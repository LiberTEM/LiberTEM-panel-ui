from __future__ import annotations
from typing import Self, TYPE_CHECKING, NamedTuple, Literal
import itertools
import functools

import numpy as np
import panel as pn
from strenum import StrEnum
from skimage.filters._fft_based import _get_nd_butterworth_filter
from skimage.registration import phase_cross_correlation
from skimage.draw import polygon2mask
from bokeh.models.tools import LassoSelectTool
from bokeh.models import Span

from libertem_ui.ui_context import UIContext  # noqa
from libertem_ui.windows.standalone import StandaloneContext
from libertem_ui.live_plot import ApertureFigure, BokehFigure
from libertem_ui.display.display_base import DiskSet, Rectangles, PointSet, Text, Polygons
from libertem_ui.display.image_db import BokehImage
# from libertem_ui.display.vectors import MultiLine
from libertem_ui.windows.base import UIWindow, WindowType, WindowProperties
from libertem_ui.utils.colormaps import get_bokeh_palette

from libertem.api import Context
from libertem.io.dataset.memory import MemoryDataSet
from libertem_holo.base.reconstr import (
    get_slice_fft,
    estimate_sideband_size,
    estimate_sideband_position,
)
from libertem_holo.base.filters import phase_unwrap as lt_phase_unwrap
from libertem_holo.base.mask import disk_aperture

from .image_transformer import ImageTransformer
from .custom_unwrap import quality_unwrap, derivative_variance, wrap


if TYPE_CHECKING:
    from libertem.io.dataset.base import DataSet
    from bokeh.events import SelectionGeometry, DoubleTap


class ViewStates(StrEnum):
    IMAGE = "Image"
    FFT = "FFT"


class OutputStates(StrEnum):
    APERTURE = "Aperture"
    CROP = "Crop"
    RECON = "Reconstruction"


class WaveViewStates(StrEnum):
    AMP = "Amplitude"
    PHASE = "Phase"


class ApertureTypes(StrEnum):
    BUT = "Butterworth"
    HARD = "Hard"


class AlignOption(StrEnum):
    ALL = "Whole image"
    SUB = "Subregion"
    MASK = "Arb. Mask"


class ApertureConfig(NamedTuple):
    sb_pos: tuple[float, float]
    radius: float
    window_size: tuple[int, int]
    ap_type: ApertureTypes = ApertureTypes.BUT
    ap_but_order: int = 4


class StackDataHandler:
    def __init__(
        self,
        stack: DataSet | np.ndarray,
        ref: DataSet | np.ndarray | None = None,
    ):
        if ref is not None:
            raise NotImplementedError
        self._data = stack

    @property
    def stack_len(self) -> int:
        try:
            return self._data.meta.shape.nav.to_tuple()[0]
        except AttributeError:
            return self._data.shape[0]

    @property
    def sig_shape(self) -> tuple[int, ...]:
        try:
            return self._data.meta.shape.sig.to_tuple()
        except AttributeError:
            return self._data.shape[1:]

    def get_fft(self, idx: int, shifted=False, abs=False) -> np.ndarray:
        frame = self.get_frame(idx)
        frame_fft = np.fft.fft2(frame)
        if shifted:
            frame_fft = np.fft.fftshift(frame_fft)
        if abs:
            frame_fft = np.abs(frame_fft)
        return frame_fft

    def get_frame(self, idx: int) -> np.ndarray:
        try:
            return self._data[idx]
        except TypeError:
            return self._data.data[idx]

    def get_crop(self, idx: int, y: float, x: float, aperture: np.ndarray):
        fft = self.get_fft(idx, abs=False, shifted=True)
        y, x = int(np.round(y)), int(np.round(x))
        h, w = fft.shape
        y -= h
        x -= w
        slice_fft = get_slice_fft(aperture.shape, fft.shape)
        rolled = np.roll(fft, (-y, -x), axis=(0, 1))
        return np.fft.fftshift(np.fft.fftshift(rolled)[slice_fft]) * aperture

    def get_recon(self, idx: int, y: float, x: float, aperture: np.ndarray):
        crop = self.get_crop(idx, y, x, aperture)
        return np.fft.ifft2(crop + 0j) * np.prod(self.sig_shape)

    def estimate_sideband_params(
        self,
        idx: int,
        sb_choice: Literal['upper', 'lower'],
        sampling: float = 1.
    ):
        im = self.get_frame(idx)
        sb_pos = estimate_sideband_position(
            im,
            (sampling, sampling),
            sb=sb_choice.lower(),
        )
        window_size = estimate_sideband_size(
            sb_pos, im.shape,
        )
        h, w = im.shape
        cy, cx = sb_pos
        cy += (h // 2)
        cy %= h
        cx += (w // 2)
        cx %= w
        window_size = int(np.round(window_size))
        window_size += (window_size % 2)
        return cx, cy, window_size

    @staticmethod
    def get_aperture(
        ap_type: ApertureTypes, radius: int, window: tuple[int, int], order: int = 1
    ):
        # as fft-shifted
        radius = int(np.round(radius))
        if ap_type == ApertureTypes.BUT:
            aperture = _get_nd_butterworth_filter(
                shape=window,
                factor=radius / min(window),
                order=order,
                high_pass=False,
                real=False,
                dtype=np.float32,
                squared_butterworth=True,
            )
        else:
            aperture = disk_aperture(
                window,
                radius,
            )
        return aperture

    def align_pair(
        self,
        static_idx: int,
        moving_idx: int,
        upsample: int = 20,
        overlap_ratio: float = 0.3,
        roi: tuple[slice, slice] | np.ndarray | None = None,
        moving_roi_offset: tuple[float, float] | None = None,
    ) -> tuple[float, float]:
        reference_mask = None
        moving_mask = None
        static_slices = moving_slices = np.s_[:, :]
        if isinstance(roi, np.ndarray):
            reference_mask = roi
            if moving_roi_offset is not None:
                raise NotImplementedError("Misunderstanding about masked xcorr or bug")
                sy, sx = map(int, moving_roi_offset)
                moving_mask = np.roll(
                    reference_mask,
                    (-sy, -sx),
                    axis=(0, 1),
                )
                # yslice = np.s_[:sy, :] if sy > 0 else np.s_[-sy:, :]
                # moving_mask[yslice] = False
                # xslice = np.s_[:, :sx] if sx > 0 else np.s_[:, -sx:]
                # moving_mask[xslice] = False
        elif roi is not None:
            if len(roi) != 2 or (not isinstance(roi[0], slice)):
                raise ValueError("Invalid roi format")
            static_slices = moving_slices = roi
            if moving_roi_offset is not None:
                # moving_roi_offset is y, x
                # FIXME bounds checking
                moving_slices = tuple(
                    slice(s.start - o, s.stop - o) for s, o in zip(
                        moving_slices, map(int, moving_roi_offset)
                    )
                )

        static = self.get_frame(static_idx)[static_slices]
        moving = self.get_frame(moving_idx)[moving_slices]
        shift, _, _ = phase_cross_correlation(
            static,
            moving,
            upsample_factor=upsample,
            normalization=None,
            reference_mask=reference_mask,
            moving_mask=moving_mask,
            overlap_ratio=overlap_ratio,
        )
        if moving_roi_offset is not None and moving_mask is not None:
            shift = (a + int(b) for a, b in zip(shift, moving_roi_offset))
        return tuple(shift)


class StackDSWindow(UIWindow):
    @classmethod
    def using(
        cls,
        dataset: DataSet | np.ndarray,
        ctx: Context | None = None,
        **init_kwargs,
    ):
        if ctx is None:
            ctx = Context.make_with('inline')
        if isinstance(dataset, np.ndarray):
            if dataset.ndim == 2:
                dataset = dataset[np.newaxis, ...]
            assert dataset.ndim == 3, 'Must be 3D stack with nav dim first'
            dataset = MemoryDataSet(data=dataset, sig_dims=2)
            dataset.initialize(ctx.executor)
        return super().using(ctx, dataset, **init_kwargs)


class ApertureBuilder(StackDSWindow, ui_type=WindowType.STANDALONE):
    @staticmethod
    def default_properties():
        return WindowProperties(
            'aperture_builder',
            'Aperture Builder',
            header_run=False,
            header_stop=False,
            self_run_only=True,
            header_activate=False,
        )

    def default_state(self):
        return ApertureConfig(
            sb_pos=(32, 32),
            radius=4,
            window_size=(16, 16),
        )

    @property
    def state(self):
        cy, cx, radius = self._disk_info()
        return ApertureConfig(
            sb_pos=(cy, cx),
            radius=radius,
            window_size=self._recon_shape(),
            ap_type=self._aperture_type(),
            ap_but_order=self._aperture_but_order(),
        )

    def initialize(self, dataset: MemoryDataSet, state: ApertureConfig | None = None) -> Self:
        self._data = StackDataHandler(dataset)
        sig_shape = self._data.sig_shape

        if state is None:
            state = self.default_state()

        self._stack_view_option = pn.widgets.RadioButtonGroup(
            name="View mode",
            options=[
                ViewStates.IMAGE,
                ViewStates.FFT,
            ],
            value=ViewStates.IMAGE,
            button_type="success",
        )
        self._stack_filtered_check = pn.widgets.Checkbox(
            name="Apply filter",
            value=False,
        )
        self._result_view_option = pn.widgets.RadioButtonGroup(
            name="View mode",
            options=[
                OutputStates.APERTURE,
                OutputStates.CROP,
                OutputStates.RECON,
            ],
            value=OutputStates.CROP,
            button_type="success",
        )
        self._recon_view_option = pn.widgets.RadioButtonGroup(
            options=[
                WaveViewStates.AMP,
                WaveViewStates.PHASE,
            ],
            value=WaveViewStates.PHASE,
            button_type="default",
            disabled=self._result_view_option.value != OutputStates.RECON,
        )
        self._per_image_sb_check = pn.widgets.Checkbox(
            name="Per-image sideband",
            value=False,
        )
        self._window_size_slider = pn.widgets.IntSlider(
            name="Window Size",
            value=min(state.window_size),
            start=2,
            end=int(max(sig_shape) * 0.75),
            step=2,
        )

        self._estimate_sb_button = pn.widgets.Button(
            name="Estimate SB",
            button_type="primary",
            disabled=self._stack_view_option.value != ViewStates.FFT,
        )
        self._sampling_val = pn.widgets.FloatInput(
            value=1.,
            start=0.1,
            end=128.,
            step=0.1,
            width=100,
        )
        self._estimate_sb_choice = pn.widgets.RadioButtonGroup(
            options=[
                "Upper",
                "Lower",
            ],
            value="Upper",
        )
        self._estimate_sb_button.on_click(self._estimate_sb_cb)

        self._unwrap_button = pn.widgets.Button(
            name="Unwrap",
            disabled=self._result_view_option.value != OutputStates.RECON,
        )
        self._unwrap_button.on_click(self._unwrap_output)

        self._aperture_type_choice = pn.widgets.Select(
            options=[
                ApertureTypes.BUT,
                ApertureTypes.HARD,
            ],
            value=state.ap_type,
            width=150,
        )
        self._aperture_but_order_int = pn.widgets.IntInput(
            value=state.ap_but_order,
            start=1,
            end=16,
            width=100,
            disabled=self._aperture_type_choice.value != ApertureTypes.BUT,
        )

        self._stack_fig = ApertureFigure.new(
            self._get_stack_image,
            title='Stack',
            channel_dimension=tuple(range(self._data.stack_len))
        )
        self._disk_annot = (
            DiskSet
            .new()
            .from_vectors(
                x=[state.sb_pos[1]],
                y=[state.sb_pos[0]],
                radius=state.radius
            )
            .on(self._stack_fig.fig)
            .editable(add=False)
            .set_visible(self._stack_view_option.value == ViewStates.FFT)
        )
        self._stack_fig.fig.toolbar.active_drag = self._stack_fig.fig.tools[-1]
        self._disk_radius_slider = self._disk_annot.get_radius_slider(
            max_r=int(max(sig_shape) * 0.75),
            label="Aperture Radius"
        )

        self._point_annot = (
            PointSet(
                self._disk_annot.cds
            )
            .on(self._stack_fig.fig)
            .set_visible(self._stack_view_option.value == ViewStates.FFT)
        )
        self._point_annot.points.marker = "cross"
        self._point_annot.points.line_color = "red"

        self._disk_annot.cds.add([self._window_size_slider.value], name="window_size")
        self._box_annot = (
            Rectangles(
                self._disk_annot.cds,
                width="window_size",
                height="window_size",
            )
            .on(self._stack_fig.fig)
            .set_visible(any(self._disk_annot.visible))
        )
        self._box_annot.rectangles.fill_color = None

        self._window_size_slider.param.watch(self._update_window_size, 'value_throttled')
        self._window_size_slider.jscallback(
            value="""
cds.data[glyph.width.field].fill(cb_obj.value);
cds.change.emit();
""",
            args={
                'cds': self._box_annot.cds,
                'glyph': self._box_annot.rectangles,
            },
        )

        # self._line_annot = (
        #     MultiLine
        #     .new()
        #     .from_vectors(xs=[[16, 48]], ys=[[48, 16]])
        #     .on(self._stack_fig.fig)
        #     .editable()
        #     .set_visible(False)
        # )
        # self._line_annot.lines.line_width = 4
        # self._line_annot.vertices.points.line_color = 'cyan'
        # self._line_annot.lines.line_color = 'cyan'
        # self._line_annot.vertices.points.fill_color = 'cyan'

        self._output_fig = ApertureFigure.new(
            np.random.uniform(size=sig_shape),
            title='Output',
            downsampling=False,
        )

        self._stack_view_option.param.watch(
            self._stack_view_cb,
            'value'
        )
        self._result_view_option.param.watch(
            self._update_output,
            'value'
        )
        self._recon_view_option.param.watch(
            self._update_output,
            'value'
        )

        self._disk_annot.cds.on_change('data', self._update_output_bk)
        self._disk_radius_slider.param.watch(self._update_output, 'value_throttled')
        self._stack_fig._channel_select.param.watch(self._update_output, 'value_throttled')
        self._aperture_type_choice.param.watch(self._update_output, 'value')
        self._aperture_but_order_int.param.watch(self._update_output, 'value')
        self._aperture_type_choice.param.watch(self._update_aperture_type, 'value')
        # self._line_annot.cds.on_change('data', _update_output)
        self._update_output()

        self._stack_fig._toolbar.insert(0, self._stack_view_option)
        self._output_fig._toolbar.insert(0, self._result_view_option)
        self._output_fig._toolbar.insert(1, self._recon_view_option)
        self._output_fig._toolbar.insert(2, self._unwrap_button)

        self._disk_radius_slider.width = 200
        self._window_size_slider.width = 200

        label_kwargs = dict(
            align="center",
            margin=(5, 5),
        )
        self.inner_layout.extend((
            pn.Column(
                self._stack_fig.layout,
                # pn.Row(
                #     self._per_image_sb_check,
                #     self._stack_filtered_check,
                # ),
                pn.Row(
                    self._disk_radius_slider,
                    self._window_size_slider,
                ),
                pn.Row(
                    self._estimate_sb_button,
                    self._estimate_sb_choice,
                    pn.widgets.StaticText(value="Sampling", **label_kwargs),
                    self._sampling_val,
                ),
            ),
            pn.Column(
                self._output_fig.layout,
                pn.Row(
                    pn.widgets.StaticText(value="Aperture type:", **label_kwargs),
                    self._aperture_type_choice,
                    pn.widgets.StaticText(value="Order", **label_kwargs),
                    self._aperture_but_order_int,
                ),
            )
        ))

        return self

    def _update_window_size(self, e):
        self._box_annot.update(width=e.new, height=e.new)
        self._update_output()

    def _update_aperture_type(self, e):
        is_but = e.new == ApertureTypes.BUT
        self._aperture_but_order_int.disabled = not is_but

    def _estimate_sb_cb(self, *e):
        cx, cy, window_size = self._data.estimate_sideband_params(
            self._current_idx(),
            sb_choice=self._estimate_sb_choice.value,
            sampling=self._sampling_val.value,
        )
        self._disk_annot.raw_update(
            cx=[cx],
            cy=[cy],
            radius=[window_size],
            window_size=[window_size],
        )
        self._disk_radius_slider.value = window_size / 2
        self._window_size_slider.value = window_size
        self._update_output()
        self._output_fig.push()

    def _get_stack_image(self, idx: int) -> tuple[np.ndarray, str]:
        state = self._stack_view_option.value
        title = f"{state} - {idx}"
        if state == ViewStates.FFT:
            frame = self._data.get_fft(idx, shifted=True, abs=True)
        else:
            frame = self._data.get_frame(idx)
        return frame, title

    def _stack_view_cb(self, e):
        state = e.new
        is_fft = state == ViewStates.FFT
        self._estimate_sb_button.disabled = not is_fft
        # self._line_annot.set_visible(is_fft)]
        self._point_annot.set_visible(is_fft)
        self._disk_annot.set_visible(is_fft)
        self._box_annot.set_visible(is_fft)
        self._stack_fig.channel_prefix = state
        self._stack_fig.change_channel(None)

    def _current_idx(self):
        return self._stack_fig._channel_select.value

    def _disk_info(self):
        cds = self._disk_annot.cds.data
        y = cds[self._disk_annot.disks.y][0]
        x = cds[self._disk_annot.disks.x][0]
        radius = self._disk_radius_slider.value
        return (y, x, radius)

    def _recon_shape(self):
        size = int(np.round(self._box_annot.cds.data["window_size"]))
        return (size, size)

    def _aperture_type(self) -> ApertureTypes:
        return self._aperture_type_choice.value

    def _aperture_but_order(self) -> int:
        return self._aperture_but_order_int.value

    def _get_aperture(self):
        _, _, radius = self._disk_info()
        return self._data.get_aperture(
            self._aperture_type(),
            radius,
            self._recon_shape(),
            order=self._aperture_but_order(),
        )

    def _update_aperture(self):
        aperture = self._get_aperture()
        self._output_fig.im.update(
            np.fft.ifftshift(aperture)
        )

    def _update_crop(self, *e):
        idx = self._current_idx()
        aperture = self._get_aperture()
        y, x, _ = self._disk_info()
        crop = self._data.get_crop(
            idx, y, x, aperture,
        )
        self._output_fig.im.update(
            np.fft.fftshift(np.abs(crop))
        )

    def _update_recon(self, *e):
        idx = self._current_idx()
        aperture = self._get_aperture()
        y, x, _ = self._disk_info()
        wave = self._data.get_recon(
            idx, y, x, aperture,
        )
        self._output_fig.im.update(
            np.angle(wave)
            if self._recon_view_option.value == WaveViewStates.PHASE
            else np.abs(wave)
        )

    def _update_output(self, *e):
        state = self._result_view_option.value
        self._recon_view_option.disabled = (state != OutputStates.RECON)
        self._unwrap_button.disabled = (state != OutputStates.RECON)

        title = state
        if state == OutputStates.APERTURE:
            self._update_aperture()
        elif state == OutputStates.CROP:
            self._update_crop()
        elif state == OutputStates.RECON:
            title = f"{title} - {self._recon_view_option.value}"
            self._update_recon()
        else:
            raise
        self._output_fig.fig.title.text = title

    def _update_output_bk(self, attr, old, new):
        self._update_output()

    def _unwrap_output(self, *e):
        if (
            self._result_view_option.value != OutputStates.RECON
            or self._recon_view_option.value != WaveViewStates.PHASE
        ):
            return
        data = self._output_fig.im.array
        self._output_fig.im.update(
            lt_phase_unwrap(data)
        )
        self._output_fig.push()


class AlignState(NamedTuple):
    static_idx: int
    shifts_y: np.ndarray
    shifts_x: np.ndarray
    skip_frame: np.ndarray

    @classmethod
    def new(cls, stack_len: int, static_idx: int = 0):
        return cls(
            static_idx=static_idx,
            shifts_y=np.zeros(stack_len).astype(float),
            shifts_x=np.zeros(stack_len).astype(float),
            skip_frame=np.zeros(stack_len).astype(bool),
        )


class StackAlignWindow(StackDSWindow, ui_type=WindowType.STANDALONE):
    @staticmethod
    def default_properties():
        return WindowProperties(
            'stack_align',
            'Stack Aligner',
            header_run=False,
            header_stop=False,
            self_run_only=True,
            header_activate=False,
        )

    @property
    def state(self) -> AlignState:
        static_cds = self._static_scatter.cds.data
        moving_cds = self._moving_scatter.cds.data
        yvals = np.asarray(list(static_cds['cy']) + list(moving_cds['cy']))
        xvals = np.asarray(list(static_cds['cx']) + list(moving_cds['cx']))
        skip = np.asarray(list(static_cds['pt_skip']) + list(moving_cds['pt_skip']))
        idx = tuple(map(int, list(static_cds['pt_label']) + list(moving_cds['pt_label'])))
        sorter = np.argsort(idx)
        return AlignState(
            static_idx=self.current_static_idx(),
            shifts_y=yvals[sorter].astype(float),
            shifts_x=xvals[sorter].astype(float),
            skip_frame=skip[sorter].astype(bool),
        )

    def current_static_idx(self) -> int:
        return self._static_idx

    def movable_keys(self) -> list[int]:
        s_idx = self.current_static_idx()
        return [
            i for i in range(self._data.stack_len)
            if i != s_idx
        ]

    def current_moving_idx(self) -> int:
        return int(self._moving_slider.value)

    def _current_colors(self) -> list[str]:
        colors = ["red" if not v else "gray" for v in self._static_scatter.cds.data['pt_skip']]
        s_idx = self.current_static_idx()
        cds = self._static_scatter.cds.data
        static_cds_idx = tuple(map(int, cds['pt_label'])).index(s_idx)
        colors[static_cds_idx] = "black"
        return colors

    def initialize(
        self, dataset: MemoryDataSet, state: AlignState | None = None, static_idx=0
    ) -> Self:

        self._data = StackDataHandler(dataset)
        if state is None:
            state = AlignState.new(self._data.stack_len, static_idx=static_idx)
        self._static_idx = state.static_idx

        self._moving_slider = pn.widgets.DiscreteSlider(
            name="Align image",
            value=(self._static_idx + 1) % self._data.stack_len,
            options=self.movable_keys(),
            width=250,
        )
        static_image = self._data.get_frame(self.current_static_idx())
        moving_image = self._data.get_frame(self.current_moving_idx())

        static_mask = np.ones(self._data.stack_len).astype(bool)
        static_mask[self.current_moving_idx()] = False
        moving_mask = np.zeros(self._data.stack_len).astype(bool)
        moving_mask[self.current_moving_idx()] = True

        next_button = pn.widgets.Button(
            name="Next",
            button_type="primary",
            width=50,
            margin=(3, 10, 30, 3),
        )
        prev_button = pn.widgets.Button(
            name="Prev",
            button_type="primary",
            width=50,
            margin=3,
        )
        next_button.on_click(self._next_moving_cb)
        prev_button.on_click(self._prev_moving_cb)

        self._image_fig = ApertureFigure.new(
            static_image,
            title='Stack',
        )
        self._moving_im = (
            BokehImage
            .new()
            .from_numpy(moving_image)
            .on(self._image_fig.fig)
        )
        self.set_image_title(skipped=state.skip_frame[moving_mask][0])
        self._moving_im.set_anchor(
            x=state.shifts_x[moving_mask][0],
            y=state.shifts_y[moving_mask][0],
        )
        self._image_fig.im.im.global_alpha = 0.5
        s_alpha_slider = self._image_fig.im.color.get_alpha_slider(
            name="Static Alpha",
            width=150,
        )
        self._moving_im.im.global_alpha = 0.5
        m_alpha_slider = self._moving_im.color.get_alpha_slider(
            name="Align Alpha",
            width=150,
        )
        toggle_alpha_btn = pn.widgets.Button(
            name="Toggle alpha",
            button_type="default",
            width=60,
        )
        toggle_alpha_btn.js_on_click(
            dict(
                s_alpha_slider=s_alpha_slider,
                m_alpha_slider=m_alpha_slider,
            ),
            code="""
let s_a = s_alpha_slider.value
let m_a = m_alpha_slider.value
if (m_a >= s_a) {
    s_alpha_slider.value = 1.
    m_alpha_slider.value = 0.
} else {
    s_alpha_slider.value = 0.
    m_alpha_slider.value = 1.
}
"""
        )

        equal_alpha_btn = pn.widgets.Button(
            name="Equal alpha",
            button_type="default",
            width=60,
        )
        equal_alpha_btn.js_on_click(
            dict(
                s_alpha_slider=s_alpha_slider,
                m_alpha_slider=m_alpha_slider,
            ),
            code="""
s_alpha_slider.value = 0.5
m_alpha_slider.value = 0.5
"""
        )

        cycle_alpha_check = pn.widgets.Checkbox(
            name="Cycle alpha",
            value=False,
        )
        cycle_alpha_check.param.watch(self._alpha_cycle_cb, 'value')
        self._cycle_cb = None

        self._image_fig.fig.x_range.bounds = (0, static_image.shape[1] - 1)
        self._image_fig.fig.y_range.bounds = (0, static_image.shape[0] - 1)

        self._image_fig.add_mask_tools()
        self._image_fig.set_mask_visiblity(rectangles=False, polygons=False)
        self._image_fig._toolbar.insert(0, self._moving_slider)
        self._image_fig._toolbar.insert(0, next_button)
        self._image_fig._toolbar.insert(0, prev_button)
        self._image_fig._toolbar.height = 60
        self._moving_slider.param.watch(self.update_moving_cb, 'value_throttled')

        drag_tool = LassoSelectTool(continuous=False)
        drag_tool.overlay.fill_alpha = 0.
        self._image_fig.fig.add_tools(drag_tool)
        self._image_fig.fig.on_event("selectiongeometry", self._drag_moving_cb)

        self._drifts_fig = BokehFigure(
            title="Shift",
            match_aspect=True,
        )
        self._drifts_fig.fig.frame_height = 400
        self._drifts_fig.fig.frame_width = 400
        self._drifts_fig.fig.y_range.flipped = True
        ver_span = Span(
            location=0.,
            dimension='height',
            line_alpha=0.5,
        )
        hor_span = Span(
            location=0.,
            dimension='width',
            line_alpha=0.5,
        )
        self._drifts_fig.fig.add_layout(ver_span)
        self._drifts_fig.fig.add_layout(hor_span)
        self._drifts_fig.fig.x_range.min_interval = 2.
        self._drifts_fig.fig.y_range.min_interval = 2.
        self._static_scatter = (
            PointSet
            .new()
            .from_vectors(
                x=state.shifts_x[static_mask],
                y=state.shifts_y[static_mask],
            )
            .on(self._drifts_fig.fig)
        )
        self._static_scatter.raw_update(
            pt_label=[
                str(i) for i in range(self._data.stack_len) if i != self.current_moving_idx()
            ],
            pt_skip=state.skip_frame[static_mask].tolist()
        )
        self._static_scatter.raw_update(
            pt_color=self._current_colors()
        )

        self._static_scatter.points.fill_color = "pt_color"
        self._static_scatter.points.line_color = "pt_color"
        self._moving_scatter = (
            PointSet
            .new()
            .from_vectors(
                x=state.shifts_x[moving_mask],
                y=state.shifts_y[moving_mask],
            )
            .on(self._drifts_fig.fig)
            .editable(add=False)
        )
        self._drifts_fig.fig.toolbar.active_drag = self._drifts_fig.fig.tools[-1]
        self._moving_scatter.raw_update(
            pt_label=[f"{self.current_moving_idx()}"],
            pt_color=["blue"],
            pt_skip=state.skip_frame[moving_mask].tolist(),
        )
        self._moving_scatter.points.fill_color = "pt_color"
        self._moving_scatter.points.line_color = "pt_color"
        self._moving_scatter.cds.on_change("data", self._move_anchor_scatter_cb)

        self._static_text = (
            Text(
                self._static_scatter.cds,
                x='cx',
                y='cy',
                text='pt_label',
            )
            .on(self._drifts_fig.fig)
        )
        text_options = dict(
            text_color='pt_color',
            x_offset=8,
            y_offset=-10,
        )
        self._static_text.glyph.update(**text_options)

        self._moving_text = (
            Text(
                self._moving_scatter.cds,
                x='cx',
                y='cy',
                text='pt_label',
            )
            .on(self._drifts_fig.fig)
        )
        self._moving_text.glyph.update(**text_options)

        self._skip_image_box = pn.widgets.Checkbox(
            name="Skip image",
            value=bool(state.skip_frame[moving_mask][0]),
        )
        self._drifts_fig._toolbar.height = 85
        self._drifts_fig._toolbar.insert(0, self._skip_image_box)
        self._skip_image_box.param.watch(self._set_validity_cb, "value")
        self._drifts_fig._toolbar.insert(
            1,
            translate_buttons(self._translate_cb),
        )
        self._translate_amount = pn.widgets.RadioButtonGroup(
            name='Translate amount',
            value=1,
            options=[0.1, 1, 10],
            align="end",
        )
        self._drifts_fig._toolbar.insert(
            2,
            self._translate_amount
        )

        align_all_btn = pn.widgets.Button(
            name="Auto-Align all",
            button_type="primary",
        )
        align_pair_btn = pn.widgets.Button(
            name="Auto-Align pair",
            button_type="primary",
        )
        reset_btn = pn.widgets.Button(
            name="Reset pair",
            button_type="warning",
        )
        reset_all_btn = pn.widgets.Button(
            name="Reset all",
            button_type="warning",
        )
        align_all_btn.on_click(self.align_all_cb)
        align_pair_btn.on_click(self.align_pair_cb)
        reset_btn.on_click(self.reset_moving_cb)
        reset_all_btn.on_click(self.reset_all_cb)

        self._align_choice = pn.widgets.Select(
            name="Align option",
            options=[
                AlignOption.ALL,
                AlignOption.SUB,
                AlignOption.MASK,
            ],
            value=AlignOption.ALL,
            width=150,
        )
        self._align_choice.param.watch(self._align_choice_cb, "value")
        self._upsample_choice = pn.widgets.Checkbox(
            name="Subpixel",
            value=True,
            align="end",
            disabled=True,
        )
        self._overlap_ratio_float = pn.widgets.FloatInput(
            name="Overlap ratio",
            value=0.3,
            start=0.05,
            step=0.05,
            end=1.,
            width=75,
            disabled=True,
        )
        self._relative_mask_box = pn.widgets.Checkbox(
            name="Relative ROI",
            value=True,
            align="end",
            disabled=True,
        )

        self._drifts_fig.fig.on_event('doubletap', self._shifts_tap_cb)

        self.inner_layout.extend((
            pn.Column(
                self._image_fig.layout,
                pn.Row(
                    s_alpha_slider,
                    m_alpha_slider,
                ),
                pn.Row(
                    cycle_alpha_check,
                    toggle_alpha_btn,
                    equal_alpha_btn,
                ),
            ),
            pn.Column(
                self._drifts_fig.layout,
                pn.Row(
                    align_all_btn,
                    align_pair_btn,
                    reset_btn,
                    reset_all_btn,
                ),
                pn.Row(
                    self._align_choice,
                    self._relative_mask_box,
                    self._upsample_choice,
                    self._overlap_ratio_float,
                ),
            )
        ))

    def _translate_cb(self, *e, x=0, y=0):
        step = float(self._translate_amount.value)
        dx = x * step
        dy = y * step
        self._update_one(dy, dx, absolute=False, push=True)

    def _align_choice_cb(self, e):
        mode = e.new
        rect_vis = poly_vis = True
        self._relative_mask_box.disabled = mode != AlignOption.SUB
        if mode in (AlignOption.ALL, AlignOption.SUB):
            self._upsample_choice.disabled = False
            self._overlap_ratio_float.disabled = True
            if mode == AlignOption.ALL:
                rect_vis = poly_vis = False
            else:
                poly_vis = False
        elif mode == AlignOption.MASK:
            self._upsample_choice.disabled = True
            self._overlap_ratio_float.disabled = False
        else:
            raise ValueError("Unrecognized option")
        self._image_fig.set_mask_visiblity(
            rectangles=rect_vis,
            polygons=poly_vis,
        )

    def _increment_moving(self, increment: int):
        options = list(self._moving_slider.options)
        num_options = len(options)
        idx = options.index(self._moving_slider.value)
        next_value = options[(idx + increment) % num_options]
        self._moving_slider.value = next_value
        self.update_moving_cb()
        self._image_fig.push(self._drifts_fig)

    def _next_moving_cb(self, *e):
        self._increment_moving(1)

    def _prev_moving_cb(self, *e):
        self._increment_moving(-1)

    def update_moving_cb(self, *e, new_idx=None):
        if new_idx is None:
            new_idx = self.current_moving_idx()
        self._change_moving_scatter(new_idx)
        self._skip_image_box.value = self._moving_scatter.cds.data['pt_skip'][0]
        self._moving_im.set_anchor(
            x=self._moving_scatter.cds.data['cx'][0],
            y=self._moving_scatter.cds.data['cy'][0],
        )
        new_frame = self._data.get_frame(new_idx)
        self._moving_im.update(new_frame)
        self.set_image_title(skipped=self._skip_image_box.value)

    def set_image_title(self, skipped=False):
        title = (
            f'Static image: {self.current_static_idx()} - Align image: {self.current_moving_idx()}'
        )
        if skipped:
            title = title + " (Skipped)"
        self._image_fig.fig.title.text = title

    def _set_validity_cb(self, e):
        self._moving_scatter.cds.data['pt_skip'][0] = bool(e.new)
        self.set_image_title(skipped=bool(e.new))

    def _change_moving_scatter(self, new_idx: str | int):
        new_idx = int(new_idx)
        old_idx = int(self._moving_scatter.cds.data['pt_label'][0])
        old_x = self._moving_scatter.cds.data['cx'][0]
        old_y = self._moving_scatter.cds.data['cy'][0]
        old_valid = self._moving_scatter.cds.data['pt_skip'][0]
        idx_in_static = list(self._static_scatter.cds.data['pt_label']).index(str(new_idx))
        new_x = self._static_scatter.cds.data['cx'][idx_in_static]
        new_y = self._static_scatter.cds.data['cy'][idx_in_static]
        new_valid = self._static_scatter.cds.data['pt_skip'][idx_in_static]
        self._static_scatter.cds.patch(
            {
                'cx': [(idx_in_static, old_x)],
                'cy': [(idx_in_static, old_y)],
                'pt_label': [(idx_in_static, str(old_idx))],
                'pt_skip': [(idx_in_static, old_valid)],
            }
        )
        self._moving_scatter.cds.patch(
            {
                'cx': [(0, new_x)],
                'cy': [(0, new_y)],
                'pt_label': [(0, str(new_idx))],
                'pt_skip': [(0, new_valid)],
            }
        )
        self._static_scatter.raw_update(
            pt_color=self._current_colors(),
        )
        self._drifts_fig.push()

    def _alpha_cycle_cb(self, e):
        if self._cycle_cb is None:
            self._cycle_cb = pn.state.add_periodic_callback(
                self._toggle_alpha_cb,
                period=400,
            )
        else:
            self._cycle_cb.stop()
            self._cycle_cb = None

    async def _toggle_alpha_cb(self):
        value = bool(self._cycle_cb.counter % 2)
        self._image_fig.im.im.global_alpha = max(0.1, float(not value))
        self._moving_im.im.global_alpha = max(0.1, float(value))
        self._image_fig.push()

    def current_upsampling(self) -> int:
        return 20 if self._upsample_choice.value else 1

    def current_overlap_r(self) -> float:
        return self._overlap_ratio_float.value

    def get_align_roi(self) -> tuple[slice, slice] | np.ndarray | None:
        mode = self._align_choice.value
        if mode == AlignOption.ALL:
            return None
        elif mode == AlignOption.MASK:
            return self._image_fig.get_mask(self._data.sig_shape)
        elif mode == AlignOption.SUB:
            slices = self._image_fig.get_mask_rect_as_slices(self._data.sig_shape)
            if not slices:
                return None
            # FIXME need a way to hide any extra rectangles / polys in this mode
            return slices[0]
        else:
            raise ValueError("Unrecognized option")

    def _get_align_roi_offset(self, idx: int | None = None) -> tuple[float, float] | None:
        if not self._relative_mask_box.value or self._align_choice != AlignOption.SUB:
            return None
        if idx is None:
            idx = self.current_moving_idx()
        shift = self.current_shift(idx)
        if np.allclose(shift, 0.):
            return None
        return shift

    def current_shift(self, idx: int) -> tuple[float, float]:
        cds = self._moving_scatter.cds.data
        for k, pt_idx in enumerate(map(int, cds['pt_label'])):
            if pt_idx == idx:
                return cds["cy"][k], cds["cx"][k]
        cds = self._static_scatter.cds.data
        for k, pt_idx in enumerate(map(int, cds['pt_label'])):
            if pt_idx == idx:
                return cds["cy"][k], cds["cx"][k]
        raise ValueError(f"Image {idx} not found")

    def align_all_cb(self, *e):
        shifts_y = []
        shifts_x = []
        static_idx = self.current_static_idx()
        roi = self.get_align_roi()
        for moving_idx in map(int, self._static_scatter.cds.data['pt_label']):
            if moving_idx == static_idx:
                shifts_y.append(0.)
                shifts_x.append(0.)
                continue
            shift_y, shift_x = self._data.align_pair(
                static_idx,
                moving_idx,
                roi=roi,
                upsample=self.current_upsampling(),
                overlap_ratio=self.current_overlap_r(),
                moving_roi_offset=self._get_align_roi_offset(moving_idx),
            )
            shifts_y.append(shift_y)
            shifts_x.append(shift_x)
        self._static_scatter.update(
            shifts_x, shifts_y
        )
        self.align_pair_cb(roi=roi)

    def _update_one(self, y: float, x: float, absolute: bool = True, push: bool = True):
        if not absolute:
            current_y, current_x = self.current_shift(self.current_moving_idx())
            x = current_x + x
            y = current_y + y
        # Need to build patch API
        self._moving_scatter.cds.patch(
            {
                'cx': [(0, x)],
                'cy': [(0, y)],
            }
        )
        self._moving_im.set_anchor(x=x, y=y)
        if push:
            self._drifts_fig.push(self._image_fig)

    def align_pair_cb(self, *e, roi: tuple[slice, slice] | np.ndarray | None = None):
        if roi is None:
            roi = self.get_align_roi()
        moving_idx = self.current_moving_idx()
        new_y, new_x = self._data.align_pair(
            self.current_static_idx(),
            moving_idx,
            roi=roi,
            upsample=self.current_upsampling(),
            overlap_ratio=self.current_overlap_r(),
            moving_roi_offset=self._get_align_roi_offset(),
        )
        self._update_one(new_y, new_x)

    def _move_anchor_scatter_cb(self, attr, old, new):
        if attr != "data":
            return
        old_x, old_y = old['cx'][0], old['cy'][0]
        new_x, new_y = new['cx'][0], new['cy'][0]
        if old_x != new_x or old_y != new_y:
            self._update_one(new_y, new_x)

    def _drag_moving_cb(self, event: SelectionGeometry):
        if not event.final or event.geometry["type"] != "poly":
            return
        xvals = event.geometry["x"]
        yvals = event.geometry["y"]

        if len(xvals) < 2:
            return

        dx = xvals[-1] - xvals[0]
        dy = yvals[-1] - yvals[0]

        self._update_one(dy, dx, absolute=False)

    def reset_moving_cb(self, e):
        self._update_one(0., 0.)

    def reset_all_cb(self, e):
        self._update_one(0., 0., push=False)
        num = self._static_scatter.data_length
        self._static_scatter.raw_update(
            cx=np.zeros((num,)),
            cy=np.zeros((num,)),
        )
        self._image_fig.push(self._drifts_fig)

    def _shifts_tap_cb(self, e: DoubleTap, radius_px: int = 30):
        fig_height = self._drifts_fig.fig.inner_height
        fig_width = self._drifts_fig.fig.inner_width
        xrange = self._drifts_fig.fig.x_range
        yrange = self._drifts_fig.fig.y_range
        data_to_px_x = fig_width / abs(xrange.end - xrange.start)
        data_to_px_y = fig_height / abs(yrange.end - yrange.start)
        click_x, click_y = e.x, e.y
        static_x = (np.asarray(self._static_scatter.cds.data["cx"]) - click_x) * data_to_px_x
        static_y = (np.asarray(self._static_scatter.cds.data["cy"]) - click_y) * data_to_px_y
        dist_2 = (static_x) ** 2 + (static_y) ** 2
        closest = np.argmin(dist_2)
        closest_dist = np.sqrt(dist_2[closest])
        closest_stack_idx = int(self._static_scatter.cds.data["pt_label"][closest])
        if closest_dist > radius_px:
            # print(f"too far to {closest_stack_idx}: {closest_dist} px")
            return
        if closest_stack_idx == self.current_static_idx():
            # print("Can't switch to static")
            return
        self._moving_slider.value = closest_stack_idx
        self.update_moving_cb(new_idx=closest_stack_idx)


class KwArgWindow(UIWindow):
    @classmethod
    def using(
        cls,
        **init_kwargs,
    ):
        return super().using(None, None, **init_kwargs)


class PointsAlignState(NamedTuple):
    static_y: np.ndarray
    static_x: np.ndarray
    moving_y: np.ndarray
    moving_x: np.ndarray
    transform_matrix: np.ndarray | None


class PointsAlignWindow(KwArgWindow, ui_type=WindowType.STANDALONE):
    @staticmethod
    def default_properties():
        return WindowProperties(
            'points_align',
            'Points Aligner',
            header_run=False,
            header_stop=False,
            self_run_only=True,
            header_activate=False,
        )

    @property
    def state(self):
        points = self.static_pointset.cds.data
        return PointsAlignState(
            static_y=np.asarray(points["cy"]),
            static_x=np.asarray(points["cx"]),
            moving_y=np.asarray(points["cy_moving"]),
            moving_x=np.asarray(points["cx_moving"]),
            transform_matrix=self.transformer.get_combined_transform().params
        )

    def initialize(self, _: None, *, static: np.ndarray, moving: np.ndarray) -> Self:
        self.transformer = ImageTransformer(moving)
        self.transformer.add_null_transform(output_shape=static.shape)

        transformations = {s.title(): s for s in ImageTransformer.available_transforms()}
        self.method_select = pn.widgets.Select(
            name='Transformation type',
            options=[*transformations.keys()],
            width=120,
        )
        run_button = pn.widgets.Button(
            name='Run',
            button_type='primary',
            width=70,
            align='end',
        )
        self.output_md = pn.pane.Markdown(
            object='No transform defined',
            width=450,
        )
        clear_button = pn.widgets.Button(
            name='Clear points',
            width=120,
            align='end',
        )

        self.static_fig = ApertureFigure.new(
            static, title='Static'
        )
        self.overlay_image = (
            BokehImage
            .new()
            .from_numpy(
                moving,
            )
            .on(self.static_fig.fig)
        )
        self.overlay_image.im.global_alpha = 0.
        alpha_slider = self.overlay_image.color.get_alpha_slider(
            name="Overlay alpha",
            width=200,
        )

        toggle_alpha_btn = pn.widgets.Button(
            name="Toggle alpha",
            button_type="default",
            width=60,
        )
        toggle_alpha_btn.js_on_click(
            dict(
                alpha_slider=alpha_slider,
            ),
            code="""
if (alpha_slider.value < 0.5) {
    alpha_slider.value = 1.
} else {
    alpha_slider.value = 0.
}
"""
        )
        self.static_fig._toolbar.insert(0, toggle_alpha_btn)
        self.static_fig._toolbar.insert(0, alpha_slider)

        self.moving_fig = ApertureFigure.new(
            moving, title='Moving'
        )

        self.static_pointset = (
            PointSet
            .new()
            .empty()
            .on(self.static_fig.fig)
            .editable()
        )
        self.static_fig.fig.toolbar.active_drag = self.static_fig.fig.tools[-1]
        self.static_pointset.raw_update(
            cx_moving=[],
            cy_moving=[],
            pt_init=[],
            color=[],
        )
        self.static_pointset.points.fill_color = "color"
        self.moving_pointset = (
            PointSet(
                self.static_pointset.cds,
                x="cx_moving",
                y="cy_moving",
            )
            .on(self.moving_fig.fig)
            .editable()
        )
        self.moving_fig.fig.toolbar.active_drag = self.moving_fig.fig.tools[-1]
        self.moving_pointset.points.fill_color = "color"

        self._sentinel_val = 3.14159
        self.setup_color_sequence()
        self.static_pointset.cds.default_values = dict(
            color="red",
            pt_init=True,
            cx_moving=self._sentinel_val,
            cy_moving=self._sentinel_val,
            cx=self._sentinel_val,
            cy=self._sentinel_val,
        )
        self.static_pointset.cds.on_change('data', self._pointset_data_change)

        clear_button.on_click(self.clear_pointsets)
        run_button.on_click(self.compute_transform)

        self.inner_layout.extend((
            pn.Column(
                self.static_fig.layout,
                pn.Row(
                    self.method_select,
                    run_button,
                    clear_button,
                )
            ),
            pn.Column(
                self.moving_fig.layout,
                pn.Row(
                    self.output_md
                ),
            )
        ))

        return self

    def setup_color_sequence(self):
        self._color_iterator = itertools.cycle(get_bokeh_palette())

    def next_color(self):
        return next(self._color_iterator)

    def _pointset_data_change(self, attr, old, new):
        patches = {
            'cx': [],
            'cy': [],
            'cx_moving': [],
            'cy_moving': [],
            "pt_init": [],
            "color": [],
        }
        for idx, is_new in enumerate(new['pt_init']):
            if not is_new:
                continue
            left_x = new['cx'][idx]
            left_y = new['cy'][idx]
            right_x = new['cx_moving'][idx]
            right_y = new['cy_moving'][idx]
            if right_x == self._sentinel_val and right_y == self._sentinel_val:
                patches["cx_moving"].append((idx, left_x))
                patches["cy_moving"].append((idx, left_y))
                patches["pt_init"].append((idx, False))
                patches["color"].append((idx, self.next_color()))
            elif left_x == self._sentinel_val and left_y == self._sentinel_val:
                patches["cx"].append((idx, right_x))
                patches["cy"].append((idx, right_y))
                patches["pt_init"].append((idx, False))
                patches["color"].append((idx, self.next_color()))
            else:
                continue

        if len(patches["pt_init"]) > 0:
            self.static_pointset.cds.patch(patches)
            self.static_fig.push(self.moving_fig)

    def clear_pointsets(self, *e):
        self.setup_color_sequence()
        self.static_pointset.clear()
        self.static_fig.push(self.moving_fig)

    def compute_transform(self, *e):
        method = self.method_select.value.lower()

        points = self.static_pointset.cds.data
        static_points = np.stack((points["cx"], points["cy"]), axis=1)
        moving_points = np.stack((points["cx_moving"], points["cy_moving"]), axis=1)

        if static_points.shape[0] == 0:
            self.output_md.object = 'No points defined'
            return
        try:
            transform = self.transformer.estimate_transform(
                static_points,
                moving_points,
                method=method,
                clear=True
            )
        except Exception:
            self.output_md.object = f'Error computing transform: {str(e)}'
            return
        try:
            self.output_md.object = self.array_format(transform.params)
        except Exception:
            self.output_md.object = 'Post-transform error (format?)'
            return

        warped_moving = self.transformer.get_transformed_image(self.transformer)
        self.overlay_image.update(warped_moving)
        self.static_fig.push()

    @staticmethod
    def array_format(array: np.ndarray, header='Transformation matrix:'):
        """
        Format a 3x3 array nicely as a Markdown string
        This is quite hacky, can be much improved
        """
        assert array.shape == (3, 3)
        str_array = np.array2string(array,
                                    precision=2,
                                    suppress_small=True,
                                    sign=' ',
                                    floatmode='fixed')
        substrings = str_array.split('\n')
        return f'''
    {header}
    ```
    {substrings[0]}
    {substrings[1]}
    {substrings[2]}
    ```'''


class UnwrapOption(StrEnum):
    SKIMAGE = "Reliability-guided"
    QUALITY = "Quality-guided"
    # GOLDSTEIN = "Goldstein's method"
    # MASK_CUT = "Mask Cut"


class PhaseUnwrapWindow(KwArgWindow, ui_type=WindowType.STANDALONE):
    @staticmethod
    def default_properties():
        return WindowProperties(
            'phase_unwrap',
            'Phase Unwrap',
            header_run=False,
            header_stop=False,
            self_run_only=True,
            header_activate=False,
        )

    def initialize(self, _: None, *, image: np.ndarray) -> Self:
        self._data = wrap(image)

        self._run_button = pn.widgets.Button(
            name='Run',
            button_type='primary',
            width=70,
            align='end',
        )
        reset_button = pn.widgets.Button(
            name='Reset',
            width=120,
            align='end',
        )
        self._method_select = pn.widgets.Select(
            name="Method",
            value=UnwrapOption.SKIMAGE,
            options=list(e.value for e in UnwrapOption),
            width=200,
            align="end",
        )

        self._phase_roll_slider = pn.widgets.FloatSlider(
            name="Offset phase (/pi)",
            value=0.,
            start=-1,
            end=1,
            step=0.01,
        )

        self._run_button.on_click(self.unwrap_cb)
        StandaloneContext._setup_run_button(self._run_button)
        reset_button.on_click(self.reset_cb)
        self._phase_roll_slider.param.watch(
            self._phase_roll_cb, 'value_throttled'
        )

        self.image_fig = ApertureFigure.new(
            self._data.copy(), title='Image'
        )
        self.image_fig.fig.on_event(
            'doubletap', self._set_point_zero_cb
        )

        drag_tool = LassoSelectTool(continuous=False)
        drag_tool.overlay.fill_alpha = 0.
        self.image_fig.fig.add_tools(drag_tool)
        self.image_fig.fig.on_event("selectiongeometry", self._drag_cb)

        self._seed_polys = (
            Polygons
            .new()
            .empty()
            .on(self.image_fig.fig)
        )

        self.inner_layout.extend((
            pn.Column(
                self.image_fig.layout,
            ),
            pn.Column(
                pn.Row(
                    self._run_button,
                    reset_button,
                ),
                self._method_select,
                self._phase_roll_slider,
            )
        ))

    def _current_seed(self):
        if self._seed_polys.data_length == 0:
            return None
        xvals = self._seed_polys.cds.data['xs'][0]
        yvals = self._seed_polys.cds.data['ys'][0]
        if len(yvals) < 2:
            return
        return polygon2mask(
            self._data.shape,
            np.round(np.stack((yvals, xvals), axis=1)).astype(int)
        )

    def unwrap_cb(self, *e):
        self._run_button.disabled = True
        self._phase_roll_slider.disabled = True
        method = self._method_select.value
        if method == UnwrapOption.SKIMAGE:
            unwrapped = lt_phase_unwrap(
                self._current_rolled_image()
            )
        else:
            phase_wrapped = self._current_rolled_image()
            quality = derivative_variance(phase_wrapped)
            unwrapped = quality_unwrap(
                phase_wrapped,
                quality,
                seed=self._current_seed(),
            )

        self.image_fig.im.update(unwrapped)
        self.image_fig.fig.title.text = f"Unwrapped ({method})"
        self.image_fig.push()
        self._run_button.disabled = False

    def reset_cb(self, *e):
        self._phase_roll_slider.disabled = False
        self._phase_roll_slider.value = 0.
        self._seed_polys.clear()
        self.image_fig.im.update(self._data)
        self.image_fig.fig.title.text = "Image"
        self.image_fig.push()

    def _current_rolled_image(self):
        offset = self._phase_roll_slider.value
        return wrap(self._data + offset * np.pi)

    def _phase_roll_cb(self, *e):
        offset = self._phase_roll_slider.value
        self.image_fig.im.update(self._current_rolled_image())
        self.image_fig.fig.title.text = f"Image (offset {offset:.2f} * pi)"
        self.image_fig.push()

    def _set_point_zero_cb(self, event: DoubleTap):
        click_x, click_y = int(event.x), int(event.y)
        h, w = self._data.shape
        if not ((0 <= click_x < w) and (0 <= click_y < h)):
            return
        ref_val = wrap(self._data[click_y, click_x])
        self._phase_roll_slider.value = min(max(-1, -1 * (ref_val / np.pi)), 1.)
        self._phase_roll_cb()

    def _drag_cb(self, event: SelectionGeometry):
        if not event.final or event.geometry["type"] != "poly":
            return
        xvals = np.asarray(event.geometry["x"])
        yvals = np.asarray(event.geometry["y"])

        if len(xvals) < 2:
            return

        self._seed_polys.update(
            [xvals], [yvals],
        )
        self.image_fig.push()


# Unicode arrow codes used for defining UI buttons
LEFT_ARROW = '\u25C1'
UP_ARROW = '\u25B3'
RIGHT_ARROW = '\u25B7'
DOWN_ARROW = '\u25BD'
ROTATE_RIGHT_ARROW = '\u21B7'
ROTATE_LEFT_ARROW = '\u21B6'


def translate_buttons(cb, width: int = 40, height: int = 40, margin: tuple[int, int] = (2, 2)):
    """
    A button array for up/down/left/right
    Configured for y-axis pointing down!!
    """
    kwargs = {
        'width': width,
        'height': height,
        'margin': margin,
        'sizing_mode': 'fixed',
    }
    get_sp = lambda: pn.Spacer(**kwargs)  # noqa
    button_kwargs = {
        'button_type': 'primary',
        **kwargs,
    }
    left = pn.widgets.Button(name=LEFT_ARROW, **button_kwargs)
    left.on_click(functools.partial(cb, x=-1))
    up = pn.widgets.Button(name=UP_ARROW, **button_kwargs)
    up.on_click(functools.partial(cb, y=-1))
    right = pn.widgets.Button(name=RIGHT_ARROW, **button_kwargs)
    right.on_click(functools.partial(cb, x=1))
    down = pn.widgets.Button(name=DOWN_ARROW, **button_kwargs)
    down.on_click(functools.partial(cb, y=1))
    return pn.Column(
        pn.Row(get_sp(), up, get_sp(), margin=(0, 0)),
        pn.Row(left, down, right, margin=(0, 0)),
        # pn.Row(get_sp(), down, get_sp(), margin=(0, 0)),
        margin=(0, 0),
    )


def rotate_buttons(cb):
    """A button array for rotate acw / cw"""
    width = height = 40
    margin = (2, 2)
    sp = pn.Spacer(width=width, height=height, margin=margin)
    kwargs = {'width': width, 'height': height, 'margin': margin, 'button_type': 'primary'}
    acw_btn = pn.widgets.Button(name=ROTATE_LEFT_ARROW, **kwargs)
    acw_btn.on_click(functools.partial(cb, dir=-1))
    cw_btn = pn.widgets.Button(name=ROTATE_RIGHT_ARROW, **kwargs)
    cw_btn.on_click(functools.partial(cb, dir=1))
    return pn.Row(sp, acw_btn, cw_btn, margin=(0, 0))


def scale_buttons(cb):
    """A button array for scaling x / y / xy up and down"""
    width = height = 40
    margin = (2, 2)
    text_kwargs = {'width': width // 2,
                   'height': height // 2,
                   'margin': margin,
                   'align': ('end', 'center')}
    button_kwargs = {'width': width,
                     'height': height,
                     'margin': margin,
                     'button_type': 'primary'}
    x_row = up_down_pair('X:',
                         cb,
                         {'xdir': 1},
                         {'xdir': -1},
                         text_kwargs,
                         button_kwargs)
    y_row = up_down_pair('Y:',
                         cb,
                         {'ydir': 1},
                         {'ydir': -1},
                         text_kwargs,
                         button_kwargs)
    xy_row = up_down_pair('XY:',
                          cb,
                          {'xdir': 1, 'ydir': 1},
                          {'xdir': -1, 'ydir': -1},
                          text_kwargs,
                          button_kwargs)
    lo = pn.Column(x_row,
                   y_row,
                   xy_row, margin=(0, 0))
    return lo


def up_down_pair(name, cb, upkwargs, downkwargs, text_kwargs, button_kwargs):
    sp = pn.Spacer(**text_kwargs)
    text = pn.widgets.StaticText(value=name, **text_kwargs)
    compress = pn.widgets.Button(name=f'{RIGHT_ARROW} {LEFT_ARROW}', **button_kwargs)
    compress.on_click(functools.partial(cb, **downkwargs))
    expand = pn.widgets.Button(name=f'{LEFT_ARROW} {RIGHT_ARROW}', **button_kwargs)
    expand.on_click(functools.partial(cb, **upkwargs))
    return pn.Row(sp, text, compress, expand, margin=(0, 0))
