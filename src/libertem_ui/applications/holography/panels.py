from __future__ import annotations
from typing import Self, TYPE_CHECKING, NamedTuple, Literal

import numpy as np
import panel as pn
from strenum import StrEnum
from skimage.filters._fft_based import _get_nd_butterworth_filter
from skimage.registration import phase_cross_correlation
from bokeh.models.tools import LassoSelectTool
from bokeh.models import Span

from libertem_ui.ui_context import UIContext  # noqa
from libertem_ui.live_plot import ApertureFigure, BokehFigure
from libertem_ui.display.display_base import DiskSet, Rectangles, PointSet
from libertem_ui.display.image_db import BokehImage
# from libertem_ui.display.vectors import MultiLine
from libertem_ui.windows.base import UIWindow, WindowType, WindowProperties

from libertem.api import Context
from libertem.io.dataset.memory import MemoryDataSet
from libertem_holo.base.reconstr import (
    get_slice_fft,
    estimate_sideband_size,
    estimate_sideband_position,
)
from libertem_holo.base.filters import phase_unwrap
from libertem_holo.base.mask import disk_aperture


if TYPE_CHECKING:
    from libertem.io.dataset.base import DataSet
    from bokeh.events import SelectionGeometry


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
        upsample: int = 20
    ) -> tuple[float, float]:
        static = self.get_frame(static_idx)
        moving = self.get_frame(moving_idx)
        shift, _, _ = phase_cross_correlation(
            static,
            moving,
            upsample_factor=upsample,
            normalization=None,
        )
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
                    self._estimate_sb_button,
                    self._estimate_sb_choice,
                    pn.widgets.StaticText(value="Sampling", **label_kwargs),
                    self._sampling_val,
                ),
                pn.Row(
                    self._disk_radius_slider,
                    self._window_size_slider,
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
        # self._line_annot.set_visible(is_fft)
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
            phase_unwrap(data)
        )
        self._output_fig.push()


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

    # @property
    # def state(self):
    #     cy, cx, radius = self._disk_info()
    #     return ApertureConfig(
    #         sb_pos=(cy, cx),
    #         radius=radius,
    #         window_size=self._recon_shape(),
    #     )

    def current_static_idx(self) -> int:
        return int(self._static_choice.value)

    def current_moving_idx(self) -> int:
        return self._moving_slider.value

    def _current_colors(self) -> list[str]:
        m_idx = self.current_moving_idx()
        return [
            "blue"
            if i == m_idx
            else "red"
            for i in range(self._drifts_scatter.data_length)
        ]

    def initialize(self, dataset: MemoryDataSet) -> Self:
        self._data = StackDataHandler(dataset)

        self._static_choice = pn.widgets.Select(
            name="Static image",
            value="0",
            options=[str(i) for i in range(self._data.stack_len)],
            width=100,
        )
        self._moving_slider = pn.widgets.IntSlider(
            name="Moving image",
            value=1,
            start=0,
            end=self._data.stack_len - 1,
            width=250,
        )
        static_image = self._data.get_frame(self.current_static_idx())
        moving_image = self._data.get_frame(self.current_moving_idx())

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
        self.set_image_title()
        self._moving_im.im.global_alpha = 0.5
        m_alpha_slider = self._moving_im.color.get_alpha_slider(name="Moving Alpha")

        self._image_fig.add_mask_tools()
        self._image_fig._toolbar.insert(0, self._moving_slider)
        self._image_fig._toolbar.insert(0, self._static_choice)
        self._image_fig._toolbar.height = 60
        self._static_choice.param.watch(self.update_static_cb, 'value')
        self._moving_slider.param.watch(self.update_moving_cb, 'value_throttled')

        drag_tool = LassoSelectTool(continuous=False)
        drag_tool.overlay.fill_alpha = 0.
        self._image_fig.fig.add_tools(drag_tool)
        self._image_fig.fig.on_event("selectiongeometry", self._drag_moving_cb)

        self._drifts_fig = BokehFigure(title="Drift")
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
                x=np.zeros((1,)),
                y=np.zeros((1,)),
            )
            .on(self._drifts_fig.fig)
        )
        self._static_scatter.points.fill_color = "black"
        self._static_scatter.points.line_color = "black"
        self._static_scatter.raw_update(
            pt_label=["static"]
        )
        self._drifts_scatter = (
            PointSet
            .new()
            .from_vectors(
                x=np.zeros(self._data.stack_len),
                y=np.zeros(self._data.stack_len),
            )
            .on(self._drifts_fig.fig)
            .editable(add=False)
        )
        self._drifts_scatter.raw_update(
            pt_label=[f"{i}" for i in range(self._drifts_scatter.data_length)],
            pt_color=self._current_colors(),
        )
        self._drifts_scatter.points.fill_color = "pt_color"
        self._drifts_scatter.points.line_color = "pt_color"
        self._drifts_scatter.cds.on_change("data", self._move_anchor_scatter_cb)

        align_all_btn = pn.widgets.Button(
            name="Auto-Align all",
            button_type="primary",
        )
        align_pair_btn = pn.widgets.Button(
            name="Auto-Align pair",
            button_type="primary",
        )
        reset_btn = pn.widgets.Button(
            name="Reset moving",
            button_type="primary",
        )
        align_all_btn.on_click(self.align_all_cb)
        align_pair_btn.on_click(self.align_pair_cb)
        reset_btn.on_click(self.reset_moving_cb)

        self.inner_layout.extend((
            pn.Column(
                self._image_fig.layout,
                m_alpha_slider,
            ),
            pn.Column(
                self._drifts_fig.layout,
                pn.Row(
                    align_all_btn,
                    align_pair_btn,
                    reset_btn,
                )
            )
        ))

    def update_static_cb(self, *e):
        frame = self._data.get_frame(self.current_static_idx())
        self._image_fig.im.update(frame)
        self.set_image_title()

    def update_moving_cb(self, e):
        m_idx = self.current_moving_idx()
        frame = self._data.get_frame(m_idx)
        self._moving_im.set_anchor(
            x=self._drifts_scatter.cds.data['cx'][m_idx],
            y=self._drifts_scatter.cds.data['cy'][m_idx],
        )
        self._moving_im.update(frame)
        self.set_image_title()
        self._sync_colors()

    def set_image_title(self):
        self._image_fig.fig.title.text = (
            f'Static {self.current_static_idx()} -  Moving {self.current_moving_idx()}'
        )

    def _sync_colors(self):
        self._drifts_scatter.raw_update(
            pt_color=self._current_colors(),
        )
        self._drifts_fig.push()

    def align_all_cb(self, *e):
        shifts_y = []
        shifts_x = []
        static_idx = self.current_static_idx()
        for moving_idx in range(self._data.stack_len):
            if moving_idx == static_idx:
                shifts_y.append(0.)
                shifts_x.append(0.)
                continue
            shift_y, shift_x = self._data.align_pair(
                static_idx,
                moving_idx,
            )
            shifts_y.append(shift_y)
            shifts_x.append(shift_x)
        self._drifts_scatter.update(
            shifts_x, shifts_y
        )
        m_idx = self.current_moving_idx()
        self._moving_im.set_anchor(
            x=shifts_x[m_idx],
            y=shifts_y[m_idx],
        )
        self._drifts_fig.push(self._image_fig)

    def _update_one(self, y, x, idx: int | None = None, absolute: bool = True):
        if idx is None:
            idx = self.current_moving_idx()
        if not absolute:
            current_x = self._drifts_scatter.cds.data['cx'][idx]
            current_y = self._drifts_scatter.cds.data['cy'][idx]
            x = current_x + x
            y = current_y + y
        # Need to build patch API
        self._drifts_scatter.cds.patch(
            {
                self._drifts_scatter.points.x: [(idx, x)],
                self._drifts_scatter.points.y: [(idx, y)],
            }
        )
        self._moving_im.set_anchor(x=x, y=y)
        self._drifts_fig.push(self._image_fig)

    def align_pair_cb(self, *e):
        moving_idx = self.current_moving_idx()
        new_y, new_x = self._data.align_pair(
            self.current_static_idx(),
            moving_idx,
        )
        self._update_one(new_y, new_x)

    def _move_anchor_scatter_cb(self, attr, old, new):
        if attr != "data":
            return
        m_idx = self.current_moving_idx()
        old_x, old_y = old['cx'][m_idx], old['cy'][m_idx]
        new_x, new_y = new['cx'][m_idx], new['cy'][m_idx]
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
