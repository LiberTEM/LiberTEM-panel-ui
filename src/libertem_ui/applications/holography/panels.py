from __future__ import annotations
from typing import Self, TYPE_CHECKING, NamedTuple

import numpy as np
import panel as pn
from strenum import StrEnum
from skimage.filters._fft_based import _get_nd_butterworth_filter

from libertem_ui.ui_context import UIContext  # noqa
from libertem_ui.live_plot import ApertureFigure
from libertem_ui.display.display_base import DiskSet, Rectangles
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

    def get_fft(self, idx, shifted=False, abs=False) -> np.ndarray:
        frame = self.get_frame(idx)
        frame_fft = np.fft.fft2(frame)
        if shifted:
            frame_fft = np.fft.fftshift(frame_fft)
        if abs:
            frame_fft = np.abs(frame_fft)
        return frame_fft

    def get_frame(self, idx) -> np.ndarray:
        try:
            return self._data[idx]
        except TypeError:
            return self._data.data[idx]


class ApertureBuilder(UIWindow, ui_type=WindowType.STANDALONE):
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
        self._estimate_sb_button.on_click(self._estimate_sb)

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

    def _estimate_sb(self, *e):
        im = self._current_frame()
        sb = self._estimate_sb_choice.value.lower()
        sampling = self._sampling_val.value
        sb_pos = estimate_sideband_position(
            im,
            (sampling, sampling),
            sb=sb,
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
            frame = self._current_frame(idx=idx)
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

    def _current_frame(self, idx=None):
        if idx is None:
            idx = self._stack_fig._channel_select.value
        return self._data.get_frame(idx)

    def _current_fft_data(self, complex=False, shifted=False, idx=None):
        if idx is None:
            idx = self._stack_fig._channel_select.value
        if complex:
            return self._data.get_fft(idx, shifted=shifted, abs=False)
        else:
            return self._data.get_fft(idx, shifted=shifted, abs=True)

    def _disk_info(self):
        cds = self._disk_annot.cds.data
        y = cds[self._disk_annot.disks.y][0]
        x = cds[self._disk_annot.disks.x][0]
        radius = self._disk_radius_slider.value
        return (y, x, radius)

    def _recon_shape(self):
        size = int(np.round(self._box_annot.cds.data["window_size"]))
        return (size, size)

    def _aperture_type(self):
        return self._aperture_type_choice.value

    def _aperture_but_order(self):
        return self._aperture_but_order_int.value

    def _get_aperture(self):
        # as fft-shifted
        _, _, radius = self._disk_info()
        radius = int(np.round(radius))
        recon_shape = self._recon_shape()
        if self._aperture_type() == ApertureTypes.BUT:
            aperture = _get_nd_butterworth_filter(
                shape=recon_shape,
                factor=radius / min(recon_shape),
                order=self._aperture_but_order(),
                high_pass=False,
                real=False,
                dtype=np.float32,
                squared_butterworth=True,
            )
        else:
            aperture = disk_aperture(
                recon_shape,
                radius,
            )
        return aperture

    def _update_aperture(self):
        aperture = self._get_aperture()
        self._output_fig.im.update(
            np.fft.ifftshift(aperture)
        )

    def _get_crop(self, complex=False):
        fft = self._current_fft_data(complex=complex, shifted=True)
        y, x, _ = map(int, np.round(self._disk_info()))
        h, w = fft.shape
        y -= h
        x -= w
        aperture = self._get_aperture()
        slice_fft = get_slice_fft(aperture.shape, fft.shape)
        rolled = np.roll(fft, (-y, -x), axis=(0, 1))
        return np.fft.fftshift(np.fft.fftshift(rolled)[slice_fft]) * aperture

    def _update_crop(self, *e):
        crop = self._get_crop()
        self._output_fig.im.update(
            np.fft.fftshift(crop)
        )

    def _get_recon(self):
        crop = self._get_crop(complex=True)
        return np.fft.ifft2(crop + 0j) * np.prod(self._current_frame().shape)

    def _update_recon(self, *e):
        wave = self._get_recon()
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