from typing import Self, TYPE_CHECKING

import numpy as np
import panel as pn
from strenum import StrEnum

from libertem_ui.ui_context import UIContext  # noqa
from libertem_ui.live_plot import ApertureFigure
from libertem_ui.display.display_base import DiskSet
from libertem_ui.display.vectors import MultiLine
from libertem_ui.windows.base import UIWindow, WindowType, WindowProperties

from libertem_holo.base.reconstr import reconstruct_frame, get_slice_fft
from libertem_holo.base.mask import disk_aperture


if TYPE_CHECKING:
    from libertem.io.dataset.memory import MemoryDataSet


class ViewStates(StrEnum):
    IMAGE = "Image"
    FFT = "FFT"


class OutputStates(StrEnum):
    APERTURE = "Aperture"
    CROP = "Crop"
    RECON = "Reconstruction"


class ApertureBuilder(UIWindow, ui_type=WindowType.STANDALONE):
    @staticmethod
    def default_properties():
        return WindowProperties(
            'aperture_builder',
            'Aperture Builder',
            self_run_only=True,
            header_activate=False,
        )

    def initialize(self, dataset: 'MemoryDataSet') -> Self:
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
                "Amplitude",
                "Phase",
            ],
            value="Phase",
            button_type="default",
            disabled=self._result_view_option.value != OutputStates.RECON,
        )
        self._per_image_sb_check = pn.widgets.Checkbox(
            name="Per-image sideband",
            value=False,
        )

        self._data = dataset.data
        # stored as shifted, real
        data_fft = np.fft.fft2(
            self._data,
        ).real
        data_fft[:, 0, 0] = np.nan  # for display
        self._data_fft = np.fft.fftshift(
            data_fft
        )

        self._stack_fig = ApertureFigure.new(
            self._data,
            title='Stack',
            channel_dimension=0,
        )
        self._disk_annot = (
            DiskSet
            .new()
            .from_vectors([32], [32], 4)
            .on(self._stack_fig.fig)
            .editable(add=False)
            .set_visible(False)
        )
        self._stack_fig.fig.toolbar.active_drag = self._stack_fig.fig.tools[-1]
        self._disk_radius_slider = self._disk_annot.get_radius_slider(max_r=32)

        self._line_annot = (
            MultiLine
            .new()
            .from_vectors(xs=[[16, 48]], ys=[[48, 16]])
            .on(self._stack_fig.fig)
            .editable()
            .set_visible(False)
        )
        self._line_annot.lines.line_width = 4
        self._line_annot.vertices.points.line_color = 'cyan'
        self._line_annot.lines.line_color = 'cyan'
        self._line_annot.vertices.points.fill_color = 'cyan'

        out_shape = dataset.shape.sig

        self._output_fig = ApertureFigure.new(
            np.random.uniform(size=out_shape),
            title='Output',
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
        # self._line_annot.cds.on_change('data', _update_output)

        self._stack_fig._toolbar.insert(0, self._stack_view_option)
        self._output_fig._toolbar.insert(0, self._result_view_option)
        self._output_fig._toolbar.insert(1, self._recon_view_option)

        self.inner_layout.extend((
            pn.Column(
                self._stack_fig.layout,
                pn.Row(
                    self._per_image_sb_check,
                    self._stack_filtered_check,
                ),
                self._disk_radius_slider,
            ),
            pn.Column(
                self._output_fig.layout,
            )
        ))

        return self

    def _stack_view_cb(self, e):
        if e.new == ViewStates.FFT:
            self._line_annot.set_visible(True)
            self._disk_annot.set_visible(True)
            self._stack_fig._setup_multichannel(self._data_fft, dim=0)
        elif e.new == ViewStates.IMAGE:
            self._line_annot.set_visible(False)
            self._disk_annot.set_visible(False)
            self._stack_fig._setup_multichannel(self._data, dim=0)
        else:
            raise
        self._stack_fig.channel_prefix = e.new
        self._stack_fig.change_channel(None)

    def _current_frame(self):
        return self._data[self._stack_fig._channel_select.value]

    def _current_fft_data(self):
        return self._data_fft[self._stack_fig._channel_select.value]

    def _disk_info(self):
        cds = self._disk_annot.cds.data
        y = cds[self._disk_annot.disks.y][0]
        x = cds[self._disk_annot.disks.x][0]
        radius = self._disk_radius_slider.value
        return (y, x, radius)

    def _recon_shape(self):
        return (32, 32)

    def _get_aperture(self):
        # as fft-shifted
        recon_shape = self._recon_shape()
        _, _, radius = self._disk_info()
        return disk_aperture(
            recon_shape,
            int(np.round(radius)),
        )

    def _update_aperture(self):
        aperture = self._get_aperture()
        self._output_fig.im.update(
            np.fft.ifftshift(aperture)
        )

    def _get_crop(self):
        im = self._current_fft_data()
        y, x, _ = map(int, map(np.round, self._disk_info()))
        aperture = self._get_aperture()
        slice_fft = get_slice_fft(aperture.shape, im.shape)

        rolled = np.roll(np.fft.fftshift(im), (y, x), axis=(0, 1))
        return np.fft.fftshift(
            np.fft.fftshift(rolled)[slice_fft] * aperture
        )

    def _update_crop(self, *e):
        crop = self._get_crop()
        self._output_fig.im.update(crop)

    def _get_recon(self):
        frame = self._current_frame()
        y, x, _ = map(int, map(np.round, self._disk_info()))
        aperture = self._get_aperture()
        slice_fft = get_slice_fft(aperture.shape, frame.shape)
        return reconstruct_frame(
            frame,
            (y, x),
            aperture,
            slice_fft,
            precision=False,
        )

    def _update_recon(self, *e):
        wave = self._get_recon()
        self._output_fig.im.update(
            np.angle(wave) if self._recon_view_option.value == "Phase" else np.abs(wave)
        )

    def _update_output(self, *e):
        state = self._result_view_option.value
        self._recon_view_option.disabled = (state != OutputStates.RECON)

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
