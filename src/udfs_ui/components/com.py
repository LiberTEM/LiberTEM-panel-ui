from __future__ import annotations
from typing import TYPE_CHECKING
from strenum import StrEnum
from typing_extensions import Self, Literal, TypedDict
from functools import partial

import numpy as np
import panel as pn
from libertem.udf.com import (
    CoMUDF, CoMParams, RegressionOptions, apply_correction, divergence, curl_2d
)

from .imaging import ImagingWindow
from .base import UIType, WindowProperties
from ..display.vectors import VectorsOverlay
from .result_containers import Numpy2DResultContainer

if TYPE_CHECKING:
    from libertem.io.dataset.base import DataSet
    from libertem_live.detectors.base.acquisition import AcquisitionProtocol
    from libertem.udf.base import UDF
    from .base import UDFWindowJob, JobResults, UIState
    from .results import ResultRow
    from libertem.udf.base import UDFResultDict


class CoMParamsUI(TypedDict):
    params: CoMParams
    result_title: str
    result_name: str


class CoMChanN(StrEnum):
    SHIFT_MAGNITUDE = 'shift_magnitude'
    RAW_X_SHIFT = 'raw_x_shift'
    RAW_Y_SHIFT = 'raw_y_shift'
    CORRECTED_X_SHIFT = 'corrected_x_shift'
    CORRECTED_Y_SHIFT = 'corrected_y_shift'
    DIVERGENCE = 'divergence'
    CURL = 'curl'
    REGRESSION_X = 'regression_x'
    REGRESSION_Y = 'regression_y'


class CoMImagingWindow(ImagingWindow, ui_type=UIType.STANDALONE):
    @staticmethod
    def default_properties():
        return WindowProperties(
            'com',
            'Centre of Mass',
        )

    def initialize(self, dataset: DataSet) -> Self:
        super().initialize(dataset, with_layout=False)

        self._current_params = CoMParamsUI()
        self.nav_plot._channel_map = {
            CoMChanN.SHIFT_MAGNITUDE: 'magnitude',
            CoMChanN.RAW_X_SHIFT: ('raw_shifts', lambda buffer: buffer[..., 1]),
            CoMChanN.RAW_Y_SHIFT: ('raw_shifts', lambda buffer: buffer[..., 0]),
            CoMChanN.CORRECTED_X_SHIFT: ('raw_shifts', partial(self._get_corrected_live, 1)),
            CoMChanN.CORRECTED_Y_SHIFT: ('raw_shifts', partial(self._get_corrected_live, 0)),
            CoMChanN.DIVERGENCE: 'divergence',
            CoMChanN.CURL: 'curl',
            CoMChanN.REGRESSION_X: self._plot_regression_x,
            CoMChanN.REGRESSION_Y: self._plot_regression_y,
        }
        self._channel_select = self.nav_plot.get_channel_select(update_title=False)
        self._channel_select.param.watch(self._update_nav_title, 'value')

        # We are overriding the default CB
        # Must take full responsibility for keeping
        # the live plot in sync with the widget
        self._channel_select.param.unwatch(
            self.nav_plot._channel_select_watcher
        )

        self._regression_mapping = {
            'NO_REGRESSION': RegressionOptions.NO_REGRESSION,
            'SUBTRACT_MEAN': RegressionOptions.SUBTRACT_MEAN,
            'SUBTRACT_LINEAR': RegressionOptions.SUBTRACT_LINEAR,
        }
        self._regression_select = pn.widgets.Select(
            name='Regression',
            options=list(self._regression_mapping.keys()),
            value='NO_REGRESSION',
            width=175,
        )
        self._mode_mapping.pop('Whole Frame')
        self._mode_mapping['Whole Frame'] = self._mode_mapping.pop('Point')
        self._mode_selector.options = list(self._mode_mapping.keys())

        self._guess_corrections_btn = pn.widgets.Button(
            name='Guess corrections',
            button_type='success',
            align='end',
            width=120,
        )

        self._flip_y_cbox = pn.widgets.Checkbox(
            name='Flip-y',
            value=False,
            align='end',
        )
        self._flip_y_cbox.param.watch(lambda e: self._vectors.flip_dir('y'), 'value')
        if self._flip_y_cbox.value:
            self._vectors.flip_dir('y')

        self._show_vectors_cbox = pn.widgets.Checkbox(
            name='Show rotation',
            value=False,
            align='end',
        )

        cx = self._disk_db.cds.data['cx'][0]
        cy = self._disk_db.cds.data['cy'][0]
        sig_dim = max(1, min(dataset.shape.sig) * 0.25)
        self._vectors = VectorsOverlay.new().from_params(
            cx, cy, sig_dim, labels=('sx', 'sy'),
        )
        self._vectors.on(self.sig_plot.fig)
        self._rotation_slider = self._vectors.with_rotation(label='Scan rotation', direction=-1)
        self._vectors.follow_point(self._disk_db.cds)
        self._vectors.set_visible(self._show_vectors_cbox.value)
        self._show_vectors_cbox.param.watch(lambda e: self._vectors.set_visible(e.new), 'value')

        self._rot_reset_btn = pn.widgets.Button(
            icon='refresh',
            button_type='light',
            width=35,
            align='end',
            margin=(3, 3),
        )
        self._rot_reset_btn.on_click(lambda e: self._rotation_slider.param.update(value=0.))

        self._rotation_slider.param.watch(self._apply_corrections, 'value_throttled')
        self._flip_y_cbox.param.watch(self._apply_corrections, 'value')
        self._channel_select.param.watch(self._apply_corrections, 'value')
        self._rot_reset_btn.on_click(self._apply_corrections)

        self.toolbox.extend((
            pn.Row(
                self._regression_select,
                self._flip_y_cbox,
                self._guess_corrections_btn,
            ),
            pn.Row(
                self._rot_reset_btn,
                self._rotation_slider,
                self._show_vectors_cbox,
            ),
        ))

        self._standard_layout(
            right_after=(
                self._mode_selector,
                self._radius_slider,
                self._radii_slider,
            ),
        )

    def _get_udf(self, dataset: DataSet) -> tuple[UDF, dict[str, float | str]]:
        regression = self._regression_mapping[self._regression_select.value]
        rotation = self._rotation_slider.value
        flip_y = self._flip_y_cbox.value
        mode = self._mode_selector.value
        glyph = self._ring_db.rings
        cx = self._ring_db.cds.data[glyph.x][0]
        cy = self._ring_db.cds.data[glyph.y][0]
        ri = self._ring_db.cds.data[glyph.inner_radius][0]
        ro = self._ring_db.cds.data[glyph.outer_radius][0]
        shared_params = {
            'cy': cy,
            'cx': cx,
            'regression': regression,
            'scan_rotation': rotation,
            'flip_y': flip_y,
        }
        if mode == 'Whole Frame':
            com_params = CoMParams(
                **shared_params,
            )
            result_title = 'Whole Frame CoM'
            result_name = 'frame_com'
        elif mode == 'Disk':
            com_params = CoMParams(
                **shared_params,
                r=ro,
            )
            result_title = 'Disk CoM'
            result_name = 'disk_com'
        elif mode == 'Annulus':
            com_params = CoMParams(
                **shared_params,
                r=ro,
                ri=ri,
            )
            result_title = 'Annular CoM'
            result_name = 'annular_com'
        else:
            raise ValueError(f'Unsupported mode {mode}')
        udf = CoMUDF(com_params=com_params)
        return udf, CoMParamsUI(
            params=com_params,
            result_title=result_title,
            result_name=result_name
        )

    def _update_nav_title(self, *e):
        selected = self._channel_select.value
        result_title = self._current_params.get('result_title', 'CoM')
        self.nav_plot.fig.title.text = f'{result_title} - {selected}'

    def _set_hold(self, val: bool):
        self._rotation_slider.disabled = val
        self._channel_select.disabled = val
        self._flip_y_cbox.disabled = val
        self._guess_corrections_btn.disabled = val
        self._rot_reset_btn.disabled = val

    def get_job(
        self,
        state: UIState,
        dataset: DataSet | AcquisitionProtocol,
        roi: np.ndarray | None,
    ):
        self._set_hold(True)
        return super().get_job(state, dataset, roi)

    def complete_job(self, job: UDFWindowJob, job_results: JobResults) -> tuple[ResultRow, ...]:
        self._set_hold(False)
        self._current_params = CoMParamsUI(**job.params)
        self._update_nav_title()

        result_name: str = job.params.pop('result_name')
        window_row = self.results_manager.new_window_run(
            self,
            job_results.run_id,
            params=job.params,
        )
        raw_shifts = job_results.udf_results[0]['raw_shifts'].data
        results = []
        for idx, key in {0: 'y', 1: 'x'}.items():
            rc = Numpy2DResultContainer(
                f'{result_name}_shift{key}',
                raw_shifts[..., idx],
            )
            results.append(
                self.results_manager.new_result(rc, job_results.run_id, window_row.window_id)
            )
        return tuple(results)

    def _plot_regression_x(self, udf_results: UDFResultDict, damage):
        regression = self._plot_regression(udf_results, 'x')
        return regression, damage

    def _plot_regression_y(self, udf_results: UDFResultDict, damage):
        regression = self._plot_regression(udf_results, 'y')
        return regression, damage

    def _plot_regression(
        self,
        udf_results: UDFResultDict,
        direction: Literal['x', 'y']
    ) -> np.ndarray:
        h, w = udf_results['magnitude'].data.shape
        column = 0 if direction == 'y' else 1
        ydims, xdims = np.ogrid[0: h, 0: w]
        regression = udf_results['regression'].data
        c, dy, dx = regression[:, column]
        return c + (dy * xdims) * (dx * ydims)

    def _get_corrected_live(self, idx: int, raw_shifts):
        shifts = apply_correction(
            y_centers=raw_shifts[..., 0],
            x_centers=raw_shifts[..., 1],
            scan_rotation=self.nav_plot.udf.params.com_params.scan_rotation,
            flip_y=self.nav_plot.udf.params.com_params.flip_y,
        )
        return shifts[idx]

    def _apply_corrections(self, *e):
        selected_channel = self._channel_select.value
        if selected_channel in (
            CoMChanN.REGRESSION_X,
            CoMChanN.REGRESSION_Y,
            CoMChanN.RAW_X_SHIFT,
            CoMChanN.RAW_Y_SHIFT,
            CoMChanN.SHIFT_MAGNITUDE,
        ):
            # Defer to standard channel change callback
            self.nav_plot.change_channel(selected_channel, update_title=False)
            return
        else:
            # Keep the nav plot in sync with the selected for the next run
            self.nav_plot._change_channel_attrs(selected_channel)
        try:
            udf_results, _ = self.nav_plot._last_res
        except (AttributeError, TypeError):
            return
        result_params = self._current_params.get('params', None)
        if result_params is None:
            return
        displayed_rotation = self._rotation_slider.value
        displayed_flip = self._flip_y_cbox.value
        raw_shifts = udf_results['raw_shifts'].data.copy()
        corrected_y, corrected_x = apply_correction(
            y_centers=raw_shifts[..., 0],
            x_centers=raw_shifts[..., 1],
            scan_rotation=displayed_rotation,
            flip_y=displayed_flip,
        )
        if selected_channel == CoMChanN.CORRECTED_X_SHIFT:
            im = corrected_x
        elif selected_channel == CoMChanN.CORRECTED_Y_SHIFT:
            im = corrected_y
        elif selected_channel == CoMChanN.DIVERGENCE:
            im = divergence(y_centers=corrected_y, x_centers=corrected_x)
        elif selected_channel == CoMChanN.CURL:
            im = curl_2d(y_centers=corrected_y, x_centers=corrected_x)
        else:
            raise ValueError('Unexpected channel name')
        self.nav_plot.im.update(im)
        self.nav_plot.push()
