from __future__ import annotations
from typing import TYPE_CHECKING
from typing_extensions import Self, Literal

import numpy as np
import panel as pn
from libertem.udf.com import CoMUDF, CoMParams, RegressionOptions

from .imaging import ImagingWindow
from .base import UIType, WindowProperties
# from .result_containers import Numpy2DResultContainer

if TYPE_CHECKING:
    from libertem.io.dataset.base import DataSet
    from libertem.udf.base import UDF
    from .base import UDFWindowJob, JobResults
    from .results import ResultRow
    from libertem.udf.base import UDFResultDict


class CoMImagingWindow(ImagingWindow, ui_type=UIType.STANDALONE):
    @staticmethod
    def default_properties():
        return WindowProperties(
            'com',
            'Centre of Mass',
        )
    
    def initialize(self, dataset: DataSet) -> Self:
        super().initialize(dataset, with_layout=False)
        self.nav_plot._channel_map = {
            'shift_magnitude': 'magnitude',
            'x-shift': ('raw_shifts', lambda buffer: buffer[..., 1]),
            'y-shift': ('raw_shifts', lambda buffer: buffer[..., 0]),
            'divergence': 'divergence',
            'curl': 'curl',
            'regression_x': self._plot_regression_x,
            'regression_y': self._plot_regression_y,
        }
        self.nav_plot.get_channel_select()        

        self._regression_mapping = {
            'NO_REGRESSION': RegressionOptions.NO_REGRESSION,
            'SUBTRACT_MEAN': RegressionOptions.SUBTRACT_MEAN,
            'SUBTRACT_LINEAR': RegressionOptions.SUBTRACT_LINEAR,
        }
        self._regression_select = pn.widgets.Select(
            name='Regression',
            options=list(self._regression_mapping.keys()),
            value='NO_REGRESSION',
            width=200,
        )
        self._mode_mapping.pop('Whole Frame')
        self._mode_mapping['Whole Frame'] = self._mode_mapping.pop('Point')
        self._mode_selector.options = list(self._mode_mapping.keys())

        self._standard_layout(
            left_after=(
                self._regression_select,
            ),
            right_after=(
                self._radius_slider,
                self._radii_slider,
            ),
        )

    def _get_udf(self, dataset: DataSet) -> tuple[UDF, dict[str, float | str]]:
        regression = self._regression_mapping[self._regression_select.value]
        mode = self._mode_selector.value
        glyph = self._ring_db.rings
        cx = self._ring_db.cds.data[glyph.x][0]
        cy = self._ring_db.cds.data[glyph.y][0]
        ri = self._ring_db.cds.data[glyph.inner_radius][0]
        ro = self._ring_db.cds.data[glyph.outer_radius][0]
        params = {'cy': cy, 'cx': cx, 'regression': regression}
        if mode == 'Whole Frame':
            com_params = CoMParams(
                cy=cy,
                cx=cx,                
                regression=regression,
            )
            params['mode'] = 'whole_frame'
            result_title = 'Whole Frame CoM'
            result_name = 'frame_com'
        elif mode == 'Disk':
            com_params = CoMParams(
                cy=cy,
                cx=cx,
                r=ro,
                regression=regression,
            )
            params['r'] = ro
            params['mode'] = 'disk'
            result_title = 'Disk CoM'
            result_name = 'disk_com'
        elif mode == 'Annulus':
            com_params = CoMParams(
                cy=cy,
                cx=cx,
                r=ro,
                ri=ri,
                regression=regression,
            )
            params['ro'] = ro
            params['ri'] = ri
            params['mode'] = 'annulus'
            result_title = 'Annular CoM'
            result_name = 'annular_com'
        else:
            raise ValueError(f'Unsupported mode {mode}')
        udf = CoMUDF(com_params)
        return udf, {**params, 'result_title': result_title, 'result_name': result_name}

    def complete_job(self, job: UDFWindowJob, job_results: JobResults) -> tuple[ResultRow, ...]:
        result_title: str = job.params.pop('result_title')
        self.nav_plot.fig.title.text = result_title
        return tuple()
        # result_name: str = job.params.pop('result_name')
        # window_row = self.results_manager.new_window_run(
        #     self,
        #     job_results.run_id,
        #     params=job.params,
        # )
        # channel: str = self.nav_plot.channel
        # buffer = job_results.udf_results[0][channel]
        # image: np.ndarray = buffer.data.squeeze()
        # rc = Numpy2DResultContainer(
        #     result_name,
        #     image,
        #     meta={
        #         'channel': channel,
        #     },
        #     title=result_title,
        # )
        # rc.tag_as(buffer.kind)
        # result = self.results_manager.new_result(rc, job_results.run_id, window_row.window_id)
        # self.nav_plot.displayed = result
        # return (result,)


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
