from __future__ import annotations
from functools import partial
import numpy as np
from typing import TYPE_CHECKING

import panel as pn
import libertem.api as lt
from libertem.udf.raw import PickUDF
from libertem.udf.sumsigudf import SumSigUDF

from .live_plot import AperturePlot
from .base import UIWindow, UIType, UIState, UDFWindowJob, JobResults
from .result_tracker import ImageResultTracker
from ..display.display_base import Cursor
from ..utils import get_initial_pos


if TYPE_CHECKING:
    from libertem_live.detectors.base.acquisition import AcquisitionProtocol
    from .results import ResultRow


class PickUDFBaseWindow(UIWindow):
    def _pick_base(self, dataset: lt.DataSet):
        self._udf_pick = self.pick_cls()
        roi = np.zeros(dataset.shape.nav, dtype=bool)
        roi[0, 0] = True
        self.sig_plot = AperturePlot.new(
            dataset,
            self._udf_pick,
            roi=roi,
            channel=('intensity', lambda buffer: buffer.squeeze()),
            title='Pick frame',
        )
        self._last_pick = (None, None)
        self._udf_plots = [self.sig_plot]

        self.nav_plot = AperturePlot.new(
            dataset,
            SumSigUDF(),
            title='Scan grid',
        )

        (ny, nx), _, _ = get_initial_pos(dataset.shape.nav)
        self._nav_cursor = Cursor.new().from_pos(nx, ny)
        self._nav_cursor.on(self.nav_plot.fig)
        self._nav_cursor.make_editable()
        self._nav_cursor.cds.on_change(
            'data',
            partial(self.run_this_bk, run_from=self._cds_pick_job),
        )
        self.nav_plot.fig.toolbar.active_drag = self.nav_plot.fig.tools[-1]
        self.toolbox = pn.Column()

    def _standard_layout(self):
        self.inner_layout.extend((
            pn.Column(
                self.nav_plot.pane,
                self.toolbox,
            ),
            pn.Column(
                self.sig_plot.pane
            )
        ))

    @staticmethod
    def _should_pick(ds: lt.DataSet, data: dict):
        try:
            x = int(data['cx'][0])
            y = int(data['cy'][0])
        except (KeyError, IndexError):
            return False
        h, w = ds.shape.nav
        if not ((0 <= x < w) and (0 <= y < h)):
            return False
        return (x, y)

    def _cds_pick_job(
        self,
        state: UIState,
        dataset: lt.DataSet | AcquisitionProtocol,
        roi: np.ndarray | None,
    ):
        if state == UIState.LIVE:
            return

        # Proceed even if ROI is not None, if this is the only window
        # then the global ROI, if any, will be ignored
        # if roi is not None:
        #     return

        coords = self._should_pick(dataset, self._nav_cursor.cds.data)
        if not coords:
            return
        cx, cy = coords

        # import sparse
        # roi = sparse.COO(
        #     np.asarray([(cy,), (cx,)]),
        #     True,
        #     shape=dataset.shape.nav,
        # )
        # sparse deactivated because of import / first-run lag
        roi = np.zeros(
            dataset.shape.nav,
            dtype=bool,
        )
        roi[cy, cx] = True

        params = {
            'cx': cx,
            'cy': cy,
        }

        return UDFWindowJob(
            self,
            [self._udf_pick],
            self._udf_plots,
            self._complete_cds_pick_job,
            params=params,
            roi=roi,
        )

    def _complete_cds_pick_job(
        self,
        job: UDFWindowJob,
        job_results: JobResults,
    ) -> tuple[ResultRow, ...]:
        if not job.udfs:
            return tuple()

        cy, cx = job.params['cy'], job.params['cx']
        self._last_pick = (
            job_results.udf_results[0]['intensity'].data.squeeze(axis=0),
            {'cx': cx, 'cy': cy},
        )
        self.sig_plot.fig.title.text = f'{self.sig_plot.title} - {(cy, cx)}'

        # Pick frame saving needs re-working to
        # avoid piling up lots of frames
        # now that everything goes via a job
        # Probably save all necessary to run the below
        # lines, but only run then when the save button is pressed
        # remember to set the 'sig' tag !

        # window_row = self.results_manager.new_window_run(self, run_id, params=job.params)
        # image: np.ndarray = results[0]['intensity'].data.squeeze(axis=0)
        # rc = Numpy2DResultContainer('intensity', image)
        # result = self.results_manager.new_result(rc, run_id, window_row.window_id)

        return tuple()  # (result,)


class PickUDFWindow(PickUDFBaseWindow, ui_type=UIType.TOOL):
    name = 'pick_frame'
    title_md = 'PickUDF'
    pick_cls = PickUDF
    self_run_only = True
    header_activate = False

    def initialize(self, dataset: lt.DataSet):
        self._pick_base(dataset)
        self._standard_layout()

        self.nav_plot_tracker = ImageResultTracker(
            self,
            self.nav_plot,
            ('nav',),
            'Nav image',
        )
        self.nav_plot_tracker.initialize()

        return self

    def on_results_registered(
        self,
        *results: ResultRow,
    ):
        self.nav_plot_tracker.on_results_registered(*results)

    def on_results_deleted(
        self,
        *results: ResultRow,
    ):
        self.nav_plot_tracker.on_results_deleted(*results)

    def get_job(
        self,
        state: UIState,
        dataset: lt.DataSet | AcquisitionProtocol,
        roi: np.ndarray | None,
    ):
        return self._cds_pick_job(state, dataset, roi)

    def complete_job(
        self,
        job: UDFWindowJob,
        job_results: JobResults,
    ):
        return self._complete_cds_pick_job(job, job_results)
