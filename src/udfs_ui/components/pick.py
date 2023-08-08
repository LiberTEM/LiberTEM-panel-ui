from __future__ import annotations
from functools import partial
import numpy as np
from typing import TYPE_CHECKING
from typing_extensions import Literal

import panel as pn
from libertem.udf.raw import PickUDF
from libertem.udf.sumsigudf import SumSigUDF

from .live_plot import AperturePlot
from .base import UIWindow, UIState, UDFWindowJob, JobResults
from ..display.display_base import Cursor
from ..utils import get_initial_pos


if TYPE_CHECKING:
    from libertem_live.detectors.base.acquisition import AcquisitionProtocol
    from .results import ResultRow
    from libertem.api import DataSet


class PickUDFBaseWindow(UIWindow):
    def _pick_base(self, dataset: DataSet):
        try:
            self._udf_pick = self.pick_cls()
        except AttributeError:
            self._udf_pick = PickUDF()
        roi = np.zeros(dataset.shape.nav, dtype=bool)
        roi[0, 0] = True
        self.sig_plot = AperturePlot.new(
            dataset,
            self._udf_pick,
            roi=roi,
            channel='intensity',
            title=self._pick_title(),
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

    def _standard_layout(
        self,
        left_before=(),
        left_after=(),
        right_before=(),
        right_after=(),
    ):
        self.inner_layout.extend((
            pn.Column(
                *left_before,
                self.nav_plot.pane,
                self.toolbox,
                *left_after,
            ),
            pn.Column(
                *right_before,
                self.sig_plot.pane,
                *right_after,
            )
        ))

    def _cds_pick_job(
        self,
        state: UIState,
        dataset: DataSet | AcquisitionProtocol,
        roi: np.ndarray | None,
        quiet: bool = True,
    ):
        if state == UIState.LIVE:
            return

        # Proceed even if ROI is not None, if this is the only window
        # then the global ROI, if any, will be ignored
        # if roi is not None:
        #     return

        coords = self._nav_cursor.current_pos(
            to_int=True,
            clip_to=dataset.shape.nav,
        )
        if coords is None:
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
            quiet=quiet,
        )

    def reset_title(self):
        _, params = self._last_pick
        cyx = (params['cy'], params['cx'])
        self.sig_plot.fig.title.text = self._pick_title(cyx)

    def _pick_title(
        self,
        cyx: tuple[int, int] | None = None,
        suffix: str | None = None,
        title_stub: str = f'Pick frame',
    ):
        if cyx is None:
            return title_stub
        cy, cx = cyx
        pad_suffix = ''
        if suffix is not None:
            pad_suffix = f' {suffix}'
        return f'{title_stub} (y={cy}, x={cx}){pad_suffix}'

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
        self.sig_plot.fig.title.text = self._pick_title((cy, cx))
        pn.io.push_notebook(self.sig_plot.pane)

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


class PickUDFWindow(PickUDFBaseWindow):
    name = 'pick_frame'
    title_md = 'PickUDF'
    pick_cls = PickUDF
    self_run_only = True
    header_activate = False

    def initialize(self, dataset: DataSet, with_layout: bool = True):
        self._pick_base(dataset)
        if with_layout:
            self._standard_layout()
        self.link_image_plot('Nav', self.nav_plot, ('nav',))
        return self

    def get_job(
        self,
        state: UIState,
        dataset: DataSet | AcquisitionProtocol,
        roi: np.ndarray | None,
    ):
        return self._cds_pick_job(state, dataset, roi)

    def complete_job(
        self,
        job: UDFWindowJob,
        job_results: JobResults,
    ):
        return self._complete_cds_pick_job(job, job_results)
