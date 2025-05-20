from __future__ import annotations
from functools import partial
import numpy as np
from typing import TYPE_CHECKING

import panel as pn
from libertem.udf.raw import PickUDF
from libertem.udf.sumsigudf import SumSigUDF

from ..live_plot import AperturePlot
from ..base import UIState, JobResults
from .base import UIWindow, UDFWindowJob, WindowProperties
from ..display.display_base import Cursor
from ..utils import get_initial_pos, PointYX


if TYPE_CHECKING:
    from libertem_live.detectors.base.acquisition import AcquisitionProtocol
    from libertem.udf.base import UDF
    from ..results.results_manager import ResultRow
    from libertem.api import DataSet


class PickUDFBaseWindow(UIWindow):
    pick_cls = PickUDF

    def _pick_base(self, dataset: DataSet):
        self._udf_pick = self.pick_cls()
        roi = np.zeros(dataset.shape.nav, dtype=bool)
        roi[0, 0] = True
        self.sig_plot = AperturePlot.new(
            dataset,
            self._udf_pick,
            roi=roi,
            channel='intensity',
            title=self._pick_title(),
        )
        self._last_pick: PointYX | None = None

        self.nav_plot = AperturePlot.new(
            dataset,
            SumSigUDF(),
            title='Scan grid',
        )

        (ny, nx), _, _ = get_initial_pos(dataset.shape.nav)
        self._nav_cursor = Cursor.new().from_pos(nx, ny)
        self._nav_cursor.on(self.nav_plot.fig)
        self._nav_cursor.editable()
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
                self.nav_plot.layout,
                self.toolbox,
                *left_after,
            ),
            pn.Column(
                *right_before,
                self.sig_plot.layout,
                *right_after,
            )
        ))

    def _cds_pick_job(
        self,
        state: UIState,
        dataset: DataSet | AcquisitionProtocol,
        roi: np.ndarray | None,
        quiet: bool = True,
        with_udfs: list[UDF] | None = None,
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

        if with_udfs is None:
            with_udfs = [self._udf_pick]

        return UDFWindowJob(
            self,
            with_udfs,
            [self.sig_plot],
            result_handler=self._complete_cds_pick_job,
            params=params,
            roi=roi,
            quiet=quiet,
        )

    def reset_title(self):
        self.sig_plot.fig.title.text = self._pick_title(cyx=self._last_pick)

    def _pick_title(
        self,
        cyx: tuple[int, int] | None = None,
        suffix: str | None = None,
        title_stub: str = 'Pick frame',
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
        with_push: bool = True,
    ) -> tuple[ResultRow, ...]:
        if not job.udfs:
            return tuple()

        cy, cx = job.params['cy'], job.params['cx']
        self._last_pick = PointYX(cy, cx)
        self.reset_title()
        if with_push:
            self.sig_plot.push()
        return tuple()


class PickUDFWindow(PickUDFBaseWindow):
    @staticmethod
    def default_properties():
        return WindowProperties(
            'pick_frame',
            'PickUDF',
            self_run_only=True,
            header_activate=False,
        )

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
