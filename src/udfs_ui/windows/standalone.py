from __future__ import annotations
from weakref import WeakValueDictionary
from typing import TYPE_CHECKING

import panel as pn

from ..results.results_manager import ResultsManager
from ..utils.logging import logger
from ..base import UIContextBase, UIState
from ..utils.progress import PanelProgressReporter

if TYPE_CHECKING:
    from .base import UIWindow, UDFWindowJob
    from ..ui_context import RunFromT
    from libertem.api import DataSet, Context


class StandaloneContext(UIContextBase):
    def __init__(self, ctx: Context, dataset: DataSet):
        self._results_manager = ResultsManager()
        self._results_manager.add_watcher(self)
        self._ctx = ctx
        self._dataset = dataset
        # As the standalone context does not control any
        # windows which are connected to it only keep a weakref to them
        self._windows: WeakValueDictionary[str, UIWindow] = WeakValueDictionary()
        self._progress: dict[str, PanelProgressReporter] = {}

    def _add_progress_bar(self, ui_window: UIWindow, insert_at: int = 1):
        pbar = pn.widgets.Tqdm(
            width=650,
            align=('start', 'center'),
            margin=(5, 5, 0, 5),
        )
        ident = self._window_ident(ui_window)
        if ident in self._progress.keys():
            raise RuntimeError('Can only create one progress bar per window')
        self._progress[ident] = PanelProgressReporter(pbar)
        # between _header_layout and _inner_layout
        ui_window.layout().insert(insert_at, pbar)

    @property
    def dataset(self):
        return self._dataset

    @property
    def logger(self):
        return logger

    @property
    def results_manager(self):
        return self._results_manager

    async def _run_job(self, run_from: list[RunFromT]):
        to_run: list[UDFWindowJob] = [
            job for job_getter in run_from
            if (job := job_getter(UIState.OFFLINE, self.dataset, None))
            is not None
        ]
        if len(to_run) == 0:
            return
        elif len(to_run) > 1:
            # Mainly due to ROI negotiation...
            raise NotImplementedError
        else:
            job = to_run[0]
        roi = job.roi
        progress = False if job.quiet else self._progress.get(job.window.ident, False)
        try:
            async for udfs_res in self._ctx.run_udf_iter(
                dataset=self.dataset,
                udf=tuple(udf for job in to_run for udf in job.udfs),
                plots=tuple(plot for job in to_run for plot in job.plots),
                progress=progress,
                sync=False,
                roi=roi,
            ):
                if job.window.should_stop():
                    self.logger.info('Job asked for early stop')
                    break
        except Exception as err:
            msg = 'Error during run_udf'
            self.logger.log_from_exception(err, reraise=True, msg=msg)

        run_record = self.results_manager.new_run(
            shape={
                'nav': tuple(self.dataset.shape.nav),
                'sig': tuple(self.dataset.shape.sig),
            },
            has_roi=roi is not None,
            state=UIState.OFFLINE,
        )

        all_results = self.unpack_results(udfs_res, to_run, run_record)
        self.notify_new_results(*all_results)
