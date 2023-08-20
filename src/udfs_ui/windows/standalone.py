from __future__ import annotations
import os
from weakref import WeakValueDictionary
from typing import TYPE_CHECKING

import panel as pn

from ..results.results_manager import ResultsManager
from ..utils.logging import logger
from .base import JobResults, UIState
from ..utils.progress import PanelProgressReporter

if TYPE_CHECKING:
    from .base import UIWindow, UDFWindowJob
    from ..ui_context import RunFromT
    from libertem.api import DataSet, Context


class StandaloneContext:
    def __init__(self, ctx: Context, dataset: DataSet, save_root: os.PathLike = '.'):
        self._results_manager = ResultsManager(save_root=save_root)
        self._ctx = ctx
        self._dataset = dataset
        # As the standalone context does not control any
        # windows which are connected to it only keep a weakref to them
        self._windows: WeakValueDictionary[str, UIWindow] = WeakValueDictionary()
        self._progress: dict[str, PanelProgressReporter] = {}

    def _register_window(self, ui_window: UIWindow):
        self._windows[self._window_ident(ui_window)] = ui_window

    @staticmethod
    def _window_ident(ui_window: UIWindow):
        # To be sure we don't keep a ref to ui_window!
        ident = '_' + ui_window.ident
        return ident[1:]

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
    def ctx(self):
        return self._ctx

    @property
    def dataset(self):
        return self._dataset

    @property
    def logger(self):
        return logger

    @property
    def results_manager(self):
        return self._results_manager

    @property
    def datset(self):
        return self._dataset

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
            async for udfs_res in self.ctx.run_udf_iter(
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
                import asyncio
                await asyncio.sleep(1.)
        except Exception as err:
            msg = 'Error during run_udf'
            self.logger.log_from_exception(err, reraise=True, msg=msg)

        run_record = self.results_manager.new_run(
            has_roi=roi is not None,
            state=UIState.OFFLINE,
            shape={
                'nav': tuple(self.dataset.shape.nav),
                'sig': tuple(self.dataset.shape.sig),
            }
        )

        # Unpack results back to their window objects
        damage = udfs_res.damage
        buffers = udfs_res.buffers
        res_iter = iter(buffers)
        for job in to_run:
            window_res = tuple(next(res_iter) for _ in job.udfs)
            job_results = JobResults(
                run_record.run_id,
                job,
                window_res,
                self.dataset.shape,
                damage=damage,
            )
            if job.result_handler is not None:
                try:
                    job.result_handler(job, job_results)
                except Exception as err:
                    msg = 'Error while unpacking result'
                    self.logger.log_from_exception(err, msg=msg)
