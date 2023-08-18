from __future__ import annotations
import os
from typing import TYPE_CHECKING

from ..results.results_manager import ResultsManager
from ..utils.logging import logger
from .base import JobResults, UIState

if TYPE_CHECKING:
    from ..ui_context import RunFromT
    from libertem.api import DataSet, Context


class StandaloneContext:
    def __init__(self, ctx: Context, dataset: DataSet, save_root: os.PathLike = '.'):
        self._results_manager = ResultsManager(save_root=save_root)
        self._ctx = ctx
        self._dataset = dataset

    @property
    def ctx(self):
        return self._ctx

    @property
    def dataset(self):
        return self._dataset

    @property
    def logger():
        return logger

    @property
    def results_manager(self):
        return self._results_manager

    async def _run_job(self, run_from: list[RunFromT]):
        to_run = [
            job for job_getter in run_from
            if (job := job_getter(UIState.OFFLINE, self.dataset, None))
            is not None
        ]
        if len(to_run) == 0:
            return
        roi = to_run[0].roi
        try:
            async for udfs_res in self.ctx.run_udf_iter(
                dataset=self.dataset,
                udf=tuple(udf for job in to_run for udf in job.udfs),
                plots=tuple(plot for job in to_run for plot in job.plots),
                progress=False,
                sync=False,
                roi=roi,
            ):
                pass
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
