from __future__ import annotations
from typing import Protocol, NewType, TYPE_CHECKING, Callable, NamedTuple

from strenum import StrEnum

if TYPE_CHECKING:
    import os
    import pathlib
    import numpy as np

    from libertem_live.detectors.base.acquisition import AcquisitionProtocol
    from libertem.io.dataset.base import DataSet
    from libertem.udf.base import UDFResults, UDFResultDict, BufferWrapper

    from .windows.base import UIWindow, UDFWindowJob
    from .results.results_manager import ResultsManager, ResultRow, RunRow
    from .utils.logging import UILogger
    from .resources import LiveResources, OfflineResources
    from .ui_context import UITools

    WindowIdent = NewType('WindowIdent', str)

    RunFromT = Callable[
        [
            'UIState',
            DataSet | AcquisitionProtocol,
            np.ndarray | None,
        ],
        'UDFWindowJob' | None
    ]

    ResultHandlerT = Callable[
        [
            'UDFWindowJob',
            'JobResults',
        ],
        tuple[ResultRow, ...]
    ]


class UIState(StrEnum):
    OFFLINE = 'Offline'
    LIVE = 'Live'
    REPLAY = 'Replay'


class JobResults(NamedTuple):
    run_row: RunRow
    job: UDFWindowJob
    udf_results: tuple[UDFResultDict]
    damage: BufferWrapper | None = None


class UIContextBase(Protocol):
    _windows: dict[WindowIdent, UIWindow]
    _results_manager: ResultsManager
    _resources: LiveResources | OfflineResources
    _tools: UITools

    @staticmethod
    def notebook_fullwidth():
        from .utils.notebook_tools import notebook_fullwidth
        notebook_fullwidth()

    @property
    def logger(self) -> UILogger:
        ...

    def _register_window(self, ui_window: UIWindow):
        self._windows[self._window_ident(ui_window)] = ui_window

    def _remove_window(self, window: UIWindow):
        ...

    @staticmethod
    def _window_ident(ui_window: UIWindow) -> WindowIdent:
        # To be sure we don't keep a ref to ui_window!
        ident = '_' + ui_window.ident
        return ident[1:]

    @property
    def results_manager(self) -> ResultsManager:
        return self._results_manager

    @property
    def save_root(self) -> pathlib.Path:
        return self.results_manager.save_root

    def change_save_root(self, save_root: os.PathLike):
        self.results_manager.change_save_root(save_root)

    def notify_new_results(self, *results: ResultRow):
        if not results:
            return
        for window in self._windows.values():
            window.on_results_registered(*results)
            for tracker in window.trackers.values():
                if tracker.auto_update:
                    tracker.on_results_registered(*results)

    def notify_deleted_results(self, *results: ResultRow):
        if not results:
            return
        for window in self._windows.values():
            window.on_results_deleted(*results)
            for tracker in window.trackers.values():
                if tracker.auto_update:
                    tracker.on_results_deleted(*results)

    async def _run_job(self, run_from: list[RunFromT]):
        ...

    def unpack_results(
        self,
        udfs_res: UDFResults,
        jobs: list[RunFromT],
        run_record: RunRow,
    ):
        # Unpack results back to their window objects
        damage = udfs_res.damage
        buffers = udfs_res.buffers
        res_iter = iter(buffers)
        all_results: list[ResultRow] = []
        for job in jobs:
            window_res = tuple(next(res_iter) for _ in job.udfs)
            job_results = JobResults(
                run_record,
                job,
                window_res,
                damage=damage,
            )
            if job.result_handler is not None:
                try:
                    result_entries = job.result_handler(job, job_results)
                except Exception as err:
                    msg = 'Error while unpacking result'
                    self.logger.log_from_exception(err, msg=msg)
                    continue
                all_results.extend(result_entries)
        return all_results
