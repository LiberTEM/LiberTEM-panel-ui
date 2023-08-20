from __future__ import annotations
from typing import Protocol, NewType, TYPE_CHECKING, Callable

from strenum import StrEnum

if TYPE_CHECKING:
    import os
    import pathlib
    import numpy as np

    from libertem_live.detectors.base.acquisition import AcquisitionProtocol
    from libertem.io.dataset.base import DataSet

    from .windows.base import UIWindow, UDFWindowJob, JobResults
    from .results.results_manager import ResultsManager, ResultRow
    from .utils.logging import UILogger

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


class UIContextBase(Protocol):
    _windows: dict[WindowIdent, UIWindow]
    _results_manager: ResultsManager

    @property
    def logger(self) -> UILogger:
        ...

    @property
    def datset(self) -> DataSet | AcquisitionProtocol:
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
