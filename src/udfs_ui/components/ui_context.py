from __future__ import annotations
import asyncio
import uuid
import time
import numpy as np
import panel as pn
from typing import Callable, TYPE_CHECKING, Type
import libertem.api as lt
import libertem_live.api as ltl

from .progress import PanelProgressReporter
from .base import UIWindow, RunnableUIWindow, UIType, UIState, UDFWindowJob
from .lifecycles import (
    UILifecycle,
    OfflineLifecycle,
    LiveLifecycle,
    ReplayLifecycle,
    ContinuousLifecycle,
)
from .resources import LiveResources, OfflineResources
from .result_containers import Numpy2DResultContainer
from .tools import ROIWindow
from .results import ResultsManager, ResultRow
from .terminal_logger import UILog
from ..utils.notebook_tools import get_ipyw_reload_button

if TYPE_CHECKING:
    from libertem_live.detectors.base.acquisition import AcquisitionProtocol
    from libertem.udf.base import UDFResults
    from libertem_live.api import LiveContext


class UITools:
    def __init__(self, ui: UIContext):
        self.ui = ui

        common_params = dict(
            width_policy='min',
            align='center',
            min_width=125,
        )

        self.title = pn.pane.Markdown(
            object='## UDFs UI',
        )

        self.run_btn = pn.widgets.Button(
            name='Run',
            button_type='success',
            **common_params,
        )

        self.mode_btn = pn.widgets.Button(
            name='Replay data',
            button_type='primary',
            **common_params,
        )

        self.stop_btn = pn.widgets.Button(
            name='STOP',
            button_type='danger',
            disabled=True,
            **common_params,
        )

        self.continuous_btn = pn.widgets.Button(
            name='Run continuous',
            button_type='success',
            **common_params,
        )

        self.add_btn = pn.widgets.Button(
            name='Add analysis',
            button_type='primary',
            **common_params,
        )

        self.add_tool_btn = pn.widgets.Button(
            name='Add tool',
            button_type='primary',
            **common_params,
        )

        common_params['min_width'] = 175

        udf_keys = list(UIWindow.get_implementations(UIType.ANALYSIS).keys())
        self.add_analysis_dropdown = pn.widgets.Select(
            value=udf_keys[0],
            options=udf_keys,
            **common_params,
        )

        tool_keys = list(UIWindow.get_implementations(UIType.TOOL).keys())
        self.add_tool_dropdown = pn.widgets.Select(
            value=tool_keys[0],
            options=tool_keys,
            **common_params,
        )

        self.replay_select = pn.widgets.Select(
            name='Dataset',
            options=[],
            **common_params,
        )

        self.pbar = pn.widgets.Tqdm(
            width=650,
            align=('start', 'center'),
        )

    def set_subtitle(self, subtitle: str):
        self.title.object = f'## UDFs UI - {subtitle}'


class UIContext:
    def __init__(self):
        self._windows: dict[str, UIWindow] = {}
        self._tools = UITools(self)
        self._p_reporter = PanelProgressReporter(self._tools.pbar)
        self._button_row = pn.Row()
        self._add_window_row = pn.Row()
        self._layout = pn.Column(
            self._button_row,
            self._tools.pbar,
            self._add_window_row,
            min_width=700,
        )
        self._state: UIState = None
        self._resources: OfflineResources | LiveResources = None
        self._run_lock = asyncio.Lock()
        self._continue_running = False
        self._continuous_acquire = False
        self._removed_from_options: dict[str, pn.widgets.Select] = {}
        self._results_manager = ResultsManager()
        self._logger = UILog()

    def log_window(self, with_reload: bool = True):
        if with_reload:
            get_ipyw_reload_button()
        return self._logger.widget

    @property
    def logger(self):
        return self._logger.logger

    def for_live(
        self,
        live_ctx: LiveContext,
        get_aq: Callable[[LiveContext], AcquisitionProtocol | None],
        offline_ctx: lt.Context | None = None,
        aq_plan: AcquisitionProtocol | None = None,
    ):
        if not isinstance(live_ctx, ltl.LiveContext):
            raise TypeError('Cannot instantiate live UIContext '
                            f'with Context of type {type(live_ctx)}')
        self._resources = LiveResources(
            live_ctx=live_ctx,
            get_aq=get_aq,
            offline_ctx=offline_ctx,
            recordings={},
            aq_plan=aq_plan,
        )
        self._state = UIState.LIVE
        self._inital_setup()
        self._set_state(UIState.LIVE)
        return self

    def for_offline(
        self,
        ctx: lt.Context,
        ds: lt.DataSet,
    ):
        if not isinstance(ctx, lt.Context):
            raise TypeError(f'Cannot instantiate UIContext with Context of type {type(ctx)}')
        if not isinstance(ds, lt.DataSet):
            raise TypeError(f'Cannot instantiate UIContext on dataset of type {type(ds)}')
        self._resources = OfflineResources(
            ctx=ctx,
            ds=ds,
        )
        self._state = UIState.OFFLINE
        self._inital_setup()
        self._set_state(UIState.OFFLINE)
        return self

    def layout(self, with_reload: bool = True):
        if with_reload:
            get_ipyw_reload_button()
        if self._state is None:
            raise ValueError('Must initialize UI before displaying')
        return self._layout

    @property
    def results_manager(self):
        return self._results_manager

    def add(
        self,
        window_cls: Type[UIWindow],
    ) -> UIWindow:
        window_id = str(uuid.uuid4())[:5]
        window: UIWindow = window_cls(self, window_id)
        self._windows[window_id] = window
        assert window.ident == window_id
        self._layout.append(window.layout())
        self.logger.info(f'Added window {window.title_md} - {window.ident}')
        window.initialize(
            self._resources.get_ds_for_init(self._state, self.current_ds_ident)
        )
        return window

    def _remove(self, window: UIWindow):
        index = tuple(i for i, _lo
                      in enumerate(self._layout)
                      if hasattr(_lo, 'ident') and _lo.ident == window.ident)
        for i in reversed(index):
            self._layout.pop(i)
        self._windows.pop(window.ident, None)
        self.logger.info(f'Removed window {window.title_md} - {window.ident}')
        if window.is_unique and window.name in self._removed_from_options:
            dropdown = self._removed_from_options.pop(window.name)
            dropdown.options = dropdown.options + [window.name]

    def _set_state(self, new_state: UIState, *e):
        if new_state == UIState.REPLAY and not self._resources.recordings:
            # Don't switch to replay mode unless we have recordings
            self.logger.info('Cannot switch to replay state without recorded data')
            return
        self.logger.info(f'Set UI-state {new_state.value.upper()}')
        old_state = self._state
        self._state = new_state
        self._setup_tools()
        for w in self._windows.values():
            w.set_state(old_state, new_state)

    def _inital_setup(self):
        base_controls_buttons, base_controls_tools = self._controls()
        self._button_row.extend(base_controls_buttons)
        self._add_window_row.extend(base_controls_tools)

        self._tools.add_btn.on_click(self._add_analysis_handler)
        self._tools.add_tool_btn.on_click(self._add_tool_handler)
        self._tools.run_btn.on_click(self._run_handler)
        self._tools.stop_btn.on_click(self._stop_handler)
        if self._state in (UIState.LIVE, UIState.REPLAY):
            self._tools.mode_btn.on_click(self._mode_handler)
            self._tools.continuous_btn.on_click(self._continuous_handler)

    async def _run_handler(
        self,
        *e,
        run_continuous: bool = False,
        windows: list[RunnableUIWindow] | None = None
    ):
        if self._run_lock.locked():
            return
        self._continue_running = True
        async with self._run_lock:
            if run_continuous:
                await self.run_continuous(*e, windows=windows)
            elif self._state == UIState.OFFLINE:
                await self.run_offline(*e, windows=windows)
            elif self._state == UIState.LIVE:
                await self.run_live(*e, windows=windows)
            elif self._state == UIState.REPLAY:
                await self.run_replay(*e, windows=windows)

    async def _add_analysis_handler(self, *e):
        dropdown = self._tools.add_analysis_dropdown
        mapper = UIWindow.get_implementations(UIType.ANALYSIS)
        await self._add_from_dropdown(dropdown, mapper)

    async def _add_tool_handler(self, *e):
        dropdown = self._tools.add_tool_dropdown
        mapper = UIWindow.get_implementations(UIType.TOOL)
        await self._add_from_dropdown(dropdown, mapper)

    async def _add_from_dropdown(self, dropdown, mapper):
        window_cls = mapper[dropdown.value]
        try:
            window = self.add(
                window_cls
            )
        except Exception as err:
            self._logger.log_from_exception(err, reraise=True)
        if window.is_unique:
            dropdown.options = [o for o in dropdown.options if o != window.name]
            self._removed_from_options[window.name] = dropdown

    async def _mode_handler(self, *e):
        if self._state == UIState.LIVE:
            self._set_state(UIState.REPLAY)
        elif self._state == UIState.REPLAY:
            self._set_state(UIState.LIVE)

    async def _stop_handler(self, *e):
        if self._continue_running:
            self._continue_running = False

    async def _continuous_handler(self, *e):
        if self._continuous_acquire:
            self._continuous_acquire = False
            # Should be on the lifecycle class, really!
            # Would need an additional method
            self._tools.continuous_btn.name = 'Stopping...'
            return
        await self._run_handler(*e, run_continuous=True)

    def _setup_tools(self):
        if self._state == UIState.OFFLINE:
            self._setup_offline()
        elif self._state == UIState.LIVE:
            self._setup_live()
        elif self._state == UIState.REPLAY:
            self._setup_replay()

    def get_recordings_map(self):
        return {str(p.stem): ident
                for ident, p
                in self._resources.recordings.items()
                if p.is_file()}

    @property
    def current_ds_ident(self) -> str | None:
        if self._state == UIState.REPLAY:
            return self.get_recordings_map()[self._tools.replay_select.value]
        return None

    def _setup_live(self):
        LiveLifecycle(self).setup()

    def _setup_replay(self):
        ReplayLifecycle(self).setup()
        self._tools.replay_select.options = [*self.get_recordings_map().keys()]

    def _setup_offline(self):
        OfflineLifecycle(self).setup()

    def _controls(self):
        self._tools.title.object = f'## UDFs UI - {self._state.value}'
        button_row = [
            self._tools.title,
            self._tools.run_btn,
            self._tools.stop_btn,
        ]
        tool_row = [
            self._tools.add_btn,
            self._tools.add_analysis_dropdown,
            self._tools.add_tool_btn,
            self._tools.add_tool_dropdown,
        ]
        if self._state in (UIState.REPLAY, UIState.LIVE):
            button_row.insert(2, self._tools.continuous_btn)
            button_row.append(self._tools.mode_btn)
            button_row.append(self._tools.replay_select)
        return button_row, tool_row

    def get_roi(self, ds: lt.DataSet) -> np.ndarray | None:
        # Get an ROI from an roi window if present and roi is set
        roi = None
        for w in self._windows.values():
            if isinstance(w, ROIWindow):
                roi = w.get_roi(ds)
                break  # only take the first, should be unique
        return roi

    @property
    def runnable_windows(self) -> dict[str, RunnableUIWindow]:
        return {k: w for k, w in self._windows.items() if isinstance(w, RunnableUIWindow)}

    async def run_live(self, *e, windows: list[RunnableUIWindow] | None = None):
        lifecycle = LiveLifecycle(self)
        lifecycle.before()
        live_ctx = self._resources.get_ctx(self._state)
        try:
            if (aq := self._resources.get_ds_for_run(self._state,
                                                     self.current_ds_ident)) is not None:
                await self._run(live_ctx, aq, lifecycle, windows=windows)
        finally:
            lifecycle.after()

    async def run_continuous(self, *e, windows: list[RunnableUIWindow] | None = None):
        lifecycle = ContinuousLifecycle(self)
        lifecycle.before()
        live_ctx = self._resources.get_ctx(self._state)
        self._continuous_acquire = True
        try:
            while self._continuous_acquire and self._continue_running:
                if (aq := self._resources.get_ds_for_run(self._state,
                                                         self.current_ds_ident)) is not None:
                    await self._run(live_ctx, aq, lifecycle, windows=windows)
        finally:
            self._continuous_acquire = False
            lifecycle.after()

    async def run_replay(self, *e, windows: list[RunnableUIWindow] | None = None):
        lifecycle = ReplayLifecycle(self)
        lifecycle.before()
        ctx = self._resources.get_ctx(self._state)
        ds = self._resources.get_ds_for_run(self._state, self.current_ds_ident)
        roi = self.get_roi(ds)

        try:
            await self._run(
                ctx,
                ds,
                lifecycle,
                roi=roi,
                windows=windows,
            )
        finally:
            lifecycle.after()

    async def run_offline(self, *e, windows: list[RunnableUIWindow] | None = None):
        lifecycle = OfflineLifecycle(self)
        lifecycle.before()

        ctx = self._resources.get_ctx(self._state)
        ds = self._resources.get_ds_for_run(self._state, self.current_ds_ident)
        roi = self.get_roi(ds)

        try:
            await self._run(ctx, ds, lifecycle, roi=roi, windows=windows)
        finally:
            lifecycle.after()

    async def _run(
        self,
        ctx: lt.Context | LiveContext,
        ds: lt.DataSet | AcquisitionProtocol,
        ui_lifecycle: UILifecycle,
        roi: np.ndarray | None = None,
        windows: list[RunnableUIWindow] | None = None,
    ):

        if ds is None:
            self.logger.error('Dataset is None')
            return

        if windows is None:
            windows = list(self.runnable_windows.values())

        to_run: list[UDFWindowJob] = [job for window in windows
                                      if (job := window.get_job(self._state, ds, roi))
                                      is not None]

        if not to_run:
            self.logger.info('No jobs to run')
            return

        self.logger.info(f'Start run, state {self._state.value.upper()} '
                         f'on {len(to_run)} jobs{" with roi" if roi is not None else ""}')

        # maybe do some backend optimisation to reduce the number
        # of SumUDF, LogsumUDF instances that run? Might need
        # to be careful about plots, though
        tstart = time.time()
        part_completed = 0
        try:
            async for udfs_res in ctx.run_udf_iter(
                dataset=ds,
                udf=tuple(udf for job in to_run for udf in job.udfs),
                plots=tuple(plot for job in to_run for plot in job.plots),
                progress=self._p_reporter,
                sync=False,
                roi=roi,
            ):
                part_completed += 1
                ui_lifecycle.during()
                if not self._continue_running:
                    self.logger.info(f'Early stop of UDF run after {part_completed} '
                                     'partitions completed')
                    # Must handle not awaiting for full acqisition!
                    break
        except Exception as err:
            self._logger.log_from_exception(err, reraise=True)

        if self._continue_running:
            self.logger.info(f'End run, completed in {time.time() - tstart:.3f} seconds')

        run_record = self.results_manager.new_run(
            has_roi=roi is not None,
            state=self._state.value,
            shape={
                'nav': tuple(ds.shape.nav),
                'sig': tuple(ds.shape.sig),
            }
        )
        self.logger.info(f'Results saved with run_id: {run_record.run_id}')
        if roi is not None:
            rc = Numpy2DResultContainer('ROI', roi)
            self.results_manager.new_result(rc, run_record)

        # Unpack results back to their window objects
        udfs_res: UDFResults
        damage = udfs_res.damage
        buffers = udfs_res.buffers
        res_iter = iter(buffers)
        all_results: list[ResultRow] = []
        for job in to_run:
            window_res = tuple(next(res_iter) for _ in job.udfs)
            results = job.window.complete_job(run_record.run_id, job, window_res, damage=damage)
            all_results.extend(results)

        self.notify_new_results(*all_results)

    def notify_new_results(self, *results: ResultRow):
        if not results:
            return
        for window in self._windows.values():
            window.on_results_registered(*results)

    def notify_deleted_results(self, *results: ResultRow):
        if not results:
            return
        for window in self._windows.values():
            window.on_results_deleted(*results)
