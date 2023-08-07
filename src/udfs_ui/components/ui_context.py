from __future__ import annotations
import os
import asyncio
import uuid
import datetime
from humanize import naturalsize, precisedelta
import numpy as np
import panel as pn
from typing import Callable, TYPE_CHECKING, Type

from .progress import PanelProgressReporter
from .base import UIWindow, UIType, UIState, UDFWindowJob, JobResults
from .lifecycles import (
    UILifecycle,
    OfflineLifecycle,
    LiveLifecycle,
    ReplayLifecycle,
    ContinuousLifecycle,
)
from .resources import LiveResources, OfflineResources
from .tools import ROIWindow
from .results import ResultsManager, ResultRow
from .terminal_logger import UILog
from ..utils.notebook_tools import get_ipyw_reload_button

if TYPE_CHECKING:
    from libertem_live.detectors.base.acquisition import AcquisitionProtocol
    from libertem.udf.base import UDFResults
    from libertem_live.api import LiveContext
    from libertem.api import DataSet, Context
    from .base import RunFromT


class UITools:
    def __init__(self):
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

        self.roi_toggle_txt = pn.widgets.StaticText(
            value='<b>Global ROI:</b>',
            align='center',
            width=80,
            margin=(5, 2, 5, 5)
        )
        self.roi_toggle_btn = pn.widgets.Switch(
            width=35,
            align='center',
            margin=(5, 5, 8, 2)
        )

        window_keys = [
            *tuple(UIWindow.get_implementations(UIType.ANALYSIS).keys()),
        ]
        self.add_window_btn = pn.widgets.MenuButton(
            name='Add window',
            button_type='primary',
            items=window_keys,
            stylesheets=[
                """.bk-menu {
  position: unset;
}"""],
            **common_params,
        )

        common_params['min_width'] = 175
        self.replay_select = pn.widgets.Select(
            name='Dataset',
            options=[],
            **common_params,
        )

        self.pbar = pn.widgets.Tqdm(
            width=650,
            align=('start', 'center'),
            margin=(5, 5, 0, 5),
        )
        self.pbar.progress.margin = (0, 10, 0, 10)
        self.pbar.progress.height = 10

    def set_subtitle(self, subtitle: str):
        self.title.object = f'## UDFs UI - {subtitle}'


class UIContext:
    def __init__(self, save_root: os.PathLike | None = '.'):
        self._save_root = save_root
        self._windows: dict[str, UIWindow] = {}
        self._state: UIState = None
        self._resources: OfflineResources | LiveResources = None
        self._run_lock = asyncio.Lock()
        self._continue_running = False
        self._continuous_acquire = False
        # Create helper classes
        self._tools = UITools()
        self._p_reporter = PanelProgressReporter(self._tools.pbar)
        self._results_manager = ResultsManager(save_root)
        self._results_manager.add_watcher(self)
        self._logger = UILog()
        # Create layout
        self._button_row = pn.Row(margin=(0, 0))
        self._add_window_row = pn.Row(margin=(0, 0))
        self._windows_area = pn.Column(
            margin=(0, 0),
        )
        self._layout = pn.Column(
            self._button_row,
            self._tools.pbar,
            self._logger.as_collapsible(),
            self._windows_area,
            pn.layout.Divider(),
            self._add_window_row,
            min_width=700,
        )
        self._roi_window: ROIWindow | None = None

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
        offline_ctx: Context | None = None,
        aq_plan: AcquisitionProtocol | None = None,
    ):
        if not hasattr(live_ctx, 'make_acquisition'):
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
        ctx: Context,
        ds: DataSet,
    ):
        import libertem.api as lt  # noqa
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

    def layout(self, with_reload: bool = False):
        if with_reload:
            get_ipyw_reload_button()
        if self._state is None:
            raise ValueError('Must initialize UI before displaying')
        # The potential to redraw the same layout by re-executing the cell
        # can cause clashes with Bokeh models. This method should re-create
        # the whole UI rather than just returning the existing layout
        return self._layout

    @property
    def results_manager(self):
        return self._results_manager

    def _find_window_implem(self, name):
        # This implies all implementations have unique names
        # probably a good idea to enforce this during registration
        all_implems = {
            **UIWindow.get_implementations(UIType.ANALYSIS),
            **UIWindow.get_implementations(UIType.TOOL),
        }
        return all_implems[name]

    def add(
        self,
        window_cls: Type[UIWindow] | str,
        insert_at: int | None = None,
    ) -> UIContext:
        # Add a window and return self to allow method chaining
        # Internal methods use _add to get the created UIWindow
        self._add(window_cls, insert_at=insert_at)
        return self

    def _add(
        self,
        window_cls: Type[UIWindow] | str,
        insert_at: int | None = None,
    ) -> UIWindow:
        if isinstance(window_cls, str):
            window_cls = self._find_window_implem(window_cls)
        window_id = str(uuid.uuid4())[:6]
        window: UIWindow = window_cls(self, window_id)
        self._windows[window_id] = window
        if window.ident != window_id:
            raise RuntimeError('Mismatching window IDs')
        if insert_at is not None:
            self._windows_area.insert(insert_at, window.layout())
        else:
            self._windows_area.append(window.layout())
        self.logger.info(f'Added window {window.title_md} - {window.ident}')
        window.initialize(
            self._resources.get_ds_for_init(self._state, self.current_ds_ident)
        )
        return window

    def _add_roi_window(self, e):
        if e.new:
            if self._roi_window is not None:
                return
            self._roi_window = self._add(ROIWindow, insert_at=0)
        else:
            if self._roi_window is None:
                return
            self._remove(self._roi_window)
            self._roi_window = None

    def _remove(self, window: UIWindow):
        index = tuple(i for i, _lo
                      in enumerate(self._windows_area)
                      if hasattr(_lo, 'ident') and _lo.ident == window.ident)
        for i in reversed(index):
            self._windows_area.pop(i)
        self._windows.pop(window.ident, None)
        self.logger.info(f'Removed window {window.title_md} - {window.ident}')

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

        self._tools.add_window_btn.on_click(self._add_handler)
        self._tools.run_btn.on_click(self._run_handler)
        self._tools.stop_btn.on_click(self._stop_handler)
        self._tools.roi_toggle_btn.param.watch(self._add_roi_window, 'value')
        if self._state in (UIState.LIVE, UIState.REPLAY):
            self._tools.mode_btn.on_click(self._mode_handler)
            self._tools.continuous_btn.on_click(self._continuous_handler)

    async def _run_handler(
        self,
        *e,
        run_continuous: bool = False,
        run_from: list[RunFromT] | None = None,
    ):
        # This is problematic if we pick a frame
        # while running a long analysis, cursor will
        # be out of sync
        if self._run_lock.locked():
            self.logger.warning('Run command dropped, lock is already held')
            return
        self._continue_running = True
        async with self._run_lock:
            if run_continuous:
                await self.run_continuous(*e, run_from=run_from)
            elif self._state == UIState.OFFLINE:
                await self.run_offline(*e, run_from=run_from)
            elif self._state == UIState.LIVE:
                await self.run_live(*e, run_from=run_from)
            elif self._state == UIState.REPLAY:
                await self.run_replay(*e, run_from=run_from)

    async def _add_handler(self, e):
        mapper = UIWindow.get_all_implementations()
        selected = e.new
        window_cls = mapper.get(selected, None)
        if window_cls is None:
            self.logger.warning(f'Cannot find window to create {selected}')
            return
        try:
            window = self._add(
                window_cls
            )
        except Exception as err:
            msg = f'Error while adding {selected}'
            self._logger.log_from_exception(err, reraise=True, msg=msg)

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
        # Record mode should just be a checkbutton on the top row
        # Save files directly into the root directory of the UIContext
        # (could provide a method to change the directory)
        # Better still use ResultManager to handle dataset recordings!
        self._tools.replay_select.options = [*self.get_recordings_map().keys()]

    def _setup_offline(self):
        OfflineLifecycle(self).setup()

    def _controls(self):
        self._tools.title.object = f'## UDFs UI - {self._state.value}'
        button_row = [
            self._tools.title,
            self._tools.run_btn,
            self._tools.stop_btn,
            self._tools.roi_toggle_txt,
            self._tools.roi_toggle_btn,
        ]
        tool_row = [
            self._tools.add_window_btn,
        ]
        if self._state in (UIState.REPLAY, UIState.LIVE):
            button_row.insert(2, self._tools.continuous_btn)
            button_row.append(self._tools.mode_btn)
            button_row.append(self._tools.replay_select)
        return button_row, tool_row

    def get_roi(self, ds: DataSet) -> np.ndarray | None:
        # Get an ROI from an roi window if present and roi is set
        roi = None
        if self._roi_window is not None:
            roi = self._roi_window.get_roi(ds)
        return roi

    async def run_live(self, *e, run_from: list[RunFromT] | None = None):
        lifecycle = LiveLifecycle(self)
        live_ctx = self._resources.get_ctx(self._state)
        try:
            if (aq := self._resources.get_ds_for_run(self._state,
                                                     self.current_ds_ident)) is not None:
                await self._run(live_ctx, aq, lifecycle, run_from=run_from)
        finally:
            lifecycle.after()

    async def run_continuous(self, *e, run_from: list[RunFromT] | None = None):
        lifecycle = ContinuousLifecycle(self)
        live_ctx = self._resources.get_ctx(self._state)
        self._continuous_acquire = True
        try:
            while self._continuous_acquire and self._continue_running:
                if (aq := self._resources.get_ds_for_run(self._state,
                                                         self.current_ds_ident)) is not None:
                    await self._run(live_ctx, aq, lifecycle, run_from=run_from)
        finally:
            self._continuous_acquire = False
            lifecycle.after()

    async def run_replay(self, *e, run_from: list[RunFromT] | None = None):
        lifecycle = ReplayLifecycle(self)
        ctx = self._resources.get_ctx(self._state)
        ds = self._resources.get_ds_for_run(self._state, self.current_ds_ident)
        roi = self.get_roi(ds)

        try:
            await self._run(
                ctx,
                ds,
                lifecycle,
                roi=roi,
                run_from=run_from,
            )
        finally:
            lifecycle.after()

    async def run_offline(self, *e, run_from: list[RunFromT] | None = None):
        lifecycle = OfflineLifecycle(self)
        ctx = self._resources.get_ctx(self._state)
        ds = self._resources.get_ds_for_run(self._state, self.current_ds_ident)
        roi = self.get_roi(ds)

        try:
            await self._run(ctx, ds, lifecycle, roi=roi, run_from=run_from)
        finally:
            lifecycle.after()

    async def _run(
        self,
        ctx: Context | LiveContext,
        ds: DataSet | AcquisitionProtocol,
        ui_lifecycle: UILifecycle,
        roi: np.ndarray | None = None,
        run_from: list[RunFromT] | None = None,
    ):

        if ds is None:
            self.logger.error('Dataset is None')
            return

        if run_from is None:
            run_from = [w.get_job for w in self._windows.values()
                        if len(self._windows) <= 1 or (not w.self_run_only)]

        to_run: list[UDFWindowJob] = [job for job_getter in run_from
                                      if (job := job_getter(self._state, ds, roi))
                                      is not None]
        num_jobs = len(to_run)

        if num_jobs == 0:
            self.logger.info('No jobs to run')
            return
        elif num_jobs == 1:
            # Single job, gets ROI priority if provided
            window_roi = to_run[0].roi
            if window_roi is not None:
                if roi is not None:
                    self.logger.info('Global ROI being overwritten by window ROI')
                roi = window_roi
        else:
            dropped_windows = tuple(j.window.ident for j in to_run if j.roi is not None)
            to_run = [j for j in to_run if j.roi is None]
            if len(dropped_windows):
                self.logger.info('Found conflicting ROIs, skipping '
                                 f'the following windows: {dropped_windows} in run.')

        quiet_mode = all(j.quiet for j in to_run)
        if quiet_mode:
            ui_lifecycle.disable()
            progress = False
        else:
            ui_lifecycle.before()

            n_frames = ds.meta.shape.nav.size
            if roi is not None:
                n_frames = roi.sum()

            roi_message = f" with ROI of {n_frames} frames" if roi is not None else ", no ROI"
            self.logger.info(f'Start run, state {self._state.value.upper()} '
                             f'on {len(to_run)} jobs{roi_message}')
            # Special optimisation for progress bar when using single-frame ROI
            progress = False if (n_frames <= 1 and roi is not None) else self._p_reporter

        tstart = datetime.datetime.now()
        part_completed = 0
        try:
            async for udfs_res in ctx.run_udf_iter(
                dataset=ds,
                udf=tuple(udf for job in to_run for udf in job.udfs),
                plots=tuple(plot for job in to_run for plot in job.plots),
                progress=progress,
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
            msg = 'Error during run_udf'
            self._logger.log_from_exception(err, reraise=True, msg=msg)

        if self._continue_running and not quiet_mode:
            proc_time = datetime.datetime.now() - tstart
            try:
                data_rate = (
                    n_frames * ds.meta.shape.sig.size * np.dtype(ds.meta.raw_dtype).itemsize
                ) / proc_time.total_seconds()
            except (TypeError, ValueError, AttributeError):
                # Missing or wrong values on dataset implementation
                data_rate = float('nan')
            self.logger.info('End run, completed in '
                             f'{precisedelta(proc_time, minimum_unit="milliseconds")} '
                             f'({naturalsize(data_rate)}/s)')

        run_record = self.results_manager.new_run(
            has_roi=roi is not None,
            state=self._state.value,
            shape={
                'nav': tuple(ds.shape.nav),
                'sig': tuple(ds.shape.sig),
            }
        )
        if not quiet_mode:
            self.logger.info(f'Results saved with run_id: {run_record.run_id}')

        # Unpack results back to their window objects
        udfs_res: UDFResults
        damage = udfs_res.damage
        buffers = udfs_res.buffers
        res_iter = iter(buffers)
        all_results: list[ResultRow] = []
        for job in to_run:
            window_res = tuple(next(res_iter) for _ in job.udfs)
            job_results = JobResults(
                run_record.run_id,
                job,
                window_res,
                damage=damage,
            )
            if job.result_handler is not None:
                result_entries = job.result_handler(job, job_results)
                all_results.extend(result_entries)

        self.notify_new_results(*all_results)

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
