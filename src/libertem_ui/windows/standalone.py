from __future__ import annotations
from weakref import WeakValueDictionary
from typing import TYPE_CHECKING

import panel as pn

from ..results.results_manager import ResultsManager
from ..utils.logging import logger
from .base import UIWindow
from ..base import UIContextBase, UIState
from ..resources import OfflineResources
from ..utils.progress import PanelProgressReporter
from ..utils.panel_components import get_spinner

if TYPE_CHECKING:
    from .base import UDFWindowJob
    from ..base import RunFromT, WindowIdent
    from libertem.api import DataSet, Context


class StandaloneContext(UIContextBase):
    def __init__(self, ctx: Context, dataset: DataSet):
        self._results_manager = ResultsManager()
        self._results_manager.add_watcher(self)
        self._resources = OfflineResources(ctx, dataset)
        self._resources: OfflineResources
        # As the standalone context does not control any
        # windows which are connected to it only keep a weakref to them
        self._windows: WeakValueDictionary[WindowIdent, UIWindow] = WeakValueDictionary()
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

    @staticmethod
    def _setup_run_button(*elements: pn.reactive.Reactive):
        code = '''
const is_disabled = cb_obj.disabled

// searching through *all* models is really a hack...
for (let model of this.document._all_models.values()){
    if (model.tags.includes("lt_disable_on_run")) {
        model.disabled = is_disabled
    } else if (model.tags.includes("lt_enable_on_run")) {
        model.disabled = !is_disabled
    } else if (model.tags.includes("lt_indicator")){
        if (is_disabled){
            model.text = spin_text
        } else {
            model.text = static_text
        }
    }
}'''
        for el in elements:
            el.jscallback(
                args={
                    'spin_text': get_spinner(True, UIWindow.SPINNER_SIZE),
                    'static_text': get_spinner(False, UIWindow.SPINNER_SIZE),
                },
                disabled=code,
            )

    @property
    def logger(self):
        return logger

    @property
    def results_manager(self):
        return self._results_manager

    async def _run_job(self, run_from: list[RunFromT]):
        ds = self._resources.dataset
        to_run: list[UDFWindowJob] = [
            job for job_getter in run_from
            if (job := job_getter(UIState.OFFLINE, ds, None))
            is not None
        ]
        if len(to_run) == 0:
            return
        elif len(to_run) > 1:
            # Mainly due to ROI negotiation...
            raise NotImplementedError
        else:
            job = to_run[0]
        quiet = job.quiet
        roi = job.roi
        progress = False if quiet else self._progress.get(job.window.ident, False)
        ctx = self._resources.ctx
        run_buttons = job.window._run_buttons()
        stop_buttons = job.window._stop_buttons()
        inital_stop_count = tuple(b.clicks for b in stop_buttons)
        try:
            if not job.quiet:
                _ = tuple(b.param.update(disabled=True) for b in run_buttons)
            async for udfs_res in ctx.run_udf_iter(
                dataset=ds,
                udf=tuple(udf for job in to_run for udf in job.udfs),
                plots=tuple(plot for job in to_run for plot in job.plots),
                progress=progress,
                sync=False,
                roi=roi,
            ):
                if any(b.clicks > i for b, i in zip(stop_buttons, inital_stop_count)):
                    self.logger.info('Early stop from button')
                    break
        except Exception as err:
            msg = 'Error during run_udf'
            self.logger.log_from_exception(err, reraise=True, msg=msg)
        finally:
            if not job.quiet:
                _ = tuple(b.param.update(disabled=False) for b in run_buttons)

        run_record = self.results_manager.new_run(
            shape={
                'nav': tuple(ds.shape.nav),
                'sig': tuple(ds.shape.sig),
            },
            has_roi=roi is not None,
            state=UIState.OFFLINE,
        )

        all_results = self.unpack_results(udfs_res, to_run, run_record)
        self.notify_new_results(*all_results)
