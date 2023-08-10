from __future__ import annotations
import uuid
import asyncio
from enum import Enum
from typing import TYPE_CHECKING, TypeVar, NamedTuple, Any, Callable

import panel as pn

from .result_tracker import ImageResultTracker

if TYPE_CHECKING:
    import numpy as np
    from libertem.api import DataSet
    from libertem_live.detectors.base.acquisition import AcquisitionProtocol
    from libertem.udf.base import UDF, BufferWrapper, UDFResultDict
    from libertem.viz.base import Live2DPlot
    from .ui_context import UIContext
    from .results import ResultRow
    from .result_tracker import ResultTracker
    from .live_plot import AperturePlot

    TWindow = TypeVar("TWindow", bound="UIWindow")

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


class UIState(Enum):
    OFFLINE = 'Offline'
    LIVE = 'Live'
    REPLAY = 'Replay'


class UIType(Enum):
    ANALYSIS = 'Analysis'
    TOOL = 'Tool'
    RESERVED = 'Reserved'


class UIWindow:
    title_md = 'UI Window'
    inner_container_cls = pn.Row

    header_activate = True
    header_remove = True
    can_self_run = True
    self_run_only = False

    _registry = {t: {} for t in UIType}

    def __init_subclass__(
        cls,
        ui_type: UIType | None = None,
        is_abstract: bool = False,
        force: bool = False,
        **kwargs,
    ):
        super().__init_subclass__(**kwargs)
        if ui_type is not None and not is_abstract:
            if ui_type not in cls._registry.keys():
                cls._registry[ui_type] = {}
            if not force and cls.name in cls._registry[ui_type]:
                raise TypeError(f'Cannot register a second UI with name {cls.name}, '
                                'use force=True in class definition to over-write '
                                f'or use another name. <{cls}>')
            cls._registry[ui_type][cls.name] = cls

    @classmethod
    def get_implementations(cls, ui_type: UIType) -> dict[str, UIWindow]:
        return cls._registry.get(ui_type, {})

    @classmethod
    def get_all_implementations(cls) -> dict[str, UIWindow]:
        implementations = {}
        for _implementations in cls._registry.values():
            implementations.update(_implementations)
        return implementations

    def __init__(self, ui_context: UIContext, ident: str | None = None):
        self._ui_context = ui_context
        if ident is None:
            ident = str(uuid.uuid4())[:5]
        self._ident = ident

        self._header_layout = self.build_header_layout()
        self._inner_layout = self.build_inner_container()
        self._layout = self.build_outer_container(
            self._header_layout,
            self._inner_layout,
        )
        self._trackers: dict[str, ResultTracker] = {}

    @property
    def ident(self) -> str:
        return self._ident

    @property
    def results_manager(self):
        return self._ui_context.results_manager

    @property
    def trackers(self):
        return self._trackers

    @property
    def logger(self):
        return self._ui_context.logger

    def build_header_layout(self):
        self._title_text = pn.pane.Markdown(
            object=f'### {self.title_md}',
            align='center',
        )
        self._id_text = pn.widgets.StaticText(
            value=f'[<b>{self._ident}</b>]',
            align='center',
        )
        self._remove_btn = pn.widgets.Button(
            name='Remove',
            button_type='danger',
            width_policy='min',
            align='center',
            visible=self.header_remove,
        )

        def _remove_self(*e):
            # Need to check if a window being removed while
            # it is being run completes gracefully
            self._ui_context._remove(self)

        self._remove_btn.on_click(_remove_self)
        self._active_cbox = pn.widgets.Checkbox(
            name='Active',
            value=True,
            align='center',
            min_width=50,
            visible=self.header_activate,
        )
        self._run_btn = pn.widgets.Button(
            name='Run this',
            button_type='success',
            width_policy='min',
            align='center',
            min_width=75,
            visible=self.can_self_run,
        )
        self._run_btn.on_click(self.run_from_btn)

        self._collapse_button = pn.widgets.Button(
            name='▼',
            button_type='default',
            width=35,
            height=35,
            align='center',
            margin=(0, 0),
        )
        self._collapse_button.param.watch(self._collapse_cb, 'value')
        return pn.Row(
            self._collapse_button,
            self._title_text,
            self._id_text,
            self._active_cbox,
            self._remove_btn,
            self._run_btn,
        )

    def build_outer_container(self, *objs) -> pn.layout.ListPanel:
        lo = pn.Column(pn.layout.Divider(), *objs, width_policy='max')
        lo.ident = self._ident
        return lo

    def build_inner_container(self, *objs) -> pn.layout.ListPanel:
        return self.inner_container_cls(*objs)

    @property
    def inner_layout(self) -> pn.layout.ListPanel:
        return self._inner_layout

    def layout(self) -> pn.layout.ListPanel | None:
        return self._layout

    @property
    def hidden(self):
        if self._layout is None:
            # Something which doesn't exist is not hidden
            return False
        return (not self._layout.visible)

    def hide(self):
        if self._layout is not None:
            self._layout.visible = False

    def unhide(self):
        if self._layout is not None:
            self._layout.visible = True

    def _collapse_cb(self, e):
        if self._inner_layout.visible:
            self._inner_layout.visible = False
            self._collapse_button.name = '▶'
        else:
            self._inner_layout.visible = True
            self._collapse_button.name = '▼'

    def set_state(self, old_state: UIState, new_state: UIState):
        # Called on UI state transitions
        pass

    def initialize(self, dataset: DataSet) -> TWindow:
        # This method is problematic as we don't always have
        # the dataset in advance to initialize the plots,
        # nor does every conceivable window type need a dataset
        # to initialize itself
        raise NotImplementedError

    @property
    def is_active(self) -> bool:
        return self._active_cbox.value

    def set_active(self, val: bool):
        self._active_cbox.value = val

    async def run_this(self, *e, run_from: RunFromT | None = None):
        # The functionality here could allow running
        # windows independently and concurrently, but would
        # need to refactor progress bar + UI state synchronisation
        if run_from is None:
            run_from = self.get_job
        await self._ui_context._run_handler(*e, run_from=[run_from])

    async def run_from_btn(self, *e):
        # Subclass can override this method if it does not
        # want the run_btn to be disabled when pressed
        # (i.e. if job.quiet == True)
        self._run_btn.disabled = True
        try:
            await self.run_this(*e, run_from=self.get_job)
        finally:
            self._run_btn.disabled = False

    def run_this_bk(self, attr, old, new, run_from: RunFromT | None = None):
        # Run a job from a Bokeh-style callback, asynchronously
        # (attr, old, new) are required in the signature by Bokeh
        return self.run_async(run_from=run_from)

    def run_async(self, run_from: RunFromT | None = None):
        self._futures.append(
            asyncio.gather(self.run_this(run_from=run_from), return_exceptions=False)
        )
        self._futures[-1].add_done_callback(self._cleanup_future)

    def _cleanup_future(self, future: asyncio.Future):
        try:
            future.result()
        except Exception as e:
            self._ui_context._logger.log_from_exception(e, reraise=True)
        finally:
            try:
                self._futures.pop(self._futures.index(future))
            except ValueError:
                pass

    def get_job(
        self,
        state: UIState,
        dataset: DataSet | AcquisitionProtocol,
        roi: np.ndarray | None,
    ) -> UDFWindowJob | None:
        return None

    def complete_job(
        self,
        job: UDFWindowJob,
        results: JobResults,
    ) -> tuple[ResultRow, ...]:
        return tuple()

    def on_results_registered(
        self,
        *results: ResultRow,
    ):
        pass

    def on_results_deleted(
        self,
        *results: ResultRow,
    ):
        pass

    def link_image_plot(
        self,
        name: str,
        plot: AperturePlot,
        tags: tuple[str, ...],
        initialize: bool = True,
    ) -> ResultTracker:
        if name in self.trackers.keys():
            raise ValueError('Result trackers must have unique names')
        tracker = ImageResultTracker(
            self.results_manager,
            plot,
            tags,
        )
        if initialize:
            tracker.initialize()
        self._trackers[name] = tracker
        self._layout_trackers()
        return tracker

    def _layout_trackers(self):
        # Assumes trackers cannot be removed!!!
        n_trackers = len(self.trackers)
        if n_trackers == 0:
            return
        elif n_trackers == 1:
            divider = pn.pane.HTML(
                R"""<div></div>""",
                styles={
                    'border-left': '2px solid #757575',
                    'height': '35px',
                }
            )
            self._tracker_display_toggle = pn.widgets.Toggle(
                name='Display',
                value=False,
                button_type='default',
                align='center',
                margin=(5, 5, 5, 5),
            )
            self._header_layout.extend((
                divider,
                self._tracker_display_toggle,
            ))
            self._tracker_display_toggle.param.watch(self._toggle_tracker_visible, 'value')
            tracker_name = tuple(self.trackers.keys())[0]
            tracker = self.trackers[tracker_name]
            self._tracker_select = pn.widgets.RadioButtonGroup(
                options=[tracker_name],
                button_type='default',
                value=tracker_name,
                align='center',
                min_width=75,
            )
            self._tracker_layouts = pn.Row(
                self._tracker_select,
                margin=(0, 0),
                visible=self._tracker_display_toggle.value,
            )
            self._header_layout.append(self._tracker_layouts)
        elif n_trackers == 2:
            self._tracker_select.param.watch(self._toggle_trackers_selected, 'value')

        self._tracker_select.options = list(self.trackers.keys())
        for tracker in self.trackers.values():
            if tracker.layout is None:
                tracker.layout = pn.Row(
                    *tracker.components(),
                    align='center',
                )
                self._tracker_layouts.append(tracker.layout)

        if n_trackers > 1:
            self._toggle_trackers_selected()

    def _toggle_tracker_visible(self, e):
        visible = e.new
        self._tracker_layouts.visible = visible

    def _toggle_trackers_selected(self, *e):
        for tracker_name, tracker in self.trackers.items():
            is_selected = self._tracker_select.value == tracker_name
            if not is_selected:
                tracker.set_visible(is_selected)
        self.trackers[self._tracker_select.value].set_visible(True)


class UDFWindowJob(NamedTuple):
    window: UIWindow
    udfs: list[UDF]
    plots: list[Live2DPlot]
    result_handler: ResultHandlerT | None = None
    params: dict[str, Any] | None = None
    roi: np.ndarray | None = None
    quiet: bool = False


class JobResults(NamedTuple):
    run_id: str
    job: UDFWindowJob
    udf_results: tuple[UDFResultDict]
    damage: BufferWrapper | None = None
