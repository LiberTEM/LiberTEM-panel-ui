from __future__ import annotations
import uuid
import asyncio
from types import SimpleNamespace
from strenum import StrEnum

from typing import TYPE_CHECKING, TypeVar, NamedTuple, Any, TypedDict
from typing_extensions import Self

import panel as pn
from bokeh.models.widgets import Div

from ..results.tracker import ImageResultTracker
from ..utils.panel_components import button_divider, get_spinner

if TYPE_CHECKING:
    import numpy as np
    from libertem.api import DataSet, Context
    from libertem_live.detectors.base.acquisition import AcquisitionProtocol
    from libertem.udf.base import UDF
    from libertem.viz.base import Live2DPlot
    from ..base import UIContextBase, RunFromT, ResultHandlerT, UIState, JobResults
    from .standalone import StandaloneContext
    from ..results.results_manager import ResultRow
    from ..results.tracker import ResultTracker
    from ..live_plot import AperturePlot

    T = TypeVar('T', bound='WindowProperties')
    W = TypeVar('W', bound='UIWindow')


class WindowType(StrEnum):
    STANDALONE = 'Standalone'
    RESERVED = 'Reserved'


class WindowProperties(NamedTuple):
    name: str
    title_md: str

    insert_at: int | None = None
    init_collapsed: bool = False
    header_indicate: bool = True
    header_activate: bool = True
    header_remove: bool = True
    header_run: bool = True
    header_stop: bool = False
    header_divider: bool = True
    header_collapse: bool = True
    self_run_only: bool = False

    def with_other(self: T, **kwargs: dict[str, str | bool]) -> T:
        return type(self)(
            **{
                **self._asdict(),
                **kwargs,
            }
        )


# Exists to help type-hint property overrides
# Ideally would use a NamedTuple but don't want
# to have the default values when overriding
class WindowPropertiesTDict(TypedDict):
    name: str
    title_md: str

    insert_at: int | None
    init_collapsed: bool
    header_indicate: bool
    header_activate: bool
    header_remove: bool
    header_run: bool
    header_stop: bool
    header_divider: bool
    header_collapse: bool
    self_run_only: bool


class UIWindow:
    SPINNER_SIZE = 30
    _registry: dict[WindowType, dict[str, type[UIWindow]]] = {t: {} for t in WindowType}

    def __init_subclass__(
        cls,
        ui_type: WindowType | None = None,
        force: bool = False,
        **kwargs,
    ):
        super().__init_subclass__(**kwargs)
        if ui_type is not None:
            if ui_type not in cls._registry.keys():
                cls._registry[ui_type] = {}
            all_names = tuple(n for implems in cls._registry.values() for n in implems.keys())
            props = cls.default_properties()
            if props.name in all_names and (not force):
                raise TypeError(f'Cannot register a second UI with name {props.name}, '
                                'use force=True in class definition to over-write, '
                                f'or use another name. {cls}')
            elif force:
                _ = tuple(implems.pop(props.name, None) for implems in cls._registry.values())
            cls._registry[ui_type][props.name] = cls

    @classmethod
    def get_implementations(cls, ui_type: WindowType) -> dict[str, type[UIWindow]]:
        return cls._registry.get(ui_type, {})

    @classmethod
    def get_all_implementations(cls) -> dict[str, type[UIWindow]]:
        implementations = {}
        for _implementations in cls._registry.values():
            implementations.update(_implementations)
        return implementations

    @staticmethod
    def default_properties() -> WindowProperties:
        raise NotImplementedError()

    def __init__(
        self,
        ui_context: UIContextBase,
        ident: str | None = None,
        prop_overrides: WindowPropertiesTDict | None = None,
        window_data: Any | None = None,
    ):
        self._ui_context = ui_context
        if ident is None:
            ident = str(uuid.uuid4())[:5]
        self._ident = ident
        prop_overrides = {} if prop_overrides is None else prop_overrides
        self._window_properties = self.default_properties().with_other(**prop_overrides)
        self._window_data = window_data
        self.validate_data()
        self._trackers: dict[str, ResultTracker] = {}
        # stores references to in-flight Futures launched
        # using asyncio, to avoid tasks being garbage collected
        self._futures: set[asyncio.Future] = set()

        self._header_ns = SimpleNamespace()
        self._header_layout = self.build_header_layout()
        self._inner_layout = self.build_inner_container()
        self._layout = self.build_outer_container(
            self._header_layout,
            self._inner_layout,
        )

    @classmethod
    def using(cls: type[W], ctx: Context, dataset: DataSet, **init_kwargs) -> W:
        from .standalone import StandaloneContext
        ui_context = StandaloneContext(ctx, dataset)
        return cls._using(ui_context, **init_kwargs)

    @classmethod
    def _using(cls: type[W], ui_context: StandaloneContext, **init_kwargs) -> W:
        default_properties = cls.default_properties()
        window = cls(
            ui_context,
            prop_overrides={
                'header_activate': False,
                'header_remove': False,
                'header_collapse': False,
                'header_divider': False,
                'header_stop': default_properties.header_run,
                'init_collapsed': False,
            }
        )
        window.initialize(ui_context._resources.init_with(), **init_kwargs)
        ui_context._register_window(window)
        if window.properties.header_run:
            ui_context._add_progress_bar(window)
            ui_context._setup_run_button(window._header_ns._run_btn)
        return window

    @classmethod
    def linked_to(cls: type[W], other: UIWindow) -> W:
        ui_context = other._ui_context
        return cls._using(ui_context)

    def validate_data(self):
        pass

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

    @property
    def properties(self) -> WindowProperties:
        return self._window_properties

    @property
    def submit_to(self):
        return self._ui_context._run_job

    def remove_self(self, *e):
        self._ui_context._remove_window(self)

    def build_header_layout(self):
        lo = pn.Row()

        if self.properties.header_collapse:
            self._header_ns._collapse_button = pn.widgets.Button(
                name='▼',
                button_type='light',
                width=35,
                height=35,
                align='center',
                margin=(3, 3),
            )
            self._header_ns._collapse_button.param.watch(self._collapse_cb, 'value')
            lo.append(self._header_ns._collapse_button)

        if self.properties.header_indicate:
            self._header_ns._indicator = Div(
                text=get_spinner(False, self.SPINNER_SIZE),
                width=self.SPINNER_SIZE,
                height=self.SPINNER_SIZE,
                tags=['lt_indicator'],
            )
            lo.append(self._header_ns._indicator)

        self._header_ns._title_text = pn.pane.Markdown(
            object=f'### {self.properties.title_md}',
            align='center',
            margin=(3, 3),
        )
        self._header_ns._id_text = pn.widgets.StaticText(
            value=f'[<b>{self._ident}</b>]',
            align='center',
            margin=(3, 3),
        )

        lo.extend((
            self._header_ns._title_text,
            self._header_ns._id_text,
        ))

        if self.properties.header_activate:
            self._header_ns._active_cbox = pn.widgets.Checkbox(
                name='Active',
                value=True,
                align='center',
                min_width=50,
            )
            lo.append(self._header_ns._active_cbox)

        if self.properties.header_remove:
            self._header_ns._remove_btn = pn.widgets.Button(
                name='Remove',
                button_type='danger',
                width_policy='min',
                align='center',
                tags=['lt_disable_on_run'],
            )

            self._header_ns._remove_btn.on_click(self.remove_self)
            lo.append(self._header_ns._remove_btn)

        if self.properties.header_run:
            self._header_ns._run_btn = pn.widgets.Button(
                name='Run this',
                button_type='success',
                width_policy='min',
                align='center',
                min_width=75,
                tags=['lt_disable_on_run'],
            )
            self._header_ns._run_btn.on_click(self.run_from_btn)
            lo.append(self._header_ns._run_btn)

        if self.properties.header_stop:
            self._header_ns._stop_btn = pn.widgets.Button(
                name='Stop',
                button_type='danger',
                width_policy='min',
                align='center',
                min_width=75,
                disabled=True,
                tags=['lt_enable_on_run'],
            )
            lo.append(self._header_ns._stop_btn)

        return lo

    def build_outer_container(self, *objs) -> pn.layout.ListPanel:
        lo = pn.Column(width_policy='max')
        if self.properties.header_divider:
            lo.append(pn.layout.Divider())
        lo.extend(objs)
        lo.ident = self._ident
        return lo

    @staticmethod
    def inner_container_cls():
        return pn.Row

    def build_inner_container(self, *objs) -> pn.layout.ListPanel:
        return self.inner_container_cls()(*objs)

    @property
    def inner_layout(self) -> pn.layout.ListPanel:
        return self._inner_layout

    def layout(self) -> pn.layout.ListPanel | None:
        return self._layout

    def _collapse_cb(self, e):
        try:
            self._header_ns._collapse_button
        except AttributeError:
            raise RuntimeError('Cannot collapse window without header_collapse option.')
        if self._inner_layout.visible:
            self._inner_layout.visible = False
            self._header_ns._collapse_button.name = '▶'
        else:
            self._inner_layout.visible = True
            self._header_ns._collapse_button.name = '▼'

    def initialize(self, dataset: DataSet) -> Self:
        # This method is problematic as we don't always have
        # the dataset in advance to initialize the plots,
        # nor does every conceivable window type need a dataset
        # to initialize itself
        raise NotImplementedError

    @property
    def is_active(self) -> bool:
        try:
            return self._header_ns._active_cbox.value
        except AttributeError:
            return True

    def set_active(self, val: bool):
        try:
            self._header_ns._active_cbox.value = val
        except AttributeError:
            raise AttributeError('Cannot set_active on window without active/disable option.')

    def should_stop(self) -> bool:
        return self.properties.header_stop and self._header_ns._stop_btn.clicks > 0

    async def run_this(self, run_from: RunFromT | None = None):
        if run_from is None:
            run_from = self.get_job
        await self.submit_to(run_from=[run_from])

    async def run_from_btn(self, e):
        await self.run_this(run_from=self.get_job)

    def run_this_bk(self, attr, old, new, run_from: RunFromT | None = None):
        # Run a job from a Bokeh-style callback, asynchronously
        # (attr, old, new) are required in the signature by Bokeh
        return self.run_async(run_from=run_from)

    def run_async(self, run_from: RunFromT | None = None):
        future = asyncio.gather(self.run_this(run_from=run_from), return_exceptions=False)
        future.add_done_callback(self._cleanup_future)
        self._futures.add(future)

    def _cleanup_future(self, future: asyncio.Future):
        try:
            future.result()
        except Exception as e:
            self.logger.log_from_exception(e, reraise=True)
        finally:
            self._futures.discard(future)

    def _run_buttons(self) -> tuple[pn.widgets.Button, ...]:
        # StandaloneContext uses this to set disabled = True
        # when the job is not quiet to trigger the JS callbacks
        if self.properties.header_run:
            return (self._header_ns._run_btn,)
        return ()

    def _stop_buttons(self) -> tuple[pn.widgets.Button, ...]:
        # StandaloneContext uses this to listen stop iteration
        if self.properties.header_stop:
            return (self._header_ns._stop_btn,)
        return ()

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
            # This is a workaround for using a JS color mapper
            # If we are initializing before the image is in the browser
            # must ensure the color limits are correct after the call
            # to img.update() inside initialize(). Should automate this.
            plot.im.color.push_clims()
        self._trackers[name] = tracker
        self._layout_trackers()
        return tracker

    def _layout_trackers(self):
        # Assumes trackers cannot be removed!!!
        n_trackers = len(self.trackers)
        if n_trackers == 0:
            return
        elif n_trackers == 1:
            self._header_ns._tracker_display_toggle = pn.widgets.Toggle(
                name='Display',
                value=False,
                button_type='default',
                align='center',
                margin=(5, 5, 5, 5),
            )
            self._header_layout.extend((
                button_divider(),
                self._header_ns._tracker_display_toggle,
            ))
            self._header_ns._tracker_display_toggle.param.watch(
                self._toggle_tracker_visible, 'value'
            )
            tracker_name = tuple(self.trackers.keys())[0]
            tracker = self.trackers[tracker_name]
            self._header_ns._tracker_select = pn.widgets.RadioButtonGroup(
                options=[tracker_name],
                button_type='default',
                value=tracker_name,
                align='center',
                min_width=75,
            )
            self._tracker_layouts = pn.Row(
                self._header_ns._tracker_select,
                margin=(0, 0),
                visible=self._header_ns._tracker_display_toggle.value,
            )
            self._header_layout.append(self._tracker_layouts)
        elif n_trackers == 2:
            self._header_ns._tracker_select.param.watch(
                self._toggle_trackers_selected, 'value'
            )

        self._header_ns._tracker_select.options = list(self.trackers.keys())
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
            is_selected = self._header_ns._tracker_select.value == tracker_name
            if not is_selected:
                tracker.set_visible(is_selected)
        self.trackers[self._header_ns._tracker_select.value].set_visible(True)


class UDFWindowJob(NamedTuple):
    window: UIWindow
    udfs: list[UDF]
    plots: list[Live2DPlot]
    result_handler: ResultHandlerT | None = None
    params: dict[str, Any] | None = None
    roi: np.ndarray | None = None
    quiet: bool = False
