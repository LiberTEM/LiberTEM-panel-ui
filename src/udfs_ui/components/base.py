from __future__ import annotations
import uuid
from enum import Enum
from typing import TYPE_CHECKING, TypeVar, NamedTuple, Any

from bidict import bidict
import panel as pn

from .live_plot import AperturePlot

if TYPE_CHECKING:
    import numpy as np
    import libertem.api as lt
    from libertem_live.detectors.base.acquisition import AcquisitionProtocol
    from libertem.udf.base import UDF, BufferWrapper, UDFResultDict
    from .ui_context import UIContext
    from .results import ResultRow


TWindow = TypeVar("TWindow", bound="UIWindow")


class UIState(Enum):
    OFFLINE = 'Offline'
    LIVE = 'Live'
    REPLAY = 'Replay'


class UIType(Enum):
    ANALYSIS = 'Analysis'
    TOOL = 'Tool'


class RemoveActivateHeaderMixin:
    """
    This is on a mixin as not all windows should
    be removable or be disable/activate-able
    """
    def build_header_layout(self):
        self._title_text = pn.pane.Markdown(
            object=f'### {self.title_md}',
            align='center',
        )
        self._id_text = pn.widgets.StaticText(
            value=f'(<b>{self._ident}</b>)',
            align='center',
        )
        self._remove_btn = pn.widgets.Button(
            name='Remove',
            button_type='danger',
            width_policy='min',
            align='center',
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
        )
        return pn.Row(
            self._title_text,
            self._id_text,
            self._active_cbox,
            self._remove_btn,
        )

    @property
    def is_active(self) -> bool:
        return self._active_cbox.value

    def set_active(self, val: bool):
        self._active_cbox.value = val


class UIWindow:
    title_md = 'UI Window'
    is_unique = False

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
            if not force and cls.name in cls._registry[ui_type]:
                raise TypeError(f'Cannot register a second UI with name {cls.name}, '
                                'use force=True in class definition to over-write '
                                f'or use another name. <{cls}>')
            cls._registry[ui_type][cls.name] = cls

    @classmethod
    def get_implementations(cls, ui_type: UIType) -> dict[str, UIWindow]:
        return cls._registry.get(ui_type, {})

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

    @property
    def ident(self) -> str:
        return self._ident

    @property
    def results_manager(self):
        return self._ui_context.results_manager

    @property
    def logger(self):
        return self._ui_context.logger

    def build_header_layout(self) -> pn.layout.ListPanel:
        # default implementation, to override in mixin
        return pn.Row(pn.pane.Markdown(object=f'### {self.title_md}'))

    def build_outer_container(self, *objs) -> pn.layout.ListPanel:
        lo = pn.Column(pn.layout.Divider(), *objs, width_policy='max')
        lo.ident = self._ident
        return lo

    def build_inner_container(self, *objs) -> pn.layout.ListPanel:
        return pn.Row(*objs)

    @property
    def inner_layout(self) -> pn.layout.ListPanel:
        return self._inner_layout

    def layout(self) -> pn.layout.ListPanel:
        return self._layout

    def set_state(self, old_state: UIState, new_state: UIState):
        # Called on UI state transitions
        pass

    def initialize(self, dataset: lt.DataSet) -> TWindow:
        # This method is problematic as we don't always have
        # the dataset in advance to initialize the plots,
        # nor does every conceivable window type need a dataset
        # to initialize itself
        raise NotImplementedError

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


class ActivateableUIWindow(RemoveActivateHeaderMixin, UIWindow):
    ...


class RunnableUIWindow(ActivateableUIWindow):
    can_self_run = True

    def build_header_layout(self) -> pn.layout.ListPanel:
        lo = super().build_header_layout()
        if not self.can_self_run:
            return lo

        self._run_btn = pn.widgets.Button(
            name='Run this',
            button_type='success',
            width_policy='min',
            align='center',
            min_width=75,
        )
        self._run_btn.on_click(self.run_this)
        lo.append(self._run_btn)
        return lo

    async def run_this(self, *e):
        # The functionality here could allow running
        # windows independently and concurrently, but would
        # need to refactor progress bar + UI state synchronisation
        self._run_btn.disabled = True
        try:
            await self._ui_context._run_handler(*e, windows=[self])
        finally:
            self._run_btn.disabled = False

    def get_job(
        self,
        state: UIState,
        dataset: lt.DataSet | AcquisitionProtocol,
        roi: np.ndarray | None,
    ) -> UDFWindowJob | None:
        raise NotImplementedError

    def complete_job(
        self,
        run_id: str,
        job: UDFWindowJob,
        results: tuple[UDFResultDict],
        damage: BufferWrapper | None = None
    ) -> tuple[ResultRow, ...]:
        return tuple()


class UDFWindowJob(NamedTuple):
    window: RunnableUIWindow
    udfs: list[UDF]
    plots: list[AperturePlot]
    params: dict[str, Any] | None = None

    # Could consider adding the result-handing callback
    # to the job object to allow this to be customised


class ImageResultTracker:
    def __init__(
        self,
        ui_window: UIWindow,
        plot: AperturePlot,
        tags: tuple[str, ...],
        select_text: str,
        plot_initialized: bool | str = False,
        auto_refresh: bool = True,
    ):
        self.window = ui_window
        self.plot = plot
        self.tags = tags
        # Records the fact we initialised from a zeros-array
        self._plot_displayed = plot_initialized

        divider = pn.pane.HTML(
            R"""<div></div>""",
            styles={
                'border-left': '2px solid #757575',
                'height': '35px',
            }
        )
        select_text = pn.widgets.StaticText(
            value=select_text,
            align='center',
            margin=(5, 5, 5, 5),
        )
        self._select_options: bidict[str, ResultRow] = bidict({})
        self.select_box = pn.widgets.Select(
            options=list(self._select_options.keys()),
            width=200,
            max_width=300,
            width_policy='min',
            align='center',
        )
        self.load_btn = pn.widgets.Button(
            name='Load image',
            align='center',
            button_type='primary',
        )
        self.load_btn.on_click(self.load_image)
        self.refresh_cbox = pn.widgets.Checkbox(
            name='Auto-refresh',
            value=auto_refresh,
            align='center',
        )

        ui_window._header_layout.extend((
            divider,
            select_text,
            self.select_box,
            self.load_btn,
            self.refresh_cbox,
        ))

    def _select_result_name(self, result: ResultRow):
        return (f'[{result.run_id}]: {result.result_id}, '
                f'{self.window.results_manager.get_window(result.window_id).window_name}')

    def on_results_registered(
        self,
        *results: ResultRow,
    ):
        new_results = tuple(
            self.window.results_manager.yield_with_tag(*self.tags, from_rows=results)
        )
        if not new_results:
            return
        self._select_options.update({
            self._select_result_name(r): r
            for r in new_results
        })
        self.select_box.options = list(reversed(sorted(
            self._select_options.keys(),
            key=lambda x: self._select_options[x].run_id
        )))
        if not self._plot_displayed:
            # Initialize plot from first nav-tagged result
            self.select_box.value = self.select_box.options[0]
            self.load_image()
            return
        if self.refresh_cbox.value:
            current_display_id = self._plot_displayed
            current = self.window.results_manager.get_result_row(current_display_id)
            if current is None:
                # Result must have been deleted
                return
            possible_results = tuple(
                r for r in new_results
                if (r.window_id == current.window_id) and (r.name == current.name)
            )
            if not possible_results:
                return
            # Take the first, result names are supposed to be unique!
            self.select_box.value = self._select_options.inverse[possible_results[0]]
            self.load_image()

    def on_results_deleted(
        self,
        *results: ResultRow,
    ):
        for r in results:
            _ = self._select_options.inverse.pop(r, None)
        self.select_box.options = list(self._select_options.keys())

    def load_image(self, *e):
        selected: str = self.select_box.value
        if not selected:
            return
        result_row = self._select_options.get(selected, None)
        if result_row is None:
            return
        rc = self.window.results_manager.get_result_container(result_row.result_id)
        if rc is None:
            return
        self.plot.im.update(rc.data)
        self.plot.fig.title.text = f'{self.plot.title} - {result_row.result_id}'
        self._plot_displayed = result_row.result_id
        pn.io.notebook.push_notebook(self.plot.pane)
