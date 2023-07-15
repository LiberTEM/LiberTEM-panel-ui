from __future__ import annotations
import uuid
from enum import Enum
from typing import TYPE_CHECKING, TypeVar, NamedTuple, Any

import panel as pn

from .live_plot import AperturePlot

if TYPE_CHECKING:
    import numpy as np
    import libertem.api as lt
    from libertem_live.detectors.base.acquisition import AcquisitionProtocol
    from libertem.udf.base import UDF, BufferWrapper, UDFResultDict
    from .ui_context import UIContext


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
        self._title_text = pn.pane.Markdown(object=f'### {self.title_md}')
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

    def get_run(
        self,
        state: UIState,
        dataset: lt.DataSet | AcquisitionProtocol,
        roi: np.ndarray | None,
    ) -> UDFWindowJob | None:
        raise NotImplementedError

    def set_results(
        self,
        run_id: str,
        job: UDFWindowJob,
        results: tuple[UDFResultDict],
        damage: BufferWrapper | None = None
    ):
        pass


class UDFWindowJob(NamedTuple):
    window: RunnableUIWindow
    udfs: list[UDF]
    plots: list[AperturePlot]
    params: dict[str, Any] | None = None

    # Could consider adding the result-handing callback
    # to the job object to allow this to be customised
