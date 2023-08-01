from __future__ import annotations
from typing import NamedTuple, Callable, TYPE_CHECKING

from .base import UIState

if TYPE_CHECKING:
    from libertem.api import DataSet, Context
    import pathlib
    from libertem_live.api import LiveContext
    from libertem_live.detectors.base.acquisition import AcquisitionProtocol


class OfflineResources(NamedTuple):
    ctx: Context
    ds: DataSet

    def get_ds_for_init(self, state: UIState, ident: str | None) -> DataSet:
        return self.get_ds_for_run(state, ident)

    def get_ds_for_run(self, state: UIState, ident: str | None) -> DataSet:
        return self.ds

    def get_ctx(self, state: UIState) -> Context:
        return self.ctx


class LiveResources(NamedTuple):
    live_ctx: LiveContext
    get_aq: Callable[[LiveContext], AcquisitionProtocol | None]
    offline_ctx: Context | None = None
    recordings: dict[str, pathlib.Path] = {}
    aq_plan: AcquisitionProtocol | None = None

    def get_ds_for_init(
        self,
        state: UIState,
        ident: str | None
    ) -> DataSet | AcquisitionProtocol | None:
        if state == UIState.LIVE:
            if self.aq_plan is not None:
                return self.aq_plan
            return self.get_aq(self.live_ctx)
        elif state == UIState.REPLAY:
            return self.get_ds_for_run(state, ident)

    def get_ds_for_run(
        self,
        state: UIState,
        ident: str | None
    ) -> DataSet | AcquisitionProtocol | None:
        if state == UIState.LIVE:
            return self.get_aq(self.live_ctx)
        elif state == UIState.REPLAY:
            if (path := self.recordings.get(ident, None)) is not None:
                return self.replay_context.load('npy', path)
            return None

    def get_ctx(self, state: UIState) -> Context:
        if state == UIState.LIVE:
            return self.live_ctx
        elif state == UIState.REPLAY:
            return self.replay_context

    @property
    def replay_context(self):
        if self.offline_ctx is None:
            return self.live_ctx
        return self.offline_ctx
