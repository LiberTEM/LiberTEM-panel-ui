from __future__ import annotations
from typing import NamedTuple, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from libertem.api import DataSet, Context
    from libertem_live.api import LiveContext
    from libertem_live.detectors.base.acquisition import AcquisitionProtocol


class OfflineResources(NamedTuple):
    ctx: Context
    dataset: DataSet

    def init_with(self) -> DataSet:
        return self.dataset


class LiveResources(NamedTuple):
    live_ctx: LiveContext
    # This could be replaced with a true AcquisitionPlan object
    # which contains the intended shapes / dtype
    aq_plan: AcquisitionProtocol | DataSet
    get_aq: Callable[[LiveContext], AcquisitionProtocol | None]
    offline_ctx: Context | None = None

    def init_with(self) -> AcquisitionProtocol | DataSet:
        return self.aq_plan

    @property
    def replay_context(self):
        if self.offline_ctx is None:
            return self.live_ctx
        return self.offline_ctx
