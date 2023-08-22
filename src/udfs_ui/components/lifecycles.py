from __future__ import annotations
from typing import TYPE_CHECKING

from .base import UIState

if TYPE_CHECKING:
    from .ui_context import UIContext


class UILifecycle:
    def __init__(self, ui_context: UIContext):
        self.ui = ui_context

    def setup(self):
        self.ui._tools.stop_btn.name = 'STOP'
        self.ui._tools.stop_btn.button_type = 'danger'

    def before(self):
        self.ui._tools.stop_btn.disabled = False

    def during(self):
        ...

    def after(self):
        self.ui._tools.stop_btn.disabled = True


class OfflineLifecycle(UILifecycle):
    def setup(self):
        super().setup()
        self.after()
        self.ui._tools.set_subtitle(UIState.OFFLINE.value)

    def before(self):
        super().before()
        self.ui._tools.run_btn.name = 'Waiting...'
        self.ui._tools.run_btn.disabled = True

    def during(self):
        self.ui._tools.run_btn.name = 'Running...'

    def after(self):
        super().after()
        self.ui._tools.run_btn.name = 'Run all'
        self.ui._tools.run_btn.disabled = False


class LiveLifecycle(UILifecycle):
    def setup(self):
        super().setup()
        self.after()
        self.ui._tools.replay_select.visible = False
        self.ui._tools.set_subtitle(UIState.LIVE.value)
        self.ui._tools.mode_btn.name = f'Go to {UIState.REPLAY.value}'
        self.ui._tools.continuous_btn.visible = True

    def before(self, is_continuous: bool = False):
        super().before()
        self.ui._tools.run_btn.disabled = True
        self.ui._tools.run_btn.name = 'Waiting trigger...'
        self.ui._tools.mode_btn.disabled = True
        self.ui._tools.continuous_btn.disabled = not is_continuous

    def during(self):
        self.ui._tools.run_btn.name = 'Acquiring...'

    def after(self):
        super().after()
        self.ui._tools.run_btn.disabled = False
        self.ui._tools.run_btn.name = 'Run once'
        self.ui._tools.mode_btn.disabled = len(self.ui._resources.recordings) == 0
        self.ui._tools.continuous_btn.disabled = False
        self.ui._tools.continuous_btn.name = 'Run continuous'
        self.ui._tools.continuous_btn.button_type = 'success'


class ContinuousLifecycle(LiveLifecycle):
    def before(self):
        super().before(is_continuous=True)
        self.ui._tools.continuous_btn.button_type = 'danger'
        self.ui._tools.continuous_btn.name = 'Stop continuous'


class ReplayLifecycle(OfflineLifecycle):
    def setup(self):
        self.after()
        self.ui._tools.replay_select.visible = True
        self.ui._tools.set_subtitle(UIState.REPLAY.value)
        self.ui._tools.mode_btn.name = f'Go to {UIState.LIVE.value}'
        self.ui._tools.mode_btn.disabled = False
        self.ui._tools.continuous_btn.visible = False

    def before(self):
        super().before()
        self.ui._tools.mode_btn.disabled = True

    def after(self):
        super().after()
        self.ui._tools.mode_btn.disabled = False
