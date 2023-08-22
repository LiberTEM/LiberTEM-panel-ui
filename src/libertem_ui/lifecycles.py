from __future__ import annotations
from typing import TYPE_CHECKING

from .base import UIState

if TYPE_CHECKING:
    from .ui_context import UIContext


class UILifecycle:
    def __init__(self, ui_context: UIContext):
        self.ui = ui_context
        self.enabled = True

    def disable(self):
        self.enabled = False

    def setup(self):
        self.ui._tools.stop_btn.param.update(
            name='STOP',
            button_type='danger',
        )

    def before(self):
        ...

    def during(self):
        ...

    def after(self):
        ...


class OfflineLifecycle(UILifecycle):
    def setup(self):
        super().setup()
        self.after()

    def before(self):
        if not self.enabled:
            return
        super().before()
        self.ui._tools.run_btn.param.update(
            name='Waiting...',
            disabled=True
        )

    def during(self):
        if not self.enabled:
            return
        self.ui._tools.run_btn.name = 'Running...'

    def after(self):
        if not self.enabled:
            return
        super().after()
        self.ui._tools.run_btn.param.update(
            name='Run all',
            disabled=False
        )


class LiveLifecycle(UILifecycle):
    def setup(self):
        super().setup()
        self.after()
        self.ui._tools.replay_select.visible = False
        self.ui._tools.mode_btn.name = f'Go to {UIState.REPLAY.value}'
        self.ui._tools.continuous_btn.visible = True
        # Toggles
        self.ui._tools.roi_toggle_btn.value = False
        self.ui._tools.roi_toggle_btn.disabled = True
        self.ui._tools.roi_toggle_btn.visible = False
        self.ui._tools.roi_toggle_txt.visible = False
        self.ui._tools.record_toggle_btn.disabled = False
        self.ui._tools.record_toggle_btn.visible = True
        self.ui._tools.record_toggle_txt.visible = True
        self.ui._tools.monitor_toggle_btn.disabled = False
        self.ui._tools.monitor_toggle_btn.visible = True
        self.ui._tools.monitor_toggle_txt.visible = True

    def before(self, is_continuous: bool = False):
        if not self.enabled:
            return
        super().before()
        self.ui._tools.run_btn.disabled = True
        self.ui._tools.run_btn.name = 'Waiting trigger...'
        self.ui._tools.mode_btn.disabled = True
        self.ui._tools.continuous_btn.disabled = not is_continuous

    def during(self):
        if not self.enabled:
            return
        self.ui._tools.run_btn.name = 'Acquiring...'

    def after(self):
        if not self.enabled:
            return
        super().after()
        self.ui._tools.run_btn.disabled = False
        self.ui._tools.run_btn.name = 'Run once'
        self.ui._tools.mode_btn.disabled = False
        self.ui._tools.continuous_btn.disabled = False
        self.ui._tools.continuous_btn.name = 'Run continuous'
        self.ui._tools.continuous_btn.button_type = 'success'


class ContinuousLifecycle(LiveLifecycle):
    def __init__(self, ui_context: UIContext):
        super().__init__(ui_context)
        self._before_done = False

    def before(self):
        if self._before_done or (not self.enabled):
            return
        super().before(is_continuous=True)
        self.ui._tools.continuous_btn.button_type = 'danger'
        self.ui._tools.continuous_btn.name = 'Stop continuous'
        self._before_done = True


class ReplayLifecycle(OfflineLifecycle):
    def setup(self):
        self.after()
        self.ui._tools.replay_select.visible = True
        self.ui._tools.mode_btn.name = f'Go to {UIState.LIVE.value}'
        self.ui._tools.mode_btn.disabled = False
        self.ui._tools.continuous_btn.visible = False
        # Toggles
        self.ui._tools.roi_toggle_btn.disabled = False
        self.ui._tools.roi_toggle_btn.visible = True
        self.ui._tools.roi_toggle_txt.visible = True
        self.ui._tools.record_toggle_btn.value = False
        self.ui._tools.record_toggle_btn.disabled = True
        self.ui._tools.record_toggle_btn.visible = False
        self.ui._tools.record_toggle_txt.visible = False
        self.ui._tools.monitor_toggle_btn.value = False
        self.ui._tools.monitor_toggle_btn.disabled = True
        self.ui._tools.monitor_toggle_btn.visible = False
        self.ui._tools.monitor_toggle_txt.visible = False

    def before(self):
        if not self.enabled:
            return
        super().before()
        self.ui._tools.mode_btn.disabled = True

    def after(self):
        if not self.enabled:
            return
        super().after()
        self.ui._tools.mode_btn.disabled = False
