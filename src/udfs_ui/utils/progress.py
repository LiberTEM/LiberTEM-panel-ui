import time
from libertem.common.progress import TQDMProgressReporter


class PanelProgressReporter(TQDMProgressReporter):
    time_inc = 1.

    def __init__(self, bar):
        super().__init__()
        self._bar_pane = bar

    def start(self, state):
        self._tc = time.time()
        self._bar = self._bar_pane(desc=self._get_description(state),
                                   total=state.num_frames_total,
                                   leave=True)

    def update(self, state):
        t = time.time()
        if (t - self._tc) > self.time_inc:
            self._tc = t
            return self._update(state, clip=True, refresh=False)
