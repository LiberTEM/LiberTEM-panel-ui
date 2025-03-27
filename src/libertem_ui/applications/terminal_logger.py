from __future__ import annotations
import logging
from io import StringIO

import panel as pn

from ..utils.logging import DEFAULT_LEVEL, logger


pn.extension(
    'terminal',
)


class UILog:
    def __init__(self):
        self._terminal = pn.widgets.Terminal(
            "UI Context Logs:\n",
            options={
                "theme": {
                    'background': '#F3F3F3',
                    'foreground': '#000000',
                },
            },
            sizing_mode='stretch_width',
            height=450,
        )

        self.stream_handler = logging.StreamHandler(self._terminal)
        self.stream_handler.terminator = "  \n"
        formatter = logging.Formatter("%(asctime)s [%(levelname)s]: %(message)s")

        self.stream_handler.setFormatter(formatter)
        self.stream_handler.setLevel(DEFAULT_LEVEL)
        self.logger.addHandler(self.stream_handler)
        self._is_held: bool = False
        self._hold_buffer = StringIO()

    @property
    def logger(self):
        return logger

    def set_level(self, log_level: int | str):
        self.logger.setLevel(log_level)
        self.stream_handler.setLevel(log_level)

    @property
    def widget(self):
        return self._terminal

    def set_hold(self, hold: bool):
        if hold == self._is_held:
            # Nothing to do
            pass
        elif hold:
            # Implementation of setStream will flush first
            self.stream_handler.setStream(self._hold_buffer)
            self._is_held = True
        else:
            # Write contents of buffer to terminal and create new buffer
            self.stream_handler.setStream(self._terminal)
            messages = self._hold_buffer.getvalue()
            self._hold_buffer = StringIO()
            self._terminal.write(messages)
            self._is_held = False

    def hold_callback(self, e):
        return self.set_hold(e.new)

    def as_collapsible(self, title: str = 'Logs', collapsed: bool = True):
        from ..utils.panel_components import minimal_card

        logger_card = minimal_card(
            title,
            self.widget,
            collapsed=collapsed,
        )
        self.set_hold(collapsed)
        logger_card.param.watch(self.hold_callback, 'collapsed')
        return logger_card
