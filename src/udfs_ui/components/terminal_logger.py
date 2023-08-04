from __future__ import annotations
import logging
import traceback
import panel as pn
pn.extension('terminal')

DEFAULT_LEVEL = logging.DEBUG
logger = logging.getLogger("ui_context")
logger.setLevel(DEFAULT_LEVEL)


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
            min_height=200,
        )

        self.stream_handler = logging.StreamHandler(self._terminal)
        self.stream_handler.terminator = "  \n"
        formatter = logging.Formatter("%(asctime)s [%(levelname)s]: %(message)s")

        self.stream_handler.setFormatter(formatter)
        self.stream_handler.setLevel(DEFAULT_LEVEL)
        self.logger.addHandler(self.stream_handler)

    @property
    def logger(self):
        return logger

    def set_level(self, log_level: int | str):
        self.logger.setLevel(log_level)
        self.stream_handler.setLevel(log_level)

    @property
    def widget(self):
        return self._terminal

    def log_from_exception(self, err: Exception, reraise: bool = False, msg: str | None = None):
        te = traceback.TracebackException.from_exception(err)
        self.logger.error('\n' + ''.join(te.stack.format()) + f'\n{type(err).__name__}: {err}')
        if msg is not None:
            self.logger.error(msg)
        if reraise:
            raise err
