import traceback
import logging
from logging import Logger

DEFAULT_LEVEL = logging.DEBUG


class UILogger(Logger):
    def log_from_exception(self, err: Exception, reraise: bool = False, msg: str | None = None):
        te = traceback.TracebackException.from_exception(err)
        self.error('\n' + ''.join(te.stack.format()) + f'\n{type(err).__name__}: {err}')
        if msg is not None:
            self.error(msg)
        if reraise:
            raise err


logging.setLoggerClass(UILogger)
logger: UILogger = logging.getLogger("ui_context")
logging.setLoggerClass(Logger)
logger.setLevel(DEFAULT_LEVEL)
