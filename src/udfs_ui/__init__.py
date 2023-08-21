import pathlib
# This needs to be refactored to reduce import times
from .windows import imaging  # noqa
from .windows import com  # noqa
from .windows import tools  # noqa

_static_root = pathlib.Path(__file__).parent / '_static'
logo_path = _static_root / 'lt_logo.svg'
icon_path = _static_root / 'lt_icon.svg'
