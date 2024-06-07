__version__ = '0.0.1'

import pathlib
_static_root = pathlib.Path(__file__).parent / '_static'
logo_path = _static_root / 'lt_logo.svg'
icon_path = _static_root / 'lt_icon.svg'


import panel as pn  # noqa

pn.extension(
    'tabulator',
    'jsoneditor',
    'terminal',
    'floatpanel',
    css_files=[
        pn.io.resources.CSS_URLS['font-awesome']
    ],
)
