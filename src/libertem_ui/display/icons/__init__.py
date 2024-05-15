from __future__ import annotations
from typing import TYPE_CHECKING
import pathlib
import functools
import base64

if TYPE_CHECKING:
    import numpy as np

_file_location = pathlib.Path(__file__).absolute().parent
png_header = 'data:image/png;base64,'


def imread(*args, **kwargs) -> np.ndarray:
    from skimage.io import imread
    return imread(*args, **kwargs)


def load_as_b64(filepath) -> str:
    with open(filepath, 'rb') as fp:
        data = fp.read()
    return png_header + base64.b64encode(data).decode('utf-8')


@functools.lru_cache(2)
def cursor_icon(as_b64: bool = False):
    path = _file_location / 'cursor.png'
    if as_b64:
        return load_as_b64(path)
    return imread(path)


@functools.lru_cache(2)
def xselect_icon(as_b64: bool = False):
    path = _file_location / 'xselect.png'
    if as_b64:
        return load_as_b64(path)
    return imread(path)


@functools.lru_cache(2)
def yselect_icon(as_b64: bool = False):
    path = _file_location / 'yselect.png'
    if as_b64:
        return load_as_b64(path)
    return imread(path)


@functools.lru_cache(2)
def line_icon(as_b64: bool = False):
    path = _file_location / 'LineEdit.png'
    if as_b64:
        return load_as_b64(path)
    return imread(path)


@functools.lru_cache(2)
def free_point(as_b64: bool = False):
    path = _file_location / 'freepoint.png'
    if as_b64:
        return load_as_b64(path)
    return imread(path)


@functools.lru_cache(2)
def free_draw(as_b64: bool = False):
    path = _file_location / 'freedraw.png'
    if as_b64:
        return load_as_b64(path)
    return imread(path)


@functools.lru_cache(2)
def options_icon(as_b64: bool = False):
    path = _file_location / 'options.png'
    if as_b64:
        return load_as_b64(path)
    return imread(path)


@functools.lru_cache(2)
def options_icon_blue(as_b64: bool = False):
    path = _file_location / 'options_blue.png'
    if as_b64:
        return load_as_b64(path)
    return imread(path)


@functools.lru_cache(2)
def sigma_icon(as_b64: bool = False):
    path = _file_location / 'sigma.png'
    if as_b64:
        return load_as_b64(path)
    return imread(path)
