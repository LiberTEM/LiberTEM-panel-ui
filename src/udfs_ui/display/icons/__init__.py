from __future__ import annotations
import pathlib
import functools

_file_location = pathlib.Path(__file__).absolute().parent


def imread(*args, **kwargs):
    from skimage.io import imread
    return imread(*args, **kwargs)


@functools.lru_cache(1)
def cursor_icon():
    return imread(_file_location / 'cursor.png')


@functools.lru_cache(1)
def xselect_icon():
    return imread(_file_location / 'xselect.png')


@functools.lru_cache(1)
def yselect_icon():
    return imread(_file_location / 'yselect.png')


@functools.lru_cache(1)
def line_icon():
    return imread(_file_location / 'LineEdit.png')


@functools.lru_cache(1)
def free_point():
    return imread(_file_location / 'freepoint.png')


@functools.lru_cache(1)
def free_draw():
    return imread(_file_location / 'freedraw.png')


@functools.lru_cache(1)
def options_icon():
    return imread(_file_location / 'options.png')
