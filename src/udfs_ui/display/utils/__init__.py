from __future__ import annotations
from collections import namedtuple
from typing import NamedTuple

from ...utils import random_hash


def slider_step_size(start, end, n=300):
    return abs(end - start) / n


def get_random_name(glyph_base):
    return f'{glyph_base.__class__.__name__}-{random_hash()}'


BokehToPanelEvent = namedtuple('BokehToPanelEvent', ['old', 'new'])


class PointYX(NamedTuple):
    y: int | float
    x: int | float

    def as_xy(self):
        return PointXY(self.x, self.y)
    
    def as_yx(self):
        return self

    def as_int(self):
        # Could be refactored with generic type on mixin
        return self.__class__(*map(int, self))


class PointXY(NamedTuple):
    x: int | float
    y: int | float

    def as_yx(self):
        return PointYX(self.y, self.x)

    def as_xy(self):
        return self

    def as_int(self):
        # Could be refactored with generic type on mixin
        return self.__class__(*map(int, self))
