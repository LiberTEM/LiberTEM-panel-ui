from __future__ import annotations
from typing import NamedTuple


def pop_from_list(sequence, el, lazy=True):
    try:
        idx = sequence.index(el)
        sequence.pop(idx)
    except ValueError as e:
        if not lazy:
            raise e


def get_initial_pos(shape: tuple[int, int]):
    h, w = shape
    cy, cx = h // 2, w // 2
    ri, r = h // 6, w // 4
    return tuple(map(float, (cy, cx))), tuple(map(float, (ri, r))), float(max(h, w)) * 0.5


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
