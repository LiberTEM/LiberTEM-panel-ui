from __future__ import annotations
from functools import partial
from typing import TYPE_CHECKING
import numpy as np

from bokeh.models.sources import ColumnDataSource
from bokeh.models.glyphs import Scatter
from bokeh.models.tools import PointDrawTool

from .display_base import DisplayBase, ConsBase
from .icons import cursor_icon
from ..utils import PointXY
from .points import get_point_tool

if TYPE_CHECKING:
    from bokeh.plotting import figure as BkFigure


class Cursor(DisplayBase):
    glyph_map = {
        'cursor': Scatter,
    }

    def __init__(
        self,
        cds: ColumnDataSource,
        x: str = 'cx',
        y: str = 'cy',
    ):
        super().__init__(cds)
        glyph = Scatter(
            marker='circle_dot',
            x=x,
            y=y,
            line_color='orange',
            line_width=2,
            fill_alpha=0,
            size=15,
            hit_dilation=2.0,
        )
        self._register_glyph('cursor', glyph)

    @property
    def cursor(self) -> Scatter:
        return self._glyphs['cursor'][0].glyph

    @classmethod
    def new(cls):
        return CursorCons()

    def update(
        self,
        x: float | None = None,
        y: float | None = None,
    ):
        data = {}
        data[self.cursor.x] = [x]
        data[self.cursor.y] = [y]
        return super().update(**data)

    def editable(
        self,
        *figs: BkFigure,
        tag_name: str = 'cursor',
        selected: bool = False,
    ) -> Cursor:
        self._add_to_tool(
            figs=figs,
            glyph_name='cursor',
            tool_filter=lambda t: tag_name in t.tags and isinstance(t, PointDrawTool),
            make_tool=partial(
                get_point_tool,
                add=False,
                drag=True,
                tag_name=tag_name,
                icon=cursor_icon(),
            ),
            selected=selected,
        )
        return self

    def current_pos(
        self,
        to_int: bool = False,
        clip_to: tuple[int, int] | None = None,
    ):
        try:
            x: float = self.cds.data[self.cursor.x][0]
            y: float = self.cds.data[self.cursor.y][0]
        except (KeyError, IndexError):
            return None
        if to_int:
            x = int(np.round(x))
            y = int(np.round(y))
        if clip_to is not None:
            h, w = clip_to
            if not ((0 <= x < w) and (0 <= y < h)):
                return None
        return PointXY(x, y)

    def reset(self, *e):
        # In case a cursor is deleted, provide button
        # to reset it to a single-point, centered CDS
        raise NotImplementedError()

    def on(self, *figs: BkFigure):
        super().on(*figs)
        # Could move render level control onto baseclass
        # or have it as a kwarg to .on when we create the rendere
        for fig in figs:
            for renderer in self.renderers_for_fig('cursor', fig):
                renderer.level = 'annotation'
        return self


class CursorCons(ConsBase):
    default_keys = ('cx', 'cy')

    @staticmethod
    def from_pos(
        x: float,
        y: float,
    ) -> Cursor:
        data = {
            k: [v] for k, v in zip(CursorCons.default_keys, (x, y))
        }
        cds = ColumnDataSource(data)
        return Cursor(cds)

    @classmethod
    def empty(cls) -> Cursor:
        return super().empty()
