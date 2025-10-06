from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from skimage.draw import polygon as draw_polygon

from bokeh.models.sources import ColumnDataSource
from bokeh.models.glyphs import Scatter, Rect, Patches
from bokeh.models.tools import BoxEditTool, PolyDrawTool, PolyEditTool

from ..utils import clip_posxy_array
from .base import DisplayBase, ConsBase
from .points import PointSet


if TYPE_CHECKING:
    from bokeh.plotting import figure as BkFigure
    from bokeh.models.tools import EditTool


class Rectangles(DisplayBase):
    glyph_map = {
        'rectangles': [Rect],
    }

    def __init__(
        self,
        cds: ColumnDataSource,
        x='cx',
        y='cy',
        width='w',
        height='h',
    ):
        super().__init__(cds)
        glyph = Rect(
            x=x,
            y=y,
            width=width,
            height=height,
            fill_alpha=0.3,
            fill_color='red',
            line_color='red',
            line_dash='dashed',
        )
        self._register_glyph('rectangles', glyph)

    def editable(
        self,
        *figs: BkFigure,
        tag_name: str = 'default',
        selected: bool = False,
    ) -> Rectangles:

        def _make_tool():
            return BoxEditTool(
                name='Rectangle Draw/Edit',
                description='Draw rectangles on figure',
                renderers=[],
                tags=[tag_name],
            )

        self._add_to_tool(
            figs=figs,
            glyph_name='rectangles',
            tool_filter=lambda t: tag_name in t.tags and isinstance(t, BoxEditTool),
            make_tool=_make_tool,
            selected=selected,
        )
        return self

    @property
    def rectangles(self) -> Rect:
        return self._glyphs['rectangles'][0].glyph

    @classmethod
    def new(cls):
        return RectanglesCons()

    def update(
        self,
        x: np.ndarray | None = None,
        y: np.ndarray | None = None,
        width: np.ndarray | float | None = None,
        height: np.ndarray | float | None = None,
    ):
        if np.isscalar(width):
            width = np.full((self.data_length,), width, dtype=float).tolist()
        if np.isscalar(height):
            height = np.full((self.data_length,), height, dtype=float).tolist()
        data = {}
        data[self.rectangles.x] = x
        data[self.rectangles.y] = y
        data[self.rectangles.width] = width
        data[self.rectangles.height] = height
        return super().update(**data)

    def as_mask(self, shape: tuple[int, int]):
        if self.data_length == 0:
            return None
        mask = np.zeros(shape, dtype=bool)
        for _, row in self.cds.to_df().iterrows():
            mask = rectangle_to_mask(
                cx=row[self.rectangles.x],
                cy=row[self.rectangles.y],
                w=abs(row[self.rectangles.width]),
                h=abs(row[self.rectangles.height]),
                mask=mask
            )
        return mask

    def as_slices(self, shape: tuple[int, int]) -> list[tuple[slice, slice]]:
        slices = []
        for _, row in self.cds.to_df().iterrows():
            slices.append(rectangle_to_slice(
                cx=row[self.rectangles.x],
                cy=row[self.rectangles.y],
                w=abs(row[self.rectangles.width]),
                h=abs(row[self.rectangles.height]),
                shape=shape
            ))
        return slices


def rectangle_to_slice(*, cx, cy, w, h, shape):
    lefttop = cx - w / 2, cy - h / 2
    rightbottom = cx + w / 2, cy + h / 2
    lefttop, _ = clip_posxy_array(lefttop, shape, round=True, to_int=True)
    rightbottom, _ = clip_posxy_array(rightbottom, shape, round=True, to_int=True)
    slice_y = slice(lefttop[1], rightbottom[1] + 1)
    slice_x = slice(lefttop[0], rightbottom[0] + 1)
    return slice_y, slice_x


def rectangle_to_mask(*, cx, cy, w, h, mask, fill_value: bool = True):
    slices = rectangle_to_slice(cx=cx, cy=cy, w=w, h=h, shape=mask.shape)
    mask[slices] = fill_value
    return mask


class RectanglesCons(ConsBase):
    default_keys = ('cx', 'cy', 'w', 'h')

    @classmethod
    def from_vectors(
        cls,
        x: np.ndarray,
        y: np.ndarray,
        width: np.ndarray | float,
        height: np.ndarray | float,
    ) -> Rectangles:
        if np.isscalar(width):
            width = np.full_like(x, width, dtype=float).tolist()
        if np.isscalar(height):
            height = np.full_like(x, height, dtype=float).tolist()
        data = {
            k: v for k, v in zip(cls.default_keys, (x, y, width, height))
        }
        cds = ColumnDataSource(data)
        return Rectangles(cds)

    @classmethod
    def empty(cls):
        return super().empty(Rectangles)


class VertexPointSetMixin:
    _vertex_pointset: PointSet

    @property
    def vertices(self) -> PointSet | None:
        try:
            return self._vertex_pointset
        except AttributeError:
            return None

    def _setup_vertex_renderer(
        self,
        where: list[tuple[BkFigure, EditTool]]
    ):
        """
        This DisplayBase is used by PolyEditTool/PolyDrawTool
        to display the vertice of polygons / lines. If the tool is first
        added to a figure by this instance, it will 'own'
        the vertex renderer, otherwise it will 'borrow' the
        PointSet / vertex renderer of another DisplayBase

        To style the vertex glyph in this case, use the 'owner'
        instance or find the associated glyph through the tool itself
        """
        if not hasattr(self, '_vertex_pointset'):
            self._vertex_pointset = PointSet.new().empty()
        for fig, poly_tool in where:
            if poly_tool.vertex_renderer is not None:
                continue
            renderers = self._vertex_pointset.renderers_for_fig('points', fig)
            if len(renderers) == 0:
                vertex_renderer = (
                    self._vertex_pointset
                    .on(fig)
                    .renderers_for_fig('points', fig)[0]
                )
            else:
                vertex_renderer = renderers[0]
            poly_tool.vertex_renderer = vertex_renderer


class Polygons(DisplayBase, VertexPointSetMixin):
    glyph_map = {
        'polys': [Patches],
    }

    def __init__(
        self,
        cds: ColumnDataSource,
        xs='xs',
        ys='ys',
    ):
        super().__init__(cds)
        glyph = Patches(
            xs=xs,
            ys=ys,
            fill_alpha=0.3,
            fill_color='red',
            line_color='red',
            line_dash='dashed',
        )
        self._register_glyph('polys', glyph)

    def editable(
        self,
        *figs: BkFigure,
        tag_name: str = 'default',
        selected: bool = False,
    ) -> Polygons:

        def _make_draw_tool():
            return PolyDrawTool(
                name='Polygon Draw',
                description='Draw polygons on figure',
                renderers=[],
                tags=[tag_name],
            )
        where = self._add_to_tool(
            figs=figs,
            glyph_name='polys',
            tool_filter=lambda t: tag_name in t.tags and isinstance(t, PolyDrawTool),
            make_tool=_make_draw_tool,
            selected=selected,
        )
        self._setup_vertex_renderer(where)

        def _make_edit_tool():
            return PolyEditTool(
                name='Polygon Draw',
                description='Edit polygons on figure',
                renderers=[],
                tags=[tag_name],
            )
        where = self._add_to_tool(
            figs=figs,
            glyph_name='polys',
            tool_filter=lambda t: tag_name in t.tags and isinstance(t, PolyEditTool),
            make_tool=_make_edit_tool,
        )
        self._setup_vertex_renderer(where)
        return self

    @property
    def polys(self) -> Patches:
        return self._glyphs['polys'][0].glyph

    @property
    def vertices(self) -> Scatter | None:
        return self._vertices_glyph

    @classmethod
    def new(cls):
        return PolygonsCons()

    def update(
        self,
        xs: list[np.ndarray],
        ys: list[np.ndarray],
    ):
        """
        Need patch methods to adjust single polygons
        Need a nicer api, too
        """
        assert all(len(x) == len(y) for x, y in zip(xs, ys))
        data = {}
        data[self.polys.xs] = xs
        data[self.polys.ys] = ys
        return super().update(**data)

    def as_mask(self, shape: tuple[int, int]):
        if self.data_length == 0:
            return None
        mask = np.zeros(shape, dtype=bool)
        for _, row in self.cds.to_df().iterrows():
            rr, cc = draw_polygon(
                row[self.polys.ys],
                row[self.polys.xs],
                shape=shape,
            )
            mask[rr, cc] = True
        return mask


class PolygonsCons(ConsBase):
    default_keys = ('xs', 'ys')

    @classmethod
    def from_pointlists(
        cls,
        *pointlists: list[tuple[float, float]],
    ) -> Polygons:
        """Note points are (x, y) pairs"""
        xs = []
        ys = []
        for pointlist in pointlists:
            xs.append([p[0] for p in pointlist])
            ys.append([p[1] for p in pointlist])
        data = {
            k: v for k, v in zip(cls.default_keys, (xs, ys))
        }
        cds = ColumnDataSource(data)
        return Polygons(cds)

    @classmethod
    def empty(cls):
        return super().empty(Polygons)
