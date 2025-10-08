from __future__ import annotations
from functools import partial
from typing import TYPE_CHECKING
import numpy as np
import panel as pn

from bokeh.models.sources import ColumnDataSource
from bokeh.models.glyphs import Scatter, Circle, Annulus
from bokeh.models.tools import PointDrawTool

from .base import DisplayBase, ConsBase


if TYPE_CHECKING:
    from bokeh.plotting import figure as BkFigure


def get_point_tool(
    add: bool = True,
    drag: bool = True,
    num_objects: int = 0,
    empty_value: float = 1.,
    tag_name: str = 'default',
    name='Point Draw/Edit',
    description='Draw points on figure',
    icon=None,
):
    return PointDrawTool(
        name=name,
        description=description,
        renderers=[],
        add=add,
        drag=drag,
        num_objects=num_objects,
        empty_value=empty_value,
        tags=[tag_name],
        icon=icon,
    )


class PointSet(DisplayBase[Scatter]):
    glyph_map = {
        'points': Scatter
    }

    def __init__(
        self,
        cds: ColumnDataSource,
        x: str = 'cx',
        y: str = 'cy',
    ):
        super().__init__(cds)
        glyph = Scatter(
            marker='circle',
            x=x,
            y=y,
            line_color=None,
            fill_color='red',
            fill_alpha=1.,
            size=10,
        )
        self._register_glyph('points', glyph)

    @property
    def points(self) -> Scatter:
        return self._glyphs['points'][0].glyph

    @classmethod
    def new(cls):
        return PointSetCons()

    def update(
        self,
        x: np.ndarray | None = None,
        y: np.ndarray | None = None,
    ):
        data = {}
        data[self.points.x] = x
        data[self.points.y] = y
        return super().update(**data)

    def editable(
        self,
        *figs: BkFigure,
        add: bool = True,
        drag: bool = True,
        tag_name: str = 'default',
        selected: bool = False,
    ) -> PointSet:
        if not (add or drag):
            raise ValueError('Cannot make editable without one of add or drag')
        self._add_to_tool(
            figs=figs,
            glyph_name='points',
            tool_filter=lambda t: tag_name in t.tags and isinstance(t, PointDrawTool),
            make_tool=partial(get_point_tool, add=add, drag=drag, tag_name=tag_name),
            selected=selected,
        )
        return self


class PointSetCons(ConsBase[PointSet]):
    constructs = PointSet
    default_keys = ('cx', 'cy')

    @staticmethod
    def from_vectors(
        x: np.ndarray,
        y: np.ndarray,
    ) -> PointSet:
        data = {
            k: v for k, v in zip(PointSetCons.default_keys, (x, y))
        }
        cds = ColumnDataSource(data)
        return PointSet(cds)

    @staticmethod
    def from_array(
        array: np.ndarray,
    ):
        raise NotImplementedError


class DiskSet(DisplayBase[Circle]):
    glyph_map = {
        'disks': Circle,
    }

    def __init__(
        self,
        cds: ColumnDataSource,
        x: str = 'cx',
        y: str = 'cy',
        radius: str = 'r0',
    ):
        super().__init__(cds)
        glyph = Circle(
            x=x,
            y=y,
            radius=radius,
            radius_units='data',
            line_color='red',
            line_width=2,
            fill_color='red',
            fill_alpha=0.5,
        )
        self._register_glyph('disks', glyph)

    @property
    def disks(self) -> Circle:
        return self._glyphs['disks'][0].glyph

    @classmethod
    def new(cls):
        return DiskSetCons()

    def update(
        self,
        x: np.ndarray | None = None,
        y: np.ndarray | None = None,
        radius: np.ndarray | float | None = None,
    ):
        if np.isscalar(radius):
            radius = np.full((self.data_length,), radius, dtype=float).tolist()
        data = {}
        data[self.disks.x] = x
        data[self.disks.y] = y
        data[self.disks.radius] = radius
        return super().update(**data)

    def editable(
        self,
        *figs: BkFigure,
        add: bool = True,
        drag: bool = True,
        tag_name: str = 'default',
        selected: bool = False,
    ) -> DiskSet:
        if not (add or drag):
            raise ValueError('Cannot make editable without one of add or drag')
        self._add_to_tool(
            figs=figs,
            glyph_name='disks',
            tool_filter=lambda t: tag_name in t.tags and isinstance(t, PointDrawTool),
            make_tool=partial(get_point_tool, add=add, drag=drag, tag_name=tag_name),
            selected=selected,
        )
        return self

    def get_radius_slider(self, max_r: float, min_r: float = 1.0, label: str = 'Disk radius'):
        try:
            initial_radius = self.cds.data[self.disks.radius][0]
        except IndexError:
            initial_radius = (max_r + min_r) / 2.

        slider = pn.widgets.FloatSlider(
            name=label,
            value=initial_radius,
            start=1.,
            end=max_r,
        )
        slider.param.watch(self._update_radius, 'value_throttled')
        slider.jscallback(
            value="""
cds.data[glyph.radius.field].fill(cb_obj.value);
cds.change.emit();
""",
            args={
                'cds': self.cds,
                'glyph': self.disks,
            },
        )
        return slider

    def _update_radius(self, e):
        self.update(radius=e.new)


class DiskSetCons(ConsBase[DiskSet]):
    constructs = DiskSet
    default_keys = ('cx', 'cy', 'r0')

    @staticmethod
    def from_vectors(
        x: np.ndarray,
        y: np.ndarray,
        radius: np.ndarray | float,
    ) -> DiskSet:
        if np.isscalar(radius):
            radius = np.full_like(x, radius, dtype=float).tolist()
        data = {
            k: v for k, v in zip(DiskSetCons.default_keys, (x, y, radius))
        }
        cds = ColumnDataSource(data)
        return DiskSet(cds)


class RingSet(DisplayBase[Annulus]):
    glyph_map = {
        'rings': Annulus,
    }

    def __init__(
        self,
        cds: ColumnDataSource,
        x: str = 'cx',
        y: str = 'cy',
        inner_radius: str = 'r0',
        outer_radius: str = 'r1',
    ):
        super().__init__(cds)
        glyph = Annulus(
            x=x,
            y=y,
            inner_radius=inner_radius,
            outer_radius=outer_radius,
            inner_radius_units='data',
            outer_radius_units='data',
            line_color='red',
            line_width=2,
            fill_color='red',
            fill_alpha=0.5,
        )
        self._register_glyph('rings', glyph)

    @property
    def rings(self) -> Annulus:
        return self._glyphs['rings'][0].glyph

    @classmethod
    def new(cls):
        return RingSetCons()

    def update(
        self,
        x: np.ndarray | None = None,
        y: np.ndarray | None = None,
        inner_radius: np.ndarray | float | None = None,
        outer_radius: np.ndarray | float | None = None,
    ):
        if np.isscalar(inner_radius):
            inner_radius = np.full((self.data_length,), inner_radius, dtype=float).tolist()
        if np.isscalar(outer_radius):
            outer_radius = np.full((self.data_length,), outer_radius, dtype=float).tolist()
        data = {}
        data[self.rings.x] = x
        data[self.rings.y] = y
        data[self.rings.inner_radius] = inner_radius
        data[self.rings.outer_radius] = outer_radius
        return super().update(**data)

    def editable(
        self,
        *figs: BkFigure,
        add: bool = True,
        drag: bool = True,
        tag_name: str = 'default',
        selected: bool = False,
    ) -> RingSet:
        if not (add or drag):
            raise ValueError('Cannot make editable without one of add or drag')
        self._add_to_tool(
            figs=figs,
            glyph_name='rings',
            tool_filter=lambda t: tag_name in t.tags and isinstance(t, PointDrawTool),
            make_tool=partial(get_point_tool, add=add, drag=drag, tag_name=tag_name),
            selected=selected,
        )
        return self


class RingSetCons(ConsBase[RingSet]):
    constructs = RingSet
    default_keys = ('cx', 'cy', 'r0', 'r1')

    @staticmethod
    def from_vectors(
        x: np.ndarray,
        y: np.ndarray,
        inner_radius: np.ndarray | float,
        outer_radius: np.ndarray | float,
    ) -> RingSet:
        if np.isscalar(inner_radius):
            inner_radius = np.full_like(x, inner_radius, dtype=float).tolist()
        if np.isscalar(outer_radius):
            outer_radius = np.full_like(x, outer_radius, dtype=float).tolist()
        data = {
            k: v for k, v in zip(RingSetCons.default_keys, (x, y, inner_radius, outer_radius))
        }
        cds = ColumnDataSource(data)
        return RingSet(cds)
