from __future__ import annotations
import abc
from functools import partial
from dataclasses import dataclass, field
import numpy as np
import itertools
import pandas as pd
import panel as pn
from typing import TYPE_CHECKING, Sequence, NamedTuple, Callable, TypeVar, Generator
from typing_extensions import Self
import colorcet as cc
from skimage.draw import polygon as draw_polygon

from bokeh.models.sources import ColumnDataSource
from bokeh.models.glyphs import Line, Scatter, Circle, Annulus, Rect, Patches
from bokeh.models.glyphs import Text as BkText
from bokeh.models.tools import PointDrawTool, BoxEditTool, PolyDrawTool, PolyEditTool

from .icons import cursor_icon
from ..utils import pop_from_list, PointXY, clip_posxy_array


if TYPE_CHECKING:
    from bokeh.plotting import figure as BkFigure
    from bokeh.models.glyphs import Glyph
    from bokeh.models.renderers import GlyphRenderer
    from bokeh.models.tools import EditTool, Tool


def unique_length(*items):
    lengths = {len(o) for o in items}
    return len(lengths) == 1


class RenderedOn(NamedTuple):
    fig: BkFigure
    renderer: GlyphRenderer


@dataclass
class GlyphOnWrapper:
    glyph: Glyph
    # Reference to where the Glyphs provided by this DisplayBase
    # are currently rendered, allows us to remove the display
    # both globally, per-figure and per-glyph (or both)
    on: list[RenderedOn] = field(default_factory=list)


class DisplayBase(abc.ABC):
    """
    Abstract class for interacting with a source of
    data and displaying it on one-or-more BokehFigure.s

    Represents one or more bokeh.models.glyphs.glyph
    which can each be rendered on one or more figures.

    Each glyph will be assigned a renderer by Bokeh on each
    figure it is added to, and each of these renderers
    will share the configuration of the underlying glyph (color etc).

    If the same data needs to be represented by a different
    glyph, e.g. to get different color / symbol / size on a
    different figure, use a new instance of the same DisplayBase
    initialised with the same data source, which will then get its
    own renderer in turn.

    Need to handle a DisplayBase spawning multiple copies
    of the same Glyph or child DisplayBase, e.g. MultiLine
    spawns multiple Line, the number of which is a runtime
    value so they can't have individual 'names' for finding
    defaults

    Don't need to set defaults at construction time
    Better to directly modify Bokeh endpoints and only
    provide convenience methods where needed

    A DisplayBase becomes a wrapper around a CDS and some
    Glyph(Renderer)s on one-or-more figures, with the ability
    to add tools, create widgets to modify the DisplayBase data.
    Don't handle exposing or propagating defaults
    Need to handle mapping CDS keys for basic construction
    and setting callbacks. Convenience methods can be provided to
    handle changing from fixed values to CDSKeys, not at construction
    time. Or the user interacts directly with the Bokeh API to do this.

    A DisplayBase could reject being placed onto other figures (Image.s)

    Multiple children could be handled by allowing a callable(idx: int)
    to be used to provide the name for a given Glyph or child

    Consider wrapping ColumnDataSource with a mechanism to
    generate fill values
    """

    # Provides the names and Glyph types or child DisplayBase
    # implemented by this DisplayBase
    glyph_map: dict[str, type[Glyph] | type[DisplayBase]] = {}

    def __init__(
        self,
        cds: ColumnDataSource,
    ):
        # The data source to be displayed through this DisplayBase
        self.cds = cds
        # The instances of Glyphs defined on this DisplayBase
        self._glyphs: dict[str, list[GlyphOnWrapper]] = {}
        # The instances of child DisplayBase defined on this DisplayBase
        self._children: dict[str, list[DisplayBase]] = {}

    def _register_glyph(self, name: str, glyph: Glyph):
        assert name in self.glyph_map
        wrapped = GlyphOnWrapper(glyph=glyph)
        try:
            self._glyphs[name].append(wrapped)
        except KeyError:
            self._glyphs[name] = [wrapped]

    def _register_child(self, name: str, child: DisplayBase):
        assert name in self.glyph_map
        try:
            self._children[name].append(child)
        except KeyError:
            self._children[name] = [child]

    def on(self, *figs: BkFigure) -> Self:
        """
        Add the DisplayBase to one-or-more figures

        In this case there is no distinction made between
        the display on each figure, so the display
        on each will be identical

        For true multi-figure DisplayBase the signature
        can should multiple fig args (with names
        to distinguish them)
        """
        for fig in figs:
            for wrappers in self._glyphs.values():
                for wrapper in wrappers:
                    renderer = fig.add_glyph(self.cds, wrapper.glyph)
                    wrapper.on.append(RenderedOn(fig, renderer))
        for children in self._children.values():
            for child in children:
                child.on(*figs)
        return self

    def remove(self) -> Self:
        """
        Remove DisplayBase from all registered figures
        """
        self.remove_from()
        return self

    def remove_from(self, *figs: BkFigure) -> Self:
        """
        Remove DisplayBase from specific figures, or
        all figures if *figs is empy
        """
        for wrappers in self._glyphs.values():
            wrappers: list[GlyphOnWrapper]
            for wrapper in wrappers:
                removed = []
                for on_idx, glyph_on in enumerate(wrapper.on):
                    if figs and glyph_on.fig not in figs:
                        continue
                    pop_from_list(glyph_on.fig.renderers, glyph_on.renderer)
                    removed.append(on_idx)
                    # Try to remove from existing tools if added
                    for tool in glyph_on.fig.tools:
                        try:
                            pop_from_list(tool.renderers, glyph_on.renderer)
                        except AttributeError:
                            pass
                _ = tuple(wrapper.on.pop(i) for i in reversed(removed))

        for children in self._children.values():
            for child in children:
                child.remove_from(*figs)
        return self

    def set_visible(self, visible: bool, children: bool = True) -> Self:
        for wrappers in self._glyphs.values():
            wrappers: list[GlyphOnWrapper]
            for wrapper in wrappers:
                for glyph_on in wrapper.on:
                    glyph_on.renderer.visible = visible

        if children:
            for _children in self._children.values():
                for child in _children:
                    child.set_visible(visible, children=children)
        return self

    @property
    def visible(self) -> Generator[bool]:
        for wrappers in self._glyphs.values():
            wrappers: list[GlyphOnWrapper]
            for wrapper in wrappers:
                for glyph_on in wrapper.on:
                    yield glyph_on.renderer.visible

    @property
    def data_length(self):
        try:
            col = self.cds.column_names[0]
            return len(self.cds.data[col])
        except IndexError:
            return 0

    @staticmethod
    def _update_filter_none(**data: np.ndarray | list | None):
        return {k: v for k, v in data.items() if v is not None}

    def raw_update(self, **data: np.ndarray | list) -> Self:
        """
        Update some-or-all columns in the CDS

        - Filters None values
        - Checks that updates have same length
        - Checks that update length matches CDS length
          except where replacing all values

        Will raise KeyError if trying to update non-existing columns
        Will raise VauleError if column lengths do not match
        """
        data = self._update_filter_none(**data)
        if not data:
            return
        current = set(self.cds.column_names)
        new = set(data.keys())
        if new > current:
            # This is a choice to prevent unexpected bugs
            # Could have an .add method which doesn't do this check
            # CDS already has an add method for this
            raise KeyError('Cannot add columns using .update(), '
                           f'current keys = {current}, '
                           f'new keys = {new}.')
        if new == current:
            # Replacing all columns, check new data lengths are consistent
            matching = unique_length(*data.values())
        else:
            # Replacing only some columns, check matching lengths
            matching = unique_length(*data.values(), *self.cds.data.values())
        if not matching:
            raise ValueError('Mismatching column lengths')
        self.cds.data.update(data)
        return self

    def update(self, **data: np.ndarray | list):
        return self.raw_update(**data)

    def clear(self) -> Self:
        """
        Can bypass custom .update method as we keep the same keys

        If clearing should be prevented, raise an exception in the subclass
        """
        empty = {k: [] for k in self.cds.data.keys()}
        DisplayBase.update(self, **empty)
        return self

    def is_on(self) -> tuple[BkFigure, ...]:
        figs = []
        for wrappers in self._glyphs.values():
            figs.extend(o.fig for wrapper in wrappers for o in wrapper.on)
        for children in self._children.values():
            for child in children:
                figs.extend(child.is_on())
        return tuple(set(figs))

    def _renderers_for_fig(self, glyph_name: str, fig: BkFigure):
        for wrappers in self._glyphs[glyph_name]:
            for glyph_on in wrappers.on:
                if glyph_on.fig is fig:
                    yield glyph_on.renderer

    def renderers_for_fig(self, glyph_name: str, fig: BkFigure) -> tuple[GlyphRenderer, ...]:
        return tuple(self._renderers_for_fig(glyph_name, fig))

    @property
    def glyph_names(self):
        return tuple(self._glyphs.keys())

    def editable(self, *figs: BkFigure) -> Self:
        raise NotImplementedError

    def tools(self, glyph_name: str, *figs: tuple[BkFigure, ...]) -> dict[BkFigure, list[Tool]]:
        if len(figs) == 0:
            figs = self.is_on()
        if len(figs) == 0:
            return {}
        tools = {}
        for fig in figs:
            tools[fig] = []
            renderers = self.renderers_for_fig(glyph_name, fig)
            for tool in fig.tools:
                if not hasattr(tool, 'renderers') or isinstance(tool.renderers, str):
                    continue
                for renderer in renderers:
                    try:
                        if renderer in tool.renderers:
                            tools[fig].append(tool)
                    except TypeError:
                        continue
        return tools

    def _add_to_tool(
        self,
        *,
        figs: tuple[BkFigure, ...],
        glyph_name: str,
        tool_filter: Callable[[Tool], bool],
        make_tool: Callable[[], EditTool],
        selected: bool = False,
    ):
        all_figs = self.is_on()
        if figs and not all(f in all_figs for f in figs):
            raise ValueError('Cannot make DiplayBase editable on a '
                             'figure before adding it to that figure')
        elif not figs:
            if not all_figs:
                raise ValueError('Cannot make DiplayBase editable before adding to figures')
            figs = all_figs

        where: list[tuple[BkFigure, EditTool]] = []
        for fig in figs:
            matching_tools = [
                t for t in fig.tools
                if tool_filter(t)
            ]
            try:
                tool = matching_tools[0]
            except IndexError:
                tool = make_tool()
                fig.add_tools(tool)
            if selected:
                # FIXME this should determine the correct tool category to activate
                fig.toolbar.active_multi = tool
            renderers = self.renderers_for_fig(glyph_name, fig)
            for renderer in renderers:
                tool.renderers.append(renderer)
            where.append((fig, tool))

        return where


T = TypeVar('T', bound='DisplayBase')


class ConsBase(abc.ABC):
    default_keys = tuple()

    @classmethod
    def empty(cls, constructs: type[T]) -> T:
        """
        Need to figure out how to give the return Type dynamically

        Self type from py3.11 or typing_extensions
        https://stackoverflow.com/a/75337086
        https://realpython.com/python-type-self/
        but these constructors do not return self...
        """
        data = {
            k: [] for k in cls.default_keys
        }
        cds = ColumnDataSource(data)
        return constructs(cds)


class PointSet(DisplayBase):
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


class PointSetCons(ConsBase):
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

    @classmethod
    def empty(cls):
        return super().empty(PointSet)


class DiskSet(DisplayBase):
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


class DiskSetCons(ConsBase):
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

    @classmethod
    def empty(cls):
        return super().empty(DiskSet)


class RingSet(DisplayBase):
    glyph_map = {
        'rings': Circle,
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


class RingSetCons(ConsBase):
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

    @classmethod
    def empty(cls):
        return super().empty(RingSet)


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


class Curve(DisplayBase):
    glyph_map = {
        'curve': Line,
    }

    def __init__(
        self,
        cds: ColumnDataSource,
        xkey: str = 'xvals',
        ykey: str = 'yvals',
    ):
        super().__init__(cds)
        glyph = Line(
            x=xkey,
            y=ykey,
        )
        self._register_glyph('curve', glyph)

    @classmethod
    def new(cls):
        return CurveCons()

    @property
    def glyph(self) -> Line:
        return self._glyphs['curve'][0].glyph

    def update(
        self,
        xvals: np.ndarray | None = None,
        yvals: np.ndarray | None = None,
    ):
        data = {}
        data[self.glyph.x] = xvals
        data[self.glyph.y] = yvals
        # Not sure where this column comes from ?
        if 'index' in self.cds.column_names:
            data['index'] = np.ones(len(xvals), dtype=int)
        return super().update(**data)


class CurveCons(ConsBase):
    default_keys = ('xvals', 'yvals')

    @staticmethod
    def from_vectors(
        xvals: np.ndarray,
        yvals: np.ndarray,
    ) -> Curve:
        return CurveCons.from_array(
            np.stack((xvals, yvals), axis=1)
        )

    @staticmethod
    def from_array(
        array: np.ndarray,
    ) -> Curve:
        df = pd.DataFrame(
            array,
            columns=CurveCons.default_keys
        )
        return CurveCons.from_dataframe(
            df,
            xkey=df.columns[0],
            ykey=df.columns[1],
        )

    @staticmethod
    def from_dataframe(
        df: pd.DataFrame,
        xkey: str = 'xvals',
        ykey: str = 'yvals',
    ):
        cds = ColumnDataSource(df)
        return Curve(
            cds,
            xkey=xkey,
            ykey=ykey,
        )

    @classmethod
    def empty(cls):
        return super().empty(Curve)


class MultiCurve(DisplayBase):
    glyph_map = {
        'curves': [Curve],
    }
    color_cycle = 'b_glasbey_category10'

    def __init__(
        self,
        cds: ColumnDataSource,
        xkey: str = 'xvals',
        ykeys: Sequence[str] | None = None,
    ):
        super().__init__(cds)
        self._children: dict[str, list[Curve]]
        self._xkey = xkey
        self._array_col_labels = None

        if ykeys is None:
            ykeys = tuple(c for c in cds.column_names if c != xkey)
        for ykey in ykeys:
            child = Curve(cds, xkey=xkey, ykey=ykey)
            self._register_child('curves', child)
            child.glyph.line_color = self._next_color()

    @classmethod
    def new(cls):
        return MultiCurveCons

    def _next_color(self):
        try:
            return next(self._ccycle)
        except AttributeError:
            self._ccycle = itertools.cycle(getattr(cc, self.color_cycle))
            return self._next_color()

    @property
    def curves(self):
        return tuple(c for c in self._children['curves'])

    def update(
        self,
        *,
        array: pd.DataFrame | np.ndarray | None = None,
        xvals: np.ndarray | None = None,
        **yvals: np.ndarray,
    ):
        if isinstance(array, np.ndarray):
            assert not yvals and xvals is None
            assert self._array_col_labels is not None
            assert array.ndim == 2 and array.shape[1] == len(self._array_col_labels)
            data = self._data_from_array(array)
        elif isinstance(array, pd.DataFrame):
            data = array.to_dict(orient='list')
        else:
            data = {**yvals}
            if xvals is not None:
                data[self._xkey] = xvals
        return super().update(**data)

    def _set_array_col_labels(self, col_labels):
        self._array_col_labels = col_labels

    def _data_from_array(
        self,
        array: np.ndarray,
    ):
        return MultiCurveCons.array_to_dict(array, self._array_col_labels)


class MultiCurveCons(ConsBase):
    default_xkey = 'xvals'

    @staticmethod
    def array_to_dict(array, labels):
        return {
            l: array[:, li] for li, l in enumerate(labels)
        }

    @staticmethod
    def default_ykey(idx: int) -> str:
        return f'y_{idx}'

    @staticmethod
    def from_array(
        array: np.ndarray,
        xcol: int = 0
    ) -> MultiCurve:
        _, ncols = array.shape
        col_labels = list(MultiCurveCons.default_ykey(i) for i in range(ncols - 1))
        col_labels.insert(xcol, MultiCurveCons.default_xkey)
        array_dict = MultiCurveCons.array_to_dict(array, col_labels)
        cds = ColumnDataSource(array_dict)
        multi_c = MultiCurve(
            cds,
            xkey=MultiCurveCons.default_xkey,
        )
        multi_c._set_array_col_labels(col_labels)
        return multi_c

    @staticmethod
    def from_dataframe(
        df: pd.DataFrame,
        xkey: str,
        ykeys: Sequence[str] | None = None,
    ) -> MultiCurve:
        cds = ColumnDataSource(df)
        return MultiCurve(
            cds,
            xkey=xkey,
            ykeys=ykeys,
        )

    @classmethod
    def empty(cls):
        return super().empty(MultiCurve)


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


class Text(DisplayBase):
    glyph_map = {
        'text': [BkText],
    }

    def __init__(
        self,
        cds: ColumnDataSource,
        x='x',
        y='y',
        text='text',
    ):
        super().__init__(cds)
        glyph = BkText(
            x=x,
            y=y,
            text=text,
        )
        self._register_glyph('text', glyph)

    @property
    def glyph(self) -> BkText:
        return self._glyphs['text'][0].glyph

    @classmethod
    def new(cls):
        return TextCons()

    def update(
        self,
        x: np.ndarray | None = None,
        y: np.ndarray | None = None,
        text: str | list[str] | None = None,
    ):
        data = {}
        data[self.glyph.x] = x
        data[self.glyph.y] = y
        if isinstance(text, str):
            text = [text] * self.data_length
        data[self.glyph.text] = text
        return super().update(**data)


class TextCons(ConsBase):
    default_keys = ('x', 'y', 'text')

    @classmethod
    def from_vectors(
        cls,
        x: np.ndarray,
        y: np.ndarray,
        text: list[str] | str,
    ):
        if isinstance(text, str):
            text = [text] * len(x)
        assert len(text) == len(x) == len(y)
        data = {
            k: v for k, v in zip(cls.default_keys, (x, y, text))
        }
        cds = ColumnDataSource(data)
        return Text(cds)

    @classmethod
    def empty(cls):
        return super().empty(Text)
