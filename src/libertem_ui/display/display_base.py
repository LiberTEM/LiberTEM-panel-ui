from __future__ import annotations
import abc
from dataclasses import dataclass, field
import numpy as np
from typing import TYPE_CHECKING, NamedTuple, Callable, TypeVar, Generator
from typing_extensions import Self

from bokeh.models.sources import ColumnDataSource

from ..utils import pop_from_list


if TYPE_CHECKING:
    from bokeh.plotting import figure as BkFigure
    from bokeh.models.glyph import Glyph
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

    @property
    def glyph(self):
        if len(self.glyph_names) > 1:
            raise NotImplementedError(
                "Default glyph implementation not available for multi-glyph components"
            )
        elif len(self.glyph_names) == 0:
            raise TypeError("No glyphs defined for display base")
        return self._glyphs[self.glyph_names[0]][0].glyph

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
