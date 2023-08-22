from __future__ import annotations
from typing import TYPE_CHECKING
from typing_extensions import Self

from bokeh.models.sources import ColumnDataSource

from .display_base import DisplayBase, ConsBase, Text, DiskSet, Curve

if TYPE_CHECKING:
    from bokeh.plotting import figure as BkFigure


class LatticeOverlay(DisplayBase):
    """
    Adds a set of circle annotations joined by a line
    with functionality to change the radius and add labels to each
    """
    glyph_map = {
        'disks': DiskSet,
        'lines': Curve,
        'labels': Text,
    }

    def __init__(self, cds: ColumnDataSource):
        super().__init__(cds)
        diskset = DiskSet(
            cds,
            x='cx',
            y='cy',
            radius='radius',
        )
        diskset.disks.line_color = 'color'
        diskset.disks.fill_color = 'color'
        self._register_child('disks', diskset)
        lines = Curve(
            cds,
            xkey=diskset.disks.x,
            ykey=diskset.disks.y,
        )
        self._register_child('lines', lines)
        lines.glyph.line_color = '#FFFFFF'
        lines.glyph.line_dash = [6, 3]
        lines.glyph.line_width = 2

    @classmethod
    def new(cls):
        return LatticeOverlayCons()

    @property
    def diskset(self) -> DiskSet:
        return self._children['disks'][0]

    def with_labels(self, centre: str = 'o', a: str = 'a', b: str = 'b') -> Self:
        self.cds.data.update({'label': [a, centre, b]})
        text = Text(
            self.cds,
            x=self.diskset.disks.x,
            y=self.diskset.disks.y,
            text='label',
        )
        text.glyph.text_color = 'color'
        text.glyph.text_baseline = 'middle'
        text.glyph.text_align = 'center'
        text.glyph.text_font = {'value': 'mono'}
        text.glyph.text_font_style = 'bold'
        self._register_child('labels', text)
        return self

    def editable(self, *figs: BkFigure) -> Self:
        self.diskset.editable(*figs, add=False)
        return self


class LatticeOverlayCons(ConsBase):
    default_keys = (
        'cx',
        'cy',
        'radius',
        'color',
        'label',
    )

    @classmethod
    def empty(cls) -> LatticeOverlay:
        """
        VectorsOverlay cannot be created empty

        raises NotImplementedError
        """
        raise NotImplementedError('LatticeOverlay cannot be instantiated empty')

    @staticmethod
    def from_lattice_vectors(
        centre: complex,
        a: complex,
        b: complex,
        radius: float,
    ) -> LatticeOverlay:
        apos = centre + a
        bpos = centre + b
        data = {}
        data['cx'] = [apos.real, centre.real, bpos.real]
        data['cy'] = [apos.imag, centre.imag, bpos.imag]
        data['radius'] = [radius] * 3
        data['color'] = ['#FF8C00', '#8FBC8F', '#00BFFF']
        cds = ColumnDataSource(data)
        return LatticeOverlay(cds)
