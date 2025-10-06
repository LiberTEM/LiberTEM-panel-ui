from __future__ import annotations
import numpy as np

from bokeh.models.sources import ColumnDataSource
from bokeh.models.glyphs import Text as BkText

from .base import DisplayBase, ConsBase


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
