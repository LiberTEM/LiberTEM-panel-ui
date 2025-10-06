from __future__ import annotations
import numpy as np
import itertools
import pandas as pd
from typing import Sequence
import colorcet as cc

from bokeh.models.sources import ColumnDataSource
from bokeh.models.glyphs import Line

from .base import DisplayBase, ConsBase


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
