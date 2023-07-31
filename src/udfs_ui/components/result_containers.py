from __future__ import annotations
from typing import Any
import pathlib

import panel as pn
import numpy as np
from bokeh.plotting import figure

from ..display.image_db import BokehImage
from .live_plot import adapt_figure


class ResultContainer:
    def __init__(
        self,
        name: str,
        data: Any,
        meta: dict[str, Any] | None = None,
        title: str | None = None
    ):

        self._name = name
        self._data = data
        if meta is None:
            meta = {'tags': []}
        elif 'tags' in meta:
            assert isinstance(meta['tags'], list)
        else:
            meta['tags'] = []
        self._meta = meta
        self._title = title

    @property
    def name(self):
        return self._name

    @property
    def data(self):
        return self._data

    @property
    def meta(self):
        return self._meta
    
    @property
    def tags(self):
        return self.meta['tags']
    
    def tag_as(self, *tags: str):
        self.meta['tags'].extend(tags)

    @property
    def title(self):
        if self._title is None:
            return self.name
        return self._title

    def table_repr(self) -> str:
        return f'{type(self.data)}'

    def show(self, standalone: bool = True) -> pn.layout.ListPanel:
        return pn.Column(pn.widgets.StaticText(value=self.table_repr()))


class ScalarResultContainer(ResultContainer):
    def table_repr(self):
        return f'{type(self.data).__name__} : {self.data}'


class NumpyResultContainer(ResultContainer):
    def __init__(
        self,
        name: str,
        data: np.ndarray,
        meta: dict[str, Any] | None = None,
        title: str | None = None
    ):
        super().__init__(name, data, meta=meta, title=title)
        assert isinstance(self.data, np.ndarray)

    @property
    def data(self) -> np.ndarray:
        return super().data

    def table_repr(self):
        return f'np.ndarray : shape {self.data.shape} : dtype {self.data.dtype}'


class Numpy2DResultContainer(NumpyResultContainer):
    def __init__(
        self,
        name: str,
        data: np.ndarray,
        meta: dict[str, Any] | None = None,
        title: str | None = None
    ):
        super().__init__(name, data, meta=meta, title=title)
        assert self.data.ndim == 2

    def show(self, standalone: bool = True):
        fig = figure()
        im = BokehImage.new().from_numpy(self.data)
        im.on(fig)
        adapt_figure(fig, im, self.data.shape, 20, 400)
        fig.title.text = self.title
        return pn.Column(pn.pane.Bokeh(fig))


class RecordResultContainer(ResultContainer):
    @property
    def filepath(self):
        return pathlib.Path(self.data).resolve()

    def table_repr(self) -> str:
        return 'Saved dataset'

    def show(self, standalone: bool = True):
        return pn.Column(
            pn.pane.Markdown(
                object=f'''**Saved dataset at:** {self.filepath}:

Load with:
```
import libertem.api as lt
ctx = lt.Context.make_with("inline")
ctx.load("npy", R"{self.filepath}")
```''',
                min_width=700,
            )
        )
