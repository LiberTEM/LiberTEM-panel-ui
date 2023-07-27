from __future__ import annotations
from typing import TYPE_CHECKING
import time
import numpy as np
import panel as pn
from bokeh.plotting import figure
from libertem.viz.base import Live2DPlot

from ..display.image_db import BokehImage
from ..display.display_base import Rectangles, DisplayBase, Polygons

if TYPE_CHECKING:
    from .results import ResultRow

def adapt_figure(fig, im, shape, mindim, maxdim):
    fig.y_range.flipped = True

    h, w = shape
    if h > w:
        fh = maxdim
        fw = max(mindim, int((w / h) * maxdim))
    else:
        fw = maxdim
        fh = max(mindim, int((h / w) * maxdim))
    fig.frame_height = fh
    fig.frame_width = fw

    if fh > 1.2 * fw:
        location = 'right'
    else:
        location = 'below'
    fig.toolbar_location = location


class AperturePlotBase(Live2DPlot):
    def __init__(
            self, dataset, udf, roi=None, channel=None, title=None, min_delta=0.25, udfresult=None
    ):
        super().__init__(
            dataset, udf,
            roi=roi, channel=channel,
            title=title, min_delta=min_delta,
            udfresult=udfresult
        )
        self._pane: pn.pane.Bokeh | None = None
        self._fig: figure | None = None
        self._im: BokehImage | None = None

    @property
    def pane(self) -> pn.pane.Bokeh | None:
        return self._pane

    @property
    def fig(self) -> figure | None:
        return self._fig

    @property
    def im(self) -> BokehImage | None:
        return self._im

    def set_plot(
        self,
        *,
        plot: 'AperturePlot' | None = None,
        fig: figure | None = None,
        im: BokehImage | None = None
    ):
        if plot is not None:
            self._pane = plot.pane
            fig = plot.fig
            im = plot.im
        else:
            self._pane = pn.pane.Bokeh(fig)
        self._fig = fig
        self._im = im

    @classmethod
    def new(
        cls,
        dataset,
        udf,
        *,
        maxdim=400,
        mindim=20,
        roi=None,
        channel=None,
        title=''
    ):
        # Live plot gets a None title if not specified so it keeps its default
        plot = cls(dataset, udf, roi=roi, channel=channel, title=title if len(title) else None)
        # Bokeh needs a string title, however, so gets the default ''
        fig = figure(title=title)
        im = BokehImage.new().from_numpy(plot.data)
        im.on(fig)
        plot.set_plot(fig=fig, im=im)
        adapt_figure(fig, im, plot.data.shape, mindim, maxdim)
        return plot

    def update(self, damage, force=False, push_nb: bool = True):
        if self.fig is None:
            raise RuntimeError('Cannot update plot before set_plot called')
        self.im.update(self.data)
        if push_nb:
            pn.io.push_notebook(self.pane)

    def display(self):
        if self.fig is None:
            raise RuntimeError('Cannot display plot before set_plot called')
        return self.fig


class AperturePlot(AperturePlotBase):
    def __init__(
            self, dataset, udf, roi=None, channel=None, title=None, min_delta=0.25, udfresult=None
    ):
        super().__init__(
            dataset, udf,
            roi=roi, channel=channel,
            title=title, min_delta=min_delta,
            udfresult=udfresult
        )
        self._mask_elements: list[DisplayBase] = []
        self._displayed = None

    @property
    def displayed(self) -> ResultRow | float | None:
        return self._displayed

    @displayed.setter
    def displayed(self, val: ResultRow | float):
        self._displayed = val

    def update(self, damage, force=False, push_nb: bool = True):
        self.displayed = time.monotonic()
        return super().update(damage, force=force, push_nb=push_nb)

    def add_mask_tools(
        self,
        rectangles: bool = True,
        polygons: bool = True,
        activate: bool = True,
    ):
        if polygons:
            self._mask_elements.append(
                Polygons
                .new()
                .empty()
                .on(self.fig)
                .make_editable()
            )
        if rectangles:
            self._mask_elements.append(
                Rectangles
                .new()
                .empty()
                .on(self.fig)
                .make_editable()
            )
        if activate and len(self.fig.tools):
            self.fig.toolbar.active_drag = self.fig.tools[-1]
        return self

    def get_mask(self, shape: tuple[int, int]) -> np.ndarray | None:
        mask = None
        for element in self._mask_elements:
            try:
                _mask = element.as_mask(shape)
            except (AttributeError, NotImplementedError):
                continue
            if _mask is not None:
                if mask is None:
                    mask = _mask
                else:
                    mask = np.logical_or(mask, _mask)
        return mask

    def clear_mask(self, *e):
        for element in self._mask_elements:
            element.clear()
        pn.io.push_notebook(self.pane)

    def get_clear_mask_btn(self, label='Clear ROI', width=100, button_type='default'):
        clear_btn = pn.widgets.Button(
            name=label,
            button_type=button_type,
            width=width,
        )
        clear_btn.on_click(self.clear_mask)
        return clear_btn
