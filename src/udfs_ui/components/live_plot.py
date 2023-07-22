from __future__ import annotations
import numpy as np
import panel as pn
from bokeh.plotting import figure
from libertem.viz.base import Live2DPlot

from ..display.image_db import BokehImage
from ..display.display_base import Rectangles, DisplayBase


def adapt_figure(fig, im, shape, mindim, maxdim):
    fig.y_range.flipped = True
    # im.flip_y()

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
        location = 'above'
    fig.toolbar_location = location


class AperturePlot(Live2DPlot):
    def __init__(self, *args, min_delta=0.25, **kwargs):
        super().__init__(*args, min_delta=min_delta, **kwargs)
        self._pane: pn.pane.Bokeh | None = None
        self._fig: figure | None = None
        self._im: BokehImage | None = None
        self._mask_elements: list[DisplayBase] = []

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

    def update(self, damage, force=False):
        if self.fig is None:
            raise RuntimeError('Cannot update plot before set_plot called')
        self.im.update(self.data)
        pn.io.push_notebook(self.pane)

    def display(self):
        if self.fig is None:
            raise RuntimeError('Cannot display plot before set_plot called')
        return self.fig

    def add_mask_tools(
        self,
        rectangles: bool = True,
        activate: bool = True,
    ):
        if rectangles:
            _rectangles = Rectangles.new().empty()
            _rectangles.on(self.fig)
            _rectangles.make_editable()
            self._mask_elements.append(_rectangles)
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
