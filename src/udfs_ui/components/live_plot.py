from __future__ import annotations
from typing import TYPE_CHECKING
import time
import numpy as np
import panel as pn
pn.extension('floatpanel')
from bokeh.plotting import figure
from libertem.viz.base import Live2DPlot

from ..display.image_db import BokehImage
from ..display.display_base import Rectangles, DisplayBase, Polygons

if TYPE_CHECKING:
    from .results import ResultRow

def adapt_figure(fig, shape, maxdim: int | None = 450, mindim: int | None = None):
    if mindim is None:
        # Don't change aspect ratio in this case
        mindim = -1

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

    fig.x_range.range_padding = 0.
    fig.y_range.range_padding = 0.


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
        self._was_rate_limited: bool = False

    @classmethod
    def new(
        cls,
        dataset,
        udf,
        *,
        maxdim: int = 450,
        # If mindim is None, plot aspect ratio is always preserved
        # even if the plot is very thin or tall
        mindim: int | None = 100,
        roi: np.ndarray | None = None,
        channel=None,
        title: str = '',
        downsampling: bool = True,
    ):
        # Live plot gets a None title if not specified so it keeps its default
        plot = cls(dataset, udf, roi=roi, channel=channel, title=title if len(title) else None)
        # Bokeh needs a string title, however, so gets the default ''
        fig = figure(title=title)
        im = BokehImage.new().from_numpy(plot.data).on(fig)
        if downsampling:
            im.enable_downsampling()
        plot.set_plot(fig=fig, im=im)
        adapt_figure(fig, plot.data.shape, maxdim=maxdim, mindim=mindim)
        return plot

    def update(self, damage, force=False, push_nb: bool = True):
        if self.fig is None:
            raise RuntimeError('Cannot update plot before set_plot called')
        if force and (not self._was_rate_limited):
            return
        self.im.update(self.data)
        self.last_update = time.time()
        if push_nb:
            pn.io.push_notebook(self.pane)

    def display(self):
        if self.fig is None:
            raise RuntimeError('Cannot display plot before set_plot called')
        return self.fig

    def new_data(self, udf_results, damage, force=False):
        """
        This method is called with the raw `udf_results` any time a new
        partition has finished processing.

        The :code:`damage` parameter is filtered to only cover finite
        values of :code:`self.data` and passed to :meth:`self.update`,
        which should then be implemented by a subclass.
        """
        t0 = time.time()
        delta = t0 - self.last_update
        rate_limited = delta < self.min_delta
        if force or not rate_limited:
            self.data, damage = self.extract(udf_results, damage)
            self.update(damage, force=force)
        else:
            self._was_rate_limited = rate_limited
            return  # don't update plot if we recently updated
        if force:
            # End of plotting, so reset this flag for the next run
            self._was_rate_limited = False


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

    def get_control_panel(
        self,
        name: str = 'Image Controls',
    ):
        initial_vis = False
        open_btn = pn.widgets.Toggle(
            name=name,
            value=initial_vis,
            margin=(5, 5, 5, 5),
        )
        floatpanel = pn.layout.FloatPanel(
            self.im.color.get_cmap_select(),
            self.im.color._cbar_freeze,
            self.im.color.get_cmap_slider(),
            self.im.color.get_cmap_invert(),
            name=name,
            config={
                "headerControls": {
                    "maximize": "remove",
                    "normalize": "remove",
                    "minimize": "remove",
                    "close": "remove",
                },
            },
            margin=20,
            visible=initial_vis,
        )
        open_btn.jslink(floatpanel, value='visible')
        return open_btn, floatpanel
