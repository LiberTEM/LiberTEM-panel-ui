from __future__ import annotations

import time
from typing import TYPE_CHECKING, Callable

import numpy as np
from libertem.viz.base import Live2DPlot
from bokeh.plotting import figure

from .display.image_db import BokehImage
from .figure import ApertureFigure, adapt_figure
from .figure import BokehFigure, PlotDataT, ChannelMapT  # noqa


if TYPE_CHECKING:
    from .results.results_manager import ResultRow
    from libertem.udf.base import UDFResultDict


class AperturePlot(Live2DPlot, ApertureFigure):
    def __init__(
            self, dataset, udf, roi=None, channel=None, title=None, min_delta=0.25, udfresult=None
    ):
        """
        ApertureFigure that implements the LiberTEM Live2DPlot interface
        """
        ApertureFigure.__init__(self)
        self._last_res: tuple[UDFResultDict, np.ndarray | bool] | None = None
        self._channel_data: dict[str, str | tuple[str, Callable] | Callable] | None = None
        if isinstance(channel, dict):
            self._channel_data = channel
            self._channel_map = 'DICT'
            channel_names = tuple(self._channel_data.keys())
            # can be {name: str | tuple[str, Callable[buffer]] | Callable(udf_result, damage)}
            # following the base class behaviour
            # name does not have to correspond to UDF buffer names
            channel = self._channel_data[channel_names[0]]

        Live2DPlot.__init__(
            self,
            dataset, udf,
            roi=roi, channel=channel,
            title=title, min_delta=min_delta,
            udfresult=udfresult
        )

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
        plot.data = np.asarray(plot.data)
        # Bokeh needs a string title, however, so gets the default ''
        fig = figure(title=title)
        im = BokehImage.new().from_numpy(plot.data).on(fig)
        if downsampling:
            im.enable_downsampling()
        plot.set_plot(fig=fig, im=im)
        adapt_figure(fig, plot.data.shape, maxdim=maxdim, mindim=mindim)
        plot._setup()
        return plot

    def update(self, damage, force=False, push_nb: bool = True):
        self.displayed = time.monotonic()
        if self.fig is None:
            raise RuntimeError('Cannot update plot before set_plot called')
        if force and (not self._was_rate_limited):
            return
        self.im.update(self.data)
        self.last_update = time.time()
        if push_nb:
            self.push()

    def display(self):
        if self.fig is None:
            raise RuntimeError('Cannot display plot before set_plot called')
        return self.fig

    def new_data(
        self,
        udf_results: UDFResultDict,
        damage: np.ndarray | bool,
        force=False,
        manual=False
    ):
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
        # The manual flag is only necessary because we are
        # assuming force= is only used at the end of a UDF run
        # when this is changed then manual= can be removed
        if manual or force or not rate_limited:
            self._last_res = (udf_results, damage)
            self.data, damage = self.extract(*self._last_res)
            self.data = np.asarray(self.data)
            self.update(damage, force=force)
        else:
            self._was_rate_limited = rate_limited
            return  # don't update plot if we recently updated
        if force:
            # End of plotting, so reset this flag for the next run
            self._was_rate_limited = False

    @property
    def displayed(self) -> ResultRow | float | None:
        return self._displayed

    @displayed.setter
    def displayed(self, val: ResultRow | float):
        self._displayed = val

    def _change_channel_attrs(self, channel_name: str):
        try:
            channel = self._channel_data[channel_name]
        except (TypeError, KeyError):
            raise RuntimeError('Invalid channel_map / channel_name')

        if callable(channel):
            extract = channel
            channel = None
        elif isinstance(channel, (tuple, list)):
            channel, func = channel

            def extract(udf_results, damage):
                return (func(udf_results[channel].data), damage)
        else:
            extract = None

        self._extract = extract
        self.channel = channel

    def change_channel(
        self,
        channel_name: str,
        push_update: bool = True,
        update_title: bool = True,
    ):
        self._change_channel_attrs(channel_name)
        if update_title:
            self.fig.title.text = channel_name
        # Will cause a double update if called during UDF run
        if push_update and self._last_res is not None:
            self.new_data(*self._last_res, manual=push_update)
