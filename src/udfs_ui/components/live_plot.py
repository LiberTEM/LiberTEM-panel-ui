from __future__ import annotations
from typing import TYPE_CHECKING, Callable
import time
from functools import partial
import uuid

import numpy as np
import panel as pn
from bokeh.plotting import figure
from bokeh.models import CustomJS
from bokeh.models.tools import CustomAction, WheelZoomTool
from bokeh.models.annotations import Title
from bokeh.models.widgets import Button
from bokeh.events import MouseMove, MouseLeave
from libertem.viz.base import Live2DPlot

from ..display.image_db import BokehImage
from ..display.display_base import Rectangles, DisplayBase, Polygons
from ..display.icons import options_icon, options_icon_blue

pn.extension('floatpanel')

if TYPE_CHECKING:
    from .results import ResultRow
    from libertem.udf.base import UDFResultDict


def adapt_figure(fig: figure, shape, maxdim: int | None = 450, mindim: int | None = None):
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

    if fh > 0.8 * fw:
        location = 'right'
    else:
        location = 'below'
    fig.toolbar_location = location

    fig.x_range.range_padding = 0.
    fig.y_range.range_padding = 0.
    fig.toolbar.active_drag = None
    zoom_tools = tuple(t for t in fig.tools if isinstance(t, WheelZoomTool))
    try:
        fig.toolbar.active_scroll = zoom_tools[0]
    except IndexError:
        pass


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
        self._last_res: tuple[UDFResultDict, np.ndarray | bool] | None = None

    @property
    def pane(self) -> pn.pane.Bokeh | None:
        return self._pane

    @property
    def fig(self) -> figure | None:
        return self._fig

    @property
    def im(self) -> BokehImage | None:
        return self._im

    @property
    def layout(self):
        return self.pane

    def _setup(self):
        # Do any custom setup after fig/im are created and set
        pass

    def set_plot(
        self,
        *,
        plot: AperturePlot | None = None,
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
        plot._setup()
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
        self._channel_map: dict[str, str | tuple[str, Callable] | Callable] | None = None
        self._channel_select = None
        if isinstance(channel, dict):
            self._channel_map = channel
            channel_names = tuple(self._channel_map.keys())
            # can be {name: str | tuple[str, Callable[buffer]] | Callable(udf_result, damage)}
            # following the base class behaviour
            # name does not have to correspond to UDF buffer names
            channel = self._channel_map[channel_names[0]]
        super().__init__(
            dataset, udf,
            roi=roi, channel=channel,
            title=title, min_delta=min_delta,
            udfresult=udfresult
        )
        self._mask_elements: list[DisplayBase] = []
        self._displayed = None
        self._toolbar = pn.Row(
            height=40,
            margin=(0, 0)
        )
        self._layout = pn.Column(
            self._toolbar,
            margin=(0, 0),
        )

        self._clear_btn: pn.widgets.Button | None = None
        self._floatpanel: pn.layout.FloatPanel | None = None

    @property
    def displayed(self) -> ResultRow | float | None:
        return self._displayed

    @displayed.setter
    def displayed(self, val: ResultRow | float):
        self._displayed = val

    @property
    def layout(self) -> pn.Column:
        return self._layout

    def set_plot(
        self,
        *,
        plot: AperturePlot | None = None,
        fig: figure | None = None,
        im: BokehImage | None = None
    ):
        super().set_plot(
            plot=plot,
            fig=fig,
            im=im,
        )
        if plot is None:
            self._layout.append(self.pane)

    def update(self, damage, force=False, push_nb: bool = True):
        self.displayed = time.monotonic()
        return super().update(damage, force=force, push_nb=push_nb)

    def _setup(self):
        self.add_hover_position_text()
        self.add_control_panel()

    def add_mask_tools(
        self,
        rectangles: bool = True,
        polygons: bool = True,
        activate: bool = False,
        clear_btn: bool = True,
    ):
        if polygons:
            self._mask_elements.append(
                Polygons
                .new()
                .empty()
                .on(self.fig)
                .editable()
            )
        if rectangles:
            self._mask_elements.append(
                Rectangles
                .new()
                .empty()
                .on(self.fig)
                .editable()
            )
        if activate and len(self.fig.tools):
            self.fig.toolbar.active_drag = self.fig.tools[-1]
        if clear_btn:
            self.get_clear_mask_btn()
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

    def get_clear_mask_btn(self):
        if self._clear_btn is None:
            self._clear_btn = pn.widgets.Button(
                name='Clear ROI',
                button_type='default',
                width=100,
            )
            self._clear_btn.on_click(self.clear_mask)
            self._toolbar.insert(0, self._clear_btn)
        return self._clear_btn

    def add_control_panel(
        self,
        name: str = 'Image Controls',
    ):
        initial_vis = False

        button_uuid = str(uuid.uuid4())
        open_btn = pn.widgets.Toggle(
            name=name,
            value=initial_vis,
            margin=(5, 5, 5, 5),
            tags=[button_uuid],
            visible=False,
            width=2,
            height=2,
        )
        close_btn = Button(
            label='✖',
            button_type='light',
        )

        self._floatpanel = pn.layout.FloatPanel(
            close_btn,
            self.im.color.get_cmap_select(),
            self.im.color.get_cmap_slider(),
            self.im.color._cbar_freeze,
            self.im.color._full_scale_btn,
            self.im.color._gamma_slider,
            self.im.color.get_cmap_invert(),
            name=name,
            config={
                "headerControls": {
                    "maximize": "remove",
                    "normalize": "remove",
                    "minimize": "remove",
                    # Really needs a close-type button !
                    # even if self-implemented
                    "close": "remove",
                },
            },
            margin=20,
            visible=initial_vis,
        )
        open_btn.jslink(self._floatpanel, value='visible')

        cb = CustomJS(
            args={
                'btn_uuid': button_uuid,
                'icon_active': options_icon_blue(as_b64=True),
                'icon_inactive': options_icon(as_b64=True),
            },
            code='''
// searching through *all* models is really a hack...
for (let model of this.document._all_models.values()){
    if (model.properties.tags._value.includes(btn_uuid)){
        model.active = !model.active
        if (model.active){
            cb_obj.properties.icon.set_value(icon_active)
        } else {
            cb_obj.properties.icon.set_value(icon_inactive)
        }
        cb_obj.properties.icon.change.emit()
        cb_obj.change.emit()
        return
    }
}
''')

        action = CustomAction(
            icon=options_icon(),
            callback=cb,
            description='Open plot toolbox',
        )
        self.fig.add_tools(action)

        close_btn.js_on_click(
            CustomJS(
                args={
                    'action': action,
                },
                code='''
action.callback.execute(action)
'''
            )
        )

        self._toolbar.extend((
            open_btn,
            self._floatpanel,
        ))

    def get_float_panel(self):
        return self._floatpanel

    def get_channel_select(
        self,
        selected: str | None = None,
        label: str = 'Display channel',
        update_title: bool = True,
        embed: bool = True,
    ) -> pn.widgets.Select:
        if self._channel_select is not None:
            return self._channel_select
        elif self._channel_map is None:
            raise RuntimeError('Cannot select channels if a channel map '
                               'was not provided')
        channel_names = list(self._channel_map.keys())
        if selected is None:
            selected = channel_names[0]
        self.change_channel(selected, push_update=False)
        display_text = pn.widgets.StaticText(
            value=label,
            align='center',
            margin=(5, 5),
        )
        self._channel_select = pn.widgets.Select(
            options=channel_names,
            value=selected,
            width=200,
            margin=(5, 5),
        )
        self._channel_select.param.watch(
            partial(self._switch_channel_cb, update_title=update_title),
            'value',
        )
        if embed:
            self._toolbar.insert(0, self._channel_select)
            self._toolbar.insert(0, display_text)
        else:
            self._channel_select.name = label
        return self._channel_select

    def _switch_channel_cb(self, e, update_title=True):
        return self.change_channel(e.new, update_title=update_title)

    def change_channel(
        self,
        channel_name: str,
        push_update: bool = True,
        update_title: bool = True,
    ):
        try:
            channel = self._channel_map[channel_name]
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
        if update_title:
            self.fig.title.text = channel_name
        # Will cause a double update if called during UDF run
        if push_update and self._last_res is not None:
            self.new_data(*self._last_res, manual=push_update)

    def add_hover_position_text(self):
        title = Title(
            text=' ',
            align='left',
            syncable=False,
        )
        self.fig.add_layout(
            title,
            place="below",
        )
        cb_code = 'pos_title.text = `[x: ${(cb_obj.x).toFixed(1)}, y: ${(cb_obj.y).toFixed(1)}]`'
        self.fig.js_on_event(
            MouseMove,
            CustomJS(
                args={
                    'pos_title': title,
                },
                code=cb_code,
            ),
        )
        self.fig.js_on_event(
            MouseLeave,
            CustomJS(
                args={
                    'pos_title': title,
                },
                code='pos_title.text = " "',
            ),
        )
