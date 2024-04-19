from __future__ import annotations
from typing import TYPE_CHECKING, Callable, Sequence
from typing_extensions import Literal
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

from .display.image_db import BokehImage
from .display.display_base import Rectangles, DisplayBase, Polygons
from .display.icons import options_icon, options_icon_blue

# pn.extension('floatpanel')

if TYPE_CHECKING:
    from .results.results_manager import ResultRow
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


class ApertureFigure:
    def __init__(self):
        """
        A wrapper around a bokeh.plotting.figure and a BokehImage with:

            - a toolbar to hold buttons, channel select etc
            - a floating toolbox with colormap widgets
            - methods to add drawing tools which define
              a mask / ROI on the image
        """
        self._pane: pn.pane.Bokeh | None = None
        self._fig: figure | None = None
        self._im: BokehImage | None = None

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

        self._channel_prefix = "Channel"
        self._channel_select: pn.widgets.Select | pn.widgets.IntSlider | None = None
        self._channel_data: np.ndarray | dict[str, np.ndarray] | Sequence[np.ndarray] = None
        self._channel_map: Literal['SEQUENCE', 'DICT'] | tuple[int, ...] | None = None

    def set_plot(
        self,
        *,
        plot: ApertureFigure | None = None,
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
        if plot is None:
            self._layout.append(self.pane)

    @classmethod
    def new(
        cls,
        data: np.ndarray | dict[str, np.ndarray] | Sequence[np.ndarray],
        *,
        maxdim: int = 450,
        # If mindim is None, plot aspect ratio is always preserved
        # even if the plot is very thin or tall
        mindim: int | None = 100,
        title: str = '',
        downsampling: bool = True,
        channel_dimension: int | tuple[int, ...] = -1,
    ):
        plot = cls()
        data = plot._setup_multichannel(data, dim=channel_dimension)
        # Bokeh needs a string title, however, so gets the default ''
        fig = figure(title=title)
        im = BokehImage.new().from_numpy(data).on(fig)
        if downsampling:
            im.enable_downsampling()
        plot.set_plot(fig=fig, im=im)
        adapt_figure(fig, data.shape, maxdim=maxdim, mindim=mindim)
        plot._setup()
        return plot

    def _setup(self):
        self.add_hover_position_text()
        self.add_control_panel()
        if self.is_multichannel:
            self.get_channel_select()

    @property
    def is_multichannel(self):
        return self._channel_map is not None

    def _setup_multichannel(
        self,
        data: np.ndarray | dict[str, np.ndarray] | Sequence[np.ndarray],
        dim: int | tuple[int, ...] = -1,
    ):
        if isinstance(data, np.ndarray):
            if data.ndim == 2:
                return data
            elif data.ndim > 2:
                if isinstance(dim, int):
                    dim = (dim,)
                try:
                    if not all(isinstance(i, int) for i in dim):
                        raise TypeError
                    # Note the dims are normalised and sorted here
                    # Needed to ensure the slicing logic works later
                    dim = tuple(sorted(d % data.ndim for d in dim))
                    num_nav = len(dim)
                except TypeError:
                    raise TypeError(
                        'Must supply sequence of int channel dims when data.ndim > 2'
                    )
                if (data.ndim - num_nav) != 2 or (num_nav != len(set(dim))):
                    raise ValueError(
                        f"Channel dim {dim} with data shape "
                        f"{data.shape} does not yield 2D arrays"
                    )
                self._channel_map = dim
                self._channel_data = data
                slices = tuple(
                    slice(None) if i not in self._channel_map else 0
                    for i in range(self._channel_data.ndim)
                )
                out_data = self._channel_data[slices]
            else:
                raise ValueError('No support for 0/1-D images')

        elif isinstance(data, dict):
            assert all(isinstance(k, str) for k in data.keys())
            assert all(isinstance(v, np.ndarray) and v.ndim == 2 for v in data.values())
            self._channel_map = 'DICT'
            self._channel_data = data
            out_data = self._channel_data[tuple(self._channel_data.keys())[0]]
        elif isinstance(data, (tuple, list)):
            assert all(isinstance(v, np.ndarray) and v.ndim == 2 for v in data)
            self._channel_data = data
            self._channel_map = 'SEQUENCE'
            out_data = data[0]
        else:
            raise ValueError('Unrecognized data format for multichannel')
        return out_data

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
    def layout(self) -> pn.Column:
        return self._layout

    def push(self, *others: ApertureFigure):
        pn.io.notebook.push_notebook(self.pane, *(o.pane for o in others))

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
            pn.Row(
                self.im.color.get_cmap_select(width=200),
                pn.layout.HSpacer(width=50),
                close_btn,
            ),
            self.im.color.get_cmap_invert(),
            self.im.color.get_cmap_slider(),
            # self.im.color._gamma_slider,
            self.im.color._cbar_freeze,
            pn.Row(
                self.im.color._full_scale_btn,
                self.im.color.clip_outliers_btn,
                self.im.color.clip_outliers_sigma_spinner,
            ),
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
        # Could use custom icons to show which tools are ROI tools
        # Could add clear ROI button as a callback on the toolbar
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
        self.push()

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

    def get_channel_select(
        self,
        selected: str | None = None,
        label: str = 'Display channel',
        update_title: bool = True,
        embed: bool = True,
        link: bool = True,
    ) -> pn.widgets.Select:
        if self._channel_select is not None:
            return self._channel_select
        elif self._channel_map is None:
            raise RuntimeError('Cannot select channels if a channel map '
                               'was not provided')
        elif self._channel_map == 'DICT':
            channel_names = list(self._channel_data.keys())
            if selected is None:
                selected = channel_names[0]
            self._channel_select = pn.widgets.Select(
                options=channel_names,
                value=selected,
                width=200,
                margin=(5, 5),
            )
        elif self._channel_map == 'SEQUENCE' or isinstance(self._channel_map, tuple):
            if self._channel_map == 'SEQUENCE':
                sequence_length = len(self._channel_data)
            else:
                dims = tuple(self._channel_data.shape[i] for i in self._channel_map)
                sequence_length = int(np.prod(dims))
            if selected is None:
                selected = 0
            self._channel_select = pn.widgets.IntSlider(
                start=0,
                end=sequence_length - 1,
                value=selected,
                width=200,
                margin=(5, 5),
            )
        else:
            raise ValueError('Unrecognized channel map format')

        self.change_channel(selected, push_update=False)
        display_text = pn.widgets.StaticText(
            value=label,
            align='center',
            margin=(5, 5),
        )
        if link:
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

    @property
    def channel_prefix(self) -> str:
        return self._channel_prefix

    @channel_prefix.setter
    def channel_prefix(self, val: str):
        self._channel_prefix = val

    def change_channel(
        self,
        channel: str | int | None,
        push_update: bool = True,
        update_title: bool = True,
    ):
        if channel is None:
            channel = self._channel_select.value
        if isinstance(self._channel_data, dict):
            data = self._channel_data[channel]
            title = channel
        elif self._channel_map == 'SEQUENCE':
            data = self._channel_data[channel]
            title = f"{self.channel_prefix}: {channel}"
        else:
            # The following relies on having sorted, normalised, unique self._channel_map
            partial_shape = tuple(
                s for i, s in enumerate(self._channel_data.shape)
                if i in self._channel_map
            )
            partial_slice = np.unravel_index(channel, partial_shape)
            slices = tuple(
                slice(None)
                if i not in self._channel_map
                else partial_slice[self._channel_map.index(i)]
                for i in range(self._channel_data.ndim)
            )
            data = self._channel_data[slices]
            slice_as_str = ', '.join(':' if s == slice(None) else str(s) for s in slices)
            title = f"{self.channel_prefix} [{slice_as_str}]"
        self.im.update(data)
        if update_title:
            self.fig.title.text = title
        if push_update:
            self.push()


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
