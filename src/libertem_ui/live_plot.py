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
from bokeh.events import MouseMove, MouseLeave
from libertem.viz.base import Live2DPlot

from .display.image_db import BokehImage
from .display.display_base import Rectangles, DisplayBase, Polygons
from .display.icons import options_icon, sigma_icon


PlotDataT = (
    np.ndarray
    | dict[str, np.ndarray]
    | Sequence[np.ndarray]
    | Callable[[int | str], tuple[np.ndarray, str]]
)
ChannelMapT = (
    Literal['SEQUENCE', 'DICT']
    | tuple[int, ...]
    | tuple[str, ...]
)

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


class BokehFigure:
    def __init__(self, title="", **kwargs):
        self._fig = figure(title=title, **kwargs)
        self._pane = pn.pane.Bokeh(self._fig)
        self._toolbar = pn.Row(
            height=40,
            margin=(0, 0)
        )
        self._layout = pn.Column(
            self._toolbar,
            self._pane,
            margin=(0, 0),
        )

    @property
    def pane(self) -> pn.pane.Bokeh:
        return self._pane

    @property
    def fig(self) -> figure:
        return self._fig

    @property
    def layout(self) -> pn.Column:
        return self._layout

    def push(self, *others: ApertureFigure | BokehFigure):
        pn.io.notebook.push_notebook(self.pane, *(o.pane for o in others))


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
        self._outer_toolbar = pn.Row(
            self._toolbar,
            height=40,
            margin=(0, 0),
        )
        self._layout = pn.Column(
            self._outer_toolbar,
            margin=(3, 3),
        )

        self._clear_btn: pn.widgets.Button | None = None

        self._channel_prefix = "Channel"
        self._channel_select: pn.widgets.Select | pn.widgets.IntSlider | None = None
        self._channel_data: PlotDataT | None = None
        self._channel_map: ChannelMapT | None = None

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
        data: PlotDataT,
        *,
        maxdim: int = 450,
        # If mindim is None, plot aspect ratio is always preserved
        # even if the plot is very thin or tall
        mindim: int | None = 100,
        title: str = '',
        downsampling: bool = True,
        channel_dimension: int | tuple[int, ...] | tuple[str, ...] = -1,
        tools: bool = True,
    ):
        plot = cls()
        image = plot._setup_multichannel(data, dim=channel_dimension)
        # Bokeh needs a string title, however, so gets the default ''
        fig = figure(title=title)
        im = BokehImage.new().from_numpy(image).on(fig)
        if downsampling:
            im.enable_downsampling()
        plot.set_plot(fig=fig, im=im)
        adapt_figure(fig, image.shape, maxdim=maxdim, mindim=mindim)
        if tools:
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
        data: PlotDataT,
        dim: int | tuple[int, ...] | tuple[str, ...] = -1,
    ):
        if callable(data):
            try:
                first_element = dim[0]
            except TypeError:
                raise TypeError("With callable data, channel_dimension must be a sequence")
            out_data, _ = data(first_element)
            self._channel_map = dim
            self._channel_data = data
        elif isinstance(data, np.ndarray):
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

    def push(self, *others: ApertureFigure | BokehFigure):
        pn.io.notebook.push_notebook(self.pane, *(o.pane for o in others))

    def add_control_panel(
        self,
        name: str = 'Image Controls',
    ):
        self._floatpanel_title = name
        self._floatpanel_items = [
            pn.Row(
                self.im.color.get_cmap_select(width=150),
                self.im.color.get_cmap_invert(
                    align='center',
                    margin=(25, 5, 5, 5),
                ),
            ),
            self.im.color.get_cmap_slider(),
            pn.Row(
                self.im.color._full_scale_btn,
                self.im.color.clip_outliers_btn,
                self.im.color.clip_outliers_sigma_spinner,
            ),
            self.im.color._cbar_freeze,
        ]

        button_uuid = str(uuid.uuid4())
        open_btn = pn.widgets.Toggle(
            name=name,
            value=False,
            margin=(5, 5, 5, 5),
            tags=[button_uuid],
            visible=False,
            width=2,
            height=2,
        )
        open_btn.param.watch(self._open_controls, 'value')
        floatpanel = self._make_floatpanel(status='closed')
        self._outer_toolbar.insert(0, open_btn)
        self._outer_toolbar.insert(0, floatpanel)

        cb = CustomJS(
            args={
                'btn_uuid': button_uuid,
            },
            code='''
// searching through *all* models is really a hack...
for (let model of this.document._all_models.values()){
    if (model.properties.tags._value.includes(btn_uuid)){
        model.active = !model.active
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

        autorange_action = CustomAction(
            icon=sigma_icon(),
            callback=self.im.color._clip_outliers_btn.js_event_callbacks['button_click'][0],
            description='Autorange color',
        )
        self.fig.add_tools(autorange_action)

    def _make_floatpanel(self, status='normalized'):
        return pn.layout.FloatPanel(
            *self._floatpanel_items,
            name=self._floatpanel_title,
            config={
                "headerControls": {
                    "maximize": "remove",
                    # "normalize": "remove",
                    "minimize": "remove",
                    "smallify": "remove",
                    # "close": "remove",
                },
            },
            contained=False,
            position='center',
            status=status,
        )

    def _open_controls(self, *e):
        self._outer_toolbar[0] = self._make_floatpanel()

    def get_float_panel(self):
        return self._outer_toolbar[0]

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
                .editable(selected=activate)
            )
        if rectangles:
            self._mask_elements.append(
                Rectangles
                .new()
                .empty()
                .on(self.fig)
                .editable(selected=activate)
            )
        # Could use custom icons to show which tools are ROI tools
        # Could add clear ROI button as a callback on the toolbar
        # if activate and len(self.fig.tools):
        #     self.fig.toolbar.active_drag = self.fig.tools[-1]
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

    def set_mask_visiblity(self, *, rectangles: bool, polygons: bool):
        visible = True
        for element in self._mask_elements:
            if isinstance(element, Rectangles):
                visible = rectangles
            if isinstance(element, Polygons):
                visible = polygons
            element.set_visible(visible)

    def get_mask_rect_as_slices(self, shape: tuple[int, int]) -> list[tuple[slice, slice]]:
        slices = []
        for element in self._mask_elements:
            if not isinstance(element, Rectangles):
                continue
            slices.extend(element.as_slices(shape))
        return slices

    def clear_mask(self, *e):
        for element in self._mask_elements:
            element.clear()
        self.push()

    def get_clear_mask_btn(self):
        if self._clear_btn is None:
            self._clear_btn = pn.widgets.Button(
                name='Clear ROI',
                button_type='default',
                width=75,
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
        elif (
            self._channel_map == 'DICT'
            or (callable(self._channel_data) and isinstance(self._channel_map[0], str))
        ):
            if callable(self._channel_data):
                channel_names = self._channel_map
            else:
                channel_names = list(self._channel_data.keys())
            if selected is None:
                selected = channel_names[0]
            self._channel_select = pn.widgets.Select(
                options=channel_names,
                value=selected,
                width=200,
                margin=(5, 5),
            )
        elif (
            self._channel_map == 'SEQUENCE'
            or (callable(self._channel_data) and isinstance(self._channel_map[0], int))
            or isinstance(self._channel_map, tuple)
        ):
            if self._channel_map == 'SEQUENCE':
                sequence_length = len(self._channel_data)
            elif callable(self._channel_data):
                sequence_length = len(self._channel_map)
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
        if callable(self._channel_data):
            data, title = self._channel_data(channel)
        elif isinstance(self._channel_data, dict):
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
