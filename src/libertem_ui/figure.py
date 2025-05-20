from __future__ import annotations
from typing import Callable, Sequence
from typing_extensions import Literal
from functools import partial
import uuid

import numpy as np
import panel as pn
from bokeh.plotting import figure
from bokeh.models import CustomJS
from bokeh.models.tools import CustomAction, WheelZoomTool
from bokeh.models.annotations import Title
from bokeh.events import MouseMove, MouseLeave

from .display.image_db import BokehImage
from .display.display_base import Rectangles, DisplayBase, Polygons
from .display.icons import options_icon, sigma_icon


pn.extension(
    'floatpanel',
)


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


def set_frame_height(fig, shape, maxdim=450, mindim=-1):
    h, w = shape
    if h > w:
        fh = maxdim
        fw = max(mindim, int((w / h) * maxdim))
    else:
        fw = maxdim
        fh = max(mindim, int((h / w) * maxdim))
    fig.frame_height = fh
    fig.frame_width = fw
    return fh, fw


def adapt_figure(fig: figure, shape, maxdim: int | None = 450, mindim: int | None = None):
    if mindim is None:
        # Don't change aspect ratio in this case
        mindim = -1

    fig.y_range.flipped = True
    fh, fw = set_frame_height(fig, shape, maxdim, mindim)

    if fh > 0.8 * fw:
        location = 'right'
    else:
        location = 'below'
    fig.toolbar_location = location

    fig.x_range.range_padding = 0.
    fig.y_range.range_padding = 0.
    fig.toolbar.active_drag = None
    fig.background_fill_alpha = 0.
    fig.border_fill_color = None
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
        pn.io.notebook.push_notebook(
            self.pane,
            *(o.pane if hasattr(o, "pane") else o for o in others)
        )


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
        self._floatpanels: dict[str, dict] = {}

        self._base_title = ""
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
        channel_dimension: int | tuple[int, ...] | list[str] = -1,
        tools: bool = True,
    ):
        plot = cls()
        plot._base_title = title
        plot._channel_data, plot._channel_map, image = (
            plot._setup_multichannel(data, dim=channel_dimension)
        )
        # Bokeh needs a string title, however, so gets the default ''
        fig = figure(title=title)
        im = BokehImage.new().from_numpy(image).on(fig)
        if downsampling:
            im.enable_downsampling()
        plot.set_plot(fig=fig, im=im)
        adapt_figure(fig, image.shape, maxdim=maxdim, mindim=mindim)
        if tools:
            plot._setup()
        if plot.is_multichannel:
            plot.get_channel_select()
        return plot

    def _setup(self):
        self.add_hover_position_text()
        self.add_control_panel(self.im)
        self.add_autorange(self.im)
        self.add_complex_select(self.im)
        self.im.color.add_colorbar()

    @property
    def is_multichannel(self):
        return self._channel_map is not None

    @staticmethod
    def _setup_multichannel(
        data: PlotDataT,
        dim: int | tuple[int, ...] | tuple[str, ...] = -1,
    ):
        if callable(data):
            try:
                first_element = dim[0]
            except TypeError:
                raise TypeError("With callable data, channel_dimension must be a sequence")
            out_data, _ = data(first_element)
            channel_map = dim
            channel_data = data
        elif isinstance(data, np.ndarray):
            if data.ndim == 2:
                return None, None, data
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
                channel_map = dim
                channel_data = data
                slices = tuple(
                    slice(None) if i not in channel_map else 0
                    for i in range(channel_data.ndim)
                )
                out_data = channel_data[slices]
            else:
                raise ValueError('No support for 0/1-D images')

        elif isinstance(data, dict):
            assert all(isinstance(k, str) for k in data.keys())
            assert all(isinstance(v, np.ndarray) and v.ndim == 2 for v in data.values())
            channel_map = 'DICT'
            channel_data = data
            out_data = channel_data[tuple(channel_data.keys())[0]]
        elif isinstance(data, (tuple, list)):
            assert all(isinstance(v, np.ndarray) and v.ndim == 2 for v in data)
            channel_data = data
            channel_map = 'SEQUENCE'
            out_data = data[0]
        else:
            raise ValueError('Unrecognized data format for multichannel')
        return channel_data, channel_map, out_data

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

    def update(self, data: PlotDataT, push_update: bool = True):
        channel_data, channel_map, _ = self._setup_multichannel(data, self._channel_map)
        if channel_map is not None or self.is_multichannel:
            assert self._channel_map == channel_map, "Cannot change multichannel mode via update"
            assert type(channel_data) is type(self._channel_data), (
                f"Cannot change multichannel data from {type(self._channel_data)} "
                f"to {type(channel_data)} via update"
            )
            if callable(channel_data):
                pass  # cannot validate a callable
            elif isinstance(channel_data, np.ndarray):
                # There is no intrinsic reason we cannot change the shape in an update
                # it would just require some clever validation and a widget update
                assert channel_data.shape == self._channel_data.shape, (
                    f"Mismatching stack shapes during update, {self._channel_data.shape} "
                    f"to {channel_data.shape}"
                )
            elif isinstance(channel_data, dict):
                # This could also be updated - would just need to update the widget too
                # and handle the case of the current slice / key no longer existing
                assert tuple(channel_data.keys()) == tuple(self._channel_data.keys()), (
                    "Mismatching data keys during update"
                )
            else:
                # Same as above, a non-matching length update is possible
                assert len(channel_data) == len(self._channel_data), (
                    f"Mismatching data sequence length during update, {len(self._channel_data)} "
                    f"to {len(channel_data)}"
                )
            self._channel_data = channel_data
            self.change_channel(None, push_update=push_update)
        else:
            self.im.update(data)
            if push_update:
                self.push()

    def push(self, *others: ApertureFigure | BokehFigure):
        pn.io.notebook.push_notebook(
            self.pane,
            *(o.pane if hasattr(o, "pane") else o for o in others)
        )

    def add_control_panel(
        self,
        im: BokehImage,
        name: str = 'Image Controls',
    ):
        cmap_slider = im.color.get_cmap_slider()
        items = [
            pn.Row(
                im.color.get_cmap_select(width=150),
                im.color.get_cmap_invert(
                    align='center',
                    margin=(25, 5, 5, 5),
                ),
            ),
            pn.Row(
                im.color._cbar_freeze,
                im.color.center_cmap_toggle,
            ),
            pn.Row(
                im.color._full_scale_btn,
                im.color.clip_outliers_btn,
                im.color.clip_outliers_sigma_spinner,
            ),
            pn.Row(
                *im.color.minmax_input
            ),
            cmap_slider,
            im.color._clim_slider_symmetric
        ]
        im.color._full_scale_btn.height = 35
        im.color.clip_outliers_btn.margin = (5, 0, 5, 5)
        im.color.clip_outliers_btn.height = 35
        im.color.clip_outliers_sigma_spinner.margin = (5, 5, 5, 0)
        im.color.clip_outliers_sigma_spinner.height = 35

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

        index = len(self._floatpanels)
        floatpanel_spec = dict(
            title=name,
            items=items,
            index=index,
        )
        self._floatpanels[button_uuid] = floatpanel_spec

        floatpanel = self._make_floatpanel(**floatpanel_spec, status='closed')
        self._outer_toolbar.insert(index, floatpanel)
        self._outer_toolbar.append(open_btn)

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

    def add_autorange(self, im: BokehFigure):
        autorange_action = CustomAction(
            icon=sigma_icon(),
            callback=im.color._clip_outliers_btn.js_event_callbacks['button_click'][0],
            description='Autorange color',
        )
        self.fig.add_tools(autorange_action)

    @staticmethod
    def _make_floatpanel(items, title, status='normalized', **kwargs):
        return pn.layout.FloatPanel(
            *items,
            name=title,
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

    def _open_controls(self, event):
        floatpanel_spec = self._floatpanels[event.obj.tags[0]]
        self._outer_toolbar[floatpanel_spec["index"]] = self._make_floatpanel(
            floatpanel_spec['items'],
            floatpanel_spec['title'],
        )

    def add_complex_select(self, im: BokehImage, label: str | None = "View"):
        if im.complex_manager is not None:
            if label is not None:
                self._toolbar.append(
                    pn.widgets.StaticText(
                        value=label,
                        align='center',
                        margin=(5, 5),
                    )
                )
            select = im.complex_manager.get_complex_select()
            self._toolbar.append(
                select
            )
            return select
        return None

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
            if len(self._base_title) > 0:
                title = f"{self._base_title} - {title}"
            self.fig.title.text = title
        if push_update:
            self.push()
