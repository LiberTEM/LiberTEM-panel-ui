import functools
import itertools
from typing import Optional, TypedDict
import numpy as np
import panel as pn
from skimage.transform import AffineTransform

from ..figure import ApertureFigure
from ..display.display_base import PointSet, Cursor
from ..display.vectors import MultiLine
from ..display.image_db import BokehImage
from ..utils.colormaps import get_bokeh_palette
from .image_transformer import ImageTransformer

from bokeh.models.widgets import CheckboxGroup, Spinner
from bokeh.models import CustomJS, ColumnDataSource


def format_transform_md(transform: AffineTransform):
    transform = AffineTransform(matrix=transform.params)
    scale_x, scale_y = transform.scale
    trans_x, trans_y = transform.translation
    return f"""| Rotation | Scale   | Shear    | Translation   |
| -------- | ------- | -------- | -------       |
| {np.rad2deg(transform.rotation):.1f}  | ({scale_x:.3f}, {scale_y:.3f}) | {transform.shear:.2f}  | ({trans_x:.1f}, {trans_y:.1f}) |
"""  # noqa


def select_roi(
    image: np.ndarray,
    maxdim: int = 500,
    downsampling: bool = True,
    **fig_kwargs,
):
    fig = (
        ApertureFigure
        .new(image, maxdim=maxdim, downsampling=downsampling, **fig_kwargs)
        .add_mask_tools(
            rectangles=True,
            polygons=False,
            activate=True,
            clear_btn=False,
        )
    )
    rectangles = fig._mask_elements[0]

    def getter() -> list[tuple[slice, slice]]:
        return fig.get_mask_rect_as_slices(fig.im.array.shape)

    return fig, rectangles, getter


class PointsDict(TypedDict):
    cx: list[float]
    cy: list[float]


def select_points(
    image: np.ndarray,
    maxdim: int = 500,
    downsampling: bool = True,
    **fig_kwargs,
):

    fig = (
        ApertureFigure
        .new(image, maxdim=maxdim, downsampling=downsampling, **fig_kwargs)
    )

    pointset = (
        PointSet
        .new()
        .empty()
        .on(fig.fig)
        .editable(selected=True)
    )

    def getter() -> PointsDict:
        return pointset.cds.clone().data

    return fig, pointset, getter


class LinesDict(TypedDict):
    xs: list[list[float]]
    ys: list[list[float]]


def define_line(
    image: np.ndarray,
    maxdim: int = 500,
    downsampling: bool = True,
    **fig_kwargs,

):
    fig = (
        ApertureFigure
        .new(image, maxdim=maxdim, downsampling=downsampling, **fig_kwargs)
    )

    lineset = (
        MultiLine
        .new()
        .empty()
        .on(fig.fig)
        .editable(selected=True)
    )

    def getter() -> LinesDict:
        return lineset.cds.clone().data

    return fig, lineset, getter


# Unicode arrow codes used for defining UI buttons
LEFT_ARROW = '\u25C1'
UP_ARROW = '\u25B3'
RIGHT_ARROW = '\u25B7'
DOWN_ARROW = '\u25BD'
ROTATE_RIGHT_ARROW = '\u21B7'
ROTATE_LEFT_ARROW = '\u21B6'
SHEAR_MORE = '+'
SHEAR_LESS = '-'


def translate_buttons(cb, width: int = 40, height: int = 40, margin: tuple[int, int] = (2, 2)):
    """
    A button array for up/down/left/right
    Configured for y-axis pointing down!!
    """
    kwargs = {
        'width': width,
        'height': height,
        'margin': margin,
        'sizing_mode': 'fixed',
    }
    get_sp = lambda: pn.Spacer(**kwargs)  # noqa
    button_kwargs = {
        'button_type': 'primary',
        **kwargs,
    }
    left = pn.widgets.Button(name=LEFT_ARROW, **button_kwargs)
    left.on_click(functools.partial(cb, x=-1))
    up = pn.widgets.Button(name=UP_ARROW, **button_kwargs)
    up.on_click(functools.partial(cb, y=-1))
    right = pn.widgets.Button(name=RIGHT_ARROW, **button_kwargs)
    right.on_click(functools.partial(cb, x=1))
    down = pn.widgets.Button(name=DOWN_ARROW, **button_kwargs)
    down.on_click(functools.partial(cb, y=1))
    return pn.Column(
        pn.Row(get_sp(), up, get_sp(), margin=(0, 0)),
        pn.Row(left, down, right, margin=(0, 0)),
        # pn.Row(get_sp(), down, get_sp(), margin=(0, 0)),
        margin=(0, 0),
    )


def shear_buttons(cb, width: int = 40, height: int = 40, margin: tuple[int, int] = (2, 2)):
    """
    Buttons for x and y shear
    """
    kwargs = {
        'width': width,
        'height': height,
        'margin': margin,
        'sizing_mode': 'fixed',
    }
    get_sp = lambda: pn.Spacer(**kwargs)  # noqa
    button_kwargs = {
        'button_type': 'primary',
        **kwargs,
    }
    left = pn.widgets.Button(name=SHEAR_LESS, **button_kwargs)
    left.on_click(functools.partial(cb, x=-1))
    up = pn.widgets.Button(name=SHEAR_MORE, **button_kwargs)
    up.on_click(functools.partial(cb, y=-1))
    right = pn.widgets.Button(name=SHEAR_MORE, **button_kwargs)
    right.on_click(functools.partial(cb, x=1))
    down = pn.widgets.Button(name=SHEAR_LESS, **button_kwargs)
    down.on_click(functools.partial(cb, y=1))
    return pn.Column(
        pn.Row(get_sp(), up, get_sp(), margin=(0, 0)),
        pn.Row(left, down, right, margin=(0, 0)),
        # pn.Row(get_sp(), down, get_sp(), margin=(0, 0)),
        margin=(0, 0),
    )


def rotate_buttons(cb):
    """A button array for rotate acw / cw"""
    width = height = 40
    margin = (2, 2)
    sp = pn.Spacer(width=width, height=height, margin=margin)
    kwargs = {'width': width, 'height': height, 'margin': margin, 'button_type': 'primary'}
    acw_btn = pn.widgets.Button(name=ROTATE_LEFT_ARROW, **kwargs)
    acw_btn.on_click(functools.partial(cb, dir=-1))
    cw_btn = pn.widgets.Button(name=ROTATE_RIGHT_ARROW, **kwargs)
    cw_btn.on_click(functools.partial(cb, dir=1))
    return pn.Row(sp, acw_btn, cw_btn, margin=(0, 0))


def scale_buttons(cb):
    """A button array for scaling x / y / xy up and down"""
    width = height = 40
    margin = (2, 2)
    text_kwargs = {'width': width // 2,
                   'height': height // 2,
                   'margin': margin,
                   'align': ('end', 'center')}
    button_kwargs = {'width': width,
                     'height': height,
                     'margin': margin,
                     'button_type': 'primary'}
    x_row = up_down_pair('X:',
                         cb,
                         {'xdir': 1},
                         {'xdir': -1},
                         text_kwargs,
                         button_kwargs)
    y_row = up_down_pair('Y:',
                         cb,
                         {'ydir': 1},
                         {'ydir': -1},
                         text_kwargs,
                         button_kwargs)
    xy_row = up_down_pair('XY:',
                          cb,
                          {'xdir': 1, 'ydir': 1},
                          {'xdir': -1, 'ydir': -1},
                          text_kwargs,
                          button_kwargs)
    lo = pn.Column(x_row,
                   y_row,
                   xy_row, margin=(0, 0))
    return lo


def up_down_pair(name, cb, upkwargs, downkwargs, text_kwargs, button_kwargs):
    sp = pn.Spacer(**text_kwargs)
    text = pn.widgets.StaticText(value=name, **text_kwargs)
    compress = pn.widgets.Button(name=f'{RIGHT_ARROW} {LEFT_ARROW}', **button_kwargs)
    compress.on_click(functools.partial(cb, **downkwargs))
    expand = pn.widgets.Button(name=f'{LEFT_ARROW} {RIGHT_ARROW}', **button_kwargs)
    expand.on_click(functools.partial(cb, **upkwargs))
    return pn.Row(sp, text, compress, expand, margin=(0, 0))


def fine_adjust(
    static: np.ndarray,
    moving: np.ndarray,
    initial_transform: Optional['AffineTransform'] = None
):
    """
    Provides a UI panel to manually align the image moving onto static
    Optionally provide a skimage.transform.GeometricTransform object
    to pre-transform moving
    """
    transformer_moving = ImageTransformer(moving)
    if initial_transform:
        transformer_moving.add_transform(initial_transform, output_shape=static.shape)
    else:
        # To be sure we set the output shape to match static
        transformer_moving.add_null_transform(output_shape=static.shape)

    # static_name = 'Static'
    moving_name = 'Moving'

    fig = (
        ApertureFigure
        .new(
            static,
            tools=False,
        )
    )
    fig.add_control_panel(fig.im, name="Static image")
    static_im = fig.im
    moving_im = (
        BokehImage
        .new()
        .from_numpy(
            transformer_moving.get_transformed_image(cval=np.nan)
        )
        .on(fig.fig)
    )
    fig.add_control_panel(moving_im, name="Moving image")

    static_im.color.change_cmap('Blues')
    moving_im.color.change_cmap('Reds')
    static_im.color.add_colorbar()
    moving_im.color.add_colorbar()
    static_im.color._lin_mapper.nan_color = (0,) * 4
    moving_im.color._lin_mapper.nan_color = (0,) * 4

    moving_im.im.global_alpha = 0.5
    overlay_alpha = moving_im.color.get_alpha_slider(
        name=f'{moving_name} alpha',
        max_width=200,
    )

    wobble_alpha_cbox = CheckboxGroup(
        labels=['Wobble alpha, step:'],
        active=[],
        align='center',
    )
    wobble_step_input = Spinner(
        low=0.05,
        high=0.25,
        value=0.1,
        step=0.05,
        align='end',
        width=100,
        format="0.2f",
    )
    wobble_callback = CustomJS(
        args=dict(
            alpha_slider=overlay_alpha,
            wobble_step=wobble_step_input,
        ),
        code=R'''
const TIMEOUT = 25
var RUNNING = false
var DIRECTION = 1

export default async function (args, obj, data, context) {
    const active = obj.active.length == 1
    if (!active) {
        RUNNING = false
        return
    }
    var stepsize = 0.05
    var current_val = args.alpha_slider.value

    RUNNING = true
    while (RUNNING) {
        current_val = args.alpha_slider.value
        stepsize = Math.min(Math.max(args.wobble_step.value, 0.01), 0.9)

        var new_val = Math.min(Math.max(current_val + DIRECTION * stepsize, 0), 1)
        if ((new_val >= 1) || (new_val <= 0)) {
            DIRECTION = -1 * DIRECTION
        }
        args.alpha_slider.value = new_val
        await new Promise(r => setTimeout(r, TIMEOUT));
    }
}
'''
    )
    wobble_alpha_cbox.js_on_change('active', wobble_callback)

    show_diff_cbox = pn.widgets.Checkbox(name='Show image difference',
                                         value=False,
                                         align='center')

    translate_step_input = pn.widgets.FloatInput(name='Translate step (px):',
                                                 value=1.,
                                                 start=0.1,
                                                 end=100.,
                                                 width=125)

    shear_step_input = pn.widgets.FloatInput(name='Shear step (deg):',
                                             value=1.,
                                             start=0.1,
                                             end=10.,
                                             width=125)

    def _transform_md():
        transform = transformer_moving.get_combined_transform()
        return format_transform_md(transform)

    transform_md = pn.pane.Markdown(
        object=_transform_md(),
    )

    def update_moving(*updates, fix_clims=True):
        moving = transformer_moving.get_transformed_image()
        if show_diff_cbox.value:
            to_display = np.float32(moving) - np.float32(static)
        else:
            to_display = moving
        moving_im.update(to_display)
        fig.push()
        transform_md.object = _transform_md()

    # def update_moving():
        # update_moving_sync()

    def switch_diff_image(event):
        if event.new:
            overlay_alpha.value = 1.
        else:
            overlay_alpha.value = 0.5
        update_moving()

    show_diff_cbox.param.watch(switch_diff_image, 'value')

    def fine_translate(event, x=0, y=0):
        if not x and not y:
            print('No translate requested')
            return
        raw_adjust = -1 * translate_step_input.value
        transformer_moving.translate(xshift=x * raw_adjust, yshift=y * raw_adjust)
        update_moving()

    def fine_shear(event, x=0, y=0):
        if not x and not y:
            print('no shear')
            return
        raw_adjust = (np.pi / 360.0 * shear_step_input.value)
        transformer_moving.shear(xshear=raw_adjust * x, yshear=raw_adjust * y)
        update_moving()

    origin_cursor = (
        Cursor
        .new()
        .from_pos(*tuple(a // 2 for a in static.shape))
        .on(fig.fig)
        .editable(selected=True)
    )
    origin_cursor.cursor.line_color = 'cyan'
    origin_cursor.cursor.line_alpha = 0.

    about_center_cbox = pn.widgets.Checkbox(name='Center-origin',
                                            value=True,
                                            width=125)

    def _set_cursor_alpha(event):
        origin_cursor.cursor.line_alpha = float(not event.new)

    about_center_cbox.param.watch(_set_cursor_alpha, 'value')

    rotate_step_input = pn.widgets.FloatInput(name='Rotate step (deg):',
                                              value=1.,
                                              start=0.1,
                                              end=100.,
                                              width=125)

    def fine_rotate(event, dir=0):
        if not dir:
            print('No rotate requested')
            return
        about_center = about_center_cbox.value
        true_rotate = -1 * rotate_step_input.value * dir
        if about_center:
            transformer_moving.rotate_about_center(rotation_degrees=true_rotate)
        else:
            cx, cy = origin_cursor.current_pos()
            transformer_moving.rotate_about_point((cy, cx), rotation_degrees=true_rotate)
        update_moving()

    scale_step_input = pn.widgets.FloatInput(name='Scale step (%):',
                                             value=1.,
                                             start=0.1,
                                             end=100.,
                                             width=125)

    def fine_scale(event, xdir=0, ydir=0):
        if not xdir and not ydir:
            print('No scaling requested')
            return
        about_center = about_center_cbox.value
        xscale = 1 - (scale_step_input.value * xdir / 100)
        yscale = 1 - (scale_step_input.value * ydir / 100)
        if about_center:
            transformer_moving.xy_scale_about_center(xscale=xscale, yscale=yscale)
        else:
            cx, cy = origin_cursor.current_pos()
            transformer_moving.xy_scale_about_point((cy, cx), xscale=xscale, yscale=yscale)
        update_moving()

    # def translate_from_path(path_dict):
    #     xs = path_dict['xs']
    #     ys = path_dict['ys']

    #     xshift = -1 * (xs[-1] - xs[0])
    #     yshift = -1 * (ys[-1] - ys[0])
    #     transformer_moving.translate(xshift=xshift, yshift=yshift)
    #     update_moving_sync()

    # fig.add_free_callback(callback=translate_from_path)

    def _undo(event):
        transformer_moving.remove_transform()
        update_moving()

    undo_button = pn.widgets.Button(name='Undo',
                                    max_width=125,
                                    button_type='primary')
    undo_button.on_click(_undo)

    def getter() -> AffineTransform:
        return transformer_moving.get_combined_transform()

    fig._toolbar.insert(0, show_diff_cbox)
    fig._toolbar.insert(0, wobble_step_input)
    fig._toolbar.insert(0, wobble_alpha_cbox)
    fig._toolbar.insert(0, overlay_alpha)

    return pn.Column(
        pn.Row(
            pn.Column(
                fig.layout,
                transform_md,
            ),
            pn.Column(
                undo_button,
                translate_step_input,
                translate_buttons(fine_translate),
                about_center_cbox,
                rotate_step_input,
                rotate_buttons(fine_rotate),
                scale_step_input,
                scale_buttons(fine_scale),
                shear_step_input,
                shear_buttons(fine_shear),
            )
        )
    ), getter


def get_joint_pointset(static_figure, moving_figure):
    """
    Place one scatter plot on each figure, and set up a callback such that
    points created or deleted on one or the other are mirrored in the other
    """
    color_iterator = itertools.cycle(get_bokeh_palette())

    defaults = {
        'cx': 0,
        'cy': 0,
        'moving_cx': 0,
        'moving_cy': 0,
        'color': '#000000',
    }

    initial_data = {k: [] for k in defaults.keys()}

    cds = ColumnDataSource(initial_data, default_values=defaults)

    static_pointset = (
        PointSet(
            cds
        )
        .on(static_figure)
        .editable(selected=True)
    )
    static_pointset.points.fill_color = "color"

    moving_pointset = (
        PointSet(
            cds,
            x="moving_cx",
            y="moving_cy",
        )
        .on(moving_figure)
        .editable(selected=True)
    )
    moving_pointset.points.fill_color = "color"

    default_fill = -1

    def _sync_points(attr, old, new):
        # The synchronization callback, will be called each time
        # the data source is changed / modified so we first
        # exit early if the source length hasn't changed !
        if not new['cx'] or len(old['cx']) == len(new['cx']):
            return
        # Use the color as a proxy to recognize which points are new
        to_patch_ix = [
            i for i, c in enumerate(new['color'])
            if c in [defaults['color'], default_fill]
        ]
        # Exit early if no points are new, this is the case for point deletion
        if not to_patch_ix:
            return

        patches = {'color': [(i, next(color_iterator)) for i in to_patch_ix]}
        cx_patches = 'cx', [(i, new['moving_cx'][i]) for i in to_patch_ix
                            if new['cx'][i] in [defaults['cx'], default_fill]]
        cy_patches = 'cy', [(i, new['moving_cy'][i]) for i in to_patch_ix
                            if new['cy'][i] in [defaults['cy'], default_fill]]
        moving_cx_patches = 'moving_cx', [(i, new['cx'][i]) for i in to_patch_ix
                                          if new['moving_cx'][i] in [defaults['moving_cx'],
                                                                     default_fill]]
        moving_cy_patches = 'moving_cy', [(i, new['cy'][i]) for i in to_patch_ix
                                          if new['moving_cy'][i] in [defaults['moving_cy'],
                                                                     default_fill]]
        valid_point_patches = {k: v for k, v in [cx_patches,
                                                 cy_patches,
                                                 moving_cx_patches,
                                                 moving_cy_patches] if v}
        patches.update(valid_point_patches)
        static_pointset.cds.patch(patches)

    static_pointset.cds.on_change('data', _sync_points)

    return static_pointset, moving_pointset


def point_registration(
    static: np.ndarray,
    moving: np.ndarray,
):
    """
    Provides a UI panel for pointset-to-pointset image registration
    from the moving image onto the static image
    Initial points can be supplied as a Pandas DataFrame
    with the following 4 columns:
        - `cx`, `cy` for points in the static image,
        - `moving_cx`, `moving_cy` for corresponding points in the moving image.
    """
    transformer_moving = ImageTransformer(moving)
    transformer_moving.add_null_transform(output_shape=static.shape)

    static_fig = (
        ApertureFigure
        .new(
            static,
            tools=False,
            title="Static",
        )
    )

    overlay_image = (
        BokehImage
        .new()
        .from_numpy(
            moving,
        )
        .on(static_fig.fig)
    )

    overlay_image.color._lin_mapper.nan_color = (0,) * 4
    overlay_image.im.global_alpha = 0.
    overlay_alpha = overlay_image.color.get_alpha_slider(
        name='Overlay alpha',
        max_width=200,
        align="end",
    )

    moving_fig = (
        ApertureFigure
        .new(
            moving,
            tools=False,
            title="Moving",
        )
    )

    static_pointset, moving_pointset = get_joint_pointset(
        static_fig.fig,
        moving_fig.fig,
    )

    transformations = {s.title(): s for s in ImageTransformer.available_transforms()}
    method_select = pn.widgets.Select(
        name='Transformation type',
        options=[*transformations.keys()],
        width=125,
        align="end",
    )
    run_button = pn.widgets.Button(
        name='Run',
        button_type='primary',
        width=100,
        align='end',
    )
    output_md = pn.pane.Markdown(
        object='No transform defined',
        align="end",
    )
    clear_button = pn.widgets.Button(
        name='Clear points',
        align='end',
        button_type="warning",
    )

    def _clear(event):
        static_pointset.clear()
        overlay_image.update(moving)
        output_md.object = 'No transform defined'
        transformer_moving.clear_transforms()

        static_fig.push(moving_fig)

    clear_button.on_click(_clear)

    def get_points():
        static_data = static_pointset.cds.data
        if len(static_data['cx']) == 0:
            output_md.object = 'No points defined'
            return np.zeros((0, 2)), np.zeros((0, 2))
        static_points = np.stack((static_data['cx'], static_data['cy']), axis=-1)
        moving_data = moving_pointset.cds.data
        moving_points = np.stack((moving_data['moving_cx'], moving_data['moving_cy']), axis=-1)
        return static_points, moving_points

    def _compute_transform(event, fix_clims=True):
        method = transformations[method_select.value]

        static_points, moving_points = get_points()
        try:
            transform = transformer_moving.estimate_transform(
                static_points,
                moving_points,
                method=method,
                clear=True,
            )
        except Exception as e:
            output_md.object = f'Error computing transform: {str(e)}'
            return
        try:
            output_md.object = format_transform_md(transform)
        except Exception:
            output_md.object = 'Post-transform error (format?)'
            return

        warped_moving = transformer_moving.get_transformed_image(output_shape=static.shape)
        overlay_image.update(warped_moving)
        static_fig.push()

    run_button.on_click(_compute_transform)

    static_fig._outer_toolbar.height = 0
    moving_fig._outer_toolbar.height = 0

    layout = pn.Column(
        pn.Row(
            run_button, method_select, overlay_alpha, clear_button, output_md,
            min_height=80,
        ),
        pn.Row(
            static_fig.layout,
            moving_fig.layout,
        ),
    )

    def getter():
        return {
            'points': get_points(),
            'transform': transformer_moving.get_combined_transform()
        }

    return layout, getter
