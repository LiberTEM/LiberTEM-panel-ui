from __future__ import annotations
import numpy as np
from itertools import pairwise
from typing import Callable

import panel as pn
from scipy.interpolate import RectBivariateSpline
from bokeh.models import CustomJS, Slider
from bokeh.plotting import figure as BkFigure

from ..utils import Margin
from ..display.display_base import Rectangles, Curve
from ..display.vectors import MultiLine
from ..figure import ApertureFigure


def deduplicate(iterable):
    retain = []
    skip = False

    for el0, el1 in pairwise(iterable):
        if skip:
            skip = (el0 == el1)
            continue
        retain.append(el0)
        skip = (el0 == el1)

    if not skip:
        retain.append(el1)
    return retain


def make_spline_image(array: np.ndarray) -> RectBivariateSpline:
    """
    Fit a :class:`scipy.interpolate.RectBivariateSpline` to the array
    Adds an extra property `.shape` to the returned spline containing
    the `(h, w)` of the original array

    FIXME: The returned spline is indexed as (x, y) and should be refactored
    """
    assert array.ndim == 2
    h, w = array.shape
    xvals = np.linspace(0, w, num=w, endpoint=False)
    yvals = np.linspace(0, h, num=h, endpoint=False)
    spline_img = RectBivariateSpline(xvals, yvals, array.T, kx=1, ky=1)
    spline_img.shape = (h, w)
    return spline_img


def interp_xy(x0, y0, x1, y1, pt_per_pixel, clip_shape=None):
    num_p = np.sqrt((x1 - x0)**2 + (y1 - y0)**2) * pt_per_pixel
    num_p = max(1, int(np.ceil(num_p)))
    xvals = np.linspace(x0, x1, num=num_p, endpoint=True)
    yvals = np.linspace(y0, y1, num=num_p, endpoint=True)
    xyvals = np.stack((xvals, yvals), axis=1).reshape(-1, 2)
    if clip_shape:
        xyvals = xyvals[filter_xy(clip_shape, xyvals), :]
    return xyvals


def interpolate_sampling(sample_points, pt_per_pixel=1, concat=True, clip_shape=None):
    sampling = []
    for (x0, y0), (x1, y1) in pairwise(sample_points):
        xyvals = interp_xy(x0, y0, x1, y1, pt_per_pixel, clip_shape=clip_shape)
        if len(sampling) == 0:
            sampling.append(xyvals)
        else:
            sampling.append(xyvals[1:, :])
    if concat:
        return np.concatenate(sampling, axis=0)
    else:
        return sampling


def compute_orthog_unit_vecs(sample_points):
    vecs = []
    for (x0, y0), (x1, y1) in pairwise(sample_points):
        lenvec = np.sqrt((x1 - x0)**2 + (y1 - y0)**2)
        dx = (x1 - x0) / lenvec
        dy = (y1 - y0) / lenvec
        vecs.append(np.asarray((-dy, dx)))
    return vecs


def get_orthogonal_offsets(orth_vecs, orthog_dist, pt_per_pixel):
    offsets = []
    for ovec in orth_vecs:
        x0, y0 = 0, 0
        x1, y1 = ovec * orthog_dist
        xyvals = interp_xy(x0, y0, x1, y1, pt_per_pixel)
        xyvals = np.concatenate((-1 * xyvals[1:, :], xyvals), axis=0)
        offsets.append(xyvals)
    return offsets


def filter_xy(array_shape, xyvals):
    h, w = array_shape[:2]
    xmask = np.logical_and(xyvals[:, 0] >= 0, xyvals[:, 0] <= w - 1)
    ymask = np.logical_and(xyvals[:, 1] >= 0, xyvals[:, 1] <= h - 1)
    mask = np.logical_and(xmask, ymask)
    return mask


def sample_orthogonal(array_spline, sampling_coords, orth_offsets):
    samples = []
    for sample_xy, offset in zip(sampling_coords, orth_offsets):
        all_coords = sample_xy[:, np.newaxis, :] + offset[np.newaxis, :, :]
        ac_shape = all_coords.shape
        all_coords = all_coords.reshape(-1, 2)
        all_samples = array_spline(all_coords[:, 0], all_coords[:, 1],
                                   grid=False).reshape(ac_shape[:-1])
        valid_mask = filter_xy(array_spline.shape, all_coords).reshape(ac_shape[:-1])
        masked_samples = np.ma.array(all_samples, mask=~valid_mask)
        line_mean_masked = masked_samples.mean(axis=1)
        line_mean_masked.set_fill_value(np.nan)
        samples.append(line_mean_masked.filled())
    return np.concatenate(samples, axis=0)


def sampling_to_tcoord(sampling):
    distances = np.linalg.norm(np.diff(sampling, axis=0), axis=1)
    return np.concatenate(([0.], np.cumsum(distances)))


def image_sampler(array_spline, sample_points, oversampling=1, orthog_dist=0):
    sampling_coords = interpolate_sampling(sample_points,
                                           pt_per_pixel=oversampling,
                                           concat=False,
                                           clip_shape=array_spline.shape)
    coords_concat = np.concatenate(sampling_coords, axis=0)
    if orthog_dist > 1e-5:
        orth_vecs = compute_orthog_unit_vecs(sample_points)
        orth_offsets = get_orthogonal_offsets(orth_vecs, orthog_dist, oversampling)
        samples = sample_orthogonal(array_spline, sampling_coords, orth_offsets)
    else:
        samples = array_spline(coords_concat[:, 0], coords_concat[:, 1], grid=False)
    tcoord = sampling_to_tcoord(coords_concat)
    return tcoord, samples, coords_concat


def extract_path(data, extract_num=0):
    if not data or not data['xs']:
        return None
    sampling_paths = [*zip(data['xs'], data['ys'])]
    sampling_paths = [[(x, y) for x, y in zip(xx, yy)] for xx, yy in sampling_paths]
    try:
        path = sampling_paths[extract_num]
    except IndexError:
        return None
    points = deduplicate(path)
    if len(points) < 2:
        return None
    return points


def _sampling_plot(
    array_spline: RectBivariateSpline,
    sample_path: MultiLine,
    sample_curve: Curve,
    oversampling_slider: pn.widgets.FloatSlider,
    perpendicular_slider: Slider,
):

    def _update_curve():
        sample_points = extract_path(sample_path.cds.data)
        if sample_points is None:
            if any(len(v) for v in sample_curve.cds.data.values()):
                sample_curve.clear()
            return

        oversampling = oversampling_slider.value
        orthog_dist = perpendicular_slider.value
        xvals, yvals, _ = image_sampler(
            array_spline,
            sample_points,
            oversampling=oversampling,
            orthog_dist=orthog_dist,
        )

        sample_curve.update(xvals=xvals, yvals=yvals)

    def _update_curve_bk(attr, old, new):
        _update_curve()

    def _update_curve_pn(event):
        _update_curve()

    sample_path.cds.on_change('data', _update_curve_bk)
    oversampling_slider.param.watch(_update_curve_pn, 'value')
    perpendicular_slider.on_change('value_throttled', _update_curve_bk)


def _sample_boxes_js():
    return '''
function pairwise(arr, func) {
    var values = [];
    for (var i = 0; i < arr.length - 1; i++) {
        values.push(func(arr[i], arr[i + 1]))
    }
    return values
}

const zip = (a, b) => a.map((k, i) => [k, b[i]]);
const mean = (p0, p1) => (p0 + p1) / 2;

function line_length(pt_xy0, pt_xy1) {
    let xlen = pt_xy1[0] - pt_xy0[0]
    let ylen = pt_xy1[1] - pt_xy0[1]
    return Math.sqrt(xlen ** 2 + ylen ** 2)
}

function line_angle(pt_xy0, pt_xy1) {
    let xlen = pt_xy1[0] - pt_xy0[0]
    let ylen = pt_xy1[1] - pt_xy0[1]
    return Math.atan2(xlen, ylen)
}

function generateBoxes(xs, ys) {
    return {
        'cx': pairwise(xs, mean),
        'cy': pairwise(ys, mean),
        'h': pairwise(zip(xs, ys), line_length),
        'angle': pairwise(zip(xs, ys), line_angle),
    }
}

let width = width_slider.value
boxes_cds.clear()

if (sample_cds.length == 0 || width == 0) {
    boxes_cds.change.emit()
    return
}

let box_data = {}

for (let i = 0; i < sample_cds.length; i++) {
    let num_box = sample_cds.data['xs'][i].length - 1
    if (num_box > 0) {
        box_data = generateBoxes(sample_cds.data['xs'][i], sample_cds.data['ys'][i])
        box_data['w'] = new Array(num_box).fill(width)
    }
}

boxes_cds.data = box_data
boxes_cds.change.emit()
'''


def sampling_tool(array: np.ndarray, fig_kwargs=None) -> tuple[pn.layout.Row, Callable[[], dict]]:
    fig = ApertureFigure.new(
        array,
        **(fig_kwargs if fig_kwargs is not None else {})
    )

    sample_path = (
        MultiLine
        .new()
        .empty()
        .on(fig.fig)
        .editable(selected=True)
    )

    lines_tool = sample_path.tools('lines', fig.fig)[fig.fig][0]
    lines_tool.num_objects = 1

    sample_boxes = (
        Rectangles
        .new()
        .empty()
        .on(fig.fig)
    )
    sample_boxes.cds.data['angle'] = []
    sample_boxes.rectangles.angle = 'angle'

    # sampling_choice = pn.widgets.Select(
    #     value="Linear",
    #     options=[
    #         "nearest".capitalize(),
    #         "linear".capitalize(),
    #         "slinear".capitalize(),
    #         "cubic".capitalize(),
    #         # "quintic".capitalize(),
    #         # "pchip".capitalize(),
    #     ],
    #     width=200,
    # )
    oversampling_slider = pn.widgets.FloatInput(
        start=0.1,
        end=5.,
        value=1.5,
        step=0.1,
        width=120,
    )
    perpendicular_slider = Slider(
        name='Width (px)',
        start=0,
        end=200,
        step=1,
        value=0,
        width=200,
    )

    sample_fig = BkFigure(title="Profile")
    sample_fig.frame_height = fig.fig.frame_height
    sample_fig.width = 500
    sample_fig_pane = pn.pane.Bokeh(sample_fig)

    clear_btn = pn.widgets.Button(
        name="Clear",
        button_type="primary",
        margin=(5, 5),
    )

    def clear_window(*e):
        sample_path.clear()
        fig.push(sample_fig_pane)
        # There is a panel notebook bug where the CDS change won't trigger
        # so the sampling plot will not update until the next update
        # oversampling_slider.param.trigger("value")

    clear_btn.on_click(clear_window)

    remove_btn = pn.widgets.Button(
        name="Remove pt",
        button_type="primary",
        margin=Margin.hv(2, 5),
    )

    def remove_pt(*e):
        if sample_path.data_length <= 0:
            return

        sample_path.cds.update(
            data={
                k: [v[:-1] if len(v) > 2 else v for v in vallists]
                for k, vallists in sample_path.cds.data.items()
            }
        )
        fig.push(sample_fig_pane)
        # There is a panel notebook bug where the CDS change won't trigger
        # so the sampling plot will not update until the next update
        # oversampling_slider.param.trigger("value")

    remove_btn.on_click(remove_pt)

    sample_curve = (
        Curve
        .new()
        .from_vectors((0,), (0,))
        .on(sample_fig)
    )

    array_spline = make_spline_image(array)

    _sampling_plot(array_spline,
                   sample_path,
                   sample_curve,
                   oversampling_slider,
                   perpendicular_slider)

    box_cb = CustomJS(code=_sample_boxes_js(), args={'boxes_cds': sample_boxes.cds,
                                                     'sample_cds': sample_path.cds,
                                                     'width_slider': perpendicular_slider})
    sample_path.cds.js_on_change('data', box_cb)
    perpendicular_slider.js_on_change('value', box_cb)

    fig._toolbar.append(
        pn.widgets.StaticText(
            value="Sampling (pt-per-px):",
            align='center',
            margin=(2, 2),
        ),
    )
    fig._toolbar.append(oversampling_slider)
    fig._toolbar.append(clear_btn)
    fig._toolbar.append(remove_btn)

    def getter():
        return {
            'points': sample_path.cds.clone().data,
            'sampling': oversampling_slider.value,
            'width': perpendicular_slider.value,
            'profile': sample_curve.cds.clone().data,
        }

    return pn.Row(
        fig.layout,
        pn.Column(
            pn.Row(
                pn.widgets.StaticText(
                    value="Averaging (px):",
                    align='end',
                    margin=(5, 5),
                ),
                perpendicular_slider,
            ),
            sample_fig_pane,
        )
    ), getter
