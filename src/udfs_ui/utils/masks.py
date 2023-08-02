from __future__ import annotations
import numpy as np
import pandas as pd
import skimage.draw as skdraw

from . import pairwise


def sanitize_val(val, clip_to=None, clip_from=0, round=True, to_int=True):
    val = np.asarray(val)
    if round:
        val = np.round(val)
    if clip_to is not None:
        in_bounds = np.logical_and(np.greater_equal(val, clip_from),
                                   np.less_equal(val, clip_to))
        val = np.clip(val, clip_from, clip_to)
    else:
        in_bounds = False
    if to_int:
        val = np.asarray(val).astype(int)
    try:
        val = val.item()
    except (AttributeError, ValueError):
        # val is not a scalar, or is already a non-numpy sequence
        pass
    return val, in_bounds


def clip_posxy_array(pos_xy, shape, round=True, to_int=True):
    pos_xy = np.asarray(pos_xy).reshape(-1, 2)
    h, w = shape
    clipped_x, ib_x = sanitize_val(pos_xy[:, 0], clip_to=w - 1, round=round, to_int=to_int)
    clipped_y, ib_y = sanitize_val(pos_xy[:, 1], clip_to=h - 1, round=round, to_int=to_int)
    is_valid = np.logical_and(ib_x, ib_y)
    return np.stack((clipped_x, clipped_y), axis=-1).squeeze(), is_valid.squeeze()


def sanitize_posxy(pos_xy, round=True, to_int=True):
    pos_xy = np.asarray(pos_xy).reshape(-1, 2)
    san_x, _ = sanitize_val(pos_xy[:, 0], round=round, to_int=to_int)
    san_y, _ = sanitize_val(pos_xy[:, 1], round=round, to_int=to_int)
    return np.stack((san_x, san_y), axis=-1).squeeze()


def in_bounds_array(pos_xy, shape):
    pos_xy = np.asarray(pos_xy).reshape(-1, 2)
    h, w = shape
    lower = pos_xy >= 0
    upper = pos_xy <= [w - 1, h - 1]
    return np.logical_and(lower, upper).all(axis=1)


def filter_valid(sequence, llim, ulim):
    return sequence[get_valid_idxs(sequence, llim, ulim)]


def get_valid_idxs(sequence, llim, ulim):
    return np.logical_and(sequence >= llim, sequence <= ulim)


def filter_rrcc(rr, cc, shape):
    h, w = shape
    rr_valid = get_valid_idxs(rr, 0, h-1)
    cc_valid = get_valid_idxs(cc, 0, w-1)
    valid = np.logical_and(rr_valid, cc_valid)
    return rr[valid], cc[valid]


def pd_to_floating(df):
    ar = np.asarray(df).astype(float)
    try:
        return ar.item()
    except (ValueError, ArithmeticError):
        return ar


def point_to_mask(point_annotation: pd.DataFrame,
                  mask: np.ndarray,
                  fill_value=True):
    """
    """
    pos_xy = pd_to_floating(point_annotation[['cx', 'cy']])
    (x, y), valid = clip_posxy_array(pos_xy, mask.shape)
    if valid.item():
        mask[y, x] = fill_value
    return mask


def circle_to_mask(circle_annotation: pd.DataFrame,
                   mask: np.ndarray,
                   fill_value=True):
    """
    """
    radius = pd_to_floating(circle_annotation['r0'])
    cxy = pd_to_floating(circle_annotation[['cx', 'cy']])
    cx, cy = sanitize_posxy(cxy)
    rr, cc = skdraw.disk((cy, cx), radius)
    rr, cc = filter_rrcc(rr, cc, mask.shape)
    mask[rr, cc] = fill_value
    return mask


def annulus_to_mask(annulus_annotation: pd.DataFrame,
                    mask: np.ndarray,
                    fill_value=True):
    """
    """
    r0, r1 = pd_to_floating(annulus_annotation[['r0', 'r1']])
    cxy = pd_to_floating(annulus_annotation[['cx', 'cy']])
    cx, cy = sanitize_posxy(cxy)
    rr, cc = skdraw.disk((cy, cx), r1)
    rr, cc = filter_rrcc(rr, cc, mask.shape)
    _temp_mask = np.zeros_like(mask)
    _temp_mask[rr, cc] = fill_value
    rr, cc = skdraw.disk((cy, cx), r0)
    rr, cc = filter_rrcc(rr, cc, mask.shape)
    _temp_mask[rr, cc] = _temp_mask.dtype.type(False)
    mask = np.logical_or(mask, _temp_mask)
    return mask


def line_to_mask(line_annotation: pd.DataFrame,
                 mask: np.ndarray,
                 fill_value=True):
    """
    """
    xs, _ = sanitize_val(pd_to_floating(line_annotation['xs']))
    ys, _ = sanitize_val(pd_to_floating(line_annotation['ys']))
    for (x0, x1), (y0, y1) in zip(pairwise(xs), pairwise(ys)):
        rr, cc = skdraw.line(y0, x0, y1, x1)
        rr, cc = filter_rrcc(rr, cc, mask.shape)
        mask[rr, cc] = fill_value
    return mask


def polygon_to_mask(polygon_annotation: pd.DataFrame,
                    mask: np.ndarray,
                    fill_value=True):
    """
    """
    xs, _ = sanitize_val(pd_to_floating(polygon_annotation['xs']))
    ys, _ = sanitize_val(pd_to_floating(polygon_annotation['ys']))
    rr, cc = skdraw.polygon(ys, xs)
    rr, cc = filter_rrcc(rr, cc, mask.shape)
    mask[rr, cc] = fill_value
    return mask


def rectangle_to_mask(rectangle_annotation: pd.DataFrame,
                      mask: np.ndarray,
                      fill_value=True):
    """
    """
    cx, cy = pd_to_floating(rectangle_annotation[['cx', 'cy']])
    h, w = pd_to_floating(rectangle_annotation[['h', 'w']])
    lefttop = cx - w / 2, cy - h / 2
    rightbottom = cx + w / 2, cy + h / 2
    lefttop, _ = clip_posxy_array(lefttop, mask.shape, round=True, to_int=True)
    rightbottom, _ = clip_posxy_array(rightbottom, mask.shape, round=True, to_int=True)
    if (lefttop[0] == rightbottom[0]) or (lefttop[1] == rightbottom[1]):
        return mask
    slice_y = slice(lefttop[1], rightbottom[1] + 1)
    slice_x = slice(lefttop[0], rightbottom[0] + 1)
    mask[slice_y, slice_x] = fill_value
    return mask


_mask_factory = {'point': point_to_mask,
                 'circle': circle_to_mask,
                 'circular_annulus': annulus_to_mask,
                 'line': line_to_mask,
                 'polygon': polygon_to_mask,
                 'rectangle': rectangle_to_mask}
