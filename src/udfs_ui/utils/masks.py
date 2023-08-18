from __future__ import annotations
import numpy as np


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
