from __future__ import annotations
from typing import Any, NewType
import numpy as np
import numpy.typing as nt
from skimage.draw import disk, circle_perimeter_aa


def aa_disk(frame: np.ndarray, cy: int, cx: int, r: int, scale: float):
    frame_shape = frame.shape
    rr, cc = disk((cy, cx), r, shape=frame_shape)
    frame[rr, cc] = scale

    rr, cc, val = circle_perimeter_aa(cy, cx, r, shape=frame_shape)
    existing = frame[rr, cc]
    newval = np.maximum(existing, val * scale)
    frame[rr, cc] = newval


def intensity_fn(val: float, falloff: float, bias: float) -> float:
    # map from 1..0 to np.inf..0, clip negative values to 0.
    val = max(0., val)
    val += bias
    if val > 1.:
        val -= 2 * (val - 1)
    return val ** falloff


def diff_frame(
    frame_shape: tuple[int, int],
    radius: float,
    centre: complex,
    a: complex | None,
    b: complex | None,
    dtype: nt.DTypeLike = np.float32,
    falloff: float = 1.,
    bias: float = 0,
    noise_level: float = 0.,
) -> np.ndarray:
    radius = int(radius)
    h, w = frame_shape
    buffer = 2 * radius

    buffered_shape = (h + 2 * buffer, w + 2 * buffer)
    if noise_level == 0.:
        frame = np.zeros(
            buffered_shape,
            dtype=dtype,
        )
    else:
        frame = np.random.uniform(
            high=noise_level, size=buffered_shape
        ).astype(dtype)

    max_r = max(
        np.abs(centre),
        np.abs(h*1j - centre),
        np.abs(w - centre),
        np.abs(w + h*1j - centre),
    )

    cx = centre.real
    cy = centre.imag
    cx += buffer
    cy += buffer

    c = cx + cy * 1j

    aa_disk(
        frame,
        int(np.round(c.imag)),
        int(np.round(c.real)),
        radius,
        intensity_fn(1., falloff, bias)
    )

    if a is not None and b is not None:
        ming = min(np.abs(a), np.abs(b))
        maxdim = np.linalg.norm(frame.shape)
        maxidx = np.ceil(maxdim / ming)

        for ai, bi in np.mgrid[-maxidx: maxidx + 1, -maxidx: maxidx + 1].reshape(2, -1).T:
            if ai == 0 and bi == 0:
                continue
            shift = ai * a + bi * b
            real_pos = c + shift
            if not (0 < real_pos.real < (w + 2 * buffer)):
                continue
            if not (0 < real_pos.imag < (h + 2 * buffer)):
                continue
            aa_disk(
                frame,
                int(np.round(real_pos.imag)),
                int(np.round(real_pos.real)),
                radius,
                intensity_fn(1. - (np.abs(shift) / max_r), falloff, bias)
            )

    return frame[buffer: -buffer, buffer: -buffer]


Degree = NewType('Degree', float)


def transform_complex(c: complex, scale: float = 1., rotation: Degree = 0.) -> complex:
    newh = np.abs(c) * scale
    dx = newh * np.cos(np.angle(c) + np.deg2rad(rotation))
    dy = newh * np.sin(np.angle(c) + np.deg2rad(rotation))
    return dx + dy * 1j


def generate_data(
    nav_shape: tuple[int, ...],
    sig_shape: tuple[int, int],
    *,
    radius: float,
    centre: complex,
    a: np.ndarray[Any, complex] | complex | None,
    b: np.ndarray[Any, complex] | complex | None,
    centre_offset: np.ndarray[Any, complex] | None = None,
    a_mult: np.ndarray[Any, float] | float = 1.,
    b_mult: np.ndarray[Any, float] | float = 1.,
    a_rot: np.ndarray | Degree = 0.,
    b_rot: np.ndarray | Degree = 0.,
    falloff: np.ndarray | float = 1.,
    bias: np.ndarray | float = 0,
    norm: np.ndarray | float | None = None,
    noise_level: np.ndarray | float = 0.,
    dtype: nt.DTypeLike = np.float32,
):
    assert len(sig_shape) == 2
    if centre_offset is None:
        centre_offset = np.zeros(nav_shape, dtype=complex)
    ab_none = (a is None, b is None)
    assert all(ab_none) or (not any(ab_none))
    if a is not None:
        try:
            a.shape
        except AttributeError:
            a = np.full(nav_shape, a, dtype=complex)
        a = transform_complex(a, scale=a_mult, rotation=a_rot)
    if b is not None:
        try:
            b.shape
        except AttributeError:
            b = np.full(nav_shape, b, dtype=complex)
        b = transform_complex(b, scale=b_mult, rotation=b_rot)
    try:
        falloff.shape
    except AttributeError:
        falloff = np.full(nav_shape, falloff, dtype=float)
    try:
        bias.shape
    except AttributeError:
        bias = np.full(nav_shape, bias, dtype=float)
    try:
        noise_level.shape
    except AttributeError:
        noise_level = np.full(nav_shape, noise_level, dtype=float)
    data = np.zeros(nav_shape + sig_shape, dtype=dtype)
    for idx, array in enumerate(data.reshape(-1, *sig_shape)):
        array[:] = diff_frame(
            sig_shape,
            radius,
            centre + centre_offset.flat[idx],
            a.flat[idx] if a is not None else a,
            b.flat[idx] if b is not None else b,
            dtype=dtype,
            falloff=falloff.flat[idx],
            bias=bias.flat[idx],
            noise_level=noise_level.flat[idx],
        )
    if norm is not None:
        data /= (data.sum(axis=(2, 3))[..., np.newaxis, np.newaxis])
        data *= norm
    return data
