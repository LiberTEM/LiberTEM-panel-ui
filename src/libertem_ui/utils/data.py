from __future__ import annotations
import warnings
from typing import Any, NewType
import numpy as np
import numpy.typing as nt
from skimage.filters import gaussian
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
    centre: complex,
    radius: float,
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

        n_drawn = 0
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
            n_drawn += 1
            if n_drawn == 128:
                warnings.warn('Drawing more than 128 disks in this '
                              'frame, this could take a while...')

    return frame[buffer: -buffer, buffer: -buffer]


Degree = NewType('Degree', float)


def transform_complex(c: complex, scale: float = 1., rotation: Degree = 0.) -> complex:
    newh = np.abs(c) * scale
    dx = newh * np.cos(np.angle(c) + np.deg2rad(rotation))
    dy = newh * np.sin(np.angle(c) + np.deg2rad(rotation))
    return dx + dy * 1j


def is_numpy(obj):
    try:
        obj.shape
        obj.size
        return True
    except AttributeError:
        return False


def generate_data(
    nav_shape: tuple[int, ...],
    sig_shape: tuple[int, int],
    *,
    centre: complex,
    radius: np.ndarray | float,
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
    centre = complex(centre)
    if not is_numpy(radius):
        radius = np.full(nav_shape, radius, dtype=float)
    if centre_offset is None:
        centre_offset = np.zeros(nav_shape, dtype=complex)
    ab_none = (a is None, b is None)
    assert all(ab_none) or (not any(ab_none))
    if a is not None:
        if not is_numpy(a):
            a = np.full(nav_shape, a, dtype=complex)
        a = transform_complex(a, scale=a_mult, rotation=a_rot)
        if (np.abs(a) < 0.5 * radius).any():
            warnings.warn('a-vector length much less than radius in some frames')
    if b is not None:
        if not is_numpy(b):
            b = np.full(nav_shape, b, dtype=complex)
        b = transform_complex(b, scale=b_mult, rotation=b_rot)
        if (np.abs(b) < 0.5 * radius).any():
            warnings.warn('b-vector length much less than radius in some frames')
    if not is_numpy(falloff):
        falloff = np.full(nav_shape, falloff, dtype=float)
    if not is_numpy(bias):
        bias = np.full(nav_shape, bias, dtype=float)
    if not is_numpy(noise_level):
        noise_level = np.full(nav_shape, noise_level, dtype=float)
    data = np.zeros(nav_shape + sig_shape, dtype=dtype)
    for idx, array in enumerate(data.reshape(-1, *sig_shape)):
        array[:] = diff_frame(
            sig_shape,
            centre + centre_offset.flat[idx],
            radius.flat[idx],
            a.flat[idx] if a is not None else a,
            b.flat[idx] if b is not None else b,
            dtype=dtype,
            falloff=falloff.flat[idx],
            bias=bias.flat[idx],
            noise_level=noise_level.flat[idx],
        )
    if norm is not None:
        if not is_numpy(norm):
            norm = np.full(nav_shape, norm, dtype=float)
        data /= (data.sum(axis=(2, 3))[..., np.newaxis, np.newaxis])
        data *= norm[..., np.newaxis, np.newaxis]
    return data


def planar_field(
    shape: tuple[int, int],
    yrange: np.ndarray | tuple[complex, complex] | tuple[float, float],
    xrange: np.ndarray | tuple[complex, complex] | tuple[float, float],
) -> np.ndarray:
    h, w = shape
    if not is_numpy(yrange):
        y0, y1 = yrange
        yrange = np.linspace(y0, y1, num=h, endpoint=True)
    if not is_numpy(xrange):
        x0, x1 = xrange
        xrange = np.linspace(x0, x1, num=w, endpoint=True)
    return yrange.reshape(-1, 1) + xrange.reshape(1, -1)


def demo_dataset():
    nav_shape = (12, 32)
    sig_shape = (128, 128)
    descan = planar_field(nav_shape, (-1-2j, 2+1j), (2-1j, -2+1j))
    feature_slice = np.s_[:, 18:25]
    feature_shift = np.zeros(nav_shape, dtype=complex)
    feature_shift[feature_slice] = 4 + 2j
    feature_shift.real = gaussian(feature_shift.real)
    feature_shift.imag = gaussian(feature_shift.imag)
    haadf_slice = np.s_[:, :8]
    bias = np.zeros(nav_shape, dtype=float)
    bias[haadf_slice] = 0.35
    bias = gaussian(bias)
    a_vec = np.full(nav_shape, 32+0j, dtype=complex)
    a_offset = np.linspace(-1, 1., num=nav_shape[1], dtype=complex)[np.newaxis, :]
    base_tilt = np.full(nav_shape, 12, dtype=float)
    ab_angle = np.full(nav_shape, 60, dtype=float)
    angle_offset = np.linspace(-2, 2., num=nav_shape[0], dtype=float)[:, np.newaxis]
    return generate_data(
        nav_shape,
        sig_shape,
        centre=58+70j,
        centre_offset=descan + feature_shift,
        radius=8,
        a=a_vec + a_offset,
        b=a_vec - a_offset,
        a_rot=base_tilt - angle_offset,
        b_rot=base_tilt + ab_angle + angle_offset,
        noise_level=0.05,
        falloff=1.5,
        bias=bias,
        norm=np.random.uniform(low=0.95, high=1.05, size=nav_shape),
    )
