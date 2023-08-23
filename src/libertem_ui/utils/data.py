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


def diff_frame(
    frame_shape: tuple[int, int],
    radius: float,
    centre: complex,
    a: complex,
    b: complex,
    dtype: nt.DTypeLike = np.float32,
) -> np.ndarray:
    h, w = frame_shape
    buffer = 2 * radius

    frame = np.zeros(
        (h + 2 * buffer, w + 2 * buffer),
        dtype=dtype,
    )

    cx = centre.real
    cy = centre.imag
    cx += buffer
    cy += buffer

    c = cx + cy * 1j

    ming = min(np.abs(a), np.abs(b))
    maxdim = np.linalg.norm(frame.shape)
    maxidx = np.ceil(maxdim / ming)
    max_r = max(frame.shape)

    for ai, bi in np.mgrid[-maxidx: maxidx + 1, -maxidx: maxidx + 1].reshape(2, -1).T:
        shift = ai * a + bi * b
        real_pos = c + shift
        if not (radius <= real_pos.real <= (w - radius)):
            continue
        if not (radius <= real_pos.imag <= (h - radius)):
            continue
        rel_radius = ((max_r - np.abs(shift)) / max_r) ** 2
        aa_disk(frame, cy, cx, radius, rel_radius)
    return frame[buffer: -buffer, buffer: -buffer]