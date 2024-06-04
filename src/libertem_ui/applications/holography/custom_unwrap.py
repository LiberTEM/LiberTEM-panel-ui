import heapq
import numpy as np
from typing_extensions import Literal
import numba


@numba.njit
def wrap(val: np.ndarray | float):
    return np.angle(np.exp(val * 1j))


def derivative_variance(array: np.ndarray, k: int = 3):
    # Adapted from
    # https://github.com/theilen/PyMRR/tree/master/mrr/unwrapping/
    diff = np.stack((
        np.diff(array, axis=0, prepend=0),
        np.diff(array, axis=1, prepend=0),
    ), axis=0)
    diff = wrap(diff)
    diff = np.pad(
        diff,
        ((0, 0), (1, 1), (1, 1)),
        mode="edge",
    )
    windows = np.lib.stride_tricks.sliding_window_view(
        diff,
        (k, k),
        axis=(1, 2),
    )
    return np.var(windows, axis=(-2, -1)).sum(axis=0)


@numba.njit
def get_neighbours_ud(idx: int, im_size: int, width: int):
    above = idx - width
    below = idx + width
    if above > 0:
        yield above
    if below < im_size:
        yield below


@numba.njit
def get_neighbours(idx: int, height: int, width: int, connectivity: int):
    im_size = height * width
    for _idx in get_neighbours_ud(idx, im_size, width):
        yield _idx
    left = idx - 1
    right = idx + 1
    col = idx % width
    if col > 0:
        yield left
        if connectivity > 4:
            for _idx in get_neighbours_ud(left, im_size, width):
                yield _idx
    if width - col > 1:
        yield right
        if connectivity > 4:
            for _idx in get_neighbours_ud(right, im_size, width):
                yield _idx


@numba.njit
def unwrap_heap(heap, flat_phase, flat_q, flat_to_q, height, width, uw_phase, connectivity):
    other_heap = [(1., -1, -1)]
    _ = heapq.heappop(other_heap)

    # Unwrap in heap order
    while len(heap) > 0:
        _, idx, parent_idx = heapq.heappop(heap)
        phase_diff = flat_phase[idx] - flat_phase[parent_idx]
        uw_phase[idx] = uw_phase[parent_idx] + wrap(phase_diff)
        # Queue neighbours
        for n_idx in get_neighbours(idx, height, width, connectivity):
            if flat_to_q[n_idx] > 0:
                heapq.heappush(heap, (flat_q[n_idx], n_idx, idx))
                flat_to_q[n_idx] = 0
            elif flat_to_q[n_idx] < 0:
                heapq.heappush(other_heap, (flat_q[n_idx], n_idx, idx))
                flat_to_q[n_idx] = 0

    # This will unwrap any disjoint additional areas in the seed mask
    # but there is no likelihood that they unwrap with the same scale
    # as the first unwrapped area, so this is disabled
    # remaining_mask = flat_to_q > 0
    # if remaining_mask.any():
    #     (nonzero_remaining,) = np.nonzero(remaining_mask)
    #     pos = nonzero_remaining[np.argmin(flat_q[remaining_mask])]
    #     heapq.heappush(heap, (flat_q[pos], pos, pos))
    #     flat_to_q[pos] = 0
    #     unwrap_heap(
    #         heap, flat_phase, flat_q, flat_to_q, height, width, uw_phase, connectivity
    #     )

    # If we had any postponed pixels set them to_q == 1 and re-run
    if len(other_heap) > 0:
        flat_to_q = np.abs(flat_to_q)
        unwrap_heap(
            other_heap, flat_phase, flat_q, flat_to_q, height, width, uw_phase, connectivity
        )


def quality_unwrap(
    phase: np.ndarray,
    quality: np.ndarray,
    seed: np.ndarray | tuple[int, int] | None = None,
    mask: np.ndarray | None = None,
    connectivity: Literal[8, 4] = 8,
):
    # quality is lowest => best
    assert -np.pi <= phase.min() <= np.pi
    assert -np.pi <= phase.max() <= np.pi
    assert phase.ndim == 2
    assert phase.shape == quality.shape
    img_shape = phase.shape
    assert connectivity in (8, 4)

    # Flat views and results array
    flat_quality = quality.ravel()
    flat_phase = phase.ravel()
    flat_uw_phase = phase.copy().ravel()  # result array
    if mask is None:
        flat_to_q = np.ones(flat_phase.shape, dtype=np.int8)
    else:
        assert mask.shape == img_shape
        flat_to_q = mask.astype(np.int8).ravel()
        # need to do this to ensure auto seed point is in mask
        flat_quality[~flat_to_q] = np.inf

    # Initialise heap
    if isinstance(seed, np.ndarray):
        # If the seed mask has disjoint regions then only
        # the one containing the best quality pixel is unwrapped first,
        # the remaining seed regions are converted to normal pixels
        flat_seed = seed.astype(bool).ravel()
        inv_seed = ~flat_seed
        # This puts other pixels into a 'postponed' state which
        # means they are processed on the second pass
        flat_to_q[inv_seed] = -1
        (nonzero_seed,) = np.nonzero(flat_seed)
        pos = nonzero_seed[np.argmin(flat_quality[flat_seed])]
    elif seed is None:
        pos = np.argmin(flat_quality)
    else:
        pos = np.ravel_multi_index(seed, img_shape)

    heap = [(flat_quality[pos], pos, pos)]
    assert flat_to_q[pos] > 0, "Starting point must be in mask"
    flat_to_q[pos] = 0  # first position already in q

    unwrap_heap(
        heap,
        flat_phase,
        flat_quality,
        flat_to_q,
        *img_shape,
        flat_uw_phase,
        connectivity,
    )

    return flat_uw_phase.reshape(img_shape)
