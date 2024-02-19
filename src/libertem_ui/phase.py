from __future__ import annotations
import warnings
from typing import Tuple, NamedTuple
from dataclasses import dataclass
import numpy as np

from .strain_decomposition import compute_strain_large_def, StrainResult


def get_pos(
    idxs: np.ndarray, g1: complex, g2: complex, as_complex: bool = False
) -> np.ndarray[float]:
    g1 = np.asarray([[g1.imag, g1.real]])
    g2 = np.asarray([[g2.imag, g2.real]])
    pos_g1 = np.asarray(idxs).reshape(-1, 2)[:, 0][..., np.newaxis] * g1
    pos_g2 = np.asarray(idxs).reshape(-1, 2)[:, 1][..., np.newaxis] * g2
    array_form = pos_g1 + pos_g2
    if as_complex:
        return array_form[:, 1] + array_form[:, 0] * 1j
    else:
        return array_form


def get_search_grid(
    array_shape: Tuple[int, int],
    g1: complex,
    g2: complex,
    max_g1: int = -1,
    max_g2: int = -1,
    border: int = -1,
    full: bool = True,
    exclude_zero: bool = False,
):
    # could extend this with a centre param!
    if g1 is None or g2 is None or np.isnan(g1) or np.isnan(g2):
        return np.empty((0, 2), dtype=int)
    h2, w2 = np.asarray(array_shape) / 2.
    diagonal = np.linalg.norm(array_shape) / 2
    len_g1 = np.abs(g1)
    len_g2 = np.abs(g2)
    if np.isclose(len_g1, 0.) or np.isclose(len_g2, 0.):
        return np.empty((0, 2), dtype=int)
    if border < 0:
        border = max(len_g1, len_g2) / 2.
    if max_g1 < 0:
        max_g1 = int(np.ceil(diagonal / len_g1))
    if max_g2 < 0:
        max_g2 = int(np.ceil(diagonal / len_g2))
    search_grid = np.mgrid[-max_g1:max_g1 + 1, -max_g2: max_g2 + 1].reshape(2, -1).T
    search_pos = get_pos(search_grid, g1, g2)
    valid_mask = np.ones(search_pos.shape[:1], dtype=bool)
    if exclude_zero:
        valid_mask = np.logical_and(valid_mask, np.abs(search_pos).sum(axis=1) > 0)
    # Note search_pos is centred on (0, 0) by default
    # x in bounds
    if full:
        valid_mask = np.logical_and(
            valid_mask, search_pos[:, 1] >= min(0., -w2 + border)
        )
    else:
        # right side of autocorr frame only!
        valid_mask = np.logical_and(valid_mask, search_pos[:, 1] >= 0)
    valid_mask = np.logical_and(
        valid_mask, search_pos[:, 1] <= max(0., w2 - border - 1)
    )
    # y in bounds
    valid_mask = np.logical_and(
        valid_mask, search_pos[:, 0] >= min(0., -h2 + border)
    )
    valid_mask = np.logical_and(
        valid_mask, search_pos[:, 0] <= max(0., h2 - border - 1)
    )
    return search_grid[valid_mask, :]


@dataclass
class Phase:
    g1: complex
    g2: complex
    ref_idx: int | None = None
    centre: complex | None = None
    label: str | None = None
    ref_region: tuple[slice | int, ...] | None = None
    fixed_ref: tuple[complex, complex] | None = None

    def search_grid(self, frame_shape: tuple[int, int]) -> np.ndarray[int]:
        return get_search_grid(frame_shape, self.g1, self.g2)

    def __eq__(self, other: Phase):
        try:
            return (
                self.g1 == other.g1
                and self.g2 == other.g2
                and self.centre == other.centre
                and self.ref_idx == other.ref_idx
            )
        except AttributeError:
            return False

    def set_ref_region(self, ref_region: tuple[slice | int, ...] | None):
        self.ref_region = ref_region

    def set_fixed_ref(
        self,
        ref_g1: complex,
        ref_g2: complex,
    ):
        self.fixed_ref = (ref_g1, ref_g2)

    def get_ref(
        self,
        g1: np.ndarray[complex],
        g2: np.ndarray[complex],
        mask: np.ndarray[bool] = None,
    ) -> tuple[complex, complex]:
        if self.fixed_ref is not None:
            return self.fixed_ref
        elif self.ref_region is not None:
            reduced_g1 = g1[self.ref_region].ravel()
            reduced_g2 = g2[self.ref_region].ravel()
            if mask is not None:
                reduced_mask = mask[self.ref_region].ravel()
                reduced_g1 = reduced_g1[reduced_mask]
                reduced_g2 = reduced_g2[reduced_mask]
            # nanmedian of empty returns nan
            g1ref = np.nanmedian(reduced_g1)
            g2ref = np.nanmedian(reduced_g2)
            if not (np.isnan(g1ref) or np.isnan(g2ref)):
                return g1ref, g2ref
            warnings.warn('ref_region provided a NaN reference, trying ref_idx')
        if self.ref_idx is not None:
            g1ref = g1.flat[self.ref_idx]
            g2ref = g2.flat[self.ref_idx]
            if not (np.isnan(g1ref) or np.isnan(g2ref)):
                return g1ref, g2ref
            warnings.warn('ref_idx provided a NaN reference, returning g1/g2')
        return self.g1, self.g2

    def ref_is_masked(self, mask: np.ndarray) -> bool:
        if self.ref_region is not None:
            reduced_mask = mask[self.ref_region].ravel()
        elif self.ref_idx is not None:
            reduced_mask = mask.flat[self.ref_idx]
        else:
            raise ValueError('No reference associated with this Phase')
        return reduced_mask.sum() == 0

    def is_crystal(self) -> bool:
        return True

    def compute_strain(
        self,
        g1fit: np.ndarray[complex],
        g2fit: np.ndarray[complex],
    ) -> StrainResult:
        return compute_strain_large_def(
            g1fit,
            g2fit,
            *self.get_ref(
                g1fit,
                g2fit,
            )
        )


class AmorphousPhase(Phase):
    def is_crystal(self) -> bool:
        # Exists to avoid an isinstance check
        return False

    def compute_strain(
        self,
        g1fit: np.ndarray[complex],
        g2fit: np.ndarray[complex],
    ) -> StrainResult:
        raise RuntimeError('Cannot compute strain for AmorphousPhase')


class PhaseMap(NamedTuple):
    idx_map: np.ndarray[int]
    phases: tuple[Phase | AmorphousPhase, ...]
    # value of the max correlation pixel, per-phase ax0
    max_val: np.ndarray | None = None
    # position of the max correlation pixel, per-phase ax0, in frame coords
    max_pos: np.ndarray | None = None

    @classmethod
    def for_single_phase(
        cls,
        nav_shape: Tuple[int, int],
        phase: Phase,
        max_val: np.ndarray | None = None,
        max_pos: np.ndarray | None = None,
    ):
        if isinstance(phase, Phase):
            phase = (phase,)
        return cls(
            idx_map=np.zeros(nav_shape, dtype=int),
            phases=phase,
            max_val=max_val,
            max_pos=max_pos,
        )

    def set_ref_region(
        self,
        region: tuple[slice | int, ...] | None,
        phase: Phase | None = None
    ):
        which = slice(None)
        if phase is not None:
            phase_idx = self.get_phase_idx(phase)
            which = slice(phase_idx, phase_idx + 1)
        for phase in self.phases[which]:
            phase.set_ref_region(region)

    def valid_phase_references(
        self,
        mask: np.ndarray[bool] | None,
    ) -> list[int]:
        if mask is None:
            return list(range(len(self.phases)))
        valid = []
        for idx, phase in enumerate(self.phases):
            if isinstance(phase, AmorphousPhase):
                # amporphous is always valid
                valid.append(idx)
                continue
            if not phase.ref_is_masked(mask):
                valid.append(idx)
        return valid

    def get_references(
        self,
        g1fit: np.ndarray[complex],
        g2fit: np.ndarray[complex],
    ) -> list[tuple[complex, complex] | tuple[None, None]]:
        refs = []
        for idx, phase in enumerate(self.phases):
            if isinstance(phase, AmorphousPhase):
                refs.append((None, None))
                continue
            phase_mask = self.idx_map == idx
            phase: Phase
            refg1, refg2 = phase.get_ref(
                g1fit,
                g2fit,
                phase_mask,
            )
            refs.append((refg1, refg2))
        return refs

    def compute_strain(
        self,
        g1fit: np.ndarray[complex],
        g2fit: np.ndarray[complex],
    ) -> StrainResult:
        full_strain = StrainResult.empty(g1fit.shape)
        references = self.get_references(g1fit, g2fit)

        for idx, (refg1, refg2) in enumerate(references):
            if refg1 is None or refg2 is None:
                continue
            phase_mask = self.idx_map == idx
            phase_strain = compute_strain_large_def(
                g1fit[phase_mask],
                g2fit[phase_mask],
                refg1,
                refg2,
            )
            full_strain.update_masked(
                phase_mask,
                phase_strain,
                inplace=True,
            )

        return full_strain

    def _phase_mask(self, idx):
        return self.idx_map == idx

    def get_phase_idx(self, phase: str | Phase) -> int:
        if isinstance(phase, str):
            phase = self.get_phase_by_label(phase)
        for idx, p in enumerate(self.phases):
            if p == phase:
                return idx
        raise RuntimeError('Phase does not exist in phase map')

    def get_phase_by_label(self, label: str) -> Phase:
        assert isinstance(label, str)
        labels = tuple(p.label for p in self.phases)
        if label not in labels:
            raise RuntimeError(f'Label {label} not in available Phases')
        if sum(lab == label for lab in labels if lab is not None) > 1:
            raise RuntimeError(f'Label {label} associated with more than one Phase')
        for p in self.phases:
            if p.label == label:
                return p
        raise RuntimeError

    def get_crystal_mask(self) -> np.ndarray | None:
        """
        Return mask where phase is not Amorphous
        If no amorphous return None (LiberTEM optimises non-ROI case)
        """
        roi = np.ones(self.idx_map.shape, dtype=bool)
        has_amorphous = False
        for idx, phase in enumerate(self.phases):
            if phase.is_crystal():
                continue
            has_amorphous = True
            phase_mask = self.idx_map == idx
            roi[phase_mask] = False
        if has_amorphous:
            return roi
        return None

    def get_absolute_positions(self, frame_shape: Tuple[int, int], max_only: bool = True):
        if self.max_pos is None:
            raise RuntimeError('Phase map needs max_pos result to compute absolute centres')
        assert self.max_pos.shape[0] == len(self.phases), 'first dimension should be phases!'
        h, w = frame_shape
        nav_dims = len(self.max_pos.shape[1:])
        im_centre = (w / 2) + (h / 2) * 1j
        phase_centres = [p.centre for p in self.phases]
        if any(c is None for c in phase_centres):
            raise RuntimeError('All phases need centres to get absolute positions')
        phase_centres = np.asarray(phase_centres).reshape(-1, *((1,) * nav_dims))
        abs_pos = phase_centres + self.max_pos - im_centre
        if max_only:
            max_idxs = self.idx_map[np.newaxis, ...]
            return np.take_along_axis(abs_pos, max_idxs, axis=0)[0, ...]
        else:
            return abs_pos

    def get_max_corr(self):
        if self.max_val is None:
            raise RuntimeError('Phase map needs corr scores to return max values')
        return np.take_along_axis(self.max_val, self.idx_map[np.newaxis, ...], 0)[0, ...]
