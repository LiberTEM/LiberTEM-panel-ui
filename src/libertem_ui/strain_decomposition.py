from __future__ import annotations
import numpy as np
from typing import NamedTuple


class StrainResult(NamedTuple):
    e_xx: float | np.ndarray
    e_xy: float | np.ndarray
    e_yy: float | np.ndarray
    theta: float | np.ndarray
    rotation: float

    @property
    def rotation_deg(self):
        return np.rad2deg(self.rotation)

    @property
    def theta_deg(self):
        return np.rad2deg(self.theta)

    @classmethod
    def empty(cls, shape, dtype=float):
        return cls(
            np.full(shape, np.nan, dtype=dtype),
            np.full(shape, np.nan, dtype=dtype),
            np.full(shape, np.nan, dtype=dtype),
            np.full(shape, np.nan, dtype=dtype),
            0.
        )

    def update_masked(
        self,
        mask: np.ndarray[bool],
        other: StrainResult,
        inplace: bool = False
    ) -> StrainResult:
        this = self
        if not inplace:
            this = this.copy()
        return this._update_masked(mask, other)

    def _update_masked(
        self,
        mask: np.ndarray[bool],
        other: StrainResult
    ) -> StrainResult:
        assert self.rotation == other.rotation
        self.e_xx[mask] = other.e_xx
        self.e_xy[mask] = other.e_xy
        self.e_yy[mask] = other.e_yy
        self.theta[mask] = other.theta
        return self

    def copy(self) -> StrainResult:
        return StrainResult(
            self.e_xx.copy(),
            self.e_xy.copy(),
            self.e_yy.copy(),
            self.theta.copy(),
            self.rotation
        )

    def rotate_rad(self, rotation: float, absolute: bool = False) -> StrainResult:
        if absolute:
            return self.to_axis_aligned().rotate_rad(rotation)
        else:
            _e_xx, _e_xy, _e_yy = _rotate_strain(
                self.e_xx,
                self.e_xy,
                self.e_yy,
                rotation,
            )
            try:
                theta = self.theta.copy()
            except AttributeError:
                theta = self.theta
            return StrainResult(
                _e_xx,
                _e_xy,
                _e_yy,
                theta,
                self.rotation + rotation,
            )

    def rotate_deg(self, rotation: float, absolute: bool = False):
        rotation_rad = np.deg2rad(rotation)
        return self.rotate_rad(rotation_rad, absolute=absolute)

    def to_axis_aligned(self) -> StrainResult:
        if np.allclose(self.rotation, 0.):
            return self
        return self.rotate_rad(-1 * self.rotation)

    def to_vector(self, vec_yx: tuple[float, float] | complex):
        # NOTE Noticed some strange behaviour here when
        # rotating from an axis-aligned to a vector-aligned
        vec_complex = self._tuple_to_complex(vec_yx)
        rotation_rad = np.angle(vec_complex)
        return self.rotate_rad(rotation_rad, absolute=True)

    @staticmethod
    def _tuple_to_complex(vec_yx: tuple[float, float] | complex):
        if not np.issubdtype(np.asarray(vec_yx).dtype, complex):
            vec_complex = vec_yx[1] + vec_yx[0] * 1j
            return vec_complex
        return vec_yx


def _rotate_strain(e_xx, e_xy, e_yy, rotation_rad):
    if np.allclose(rotation_rad, 0.):
        return e_xx, e_xy, e_yy

    # JLR rotate implementation from GEM_ED
    theta = rotation_rad
    _2theta = 2 * theta
    costheta2 = (1 + np.cos(_2theta)) / 2
    sintheta2 = (1 - np.cos(_2theta)) / 2
    sin2theta = np.sin(_2theta)
    cos2theta = np.cos(_2theta)

    _e_xx = e_xx * costheta2 + e_xy * sin2theta + e_yy * sintheta2
    _e_xy = (e_yy - e_xx) * sin2theta / 2 + e_xy * cos2theta
    _e_yy = e_yy * costheta2 - e_xy * sin2theta + e_xx * sintheta2

    return _e_xx, _e_xy, _e_yy


def compute_strain_large_def(
    g1: np.ndarray[complex],
    g2: np.ndarray[complex],
    g1_ref: complex,
    g2_ref: complex
) -> StrainResult:

    scalar = False
    if np.isscalar(g1):
        g1 = np.asarray([g1])
        scalar = True
    if np.isscalar(g2):
        g2 = np.asarray([g2])
        assert scalar, 'Cannot handle mixed scalar / array inputs'

    output_shape = g1.shape
    gx1 = g1.real.ravel()
    gy1 = g1.imag.ravel()
    gx2 = g2.real.ravel()
    gy2 = g2.imag.ravel()

    gx1_0, gy1_0 = g1_ref.real, g1_ref.imag
    gx2_0, gy2_0 = g2_ref.real, g2_ref.imag

    det = gx1 * gy2 - gy1 * gx2
    d11 = (gx1_0 * gy2 - gx2_0 * gy1) / det
    d21 = (gx2_0 * gx1 - gx1_0 * gx2) / det
    d12 = (gy1_0 * gy2 - gy2_0 * gy1) / det
    d22 = (gy2_0 * gx1 - gy1_0 * gx2) / det

    aux = np.arctan2(d21 - d12, d11 + d22)
    jcos = np.cos(aux)
    jsin = np.sin(aux)

    exx = d11 * jcos + d21 * jsin - 1
    exy = d12 * jcos + d22 * jsin
    eyy = -d12 * jsin + d22 * jcos - 1
    rot = 0.5 * (d21 - d12)

    exx = exx.reshape(output_shape)
    exy = exy.reshape(output_shape)
    eyy = eyy.reshape(output_shape)
    rot = rot.reshape(output_shape)

    if scalar:
        exx = exx.item()
        exy = exy.item()
        eyy = eyy.item()
        rot = rot.item()

    return StrainResult(
        exx,
        exy,
        eyy,
        rot,
        0.
    )
