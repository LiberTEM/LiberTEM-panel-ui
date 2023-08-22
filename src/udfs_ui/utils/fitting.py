from __future__ import annotations
import numpy as np
from scipy.interpolate import RectBivariateSpline


def polyfit_array(array: np.ndarray,
                  degree: int = 2,
                  mask: np.ndarray = None,
                  **regression_kwargs) -> np.ndarray:
    """
    Fits and predicts a polynomial to (an optional subset of) an array

    If mask is provided restrict the fitting to its True pixels, but
    the prediction is always returned over the whole array

    Parameters
    ----------
    array : np.ndarray
        The array to fit, 2-D (height, width) or 3-D (height, width, components)
    degree : int, optional
        Te degree of the fitted Polynomial, by default 2
    mask : np.ndarray, optional
        A boolean mask of pixels to use in the fit, by default None
        in which case the whole array is used for fitting.
    regression_kwargs : dict, optional
        Keyword arguments passed to :code:`sklearn.linear_model.Ridge`

    Returns
    -------
    np.ndarray
        The fitted array with same shape as the input array
    """
    # Delay import to save import time if this is unused
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler
    from sklearn.pipeline import make_pipeline

    squeeze = False
    if array.ndim == 2:
        array = array[..., np.newaxis]
        squeeze = True

    assert array.ndim == 3
    h, w, c = array.shape

    valid_mask = np.isfinite(array).all(axis=-1)
    if mask is None:
        mask = valid_mask
    else:
        mask = np.logical_and(valid_mask, mask)
    mask = mask.ravel()

    # For full array
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))
    coords = np.stack((yy, xx), axis=-1).reshape(-1, 2)
    array = array.reshape(-1, c)

    model = make_pipeline(StandardScaler(), PolynomialFeatures(degree), Ridge(**regression_kwargs))
    model.fit(coords[mask, :], array[mask, :])
    components = model.predict(coords).reshape(h, w, c)

    if squeeze:
        components = components.squeeze(axis=-1)
    return components


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
