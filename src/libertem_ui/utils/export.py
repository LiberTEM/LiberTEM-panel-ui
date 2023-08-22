from __future__ import annotations
import pathlib
import collections
import tifffile
import numpy as np
import skimage.io as skio
import skimage.util as sku
import skimage.exposure as ske


def encode_val(val, writeonce=True):
    if isinstance(val, (float, np.float32, np.float64)):
        _val = np.float64(val).tobytes()
        type_code = tifffile.TIFF.DATATYPES.DOUBLE.value
        count = 1
    elif isinstance(val, (int, np.int32, np.int64)):
        _val = np.int32(val).tobytes()
        type_code = tifffile.TIFF.DATATYPES.SLONG.value
        count = 1
    elif isinstance(val, str):
        type_code = tifffile.TIFF.DATATYPES.ASCII.value
        _val = val.encode('ascii')
        count = len(_val)
    else:
        raise ValueError(f'{val} - {type(val)}')
    return type_code, count, _val, writeonce


def dm_tags(stack, *, xy_unit='', xy_scale=1., xy_origin=0.,
            stack_unit='', stack_scale=1., stack_origin=0.,
            intensity_os=None, intensity_unit=None):
    if isinstance(xy_unit, str):
        xy_unit = (xy_unit, xy_unit)
    xy_unit_short, xy_unit_long = xy_unit
    if isinstance(xy_scale, (collections.abc.Sequence, np.ndarray)):
        xscale, yscale = xy_scale
    else:
        xscale, yscale = xy_scale, xy_scale
    if isinstance(xy_origin, (collections.abc.Sequence, np.ndarray)):
        xorigin, yorigin = xy_origin
    else:
        xorigin, yorigin = xy_origin, xy_origin

    tags = {65003: str(xy_unit_short),  # hor dim unit shortname
            65004: str(xy_unit_short),  # ver dim unit shortname
            65012: str(xy_unit_long),  # hor dim unit longname
            65013: str(xy_unit_long),  # ver dim unit longname
            65006: np.float64(xorigin),  # hor dim left origin
            65007: np.float64(yorigin),  # ver dim top origin
            65009: np.float64(xscale),  # hor dim scale
            65010: np.float64(yscale)}  # ver dim scale
    if stack:
        tags[65005] = str(stack_unit)
        tags[65014] = str(stack_unit)
        tags[65008] = np.float64(stack_origin)
        tags[65011] = np.float64(stack_scale)
    if intensity_os is not None:
        intensity_origin, intensity_scale = intensity_os
        tags[65024] = np.float64(intensity_origin)  # intensity origin
        tags[65025] = np.float64(intensity_scale)  # intensity scale
    if intensity_unit is not None:
        tags[65022] = str(intensity_unit)  # intensity units
        tags[65023] = str(intensity_unit)  # intensity units
    return [(code, *encode_val(value)) for code, value in tags.items()]


def write_dm_tiff(fpath, array, stack_dimension=-1, verbose=True, dry_run=False, **kwargs):
    fpath = pathlib.Path(fpath).absolute()
    assert fpath.suffix == '.tif', 'Must use .tif suffix on filepath'

    if array.ndim > 3:
        ValueError("Saving >3D images as tif not yet supported")
    elif array.ndim < 2:
        ValueError("Cannot save 1D arrays as tif")

    supported_dtypes = [np.uint8, np.uint16, np.uint32,
                        np.int8, np.int16, np.int32,
                        np.float32]
    unsupported_dtypes = {
        np.int64: np.int32,
        np.float64: np.float32,
        np.float16: np.float32,
        bool: np.uint8,
        float: np.float32,
        int: np.int32,
        complex: np.float32,
        np.complex128: np.float32,
        np.complex64: np.float32
    }
    dtype_mapper = {**unsupported_dtypes, **{k: k for k in supported_dtypes}}
    dtype_mapper = {np.dtype(k): np.dtype(v) for k, v in dtype_mapper.items()}

    source_dtype = np.dtype(array.dtype)
    target_dtype = dtype_mapper.get(source_dtype, None)
    if target_dtype is None:
        raise ValueError(f'Cannot save image of type {source_dtype} '
                         'in a DM-compatible format.')
    elif np.issubdtype(source_dtype, complex):
        assert array.ndim == 2, ('Cannot convert stacked complex images '
                                 'to a sensible DM format. Save array.real '
                                 'and array.imag separately.')
        print('Complex tif images cannot be read by DM, '
              'saving real and imaginary parts as a 2-stack '
              f'of type {target_dtype}. This may incurr a loss of precision.')
        array = np.stack((array.real, array.imag), axis=-1)
        stack_dimension = -1
    elif verbose and source_dtype != target_dtype:
        print(f'Casting from {source_dtype} to {target_dtype} '
              'while saving for DM; this may incurr a loss of precision.')
    if dry_run:
        return
    _array = array.astype(target_dtype)
    is_stack = _array.ndim == 3
    tags = dm_tags(is_stack, **kwargs)

    if array.ndim == 3:
        stack_length = _array.shape[stack_dimension]
        arrays = np.split(_array, stack_length, axis=stack_dimension)
        arrays = [a.squeeze(axis=stack_dimension) for a in arrays]
    else:
        arrays = [_array]

    with tifffile.TiffWriter(fpath) as tif:
        for _array in arrays:
            tif.write(_array, photometric='minisblack', extratags=tags)
    return fpath


def save_npy(filepath, result, check=False):
    assert isinstance(result, np.ndarray)
    if check:
        return
    np.save(filepath, result)


def save_tif(filepath, result, check=False):
    assert isinstance(result, np.ndarray)
    write_dm_tiff(filepath, result, dry_run=True)
    if check:
        return
    write_dm_tiff(filepath, result)


def save_skimg(filepath, result, check=False):
    assert isinstance(result, np.ndarray)
    assert result.ndim in (2, 3)
    if result.ndim == 3:
        assert result.shape[-1] in [3, 4]
    if check:
        return
    img_scaled = ske.rescale_intensity(result.astype(np.float64))
    img_cast = sku.img_as_ubyte(img_scaled)
    skio.imsave(filepath, img_cast)


image_save_functions = {
    '.tif': save_tif,
    '.png': save_skimg,
    '.jpg': save_skimg,
    '.npy': save_npy,
}
