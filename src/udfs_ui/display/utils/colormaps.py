from __future__ import annotations
import functools
import warnings
import bokeh.colors as bc
import bokeh.palettes as bp


def hex2rgb(hex_color) -> bc.RGB:
    _hex = hex_color.replace('#', '')
    comps = tuple(int(_hex[2 * i:2 * (i + 1)], base=16) for i in range(len(_hex) // 2))
    rgb = bc.RGB(*comps)
    return rgb


def hex2hsv(hex_color):
    return hex2rgb(hex_color).to_hsl()


def hue_shift(hex_color, degrees):
    hsv = hex2hsv(hex_color)
    hsv.h = (hsv.h + degrees) % 360
    return hsv.to_rgb().to_hex()


@functools.lru_cache(1)
def _default_colormaps():
    import colorcet as cc
    # Necessary because Bokeh HSL class is being deprecated
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        image_colormaps = {}
        image_colormaps['Greys'] = cc.gray
        image_colormaps['Viridis'] = bp.Viridis256
        image_colormaps['Cividis'] = bp.Cividis256
        image_colormaps['Blues'] = cc.blues  # bp.Blues[256]
        # This mapping could be vectorized if used a lot
        # To map 256 colors takes about 5 ms but is only done once at startup
        blue_to_red = functools.partial(hue_shift, degrees=150)
        image_colormaps['Reds'] = [*map(blue_to_red, cc.blues)]
        image_colormaps['Greens'] = bp.Greens[256]
        image_colormaps['Oranges'] = bp.Oranges[256]
        image_colormaps['Diverging Blue/Red'] = cc.coolwarm
        image_colormaps['Spectrum'] = bp.Turbo256
    return image_colormaps


@functools.lru_cache(10)
def get_colormap(name, inverted=False):
    cmap = _default_colormaps()[name]
    if inverted:
        cmap = [*reversed(cmap)]
    return cmap


def available_colormaps():
    return [*_default_colormaps().keys()]


def default_image_cmap():
    return get_colormap(default_image_cmap_name())


def default_image_cmap_name():
    return 'Greys'


def get_bokeh_palette(name='Category10_10') -> bp.Palette:
    return getattr(bp, name, bp.Category10_10)
