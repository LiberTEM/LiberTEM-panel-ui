from __future__ import annotations
import functools
import warnings
import colorcet as cc
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


# Necessary because Bokeh HSL class is being deprecated
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    image_colormaps = {}
    image_colormaps['Greys'] = cc.gray
    image_colormaps['Viridis'] = bp.Viridis256
    image_colormaps['Cividis'] = bp.Cividis256
    image_colormaps['Blues'] = cc.blues
    blue_to_red = functools.partial(hue_shift, degrees=150)
    image_colormaps['Reds'] = [*map(blue_to_red, cc.blues)]
    image_colormaps['Greens'] = bp.Greens[256]
    image_colormaps['Oranges'] = bp.Oranges[256]
    image_colormaps['Diverging Blue/Red'] = cc.coolwarm
    image_colormaps['Cyclic Isoluminant'] = cc.cyclic_isoluminant
    image_colormaps['Isoluminant'] = cc.isolum
    image_colormaps['Spectrum'] = bp.Turbo256
    image_colormaps['Temperature'] = (
        '#000000', '#000000', '#000000', '#000005', '#00000a', '#00000f',
        '#000014', '#000019', '#00001e', '#000023', '#000028', '#00002d',
        '#000032', '#000037', '#00003c', '#000041', '#000046', '#00004b',
        '#000050', '#000055', '#00005a', '#00005f', '#000064', '#000069',
        '#00006e', '#000073', '#000078', '#00007d', '#000082', '#000087',
        '#00008c', '#000091', '#000096', '#00009b', '#0000a0', '#0000a5',
        '#0000aa', '#0000af', '#0000b4', '#0000b9', '#0000be', '#0000c3',
        '#0000c8', '#0000cd', '#0000d2', '#0000d7', '#0000dc', '#0000e1',
        '#0000e6', '#0000eb', '#0000f0', '#0000f5', '#0000fa', '#0000ff',
        '#0003fa', '#0007f5', '#000af0', '#000deb', '#0011e6', '#0014e1',
        '#0017dc', '#001bd7', '#001ed2', '#0021cd', '#0025c8', '#0028c3',
        '#002bbe', '#002fb9', '#0032b4', '#0035af', '#0039aa', '#003ca5',
        '#003fa0', '#00439b', '#004696', '#004991', '#004d8c', '#005087',
        '#005382', '#00577d', '#005a78', '#005d73', '#00616e', '#006469',
        '#006764', '#006b5f', '#007155', '#007550', '#00784b', '#007b46',
        '#007f41', '#00823c', '#008537', '#008932', '#008c2d', '#008f28',
        '#009323', '#00961e', '#009919', '#009d14', '#00a00f', '#00a30a',
        '#00a705', '#00aa00', '#05a700', '#0aa300', '#0fa000', '#149d00',
        '#199900', '#1e9600', '#239300', '#288f00', '#2d8c00', '#328900',
        '#378500', '#3c8200', '#417f00', '#467b00', '#4b7800', '#507500',
        '#557100', '#5a6e00', '#5f6b00', '#646700', '#696400', '#6e6100',
        '#735d00', '#785a00', '#7d5700', '#825300', '#875000', '#8c4d00',
        '#914900', '#964600', '#9b4300', '#a03f00', '#a53c00', '#aa3900',
        '#af3500', '#b43200', '#b92f00', '#be2b00', '#c32800', '#c82500',
        '#cd2100', '#d21e00', '#d71b00', '#dc1700', '#e11400', '#e61100',
        '#eb0d00', '#f00a00', '#f50700', '#fa0300', '#ff0000', '#ff0500',
        '#ff0a00', '#ff0f00', '#ff1400', '#ff1900', '#ff1e00', '#ff2300',
        '#ff2800', '#ff2d00', '#ff3200', '#ff3700', '#ff3c00', '#ff4100',
        '#ff4600', '#ff4b00', '#ff5500', '#ff5a00', '#ff5f00', '#ff6400',
        '#ff6900', '#ff6e00', '#ff7300', '#ff7800', '#ff7d00', '#ff8200',
        '#ff8700', '#ff8c00', '#ff9100', '#ff9600', '#ff9b00', '#ffa000',
        '#ffa500', '#ffaa00', '#ffaf00', '#ffb400', '#ffb900', '#ffbe00',
        '#ffc300', '#ffc800', '#ffcd00', '#ffd200', '#ffd700', '#ffdc00',
        '#ffe100', '#ffe600', '#ffeb00', '#fff000', '#fff500', '#fffa00',
        '#ffff00', '#ffff05', '#ffff0a', '#ffff0f', '#ffff14', '#ffff19',
        '#ffff1e', '#ffff23', '#ffff28', '#ffff2d', '#ffff32', '#ffff37',
        '#ffff3c', '#ffff41', '#ffff46', '#ffff4b', '#ffff50', '#ffff55',
        '#ffff5a', '#ffff5f', '#ffff64', '#ffff69', '#ffff6e', '#ffff73',
        '#ffff78', '#ffff7d', '#ffff82', '#ffff87', '#ffff8c', '#ffff91',
        '#ffff96', '#ffff9b', '#ffffa0', '#ffffa5', '#ffffaa', '#ffffaf',
        '#ffffb4', '#ffffb9', '#ffffbe', '#ffffc3', '#ffffc8', '#ffffcd',
        '#ffffd2', '#ffffd7', '#ffffdc', '#ffffe1', '#ffffe6', '#ffffeb',
        '#fffff0', '#fffff5', '#ffffff', '#ffffff',
    )


def _default_colormaps():
    return image_colormaps


@functools.lru_cache(10)
def get_colormap(name: str, inverted: bool = False):
    cmaps_lower = {k.lower(): v for k, v in _default_colormaps().items()}
    cmap = cmaps_lower[name.lower()]
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
