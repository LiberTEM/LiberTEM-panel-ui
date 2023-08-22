from __future__ import annotations
from collections import namedtuple

from ...utils import random_hash


def slider_step_size(start, end, n=300):
    return abs(end - start) / n


def get_random_name(glyph_base):
    return f'{glyph_base.__class__.__name__}-{random_hash()}'


BokehToPanelEvent = namedtuple('BokehToPanelEvent', ['old', 'new'])
