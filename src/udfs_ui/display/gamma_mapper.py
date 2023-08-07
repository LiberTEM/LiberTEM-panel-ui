import pathlib

from bokeh.core.properties import Float
from bokeh.models.mappers import LinearColorMapper


class GammaColorMapper(LinearColorMapper):
    __implementation__ = str(pathlib.Path(__file__).parent / "gamma_mapper.ts")

    gamma = Float(default=0.0)
