from __future__ import annotations
from typing import TYPE_CHECKING, NewType
from typing_extensions import Self, Literal
import numpy as np

import panel as pn
from bokeh.models import CustomJS
from bokeh.models.glyphs import MultiLine as BkMultiLine
from bokeh.models.sources import ColumnDataSource
from bokeh.models.tools import PolyEditTool, PolyDrawTool

from .display_base import DisplayBase, ConsBase, Text, VertexPointSetMixin
from .icons import line_icon


if TYPE_CHECKING:
    from bokeh.plotting import figure as BkFigure


class MultiLine(DisplayBase, VertexPointSetMixin):
    glyph_map = {
        'lines': BkMultiLine
    }

    def __init__(
        self,
        cds: ColumnDataSource,
        xs: str = 'xs',
        ys: str = 'ys',
    ):
        super().__init__(cds)
        glyph = BkMultiLine(
            xs=xs,
            ys=ys,
            line_color='red',
            line_width=2,
        )
        self._register_glyph('lines', glyph)

    @property
    def lines(self) -> BkMultiLine:
        return self._glyphs['lines'][0].glyph

    @classmethod
    def new(cls):
        return MultiLineCons()

    def update(
        self,
        xs: np.ndarray | list | None = None,
        ys: np.ndarray | list | None = None,
    ):
        data = {}
        data[self.lines.xs] = xs
        data[self.lines.ys] = ys
        return super().update(**data)

    def editable(
        self,
        *figs: BkFigure,
        tag_name: str = 'default',
        selected: bool = False,
    ) -> Self:

        def _make_draw_tool():
            return PolyDrawTool(
                name='Line Draw',
                description='Draw lines on figure',
                renderers=[],
                icon=line_icon(),
                tags=[tag_name],
            )

        where = self._add_to_tool(
            figs=figs,
            glyph_name='lines',
            tool_filter=lambda t: tag_name in t.tags and isinstance(t, PolyDrawTool),
            make_tool=_make_draw_tool,
            selected=selected,
        )
        self._setup_vertex_renderer(where)

        def _make_edit_tool():
            return PolyEditTool(
                name='Line Edit',
                description='Edit lines on figure',
                renderers=[],
                tags=[tag_name],
            )

        where = self._add_to_tool(
            figs=figs,
            glyph_name='lines',
            tool_filter=lambda t: tag_name in t.tags and isinstance(t, PolyEditTool),
            make_tool=_make_edit_tool,
        )
        self._setup_vertex_renderer(where)
        return self


class MultiLineCons(ConsBase):
    default_keys = ('xs', 'ys')

    @staticmethod
    def from_vectors(
        xs: np.ndarray,
        ys: np.ndarray,
    ) -> MultiLine:
        data = {
            k: v for k, v in zip(MultiLineCons.default_keys, (xs, ys))
        }
        cds = ColumnDataSource(data)
        return MultiLine(cds)

    @classmethod
    def empty(cls):
        return super().empty(MultiLine)


class VectorsOverlay(DisplayBase):
    """
    Adds two lines to the figure at a user-specified angle. Has functionality
    to interactively set the vector angle using a slider.
    """
    glyph_map = {
        'vectors': MultiLine,
        'labels': Text,
    }

    def __init__(self, cds: ColumnDataSource):
        super().__init__(cds)
        lines = MultiLine(cds)
        lines.lines.line_color = 'line_color'
        self._register_child('vectors', lines)

    @classmethod
    def new(cls):
        return VectorsOverlayCons()

    def with_labels(
        self,
        labels: tuple[str, str],
        length_mult: float = 1.15,
    ) -> Self:
        cx = self.cds.data['xs'][0][0]
        cy = self.cds.data['ys'][0][0]
        angles = self.cds.data['angle']
        lengths = self.cds.data['length']
        initial_pos_text = VectorsOverlayCons._vectors_init(
            cx,
            cy,
            angles,
            lengths,
            mult=length_mult,
        )
        self.cds.data.update({
            'text_x': [initial_pos_text['xs'][0][1], initial_pos_text['xs'][1][1]],
            'text_y': [initial_pos_text['ys'][0][1], initial_pos_text['ys'][1][1]],
            'length_mult': [length_mult] * 2,
            'labels': labels,
        })
        labels = Text(
            self.cds,
            x='text_x',
            y='text_y',
            text='labels',
        )
        labels.glyph.text_color = 'line_color'
        labels.glyph.text_baseline = 'middle'
        labels.glyph.text_align = 'center'
        self._register_child('labels', labels)
        return self

    def refresh(self):
        # FIXME sync issues!!!
        cx = self.cds.data['xs'][0][0]
        cy = self.cds.data['ys'][0][0]
        self.move_centre(cx, cy)

    def move_centre(self, cx: float, cy: float):
        new_data = self._new_coords_for_centre(cx, cy)
        self.update(**new_data)

    def _new_coords_for_centre(self, cx: float, cy: float):
        angles = self.cds.data['angle']
        lengths = self.cds.data['length']
        new_data = VectorsOverlayCons._vectors_init(
            cx,
            cy,
            angles,
            lengths,
        )
        if 'labels' in self.cds.data.keys():
            length_mult = self.cds.data['length_mult'][0]
            pos_text = VectorsOverlayCons._vectors_init(
                cx,
                cy,
                angles,
                lengths,
                mult=length_mult,
            )
            new_data['text_x'] = [pos_text['xs'][0][1], pos_text['xs'][1][1]]
            new_data['text_y'] = [pos_text['ys'][0][1], pos_text['ys'][1][1]]
        return new_data

    def flip_dir(self, dir: Literal['x', 'y']):
        if dir not in ('x', 'y'):
            raise ValueError("Direction either 'x' or 'y'")
        idx = 0 if dir == 'x' else 1
        current_length = self.cds.data['length'][idx]
        self.cds.patch({
            'length': [(idx, -1 * current_length)],
        })
        try:
            # HACK !!!
            # Cannot trigger the JSCallback directly
            current_dir = np.sign(current_length)
            self._rotation_slider.param.update(
                value=self._rotation_slider.value + (current_dir * 1e-3)
            )
        except AttributeError:
            # FIXME Will still bug-out due to JSCallback of .follow_point
            self.refresh()

    def follow_point(
        self,
        source_cds: ColumnDataSource,
        index: int = 0,
        xkey: str = 'cx',
        ykey: str = 'cy',
    ):
        cx = source_cds.data[xkey][index]
        cy = source_cds.data[ykey][index]
        self.move_centre(cx, cy)
        callback = CustomJS(
            args={
                'xkey': xkey,
                'ykey': ykey,
                'index': index,
                'vector_source': self.cds,
            },
            code=self._move_centre_js(),
        )
        source_cds.js_on_change(
            'data',
            callback,
        )

    @staticmethod
    def _move_centre_js():
        return """
const new_centre_x = cb_obj.data[xkey][index]
const new_centre_y = cb_obj.data[ykey][index]
const old_centre_x = vector_source.data.xs[0][0]
const old_centre_y = vector_source.data.ys[0][0]

const dx = new_centre_x - old_centre_x
const dy = new_centre_y - old_centre_y

vector_source.data.xs = vector_source.data.xs.map(xs => xs.map(x => x + dx));
vector_source.data.ys = vector_source.data.ys.map(ys => ys.map(y => y + dy));

if (vector_source.data.hasOwnProperty('labels')) {
    vector_source.data.text_x = vector_source.data.text_x.map(x => x + dx);
    vector_source.data.text_y = vector_source.data.text_y.map(y => y + dy);
}
vector_source.change.emit()
"""

    def with_rotation(self, label='Rotation', direction: Literal[1, -1] = 1):
        self._rotation_slider = pn.widgets.FloatSlider(
            name=label,
            # FIXME make this wrap to the correct range
            value=self.cds.data['angle'][0],
            start=-180.,
            end=180.,
        )
        if direction not in (-1, 1):
            raise ValueError('Rotation direction must be one of -1, 1')

        args = {
            'vector_source': self.cds,
            'direction': direction,
        }

        args['angle_slider'] = self._rotation_slider
        has_text = 'labels' in self.cds.data.keys()
        rotation_code = (
            self._vectors_cb_rotation_js()
            + self._vectors_cb_base_js(with_emit=(not has_text))
        )
        if has_text:
            rotation_code = rotation_code + self._text_cb_js()
        self._rotation_slider.jscallback(
            args,
            value=rotation_code,
        )
        self._rotation_slider.param.watch(
            self._sync_angle,
            'value_throttled',
        )
        return self._rotation_slider

    def _sync_angle(self, e):
        angle0 = e.new
        old_angle0, old_angle1 = self.cds.data['angle']
        dtheta = old_angle1 - old_angle0
        self.update(angle=[angle0, angle0 + dtheta])

    @staticmethod
    def _vectors_cb_rotation_js():
        return """
const centre_x = vector_source.data.xs[0][0]
const centre_y = vector_source.data.ys[0][0]

"""

    @staticmethod
    def _vectors_cb_base_js(with_emit: bool = True):
        code = """
const angle_deg = angle_slider.value * direction
const angle_rad0 = angle_deg * Math.PI/180;
const length = vector_source.data.length

const delta_theta_deg = vector_source.data.angle[1] - vector_source.data.angle[0]
const delta_theta_rad = delta_theta_deg  * Math.PI/180

const cos_dir0 = Math.cos(angle_rad0)
const sin_dir0 = Math.sin(angle_rad0)
const cos_dir1 = Math.cos(angle_rad0 + delta_theta_rad)
const sin_dir1 = Math.sin(angle_rad0 + delta_theta_rad)

vector_source.data.xs = []
vector_source.data.ys = []
vector_source.data.angle = [angle_deg, angle_deg + delta_theta_deg]
vector_source.data.xs.push([centre_x, centre_x + length[0] * cos_dir0])
vector_source.data.ys.push([centre_y, centre_y + length[0] * sin_dir0])
vector_source.data.xs.push([centre_x, centre_x + length[1] * cos_dir1])
vector_source.data.ys.push([centre_y, centre_y + length[1] * sin_dir1])

"""
        if not with_emit:
            return code
        else:
            return code + """
vector_source.change.emit()

"""

    @staticmethod
    def _text_cb_js():
        return '''
vector_source.data.text_x = []
vector_source.data.text_y = []
vector_source.data.text_x.push(centre_x + length[0] * \
    vector_source.data.length_mult[0] * cos_dir0)
vector_source.data.text_x.push(centre_x + length[1] * \
    vector_source.data.length_mult[1] * cos_dir1)
vector_source.data.text_y.push(centre_y + length[0] * \
    vector_source.data.length_mult[0] * sin_dir0)
vector_source.data.text_y.push(centre_y + length[1] * \
    vector_source.data.length_mult[1] * sin_dir1)
vector_source.change.emit()
'''


Degrees = NewType('Degrees', float)


class VectorsOverlayCons(ConsBase):
    @classmethod
    def from_params(
        cls,
        cx: float,
        cy: float,
        length: tuple[float, float] | float,
        angle: tuple[Degrees, Degrees] | Degrees = 0.,
        colors: tuple[str, str] | str = ('#FF8C00', '#00BFFF'),
        labels: tuple[str, str] | None = None,
    ) -> VectorsOverlay:
        try:
            a0, a1 = angle
        except TypeError:
            a0, a1 = angle, angle + 90.
        try:
            l0, l1 = length
        except TypeError:
            l0, l1 = length, length
        if isinstance(colors, str):
            colors = [colors] * 2
        lengths = [l0, l1]
        angles = [a0, a1]
        data = {
            **cls._vectors_init(cx, cy, angles, lengths),
            'line_color': colors,
            'length': lengths,
            'angle': angles,
        }
        cds = ColumnDataSource(data)
        dbase = VectorsOverlay(cds)
        if labels is not None:
            return dbase.with_labels(labels)
        return dbase

    @classmethod
    def empty(cls) -> VectorsOverlay:
        """
        VectorsOverlay cannot be created empty

        raises NotImplementedError
        """
        raise NotImplementedError('VectorsOverlay cannot be instantiated empty')

    @staticmethod
    def _vectors_init(cx, cy, angles_deg, lengths, mult=1.):
        angle0, angle1 = np.deg2rad(angles_deg)
        length0, length1 = lengths
        dx0, dy0 = (
            np.cos(angle0) * length0 * mult,
            np.sin(angle0) * length0 * mult,
        )
        dx1, dy1 = (
            np.cos(angle1) * length1 * mult,
            np.sin(angle1) * length1 * mult,
        )
        return {
            'xs': [[cx, cx + dx0], [cx, cx + dx1]],
            'ys': [[cy, cy + dy0], [cy, cy + dy1]],
        }
