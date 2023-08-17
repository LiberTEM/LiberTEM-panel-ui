from __future__ import annotations
from typing import TYPE_CHECKING, NewType
import numpy as np

# from bokeh.models import CustomJS, Slider, Select
from bokeh.models.glyphs import MultiLine as BkMultiLine
from bokeh.models.sources import ColumnDataSource

from .display_base import DisplayBase, ConsBase


if TYPE_CHECKING:
    pass


class MultiLine(DisplayBase):
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
    Adds two lines to the figure at right-angles. Has functionality
    to interactively set the vector angle using a slider.
    """
    glyph_map = {
        'vectors': MultiLine
    }

    def __init__(self, cds: ColumnDataSource):
        super().__init__(cds)
        lines = MultiLine(cds)
        self._register_child('vectors', lines)

    @classmethod
    def new(cls):
        return VectorsOverlayCons()


Degrees = NewType('Degrees', float)


class VectorsOverlayCons(ConsBase):
    @classmethod
    def from_params(
        cls,
        cx: float,
        cy: float,
        length: float,
        initial_angle: Degrees = 0.,
        colors: tuple[str, str] | str = ('#FF0000', '#0000FF'),
    ) -> VectorsOverlay:
        if isinstance(colors, str):
            colors = [colors] * 2
        data = {
            **cls._vectors_init(cx, cy, initial_angle, length),
            'line_color': colors,
            'length': [length] * 2,
            'angle': [initial_angle] * 2,
        }
        cds = ColumnDataSource(data)
        return VectorsOverlay(cds)

    @classmethod
    def empty(cls) -> VectorsOverlay:
        """
        VectorsOverlay cannot be created empty

        raises NotImplementedError
        """
        raise NotImplementedError('VectorsOverlay cannot be instantiated empty')

    @staticmethod
    def _vectors_init(cx, cy, angle_deg, length, mult=1.):
        angle = np.deg2rad(angle_deg)
        dx0, dy0 = np.cos(angle) * length * mult, np.sin(angle) * length * mult
        dx1, dy1 = np.cos(angle + np.pi/2) * length * mult, np.sin(angle + np.pi/2) * length * mult
        return {'xs': [[cx, cx + dx0], [cx, cx + dx1]],
                'ys': [[cy, cy + dy0], [cy, cy + dy1]]}

#     @property
#     def rotation_slider(self) -> Slider:
#         """
#         Get or make the rotation slider to control the vectors

#         If control is desired over the slider params, call
#         :meth:`~VectorsOverlay.make_rotation_slider` first.

#         Returns
#         -------
#         Slider
#             The rotation slider
#         """
#         try:
#             return self._rotation_slider
#         except AttributeError:
#             return self.make_rotation_slider()

#     def make_rotation_slider(self, start_angle: float = None,
#                              name: str = 'Angle (rad)', **kwargs) -> Slider:
#         """
#         Make the rotation slider to control the vectors, in radians


#         Parameters
#         ----------
#         start_angle : float, optional
#             Set the initial slider angle, by default 0. radians
#         name : str, optional
#             The label to apply to the slider, by default 'Angle'
#         **kwargs : dict, optional
#             Optional :code:`kwargs` passed to the Slider constructor,
#             which can be used to set style options.

#         Returns
#         -------
#         Slider
#             The rotation slider
#         """
#         if start_angle is None:
#             start_angle = 0.
#             if self.cds is not None:
#                 start_angle = self.cds.data['angle'][0]
#         assert -np.pi < start_angle <= np.pi
#         self._rotation_slider = Slider(title=name,
#                                        value=start_angle,
#                                        start=-np.pi,
#                                        end=np.pi,
#                                        step=0.01,
#                                        **kwargs)
#         return self.rotation_slider

#     def build_figure(self,
#                      pointset: PointSet = None,
#                      pos_xy: tuple[float, float] = None,
#                      length: float = 100.,
#                      colors: tuple[str, str] = None,
#                      labels: tuple[str, str] = None,
#                      text_length_mult: float = 1.1,
#                      initial_idx: int = 0,
#                      initial_angle: float = 0.,
#                      labels_kwargs: dict = None,
#                      **kwargs):
#         if colors is None:
#             colors = ['red', 'blue']
#         assert len(colors) == 2

#         if pointset is not None:
#             assert pos_xy is None, 'can build VectorsOverlay either
# from PointSet or fixed position'
#             pos_xy = self._get_initial_pos_xy(pointset, initial_idx)
#         else:
#             assert pos_xy is not None, ('can build VectorsOverlay either from '
#                                         'PointSet or fixed position')

#         # Build vectors
#         self.cds = ColumnDataSource({**self._vectors_init(pos_xy, initial_angle, length),
#                                      'line_color': colors,
#                                      'length': [length] * 2,
#                                      'angle': [initial_angle] * 2,
#                                      'idx': [initial_idx] * 2})
#         glyph = MultiLine(xs="xs",
#                           ys="ys",
#                           line_color='line_color',
#                           line_width=kwargs.pop('line_width', 2),
#                           **kwargs)
#         self.add_glyph(self.cds, glyph)

#         # Add Text overlay for labels
#         self._text = None
#         if labels is not None:
#             if labels_kwargs is None:
#                 labels_kwargs = {}
#             assert len(labels) == 2
#             assert all(isinstance(label, str) for label in labels)
#             initial_pos_text = self._vectors_init(pos_xy, initial_angle, length,
#                                                   mult=text_length_mult)
#             initial_pos_text = [(initial_pos_text['xs'][0][1], initial_pos_text['ys'][0][1]),
#                                 (initial_pos_text['xs'][1][1], initial_pos_text['ys'][1][1])]
#             with self.fig.as_parent(self):
#                 self._text = self.fig.add_text_overlay(pos_xy=initial_pos_text,
#                                                     strings=labels,
#                                                     other_data={'text_color': colors,
#                                                                 'length_mult': [text_length_mult,
#                                                                                 text_length_mult]},
#                                                     text_color='text_color',
#                                                     text_baseline=labels_kwargs.pop('text_baseline',
#                                                                                     'middle'),
#                                                     text_align=labels_kwargs.pop('text_align',
#                                                                                  'center'),
#                                                     **labels_kwargs)

#     @staticmethod
#     def _get_initial_pos_xy(pointset, initial_idx):
#         try:
#             return pointset.cds.data['cx'][initial_idx], pointset.cds.data['cy'][initial_idx]
#         except IndexError:
#             raise IndexError(f'Initialising VectorsOverlay with initial_idx {initial_idx}'
#                              f'but PointSet has length {len(pointset.cds.data["cx"])}')

#     @staticmethod
#     def _vectors_init(pos_xy, angle, length, mult=1.):
#         cx, cy = pos_xy
#         dx0, dy0 = np.cos(angle) * length * mult, np.sin(angle) * length * mult
#         dx1, dy1 = np.cos(angle + np.pi/2) * length * mult, np.sin(angle + np.pi/2)
# * length * mult
#         return {'xs': [[cx, cx + dx0], [cx, cx + dx1]],
#                 'ys': [[cy, cy + dy0], [cy, cy + dy1]]}

#     def setup_callbacks(self,
#                         rotation: bool = True,
#                         pointset: PointSet | None = None,
#                         selectable: bool = False):
#         """
#         Configure callbacks on the VectorsOverlay depending on requirements

#         Currently the user *must* call this to set up the desired callbacks.

#         Parameters
#         ----------
#         rotation : bool, optional
#             Add rotation via the slider returned by :meth:`~VectorsOverlay.rotation_slider`,
#             by default True. If control is desired over the slider creation,
#             call :meth:`VectorsOverlay.self.make_rotation_slider` first
#             to set this property.
#         pointset : Optional[PointSet], optional
#             Link the vector origin to the supplied :class:`~PointSet`,
#             by default None in which case the origin is fixed.
#         selectable : bool, optional
#             If linking to a :class:`~PointSet` enable selection of the
#             point to follow by setting selectable to True, by default False.
#             If set up, use the select box returned by :meth:`~VectorsOverlay.point_select`
#             to control the vector location.
#         """
#         args = {'vector_source': self.cds}

#         if self._text is not None:
#             args['text_source'] = self._text.cds
#             text_code = self._text_cb_js()
#         else:
#             text_code = ''

#         if pointset is not None:
#             args['point_source'] = pointset.cds
#             if selectable:
#                 if self.point_select is None:
#                     self.get_point_select(pointset)
#                 args['point_select'] = self.point_select

    # if rotation:
    #     args['angle_slider'] = self.rotation_slider
    #     rotation_code = self._vectors_cb_rotation_js() + self._vectors_cb_base_js() + text_code
    #     callback_rot = CustomJS(args=args,
    #                             code=rotation_code)
    #     self.rotation_slider.js_on_change('value', callback_rot)

    # if pointset is not None:
    #     pointset_code = self._vectors_cb_pointset_js() + self._vectors_cb_base_js() + text_code
    #     callback_pointset = CustomJS(args=args,
    #                                     code=pointset_code)
    #     pointset.cds.js_on_change('data', callback_pointset)

    #     if selectable:
    #         callback_select = CustomJS(args=args,
    #                                     code=pointset_code)
    #         self.point_select.js_on_change('value', callback_select)

#     def move_centre(self, pos_xy: tuple[float, float]):
#         """
#         Moves the centre of the vectors display to pos_xy

#         Parameters
#         ----------
#         pos_xy : tuple[float, float]
#             The new coordinates of the vectors origin
#         """
#         current_angle = self.rotation_slider.value
#         original_length = self.cds.data['length'][0]
#         new_cds_coords = self._vectors_init(pos_xy, current_angle, original_length)
#         self._update_cds_datadict(self.cds, new_cds_coords)
#         if self._text is not None:
#             text_mult = self._text.cds.data['length_mult'][0]
#             new_text_cds_coords = self._vectors_init(pos_xy,
#                                                      current_angle,
#                                                      original_length,
#                                                      mult=text_mult)
#             self._text._update_cds_datadict(self._text.cds, new_text_cds_coords)

#     @property
#     def point_select(self) -> Select | None:
#         """
#         Get the current point selection box, if defined

#         Returns
#         -------
#         Optional[Select]
#             The selection box
#         """
#         try:
#             return self._point_select
#         except AttributeError:
#             return None

#     def get_point_select(self, pointset: PointSet, **kwargs) -> Select:
#         """
#         Make a selection box on :code:`pointset` and set it as the
#         selection box for this VectorsOverlay.

#         Parameters
#         ----------
#         pointset : PointSet
#             The pointset to link to
#         **kwargs : dict, optional
#             Optional :code:`kwargs` passed to the Selec constructor, which
#             can be used to set style options.

#         Returns
#         -------
#         Select
#             The selection box
#         """
#         self._point_select = pointset.make_point_select(**kwargs)
#         return self._point_select

#     @staticmethod
#     def _vectors_cb_pointset_js():
#         return '''
# var point_idx;
# if (typeof point_select !== 'undefined') {
#   if (point_source.data.hasOwnProperty('label')) {
#       point_idx = point_source.data.label.findIndex(element =>  element === point_select.value)
#       if (point_idx == -1) {
#           point_idx = 0
#       }
#   } else {
#       point_idx = point_select.value;
#   }
# } else {
#   point_idx = vector_source.data.idx[0];
# }
# const centre_x = point_source.data.cx[point_idx]
# const centre_y = point_source.data.cy[point_idx]
# '''

#     @staticmethod
#     def _vectors_cb_rotation_js():
#         return '''
# const centre_x = vector_source.data.xs[0][0]
# const centre_y = vector_source.data.ys[0][0]
# '''

#     @staticmethod
#     def _vectors_cb_base_js():
#         return '''
# var angle;
# if (typeof angle_slider !== 'undefined') {
#   angle = angle_slider.value;
# } else {
#   angle = vector_source.data.angle[0];
# }

# const length = vector_source.data.length

# const cos_dir = Math.cos(angle)
# const sin_dir = Math.sin(angle)
# const cos_dir_pi2 = Math.cos(angle + Math.PI/2)
# const sin_dir_pi2 = Math.sin(angle + Math.PI/2)

# vector_source.data.xs = []
# vector_source.data.ys = []
# vector_source.data.angle = [angle, angle]
# vector_source.data.xs.push([centre_x, centre_x + length[0] * cos_dir])
# vector_source.data.ys.push([centre_y, centre_y + length[0] * sin_dir])
# vector_source.data.xs.push([centre_x, centre_x + length[1] * cos_dir_pi2])
# vector_source.data.ys.push([centre_y, centre_y + length[1] * sin_dir_pi2])
# vector_source.change.emit()
# '''

#     @staticmethod
#     def _text_cb_js():
#         return '''

# text_source.data.x = []
# text_source.data.y = []
# text_source.data.x.push(centre_x + length[0] * text_source.data.length_mult[0] * cos_dir)
# text_source.data.x.push(centre_x + length[1] * text_source.data.length_mult[1] * cos_dir_pi2)
# text_source.data.y.push(centre_y + length[0] * text_source.data.length_mult[0] * sin_dir)
# text_source.data.y.push(centre_y + length[1] * text_source.data.length_mult[1] * sin_dir_pi2)
# text_source.change.emit()
# '''
