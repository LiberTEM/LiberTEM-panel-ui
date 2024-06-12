from __future__ import annotations
import time
import numpy as np
from typing import TYPE_CHECKING, Callable

import panel as pn
from bokeh.models import Image
from bokeh.models.mappers import LogColorMapper, LinearColorMapper
from bokeh.models.sources import ColumnDataSource
from bokeh.models.annotations import ColorBar

from .display_base import DisplayBase, PointSet
# from .gamma_mapper import GammaColorMapper
from ..utils import colormaps as cmaps
from bokeh.models.widgets import RangeSlider, CheckboxGroup, Button, Spinner  # Slider
from bokeh.models import CustomJS
from bokeh.events import RangesUpdate

if TYPE_CHECKING:
    from bokeh.models.mappers import ColorMapper
    from .image_datashader import DatashadeHelper


def slider_step_size(start, end, n=300):
    return abs(end - start) / n


class BokehImageCons:
    @classmethod
    def from_numpy(cls, array: np.ndarray) -> BokehImage:
        """
        Construct a :class:`~BokehImage` from a 2-D numpy array

        Returns
        -------
        BokehImage
            The image constructed from the array
        """
        cls.check_nparray(array)
        array = cls._cast_if_needed(array)
        minmax = cls._calc_minmax(array)
        cds_dict = cls._get_datadict(
            array,
            minmax,
        )
        cds = ColumnDataSource(cds_dict)
        return BokehImage(cds=cds)

    @classmethod
    def from_shape(cls, *, height: int, width: int, dtype: np.dtype = np.float32,
                   constructor: Callable[[tuple], np.ndarray] = np.zeros, **kwargs) -> BokehImage:
        """
        Convenience method to construct an :class:`~BokehImage` from a shape tuple

        All other kwargs are forwarded to :meth:`~BokehImage.from_numpy`

        Parameters
        ----------
        height : int
            Height of the array
        width : int
            Width of the array
        dtype : np.dtype, optional
            Dtype of the constructed array, by default np.float32
        constructor : callable, optional
            Cosntructor for the array, by default np.zeros

        Returns
        -------
        BokehImage
            The image constructed from the array
        """
        assert isinstance(height, int) and isinstance(width, int)
        array = constructor(shape=(height, width)).astype(dtype)
        return cls.from_numpy(array=array, **kwargs)

    @classmethod
    def from_url(cls, url):
        import imageio.v3 as iio
        array = iio.imread(url, index=None)
        if array.ndim == 3:
            from skimage.color import rgb2gray
            array = rgb2gray(array)
        return cls.from_numpy(array)

    @classmethod
    def _get_datadict(
        cls,
        array: np.ndarray,
        minmax: tuple[float, float],
        anchor_xy: tuple[float, float] = (0., 0.),
        px_offset: float = -0.5,
    ) -> dict[str, list]:
        return {
            **cls._get_geometry(array.shape[:2], anchor_xy, px_offset),
            **cls._get_array(array),
            **cls._get_minmax(minmax),
            'cbar_slider': [False],
            'cbar_centered': [False],
            'clim_update': [True],  # when False this will disable clim updates in the jscallback
        }

    @staticmethod
    def _get_geometry(shape, anchor_xy, px_offset):
        height, width = shape
        left, top = anchor_xy
        return {
            'x': [left + px_offset],
            'y': [top + px_offset],
            'dw': [width],
            'dh': [height],
        }

    @classmethod
    def _get_array(cls, array: np.ndarray):
        return {
            'image': [array],
        }

    @staticmethod
    def _get_minmax(minmax):
        return {
            'val_low': [minmax[0]],
            'val_high': [minmax[1]],
        }

    @staticmethod
    def _calc_minmax(array) -> tuple[float, float]:
        data = array.ravel()
        mmin, mmax = np.min(data), np.max(data)
        if np.isnan(mmin) or np.isnan(mmax):
            mmin, mmax = np.nanmin(data), np.nanmax(data)
        if np.isnan(mmin) or np.isnan(mmax):
            mmin, mmax = 0., 1.
        if mmin == mmax:
            mmax = mmin + 1.
        return mmin, mmax

    @staticmethod
    def _cast_if_needed(array: np.ndarray) -> np.ndarray:
        if np.dtype(array.dtype).itemsize > 4:
            return array.astype(np.float32)
        return array

    @staticmethod
    def check_nparray(array):
        assert isinstance(array, np.ndarray)
        assert array.ndim == 2, 'No support for multi-channel images yet'
        assert all(s > 0 for s in array.shape)
        assert not np.issubdtype(array.dtype, complex), 'No support for complex images yet'
        return True


class BokehImage(DisplayBase):
    """
    Adds a Bokeh Image glyph to a figure and provides convenience methods on it

    Attributes
    ----------
    _px_offset: float
        -0.5 by default, used to offset the image origin by a half pixel
        so that the pixels as-displayed are centred on a whole coordinate
        and not anchored at their top-left. Set this to zero to anchor pixels
        at their top-left.
    """
    _px_offset: float = -0.5
    glyph_map = {
        'image': Image,
        'bounds': PointSet,
    }
    constructor = BokehImageCons

    def __init__(
        self,
        cds: ColumnDataSource,
    ):
        super().__init__(cds)

        glyph = Image(image='image', x='x', y='y', dw='dw', dh='dh')
        glyph.color_mapper.palette = cmaps.get_colormap('Greys')
        self._register_glyph('image', glyph)
        self._ds_helper: DatashadeHelper | None = None
        self._array: np.ndarray = self.cds.data['image'][0].copy()

    @staticmethod
    def new():
        return BokehImageCons()

    @property
    def im(self) -> Image:
        return self._glyphs['image'][0].glyph

    @property
    def array(self) -> np.ndarray:
        # A reference to the full, most-recent image array (after casting)
        return self._array

    @property
    def color(self) -> BokehImageColor:
        try:
            return self._color_manager
        except AttributeError:
            self._color_manager = BokehImageColor(self)
            return self.color

    def _update_geometry(self):
        # Get true array shape not from CDS
        height, width = self.cds.data['image'][0].shape[:2]
        left, top = self.anchor_offset
        ypos = top + self._px_offset
        dh = height

        self.raw_update(
            x=[left + self._px_offset],
            dw=[width],
            y=[ypos],
            dh=[dh],
        )
        return self

    # @staticmethod
    # def convert_dtype(array):
    #     if np.issubdtype(array.dtype, np.floating) and array.itemsize != 4:
    #         return array.astype(np.float32)
    #     return array

    def update(
        self,
        array: np.ndarray,
    ):
        """
        Update the current displayed image with the supplied array

        This is the user-facing endpoint for triggering image updates

        Parameters
        ----------
        array : np.ndarray
            The array to update with
        """
        self.constructor.check_nparray(array)
        array = self.constructor._cast_if_needed(array)
        # Store ref to most recent full array on this class
        self._array = array
        minmax = self.constructor._calc_minmax(array)
        if self.use_downsampling():
            data = self.downsampler.compute_update(array)
        else:
            data = {
                **self.constructor._get_array(array),
                **self.constructor._get_geometry(
                    array.shape,
                    self.anchor_offset,
                    self._px_offset,
                ),
            }
        data['clim_update'] = [True]
        super().update(
            **data,
            **self.constructor._get_minmax(minmax),
        )

    def enable_downsampling(self, dimension: int = 600, threshold_bytesize: float = 0.25 * 2**20):
        array_bytesize = np.dtype(self.array.dtype).itemsize * self.array.size

        if self.use_downsampling():
            # Already active
            self.downsampler.set_dimension(dimension)
            self.downsampler.redraw(self.array)
        elif self.downsampler is None and array_bytesize > threshold_bytesize:
            self._create_downsampler(dimension)
            # Push an update to the CDS to ensure we initialize in a low resolution
            # This will go through the downsampler and update its internal self.array
            self.update(self.array)
        elif array_bytesize > threshold_bytesize:
            self.downsampler.enable()
            self.downsampler.set_dimension(dimension)
            # If we have changed the array size on this class then the downsampler
            # will raise because it does not support changing array size yet
            self.downsampler.redraw(self.array)
        return self

    def disable_downsampling(self):
        if not self.use_downsampling():
            return
        self.downsampler.disable()  # Nullifies the callback
        self.update(self.array)
        return self

    def _create_downsampler(self, dimension):
        figures = self.is_on()
        if not figures:
            raise NotImplementedError('Must add image to figure before enabling downsampling')
        # Responsive downsampling breaks some assumptions of DisplayBase
        # because it makes no sense for multiple figures, and breaks
        # when an image is removed from a figure because Bokeh doesn't support
        # removing an on_event callback
        fig = figures[0]

        from .image_datashader import DatashadeHelper
        self._ds_helper = DatashadeHelper(fig, self, dimension=dimension)
        # Could use the inner_ values to set the canvas size but they are
        # not available until the figure is actually displayed on the screen
        # fig.inner_height, fig.inner_width
        fig.on_event(RangesUpdate, self._ds_helper.update_view)
        # Create a pointset to retain the bounds of the image
        # This isn't updated if we change the array size (normally)
        # forbidden by the downsampler, but could be later added
        h, w = self._ds_helper.array.shape
        self._bound_ps = PointSet.new().from_vectors([0, w], [0, h]).on(fig)
        self._bound_ps.points.fill_alpha = 0.
        self._bound_ps.points.line_alpha = 0.
        self._register_child('bounds', self._bound_ps)

    @property
    def downsampler(self):
        return self._ds_helper

    def use_downsampling(self):
        return self.downsampler is not None and self.downsampler.active

    # def patch(self, array: np.ndarray, slice_0: slice, slice_1: slice):
    #     # This needs to defer to datashade helper if defined
    #     patched_data = array[slice_0, slice_1].ravel()
    #     patches = {'image': [([0, slice_0, slice_1], patched_data)]}
    #     self.cds.patch(patches)

    @property
    def current_minmax(self) -> tuple[float, float] | tuple[None, None]:
        try:
            return self.cds.data['val_low'][0], self.cds.data['val_high'][0]
        except (KeyError, IndexError):
            return None, None

    @property
    def last_update(self) -> float:
        """
        Used to track when the CDS was last updated

        :meta private:
        """
        try:
            return self._last_update
        except AttributeError:
            return 0.

    def update_tick(self) -> None:
        """
        Signal that we did an update

        :meta private:
        """
        self._last_update = time.time()

    def rate_limit(self, maxrate: int) -> bool:
        """
        Computes whether we should rate limit

        To be practical this needs to schedule the final update
        to run automatically if we drop some updates due to
        applying a rate limit

        :meta private:
        """
        if maxrate == 0:
            return False
        assert maxrate > 0
        now = time.time()
        if (now - self.last_update) > (1. / maxrate):
            self.update_tick()
            return False
        else:
            return True

    @property
    def anchor_offset(self) -> tuple[float, float]:
        try:
            return self._anchor_offset
        except AttributeError:
            self._anchor_offset = (0., 0.)
            return self.anchor_offset

    def set_anchor(self, *, x: float, y: float):
        """
        Sets the image anchor point

        :meta private:
        """
        self._anchor_offset = (x, y)
        self._update_geometry()

    def reset_anchor(self):
        if self.anchor_offset == (0., 0.):
            return
        self.set_anchor(x=0., y=0.)


class BokehImageColor():
    def __init__(self, img: BokehImage):
        self.img = img
        self._colorbars: list[ColorBar] = []

        low, high = self.img.current_minmax
        palette = cmaps.get_colormap('Greys')
        self._lin_mapper: LinearColorMapper = LinearColorMapper(low=low, high=high)
        self._lin_mapper.palette = palette
        self._log_mapper: LogColorMapper = LogColorMapper(**self._log_norm_py(low, high))
        self._log_mapper.palette = palette
        self.img.im.color_mapper = self._lin_mapper

    def push_clims(self):
        # Force current_minmax onto both colormappers
        low, high = self.img.current_minmax
        self._lin_mapper.update(low=low, high=high)
        self._log_mapper.update(**self._log_norm_py(low, high))

    @property
    def color_mapper(self) -> ColorMapper:
        return self.img.im.color_mapper

    @property
    def cbar_slider(self) -> RangeSlider | None:
        """
        Returns the current colormap slider, if initialized

        Returns
        -------
        Optional[RangeSlider]
            Slider if present, else None
        """
        try:
            return self._cbar_slider
        except AttributeError:
            return None

    def has_cbar_slider(self) -> bool:
        return self.cbar_slider is not None

    def get_cmap_slider(self,
                        title: str = 'Colormap range',
                        init_range: tuple[float, float] | None = None,
                        nstep: int = 300,
                        **kwargs) -> RangeSlider:
        """
        Create a two-handled RangeSlider used to interactively modify
        the colormapper lower/upper limits.

        Parameters
        ----------
        title : str, optional
            A label to apply to the slider, by default 'Colormap range'
        fixed_range : Optional[tuple[float, float]], optional
            Explicit initial range for the slider, by default None, in which
            case the current image min/max will be used.
        **kwargs : dict, optional
            Optional :code:`kwargs` passed to the RangeSlider constructor, which
            can be used to set style and other options.

        Returns
        -------
        RangeSlider
            The colormap slider with callbacks registered.
        """
        if self.has_cbar_slider():
            return self.cbar_slider

        if not init_range:
            init_range = self.img.current_minmax

        self._cbar_freeze = CheckboxGroup(labels=['Freeze color limits'], active=[])
        self._full_scale_btn = Button(label="Full scale", button_type="default")

        self._cbar_slider = RangeSlider(title=title,
                                        start=init_range[0],
                                        end=init_range[1],
                                        value=init_range,
                                        step=slider_step_size(*init_range, n=nstep),
                                        syncable=False,
                                        **kwargs)

        clim_value_callback = CustomJS(args={'lin_mapper': self._lin_mapper,
                                             'log_mapper': self._log_mapper,
                                             'cds': self.img.cds,
                                             'freeze': self._cbar_freeze},
                                       code=self._clim_slider_value_js())
        self._cbar_slider.js_on_change('value', clim_value_callback)

        autorange_btn = self.get_clip_outliers_btn(nstep=nstep)
        clim_freeze_callback = CustomJS(args={'cds': self.img.cds,
                                              'clim_slider': self._cbar_slider,
                                              'nstep': nstep,
                                              'full_btn': self._full_scale_btn,
                                              'autorange_btn': autorange_btn},
                                        code=self._clim_freeze_toggle_js())
        self._cbar_freeze.js_on_change('active', clim_freeze_callback)

        self.img.raw_update(cbar_slider=[True])
        clim_update_callback = CustomJS(args={'clim_slider': self._cbar_slider,
                                              'lin_mapper': self._lin_mapper,
                                              'log_mapper': self._log_mapper,
                                              'nstep': nstep,
                                              'freeze': self._cbar_freeze},
                                        code=self._clim_slider_update_image_js())
        self.img.cds.js_on_change('data', clim_update_callback)

        full_scale_callback = CustomJS(args={'clim_slider': self._cbar_slider,
                                             'cds': self.img.cds,
                                             'freeze': self._cbar_freeze,
                                             'nstep': nstep,
                                             'im_glyph': self.img.im},
                                       code=self._clim_full_scale_js())
        self._full_scale_btn.js_on_event("button_click", full_scale_callback)

        self._log_color_btn = pn.widgets.Toggle(
            name='Log color',
            value=False,
        )
        self._log_color_btn.param.watch(self._toggle_log_color, 'value')

        # self._gamma_slider = Slider(
        #     title='Gamma',
        #     start=-2.,
        #     end=2.,
        #     value=0.,
        #     step=0.01,
        # )
        # self._gamma_slider.js_link('value_throttled', self._lin_mapper, 'gamma')

        # self._gamma_reset_btn = Button(
        #     label="Reset gamma", button_type="default"
        # )
#         reset_gamma_callback = CustomJS(
#             args={
#                 'gamma_slider': self._gamma_slider,
#                 'lin_mapper': self._lin_mapper
#             },
#             code='''
# lin_mapper.gamma = 0.
# lin_mapper.change.emit()
# gamma_slider.value = 0.
# gamma_slider.change.emit()
# ''')
#         self._gamma_reset_btn.js_on_event("button_click", reset_gamma_callback)
        return self.cbar_slider

    def _toggle_log_color(self, e):
        if e.new:
            self.img.im.color_mapper = self._log_mapper
        else:
            self.img.im.color_mapper = self._lin_mapper
        for colorbar in self._colorbars:
            colorbar.color_mapper = self.img.im.color_mapper

    @staticmethod
    def _log_norm_py(low, high):
        if low <= 0. and high <= 0.:
            low = 0.01
            high = 1.
        elif low <= 0.:
            low = min(0.01, high * 0.1)
        return {'low': low, 'high': high}

    @staticmethod
    def _log_norm_js():
        return '''
if (low <= 0.  && high <= 0.) {
    low = 0.01
    high = 1.
} else if (low <= 0.){
    low = Math.min(0.01, high * 0.1)
}'''

    @staticmethod
    def _clim_slider_update_image_js():
        return '''

const clim_update = cb_obj.data.clim_update[0]
if (!clim_update){
    return
}

if (freeze.active.length == 1){
    return
}

var low = cb_obj.data.val_low[0]
var high = cb_obj.data.val_high[0]

clim_slider.start = low
clim_slider.end = high
clim_slider.value = [low, high];
clim_slider.step = (high - low) / nstep;

if (cb_obj.data.cbar_centered[0]){
    const val = Math.max(Math.abs(low), Math.abs(high))
    low = -val
    high = val
}

lin_mapper.low = low;
lin_mapper.high = high;
''' + BokehImageColor._log_norm_js() + '''
log_mapper.low = low;
log_mapper.high = high;
'''

    @staticmethod
    def _clim_slider_value_js():
        return '''
var low = cb_obj.value[0]
var high = cb_obj.value[1]

if (cds.data.cbar_centered[0]){
    const val = Math.max(Math.abs(low), Math.abs(high))
    low = -val
    high = val
}

lin_mapper.low = low;
lin_mapper.high = high;
''' + BokehImageColor._log_norm_js() + '''
log_mapper.low = low;
log_mapper.high = high;
'''

    @staticmethod
    def _clim_freeze_toggle_js():
        return '''
if (cb_obj.active.length == 1){
    clim_slider.disabled = true
    full_btn.disabled = true
    autorange_btn.disabled = true
    return
}

clim_slider.disabled = false
full_btn.disabled = false
autorange_btn.disabled = false

var low = cds.data.val_low[0]
var high = cds.data.val_high[0]

var new_val_low = Math.max(low, clim_slider.value[0])
var new_val_high = Math.min(high, clim_slider.value[1])

clim_slider.start = low
clim_slider.end = high
clim_slider.value = [new_val_low, new_val_high];
clim_slider.step = (high - low) / nstep;
'''

    @staticmethod
    def _clim_full_scale_js():
        return '''
if (freeze.active.length == 1){
    return
}


const low = cds.data.val_low[0]
const high = cds.data.val_high[0]

clim_slider.start = low
clim_slider.end = high
clim_slider.value = [low, high];
clim_slider.step = (high - low) / nstep;
'''

    @property
    def current_palette_name(self) -> str:
        try:
            return self._current_palette_name
        except AttributeError:
            return cmaps.default_image_cmap_name()

    def change_cmap(self, palette: str):
        if self.cmap_select:
            self.cmap_select.value = cmaps.get_colormap(palette, inverted=self.is_cmap_inverted())
        elif self.color_mapper:
            self.change_cmap_cb(None, palette=palette)
        else:
            pass

    def change_cmap_cb(self, event, palette: str | None = None):
        """
        Callback triggered by changing the colormap select box

        :meta private:
        """
        if palette is None:
            palette = event.new
        self._current_palette_name = palette
        # can definitely be JS-linked!!!
        _palette_list = cmaps.get_colormap(palette, inverted=self.is_cmap_inverted())
        self._lin_mapper.palette = _palette_list
        self._log_mapper.palette = _palette_list

    def invert_cmap(self, event):
        if self.cmap_select is not None:
            self.cmap_select.param.trigger('value')

    @property
    def cmap_select(self):
        try:
            return self._cmap_select
        except AttributeError:
            return None

    def get_cmap_select(self,
                        title: str = 'Colormap',
                        default: str | None = None,
                        options: list[str] | None = None,
                        **kwargs) -> pn.widgets.Select:
        """
        Create a select box to change the image colormap interactively

        Parameters
        ----------
        title : str, optional
            A label to apply on the box, by default 'Colormap'
        default : Optional[str], optional
            _description_, by default None
        options : Optional[list[str]], optional
            _description_, by default None
        **kwargs : dict, optional
            Optional :code:`kwargs` passed to the Select constructor, which
            can be used to set style and other options.

        Returns
        -------
        pn.widgets.Select
            The select box with callbacks registered
        """
        if self.cmap_select is not None:
            return self.cmap_select
        self._cmap_select = pn.widgets.Select(name=title,
                                              options=(cmaps.available_colormaps()
                                                       if options is None else options),
                                              value=(self.current_palette_name
                                                     if default is None else default),
                                              **kwargs)
        self._cmap_select.param.watch(self.change_cmap_cb, 'value')
        return self._cmap_select

    @property
    def invert_cmap_box(self) -> pn.widgets.Checkbox:
        try:
            return self._invert_cmap_box
        except AttributeError:
            return None

    def get_cmap_invert(self, name='Invert colormap', **kwargs):
        if self.invert_cmap_box is not None:
            return self.invert_cmap_box
        self._invert_cmap_box = pn.widgets.Checkbox(name=name, value=False, **kwargs)
        self._invert_cmap_box.param.watch(self.invert_cmap, 'value')
        return self.invert_cmap_box

    def is_cmap_inverted(self) -> bool:
        if self.invert_cmap_box is not None:
            return self.invert_cmap_box.value
        else:
            return False

    @property
    def center_cmap_toggle(self):
        try:
            return self._centre_cmap_toggle
        except AttributeError:
            return None

    def get_cmap_center(self, title='Centered colormap', value=False) -> pn.widgets.Checkbox:
        if self.center_cmap_toggle is not None:
            return self.center_cmap_toggle
        self._centre_cmap_toggle = pn.widgets.Checkbox(name=title, value=value)
        self._centre_cmap_toggle.param.watch(self.toggle_center_cmap_cb, 'value')
        self._centre_cmap_toggle.param.trigger('value')
        return self.center_cmap_toggle

    def toggle_center_cmap_cb(self, event):
        current = self.img.cds.data['cbar_centered'][0]
        if event.new == current:
            return
        self.img.raw_update(cbar_centered=[event.new])

    def add_colorbar(self, *figs, width=10, padding=2, position='right'):
        if not figs:
            figs = self.img.is_on()
        for fig in figs:
            for renderer in self.img.renderers_for_fig('image', fig):
                color_bar = renderer.construct_color_bar(width=width, padding=padding)
                fig.add_layout(color_bar, position)
                # Need reference so we can change the mapper
                # when switching from log to lin
                self._colorbars.append(color_bar)

    @property
    def clip_outliers_btn(self):
        try:
            return self._clip_outliers_btn
        except AttributeError:
            return None

    @property
    def clip_outliers_sigma_spinner(self):
        try:
            return self._clip_outliers_sigma_spinner
        except AttributeError:
            return None

    def get_clip_outliers_btn(self, nstep: int = 300) -> Button:
        if self.clip_outliers_btn is not None:
            return self.clip_outliers_btn

        self._clip_outliers_btn = Button(
            label="Autorange (n-sigma)",
            button_type="default"
        )
        self._clip_outliers_sigma_spinner = Spinner(
            value=2,
            low=0.1,
            high=8.,
            step=1.,
            mode='float',
            syncable=False,
            width=75,
        )
        autorange_callback = CustomJS(args={'clim_slider': self._cbar_slider,
                                            'freeze': self._cbar_freeze,
                                            'cds': self.img.cds,
                                            'nstep': nstep,
                                            'nsigma': self._clip_outliers_sigma_spinner},
                                      code=self._clim_autorange_js())
        self._clip_outliers_btn.js_on_event("button_click", autorange_callback)

        return self.clip_outliers_btn

    @staticmethod
    def _clim_autorange_js():
        return """
if (freeze.active.length == 1){
    return
}

const data = cds.data.image[0]
var non_nan = Math.max(data.length, 1)
const mean = data.reduce((acc, v) => {
    if (!Number.isNaN(v)) {
        return acc + v
    }
    non_nan -= 1
    return acc
}, 0) / Math.max(non_nan, 1)
var std = Math.sqrt(data.reduce((acc, v) => {
    return Number.isNaN(v) ? acc : acc + (Math.abs(v - mean) ** 2)}
, 0) / Math.max(non_nan, 1))
if (std == 0.) {
    std = 0.1
}

const data_low = cds.data.val_low[0]
const data_high = cds.data.val_high[0]

const nsig = nsigma.value

const low = Math.max(data_low, mean - nsig * std)
const high = Math.min(data_high, mean + nsig * std)
const bar_low = Math.max(data_low, mean - (nsig + 1) * std)
const bar_high = Math.min(data_high, mean + (nsig + 1) * std)

clim_slider.value = [low, high]
clim_slider.start = bar_low
clim_slider.end = bar_high
clim_slider.step = (bar_high - bar_low) / nstep;
"""
