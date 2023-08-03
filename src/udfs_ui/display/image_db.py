from __future__ import annotations
import time
import numpy as np
from typing import TYPE_CHECKING, Callable

import panel as pn
from bokeh.models import Image
from bokeh.models.sources import ColumnDataSource

from .display_base import DisplayBase, PointSet
from .utils import slider_step_size
from .utils import colormaps as cmaps
from bokeh.models.widgets import RangeSlider
from bokeh.models import CustomJS
from bokeh.events import RangesUpdate

if TYPE_CHECKING:
    from bokeh.models.mappers import ColorMapper
    from .image_datashader import DatashadeHelper


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
        cds_dict = cls._get_datadict(
            array,
            BokehImage.calc_minmax(array),
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
        array = cls._cast_if_needed(array)
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

    @staticmethod
    def new():
        return BokehImageCons()

    @property
    def im(self) -> Image:
        return self._glyphs['image'][0].glyph

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
        minmax = self.calc_minmax(array)
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
        super().update(
            **data,
            **self.constructor._get_minmax(minmax),
        )

    def enable_downsampling(self, dimension: int = 600):
        figures = self.is_on()
        if not figures:
            raise NotImplementedError('Must add image to figure before enabling downsampling')

        from .image_datashader import DatashadeHelper
        self._ds_helper = DatashadeHelper(self, dimension=dimension)
        # Responsive downsampling breaks some assumptions of DisplayBase
        # because it makes no sense for multiple figures, and breaks
        # when an image is removed from a figure because Bokeh doesn't support
        # removing an on_event callback
        fig = figures[0]
        # Could use the inner_ values to set the canvas size but they are
        # not available until the figure is actually displayed on the screen
        # fig.inner_height, fig.inner_width
        fig.on_event(RangesUpdate, self._ds_helper.update_view)
        # Create a pointset to retain the bounds of the image
        h, w = self._ds_helper.array.shape
        self._bound_ps = PointSet.new().from_vectors([0, w], [0, h]).on(fig)
        self._bound_ps.points.fill_alpha = 0.
        self._bound_ps.points.line_alpha = 0.
        self._register_child('bounds', self._bound_ps)
        # Push an update to the CDS to ensure we initialize in a low resolution
        self.update(self.downsampler.array.data)
        return self

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

    @staticmethod
    def calc_minmax(array) -> tuple[float, float]:
        data = array.ravel()
        mmin, mmax = np.min(data), np.max(data)
        if np.isnan(mmin) or np.isnan(mmax):
            mmin, mmax = np.nanmin(data), np.nanmax(data)
        if np.isnan(mmin) or np.isnan(mmax):
            mmin, mmax = 0., 1.
        if mmin == mmax:
            mmax = mmin + 1.
        return mmin, mmax

    @property
    def current_minmax(self):
        return self.cds.data['val_low'][0], self.cds.data['val_high'][0]

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

        self._cbar_slider = RangeSlider(title=title,
                                        start=init_range[0],
                                        end=init_range[1],
                                        value=init_range,
                                        step=slider_step_size(*init_range, n=nstep),
                                        syncable=False,
                                        **kwargs)

        clim_value_callback = CustomJS(args={'cmapper': self.color_mapper,
                                             'cds': self.img.cds},
                                       code=self._clim_slider_value_js())
        self._cbar_slider.js_on_change('value', clim_value_callback)

        self.img.raw_update(cbar_slider=[True])
        clim_update_callback = CustomJS(args={'clim_slider': self._cbar_slider,
                                              'cmapper': self.color_mapper,
                                              'nstep': nstep},
                                        code=self._clim_slider_update_image_js())
        self.img.cds.js_on_change('data', clim_update_callback)
        return self.cbar_slider

    @staticmethod
    def _clim_slider_update_image_js():
        return '''
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

cmapper.low = low;
cmapper.high = high;
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

cmapper.low = low;
cmapper.high = high;
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
        if self.color_mapper is not None:
            self._current_palette_name = palette
            # can definitely be JS-linked!!!
            self.color_mapper.palette = cmaps.get_colormap(palette,
                                                           inverted=self.is_cmap_inverted())

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
