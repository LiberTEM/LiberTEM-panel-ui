from __future__ import annotations
import numpy as np
import datashader as ds
import xarray as xr
from typing import TYPE_CHECKING

from bokeh.events import RangesUpdate

from .image_db import BokehImage, BokehImageCons

if TYPE_CHECKING:
    from bokeh.plotting import figure

VERBOSE = False


class DatashadeHelper:
    def __init__(
        self,
        fig: figure,
        im: BokehImage,
        dimension: int = 600,
        downsampling_method: str = 'mean',
    ):
        # This class is designed to never upsample, but specify nearest just in case!
        self._upsampling_method = 'nearest'
        self._downsampling_method = downsampling_method
        self._active = True
        # Assumes the image is anchored at 0, 0 for simplicitly
        # but could be relaxed if there is ever a need
        self._fig = fig
        self._im = im
        self._dimension = dimension
        self._setup_canvas(self.im.array)

    def _setup_canvas(self, array: np.ndarray):
        h, w = array.shape
        ys = np.arange(h).astype(float)
        xs = np.arange(w).astype(float)
        self._array_da = xr.DataArray(
            array,
            coords=[('y', ys), ('x', xs)],
        )
        self._array_da_minimum: np.ndarray | None = None

        height, width = self._determine_canvas(
            self.array.shape,
            self._dimension,
        )
        self._canvas = ds.Canvas(
            plot_width=width,
            plot_height=height,
        )
        self._check_enable()

    def _check_enable(self):
        h, w = self.im.array.shape
        canvas_height = self._canvas.plot_height
        canvas_width = self._canvas.plot_width
        if h <= canvas_height * 1.2 and w <= canvas_width * 1.2:
            # When the image is actually smaller than the canvas size
            # we gain nothing, so we disable the downsampler
            self.disable()
        else:
            self.enable()

    @property
    def fig(self):
        return self._fig

    @property
    def im(self):
        return self._im

    @property
    def array(self):
        return self._array_da

    @property
    def px_offset(self):
        return self.im._px_offset

    @property
    def active(self):
        return self._active

    def enable(self):
        # This could push an downsampled update
        self._active = True

    def disable(self):
        # This could push a full size update
        self._active = False

    @property
    def canvas(self) -> ds.Canvas:
        """
        The Datashader object which does the re-rasterizing

        :meta private:
        """
        return self._canvas

    @property
    def canvas_shape(self):
        return (self.canvas.plot_height, self.canvas.plot_width)

    def shade(self, xrange=None, yrange=None) -> xr.DataArray:
        """
        Shades the display_image with new x- and y- ranges
        ranges with None values will shade the whole dimension

        :meta private:
        """
        self.set_canvas_ranges(xrange=xrange, yrange=yrange)
        return self.reshade()

    def set_dimension(self, dimension: int):
        height, width = self._determine_canvas(self.array.shape, dimension)
        self.canvas.plot_width = width
        self.canvas.plot_height = height

    def set_canvas_ranges(self, xrange=None, yrange=None):
        self.canvas.x_range = xrange
        self.canvas.y_range = yrange

    @staticmethod
    def _determine_canvas(shape: tuple[int, int], canvas_diagonal: int) -> tuple[int, int]:
        h, w = shape
        wh2 = (w / h) ** 2
        d2 = canvas_diagonal ** 2
        hh = np.sqrt(d2 / (1 + wh2))
        ww = np.sqrt(d2 - (hh ** 2))
        return int(np.ceil(hh)), int(np.ceil(ww))

    def reshade(self):
        """
        Re-rasters the display_image using the x_range and y_range stored
        on self.canvas. This implicitly supports changing the image data
        (assuming same image array size) and updating the in-browser view
        without changing the CDS geometry or displayed crop of the image

        :meta private:
        """
        return self.canvas.raster(
            self.array,
            interpolate=self._upsampling_method,
            agg=self._downsampling_method,
        )

    @staticmethod
    def cds_to_ltrb(cds: dict[str, list[float]]) -> tuple[float, float, float, float]:
        """
        Convert the image CDS dict to left, top, right, bottom representation

        Returns
        -------
        tuple[float, float, float, float]
            ltrb geometry


        :meta private:
        """
        left = cds['x'][0]
        bottom = cds['y'][0]
        height = cds['dh'][0]
        width = cds['dw'][0]
        return left, bottom, left + width, bottom + height

    def current_cds_dims(self):
        """
        Current geometry of the glyph displaying the image

        :meta private:
        """
        return {k: self.im.cds.data[k] for k in ['x', 'y', 'dw', 'dh']}

    def continuous_bounds(self, apply_px_offset=True):
        """
        The bounds of the full image in continuous coordinates
        Returns position of outer edges of pixels as-drawn

        :meta private:
        """
        h, w = self.array.shape
        if apply_px_offset:
            anchor_x = anchor_y = self.px_offset
        else:
            anchor_x = anchor_y = 0.
        # anchor_x, anchor_y = self.shift_anchor(anchor_x, anchor_y)
        return anchor_x, anchor_y, anchor_x + w, anchor_y + h

    def array_bounds(self):
        """
        The array bounds in integer pixels

        :meta private:
        """
        h, w = self.array.shape
        return 0, 0, w, h

    def unpack_event(
        self,
        event: RangesUpdate
    ) -> tuple[bool, dict[str, list[float]], tuple[float, float]]:
        """
        Unpack the Bokeh event into the geometry used to perform datashading

        Parameters
        ----------
        event : RangesUpdate
            The callback event containing the current view information

        Returns
        -------
        bool
            is_visisble, whether the current view overlaps with the
            current array
        dict[str, list[float]]]
            a partial ColumnDataSource dictionary without image data
            matching the current view geometry


        :meta private:
        """
        # event contains viewport limits in continuous coordinates
        # implicitly the coordinates are in units of data/array pixels
        # not in screen pixels, but are not necessarily aligned with
        # the array origin (depends on what our self.px_offset is)
        # First step is to reverse y1 and y0 because we assume an inverted y-axis
        x0: float = event.x0
        y0: float = event.y0
        x1: float = event.x1
        y1: float = event.y1

        fig = event.model
        if fig.x_range.flipped:
            x0, x1 = x1, x0
        if fig.y_range.flipped:
            y0, y1 = y1, y0

        if VERBOSE:
            print(f'Event data {x0, y0, x1, y1}, width {x1 - x0}, height {y1 - y0}')
        viewport_width = x1 - x0
        viewport_height = y1 - y0
        # The full array is bounded in continuous coordinates
        # by the following geometry (includes self.px_offset)
        l, b, r, t = self.continuous_bounds()
        # Test if our viewport overlaps with the full array
        # Used for an early return if we don't have anything to show
        # print(f'LR: {(l, r)}, {(x0, x1)}')
        # print(f'BT: {(b, t)}, {(y0, y1)}')
        is_visible = (
            self.axis_overlaps((l, r), (x0, x1))
            and self.axis_overlaps((b, t), (y0, y1))
        )
        # Next we expand the coordinates outwards
        # This gives us a new 'viewport' guaranteed to contain the current
        # continuous coordinate viewport even when very zoomed in
        # We also expand again to align to the pixel grid
        x0 = np.floor(x0) + self.px_offset
        y0 = np.floor(y0) + self.px_offset
        x1 = np.ceil(x1) - self.px_offset
        y1 = np.ceil(y1) - self.px_offset
        # Clip the eventual geometry of the new view to the full
        # array, in continuous coords. This allows us to return a
        # partial crop of the array if we are partially outside of its range
        x0 = max(x0, l)
        y0 = max(y0, b)
        x1 = min(x1, r)
        y1 = min(y1, t)
        # h and w should work out because we are working in
        # continuous coordinates here, but should test this
        h = abs(y1 - y0)
        w = abs(x1 - x0)
        return (
            is_visible,
            {'x': [x0], 'y': [y0], 'dw': [w], 'dh': [h]},
            (viewport_width, viewport_height),
        )

    def full_cds_coords(self):
        h, w = self.array.shape
        return {
            'x': [self.px_offset],
            'y': [self.px_offset],
            'dw': [w],
            'dh': [h]
        }

    def ranges_from_cds_dict(self, cds: dict[str, list[float]], as_int: bool = False):
        """
        Gets ranges to give to canvas.raster from the display cds
        Assumes that the cds anchor has been shifted by self.px_offset

        :meta private:
        """
        x0 = cds['x'][0] - self.px_offset
        y0 = cds['y'][0] - self.px_offset
        # x0, y1 = self.unshift_anchor(x0, y1)
        w = cds['dw'][0]
        h = cds['dh'][0]
        if as_int:
            w = int(np.round(w))
            h = int(np.round(h))
            y0 = int(y0)
            x0 = int(x0)
        x1 = x0 + w
        y1 = y0 + h
        return (x0, x1), (y0, y1)

    def is_complete(
        self,
        xrange: tuple[float, float],
        yrange: tuple[float, float]
    ) -> bool:
        """
        Check if the given ranges cover the whole array in
        one or the other axis
        """
        full_h, full_w = self.array.shape
        x0, x1 = xrange
        y0, y1 = yrange
        return full_w <= abs(x1 - x0) or full_h <= abs(y1 - y0)

    @staticmethod
    def axis_overlaps(obj_range, view_range):
        return (obj_range[0] <= view_range[1]) and (view_range[0] <= obj_range[1])

    def update_view(
        self, event: RangesUpdate, force: bool = False,
        do_update: bool = True, with_clims: bool = False
    ):
        """
        This is the main callback linked to the RangesUpdate event

        Triggered every time the axes ranges change due to zooming,
        scrolling, reset etc.

        Unpacks the event information then determines the best array to update
        the view with (or to do nothing). In several cases no datashading
        is performed, notably when we cannot see the image (out of bounds)
        or we are too-far zoomed in, i.e datashading would be oversampling.

        As the zoom level reaches 1 pixel of the array per pixel on the screen
        this function essentially disables datashading by just transferring a
        1:1 slice of the array to the browser. If we over-zoom on this block
        no update is performed until we leave its bounds, forcing another
        1:1 update.

        :meta private:

        # IDEA Why not pre-generate an image pyramid, choose the closest LOD
        # to the viewport in the event, then send by slicing a centered window
        # into the array, avoids the time repeatedly calling .reshade
        # and most of the logic for over-zooming etc will be the same
        # the pyramid could also be generated lazily to avoid runtime overhead
        # In this case we don't even need to rely on xarray or datashader
        # unless their image resampling is faster than scikit-image
        """

        if not force and not self.active:
            # If disabled, assume nothing to do
            if VERBOSE:
                print('Inactive, skipping')
            return

        # Need to thorougly test this function for off-by-one errors...
        is_visible, new_cds_coords, viewport_wh = self.unpack_event(event)
        if not force and not is_visible:
            # Do nothing
            if VERBOSE:
                print('Out of bounds, skipping')
            return

        if not force and self.matching_bounds(self.current_cds_dims(), new_cds_coords):
            # Returns when the new displayed bounds match the old bounds
            # Implicitly there is nothing to do here
            # Covers the case when we zoom out from the full shaded image
            if VERBOSE:
                print('Matching bounds, skip update')
            return

        if not force and self.data_matches_scale(
            self.current_cds_dims(),
            new_cds_coords,
            viewport_wh
        ):
            # Cover the case when the data already in the CDS
            # fills the new viewport correctly, either if we are
            # zoomed out or we are panning around beyond the image
            if VERBOSE:
                print('Scale unchanged bounds, skip update')
            return

        if not force and self.is_oversampled(new_cds_coords):
            if self.in_bounds(self.current_cds_dims(), new_cds_coords):
                if self.is_oversampled(self.current_cds_dims()):
                    # Returns when we are oversampling wholly into a
                    # displayed image which is already oversampled
                    # Avoids unecessary updates when zoomed in
                    if VERBOSE:
                        print('Over-zoomed, skip update')
                    return
            # If we are oversampling just return a slice directly from the array
            # First update the view dimension so that it fills the canvas size
            # completely. This prevents some extra updates
            # TODO must see how this behaves when updating the image when already oversampled
            new_cds_coords, (xrange, yrange) = self.maximum_view(new_cds_coords)
            self.set_canvas_ranges(xrange=xrange, yrange=yrange)
            shaded = self.direct_sample(self.array, xrange, yrange)
            if VERBOSE:
                print('Update from slice')
        else:
            # TODO test / investigate slight shifts in the image when datashading
            # It might be related to the fact that the pixel [0, 0] in the datashaded
            # image is an average of several pixels, (the xarray list the average
            # coordinates). Here I just anchor the image to the top-left point
            # which we used to define the datashading ranges, but this might not
            # be the right choice (should perhaps offset by some amount...)
            xrange, yrange = self.ranges_from_cds_dict(new_cds_coords)
            is_complete = self.is_complete(xrange, yrange)
            if is_complete and self._array_da_minimum is not None:
                shaded = self._array_da_minimum
                new_cds_coords = self.full_cds_coords()
                if VERBOSE:
                    print('Update from existing fullsize')
            else:
                shaded = self.shade(xrange=xrange, yrange=yrange).to_numpy()
                if VERBOSE:
                    print('Update from shade')
                if is_complete:
                    self._array_da_minimum = shaded
                    if VERBOSE:
                        print('Storing fullsize')
            # A further path to cover is when we are 'close' to displaying
            # the full image, but not quite there (when zooming out, particularly)
            # This would save an update and a strange partial completion effect

        if VERBOSE:
            print(f'New CDS {new_cds_coords}, array_shape {shaded.shape}')
        new_data = {
            **new_cds_coords,
            **BokehImageCons._get_array(shaded),
            'clim_update': [with_clims],
        }
        if do_update:
            self.im.raw_update(**new_data)
        return new_data

    @staticmethod
    def direct_sample(xar: xr.DataArray,
                      xrange: tuple[int, int],
                      yrange: tuple[int, int]) -> np.ndarray:
        """
        Take a slice from an xarray based on two
        int tuples for x and y-ranges

        :meta private:
        """
        assert all(isinstance(x, int) for x in xrange)
        assert all(isinstance(y, int) for y in yrange)
        return xar.data[slice(*yrange), slice(*xrange)]

    def maximum_view(self, new_cds):
        """
        When over-zoomed compute the maximum array slice to transfer
        to the browser which contains the current view and does not
        go out of bounds of the array. The computed array slice will
        have the same shape as the current frame size even
        if we are so-far zoomed that we see a smaller array on screen.
        This prevents some future updates if we then zoom/scroll within this
        maximum view/slice.

        Returns a new_cds aligned with the array grid where the
        current view is contained and the pixel offset is applied

        new_cds is assumed to have already been offset

        :meta private:
        """
        # This needs to assume that each index/axis is uniformly sampled
        target_height = self.canvas.plot_height
        target_width = self.canvas.plot_width

        xrange, yrange = self.ranges_from_cds_dict(new_cds, as_int=True)
        # These will be ints as float, so map to int
        max_l, max_t, max_r, max_b = map(int, self.array_bounds())
        new_yrange = self.bounded_slice((max_t, max_b), target_height, yrange)
        new_xrange = self.bounded_slice((max_l, max_r), target_width, xrange)
        # these are int ranges for giving to datashader

        w = new_xrange[1] - new_xrange[0]
        h = new_yrange[1] - new_yrange[0]
        # Need to shift anchor point to get correct position in continuous coords
        x0 = new_xrange[0] + self.px_offset
        y0 = new_yrange[0] + self.px_offset
        # x0, y0 = self.shift_anchor(x0, y0)
        return {'x': [x0], 'y': [y0], 'dw': [w], 'dh': [h]}, (new_xrange, new_yrange)

    @staticmethod
    def bounded_slice(full_bounds: tuple[int, int],
                      target_dimension: int,
                      original_bounds: tuple[int, int]) -> tuple[int, int]:
        """
        Return an int slice within full_bounds of length target_dimension
        ideally centred on original_bounds, otherwise shifted to lie
        within full_bounds

        :meta private:
        """
        ll, ul = full_bounds
        if ul - ll <= target_dimension:
            return full_bounds
        half_slice = target_dimension / 2
        centre = sum(original_bounds) / 2
        target_ll, target_ul = centre - half_slice, centre + half_slice
        if target_ll >= ll and target_ul < ul:
            pass
        elif target_ll < ll:
            target_ll = ll
            target_ul = target_ll + target_dimension
        elif target_ul >= ul:
            target_ul = ul
            target_ll = target_ul - target_dimension
        else:
            print(full_bounds, target_dimension, original_bounds)
            raise ValueError('Unexpected bounded_slice behaviour')
        return (int(target_ll), int(target_ul))

    def is_oversampled(self, cds) -> bool:
        """
        Test if the cds width/height gives an array smaller than the canvas size

        :meta private:
        """
        sampled_shape = (cds['dh'][0], cds['dw'][0])
        return all(s <= c for s, c in zip(sampled_shape, self.canvas_shape))

    def in_bounds(self,
                  old_cds_coords: dict[str, list[float]],
                  new_cds_coords: dict[str, list[float]]):
        """
        Test if new_cds_coords are within old_cds_coords
        {'x':[x0], 'y':[y0], 'dw': [w], 'dh': [h]}
        y0 is the bottom coordinate!!!

        :meta private:
        """
        o_l, o_t, o_r, o_b = self.cds_to_ltrb(old_cds_coords)
        n_l, n_t, n_r, n_b = self.cds_to_ltrb(new_cds_coords)
        if n_l < o_l:
            return False
        if n_t < o_t:
            return False
        if n_r > o_r:
            return False
        if n_b > o_b:
            return False
        return True

    def matching_bounds(
        self,
        old_cds_coords: dict[str, list[float]],
        new_cds_coords: dict[str, list[float]]
    ) -> bool:
        """
        Test if new_cds_coords bounds match old_cds_coords after being clipped
        to the array dimension. Used to return early if we are really zoomed out.

        :meta private:
        """
        return all(a == b for a, b in zip(self.cds_to_ltrb(old_cds_coords),
                                          self.cds_to_ltrb(new_cds_coords)))

    def data_matches_scale(
        self,
        old_cds_coords: dict[str, list[float]],
        new_cds_coords: dict[str, list[float]],
        viewport_wh: tuple[float, float],
    ) -> bool:
        old_l, old_t, old_r, old_b = self.cds_to_ltrb(old_cds_coords)
        new_l, new_t, new_r, new_b = self.cds_to_ltrb(new_cds_coords)

        is_contained = (
            old_l <= new_l
            and old_t <= new_t
            and old_r >= new_r
            and old_b >= new_b
        )

        if not is_contained:
            # Need new data
            return False

        current_data = self.im.array
        current_array_h, current_array_w = current_data.shape
        viewport_w, viewport_h = viewport_wh

        old_dh = old_cds_coords['dh'][0]
        old_dw = old_cds_coords['dw'][0]
        new_dh = viewport_h
        new_dw = viewport_w

        # These have values of axis units per array unit
        old_data_ratio_h = old_dh / current_array_h
        new_data_ratio_h = new_dh / current_array_h
        old_data_ratio_w = old_dw / current_array_w
        new_data_ratio_w = new_dw / current_array_w

        # i.e. if the new viewport shows fewer array units per screen unit
        if new_data_ratio_w >= old_data_ratio_w and new_data_ratio_h >= old_data_ratio_h:
            return True
        return False

    def _update_array(self, array: np.ndarray):
        if self.array.shape != array.shape:
            self._setup_canvas(array)
        else:
            # Replace the data
            self._array_da[:] = array
            self._array_da_minimum = None

    def compute_update(self, array: np.ndarray) -> dict[str, list[float | np.ndarray]]:
        if not self.active:
            # Should never be called, but just in case!
            return {
                **self.full_cds_coords(),
                **BokehImageCons._get_array(array),
            }
        return self.redraw(array, do_update=False)

        # current_cds_dims = self.current_cds_dims()
        # if self.is_oversampled(current_cds_dims):
        #     xrange, yrange = self.ranges_from_cds_dict(current_cds_dims, as_int=True)
        #     image_data = self.direct_sample(self.array, xrange, yrange)
        #     if VERBOSE:
        #         print('Update from direct sample')
        # else:
        #     # Use .reshade and not .shade here to preserve previous data ranges
        #     image_data = self.reshade().data
        #     xrange, yrange = self.ranges_from_cds_dict(current_cds_dims)
        #     if self.is_complete(xrange, yrange):
        #         self._array_da_minimum = image_data.copy()
        #         if VERBOSE:
        #             print('Update is full-view (shaded)')
        #     if VERBOSE:
        #         print('Update from reshade')
        # return {**current_cds_dims, **BokehImageCons._get_array(image_data)}

    def redraw(self, array: np.ndarray, do_update: bool = True):
        h, w = array.shape
        self._update_array(array)
        fig = self.fig
        x_range = fig.x_range
        x0, x1 = x_range.start, x_range.end
        if np.isnan(x0) or np.isnan(x1):
            x0, x1 = 0., float(w)
            if x_range.flipped:
                x0, x1 = x1, x0
        y_range = fig.y_range
        y0, y1 = y_range.start, y_range.end
        if np.isnan(y0) or np.isnan(y1):
            y0, y1 = 0., float(h)
            if y_range.flipped:
                y0, y1 = y1, y0
        # Manually trigger an event
        event = RangesUpdate(
            fig,
            x0=x0,
            x1=x1,
            y0=y0,
            y1=y1,
        )
        return self.update_view(event, force=True, do_update=do_update)
