from __future__ import annotations
import numpy as np
import datashader as ds
import xarray as xr
from typing import TYPE_CHECKING

from .image_db import BokehImage

if TYPE_CHECKING:
    from bokeh.events import RangesUpdate

VERBOSE = False


class DatashadeHelper:
    """
    Attributes
    ----------
    _datashade_threshold: float, default 1.05 * (4 * 1024**2)
        Threshold size in bytes below which datashading will
        be disabled, even if this class is used. The size is
        chosen so as to not datashade a 1k x 1k np.float32 image
        which corresponds to 4 MB.
    _datashade_method: str, default 'nearest'
        Method used when downsampling, supported modes are
        :code:`'nearest'` and :code:`'linear'`. This is a limitation
        of the datashader library
    """
    # datashade for images above 1024*1024 @ float32
    _datashade_threshold = 1.05 * (4 * 1024**2)
    _datashade_method = 'nearest'

    def __init__(self, im: BokehImage, width: int = 400, height: int = 400):
        self._im = im
        self._canvas = ds.Canvas(
            plot_height=height,
            plot_width=width,
        )

        array: np.ndarray = self.im.cds.data['image'][0]
        h, w = array.shape
        ys = np.arange(h).astype(float)
        xs = np.arange(w).astype(float)
        self._array_da = xr.DataArray(
            array,
            coords=[('y', ys), ('x', xs)],
        )

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
    def canvas(self) -> ds.Canvas:
        """
        The Datashader object which does the re-rasterizing

        :meta private:
        """
        return self._canvas

    @property
    def canvas_shape(self):
        return (self.canvas.plot_height, self.canvas.plot_width)

    def shade(self, xrange=None, yrange=None):
        """
        Shades the display_image with new x- and y- ranges
        ranges with None values will shade the whole dimension

        :meta private:
        """
        self.set_canvas_ranges(xrange=xrange, yrange=yrange)
        return self.reshade()

    def set_canvas_ranges(self, xrange=None, yrange=None):
        self.canvas.x_range = xrange
        self.canvas.y_range = yrange

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
            interpolate=self._datashade_method,
            agg='first',
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

    def unpack_event(self, event: RangesUpdate) -> tuple[bool, dict[str, list[float]]]:
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
        x0 = event.x0
        y0 = event.y0
        x1 = event.x1
        y1 = event.y1
        if VERBOSE:
            print(f'Event data {x0, y0, x1, y1}')
        # The full array is bounded in continuous coordinates
        # by the following geometry (includes self.px_offset)
        l, b, r, t = self.continuous_bounds()
        # Test if our viewport overlaps with the full array
        # Used for an early return if we don't have anything to show
        # print(f'LR: {(l, r)}, {(x0, x1)}')
        # print(f'BT: {(b, t)}, {(y0, y1)}')
        is_visible = (self.axis_overlaps((l, r), (x0, x1))
                      and self.axis_overlaps((b, t), (y0, y1)))
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
        return is_visible, {'x': [x0], 'y': [y0], 'dw': [w], 'dh': [h]}

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

    @staticmethod
    def axis_overlaps(obj_range, view_range):
        return (obj_range[0] <= view_range[1]) and (view_range[0] <= obj_range[1])

    def update_view(self, event: RangesUpdate):
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
        """
        # Need to thorougly test this function for off-by-one errors...
        is_visible, new_cds_coords = self.unpack_event(event)
        if not is_visible:
            # Do nothing
            if VERBOSE:
                print('Out of bounds, skipping')
            return

        if self.matching_bounds(self.current_cds_dims(), new_cds_coords):
            # Returns when the new displayed bounds match the old bounds
            # Implicitly there is nothing to do here
            # Covers the case when we zoom out from the full shaded image
            if VERBOSE:
                print('Matching bounds, skip update')
            return

        if self.is_oversampled(new_cds_coords):
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
            shaded = self.shade(xrange=xrange, yrange=yrange).data
            if VERBOSE:
                print('Update from shade')

        if VERBOSE:
            print(f'New CDS {new_cds_coords}, array_shape {shaded.shape}')
        new_data = {
            **new_cds_coords,
            'image': [shaded],  # 'image': [shaded[::-1, :]]}
        }
        self.im.raw_update(**new_data)

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

    def matching_bounds(self,
                        old_cds_coords: dict[str, list[float]],
                        new_cds_coords: dict[str, list[float]]):
        """
        Test if new_cds_coords bounds match old_cds_coords after being clipped
        to the array dimension. Used to return early if we are really zoomed out.

        :meta private:
        """
        return all(a == b for a, b in zip(self.cds_to_ltrb(old_cds_coords),
                                          self.cds_to_ltrb(new_cds_coords)))

    # def get_shaded_datadict(self, force_full=False):
    #     """
    #     Called when updating the image data
    #     Will shade and return the CDS with the same display/view
    #     as the previous CDS, except for when initialising the plot
    #     where the shaded image/geometry will be for the whole array

    #     :meta private:
    #     """
    #     if self.cds is None or force_full:
    #         image_data = self.shade()
    #         return self.get_full_datadict(image_data=image_data.data)
    #     else:
    #         current_cds_dims = self.current_cds_dims()
    #         if self.is_oversampled(current_cds_dims):
    #             xrange, yrange = self.ranges_from_cds_dict(current_cds_dims, as_int=True)
    #             image_data = self.direct_sample(self.array, xrange, yrange)
    #         else:
    #             # Use .reshade and not .shade here to preserve previous data ranges
    #             image_data = self.reshade().data
    #         return {**current_cds_dims, 'image': [image_data[::-1, :]]}

    # def _update_data_ranges(self, event: RangesUpdate):
    #     fig = event.model
    #     x_flipped = fig.x_range.flipped
    #     y_flipped = fig.y_range.flipped
    #     full_h, full_w = self._array_da.shape
    #     x0, x1 = event.x0, event.x1
    #     if x_flipped:
    #         x0, x1 = x1, x0
    #     y0, y1 = event.y0, event.y1
    #     if y_flipped:
    #         y0, y1 = y1, y0
    #     x0 -= 1.
    #     y0 -= 1.
    #     x1 += 1.
    #     y1 += 1.
    #     x0 = max(0, x0)
    #     y0 = max(0, y0)
    #     x1 = min(full_h - 1, x1)
    #     y1 = min(full_w - 1, y1)
    #     self._canvas.x_range = (x0, x1)
    #     self._canvas.y_range = (y0, y1)
    #     sampled: 'xr.DataArray' = self._canvas.raster(
    #         self._array_da,
    #         interpolate='nearest',
    #         agg='first',
    #     )
    #     self.raw_update(
    #         x=[x0 + self._px_offset],
    #         dw=[x1 - x0],
    #         y=[y0 + self._px_offset],
    #         dh=[y1 - y0],
    #         image=[sampled.to_numpy()]
    #     )