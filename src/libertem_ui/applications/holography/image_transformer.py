import functools
import numpy as np
from skimage import transform as sktransform


class ImageTransformer:
    """
    Provides method to compute image transformations
    Supports chaining multiple transforms and changes to output shape
    Provides the final transformation matrix via the
    :code:`get_combined_transform()` method

    Based entirely on `skimage.transform`
    """
    def __init__(self, image):
        self._image = image
        self._transforms = []
        self._reshapes = []
        self._frozen_len = -1

    def set_image(self, image):
        self._image = image

    @property
    def transforms(self):
        return self._transforms

    def add_transform(self, *transforms, output_shape=None, frozen=False):
        self.transforms.append(self._combine_transforms(*transforms))
        self._reshapes.append(output_shape)
        if frozen:
            self._frozen_len = len(self.transforms)

    def add_null_transform(self, output_shape=None, frozen=False):
        self.transforms.append(self._null_transform())
        self._reshapes.append(output_shape)
        if frozen:
            self._frozen_len = len(self.transforms)

    def remove_transform(self, n=1):
        for _ in range(n):
            if len(self.transforms) <= self._frozen_len:
                break
            try:
                self.transforms.pop(-1)
                self._reshapes.pop(-1)
            except (IndexError, TypeError):
                break

    @staticmethod
    def _null_transform():
        return sktransform.EuclideanTransform()

    def clear_transforms(self):
        self.transforms.clear()
        self._reshapes.clear()

    def current_shape(self):
        reshapes = [r for r in self._reshapes if r is not None]
        if reshapes:
            return reshapes[-1]
        return self._image.shape

    def get_combined_transform(self):
        transform_mxs = [t.params for t in self.transforms]
        if not transform_mxs:
            transform_mat = self._null_transform().params
        elif len(transform_mxs) >= 2:
            transform_mat = functools.reduce(np.matmul, transform_mxs)
        else:
            transform_mat = transform_mxs[0]
        combined = sktransform.AffineTransform(matrix=transform_mat)
        return combined

    @staticmethod
    def _combine_transforms(*transforms):
        transforms = [sktransform.AffineTransform(matrix=t)
                      if isinstance(t, np.ndarray)
                      else t for t in transforms]
        if not transforms:
            return ImageTransformer._null_transform()
        elif len(transforms) == 1:
            return transforms[0]
        else:
            transform_mat = functools.reduce(np.matmul, [t.params for t in transforms])
            return sktransform.AffineTransform(matrix=transform_mat)

    def get_transformed_image(self, preserve_range=True, order=None, cval=np.nan, **kwargs):
        if not self.transforms:
            return self._image
        combined_transform = self.get_combined_transform()
        return sktransform.warp(self._image,
                                combined_transform,
                                order=order,
                                output_shape=kwargs.pop('output_shape', self.current_shape()),
                                preserve_range=preserve_range,
                                cval=cval,
                                **kwargs)

    def get_current_center(self):
        current_shape = np.asarray(self.current_shape())
        return current_shape / 2.

    def translate(self, xshift=0., yshift=0., output_shape=None):
        transform = sktransform.EuclideanTransform(translation=(xshift, yshift))
        self.add_transform(transform, output_shape=output_shape)

    def rotate_about_point(self, point_yx, rotation_degrees=None, rotation_rad=None):
        if rotation_degrees and rotation_rad:
            raise ValueError('Cannot specify both degrees and radians')
        elif rotation_degrees:
            rotation_rad = np.deg2rad(rotation_degrees)
        if not rotation_rad:
            if rotation_rad == 0.:
                return
            raise ValueError('Must specify one of degrees or radians')

        transform = sktransform.EuclideanTransform(rotation=rotation_rad)
        self._operation_with_origin(point_yx, transform)

    def rotate_about_center(self, **kwargs):
        current_center = self.get_current_center()
        return self.rotate_about_point(current_center, **kwargs)

    def uniform_scale_centered(self, scale_factor, output_shape=None):
        transform = sktransform.SimilarityTransform(scale=scale_factor)
        current_center = self.get_current_center()
        self._operation_with_origin(current_center, transform)

    def xy_scale_about_point(self, point_yx, xscale=1., yscale=1.):
        transform = sktransform.AffineTransform(scale=(xscale, yscale))
        self._operation_with_origin(point_yx, transform)

    def xy_scale_about_center(self, **kwargs):
        current_center = self.get_current_center()
        return self.xy_scale_about_point(current_center, **kwargs)

    def _operation_with_origin(self, origin_yx, transform):
        origin_xy = np.asarray(origin_yx).astype(float)[::-1]
        forward_shift = sktransform.EuclideanTransform(translation=origin_xy)
        backward_shift = sktransform.EuclideanTransform(translation=-1 * origin_xy)
        self.add_transform(forward_shift, transform, backward_shift)

    @staticmethod
    def available_transforms():
        """The transformation types which can be estimated, compatible with this class"""
        return ['affine', 'euclidean', 'similarity', 'projective']

    def estimate_transform(self, static_points, moving_points,
                           method='affine', output_shape=None, clear=False):
        assert method in self.available_transforms()
        assert static_points.size and moving_points.size, 'Need points to match'
        assert static_points.size == moving_points.size, 'Must supply matching pointsets'
        transform = sktransform.estimate_transform(method,
                                                   static_points.reshape(-1, 2),
                                                   moving_points.reshape(-1, 2))
        if clear:
            self.clear_transforms()
        self.add_transform(transform, output_shape=output_shape)
        return transform
