import torch
from ..functional import distort, DistortionType
from ..layer_utils import InterpolationType


class Distortion(object):

    def __init__(self,
                 params,
                 dist_type=DistortionType.BROWN,
                 crop=True,
                 keep_shape=True,
                 interpolation=InterpolationType.BILINEAR):
        """
        params: k dimensional parameter tensor
        crop:   True to crop out blank pixels during distortion (i.e. all 
                resulting pixels have values), False to include all pixels from input image
        keep_shape: Crop or zero-pad output to maintain the original shape of 
                the input
        """

        # Make sure the parameters are float64
        self.params = params.double()

        # Cropping instructions
        self.crop = crop
        self.keep_shape = keep_shape

        # Assign distortion function based on the requested distortion type
        self.dist_type = dist_type

        # Type of interpolation to use
        self.interpolation = interpolation

    def __call__(self, x):
        return distort(x, self.params, self.dist_type, self.crop,
                       self.keep_shape, self.interpolation)


class DistortionSimpleRadial(Distortion):
    """
    Adds radial distortion according to a single k1 parameter
    """

    def __init__(self,
                 k1=0,
                 crop=True,
                 keep_shape=True,
                 interpolation=InterpolationType.BILINEAR):
        super().__init__(
            torch.tensor([k1]), DistortionType.SIMPLE_RADIAL, crop, keep_shape,
            interpolation)

    def __call__(self, x):
        return super(DistortionSimpleRadial, self).__call__(x)


class RandomDistortionSimpleRadial(Distortion):
    """
    Adds radial distortion according to uniformly random k1 parameter within a provided range
    """

    def __init__(self,
                 k1_range,
                 crop=True,
                 keep_shape=True,
                 interpolation=InterpolationType.BILINEAR):
        super().__init__(
            torch.tensor([0.0]), DistortionType.SIMPLE_RADIAL, crop,
            keep_shape, interpolation)

        assert type(k1_range) in [tuple, list] and len(k1_range) == 2, \
            'k1_range expects a tuple or list as (min_k1, max_k1), inclusive'

        self.k1_range = k1_range

    def __call__(self, x):
        # Randomly sample k1 uniformly from k1_range
        min_k1 = self.k1_range[0]
        max_k1 = self.k1_range[1]
        diff = max_k1 - min_k1
        self.params = (torch.rand(1) * diff + min_k1).double()
        return super(RandomDistortionSimpleRadial, self).__call__(x)


class DistortionBrown(Distortion):

    def __init__(self,
                 k1=0,
                 k2=0,
                 k3=0,
                 t1=0,
                 t2=0,
                 crop=True,
                 keep_shape=True,
                 interpolation=InterpolationType.BILINEAR):
        super().__init__(
            torch.tensor([k1, k2, k3, t1, t2]), DistortionType.BROWN, crop,
            keep_shape, interpolation)

    def __call__(self, x):
        return super(DistortionBrown, self).__call__(x)


class DistortionFisheye(Distortion):

    def __init__(self,
                 k1=0,
                 k2=0,
                 k3=0,
                 k4=0,
                 crop=True,
                 keep_shape=True,
                 interpolation=InterpolationType.BILINEAR):
        super().__init__(
            torch.tensor([k1, k2, k3, k4]), DistortionType.FISHEYE, crop,
            keep_shape, interpolation)

    def __call__(self, x):
        return super(DistortionFisheye, self).__call__(x)