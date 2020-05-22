import torch.nn as nn

from ..util import tangent_image_dim, create_equirectangular_to_tangent_images_sample_map, create_tangent_images_to_equirectangular_uv_sample_map
from ..functional import unresample, uv_resample
from ..layer_utils import InterpolationType


class CreateTangentImages(nn.Module):
    """Creates tangent images"""

    def __init__(self, image_shape, base_order, sample_order):

        super(CreateTangentImages, self).__init__()

        # Sample map
        self.register_buffer(
            'sample_map',
            create_equirectangular_to_tangent_images_sample_map(
                image_shape, base_order, sample_order))

        # Dimension of square tangent image grid
        self.grid_dim = tangent_image_dim(base_order, sample_order,
                                            kernel_size)

    def forward(self, x, interpolation=InterpolationType.BISPHERICAL):
        '''
        Creates a set of tangent images from an equirectangular input

        x: B x C x H x W

        returns B x C x F_base x H x W
        '''

        # Resample the image to the tangent planes as (B x F_base x C x num_samples^2)
        tangent_images = unresample(x, self.sample_map, interpolation)

        # Reshape
        B, C, N = tangent_images.shape[:3]
        return tangent_images.view(B, C, N, self.grid_dim, self.grid_dim)


class TangentImagesToEquirectangular(nn.Module):

    def __init__(self, image_shape, base_order, sample_order):

        super(TangentImagesToEquirectangular, self).__init__()

        # Create the quad/UV mapping sample map
        quad, uv = create_tangent_images_to_equirectangular_uv_sample_map(
            image_shape, base_order, sample_order)

        # Register the quad indices and UV coords as a buffer to bind it to module
        self.register_buffer('quad', quad)
        self.register_buffer('uv', uv)

        # Store number of faces
        self.num_faces = compute_num_faces(base_order)

    def forward(self, x, interpolation=InterpolationType.BILINEAR):
        '''
        x: B x C x F x N x N tensor
        '''
        # Expand the tensor to a separate patches
        B = x.shape[0]
        N = x.shape[-1]
        x = x.view(B, -1, self.num_faces, N, N)

        # Resample the image to the tangent planes as (B x C x OH x OW)
        return uv_resample(x, self.quad, self.uv, interpolation)