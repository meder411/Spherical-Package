from ..util import create_equirectangular_to_tangent_images_sample_map, create_tangent_images_to_equirectangular_uv_sample_map, tangent_image_dim, compute_num_faces
from ..functional import unresample, uv_resample
from ..layer_utils import InterpolationType


def create_tangent_images(
        input,  # [B] x C x H x W
        base_order,
        sample_order,
        interpolation=InterpolationType.BISPHERICAL):

    assert input.dim() in [3, 4], \
        'input must be a 3D or 4D tensor (input.dim() == {})'.format(input.dim())

    sample_map = create_equirectangular_to_tangent_images_sample_map(
        input.shape[-2:], base_order, sample_order)

    if input.is_cuda:
        sample_map = sample_map.to(input.get_device())

    # Resample to the tangent images
    tangent_images = unresample(input, sample_map, interpolation)

    # Reshape to a separate each patch
    num_samples = tangent_image_dim(base_order, sample_order)
    tangent_images = tangent_images.view(*tangent_images.shape[:-1],
                                         num_samples, num_samples)

    return tangent_images


def tangent_images_to_equirectangular(
        tangent_images,  # [B] x C x faces x D x D
        image_shape,
        base_order,
        sample_order,
        interpolation=InterpolationType.BILINEAR):

    assert tangent_images.dim() in [4, 5], \
        'tangent_images must be a 4D or 5D tensor (tangent_images.dim() == {})'.format(tangent_images.dim())

    quad, uv = create_tangent_images_to_equirectangular_uv_sample_map(
        image_shape, base_order, sample_order)

    if tangent_images.is_cuda:
        quad = quad.to(tangent_images.get_device())
        uv = uv.to(tangent_images.get_device())

    # Resample the tangent images back to an equirectangular image
    output = uv_resample(tangent_images, quad, uv, interpolation)

    return output