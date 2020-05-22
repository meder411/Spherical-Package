import torch
from ..util import sphere_to_image_resample_map, sphere_to_cube_resample_map, equirectangular_to_sphere_resample_map, sphere_from_cube_resample_map, equirectangular_from_cube_resample_map, compute_num_vertices
from ..functional import unresample, resample
from ..layer_utils import InterpolationType


def resample_vertex_to_equirectangular(vertices,
                                       image_shape,
                                       order,
                                       nearest=False):
    '''
    Returns a tensor of RGB values that correspond to each vertex of the provided icosphere.

    Computes a color value for each vertex in the provided icosphere by texturing it with the provided image using barycentric interpolation
    '''

    # Get resampling map with barycentric interpolation weights
    sample_map, interp_map = sphere_to_image_resample_map(
        order, image_shape, nearest)
    if vertices.is_cuda:
        sample_map = sample_map.to(vertices.get_device())
        interp_map = interp_map.to(vertices.get_device())

    # Unresample the image to the sphere
    if nearest:
        interp = InterpolationType.NEAREST
    else:
        interp = InterpolationType.BISPHERICAL
    rgb_rect = unresample(vertices, sample_map, interp, interp_map)

    return rgb_rect


# -----------------------------------------------------------------------------
def resample_vertex_from_equirectangular(img, icosphere, nearest=False):
    """
    Assigns information from an equirectangular input to each vertex on an icosphere. This approach samples *from* the equirectangular image, so no barycentric interpolation is applies. The result on the sphere is slightly coarser, but there is no need to worry about a resolution mismatch. This method is also faster.
    """

    if nearest:
        interp = InterpolationType.NEAREST
    else:
        interp = InterpolationType.BISPHERICAL

    # Create the sample map to go from equirectangular image to sphere
    sample_map = equirectangular_to_sphere_resample_map(
        img.shape[-2:], icosphere)

    # Unresample the images to vertices
    return unresample(img, sample_map, interp)


def resample_equirectangular_to_vertex(img, order, nearest=False):
    '''
    Returns a tensor of RGB values that correspond to each vertex of the provided icosphere.

    Computes a color value for each vertex in the provided icosphere by texturing it with the provided image using barycentric interpolation
    '''

    # Get resampling map with barycentric interpolation weights
    sample_map, interp_map = sphere_to_image_resample_map(
        order, img.shape[-2:], nearest)
    if img.is_cuda:
        sample_map = sample_map.to(img.get_device())
        interp_map = interp_map.to(img.get_device())

    if nearest:
        interp = InterpolationType.NEAREST
    else:
        interp = InterpolationType.BISPHERICAL
    rgb_vertices = resample(img, sample_map, (1, compute_num_vertices(order)),
                            interp, interp_map)

    # Normalize color
    sum_weights = torch.zeros(rgb_vertices.shape[-1])
    if img.is_cuda:
        sum_weights = sum_weights.cuda()
    sum_weights.index_add_(0, sample_map[..., 0].long().view(-1),
                           interp_map.view(-1))
    rgb_vertices /= (sum_weights + 1e-12)

    return rgb_vertices


# -----------------------------------------------------------------------------


def resample_cube_to_vertex(cube, order, nearest=False):
    '''
    Returns a tensor of RGB values that correspond to each vertex of the provided icosphere.

    Computes a color value for each vertex in the provided icosphere by texturing it with the provided image using barycentric interpolation
    '''

    # Get resampling map with barycentric interpolation weights
    sample_map, interp_map = sphere_to_cube_resample_map(
        order, cube.shape[-2], nearest)
    if cube.is_cuda:
        sample_map = sample_map.to(cube.get_device())
        interp_map = interp_map.to(cube.get_device())

    # Resample the image to the sphere
    if nearest:
        interp = InterpolationType.NEAREST
    else:
        interp = InterpolationType.BILINEAR
    rgb_vertices = resample(cube, sample_map, (1, compute_num_vertices(order)),
                            interp, interp_map)

    # Normalize color
    sum_weights = torch.zeros(rgb_vertices.shape[-1])
    if cube.is_cuda:
        sum_weights = sum_weights.cuda()
    sum_weights.index_add_(0, sample_map[..., 0].long().view(-1),
                           interp_map.view(-1))
    rgb_vertices /= (sum_weights + 1e-12)

    return rgb_vertices


def resample_equirectangular_from_cube(
        cube, output_shape, interpolation=InterpolationType.BILINEAR):
    """
    cube: * x * x cube_dim x 6 * cube_dim
    """

    # Dimension of a cube face
    cube_dim = cube.shape[-2]

    # Create the sample map
    sample_map = equirectangular_from_cube_resample_map(cube_dim, output_shape)

    # Put the map on the GPU if the cube map is already there
    if cube.is_cuda:
        sample_map = sample_map.to(cube.get_device())

    # Perform the unresampling
    return unresample(cube, sample_map, interpolation)


def resample_sphere_from_cube(cube,
                              order,
                              interpolation=InterpolationType.BILINEAR):
    """
    cube: * x * x cube_dim x 6 * cube_dim
    """

    # Dimension of a cube face
    cube_dim = cube.shape[-2]

    # Create the sample map
    sample_map = sphere_from_cube_resample_map(cube_dim, order)

    # Put the map on the GPU if the cube map is already there
    if cube.is_cuda:
        sample_map = sample_map.to(cube.get_device())

    # Perform the unresampling
    return unresample(cube, sample_map, interpolation)
