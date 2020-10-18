import torch.nn as nn
import torch.nn.functional as F

import math

from .conversions import *
from .grids import *
from .icosahedron_functions import *
from .spherical_projections import forward_gnomonic_projection, inverse_gnomonic_projection
from .util import points_in_triangle_2d
import _spherical_distortion_ext._mesh as _mesh
from _spherical_distortion_ext._enums import InterpolationType

# -----------------------------------------------------------------------------


def tangent_image_dim(base_order, sample_order):
    '''Computes the number of samples for a tangent image dimension.'''
    return 2**(sample_order - base_order)


# -----------------------------------------------------------------------------


def get_sampling_resolution(base_order):
    '''
    This returns the angular resolution of the <base_order> - 1 icosahedron. Using b-1 ensures that the spatial extent of the tangent image will cover the entire triangular face.

    Note: After level 4, the vertex resolution comes pretty close to exactly halving at each subsequent order. This means we don't need to generate the sphere to compute the resolution. However, at lower levels of subdivision, we ought to compute the vertex resolution as it's not fixed.
    '''
    if base_order < 5:
        sampling_resolution = generate_icosphere(max(
            0, base_order - 1)).get_angular_resolution()
        if base_order == 0:
            sampling_resolution *= 2
    else:
        sampling_resolution = generate_icosphere(4).get_angular_resolution()
        sampling_resolution /= (2**(base_order - 5))
    return sampling_resolution


# -----------------------------------------------------------------------------


def compute_tangent_image_angular_resolution(corners):
    '''
    corners: num_tangent_images x 4 x 3 (3d points)
    '''
    A = F.normalize(corners[..., 0, :], dim=-1)
    B = F.normalize(corners[..., 1, :], dim=-1)
    C = F.normalize(corners[..., 2, :], dim=-1)
    D = F.normalize(corners[..., 3, :], dim=-1)
    fov_x = (torch.acos((A * B).sum(-1)) * 180 / math.pi).mean()
    fov_y = (torch.acos((A * C).sum(-1)) * 180 / math.pi).mean()

    return fov_x, fov_y


# -----------------------------------------------------------------------------


def tangent_images_spherical_sample_map(base_order, sample_order):

    assert sample_order >= base_order, 'Sample order must be greater than or equal to the base order ({} <{ })'.format(
        sample_order, base_order)

    # Generate the base icosphere
    base_sphere = generate_icosphere(base_order)

    # Get sampling resolution
    sampling_resolution = get_sampling_resolution(base_order)

    # Determine how many samples to grab in each direction
    num_samples = tangent_image_dim(base_order, sample_order)

    # Generate spherical sample map s.t. each face is projected onto a tangent grid of size (num_samples x num_samples) and the samples are spaced (sampling_resolution/num_samples x sampling_resolution/num_samples apart)
    spherical_sample_map = gnomonic_kernel_from_sphere(
        base_sphere,
        num_samples,
        num_samples,
        sampling_resolution / num_samples,
        sampling_resolution / num_samples,
        source='face')

    return spherical_sample_map


# -----------------------------------------------------------------------------


def create_equirectangular_to_tangent_images_sample_map(
    image_shape, base_order, sample_order):

    assert sample_order >= base_order, 'Sample order must be greater than or equal to the base order ({} <{ })'.format(
        sample_order, base_order)

    # Generate the spherical sample map
    spherical_sample_map = tangent_images_spherical_sample_map(
        base_order, sample_order)

    # Produces a sample map to turn the image into tangent planes
    image_sample_map = convert_spherical_to_image(spherical_sample_map,
                                                  image_shape)

    # Returns F_base x num_samples^2 x 2 sample map
    return image_sample_map.squeeze(0)


# -----------------------------------------------------------------------------


def create_tangent_images_to_equirectangular_uv_sample_map(
    image_shape, base_order, sample_order):
    # Create base icosphere
    icosphere = generate_icosphere(base_order)

    # Compute number of samples and sampling resolution based on base and sample orders
    num_samples = tangent_image_dim(base_order, sample_order)
    sampling_resolution = get_sampling_resolution(base_order)

    # Find the boundaries of the tangent planes in 3D
    corners = tangent_image_corners(base_order, sample_order)

    # Compute the rays for each pixel in the equirectangular image
    rays = torch.stack(spherical_meshgrid(image_shape), -1).view(-1, 2)

    # Find the intersection points of the rays with the tangent images
    # Note: This can probably be done analytically (TODO)
    quad, uv = _mesh.find_tangent_plane_intersections(corners, rays)

    # Reshape quad and uv back to image dims
    # Scale normalized UV coords to actual (floating point) pixel coords
    quad = quad.view(*image_shape)
    uv = uv.view(*image_shape, 2) * (num_samples - 1)

    return quad, uv


# -----------------------------------------------------------------------------


def tangent_image_centers(base_order):
    # Get the centers of each face in spherical coordinates
    return generate_icosphere(base_order).get_face_barycenters()


# -----------------------------------------------------------------------------


def tangent_image_corners(base_order, sample_order):

    # Tangent image dimension
    dim = tangent_image_dim(base_order, sample_order)

    # Number of tangent images
    num_faces = compute_num_faces(base_order)

    # Sampling resolution
    sampling_resolution = get_sampling_resolution(base_order)

    # Get the centers of each face in spherical coordinates
    centers = convert_3d_to_spherical(tangent_image_centers(base_order))

    # Corners of tangent planes in 3D coordinates
    # The "minus 1" term means that the center of the TL pixel is (0,0)
    corners = gnomonic_kernel(centers, 2, 2,
                              sampling_resolution * (dim - 1) / dim,
                              sampling_resolution * (dim - 1) / dim)

    # Convert the corners to 3D coordinates
    corners = convert_spherical_to_3d(corners).squeeze()

    return corners


def face_corners_on_tangent_images(base_order, sample_order, face_idx=None):
    """
    Returns the x,y coordinates of the icosahedron face projected onto the tangent images
    """
    # Vertices of the face in spherical coords
    spherical_vert = convert_3d_to_spherical(
        generate_icosphere(base_order).get_all_face_vertex_coords())

    # Tangent image corners and center in spherical coords
    spherical_corners = convert_3d_to_spherical(
        tangent_image_corners(base_order, sample_order))
    spherical_center = convert_3d_to_spherical(
        tangent_image_centers(base_order))

    # Select a specific face if desired
    if face_idx is not None:
        assert isinstance(face_idx, int), 'Face index must be an integer'
        assert 0 <= face_idx < compute_num_faces(
            base_order), 'Face index must be in the range [{}, {})'.format(
                0, compute_num_faces(base_order))
        spherical_vert = spherical_vert[face_idx]
        spherical_corners = spherical_corners[face_idx]
        spherical_center = spherical_center[face_idx]

    # Project the face onto the image via forward gnomonic projection
    x_corners, y_corners = forward_gnomonic_projection(
        spherical_corners[..., 0], spherical_corners[..., 1], spherical_center)
    x, y = forward_gnomonic_projection(spherical_vert[..., 0],
                                       spherical_vert[...,
                                                      1], spherical_center)

    # Re-normalize to the image dimensions
    dim = tangent_image_dim(base_order, sample_order)
    x_max = x_corners.max()
    x_min = x_corners.min()
    y_max = y_corners.max()
    y_min = y_corners.min()
    vert_x = renormalize(x, x_min, x_max, 0, dim)
    vert_y = renormalize(y, y_min, y_max, 0, dim)
    vert_xy = torch.stack((vert_x, vert_y), -1)

    # Returns the (x,y) in each face
    return vert_xy


def compute_icosahedron_face_mask(base_order, sample_order, face_idx=None):
    """
    Returns the mask of points inside the projection of the icosahedral face onto the correpsonding tangent images
    """
    # Get corners of face triangles as 2D coordinates on the tangent image
    vert_xy = face_corners_on_tangent_images(base_order, sample_order,
                                             face_idx)

    dim = tangent_image_dim(base_order, sample_order)
    xy_coords = torch.stack(meshgrid((dim, dim)), -1)

    if vert_xy.dim() == 3:
        masks = []
        for i in range(vert_xy.shape[0]):
            masks.append(points_in_triangle_2d(xy_coords, vert_xy[i]))
        return torch.stack(masks, 0)
    else:
        return points_in_triangle_2d(xy_coords, vert_xy)


def get_valid_coordinates(base_order,
                          sample_order,
                          face_idx,
                          coordinates,
                          return_mask=False):
    """
    Given a set of 2D pixel coordinates on the tangent image, returns only those that fall within the valid region

    coordinates: N x 2

    returns K x 2 (K <= N), and optionally the length-N boolean mask
    """
    # Project face vertices onto tangent image
    vert_xy = face_corners_on_tangent_images(base_order, sample_order,
                                             face_idx)

    # Check which of the provided coordinates fall within that triangle
    valid = points_in_triangle_2d(coordinates, vert_xy)

    # Return the subset of points that are valid
    if return_mask:
        return coordinates[valid], valid
    return coordinates[valid]


def convert_tangent_image_coordinates_to_spherical(base_order, sample_order,
                                                   tangent_img_idx, uv):
    """
    base_order: tangent image base order
    sample_order: tangent image sample order
    tangest_img_idx: which tangent image is this (i.e. which face of icosahedron)
    uv: M x 2 matrix of pixel coordinates on tangent image
    """

    # Get the tangent points (i.e. center of each tangent image in spherical coordinates)
    centers = convert_3d_to_spherical(tangent_image_centers(base_order))

    # We will need the dimension and sample resolution of the tangent images for this computation
    dim = tangent_image_dim(base_order, sample_order)
    sample_resolution = get_sampling_resolution(base_order) / dim

    # Convert UV coordinates to angular distance and shift them so the origin is at the center pixel
    un = (uv[:, 0] - dim / 2.0) * sample_resolution + sample_resolution / 2.0
    vn = (uv[:, 1] - dim / 2.0) * sample_resolution + sample_resolution / 2.0

    # Return (lon, lat)
    return inverse_gnomonic_projection(un, vn, centers[tangent_img_idx])