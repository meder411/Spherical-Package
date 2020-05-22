import torch
import math

from .grids import *
from .conversions import *

# =============================================================================
# Equirectangular mapping functions
# =============================================================================
#
# Note that there is no concept of padding for spherical images because there
# are no image boundaries.
# #


def equirectangular_kernel(shape, kernel_size, dilation=1):
    """
    Returns a kernel sampling grid with angular spacing according to the provided shape (and associated computed angular resolution) of an equirectangular image

    shape: (H, W)
    kernel_size: (kh, kw)
    """
    # For convenience
    kh, kw = kernel_size

    # Get equirectangular grid resolution
    res_lon, res_lat = get_equirectangular_grid_resolution(shape)

    # Build the kernel according to the angular resolution of the equirectangular image
    dlon = torch.zeros(kernel_size)
    dlat = torch.zeros(kernel_size)
    for i in range(kh):
        cur_i = i - (kh // 2)
        for j in range(kw):
            cur_j = j - (kw // 2)
            dlon[i, j] = cur_j * dilation * res_lon
            # Flip sign is because +Y is down
            dlat[i, j] = cur_i * dilation * -res_lat

    # Returns the kernel differentials as kh x kw
    return dlon, dlat


def grid_projection_map(shape, kernel_size, stride=1, dilation=1):
    # For convenience
    H, W = shape
    kh, kw = kernel_size

    # Get lat/lon mesh grid and resolution
    lon, lat = spherical_meshgrid(shape)

    # Get the kernel differentials
    dlon, dlat = equirectangular_kernel(shape, kernel_size, dilation)

    # Equalize views
    lat = lat.view(H, W, 1)
    lon = lon.view(H, W, 1)
    dlon = dlon.view(1, 1, kh * kw)
    dlat = dlat.view(1, 1, kh * kw)

    # Compute the "projection"
    map_lat = lat + dlat
    map_lon = lon + dlon

    # Convert the spherical coordinates to pixel coordinates
    # H x W x KH*KW x 2
    map_pixels = convert_spherical_to_image(
        torch.stack((map_lon, map_lat), -1), shape)

    # Adjust the stride of the map accordingly
    map_pixels = map_pixels[::stride, ::stride, ...].contiguous()

    # Return the pixel sampling map
    # H x W x KH*KW x 2
    return map_pixels


def inverse_gnomonic_projection_map(shape, kernel_size, stride=1, dilation=1):
    # For convenience
    H, W = shape
    kh, kw = kernel_size

    # Get lat/lon mesh grid and resolution
    lon, lat = spherical_meshgrid(shape)

    # Get the kernel differentials
    dlon, dlat = equirectangular_kernel(shape, kernel_size, dilation)

    # Equalize views
    lat = lat.view(H, W, 1)
    lon = lon.view(H, W, 1)
    dlon = dlon.view(1, 1, kh * kw)
    dlat = dlat.view(1, 1, kh * kw)

    # Compute the inverse gnomonic projection of each tangent grid (the kernel) back onto sphere at each pixel of the equirectangular image.
    rho = (dlon**2 + dlat**2).sqrt()
    nu = rho.atan()
    map_lat = (nu.cos() * lat.sin() + dlat * nu.sin() * lat.cos() / rho).asin()
    map_lon = lon + torch.atan2(
        dlon * nu.sin(),
        rho * lat.cos() * nu.cos() - dlat * lat.sin() * nu.sin())

    # Handle the (0,0) case
    map_lat[..., [kh * kw // 2]] = lat
    map_lon[..., [kh * kw // 2]] = lon

    # Compensate for longitudinal wrap around
    map_lon = ((map_lon + math.pi) % (2 * math.pi)) - math.pi

    # Convert the spherical coordinates to pixel coordinates
    # H x W x KH*KW x 2
    map_pixels = convert_spherical_to_image(
        torch.stack((map_lon, map_lat), -1), shape)

    # Adjust the stride of the map accordingly
    map_pixels = map_pixels[::stride, ::stride, ...].contiguous()

    # Return the pixel sampling map
    # H x W x KH*KW x 2
    return map_pixels


def inverse_equirectangular_projection_map(shape,
                                           kernel_size,
                                           stride=1,
                                           dilation=1):
    # For convenience
    H, W = shape
    kh, kw = kernel_size

    # Get lat/lon mesh grid and resolution
    lon, lat = spherical_meshgrid(shape)

    # Get the kernel differentials
    dlon, dlat = equirectangular_kernel(shape, kernel_size, dilation)

    # Equalize views
    lat = lat.view(H, W, 1)
    lon = lon.view(H, W, 1)
    dlon = dlon.view(1, 1, kh * kw)
    dlat = dlat.view(1, 1, kh * kw)

    # Compute the inverse equirectangular projection of each tangent grid (the kernel) back onto sphere at each pixel of the equirectangular image.
    # Compute the projection back onto sphere
    map_lat = lat + dlat
    map_lon = lon + dlon / map_lat.cos()

    # Compensate for longitudinal wrap around
    map_lon = ((map_lon + math.pi) % (2 * math.pi)) - math.pi

    # Convert the spherical coordinates to pixel coordinates
    # H x W x KH*KW x 2
    map_pixels = convert_spherical_to_image(
        torch.stack((map_lon, map_lat), -1), shape)

    # Adjust the stride of the map accordingly
    map_pixels = map_pixels[::stride, ::stride, ...].contiguous()

    # Return the pixel sampling map
    # H x W x KH*KW x 2
    return map_pixels


# =============================================================================
# Cube map mapping functions
# =============================================================================


def cube_kernel(cube_dim, kernel_size, dilation=1):
    """
    Returns a kernel sampling grid with angular spacing according to the provided cube dimension (and associated computed angular resolution) of a cube map

    cube_dim: length of side of square face of cube map
    kernel_size: (kh, kw)
    """
    # For convenience
    kh, kw = kernel_size

    cube_res = 1 / cube_dim

    # Build the kernel according to the angular resolution of the cube face
    dx = torch.zeros(kernel_size)
    dy = torch.zeros(kernel_size)
    for i in range(kh):
        cur_i = i - (kh // 2)
        for j in range(kw):
            cur_j = j - (kw // 2)
            dx[i, j] = cur_j * dilation * cube_res
            # Flip sign is because +Y is down
            dy[i, j] = cur_i * dilation * -cube_res

    # Returns the kernel differentials as kh x kw
    return dx, dy


def inverse_cube_face_projection_map(cube_dim,
                                     kernel_size,
                                     stride=1,
                                     dilation=1,
                                     polar=False):
    """
    Creates a sampling map which models each face of the cube as an gnomonic projection (equatorial aspect) of the sphere. Warps the kernel according to the inverse gnomonic projection for the face.
    """
    # For convenience
    kh, kw = kernel_size

    # Get a meshgrid of a cube face in terms of spherical coordinates
    face_lon, face_lat = cube_face_spherical_meshgrid(cube_dim, polar)

    # Get the kernel differentials
    dx, dy = cube_kernel(cube_dim, kernel_size, dilation)

    # Equalize views
    face_lat = face_lat.view(cube_dim, cube_dim, 1)
    face_lon = face_lon.view(cube_dim, cube_dim, 1)
    dx = dx.view(1, 1, kh * kw)
    dy = dy.view(1, 1, kh * kw)

    # Compute the inverse gnomonic projection of each tangent grid (the kernel) back onto sphere at each pixel of the cube face
    rho = (dx**2 + dy**2).sqrt()
    nu = rho.atan()
    map_lat = (nu.cos() * face_lat.sin() +
               dy * nu.sin() * face_lat.cos() / rho).asin()
    map_lon = face_lon + torch.atan2(
        dx * nu.sin(),
        rho * face_lat.cos() * nu.cos() - dy * face_lat.sin() * nu.sin())

    # Handle the (0,0) case
    map_lat[..., [kh * kw // 2]] = face_lat
    map_lon[..., [kh * kw // 2]] = face_lon

    # Create the sample map in terms of spherical coordinates
    map_face = torch.stack((map_lon, map_lat), -1)

    # Convert the cube coordinates on the sphere to pixels in the cube map
    map_pixels = convert_spherical_to_cube_face(map_face, cube_dim)

    # Adjust the stride of the map accordingly
    map_pixels = map_pixels[::stride, ::stride, ...].contiguous()

    # Return the pixel sampling map
    # cube_dime x cube_dim x KH*KW x 2
    return map_pixels