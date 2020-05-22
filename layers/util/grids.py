import torch
import math


def get_equirectangular_grid_resolution(shape):
    '''Returns the resolution between adjacency grid indices for an equirectangular grid'''
    H, W = shape
    res_lon = 2 * math.pi / W
    res_lat = math.pi / (H - 1)
    return res_lon, res_lat


def spherical_meshgrid(shape,
                       lon_limit=(-math.pi, math.pi),
                       lat_limit=(-math.pi / 2, math.pi / 2)):
    """
    Default limits are according to an equirectangular image
    """
    H, W = shape
    lon = torch.linspace(
        lon_limit[0], lon_limit[1], steps=W + 1)[:-1].view(1, -1).expand(
            H, -1)
    lat = torch.linspace(
        lat_limit[0], lat_limit[1], steps=H).view(-1, 1).expand(-1, W)
    return lon, lat


def meshgrid(shape):
    """
    Wrapper for PyTorch's meshgrid function
    """
    H, W = shape
    y, x = torch.meshgrid(torch.arange(H), torch.arange(W))
    return x, y


def get_cube_face_resolution(cube_dim):
    '''Returns the angular resolution per pixel of a cube map face'''
    return math.pi / (2.0 * cube_dim)


def cube_face_spherical_meshgrid(cube_dim, polar=False):
    """
    Returns a mesh grid of a face of a cube in spherical coordinates, assuming that the center pixel of the image is (0,0). This is appropriate for cube maps as each face uses the same equatorial aspect of the gnomonic projection.
    """
    if not polar:
        lat = torch.linspace(
            -math.pi / 4, math.pi / 4, steps=cube_dim).view(-1, 1).expand(
                -1, cube_dim)
        lon = torch.linspace(
            -math.pi / 4, math.pi / 4, steps=cube_dim).view(1, -1).expand(
                cube_dim, -1)
    else:
        lat = torch.linspace(
            math.pi / 4, 3 * math.pi / 4, steps=cube_dim).view(-1, 1).expand(
                -1, cube_dim)
        lon = torch.linspace(
            0, math.pi, steps=cube_dim).view(1, -1).expand(cube_dim, -1)
    return lon, lat


def cube_meshgrid(cube_dim):
    '''
    returns (u, v, index)
    '''
    H = cube_dim
    W = cube_dim * 6

    v = torch.arange(cube_dim).view(-1, 1).expand(-1, W)
    u = torch.arange(cube_dim).view(1, -1).expand(H, -1).repeat(1, 6)
    index = torch.zeros((H, W))
    for i in range(6):
        index[:, i * cube_dim:(i + 1) * cube_dim] += i

    return u, v, index


def uniformly_spaced_samples(shape,
                             xlim,
                             ylim,
                             include_boundary=(False, False)):
    """
    Provides a set of samples in 2D such that all samples are evenly spaced between each other and (optionally) the image boundaries in each dimension
    """
    H, W = shape
    xmin, xmax = xlim
    ymin, ymax = ylim
    include_boundary_x, include_boundary_y = include_boundary
    if not include_boundary_x:
        x = torch.linspace(xmin, xmax, steps=W + 2)[1:-1]
    else:
        x = torch.linspace(xmin, xmax, steps=W)
    if not include_boundary_y:
        y = torch.linspace(ymin, ymax, steps=H + 2)[1:-1]
    else:
        y = torch.linspace(ymin, ymax, steps=H)
    x = x.view(1, -1).expand(H, -1)
    y = y.view(-1, 1).expand(-1, W)
    return x, y