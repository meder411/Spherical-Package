import torch
import math
"""
All of the spherical to Cartesian conversions in this file transform coordinates according to this coordinate system:

-Y   +Z
 |   /
 |  /
 | /
 |/
 --------- +X

This 3D rectangular coordinate system is consistently used within computer vision, and aligns with the 2D image coordinate system that places the origin at the top-left of an image.

Some useful identities:
* (lon, lat) == (0, 0) along the +X axis
* (lon, lat) == (-pi, 0) along the -X axis
* (lon, lat) == (0, p/2) along the +Y axis
* (lon, lat) == (0, -p/2) along the -Y axis
* (lon, lat) == (pi/2, 0) along the +Z axis
* (lon, lat) == (3*pi/2, 0) along the -Z axis
"""


def renormalize(values, old_min, old_max, new_min, new_max):
    """
    Transforms values in range [old_min, old_max] to values in range [new_min, new_max]
    """
    return (new_max - new_min) * (values - old_min) / (old_max -
                                                       old_min) + new_min


def convert_spherical_to_image(rad, shape):
    '''
    This function converts spherical coordinates to pixel coordinates on an equirectangular image. It assumes the image is laid out as:

           0 <--> -pi            W-1 <--> pi*(W-1/W)
              --------------------------
 0 <--> -pi/2 |                        |
              |                        |
              |                        |
H-1 <--> pi/2 |                        |
              --------------------------

    rad: * x ... x * x 2 (lon, lat)

    returns: * x ... x * x 2 (x,y)
    '''
    H, W = shape
    xy = torch.zeros_like(rad)
    xy[..., 0] = renormalize(rad[..., 0], -math.pi, math.pi, 0, W)
    xy[..., 1] = renormalize(rad[..., 1], -math.pi / 2, math.pi / 2, 0, H - 1)
    return xy


def convert_image_to_spherical(xy, shape):
    '''
    This function converts pixels coordinates:

    [0, H-1] x [0, W-1]

    into spherical coordinates:

    [-pi/2, pi/2] x [-pi, (W-1)*pi/W]

    assuming an equirectangular image. This conversion assumes that the rightmost pixel is comes immediately before the leftmost pixel as the image wraps around a sphere.

    xy: * x ... x * x 2 (x, y)

    returns: * x ... x * x 2 (lon, lat)
    '''
    H, W = shape
    rad = torch.zeros_like(xy)
    rad[..., 0] = renormalize(xy[..., 0], 0, W, -math.pi, math.pi)
    rad[..., 1] = renormalize(xy[..., 1], 0, H - 1, -math.pi / 2, math.pi / 2)
    return rad


def convert_3d_to_spherical(xyz, return_separate=False):
    '''
    xyz : * x 3
    returns : * x 2 (lon, lat)
    '''
    x = xyz[..., 0]
    y = xyz[..., 1]
    z = xyz[..., 2]
    lon = torch.atan2(z, x)
    lat = torch.atan2(-y, (x**2 + z**2).sqrt())
    if return_separate:
        return lon, lat
    return torch.stack((lon, lat), -1)


def convert_spherical_to_3d(lonlat, return_separate=False):
    '''
    lonlat : * x 2 (lon, lat)
    returns : * x 3 (x,y,z)
    '''
    x = lonlat[..., 1].cos() * lonlat[..., 0].cos()
    y = -lonlat[..., 1].sin()
    z = lonlat[..., 1].cos() * lonlat[..., 0].sin()
    if return_separate:
        return x, y, z
    return torch.stack((x, y, z), -1)


def convert_cubemap_tuple_to_pixels(uv, idx, cube_dim):
    '''Converts cubemap coordinates from tuple form (uv, cube_idx) to coordinates on the (cube_dim x 6*cube_dim) cubemap image'''
    u = uv[..., 0]
    v = uv[..., 1]
    idx = idx[..., 0]
    return torch.stack((idx.float() * cube_dim + u, v), -1)


def convert_3d_to_cubemap_pixels(xyz, cube_dim):
    """
    lonlat : * x 3 (x,y,z)

    returns: * x 2 (x,y)
    """
    return convert_cubemap_tuple_to_pixels(*convert_3d_to_cube(xyz, cube_dim),
                                           cube_dim)


def convert_spherical_to_cubemap_pixels(lonlat, cube_dim):
    """
    lonlat : * x 2 (lon, lat)

    returns: * x 2 (x,y)
    """
    return convert_3d_to_cubemap_pixels(convert_spherical_to_3d(lonlat),
                                        cube_dim)


def convert_spherical_to_cube_face(rad, cube_dim, polar=False):
    '''
    This function converts spherical coordinates to pixel coordinates on a face of a cube map. It assumes the face is laid out as:

            0 <--> -pi/4    cube_dim-1 <--> pi/4
                     ------------
        0 <--> -pi/4 |          |
                     |          |
                     |          |
cube_dim-1 <--> pi/4 |          |
                     ------------

    rad: * x ... x * x 2 (lon, lat)

    returns: * x ... x * x 2 (x,y)
    '''
    xy = torch.zeros_like(rad)
    xy[..., 0] = renormalize(rad[..., 0], -math.pi / 4, math.pi / 4, 0,
                             cube_dim - 1)
    if not polar:
        xy[..., 1] = renormalize(rad[..., 1], -math.pi / 4, math.pi / 4, 0,
                                 cube_dim - 1)
    else:
        xy[..., 1] = renormalize(rad[..., 1], math.pi / 4, 3 * math.pi / 4, 0,
                                 cube_dim - 1)
    return xy


def convert_quad_coord_to_uv(quad_shape, coord):
    """
    quad_shape: (H, W)
    coord: N x 2 (X, Y)

    returns: N x 2 (u, v)
    """
    uv = torch.zeros_like(coord)
    H, W = quad_shape
    uv[:, 0] = renormalize(coord[:, 0], 0, W, 0, 1)
    uv[:, 1] = renormalize(coord[:, 1], 0, H, 0, 1)
    return uv


def convert_quad_uv_to_3d(quad_idx, uv, quad_corners):
    """
    quad_idx: M
    uv: M x 2
    quad_corners: N x 4 x 3

    returns: M x 3 3D points
    """
    # Grab the relevant quad data (out: M x 4 x 3)
    relevant_quads = quad_corners[quad_idx, ...]

    # Vectors defining the quads (each out: M x 3)
    u_vec = relevant_quads[:, 1, :] - relevant_quads[:, 0, :]
    v_vec = relevant_quads[:, 2, :] - relevant_quads[:, 0, :]

    # Convenience
    u = uv[:, [0]]
    v = uv[:, [1]]

    # Compute 3D point on the quad as vector addition
    pts_3d = relevant_quads[:, 0, :] + u * u_vec + v * v_vec

    return pts_3d


def convert_cube_to_3d(uv, index, cube_dim):
    '''
    Indexing is [-z, -x, +z, +x, +y, -y]

    Assumes that pixel centers are (u + 0.5, v + 0.5)
    uv : * x 2
    index : * x 1
    returns : * x 3 (xyz)
    '''
    # Convert from cube coord to [0,1]
    uv = uv.float()
    uv = (uv + 0.5) / cube_dim

    # Convert from [0,1] range to [-1,1]
    uv = 2 * uv - 1

    # For convenience
    u = uv[..., 0]
    v = uv[..., 1]

    xyz = torch.zeros(*uv.shape[:-1], 3).float()

    # POSITIVE X
    case_3 = index == 3
    xyz[..., 0][case_3] = 1.0
    xyz[..., 1][case_3] = -v[case_3]
    xyz[..., 2][case_3] = -u[case_3]

    # NEGATIVE X
    case_1 = index == 1
    xyz[..., 0][case_1] = -1.0
    xyz[..., 1][case_1] = -v[case_1]
    xyz[..., 2][case_1] = u[case_1]

    # POSITIVE Y
    case_4 = index == 4
    xyz[..., 0][case_4] = u[case_4]
    xyz[..., 1][case_4] = 1.0
    xyz[..., 2][case_4] = v[case_4]

    # NEGATIVE Y
    case_5 = index == 5
    xyz[..., 0][case_5] = u[case_5]
    xyz[..., 1][case_5] = -1.0
    xyz[..., 2][case_5] = -v[case_5]

    # POSITIVE Z
    case_2 = index == 2
    xyz[..., 0][case_2] = u[case_2]
    xyz[..., 1][case_2] = -v[case_2]
    xyz[..., 2][case_2] = 1.0

    # NEGATIVE Z
    case_0 = index == 0
    xyz[..., 0][case_0] = -u[case_0]
    xyz[..., 1][case_0] = -v[case_0]
    xyz[..., 2][case_0] = -1.0

    return xyz


def convert_3d_to_cube(xyz, cube_dim):
    """
    Assumes cube map faces are indexed is [-z, -x, +z, +x, +y, -y]
    """

    def normalize_coord(u_or_v, max_axis):
        """
        Normalizes (-1.0, 1.0)  --> (0, cube_dim - 1)
        """
        return (cube_dim - 1) * (u_or_v / max_axis + 1.0) / 2

    # For convenience
    x = xyz[..., 0]
    y = xyz[..., 1]
    z = xyz[..., 2]
    abs_x = abs(x)
    abs_y = abs(y)
    abs_z = abs(z)
    is_x_positive = x > 0
    is_y_positive = y > 0
    is_z_positive = z > 0

    uv = torch.zeros(*xyz.shape[:-1], 2).float()
    idx = torch.zeros(*xyz.shape[:-1], 1).float()

    # POSITIVE X
    # u (0 to 1) goes from +z to -z
    # v (0 to 1) goes from +y to -y
    pos_x = is_x_positive & (abs_x >= abs_y) & (abs_x >= abs_z)
    uv[..., 0][pos_x] = normalize_coord(-z[pos_x], abs_x[pos_x])
    uv[..., 1][pos_x] = normalize_coord(-y[pos_x], abs_x[pos_x])
    idx[pos_x] = 3

    # NEGATIVE X
    # u (0 to 1) goes from -z to +z
    # v (0 to 1) goes from +y to -y
    neg_x = (~is_x_positive) & (abs_x >= abs_y) & (abs_x >= abs_z)
    uv[..., 0][neg_x] = normalize_coord(z[neg_x], abs_x[neg_x])
    uv[..., 1][neg_x] = normalize_coord(-y[neg_x], abs_x[neg_x])
    idx[neg_x] = 1

    # POSITIVE Y
    # u (0 to 1) goes from -x to +x
    # v (0 to 1) goes from -z to +z
    pos_y = is_y_positive & (abs_y >= abs_x) & (abs_y >= abs_z)
    uv[..., 0][pos_y] = normalize_coord(x[pos_y], abs_y[pos_y])
    uv[..., 1][pos_y] = normalize_coord(z[pos_y], abs_y[pos_y])
    idx[pos_y] = 4

    # NEGATIVE Y
    # u (0 to 1) goes from -x to +x
    # v (0 to 1) goes from +z to -z
    neg_y = (~is_y_positive) & (abs_y >= abs_x) & (abs_y >= abs_z)
    uv[..., 0][neg_y] = normalize_coord(x[neg_y], abs_y[neg_y])
    uv[..., 1][neg_y] = normalize_coord(-z[neg_y], abs_y[neg_y])
    idx[neg_y] = 5

    # POSITIVE Z
    # u (0 to 1) goes from -x to +x
    # v (0 to 1) goes from +y to -y
    pos_z = is_z_positive & (abs_z >= abs_x) & (abs_z >= abs_y)
    uv[..., 0][pos_z] = normalize_coord(x[pos_z], abs_z[pos_z])
    uv[..., 1][pos_z] = normalize_coord(-y[pos_z], abs_z[pos_z])
    idx[pos_z] = 2

    # NEGATIVE Z
    # u (0 to 1) goes from +x to -x
    # v (0 to 1) goes from +y to -y
    neg_z = (~is_z_positive) & (abs_z >= abs_x) & (abs_z >= abs_y)
    uv[..., 0][neg_z] = normalize_coord(-x[neg_z], abs_z[neg_z])
    uv[..., 1][neg_z] = normalize_coord(-y[neg_z], abs_z[neg_z])
    idx[neg_z] = 0

    return uv, idx


def create_pinhole_camera(height, width):
    """
    Creates a pinhole camera according to height and width, assuming the principal point is in the center of the image
    """
    cx = (width - 1) / 2
    cy = (height - 1) / 2
    f = max(cx, cy)
    return f, cx, cy


def normalize_pinhole_camera(x, y, f, cx, cy):
    """
    Normalizes coordinates to camera coordinates in [-1, 1] x [-1, 1]
    """
    xn = (x - cx) / f
    yn = (y - cy) / f
    return xn, yn


def denormalize_pinhole_camera(xn, yn, f, cx, cy):
    """
    De-normalizes camera coordinates to pixel coordinates, according to provided camera parameters
    """
    x = f * xn + cx
    y = f * yn + cy
    return x, y


def bound_longitude(lon, rad=False):
    """
    Return a value between (-180, 180], handling the wrap-around
    """
    if rad:
        lon = lon * 180 / math.pi

    # Keeps longitude in [0, 360) range
    lon = lon % 360
    lon[lon > 180] = lon[lon > 180] % 180 - 180

    if rad:
        lon = lon * math.pi / 180
    return lon


def bound_latitude(lat, rad=False):
    """
    Return a value between [-90, 90], properly handling the poles
    """
    if rad:
        lat = lat * 180 / math.pi

    # Flip the sign of the output if it crosses the equator an odd num of times
    flip = (abs(lat) // 180) % 2 == 1

    # If latitude is negative to begin with, do a second flip at the end
    is_neg = lat < 0

    # Keep latitude in [0, 180] range
    lat = lat % 180

    # Handle the poles
    lat[lat > 90] = 90 - (lat[lat > 90] % 90)

    # Logical XOR, only flip the sign if only one of the flip cases is true
    lat[flip != is_neg] *= -1

    if rad:
        lat = lat * math.pi / 180
    return lat


def rad2deg(rad):
    return rad * 180.0 / math.pi


def deg2rad(deg):
    return deg * math.pi / 180.0
