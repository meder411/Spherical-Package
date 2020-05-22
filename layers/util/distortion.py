import torch
from matplotlib import patches

from .conversions import *
from ..layer_utils import DistortionType

import _spherical_distortion_ext._mesh as _mesh

# Expose the ImageGrid class
ImageGrid = _mesh.ImageGrid


def brown_distortion_function(x, y, params):
    k1 = params[0]
    k2 = params[1]
    k3 = params[2]
    t1 = params[3]
    t2 = params[4]
    r2 = x * x + y * y
    r4 = r2 * r2
    r6 = r4 * r2
    k_diff = k1 * r2 + k2 * r4 + k3 * r6
    t_x = t2 * (r2 + 2 * x * x) + 2 * t1 * x * y
    t_y = t1 * (r2 + 2 * y * y) + 2 * t2 * x * y
    dx = x * k_diff + t_x
    dy = y * k_diff + t_y
    return dx, dy


def simple_radial_distortion_function(x, y, params):
    k1 = params[0]
    r2 = x * x + y * y
    dr = k1 * r2
    dx = x * dr
    dy = y * dr
    return dx, dy


def fisheye_distortion_function(x, y, params):
    k1 = params[0]
    k2 = params[1]
    r = (x * x + y * y).sqrt()

    if r > 1e-8:  # A machine epsilon value
        theta = r.atan()
        theta2 = theta * theta
        theta4 = theta2 * theta2
        thetad = theta * (T(1) + k1 * theta2 + k2 * theta4)
        dx = x * thetad / r - x
        dy = y * thetad / r - y
    else:
        dx = T(0)
        dy = T(0)
    return dx, dy


def spherical_to_3d_conversion(lon, lat):
    """
    Wraps the convert_spherical_to_3d function from .conversions. Changes the parameter and return layout to be consistent with other conversion functions so it can be used with the numerical Tissot analysis
    """
    return convert_spherical_to_3d(torch.stack((lon, lat), -1), True)


def distortion_to_3d_conversion(x, y, distortion_func, params):
    """
    Models image distortion as a warps image manifold
    """
    dx, dy = distortion_func(x, y, params)
    return x + dx, y + dy, torch.zeros_like(x)


def compute_jacobian_on_surface(u, v, forward_transform, eps=0.01):
    """
    Computes the differentials: 

    [dX/dv, dY/dv, dX/dv, dX/du, dY/du, dX/du] 

    for the given projection function) using central differences. u and v are an orthogonal coordinate system on the surface and X, Y, Z are 3D Cartesian coordinates..

    Returns (u.shape[0], u.shape[1], 2, 2)
    """

    # Compute dX/du, dY/du, dZ, du
    x0, y0, z0 = forward_transform(u - eps, v)
    x1, y1, z1 = forward_transform(u + eps, v)
    dx_du = (x1 - x0) / (2 * eps)
    dy_du = (y1 - y0) / (2 * eps)
    dz_du = (z1 - z0) / (2 * eps)

    # Compute dX/dv, dY/dv, dZ/dv
    x2, y2, z2 = forward_transform(u, v - eps)
    x3, y3, z3 = forward_transform(u, v + eps)
    dx_dv = (x3 - x2) / (2 * eps)
    dy_dv = (y3 - y2) / (2 * eps)
    dz_dv = (z3 - z2) / (2 * eps)

    return torch.stack((torch.stack(
        (dx_du, dy_du, dz_du), -1), torch.stack((dx_dv, dy_dv, dz_dv), -1)),
                       -1)


def compute_jacobian_on_map(x, y, forward_transform, eps=0.01):
    """
    Computes the differentials dx/du, dy/du, dx/dv, dy/dv for the given projection function) using central differences. For spherical projections, (lon, lat = u, v)

    Returns (x.shape[0], x.shape[1], 2, 2)
    """

    # Compute dx/dv, dy/dv
    x0, y0 = forward_transform(x - eps, y)
    x1, y1 = forward_transform(x + eps, y)
    dx_du = (x1 - x0) / (2 * eps)
    dy_du = (y1 - y0) / (2 * eps)

    # Compute dx/du, dy/du
    x2, y2 = forward_transform(x, y - eps)
    x3, y3 = forward_transform(x, y + eps)
    dx_dv = (x3 - x2) / (2 * eps)
    dy_dv = (y3 - y2) / (2 * eps)

    return torch.stack((torch.stack(
        (dx_du, dy_du), -1), torch.stack((dx_dv, dy_dv), -1)), -1)


def compute_differential_distance(x,
                                  y,
                                  forward_transform,
                                  conversion_func,
                                  eps=0.01):
    """
    Computes the semi-major and semi-minor axes, as well as the orientation  and area of the ellipse location at each index of (x,y)
    """
    # Compute the Jacobian of the projection using central differences
    # The forward transform in this case is the surface-to-map projection, e.g. a spherical projection or image distortion function
    J_map = compute_jacobian_on_map(x, y, forward_transform, eps)

    # Compute the first fundamental form coefficients from computing distance on the surface
    E_map = (J_map[..., 0]**2).sum(-1)  # (du)^2
    F_map = (J_map[..., 0] * J_map[..., 1]).sum(-1)  # (dudv)
    G_map = (J_map[..., 1]**2).sum(-1)  # (dv)^2

    # Compute the metric tensor. This is used to compute squared distances ds^2
    I_map = torch.stack((torch.stack(
        (E_map, F_map), -1), torch.stack((F_map, G_map), -1)), -1)

    # Compute the Jacobian on the surface using central differences. This measured the differentials in (X, Y, Z) w.r.t (u,v) orthogonal coordinate system on the surface. The conversion function should be a surface-to-3d conversion.
    J_surf = compute_jacobian_on_surface(x, y, conversion_func, eps)

    # Compute the first fundamental form coefficients from computing distance on the surface
    E_surf = (J_surf[..., 0]**2).sum(-1)  # (du)^2
    F_surf = (J_surf[..., 0] * J_surf[..., 1]).sum(-1)  # (dudv)
    G_surf = (J_surf[..., 1]**2).sum(-1)  # (dv)^2

    # Compute the metric tensor. This is used to compute squared distances dS^2
    I_surf = torch.stack((torch.stack(
        (E_surf, F_surf), -1), torch.stack((F_surf, G_surf), -1)), -1)

    # Set du, dv
    du = abs(x[0, 0] - x[0, 1])
    dv = abs(y[0, 0] - y[1, 0])

    # Compute ds and dS
    ds = (E_map * (du**2) + 2 * F_map * (du * dv) + G_map * (dv**2)).sqrt()
    dS = (E_surf * (du**2) + 2 * F_surf * (du * dv) + G_surf * (dv**2)).sqrt()

    # Ratio of differential distance (units are map-distance-per-spherical-distance traveled)
    diff_dist = ds / dS
    return diff_dist


def compute_spherical_tissot_measurements_numerical(x,
                                                    y,
                                                    forward_transform,
                                                    conversion_func,
                                                    eps=0.01):
    """
    Computes the semi-major and semi-minor axes, as well as the orientation  and area of the ellipse location at each index of (x,y)
    """
    # Compute the Jacobian of the projection using central differences
    # The forward transform in this case is the surface-to-map projection, e.g. a spherical projection or image distortion function
    J = compute_jacobian_on_map(x, y, forward_transform, eps)

    # Compute the Jacobian on the surface using central differences. This measured the differentials in (X, Y, Z) w.r.t (u,v) orthogonal coordinate system on the surface. The conversion function should be a surface-to-3d conversion.
    J_surf = compute_jacobian_on_surface(x, y, conversion_func, eps)

    # Compute the first fundamental form coefficients
    E = (J_surf[..., 0]**2).sum(-1)  # (du)^2
    F = (J_surf[..., 0] * J_surf[..., 1]).sum(-1)  # (dudv)
    G = (J_surf[..., 1]**2).sum(-1)  # (dv)^2

    # Compute the metric tensor. This is used to compute squared distances ds^2
    I = torch.stack((torch.stack((E, F), -1), torch.stack((F, G), -1)), -1)

    # For the Tissot details, we want the actual distance along the u and v. For that we take the values along the diagonal (E and G, no cross-terms)and then take the square root (to get ds from ds^2). We then use that info to form K_inv
    K_inv = torch.inverse(
        torch.diag_embed(
            I.diagonal(dim1=-2, dim2=-1).sqrt(), dim1=-2, dim2=-1))

    # Create the transformation matrix as A = JK^-1
    A = J @ K_inv

    # Run SVD on A
    U, S, _ = torch.svd(A)

    # Extract ellipse parameters
    a = S[..., 0]  # Semi-major axis
    b = S[..., 1]  # Semi-minor axis
    theta = torch.atan2(U[..., 1, 0], U[..., 0, 0])  # Orientation
    area = a * b

    return a, b, theta, area


def compute_spherical_tissot_measurements(lon,
                                          lat,
                                          forward_transform,
                                          eps=0.01):
    """
    Computes the semi-major and semi-minor axes, as well as the orientation  and area of the ellipse location at each index of lon/lat
    """
    # Compute the Jacobian using central differences
    J = compute_jacobian_on_map(lon, lat, forward_transform, eps)

    # Create the curvature matrix K_inv with shape (*trix_shape, 2, 2)
    K_inv = torch.stack(
        (torch.stack((1 / (lat.cos()), torch.zeros_like(lon)), -1),
         torch.stack((torch.zeros_like(lon), torch.ones_like(lon)), -1)), -1)

    # Create the transformation matrix as A = JK^-1
    A = J @ K_inv

    # Run SVD on A
    U, S, _ = torch.svd(A)

    # Extract ellipse parameters
    a = S[..., 0]  # Semi-major axis
    b = S[..., 1]  # Semi-minor axis
    theta = torch.atan2(U[..., 1, 0], U[..., 0, 0])  # Orientation
    area = a * b
    h = (A[..., 1]**2).sum(-1).sqrt()
    k = (A[..., 0]**2).sum(-1).sqrt()

    return a, b, theta, area, h, k


def create_ellipse_objects(x, y, width, height, rotation):
    assert x.shape == y.shape, 'Must have a corresponding x-coord for each y-coord'

    if x.dim() == 2:
        nH = x.shape[0]
        nW = x.shape[1]

        # Create ellipse objects
        el = [
            patches.Ellipse(
                (x[r, c].numpy(), y[r, c].numpy()),
                width[r, c].numpy(),
                height[r, c].numpy(),
                angle=rotation[r, c].numpy(),
                color='r',
                linewidth=None,
                alpha=0.6) for c in torch.arange(nW) for r in torch.arange(nH)
        ]

    elif x.dim() == 1:
        n = x.shape[0]

        # Create ellipse objects
        el = [
            patches.Ellipse(
                (x[i].numpy(), y[i].numpy()),
                width[i].numpy(),
                height[i].numpy(),
                angle=rotation[i].numpy(),
                color='r',
                linewidth=None,
                alpha=0.6) for i in torch.arange(n)
        ]

    return el


def eccentricity(a, b):
    """
    a : Semi-major axis
    b : Semi-minor axis
    """
    # Handle tensor vs not tensor inputs
    a_not_tensor = False
    if not torch.is_tensor(a):
        a_not_tensor = True
        a = torch.tensor([a])
    b_not_tensor = False
    if not torch.is_tensor(b):
        b_not_tensor = True
        b = torch.tensor([b])

    # Compute the eccentricity
    e = torch.sqrt(1 - b**2 / a**2)

    # If both inputs are Python scalars, return a Python scalar
    if a_not_tensor and b_not_tensor:
        return e.item()

    # If either input was a tensor, return a tensor
    return e
