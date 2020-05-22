import torch
import math


def parse_center_coord(center_coord):
    assert isinstance(
        center_coord, tuple) or isinstance(center_coord, list) or isinstance(
            center_coord, torch.Tensor
        ), 'center_coord should be a tuple, list, or torch.Tensor type'
    if isinstance(center_coord, tuple) or isinstance(center_coord, list):
        center_lon, center_lat = center_coord
    elif isinstance(center_coord, torch.Tensor):
        center_lon = center_coord[..., 0]
        center_lat = center_coord[..., 1]
    return center_lon, center_lat


def match_dims(a, b, center_coord):
    assert a.dim() == b.dim(), 'Coordinate tensors must have the same shape'
    # If it's a tensor it, return the expanded view needed for matching dimensions
    if isinstance(center_coord, torch.Tensor):
        return [-1] + [1 for i in range(a.dim() - 1)]
    else:
        return None


def sin(x):
    if isinstance(x, torch.Tensor):
        return torch.sin(x)
    else:
        return math.sin(x)


def cos(x):
    if isinstance(x, torch.Tensor):
        return torch.cos(x)
    else:
        return math.cos(x)


def forward_equirectangular_projection(lon, lat, center_coord=(0.0, 0.0)):
    """
    Computes the projection from the sphere to the map
    """
    center_lon, center_lat = center_coord
    x = (lon - center_lon) * math.cos(center_lat)
    y = lat - center_lat
    return x, y


def inverse_equirectangular_projection(x, y, center_coord=(0.0, 0.0)):
    """
    Computes the projection from the map to the sphere
    """
    center_lon, center_lat = center_coord
    lon = x / math.cos(center_lat) + center_lon
    lat = y + center_lat
    return lon, lat


def forward_mercator_projection(lon, lat, center_coord=(0.0, 0.0)):
    """
    Computes the projection from the sphere to the map
    """
    center_lon, _ = center_coord
    x = lon - center_lon
    y = (lat.tan() + 1 / lat.cos()).log()
    return x, y


def inverse_mercator_projection(x, y, center_coord=(0.0, 0.0)):
    """
    Computes the projection from the map to the sphere
    """
    center_lon, _ = center_coord
    lon = x + center_lon
    lat = torch.atan2(y.sinh(), torch.ones_like(y))
    return lon, lat


def forward_gnomonic_projection(lon, lat, center_coord=(0.0, 0.0)):
    """
    Computes the projection from the sphere to the map
    """
    center_lon, center_lat = parse_center_coord(center_coord)
    view = match_dims(lon, lat, center_coord)
    if view is not None:
        center_lon = center_lon.view(view)
        center_lat = center_lat.view(view)
    cos_c = sin(center_lat) * sin(lat) + cos(center_lat) * cos(lat) * cos(
        lon - center_lon)
    x = cos(lat) * sin(lon - center_lon) / cos_c
    y = (cos(center_lat) * sin(lat) -
         sin(center_lat) * cos(lat) * cos(lon - center_lon)) / cos_c
    return x, y


def inverse_gnomonic_projection(x, y, center_coord=(0.0, 0.0)):
    """
    Computes the projection from the map to the sphere
    """
    center_lon, center_lat = center_coord
    rho = (x**2 + y**2).sqrt()
    c = rho.atan()
    lat = (c.cos() * math.sin(center_lat) +
           y * c.sin() * math.cos(center_lat) / rho).asin()
    lon = center_lon + torch.atan2(x * c.sin(),
                                   (rho * math.cos(center_lat) * c.cos() -
                                    y * math.sin(center_lat) * c.sin()))

    # If image has an odd-valued dimension and the center coordinate is 0.0, handle the case which resolves to NaN above
    if x.dim() == 2:
        H, W = x.shape
        if (H % 2 == 1) and abs(center_coord[0]) <= 1e-10:
            lat[..., H // 2, W // 2] = center_coord[0]
        if (W % 2 == 1) and abs(center_coord[1]) <= 1e-10:
            lon[..., H // 2, W // 2] = center_coord[1]

    return lon, lat


def forward_lambert_cylindrical_projection(lon, lat, center_coord=(0.0, 0.0)):
    center_lon, center_lat = center_coord
    x = (lon - center_lon) * math.cos(center_lat)
    y = lat.sin() / math.cos(center_lat)
    return x, y


def inverse_lambert_cylindrical_projection(x, y, center_coord=(0.0, 0.0)):
    center_lon, center_lat = center_coord
    lat = (y * math.cos(center_lat)).asin()
    lon = x / math.cos(center_lat) + center_lon
    return lon, lat


def lambert_cylindrical_width(height, center_lat):
    return math.floor(height * math.pi * math.cos(center_lat)**2)