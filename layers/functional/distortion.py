import torch.nn as nn

import _spherical_distortion_ext._distort as _distort
import _spherical_distortion_ext._enums as _enums

from .resample import unresample
from ..layer_utils import InterpolationType

# Wrap the exported C++ _distort.DistortionType enums
DistortionType = _enums.DistortionType


def create_distortion_map(input_shape,
                          params,
                          dist_type,
                          crop=True,
                          keep_shape=True):
    # First compute the distorted location of all locations of the input image grid
    H, W = input_shape
    distortion_map = _distort.create_distortion_sample_map(
        H, W, dist_type, params, crop)

    # If keep_shape is True, then the output should be the same size as the input
    if keep_shape:
        DH, DW = distortion_map.shape[:2]
        if DH > H:
            top_crop = (DH - H) // 2
            bottom_crop = (DH - H) // 2
            if (DH - H) % 2 == 1:
                bottom_crop += 1
            distortion_map = distortion_map[top_crop:-bottom_crop, ...]
        elif H > DH:
            top_pad = (H - DH) // 2
            bottom_pad = (H - DH) // 2
            if (H - DH) % 2 == 1:
                bottom_pad += 1
            distortion_map = nn.functional.pad(
                distortion_map, (0, 0, 0, 0, top_pad, bottom_pad), value=-1)
        if DW > W:
            left_crop = (DW - W) // 2
            right_crop = (DW - W) // 2
            if (DW - W) % 2 == 1:
                right_crop += 1
            distortion_map = distortion_map[:, left_crop:-right_crop, :]
        elif W > DW:
            left_pad = (W - DW) // 2
            right_pad = (W - DW) // 2
            if (W - DW) % 2 == 1:
                right_pad += 1
            distortion_map = nn.functional.pad(
                distortion_map, (0, 0, left_pad, right_pad, 0, 0), value=-1)

    return distortion_map.contiguous()


def distort(x,
            params,
            dist_type,
            crop=True,
            keep_shape=True,
            interpolation=InterpolationType.BILINEAR):
    """
    x:      Either 3D or 4D tensor ([batch] x channels x input_height x 
            input_width)
    """
    assert x.dim() in [3, 4], \
        'x must be a 3D or 4D tensor (x.dim() == {})'.format(x.dim())
    assert x.get_device() == params.get_device(
    ), 'x and params must be on the same device'

    distortion_map = create_distortion_map(x.shape[-2:], params, dist_type,
                                           crop, keep_shape)

    # Resample the input according to the distortion map
    output = unresample(x, distortion_map.to(x.dtype), interpolation)

    return output