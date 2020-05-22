import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import gradcheck

import math
import time

from ..functional import resample, unresample, uv_resample
from ..layer_utils import InterpolationType


class Resample(nn.Module):
    '''
    A class that maps integer-valued input locations to real-valued output locations according to a function.
    '''

    def __init__(self, interpolation=InterpolationType.BILINEAR):

        super(Resample, self).__init__()

        self.interp = interpolation

    def forward(self, x, sample_map, output_shape, interp_weights=None):
        '''
        x:              batch x channels x input_height x input_width
        sample_map: input_height x input_width x 2 (x, y)
        output_shape:   (output_height, output_width)
        interp_weights: [OPTIONAL] input_height x input_width x num_interp_points x 2 (x, y)
        '''
        return resample(x, sample_map, output_shape, self.interp,
                        interp_weights)


class Unresample(nn.Module):
    '''
    A class that maps real-valued input locations to a integer-valued output location according to a function. Essentially a mapped convolution with unit weights, no bias, and a kernel size of 1.
    '''

    def __init__(self, interpolation=InterpolationType.BILINEAR):

        super(Unresample, self).__init__()

        self.interp = interpolation

    def forward(self, x, sample_map, interp_weights=None):
        '''
        x:              batch x channels x input_height x input_width
        sample_map:     output_height x output_width x 2 (x, y)
        interp_weights: output_height x output_width x num_interp_points x 2 (x, y)
        '''
        return unresample(x, sample_map, self.interp, interp_weights)


class ResampleFromUV(nn.Module):
    '''
    A class that maps real-valued input locations on a B x C x N x H x W set of N H x W textures to a integer-valued output location.
    '''

    def __init__(self, interpolation=InterpolationType.BILINEAR):

        super(ResampleFromUV, self).__init__()

        assert interpolation in [
            InterpolationType.NEAREST, InterpolationType.BILINEAR
        ], 'Unsupported interpolation type'
        self.interp = interpolation

    def forward(self, x, quad_idx, tex_uv):
        '''
        x:           batch x channels x num_textures x tex_height x tex_width
        quad_idx:    output_height x output_width
        tex_uv:      output_height x output_width x 2 (x, y)
        '''

        assert x.dim() == 5, \
            'Input expected to be 5 dimensional tensor ({} != {})'.format(
            x.dim(), 5)
        assert (quad_idx.dim() == 2) and (quad_idx.dtype == torch.int64), \
            'quad_idx expected to be 2 dimension tensor of type long'
        assert tex_uv.shape[:2] == quad_idx.shape, \
            'tex_uv expected to have same first two dimensions as quad_idx ({} != {})'.format(tex_uv.shape[:2], quad_idx.shape)

        return uv_resample(x, quad_idx, tex_uv, self.interp)