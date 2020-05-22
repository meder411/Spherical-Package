import torch
import torch.nn as nn

import math

from ..functional import mapped_transposed_convolution
from ..layer_utils import InterpolationType


class MappedTransposedConvolution(nn.Module):
    '''
    Contrary to the standard mapped convolution, for the mapped transposed convolution, the mapping should be a mapping from the output to the input. The map should be the same size as the input, but point to locations in the range of the output.
    '''

    def __init__(
            self,
            in_channels,  # Input channels to convolution
            out_channels,  # Output channels from convolution
            kernel_size=1,  # Note the single dimension
            interpolation=InterpolationType.BILINEAR,
            bias=True):
        super(MappedTransposedConvolution, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.interp = interpolation

        # Initialize parameters of the layer
        self.weight = nn.Parameter(
            torch.Tensor(self.in_channels, self.out_channels,
                         self.kernel_size))
        self.bias = nn.Parameter(torch.Tensor(self.out_channels))

        self.reset_parameters()

        if not bias:
            self.bias = nn.Parameter(torch.zeros(self.out_channels))

    def reset_parameters(self):
        n = self.in_channels * self.kernel_size
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, oh, ow, sample_map, interp_weights=None):
        '''
        x:          batch x channels x input_height x input_width
        sample_map: input_height x input_width x kernel_size x 2 (x, y)
        oh:         scalar output height
        ow:         scalar output width
        interp_weights: [OPTIONAL] input_height x input_width x kernel_size x num_interp_points x 2 (x, y)
        '''
        return mapped_transposed_convolution(
            x, self.weight, self.bias, sample_map, oh, ow, self.kernel_size,
            self.interp, interp_weights)
