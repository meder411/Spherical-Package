import torch
import torch.nn as nn
import torch.nn.functional as F

import math

from ..functional import mapped_convolution
from ..layer_utils import _pair, InterpolationType


class MappedConvolution(nn.Module):
    '''
    A class that performs the mapped convolution operation. This operations requires a map tensor that maps from the input to some output. The output dimension is determined by the dimension of the map.
    '''

    def __init__(
            self,
            in_channels,  # Input channels to convolution
            out_channels,  # Output channels from convolution
            kernel_size=1,  # One dimension
            interpolation=InterpolationType.BILINEAR,
            bias=True):

        super(MappedConvolution, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.interp = interpolation

        # Initialize parameters of the layer
        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels, self.kernel_size))
        self.bias = nn.Parameter(torch.Tensor(out_channels))

        # Initializes the parameters of the layer
        self.reset_parameters()

        if not bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))

    def reset_parameters(self):
        '''
        Sets up initial weights for the parameters
        '''
        n = self.in_channels * self.kernel_size
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, sample_map, interp_weights=None):
        '''
        x:              batch x channels x input_height x input_width
        sample_map:     output_height x output_width x kernel_size x 2 (x, y)
        interp_weights: [OPTIONAL] output_height x output_width x num_interp_points x 2
        '''
        return mapped_convolution(x, self.weight, self.bias, sample_map,
                                  self.kernel_size, self.interp,
                                  interp_weights)
