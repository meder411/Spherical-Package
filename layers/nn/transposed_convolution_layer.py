import torch
import torch.nn as nn

import math

from ..functional import transposed_convolution
from ..layer_utils import _pair


class TransposedConvolution(nn.Module):

    def __init__(
            self,
            in_channels,  # Input channels to convolution
            out_channels,  # Output channels from convolution
            kernel_size=1,  # Filter size
            stride=1,  # Stride
            padding=0,  # Padding
            dilation=1):  # Dilation

        super(TransposedConvolution, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.padding = _pair(padding)
        self.stride = _pair(stride)
        self.dilation = _pair(dilation)

        # Initialize parameters of the layer
        self.weight = nn.Parameter(
            torch.Tensor(in_channels, out_channels, self.kernel_size[0],
                         self.kernel_size[1]))
        self.bias = nn.Parameter(torch.Tensor(out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
            stdv = 1. / math.sqrt(n)
            self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        '''
        x: B x C x H x W
        '''
        return transposed_convolution(x, self.weight, self.bias,
                                      self.kernel_size, self.stride,
                                      self.padding, self.dilation)
