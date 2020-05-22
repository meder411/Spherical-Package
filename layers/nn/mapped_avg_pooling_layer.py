import torch.nn as nn

from ..functional import mapped_avg_pool, mapped_avg_unpool
from ..layer_utils import InterpolationType, check_args


class MappedAvgPool(nn.Module):

    def __init__(self, kernel_size, interpolation=InterpolationType.BILINEAR):

        super(MappedAvgPool, self).__init__()

        self.kernel_size = kernel_size
        self.interp = interpolation

    def forward(self, x, sample_map, interp_weights=None):

        return mapped_avg_pool(x, sample_map, self.kernel_size, self.interp,
                               interp_weights)


class MappedAvgUnpool(nn.Module):

    def __init__(self, kernel_size, interpolation=InterpolationType.BILINEAR):

        super(MappedAvgUnpool, self).__init__()

        self.kernel_size = kernel_size
        self.interp = interpolation

    def forward(self, x, oh, ow, sample_map, interp_weights=None):
        '''
        x:          batch x channels x input_height x input_width
        oh:         scalar output height
        ow:         scalar output width
        sample_map: input_height x input_width x kernel_size x 2 (x, y)
        interp_weights: [OPTIONAL] input_height x input_width x kernel_size x num_interp_points x 2 (x, y)
        '''

        return mapped_avg_unpool(x, oh, ow, sample_map, self.kernel_size,
                                 self.interp, interp_weights)
