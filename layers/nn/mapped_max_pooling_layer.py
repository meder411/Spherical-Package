import torch
import torch.nn as nn

from ..functional import mapped_max_pool, mapped_max_unpool
from ..layer_utils import InterpolationType, check_args


class MappedMaxPool(nn.Module):

    def __init__(self,
                 kernel_size,
                 interpolation=InterpolationType.BILINEAR,
                 return_indices=False):

        super(MappedMaxPool, self).__init__()

        self.kernel_size = kernel_size
        self.return_indices = return_indices
        self.interp = interpolation

    def forward(self, x, sample_map, interp_weights=None):

        pooled, idx_mask = mapped_max_pool(x, sample_map, self.kernel_size,
                                           self.interp, interp_weights)

        if self.return_indices:
            return pooled, idx_mask
        else:
            return pooled


# --------------------------


class MappedMaxUnpool(nn.Module):

    def __init__(self, kernel_size, interpolation=InterpolationType.BILINEAR):

        super(MappedMaxUnpool, self).__init__()

        self.kernel_size = kernel_size
        self.interp = interpolation

    def forward(self, x, idx_mask, sample_map, interp_weights=None):

        return mapped_max_unpool(x, idx_mask, sample_map, self.kernel_size,
                                 self.interp, interp_weights)
