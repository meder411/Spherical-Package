import torch

import _spherical_distortion_ext._weighted_mapped_max_pooling as _weighted_mapped_max_pool
import _spherical_distortion_ext._mapped_max_pooling as _mapped_max_pool
import _spherical_distortion_ext._weighted_mapped_avg_pooling as _weighted_mapped_avg_pool
import _spherical_distortion_ext._mapped_avg_pooling as _mapped_avg_pool
from ..layer_utils import InterpolationType, check_args, check_input_map_shape


class MappedAvgPoolFunction(torch.autograd.Function):

    @staticmethod
    def forward(self,
                input,
                sample_map,
                kernel_size,
                interp,
                interp_weights=None):

        if interp_weights is not None:
            pooled_output = _weighted_mapped_avg_pool.weighted_mapped_avg_pool(
                input, sample_map, interp_weights, kernel_size, interp)
        else:
            pooled_output = _mapped_avg_pool.mapped_avg_pool(
                input, sample_map, kernel_size, interp)

        self.save_for_backward(
            torch.tensor([input.shape[2], input.shape[3]]), sample_map,
            torch.tensor(kernel_size), torch.tensor(int(interp)),
            interp_weights)

        return pooled_output

    @staticmethod
    def backward(self, grad_output):

        input_shape, \
            sample_map, \
            kernel_size, \
            interp, \
            interp_weights = self.saved_tensors

        # Cast back to enum
        interp = InterpolationType(interp)

        if interp_weights is not None:
            grad_input = _weighted_mapped_avg_pool.weighted_mapped_avg_unpool(
                grad_output, sample_map, interp_weights, input_shape[0],
                input_shape[1], kernel_size, interp)
        else:
            grad_input = _mapped_avg_pool.mapped_avg_unpool(
                grad_output, sample_map, input_shape[0], input_shape[1],
                kernel_size, interp)

        return grad_input, None, None, None, None


class MappedAvgUnpoolFunction(torch.autograd.Function):

    @staticmethod
    def forward(self,
                input,
                oh,
                ow,
                sample_map,
                kernel_size,
                interp,
                interp_weights=None):

        if interp_weights is not None:
            pooled_output = _weighted_mapped_avg_pool.weighted_mapped_avg_unpool(
                input, sample_map, oh, ow, interp_weights, kernel_size, interp)
        else:
            pooled_output = _mapped_avg_pool.mapped_avg_unpool(
                input, sample_map, oh, ow, kernel_size, interp)

        self.save_for_backward(
            torch.tensor([input.shape[2], input.shape[3]]), sample_map,
            torch.tensor(kernel_size), torch.tensor(int(interp)),
            interp_weights)

        return pooled_output

    @staticmethod
    def backward(self, grad_output):

        input_shape, \
            sample_map, \
            kernel_size, \
            interp, \
            interp_weights = self.saved_tensors

        # Cast back to enum
        interp = InterpolationType(interp)

        if interp_weights is not None:
            grad_input = _weighted_mapped_avg_pool.weighted_mapped_avg_pool(
                grad_output, sample_map, interp_weights, kernel_size, interp)
        else:
            grad_input = _mapped_avg_pool.mapped_avg_pool(
                grad_output, sample_map, kernel_size, interp)

        return grad_input, None, None, None, None, None, None, None


class MappedMaxPoolFunction(torch.autograd.Function):

    @staticmethod
    def forward(self,
                input,
                sample_map,
                kernel_size,
                interp,
                interp_weights=None):

        if interp_weights is not None:
            pooled_output, idx_mask = _weighted_mapped_max_pool.weighted_mapped_max_pool(
                input, sample_map, interp_weights, kernel_size, interp)
        else:
            pooled_output, idx_mask = _mapped_max_pool.mapped_max_pool(
                input, sample_map, kernel_size, interp)

        self.mark_non_differentiable(idx_mask)

        self.save_for_backward(
            torch.tensor([input.shape[2], input.shape[3]]),
            idx_mask, sample_map, torch.tensor(kernel_size),
            torch.tensor(int(interp)), interp_weights)

        return pooled_output, idx_mask

    @staticmethod
    def backward(self, grad_output, idx_mask_grad=None):
        input_shape, \
            idx_mask, \
            sample_map, \
            kernel_size, \
            interp, \
            interp_weights = self.saved_tensors

        # Cast back to enum
        interp = InterpolationType(interp)

        if interp_weights is not None:
            grad_input = _weighted_mapped_max_pool.weighted_mapped_max_unpool(
                grad_output, idx_mask, sample_map, interp_weights,
                input_shape[0], input_shape[1], kernel_size, interp)
        else:
            grad_input = _mapped_max_pool.mapped_max_unpool(
                grad_output, idx_mask, sample_map, input_shape[0],
                input_shape[1], kernel_size, interp)

        return grad_input, None, None, None, None


class MappedMaxUnpoolFunction(torch.autograd.Function):

    @staticmethod
    def forward(self,
                input,
                idx_mask,
                sample_map,
                kernel_size,
                interp,
                interp_weights=None):

        if interp_weights is not None:
            unpooled_input = _weighted_mapped_max_pool.weighted_mapped_max_unpool(
                input, idx_mask, sample_map, interp_weights, input_shape[0],
                input_shape[1], kernel_size, interp)
        else:
            unpooled_input = _mapped_max_pool.mapped_max_unpool(
                intput, idx_mask, sample_map, input_shape[0], input_shape[1],
                kernel_size, interp)

        self.save_for_backward(sample_map, torch.tensor(kernel_size),
                               torch.tensor(int(interp)), interp_weights)

        return unpooled_input

    @staticmethod
    def backward(self, grad_output):

        sample_map, \
            kernel_size, \
            interp, \
            interp_weights = self.saved_tensors

        # Cast back to enum
        interp = InterpolationType(interp)

        if interp_weights is not None:
            pooled_grad, _ = _weighted_mapped_max_pool.weighted_mapped_max_pool(
                grad_output, sample_map, interp_weights, kernel_size, interp)
        else:
            pooled_grad, _ = _mapped_max_pool.mapped_max_pool(
                grad_output, sample_map, kernel_size, interp)

        return grad_input, None, None, None, None


def mapped_avg_pool(input,
                    sample_map,
                    kernel_size,
                    interp,
                    interp_weights=None):
    check_args(input, sample_map, interp_weights, None, kernel_size)
    return MappedAvgPoolFunction.apply(input, sample_map, kernel_size, interp,
                                       interp_weights)


def mapped_avg_unpool(input,
                      oh,
                      ow,
                      sample_map,
                      kernel_size,
                      interp,
                      interp_weights=None):
    check_args(input, sample_map, interp_weights, None, kernel_size)
    check_input_map_shape(input, sample_map)
    return MappedAvgUnpoolFunction.apply(input, oh, ow, sample_map,
                                         kernel_size, interp, interp_weights)


def mapped_max_pool(input,
                    sample_map,
                    kernel_size,
                    interp,
                    interp_weights=None):
    check_args(input, sample_map, interp_weights, None, kernel_size)
    return MappedMaxPoolFunction.apply(input, sample_map, kernel_size, interp,
                                       interp_weights)


def mapped_max_unpool(input,
                      idx_mask,
                      sample_map,
                      kernel_size,
                      interp,
                      interp_weights=None):
    return MappedMaxPoolFunction.apply(input, idx_mask, sample_map,
                                       kernel_size, interp, interp_weights)