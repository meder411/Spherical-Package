import torch

import _spherical_distortion_ext._resample as _resample
import _spherical_distortion_ext._weighted_resample as _weighted_resample
import _spherical_distortion_ext._uv_resample as _uv_resample
import _spherical_distortion_ext._voting_resample as _voting_resample

from ..layer_utils import InterpolationType, check_args, check_input_map_shape


class ResampleFunction(torch.autograd.Function):

    @staticmethod
    def forward(self,
                input,
                sample_map,
                output_shape,
                interp,
                interp_weights=None):
        self.save_for_backward(input, sample_map, torch.tensor(int(interp)),
                               interp_weights)

        if interp_weights is not None:
            return _weighted_resample.weighted_resample_to_map(
                input, sample_map, interp_weights, output_shape[0],
                output_shape[1], interp)
        else:
            return _resample.resample_to_map(
                input, sample_map, output_shape[0], output_shape[1], interp)

    @staticmethod
    def backward(self, grad_output):
        input, \
            sample_map, \
            interp, \
            interp_weights = self.saved_tensors

        # Cast back to enum
        interp = InterpolationType(interp)

        if interp_weights is not None:
            unresampled_grad_output = _weighted_resample.weighted_resample_from_map(
                grad_output, sample_map, interp_weights, interp)
        else:
            unresampled_grad_output = _resample.resample_from_map(
                grad_output, sample_map, interp)

        return unresampled_grad_output, None, None, None, None


class UnresampleFunction(torch.autograd.Function):

    @staticmethod
    def forward(self, input, sample_map, interp, interp_weights=None):
        self.save_for_backward(input, sample_map, torch.tensor(int(interp)),
                               interp_weights)

        if interp_weights is not None:
            return _weighted_resample.weighted_resample_from_map(
                input, sample_map, interp_weights, interp)
        else:
            return _resample.resample_from_map(input, sample_map, interp)

    @staticmethod
    def backward(self, grad_output):
        input, \
            sample_map, \
            interp, \
            interp_weights = self.saved_tensors

        # Cast back to enum
        interp = InterpolationType(interp)

        if interp_weights is not None:
            resampled_grad_output = _weighted_resample.weighted_resample_to_map(
                grad_output, sample_map, interp_weights, input.shape[2],
                input.shape[3], interp)
        else:
            resampled_grad_output = _resample.resample_to_map(
                grad_output, sample_map, input.shape[2], input.shape[3],
                interp)

        return resampled_grad_output, None, None, None, None


class UVResampleFunction(torch.autograd.Function):

    @staticmethod
    def forward(self, input, quad_idx, tex_uv, interp):
        self.save_for_backward(quad_idx, tex_uv, torch.tensor(int(interp)),
                               torch.tensor(input.shape[-3:]))

        return _uv_resample.resample_from_uv_maps(input, quad_idx, tex_uv,
                                                  interp)

    @staticmethod
    def backward(self, grad_output):
        quad_idx, \
        tex_uv, \
        interp, \
        input_shape = self.saved_tensors

        # Cast back to enum
        interp = InterpolationType(interp)

        resampled_grad_output = _uv_resample.resample_to_uv_maps(
            grad_output, quad_idx, tex_uv, input_shape[0], input_shape[1],
            input_shape[2], interp)

        return resampled_grad_output, None, None, None


def resample(input, sample_map, output_shape, interp, interp_weights=None):
    # Add the batch dimension, if necessary, for the resample call

    assert input.dim() in [3, 4], \
        'Input must have 3 or 4 dimensions (input.dim() == {})'.format(input.dim())

    added_batch_dim = False
    if input.dim() == 3:
        input.unsqueeze_(0)
        added_batch_dim = True

    check_args(input, sample_map, interp_weights, None, None)
    check_input_map_shape(input, sample_map)
    resampled = ResampleFunction.apply(input, sample_map, output_shape, interp,
                                       interp_weights)

    # Remove the batch dimension if it wasn't originally there
    if added_batch_dim:
        input.squeeze_(0)
        resampled.squeeze_(0)

    return resampled


def unresample(input, sample_map, interp, interp_weights=None):
    # Add the batch dimension, if necessary, for the resample call

    assert input.dim() in [3, 4], \
        'Input must have 3 or 4 dimensions (input.dim() == {})'.format(input.dim())

    added_batch_dim = False
    if input.dim() == 3:
        input.unsqueeze_(0)
        added_batch_dim = True

    check_args(input, sample_map, interp_weights, None, None)
    unresampled = UnresampleFunction.apply(input, sample_map, interp,
                                           interp_weights)

    # Remove the batch dimension if it wasn't originally there
    if added_batch_dim:
        input.squeeze_(0)
        unresampled.squeeze_(0)

    return unresampled


def voting_resample(input, sample_map, interp_weights, output_shape,
                    num_candidates):
    # Add the batch dimension, if necessary, for the resample call

    assert input.dim() in [3, 4], \
        'Input must have 3 or 4 dimensions (input.dim() == {})'.format(input.dim())

    added_batch_dim = False
    if input.dim() == 3:
        input.unsqueeze_(0)
        added_batch_dim = True

    check_args(input, sample_map, interp_weights, None, None)
    check_input_map_shape(input, sample_map)

    # Adjust the sample map to assign to the nearest neighbor according to the interp_weights
    # First set the max weight to 1 and zero-out the others
    nearest_interp_weights = (interp_weights == interp_weights.max(
        -1, keepdim=True)[0]).long()
    # Then rebuild the sample map

    resampled = _voting_resample.voting_resample_to_map(
        input.long(), sample_map.long(), *output_shape, num_candidates)

    # Remove the batch dimension if it wasn't originally there
    if added_batch_dim:
        input.squeeze_(0)
        resampled.squeeze_(0)

    return resampled


def uv_resample(input, quad_idx, tex_uv, interp):
    # Add the batch dimension, if necessary, for the resample call

    assert input.dim() in [4, 5], \
        'Input must have 4 or 5 dimensions (input.dim() == {})'.format(input.dim())

    added_batch_dim = False
    if input.dim() == 4:
        input.unsqueeze_(0)
        added_batch_dim = True
    uv_resampled = UVResampleFunction.apply(input, quad_idx, tex_uv, interp)

    # Remove the batch dimension if it wasn't originally there
    if added_batch_dim:
        input.squeeze_(0)
        uv_resampled.squeeze_(0)

    return uv_resampled