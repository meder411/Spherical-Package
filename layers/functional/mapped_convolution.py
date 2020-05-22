import torch

import _spherical_distortion_ext._mapped_convolution as _mapped_conv
import _spherical_distortion_ext._weighted_mapped_convolution as _weighted_mapped_conv
import _spherical_distortion_ext._mapped_transposed_convolution as _mapped_transposed_conv
import _spherical_distortion_ext._weighted_mapped_transposed_convolution as _weighted_mapped_transposed_conv
from ..layer_utils import InterpolationType, check_args, check_input_map_shape

# =============================================================================
# AUTOGRAD FUNCTIONS
# =============================================================================


class MappedConvolutionFunction(torch.autograd.Function):

    @staticmethod
    def forward(self,
                input,
                weight,
                bias,
                sample_map,
                kernel_size,
                interp,
                interp_weights=None):

        self.save_for_backward(input, weight, bias, sample_map,
                               torch.tensor(kernel_size),
                               torch.tensor(int(interp)), interp_weights)

        if interp_weights is not None:
            return _weighted_mapped_conv.weighted_mapped_conv_forward(
                input, sample_map, interp_weights, weight, bias, kernel_size,
                interp)

        else:
            return _mapped_conv.mapped_conv_forward(input, sample_map, weight,
                                                    bias, kernel_size, interp)

    @staticmethod
    def backward(self, grad_output):
        input, \
            weight, \
            bias, \
            sample_map, \
            kernel_size, \
            interp, \
            interp_weights = self.saved_tensors

        # Cast back to enum
        interp = InterpolationType(interp)

        if interp_weights is not None:
            grad_input, grad_weight, grad_bias = _weighted_mapped_conv.weighted_mapped_conv_backward(
                grad_output, sample_map, interp_weights, input, weight, bias,
                kernel_size, interp)
        else:
            grad_input, grad_weight, grad_bias = _mapped_conv.mapped_conv_backward(
                grad_output, sample_map, input, weight, bias, kernel_size,
                interp)

        return grad_input, grad_weight, grad_bias, None, None, None, None


class MappedTransposedConvolutionFunction(torch.autograd.Function):

    @staticmethod
    def forward(self,
                input,
                weight,
                bias,
                sample_map,
                output_height,
                output_width,
                kernel_size,
                interp,
                interp_weights=None):

        self.save_for_backward(input, sample_map, weight, bias,
                               torch.tensor(kernel_size),
                               torch.tensor(int(interp)), interp_weights)

        if interp_weights is not None:
            return _weighted_mapped_transposed_conv.weighted_mapped_transposed_conv_forward(
                input, sample_map, interp_weights, weight, bias, output_height,
                output_width, kernel_size, interp)
        else:
            return _mapped_transposed_conv.mapped_transposed_conv_forward(
                input, sample_map, weight, bias, output_height, output_width,
                kernel_size, interp)

    @staticmethod
    def backward(self, grad_output):
        input, \
            sample_map, \
            weight, \
            bias, \
            kernel_size, \
            interp, \
            interp_weights = self.saved_tensors

        # Cast back to enum
        interp = InterpolationType(interp)

        if interp_weights is not None:
            grad_input, grad_weight, grad_bias = _weighted_mapped_transposed_conv.weighted_mapped_transposed_conv_backward(
                grad_output, sample_map, interp_weights, input, weight, bias,
                kernel_size, interp)
        else:
            grad_input, grad_weight, grad_bias = _mapped_transposed_conv.mapped_transposed_conv_backward(
                grad_output, sample_map, input, weight, bias, kernel_size,
                interp)

        return grad_input, grad_weight, grad_bias, None, None, None, None, None, None, None, None


# =============================================================================
# EXPOSED FUNCTIONS
# =============================================================================


def mapped_convolution(input,
                       weight,
                       bias,
                       sample_map,
                       kernel_size,
                       interp,
                       interp_weights=None):
    check_args(input, sample_map, interp_weights, weight.shape[1],
               weight.shape[2])
    return MappedConvolutionFunction.apply(input, weight, bias, sample_map,
                                           kernel_size, interp, interp_weights)


def mapped_transposed_convolution(input,
                                  weight,
                                  bias,
                                  sample_map,
                                  output_height,
                                  output_width,
                                  kernel_size,
                                  interp,
                                  interp_weights=None):
    check_args(input, sample_map, interp_weights, weight.shape[0],
               weight.shape[2])
    check_input_map_shape(input, sample_map)
    return MappedTransposedConvolutionFunction.apply(
        input, weight, bias, sample_map, output_height, output_width,
        kernel_size, interp, interp_weights)
