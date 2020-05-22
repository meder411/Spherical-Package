import torch
import _spherical_distortion_ext._convolution as _conv
import _spherical_distortion_ext._transposed_convolution as _transposed_conv

# =============================================================================
# AUTOGRAD FUNCTIONS
# =============================================================================


class ConvolutionFunction(torch.autograd.Function):

    @staticmethod
    def forward(self, input, weight, bias, kernel_size, stride, padding,
                dilation):
        self.save_for_backward(input, weight, bias, torch.tensor(kernel_size),
                               torch.tensor(stride), torch.tensor(padding),
                               torch.tensor(dilation))
        return _conv.conv_forward(
            input, weight, bias, kernel_size[0], kernel_size[1], stride[0],
            stride[1], padding[0], padding[1], dilation[0], dilation[1])

    @staticmethod
    def backward(self, grad_output):
        input, weight, bias, kernel_size, stride, padding, dilation = self.saved_tensors
        grad_input, grad_weight, grad_bias = _conv.conv_backward(
            grad_output, input, weight, bias, kernel_size[0], kernel_size[1],
            stride[0], stride[1], padding[0], padding[1], dilation[0],
            dilation[1])

        return grad_input, grad_weight, grad_bias, None, None, None, None


class TransposedConvolutionFunction(torch.autograd.Function):

    @staticmethod
    def forward(self, input, weight, bias, kernel_size, stride, padding,
                dilation):
        self.save_for_backward(input, weight, bias, torch.tensor(kernel_size),
                               torch.tensor(stride), torch.tensor(padding),
                               torch.tensor(dilation))
        return _transposed_conv.transposed_conv_forward(
            input, weight, bias, kernel_size[0], kernel_size[1], stride[0],
            stride[1], padding[0], padding[1], dilation[0], dilation[1])

    @staticmethod
    def backward(self, grad_output):
        input, weight, bias, kernel_size, stride, padding, dilation = self.saved_tensors
        grad_input, grad_weight, grad_bias = _transposed_conv.transposed_conv_backward(
            grad_output, input, weight, bias, kernel_size[0], kernel_size[1],
            stride[0], stride[1], padding[0], padding[1], dilation[0],
            dilation[1])

        return grad_input, grad_weight, grad_bias, None, None, None, None


# =============================================================================
# EXPOSED FUNCTIONS
# =============================================================================


def convolution(input, weight, bias, kernel_size, stride, padding, dilation):
    return ConvolutionFunction.apply(input, weight, bias, kernel_size, stride,
                                     padding, dilation)


def transposed_convolution(input, weight, bias, kernel_size, stride, padding,
                           dilation):
    return TransposedConvolutionFunction.apply(
        input, weight, bias, kernel_size, stride, padding, dilation)