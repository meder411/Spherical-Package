#ifndef MAPPED_TRANSPOSED_CONVOLUTION_LAYER_H_
#define MAPPED_TRANSPOSED_CONVOLUTION_LAYER_H_

#include <torch/extension.h>
#include <vector>

#include "cuda_helper.h"
#include "enums.h"

namespace spherical {

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
// FORWARD DECLARATIONS
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

#ifndef __NO_CUDA__  // CUDA compilation only
namespace cuda {
torch::Tensor MappedTransposedConvForward(torch::Tensor input,
                                          torch::Tensor sample_map,
                                          torch::Tensor weight,
                                          torch::Tensor bias, int outputHeight,
                                          int outputWidth, int kernel_size,
                                          InterpolationType interpolation);

torch::Tensor MappedTransposedConvBackwardInput(
    torch::Tensor grad_output, torch::Tensor sample_map, torch::Tensor weight,
    int inputHeight, int inputWidth, int kernel_size,
    InterpolationType interpolation);

torch::Tensor MappedTransposedConvBackwardWeight(
    torch::Tensor grad_output, torch::Tensor sample_map, torch::Tensor input,
    int kernel_size, InterpolationType interpolation);
}  // namespace cuda
#endif

namespace cpu {
torch::Tensor MappedTransposedConvForward(torch::Tensor input,
                                          torch::Tensor sample_map,
                                          torch::Tensor weight,
                                          torch::Tensor bias, int outputHeight,
                                          int outputWidth, int kernel_size,
                                          InterpolationType interpolation);

torch::Tensor MappedTransposedConvBackwardInput(
    torch::Tensor grad_output, torch::Tensor sample_map, torch::Tensor weight,
    int inputHeight, int inputWidth, int kernel_size,
    InterpolationType interpolation);

torch::Tensor MappedTransposedConvBackwardWeight(
    torch::Tensor grad_output, torch::Tensor sample_map, torch::Tensor input,
    int kernel_size, InterpolationType interpolation);
}  // namespace cpu

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
// CALL FUNCTION IMPLEMENTATIONS
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

torch::Tensor MappedTransposedConvForward(torch::Tensor input,
                                          torch::Tensor sample_map,
                                          torch::Tensor weight,
                                          torch::Tensor bias, int outputHeight,
                                          int outputWidth, int kernel_size,
                                          InterpolationType interpolation) {
  CHECK_CONTIGUOUS(input);
  CHECK_CONTIGUOUS(sample_map);
  CHECK_CONTIGUOUS(weight);
  CHECK_CONTIGUOUS(bias);

#ifndef __NO_CUDA__  // CUDA compilation only
  if (input.is_cuda()) {
    CHECK_CUDA(weight);
    CHECK_CUDA(bias);
    CHECK_CUDA(sample_map);

    return cuda::MappedTransposedConvForward(input, sample_map, weight, bias,
                                             outputHeight, outputWidth,
                                             kernel_size, interpolation);
  } else
#endif
  {
    CHECK_CPU(sample_map);
    CHECK_CPU(weight);
    CHECK_CPU(bias);

    return cpu::MappedTransposedConvForward(input, sample_map, weight, bias,
                                            outputHeight, outputWidth,
                                            kernel_size, interpolation);
  }
}

std::vector<torch::Tensor> MappedTransposedConvBackward(
    torch::Tensor grad_output, torch::Tensor sample_map, torch::Tensor input,
    torch::Tensor weight, torch::Tensor bias, int kernel_size,
    InterpolationType interpolation) {
  CHECK_CONTIGUOUS(sample_map);
  CHECK_CONTIGUOUS(input);
  CHECK_CONTIGUOUS(weight);
  CHECK_CONTIGUOUS(bias);

#ifndef __NO_CUDA__  // CUDA compilation only
  if (grad_output.is_cuda()) {
    CHECK_CUDA(sample_map);
    CHECK_CUDA(input);
    CHECK_CUDA(weight);
    CHECK_CUDA(bias);

    torch::Tensor grad_input = cuda::MappedTransposedConvBackwardInput(
        grad_output, sample_map, weight, input.size(2), input.size(3),
        kernel_size, interpolation);

    torch::Tensor grad_weight = cuda::MappedTransposedConvBackwardWeight(
        grad_output, sample_map, input, kernel_size, interpolation);

    torch::Tensor grad_bias = grad_output.sum({0, 2, 3});

    return {grad_input, grad_weight, grad_bias};
  } else
#endif
  {
    CHECK_CPU(sample_map);
    CHECK_CPU(input);
    CHECK_CPU(weight);
    CHECK_CPU(bias);

    torch::Tensor grad_input = cpu::MappedTransposedConvBackwardInput(
        grad_output, sample_map, weight, input.size(2), input.size(3),
        kernel_size, interpolation);

    torch::Tensor grad_weight = cpu::MappedTransposedConvBackwardWeight(
        grad_output, sample_map, input, kernel_size, interpolation);

    torch::Tensor grad_bias = grad_output.sum({0, 2, 3});

    return {grad_input, grad_weight, grad_bias};
  }
}

}  // namespace spherical

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("mapped_transposed_conv_forward",
        &spherical::MappedTransposedConvForward, "Forward convolution");
  m.def("mapped_transposed_conv_backward",
        &spherical::MappedTransposedConvBackward, "Backward convolution");
}

#endif