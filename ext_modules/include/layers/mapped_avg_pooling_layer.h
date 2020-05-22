#ifndef MAPPED_AVG_POOLING_LAYER_H_
#define MAPPED_AVG_POOLING_LAYER_H_

#include <torch/extension.h>
#include <vector>

#include "cuda_helper.h"
#include "enums.h"

namespace spherical {

#ifndef __NO_CUDA__  // CUDA compilation only
namespace cuda {
torch::Tensor MappedAvgPoolForward(torch::Tensor input,
                                   torch::Tensor sample_map, int kernel_size,
                                   InterpolationType interpolation);

torch::Tensor MappedAvgPoolBackward(torch::Tensor input,
                                    torch::Tensor sample_map, int inputHeight,
                                    int inputWidth, int kernel_size,
                                    InterpolationType interpolation);
}  // namespace cuda
#endif

namespace cpu {
torch::Tensor MappedAvgPoolForward(torch::Tensor input,
                                   torch::Tensor sample_map, int kernel_size,
                                   InterpolationType interpolation);

torch::Tensor MappedAvgPoolBackward(torch::Tensor input,
                                    torch::Tensor sample_map, int inputHeight,
                                    int inputWidth, int kernel_size,
                                    InterpolationType interpolation);
}  // namespace cpu

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
// CALL FUNCTION IMPLEMENTATIONS
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

torch::Tensor MappedAvgPoolforward(torch::Tensor input,
                                   torch::Tensor sample_map, int kernel_size,
                                   InterpolationType interpolation) {
  CHECK_CONTIGUOUS(input);
  CHECK_CONTIGUOUS(sample_map);

#ifndef __NO_CUDA__  // CUDA compilation only
  if (input.is_cuda()) {
    CHECK_CUDA(input);
    CHECK_CUDA(sample_map);
    return cuda::MappedAvgPoolForward(input, sample_map, kernel_size,
                                      interpolation);
  } else
#endif
  {
    CHECK_CPU(input);
    CHECK_CPU(sample_map);
    return cpu::MappedAvgPoolForward(input, sample_map, kernel_size,
                                     interpolation);
  }
}

torch::Tensor MappedAvgPoolbackward(torch::Tensor input,
                                    torch::Tensor sample_map, int inputHeight,
                                    int inputWidth, int kernel_size,
                                    InterpolationType interpolation) {
  CHECK_CONTIGUOUS(input);
  CHECK_CONTIGUOUS(sample_map);

#ifndef __NO_CUDA__  // CUDA compilation only
  if (input.is_cuda()) {
    CHECK_CUDA(input);
    CHECK_CUDA(sample_map);
    return cuda::MappedAvgPoolBackward(input, sample_map, inputHeight,
                                       inputWidth, kernel_size, interpolation);
  } else
#endif
  {
    CHECK_CPU(input);
    CHECK_CPU(sample_map);
    return cpu::MappedAvgPoolBackward(input, sample_map, inputHeight,
                                      inputWidth, kernel_size, interpolation);
  }
}

}  // namespace spherical

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("mapped_avg_pool", &spherical::MappedAvgPoolforward,
        "Mapped max pooling operation");
  m.def("mapped_avg_unpool", &spherical::MappedAvgPoolbackward,
        "Mapped max unpooling operation");
}

#endif