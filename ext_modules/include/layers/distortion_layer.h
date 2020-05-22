#ifndef DISTORTION_LAYER_H_
#define DISTORTION_LAYER_H_

#include <torch/extension.h>
#include <vector>

#include "cuda_helper.h"
#include "enums.h"

namespace spherical {

#ifndef __NO_CUDA__  // CUDA compilation only
namespace cuda {
torch::Tensor CreateDistortionSampleMap(const int64_t input_height,
                                        const int64_t input_width,
                                        const DistortionType type,
                                        torch::Tensor params, const bool crop);

}  // namespace cuda
#endif

namespace cpu {
torch::Tensor CreateDistortionSampleMap(const int64_t input_height,
                                        const int64_t input_width,
                                        const DistortionType type,
                                        torch::Tensor params, const bool crop);
}  // namespace cpu

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
// CALL FUNCTION IMPLEMENTATIONS
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

torch::Tensor CreateDistortionSampleMap(const int64_t input_height,
                                        const int64_t input_width,
                                        const DistortionType type,
                                        torch::Tensor params, const bool crop) {
  CHECK_CONTIGUOUS(params);

#ifndef __NO_CUDA__  // CUDA compilation only
  if (params.is_cuda()) {
    CHECK_CUDA(params);
    return cuda::CreateDistortionSampleMap(input_height, input_width, type,
                                           params, crop);
  } else
#endif
  {
    CHECK_CPU(params);
    return cpu::CreateDistortionSampleMap(input_height, input_width, type,
                                          params, crop);
  }
}

}  // namespace spherical

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("create_distortion_sample_map", &spherical::CreateDistortionSampleMap,
        "Returns a sample map (H x W x 2) defining where to resample from for "
        "a given distortion model");
}

#endif
