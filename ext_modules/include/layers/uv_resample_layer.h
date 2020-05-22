#ifndef UV_RESAMPLE_LAYER_H_
#define UV_RESAMPLE_LAYER_H_

#include <torch/extension.h>
#include <vector>

#include "cuda_helper.h"
#include "enums.h"

namespace spherical {

#ifndef __NO_CUDA__  // CUDA compilation only
namespace cuda {
torch::Tensor EquirectangularToTangentImages(torch::Tensor input,
                                             torch::Tensor quad_idx,
                                             torch::Tensor tex_uv,
                                             int numTextures, int texHeight,
                                             int texWidth,
                                             InterpolationType interpolation);

torch::Tensor EquirectangularFromTangentImages(torch::Tensor grad_output,
                                               torch::Tensor quad_idx,
                                               torch::Tensor tex_uv,
                                               InterpolationType interpolation);
}  // namespace cuda
#endif

namespace cpu {
torch::Tensor EquirectangularToTangentImages(torch::Tensor input,
                                             torch::Tensor quad_idx,
                                             torch::Tensor tex_uv,
                                             int numTextures, int texHeight,
                                             int texWidth,
                                             InterpolationType interpolation);

torch::Tensor EquirectangularFromTangentImages(torch::Tensor grad_output,
                                               torch::Tensor quad_idx,
                                               torch::Tensor tex_uv,
                                               InterpolationType interpolation);
}  // namespace cpu

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
// CALL FUNCTION IMPLEMENTATIONS
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

torch::Tensor ResampleToUVMaps(torch::Tensor input, torch::Tensor quad_idx,
                               torch::Tensor tex_uv, int numTextures,
                               int texHeight, int texWidth,
                               InterpolationType interpolation) {
  CHECK_CONTIGUOUS(input);
  CHECK_CONTIGUOUS(quad_idx);
  CHECK_CONTIGUOUS(tex_uv);

#ifndef __NO_CUDA__  // CUDA compilation only
  if (input.is_cuda()) {
    CHECK_CUDA(input);
    CHECK_CUDA(quad_idx);
    CHECK_CUDA(tex_uv);
    return cuda::EquirectangularToTangentImages(input, quad_idx, tex_uv,
                                                numTextures, texHeight,
                                                texWidth, interpolation);
  } else
#endif
  {
    CHECK_CPU(input);
    CHECK_CPU(quad_idx);
    CHECK_CPU(tex_uv);
    return cpu::EquirectangularToTangentImages(input, quad_idx, tex_uv,
                                               numTextures, texHeight, texWidth,
                                               interpolation);
  }
}

torch::Tensor ResampleFromUVMaps(torch::Tensor grad_output,
                                 torch::Tensor quad_idx, torch::Tensor tex_uv,
                                 InterpolationType interpolation) {
  CHECK_CONTIGUOUS(grad_output);
  CHECK_CONTIGUOUS(quad_idx);
  CHECK_CONTIGUOUS(tex_uv);

#ifndef __NO_CUDA__  // CUDA compilation only
  if (grad_output.is_cuda()) {
    CHECK_CUDA(grad_output);
    CHECK_CUDA(quad_idx);
    CHECK_CUDA(tex_uv);
    return cuda::EquirectangularFromTangentImages(grad_output, quad_idx, tex_uv,
                                                  interpolation);
  } else
#endif
  {
    CHECK_CPU(grad_output);
    CHECK_CPU(quad_idx);
    CHECK_CPU(tex_uv);
    return cpu::EquirectangularFromTangentImages(grad_output, quad_idx, tex_uv,
                                                 interpolation);
  }
}

}  // namespace spherical

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("resample_to_uv_maps", &spherical::ResampleToUVMaps,
        "Resample to UV maps operation");
  m.def("resample_from_uv_maps", &spherical::ResampleFromUVMaps,
        "Resample from UV maps operation");
}

#endif