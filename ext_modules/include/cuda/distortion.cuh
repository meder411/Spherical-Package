#ifndef RESAMPLE_CUH_
#define RESAMPLE_CUH_

#include <torch/extension.h>

#include "common/distortion.h"
#include "core/distortion.h"
#include "cuda_helper.h"
#include "enums.h"

namespace spherical {
namespace cuda {

template <typename T,
          void (*ApplyFunction)(const int64_t, T *, const int64_t, const T,
                                const T, const T, const T, const T, const T,
                                const DistortionType, const T *)>
__global__ void ComputeSampleMapKernel(
    const int64_t n, T *sample_map_ptr, const int64_t sample_map_width,
    const T f_in, const T cx_in, const T cy_in, const T f_out, const T cx_out,
    const T cy_out, const DistortionType type, const T *params_ptr) {
  int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= n) { return; }
  // if (index != 0) { return; }

  ApplyFunction(index, sample_map_ptr, sample_map_width, f_in, cx_in, cy_in,
                f_out, cx_out, cy_out, type, params_ptr);
}

void SampleMapLauncher(torch::Tensor sample_map, const double f_in,
                       const double cx_in, const double cy_in,
                       const double f_out, const double cx_out,
                       const double cy_out, const bool undistort,
                       const DistortionType type, torch::Tensor params) {
  const int64_t num_kernels = sample_map.size(0) * sample_map.size(1);
  const dim3 blocks((num_kernels + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS);

  if (undistort) {
    ComputeSampleMapKernel<double, common::Undistort>
        <<<blocks, CUDA_NUM_THREADS>>>(
            num_kernels, sample_map.data_ptr<double>(), sample_map.size(1),
            f_in, cx_in, cy_in, f_out, cx_out, cy_out, type,
            params.data_ptr<double>());
  } else {
    ComputeSampleMapKernel<double, common::Distort>
        <<<blocks, CUDA_NUM_THREADS>>>(
            num_kernels, sample_map.data_ptr<double>(), sample_map.size(1),
            f_in, cx_in, cy_in, f_out, cx_out, cy_out, type,
            params.data_ptr<double>());
  }
}

}  // namespace cuda
}  // namespace spherical
#endif