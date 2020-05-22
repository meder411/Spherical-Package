#ifndef RESAMPLE_H_
#define RESAMPLE_H_

#include <omp.h>
#include <torch/extension.h>

#include "common/distortion.h"
#include "core/distortion.h"

#include "enums.h"

namespace spherical {
namespace cpu {

template <typename T,
          void (*ApplyFunction)(const int64_t, T *, const int64_t, const T,
                                const T, const T, const T, const T, const T,
                                const DistortionType, const T *)>
void ComputeSampleMap(torch::Tensor sample_map, const T f_in, const T cx_in,
                      const T cy_in, const T f_out, const T cx_out,
                      const T cy_out, const DistortionType type,
                      torch::Tensor params) {
  const int64_t num_kernels = sample_map.size(0) * sample_map.size(1);
  T *sample_map_ptr         = sample_map.data_ptr<T>();
  const T *params_ptr       = params.data_ptr<T>();

  int64_t index;
#pragma omp parallel for shared(sample_map_ptr, params_ptr) private(index) \
    schedule(static)
  for (index = 0; index < num_kernels; index++) {
    ApplyFunction(index, sample_map_ptr, sample_map.size(1), f_in, cx_in,
                  cy_in, f_out, cx_out, cy_out, type, params_ptr);
  }
}

template <typename T>
void SampleMapSelector(torch::Tensor sample_map, const T f_in, const T cx_in,
                       const T cy_in, const T f_out, const T cx_out,
                       const T cy_out, const bool undistort,
                       const DistortionType type, torch::Tensor params) {
  if (undistort) {
    ComputeSampleMap<T, common::Undistort>(
        sample_map, f_in, cx_in, cy_in, f_out, cx_out, cy_out, type, params);
  } else {
    ComputeSampleMap<T, common::Distort>(sample_map, f_in, cx_in, cy_in, f_out,
                                         cx_out, cy_out, type, params);
  }
}

}  // namespace cpu
}  // namespace spherical
#endif