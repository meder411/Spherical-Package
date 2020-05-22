#ifndef DISTORTION_H_
#define DISTORTION_H_

#include <torch/extension.h>

#include "core/distortion.h"
#include "core/interpolation.h"
#include "core/util.h"
#include "enums.h"

namespace spherical {
namespace common {

template <typename T>
__host__ __device__ void Distort(const int64_t index, T *sample_map_ptr,
                                 const int64_t map_width, const T f_in,
                                 const T cx_in, const T cy_in, const T f_out,
                                 const T cx_out, const T cy_out,
                                 const DistortionType type,
                                 const T *params_ptr) {
  // Compute location in sample map
  int64_t y_in = 0;
  int64_t x_in = 0;
  IndToSubHW(index, map_width, y_in, x_in);

  // Normalize coordinates
  T x_n = static_cast<T>(x_in);
  T y_n = static_cast<T>(y_in);
  NormalizePinholeCamera(x_n, y_n, f_in, cx_in, cy_in);

  // Compute the distorted location
  T x_out = x_n;
  T y_out = y_n;
  core::AddDistortion(x_n, y_n, type, params_ptr, x_out, y_out);

  // Denormalize coordinates
  DenormalizePinholeCamera(x_out, y_out, f_out, cx_out, cy_out);

  // Fill the coordinates tensor
  const int64_t loc       = y_in * map_width * 2 + x_in * 2;
  sample_map_ptr[loc]     = x_out;  // First channel is X
  sample_map_ptr[loc + 1] = y_out;  // Second channel is Y
}

template <typename T>
__host__ __device__ void Undistort(const int64_t index, T *sample_map_ptr,
                                   const int64_t map_width, const T f_in,
                                   const T cx_in, const T cy_in, const T f_out,
                                   const T cx_out, const T cy_out,
                                   const DistortionType type,
                                   const T *params_ptr) {
  // Compute location in sample map
  int64_t y_in = 0;
  int64_t x_in = 0;
  IndToSubHW(index, map_width, y_in, x_in);

  // Normalize coordinates
  T x_n = static_cast<T>(x_in);
  T y_n = static_cast<T>(y_in);
  NormalizePinholeCamera(x_n, y_n, f_in, cx_in, cy_in);

  // Compute the undistorted location
  T x_out = x_n;
  T y_out = y_n;
  core::RemoveDistortion(x_n, y_n, type, params_ptr, x_out, y_out);

  // Denormalize coordinates
  DenormalizePinholeCamera(x_out, y_out, f_out, cx_out, cy_out);

  // Fill the coordinates tensor
  const int64_t loc       = y_in * map_width * 2 + x_in * 2;
  sample_map_ptr[loc]     = x_out;  // First channel is X
  sample_map_ptr[loc + 1] = y_out;  // Second channel is Y
}

}  // namespace common
}  // namespace spherical
#endif