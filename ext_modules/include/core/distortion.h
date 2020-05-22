#ifndef CORE_DISTORTION_H_
#define CORE_DISTORTION_H_

#include <math.h>
#include <limits>

#include <Eigen/Core>
#include <Eigen/Dense>

#include "core/util.h"
#include "enums.h"

namespace spherical {
namespace core {

// C++ typedef for distortion functions
template <typename T>
using dist_func_t = void (*)(const T, const T, const T *, T &, T &);

template <typename T>
__host__ __device__ void BrownDistortionFunction(const T x_in, const T y_in,
                                                 const T *params_ptr, T &dx,
                                                 T &dy) {
  // Brown distortion function, courtesy of OpenMVG
  const T k1 = params_ptr[0];
  const T k2 = params_ptr[1];
  const T k3 = params_ptr[2];
  const T t1 = params_ptr[3];
  const T t2 = params_ptr[4];
  const T r2 = x_in * x_in + y_in * y_in;
  const T r4 = r2 * r2;
  const T r6 = r4 * r2;
  const T k_diff = k1 * r2 + k2 * r4 + k3 * r6;
  const T t_x = t2 * (r2 + 2 * x_in * x_in) + 2 * t1 * x_in * y_in;
  const T t_y = t1 * (r2 + 2 * y_in * y_in) + 2 * t2 * x_in * y_in;
  dx = x_in * k_diff + t_x;
  dy = y_in * k_diff + t_y;
}

template <typename T>
__host__ __device__ void SimpleRadialDistortionFunction(const T x_in,
                                                        const T y_in,
                                                        const T *params_ptr,
                                                        T &dx, T &dy) {
  const T k1 = params_ptr[0];
  const T r2 = x_in * x_in + y_in * y_in;
  const T dr = k1 * r2;
  dx = x_in * dr;
  dy = y_in * dr;
}

template <typename T>
__host__ __device__ void FisheyeDistortionFunction(const T x_in, const T y_in,
                                                   const T *params_ptr, T &dx,
                                                   T &dy) {
  const T k1 = params_ptr[0];
  const T k2 = params_ptr[1];
  const T r = std::sqrt(x_in * x_in + y_in * y_in);

  if (r > T(std::numeric_limits<double>::epsilon())) {
    const T theta = std::atan(r);
    const T theta2 = theta * theta;
    const T theta4 = theta2 * theta2;
    const T thetad = theta * (T(1) + k1 * theta2 + k2 * theta4);
    dx = x_in * thetad / r - x_in;
    dy = y_in * thetad / r - y_in;
  } else {
    dx = T(0);
    dy = T(0);
  }
}

template <typename T>
__host__ __device__ dist_func_t<T> GetDistortionFunction(
    const DistortionType type) {
  switch (type) {
    case DistortionType::SIMPLE_RADIAL:
    default:
      return SimpleRadialDistortionFunction<T>;
    case DistortionType::BROWN:
      return BrownDistortionFunction<T>;
    case DistortionType::FISHEYE:
      return FisheyeDistortionFunction<T>;
  }
}

template <typename T>
__host__ __device__ void AddDistortion(const T x_ud, const T y_ud,
                                       const DistortionType type,
                                       const T *params_ptr, T &x_out,
                                       T &y_out) {
  // Get the function pointer to the relevant distortion function
  dist_func_t<T> DistFunc = GetDistortionFunction<T>(type);

  // Compute distortion offsets
  T dx = T(0);
  T dy = T(0);
  DistFunc(x_ud, y_ud, params_ptr, dx, dy);
  x_out = x_ud + dx;
  y_out = y_ud + dy;
}

template <typename T>
__host__ __device__ void RemoveDistortion(const T x_d, const T y_d,
                                          const DistortionType type,
                                          const T *params_ptr, T &x_ud,
                                          T &y_ud) {
  // Gauss-Newton minimization using central differences
  // 100 iterations should be enough even for complex camera models with
  // higher order terms according to COLMAP. This code is from COLMAP.

  // Get the function pointer to the relevant distortion function
  dist_func_t<T> DistFunc = GetDistortionFunction<T>(type);

  // Parameters
  const size_t num_itt = 100;
  const double max_step_norm = 1e-10;
  const double rel_step_size = 1e-10;

  // Starting point and working variables
  Eigen::Matrix2d J;
  const Eigen::Vector2d x0(x_d, y_d);
  Eigen::Vector2d x(x_d, y_d);
  Eigen::Vector2d dx;
  Eigen::Vector2d dx_0b;
  Eigen::Vector2d dx_0f;
  Eigen::Vector2d dx_1b;
  Eigen::Vector2d dx_1f;

  for (size_t i = 0; i < num_itt; ++i) {
    // Bound the step size by the minimal floating point difference
    const double step0 = host_device_max(std::numeric_limits<double>::epsilon(),
                                         abs(rel_step_size * x(0)));
    const double step1 = host_device_max(std::numeric_limits<double>::epsilon(),
                                         abs(rel_step_size * x(1)));

    // Compute the central differences
    DistFunc(x(0), x(1), params_ptr, dx(0), dx(1));
    DistFunc(x(0) - step0, x(1), params_ptr, dx_0b(0), dx_0b(1));
    DistFunc(x(0) + step0, x(1), params_ptr, dx_0f(0), dx_0f(1));
    DistFunc(x(0), x(1) - step1, params_ptr, dx_1b(0), dx_1b(1));
    DistFunc(x(0), x(1) + step1, params_ptr, dx_1f(0), dx_1f(1));

    // Create Jacobian
    J(0, 0) = 1 + (dx_0f(0) - dx_0b(0)) / (2 * step0);
    J(0, 1) = (dx_1f(0) - dx_1b(0)) / (2 * step1);
    J(1, 0) = (dx_0f(1) - dx_0b(1)) / (2 * step0);
    J(1, 1) = 1 + (dx_1f(1) - dx_1b(1)) / (2 * step1);

    // Compute the step from the Jacobian
    const Eigen::Vector2d step_x = J.inverse() * (x + dx - x0);

    // Take the step
    x -= step_x;

    // Early termination if convergence
    if (step_x.squaredNorm() < max_step_norm) {
      break;
    }
  }

  // Assign the result to the undistorted coordinates
  x_ud = x(0);
  y_ud = x(1);
}

}  // namespace core
}  // namespace spherical

#endif