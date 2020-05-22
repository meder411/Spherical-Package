#ifndef CORE_UTIL_H_
#define CORE_UTIL_H_

#include <math.h>
#include <omp.h>
#include <cstdio>

// For GCC
#ifndef __host__
#define __host__
#endif
#ifndef __device__
#define __device__
#endif

// From
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomic-functions
// With help from https://stackoverflow.com/a/39287554/3427580
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 600
static __inline__ __device__ double atomicAdd(double *address, double val) {
  unsigned long long int *address_as_ull = (unsigned long long int *)address;
  unsigned long long int old = *address_as_ull, assumed;
  if (val == 0.0) {
    return __longlong_as_double(old);
  }
  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
                    __double_as_longlong(val + __longlong_as_double(assumed)));
  } while (assumed != old);
  return __longlong_as_double(old);
}
#endif

template <typename T>
__host__ __device__ inline void atomic_add(T *address, T val) {
#ifdef __CUDACC__  // CUDA versions of atomic add
  atomicAdd(address, val);
#else  // C++ version of atomic add
#pragma omp atomic
  *address += val;
#endif
}

template <typename T>
__host__ __device__ inline T host_device_max(const T val1, const T val2) {
#ifdef __CUDACC__  // CUDA versions of atomic add
  return max(val1, val2);
#else  // C++ version of atomic add
  return std::max(val1, val2);
#endif
}

template <typename T>
__host__ __device__ inline const T fnegmod(const T lval, const T rval) {
  return fmod(fmod(lval, rval) + rval, rval);
}

__host__ __device__ inline int64_t negmod(const int64_t lval,
                                          const int64_t rval) {
  return ((lval % rval) + rval) % rval;
}

__host__ __device__ inline void IndToSubHW(const int64_t ind,
                                           const int64_t width, int64_t &y,
                                           int64_t &x) {
  x = ind % width;
  y = ind / width;
}
__host__ __device__ inline int64_t SubToIndHW(const int64_t y, const int64_t x,
                                              const int64_t width) {
  return y * width + x;
}

__host__ __device__ inline void IndToSubCHW(const int64_t ind,
                                            const int64_t height,
                                            const int64_t width, int64_t &c,
                                            int64_t &y, int64_t &x) {
  x = ind % width;
  y = (ind / width) % height;
  c = ind / (width * height);
}

__host__ __device__ inline int64_t SubToIndCHW(const int64_t c, const int64_t y,
                                               const int64_t x,
                                               const int64_t height,
                                               const int64_t width) {
  return c * height * width + y * width + x;
}

__host__ __device__ inline void IndToSubNCHW(
    const int64_t ind, const int64_t channels, const int64_t height,
    const int64_t width, int64_t &n, int64_t &c, int64_t &y, int64_t &x) {
  x = ind % width;
  y = (ind / width) % height;
  c = (ind / (width * height)) % channels;
  n = ind / (width * height * channels);
}

__host__ __device__ inline int64_t SubToIndNCHW(
    const int64_t n, const int64_t c, const int64_t y, const int64_t x,
    const int64_t channels, const int64_t height, const int64_t width) {
  const int64_t hw = height * width;
  return n * channels * hw + c * hw + y * width + x;
}

template <typename T>
__host__ __device__ inline void NormalizePinholeCamera(T &x, T &y, const T f,
                                                       const T cx, const T cy) {
  x = (x - cx) / f;
  y = (y - cy) / f;
}

template <typename T>
__host__ __device__ inline void DenormalizePinholeCamera(T &x, T &y, const T f,
                                                         const T cx,
                                                         const T cy) {
  x = f * x + cx;
  y = f * y + cy;
}

template <typename T>
__host__ __device__ inline void CreatePinholeCamera(T &f, T &cx, T &cy,
                                                    const int64_t height,
                                                    const int64_t width) {
  // Create a simple pinhole camera
  cx = static_cast<T>(width - 1) / T(2);
  cy = static_cast<T>(height - 1) / T(2);
  f = host_device_max(cx, cy);
}

template <typename T>
__host__ __device__ inline void NormalizeCoordinates(T &x, T &y,
                                                     const int64_t height,
                                                     const int64_t width) {
  // Create a simple pinhole camera
  T f;
  T cx;
  T cy;
  CreatePinholeCamera(f, cx, cy, height, width);
  NormalizePinholeCamera(x, y, f, cx, cy);
}

template <typename T>
__host__ __device__ inline void DenormalizeCoordinates(T &x, T &y,
                                                       const int64_t height,
                                                       const int64_t width) {
  // Create a simple pinhole camera
  T f;
  T cx;
  T cy;
  CreatePinholeCamera(f, cx, cy, height, width);
  DenormalizePinholeCamera(x, y, f, cx, cy);
}

#endif