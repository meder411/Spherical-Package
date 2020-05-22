#ifndef ENUM_EXPORT_H_
#define ENUM_EXPORT_H_

#include <torch/extension.h>

#include "enums.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // Interpolation type enum
  py::enum_<spherical::InterpolationType>(
      m, "InterpolationType", py::arithmetic(),
      "Enum for the type of distortion operation")
      .value("NEAREST", spherical::InterpolationType::NEAREST,
             "Nearest neighbor interpolation")
      .value("BILINEAR", spherical::InterpolationType::BILINEAR,
             "Bilinear interpolation")
      .value("BISPHERICAL", spherical::InterpolationType::BISPHERICAL,
             "Bilinear interpolation that wraps around the X-axis");

  // Distortion type enum
  py::enum_<spherical::DistortionType>(
      m, "DistortionType", py::arithmetic(),
      "Enum for the type of distortion operation")
      .value("SIMPLE_RADIAL", spherical::DistortionType::SIMPLE_RADIAL,
             "Simple radial distortion (k1 only)")
      .value("BROWN", spherical::DistortionType::BROWN, "Brown distortion")
      .value("FISHEYE", spherical::DistortionType::FISHEYE,
             "Fisheye distortion");
}

#endif