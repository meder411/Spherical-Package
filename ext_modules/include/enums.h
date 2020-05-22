#ifndef ENUMS_H_
#define ENUMS_H_

namespace spherical {

enum class DistortionType {
  SIMPLE_RADIAL,
  BROWN,
  FISHEYE,
};

enum class InterpolationType {
  NEAREST,
  BILINEAR,
  BISPHERICAL,
};

}  // namespace spherical

#endif