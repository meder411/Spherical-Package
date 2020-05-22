#include <torch/extension.h>

#include "core/util.h"
#include "cuda/distortion.cuh"

#include <iostream>
#include "enums.h"

namespace spherical {
namespace cuda {

torch::Tensor CreateDistortionSampleMap(const int64_t height,
                                        const int64_t width,
                                        DistortionType type,
                                        torch::Tensor params,
                                        const bool crop) {
  // Create a tensor to store distortion sampling map (H x W x 2)
  torch::Tensor sample_map =
      torch::zeros({height, width, 2}, params.options());

  // Create a pinhole camera for the input image
  double f;
  double cx;
  double cy;
  CreatePinholeCamera(f, cx, cy, height, width);

  // Compute the locations to *sample to*  using the *distortion* function
  SampleMapLauncher(sample_map, f, cx, cy, f, cx, cy, false, type, params);

  // Find the bounds of the new image
  torch::Tensor top    = sample_map.narrow(0, 0, 1);   // sample_map[[0],...]
  torch::Tensor bottom = sample_map.narrow(0, -1, 1);  // sample_map[[-1],...]
  torch::Tensor left   = sample_map.narrow(1, 0, 1);   // sample_map[:,[0],:]
  torch::Tensor right  = sample_map.narrow(1, -1, 1);  // sample_map[:,[-1],:]
  int64_t min_x        = 0;
  int64_t max_x        = 0;
  int64_t min_y        = 0;
  int64_t max_y        = 0;
  if (crop) {
    // To crop out all blank areas, find the min bounds of the distorted image
    // Top max y value
    min_y = static_cast<int64_t>(top.narrow(-1, 1, 1).max().item<double>());
    // Bottom min y value
    max_y = static_cast<int64_t>(bottom.narrow(-1, 1, 1).min().item<double>());
    // Left max x value
    min_x = static_cast<int64_t>(left.narrow(-1, 0, 1).max().item<double>());
    // Right min x value
    max_x = static_cast<int64_t>(right.narrow(-1, 0, 1).min().item<double>());

  } else {
    // Otherwise, we want the max bounds
    // Top min y value
    min_y = static_cast<int64_t>(top.narrow(-1, 1, 1).min().item<double>());
    // Bottom max y value
    max_y = static_cast<int64_t>(bottom.narrow(-1, 1, 1).max().item<double>());
    // Left min x value
    min_x = static_cast<int64_t>(left.narrow(-1, 0, 1).min().item<double>());
    // Right max x value
    max_x = static_cast<int64_t>(right.narrow(-1, 0, 1).max().item<double>());
  }

  // Bound the output dimensions
  const int64_t out_width  = std::min(std::max(1L, max_x - min_x + 1), 10000L);
  const int64_t out_height = std::min(std::max(1L, max_y - min_y + 1), 10000L);

  // Use the inverse operation to compute a resampling from the original image
  // Re-initialize the sampling map to (OH x OW x 2)
  sample_map = -1 * torch ::ones({out_height, out_width, 2}, params.options());

  // Create a pinhole camera for the output image
  double cx_p;
  double cy_p;
  double tmp;  // To store garbage focal length
  CreatePinholeCamera(tmp, cx_p, cy_p, out_height, out_width);

  // Compute the sampling locations
  SampleMapLauncher(sample_map, f, cx_p, cy_p, f, cx, cy, true, type, params);

  // Return the sample map
  return sample_map;
}

}  // namespace cuda
}  // namespace spherical