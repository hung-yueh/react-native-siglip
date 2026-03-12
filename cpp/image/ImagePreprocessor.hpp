#pragma once

#include <vector>

#include "ImageTypes.hpp"

namespace rnsiglip::image {

struct PreprocessConfig {
  int targetWidth = 224;
  int targetHeight = 224;
  float mean[3] = {0.5f, 0.5f, 0.5f};
  float std[3] = {0.5f, 0.5f, 0.5f};
};

/**
 * Converts a decoded RGB image into CHW Float32 tensor data:
 * shape [1, 3, targetHeight, targetWidth].
 */
std::vector<float> preprocessImageToCHW(
    const DecodedImage& image,
    const PreprocessConfig& config);

}  // namespace rnsiglip::image
