#pragma once

#include <cstdint>
#include <vector>

namespace rnsiglip::image {

struct DecodedImage {
  int width = 0;
  int height = 0;
  // Interleaved RGB, 8-bit per channel.
  std::vector<uint8_t> pixels;
};

}  // namespace rnsiglip::image
