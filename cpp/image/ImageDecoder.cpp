#include "ImageDecoder.hpp"

#if !defined(__APPLE__) && !defined(__ANDROID__)

#include <stdexcept>

namespace rnsiglip::image {

DecodedImage decodeImageFromPath(const std::string& absolutePath) {
  (void)absolutePath;
  throw std::runtime_error(
      "No platform decoder compiled. Use iOS or Android decoder implementation.");
}

}  // namespace rnsiglip::image

#endif
