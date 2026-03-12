#pragma once

#include <string>

#include "ImageTypes.hpp"

namespace rnsiglip::image {

/**
 * Platform-specific image decode implementation.
 * Returns RGB8 interleaved pixels.
 */
DecodedImage decodeImageFromPath(const std::string& absolutePath);

}  // namespace rnsiglip::image
