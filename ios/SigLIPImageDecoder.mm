#import <CoreGraphics/CoreGraphics.h>
#import <Foundation/Foundation.h>
#import <UIKit/UIKit.h>

#include "cpp/image/ImageDecoder.hpp"

#include <stdexcept>

namespace rnsiglip::image {

DecodedImage decodeImageFromPath(const std::string& absolutePath) {
  @autoreleasepool {
    NSString* nsPath = [NSString stringWithUTF8String:absolutePath.c_str()];
    UIImage* image = [UIImage imageWithContentsOfFile:nsPath];
    if (image == nil || image.CGImage == nil) {
      throw std::runtime_error("Unable to decode image from path.");
    }

    CGImageRef cgImage = image.CGImage;
    const size_t width = CGImageGetWidth(cgImage);
    const size_t height = CGImageGetHeight(cgImage);
    if (width == 0 || height == 0) {
      throw std::runtime_error("Decoded image has invalid size.");
    }

    // Draw into RGBA then strip alpha into RGB interleaved.
    const size_t bytesPerPixel = 4;
    const size_t bytesPerRow = width * bytesPerPixel;
    const size_t byteCount = bytesPerRow * height;

    std::vector<uint8_t> rgba(byteCount);
    CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
    CGContextRef context = CGBitmapContextCreate(
        rgba.data(),
        width,
        height,
        8,
        bytesPerRow,
        colorSpace,
        (uint32_t)kCGImageAlphaNoneSkipLast | (uint32_t)kCGBitmapByteOrder32Big);
    CGColorSpaceRelease(colorSpace);

    if (context == nil) {
      throw std::runtime_error("Failed to create bitmap context.");
    }

    CGContextDrawImage(
        context,
        CGRectMake(0, 0, static_cast<CGFloat>(width), static_cast<CGFloat>(height)),
        cgImage);
    CGContextRelease(context);

    std::vector<uint8_t> rgb(width * height * 3);
    for (size_t i = 0; i < width * height; ++i) {
      rgb[i * 3 + 0] = rgba[i * 4 + 0];
      rgb[i * 3 + 1] = rgba[i * 4 + 1];
      rgb[i * 3 + 2] = rgba[i * 4 + 2];
    }

    DecodedImage decoded;
    decoded.width = static_cast<int>(width);
    decoded.height = static_cast<int>(height);
    decoded.pixels = std::move(rgb);
    return decoded;
  }
}

}  // namespace rnsiglip::image
