#include "ImagePreprocessor.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>
#include <string>

#if defined(__ANDROID__)
#include <android/log.h>
#define ALOGE(...) __android_log_print(ANDROID_LOG_ERROR, "SigLIP", __VA_ARGS__)
#define ALOGI(...) __android_log_print(ANDROID_LOG_INFO, "SigLIP", __VA_ARGS__)
#elif defined(__APPLE__)
#include <os/log.h>
#define ALOGE(fmt, ...) os_log_error(OS_LOG_DEFAULT, "SigLIP [ERROR]: " fmt, ##__VA_ARGS__)
#define ALOGI(fmt, ...) os_log_info(OS_LOG_DEFAULT, "SigLIP [INFO]: " fmt, ##__VA_ARGS__)
#else
#include <cstdio>
#define ALOGE(fmt, ...) printf("SigLIP [ERROR]: " fmt "\n", ##__VA_ARGS__)
#define ALOGI(fmt, ...) printf("SigLIP [INFO]: " fmt "\n", ##__VA_ARGS__)
#endif

namespace rnsiglip::image {
namespace {

struct FloatRGB {
  float r;
  float g;
  float b;
};

inline FloatRGB sampleBilinear(
    const DecodedImage& image,
    float srcX,
    float srcY) {
  const int maxX = std::max(0, image.width - 1);
  const int maxY = std::max(0, image.height - 1);

  const float x = std::clamp(srcX, 0.0f, static_cast<float>(maxX));
  const float y = std::clamp(srcY, 0.0f, static_cast<float>(maxY));

  const int x0 = static_cast<int>(std::floor(x));
  const int y0 = static_cast<int>(std::floor(y));
  const int x1 = std::min(x0 + 1, maxX);
  const int y1 = std::min(y0 + 1, maxY);

  const float dx = x - static_cast<float>(x0);
  const float dy = y - static_cast<float>(y0);

  const auto pixelAt = [&](int px, int py) {
    const size_t offset =
        static_cast<size_t>((py * image.width + px) * 3);
    return FloatRGB{
        static_cast<float>(image.pixels[offset]),
        static_cast<float>(image.pixels[offset + 1]),
        static_cast<float>(image.pixels[offset + 2]),
    };
  };

  const FloatRGB p00 = pixelAt(x0, y0);
  const FloatRGB p10 = pixelAt(x1, y0);
  const FloatRGB p01 = pixelAt(x0, y1);
  const FloatRGB p11 = pixelAt(x1, y1);

  const auto lerp = [](float a, float b, float t) { return a + (b - a) * t; };

  const float r0 = lerp(p00.r, p10.r, dx);
  const float g0 = lerp(p00.g, p10.g, dx);
  const float b0 = lerp(p00.b, p10.b, dx);

  const float r1 = lerp(p01.r, p11.r, dx);
  const float g1 = lerp(p01.g, p11.g, dx);
  const float b1 = lerp(p01.b, p11.b, dx);

  return {
      lerp(r0, r1, dy),
      lerp(g0, g1, dy),
      lerp(b0, b1, dy),
  };
}

}  // namespace

std::vector<float> preprocessImageToCHW(
    const DecodedImage& image,
    const PreprocessConfig& config) {
  if (image.width <= 0 || image.height <= 0 || image.pixels.empty()) {
    throw std::invalid_argument("Invalid image data for preprocessing.");
  }
  if (config.targetWidth <= 0 || config.targetHeight <= 0) {
    throw std::invalid_argument("Invalid target size for preprocessing.");
  }

  const int outW = config.targetWidth;
  const int outH = config.targetHeight;
  const size_t planeSize = static_cast<size_t>(outW * outH);

  std::vector<float> output(planeSize * 3);
  float* outR = output.data();
  float* outG = output.data() + planeSize;
  float* outB = output.data() + (planeSize * 2);

  const float srcW = static_cast<float>(image.width);
  const float srcH = static_cast<float>(image.height);
  const float srcAspect = srcW / srcH;
  const float dstAspect = static_cast<float>(outW) / static_cast<float>(outH);

  // Center-crop to preserve aspect ratio before resize.
  float cropX = 0.0f;
  float cropY = 0.0f;
  float cropW = srcW;
  float cropH = srcH;

  if (srcAspect > dstAspect) {
    cropW = srcH * dstAspect;
    cropX = (srcW - cropW) * 0.5f;
  } else if (srcAspect < dstAspect) {
    cropH = srcW / dstAspect;
    cropY = (srcH - cropH) * 0.5f;
  }

  for (int y = 0; y < outH; ++y) {
    for (int x = 0; x < outW; ++x) {
      const float u = (static_cast<float>(x) + 0.5f) / static_cast<float>(outW);
      const float v = (static_cast<float>(y) + 0.5f) / static_cast<float>(outH);

      const float sx = cropX + u * cropW;
      const float sy = cropY + v * cropH;
      const FloatRGB rgb = sampleBilinear(image, sx, sy);

      const float r01 = rgb.r / 255.0f;
      const float g01 = rgb.g / 255.0f;
      const float b01 = rgb.b / 255.0f;

      const size_t idx = static_cast<size_t>(y * outW + x);
      outR[idx] = (r01 - config.mean[0]) / config.std[0];
      outG[idx] = (g01 - config.mean[1]) / config.std[1];
      outB[idx] = (b01 - config.mean[2]) / config.std[2];

      if (x == 0 && y == 0) {
        ALOGI("SigLIP Preprocess Pixel(0,0): RGB=%f,%f,%f, Norm=%f,%f,%f", 
              rgb.r, rgb.g, rgb.b, outR[idx], outG[idx], outB[idx]);
      }
    }
  }

  ALOGI("SigLIP Input Tensor first 5: [%.4f, %.4f, %.4f, %.4f, %.4f]", 
        output[0], output[1], output[2], output[3], output[4]);

  return output;
}

}  // namespace rnsiglip::image
