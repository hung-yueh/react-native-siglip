#include "image/ImageDecoder.hpp"
#include <android/bitmap.h>
#include <android/log.h>
#include <stdexcept>
#include <string>
#include <vector>
#include "JNIVMHelper.hpp" // Keep because it defines the fbjni wrappers!
#include <fbjni/fbjni.h>

#define TAG "SigLIPDecoder"
#define ALOGD(...) __android_log_print(ANDROID_LOG_DEBUG, TAG, __VA_ARGS__)

namespace rnsiglip::image {

DecodedImage decodeImageFromPath(const std::string& absolutePath) {
  ALOGD("Decoding image: %s", absolutePath.c_str());

  // 1. Create a Java String using fbjni
  auto jPath = facebook::jni::make_jstring(absolutePath);

  // 2. Load the Bitmap using our fbjni wrapper
  ALOGD("Calling BitmapFactory.decodeFile for: %s", absolutePath.c_str());
  auto bitmap = BitmapFactory::decodeFile(jPath);

  if (!bitmap) {
    ALOGD("BitmapFactory.decodeFile returned null for: %s", absolutePath.c_str());
    throw std::runtime_error("Failed to decode image at path (returned null): " + absolutePath);
  }
  ALOGD("Bitmap decoded successfully.");

  // For AndroidBitmap NDK APIs, we still need the raw JNIEnv* and jobject.
  // fbjni makes it easy to get them safely:
  facebook::jni::Environment::ensureCurrentThreadIsAttached();
  JNIEnv* env = facebook::jni::Environment::current();
  jobject rawBitmap = bitmap.get();

  // 3. Get Bitmap info and lock pixels
  AndroidBitmapInfo info;
  if (AndroidBitmap_getInfo(env, rawBitmap, &info) < 0) {
    ALOGD("Failed to get bitmap info");
    throw std::runtime_error("Failed to get bitmap info for: " + absolutePath);
  }
  ALOGD("Bitmap info: %ux%u, format: %d", info.width, info.height, info.format);

  void* pixels;
  ALOGD("Locking pixels...");
  if (AndroidBitmap_lockPixels(env, rawBitmap, &pixels) < 0) {
    ALOGD("Failed to lock pixels");
    throw std::runtime_error("Failed to lock bitmap pixels for: " + absolutePath);
  }
  ALOGD("Pixels locked at: %p", pixels);

  // 4. Extract RGB data
  DecodedImage result;
  result.width = static_cast<int>(info.width);
  result.height = static_cast<int>(info.height);
  result.pixels.resize(result.width * result.height * 3);

  uint8_t* dst = result.pixels.data();
  const uint8_t* src = static_cast<const uint8_t*>(pixels);

  if (info.format == ANDROID_BITMAP_FORMAT_RGBA_8888) {
    for (uint32_t i = 0; i < info.width * info.height; ++i) {
      dst[i * 3 + 0] = src[i * 4 + 0]; // R
      dst[i * 3 + 1] = src[i * 4 + 1]; // G
      dst[i * 3 + 2] = src[i * 4 + 2]; // B
    }
  } else if (info.format == ANDROID_BITMAP_FORMAT_RGB_565) {
    const uint16_t* src16 = static_cast<const uint16_t*>(pixels);
    for (uint32_t i = 0; i < info.width * info.height; ++i) {
      uint16_t p = src16[i];
      dst[i * 3 + 0] = ((p >> 11) & 0x1F) << 3; // R
      dst[i * 3 + 1] = ((p >> 5) & 0x3F) << 2;  // G
      dst[i * 3 + 2] = (p & 0x1F) << 3;         // B
    }
  } else {
    AndroidBitmap_unlockPixels(env, rawBitmap);
    throw std::runtime_error("Unsupported bitmap format: " + std::to_string(info.format));
  }

  AndroidBitmap_unlockPixels(env, rawBitmap);
  // No need to call env->DeleteLocalRef(bitmap) — fbjni's local_ref handles it automatically when `bitmap` goes out of scope!

  return result;
}

} // namespace rnsiglip::image
