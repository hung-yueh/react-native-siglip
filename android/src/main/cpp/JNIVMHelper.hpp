#pragma once
#include <fbjni/fbjni.h>
#include <string>

namespace rnsiglip {

  // React Native fbjni wrapper for android.graphics.Bitmap
  struct Bitmap : facebook::jni::JavaClass<Bitmap> {
    static constexpr auto kJavaDescriptor = "Landroid/graphics/Bitmap;";
  };

  // React Native fbjni wrapper for android.graphics.BitmapFactory
  struct BitmapFactory : facebook::jni::JavaClass<BitmapFactory> {
    static constexpr auto kJavaDescriptor = "Landroid/graphics/BitmapFactory;";

    // Binds to public static Bitmap decodeFile(String pathName)
    static facebook::jni::local_ref<Bitmap> decodeFile(facebook::jni::alias_ref<facebook::jni::JString> path) {
      static auto method = javaClassStatic()->getStaticMethod<Bitmap(facebook::jni::alias_ref<facebook::jni::JString>)>("decodeFile");
      return method(javaClassStatic(), path);
    }
  };

} // namespace rnsiglip
