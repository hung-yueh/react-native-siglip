#include <stdexcept>

#include "JNIVMHelper.hpp"
#include "NitroSigLIPOnLoad.hpp"
#include <fbjni/fbjni.h>

JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM* vm, void* reserved) {
  return facebook::jni::initialize(vm, []() {
    // 1. Register Nitro's auto-generated wrappers
    margelo::nitro::rnsiglip::registerAllNatives();
  });
}

