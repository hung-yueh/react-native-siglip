#if RNSIGLIP_HAS_EXECUTORCH
#include <executorch/extension/module/module.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#endif

#ifdef __APPLE__
#include <mach/mach.h>
#endif
#include <sys/stat.h>

#include "SigLIPHybridObject.hpp"

#include <filesystem>
#include <utility>
#include <vector>
#include <string>

#include "image/ImageDecoder.hpp"
#include "image/ImagePreprocessor.hpp"

#if RNSIGLIP_HAS_EXECUTORCH
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
#endif

using namespace rnsiglip;
using namespace margelo::nitro;

namespace margelo::nitro::rnsiglip {

struct SigLIPHybridObject::Impl {
#if RNSIGLIP_HAS_EXECUTORCH
  std::unique_ptr<executorch::extension::Module> visionModule;
  std::unique_ptr<executorch::extension::Module> textModule;
#endif
};

SigLIPHybridObject::SigLIPHybridObject() 
    : HybridObject(TAG),
      HybridSigLIPSpec(),
      pimpl_(std::make_unique<Impl>()) {
#if RNSIGLIP_HAS_EXECUTORCH
#endif
}

SigLIPHybridObject::~SigLIPHybridObject() = default;

size_t SigLIPHybridObject::getExternalMemorySize() noexcept {
  size_t totalMemoryBytes = sizeof(SigLIPHybridObject);

  if (pimpl_) {
    // Add size of implementation struct
    totalMemoryBytes += sizeof(Impl);

#if RNSIGLIP_HAS_EXECUTORCH
    if (pimpl_->visionModule) {
      totalMemoryBytes += 150 * 1024 * 1024; // ~150MB generic Vision Model Size
    }
    
    if (pimpl_->textModule) {
      totalMemoryBytes += 100 * 1024 * 1024; // ~100MB generic Text Model Size
    }
#endif
  }
  return totalMemoryBytes;
}

std::string SigLIPHybridObject::normalizePath(const std::string& path) const {
  constexpr const char* kFileScheme = "file://";
  if (path.empty()) {
    throw SigLIPException(
        SigLIPErrorCode::InvalidArgument,
        "Path must be a non-empty absolute path or file:// URI.");
  }
  if (path.rfind(kFileScheme, 0) == 0) {
    return path.substr(7);
  }
  return path;
}

void SigLIPHybridObject::validateReadableFile(
    const std::string& absolutePath,
    SigLIPErrorCode notFoundCode) const {
  std::error_code ec;
  if (!std::filesystem::exists(absolutePath, ec) ||
      !std::filesystem::is_regular_file(absolutePath, ec)) {
    throw SigLIPException(notFoundCode, "File not found: " + absolutePath);
  }
}

std::shared_ptr<Promise<void>> SigLIPHybridObject::loadVisionModel(const std::string& path) {
  auto promise = Promise<void>::create();
  try {
    const std::string normalizedPath = normalizePath(path);
    validateReadableFile(normalizedPath, SigLIPErrorCode::ModelNotFound);
    std::lock_guard<std::mutex> lock(mutex_);

#if RNSIGLIP_HAS_EXECUTORCH
    ALOGI("Creating ExecuTorch Vision Module for: %s", normalizedPath.c_str());
    pimpl_->visionModule.reset();
    pimpl_->visionModule = std::make_unique<executorch::extension::Module>(
        normalizedPath,
        executorch::extension::Module::LoadMode::Mmap);

    auto load_error = pimpl_->visionModule->load();
    if (load_error != executorch::runtime::Error::Ok) {
      throw SigLIPException(
          SigLIPErrorCode::ModelLoad,
          "Failed to load ExecuTorch vision module. Error code: " + std::to_string((int)load_error));
    }
    ALOGI("Successfully loaded ExecuTorch vision module.");
    
    SigLIPModelInfo info;
    info.width = 224;
    info.height = 224;
    info.dimension = 768;
    
    // Default to CPU, but try to detect if a delegate is present based on the path
    info.backend = SigLIPBackend::CPU;
    if (normalizedPath.find("coreml") != std::string::npos || 
        normalizedPath.find("ios18") != std::string::npos) {
      info.backend = SigLIPBackend::COREML;
    } else if (normalizedPath.find("qnn") != std::string::npos) {
      info.backend = SigLIPBackend::QNN;
    } else if (normalizedPath.find("xnnpack") != std::string::npos) {
      info.backend = SigLIPBackend::XNNPACK;
    }

    modelInfo_ = info;
    promise->resolve();
#else
    throw SigLIPException(
        SigLIPErrorCode::ModelLoad,
        "react-native-siglip was built without ExecuTorch headers.");
#endif
  } catch (const std::exception& e) {
    promise->reject(std::current_exception());
  }
  return promise;
}

std::shared_ptr<Promise<void>> SigLIPHybridObject::loadTextModel(const std::string& path) {
  auto promise = Promise<void>::create();
  try {
    const std::string normalizedPath = normalizePath(path);
    validateReadableFile(normalizedPath, SigLIPErrorCode::ModelNotFound);
    std::lock_guard<std::mutex> lock(mutex_);

#if RNSIGLIP_HAS_EXECUTORCH
    ALOGI("Creating ExecuTorch Text Module for: %s", normalizedPath.c_str());
    pimpl_->textModule.reset();
    pimpl_->textModule = std::make_unique<executorch::extension::Module>(
        normalizedPath,
        executorch::extension::Module::LoadMode::Mmap);

    auto load_error = pimpl_->textModule->load();
    if (load_error != executorch::runtime::Error::Ok) {
      throw SigLIPException(
          SigLIPErrorCode::ModelLoad,
          "Failed to load ExecuTorch text module. Error code: " + std::to_string((int)load_error));
    }
    ALOGI("Successfully loaded ExecuTorch text module.");
    promise->resolve();
#else
    throw SigLIPException(
        SigLIPErrorCode::ModelLoad,
        "react-native-siglip was built without ExecuTorch headers.");
#endif
  } catch (const std::exception& e) {
    promise->reject(std::current_exception());
  }
  return promise;
}

std::shared_ptr<Promise<std::shared_ptr<ArrayBuffer>>> SigLIPHybridObject::getImageEmbedding(
    const std::string& imageUri) {
  auto promise = Promise<std::shared_ptr<ArrayBuffer>>::create();
  
  try {
    const std::string normalizedPath = normalizePath(imageUri);
    validateReadableFile(normalizedPath, SigLIPErrorCode::ImageNotFound);

    std::lock_guard<std::mutex> lock(mutex_);
    if (!modelInfo_) {
      throw SigLIPException(
          SigLIPErrorCode::NotLoaded,
          "No vision model loaded. Call loadVisionModel() before getImageEmbedding().");
    }

    image::DecodedImage decoded;
    try {
      decoded = image::decodeImageFromPath(normalizedPath);
    } catch (const std::exception& e) {
      throw SigLIPException(
          SigLIPErrorCode::ImageDecode,
          std::string("Image decode failed: ") + e.what());
    }

    std::vector<float> inputData;
    try {
      image::PreprocessConfig preprocessConfig;
      preprocessConfig.targetWidth = modelInfo_->width;
      preprocessConfig.targetHeight = modelInfo_->height;
      preprocessConfig.mean[0] = 0.5f;
      preprocessConfig.mean[1] = 0.5f;
      preprocessConfig.mean[2] = 0.5f;
      preprocessConfig.std[0] = 0.5f;
      preprocessConfig.std[1] = 0.5f;
      preprocessConfig.std[2] = 0.5f;
      inputData = image::preprocessImageToCHW(decoded, preprocessConfig);
    } catch (const std::exception& e) {
      throw SigLIPException(
          SigLIPErrorCode::ImageDecode,
          std::string("Image preprocessing failed: ") + e.what());
    }

#if RNSIGLIP_HAS_EXECUTORCH
    if (!pimpl_->visionModule) {
      throw SigLIPException(
          SigLIPErrorCode::NotLoaded,
          "Vision Module pointer is null. Call loadVisionModel() again.");
    }

    auto method_name = "forward";
    auto method_meta_result = pimpl_->visionModule->method_meta(method_name);
    if (!method_meta_result.ok()) {
      ALOGE("Failed to get method_meta for '%s'. Error: %i", method_name, (int)method_meta_result.error());
    }

  const int batchSize = 1;
  const int channels = 3;
  const int height = modelInfo_->height;
  const int width = modelInfo_->width;
  
  ALOGI("Preparing input tensor: [%d, %d, %d, %d]", batchSize, channels, height, width);

  std::vector<executorch::aten::SizesType> sizes = {
      batchSize, channels, height, width};
  std::vector<executorch::aten::DimOrderType> dim_order = {0, 1, 2, 3};
  std::vector<executorch::aten::StridesType> strides = {
      channels * height * width,
      height * width,
      width,
      1};
  
  executorch::aten::TensorImpl tensorImpl(
      executorch::aten::ScalarType::Float,
      sizes.size(),
      sizes.data(),
      inputData.data(),
      dim_order.data(),
      strides.data(),
      executorch::aten::TensorShapeDynamism::STATIC);
  
  executorch::aten::Tensor inputTensor(&tensorImpl);
  
  std::vector<executorch::runtime::EValue> inputs;
  inputs.emplace_back(inputTensor);
  
  float sum = 0;
  float minVal = inputData.empty() ? 0 : inputData[0];
  float maxVal = inputData.empty() ? 0 : inputData[0];
  for (float f : inputData) {
    sum += f;
    if (f < minVal) minVal = f;
    if (f > maxVal) maxVal = f;
  }
  float mean = inputData.empty() ? 0 : sum / inputData.size();

  ALOGI("Vision Input Tensor: Mean=%f, Min=%f, Max=%f, Size=%zu, Dim=[%d,%d,%d,%d]", 
        mean, minVal, maxVal, inputData.size(), batchSize, channels, height, width);
  ALOGI("Vision Input First 5: [%.4f, %.4f, %.4f, %.4f, %.4f]",
        inputData.size() > 0 ? inputData[0] : 0,
        inputData.size() > 1 ? inputData[1] : 0,
        inputData.size() > 2 ? inputData[2] : 0,
        inputData.size() > 3 ? inputData[3] : 0,
        inputData.size() > 4 ? inputData[4] : 0);
  
  auto result = pimpl_->visionModule->forward(inputs);
  if (!result.ok()) {
    auto error = result.error();
    throw SigLIPException(
        SigLIPErrorCode::Inference,
        "Module forward() failed during vision inference. Error Code: " + std::to_string((int)error));
  }
  
  auto& outputs = result.get();
  ALOGI("Vision Inference completed. Output tensors: %zu", outputs.size());
    const size_t targetDim = static_cast<size_t>(modelInfo_->dimension);
    
    std::vector<float> outputDataVector;
    bool found = false;
    size_t bestOutputIndex = 0;

    for (size_t i = 0; i < outputs.size(); ++i) {
      if (outputs[i].isTensor()) {
        const auto& tensor = outputs[i].toTensor();
        const size_t numel = static_cast<size_t>(tensor.numel());
        auto scalarType = tensor.scalar_type();
        
        std::string shapeStr = "[";
        for (int d = 0; d < tensor.dim(); ++d) {
          shapeStr += std::to_string(tensor.size(d)) + (d == tensor.dim() - 1 ? "" : ", ");
        }
        shapeStr += "]";

        ALOGI("  Vision Output %zu: Size=%zu, Rank=%d, Shape=%s, Type=%i", 
              i, numel, (int)tensor.dim(), shapeStr.c_str(), (int)scalarType);
        
        if (numel == targetDim) {
          ALOGI("    -> Potential vision match at output %zu", i);
          found = true;
          bestOutputIndex = i; // Overwrite to pick the LAST matching 768-dim tensor (likely head)
        }
      }
    }

    if (found) {
      ALOGI("  -> Final Selection: Output %zu", bestOutputIndex);
      const auto& tensor = outputs[bestOutputIndex].toTensor();
      const auto scalarType = tensor.scalar_type();
      const size_t numel = static_cast<size_t>(tensor.numel());

      outputDataVector.resize(numel);
      if (scalarType == executorch::aten::ScalarType::Float) {
        const float* data = tensor.const_data_ptr<float>();
        std::copy(data, data + numel, outputDataVector.begin());
      } else if (scalarType == executorch::aten::ScalarType::Half) {
        const _Float16* halfPtr = reinterpret_cast<const _Float16*>(tensor.const_data_ptr<uint16_t>());
        for (size_t j = 0; j < numel; j++) {
          outputDataVector[j] = static_cast<float>(halfPtr[j]);
        }
      } else {
      }
    }

    // 2. Fallback: If we have a sequence output in the FIRST tensor, GAP it
    if (!found && !outputs.empty() && outputs[0].isTensor()) {
      const auto& outputTensor = outputs[0].toTensor();
      const float* outputData = outputTensor.const_data_ptr<float>();
      const size_t outputSize = static_cast<size_t>(outputTensor.numel());
      
      if (outputSize > 0 && outputSize % targetDim == 0) {
        size_t numTokens = outputSize / targetDim;
        outputDataVector.assign(targetDim, 0.0f);
        for (size_t k = 0; k < numTokens; ++k) {
          for (size_t j = 0; j < targetDim; ++j) {
            outputDataVector[j] += outputData[k * targetDim + j];
          }
        }
        for (size_t j = 0; j < targetDim; ++j) {
          outputDataVector[j] /= static_cast<float>(numTokens);
        }
        found = true;
      } else {
        throw SigLIPException(
            SigLIPErrorCode::Inference,
            "Output dimension mismatch. Primary output size: " + std::to_string(outputSize));
      }
    }
    
    if (!found) {
      throw SigLIPException(
          SigLIPErrorCode::Inference,
          "Inference failed: Model returned no applicable tensor outputs.");
    }

    uint8_t* byteData = reinterpret_cast<uint8_t*>(outputDataVector.data());
    size_t byteLength = outputDataVector.size() * sizeof(float);
    
    auto arrayBuffer = ArrayBuffer::allocate(byteLength);
    std::memcpy(arrayBuffer->data(), byteData, byteLength);
    promise->resolve(arrayBuffer);

#else
    throw SigLIPException(
        SigLIPErrorCode::Inference,
        "Inference unavailable because ExecuTorch is not linked.");
#endif
  } catch (const std::exception& e) {
    promise->reject(std::current_exception());
  }

  return promise;
}

std::shared_ptr<Promise<std::shared_ptr<ArrayBuffer>>> SigLIPHybridObject::getTextEmbedding(
    const std::shared_ptr<ArrayBuffer>& tokens) {
  auto promise = Promise<std::shared_ptr<ArrayBuffer>>::create();

  try {
    std::lock_guard<std::mutex> lock(mutex_);
#if RNSIGLIP_HAS_EXECUTORCH
    if (!pimpl_->textModule) {
      throw SigLIPException(
          SigLIPErrorCode::NotLoaded,
          "Text Module pointer is null. Call loadTextModel() before getTextEmbedding().");
    }

    if (!modelInfo_) {
      throw SigLIPException(
          SigLIPErrorCode::NotLoaded,
          "Model info is null.");
    }

    // The JS `tokens` is a NativeArrayBuffer populated using BigInt64Array.
    // We can directly cast its memory to int64_t, achieving zero-copy transfer.
    int64_t* inputData = reinterpret_cast<int64_t*>(tokens->data());
    
    // Calculate how many 64-bit integers we have in the buffer
    const size_t numTokens = tokens->size() / sizeof(int64_t);

    const int batchSize = 1;
    const int sequenceLength = static_cast<int>(numTokens);
    
    std::vector<executorch::aten::SizesType> sizes = {batchSize, sequenceLength};
    std::vector<executorch::aten::DimOrderType> dim_order = {0, 1};
    std::vector<executorch::aten::StridesType> strides = {sequenceLength, 1};
    
    executorch::aten::TensorImpl inputTensorImpl(
        executorch::aten::ScalarType::Long,
        sizes.size(),
        sizes.data(),
        inputData,
        dim_order.data(),
        strides.data(),
        executorch::aten::TensorShapeDynamism::STATIC);
    
    executorch::aten::Tensor inputTensor(&inputTensorImpl);
    
    std::vector<executorch::runtime::EValue> inputs;
    inputs.emplace_back(inputTensor);
    
    auto result = pimpl_->textModule->forward(inputs);
    if (!result.ok()) {
      auto error = result.error();
      throw SigLIPException(
          SigLIPErrorCode::Inference,
          "Module forward() failed during text inference. Error Code: " + std::to_string((int)error));
    }
    
    auto& outputs = result.get();
    ALOGI("Text Inference completed. Output tensors: %zu", outputs.size());
    const size_t targetDim = static_cast<size_t>(modelInfo_->dimension);
    
    std::vector<float> outputDataVector;
    bool found = false;

    for (size_t i = 0; i < outputs.size(); ++i) {
      if (outputs[i].isTensor()) {
        const auto& tensor = outputs[i].toTensor();
        const size_t numel = static_cast<size_t>(tensor.numel());
        auto scalarType = tensor.scalar_type();
        
        ALOGI("  Text Output %zu: Size=%zu, Rank=%d, Type=%i", 
              i, numel, (int)tensor.dim(), (int)scalarType);
        
        if (!found && numel == targetDim) {
          found = true;
          outputDataVector.resize(numel);
          if (scalarType == executorch::aten::ScalarType::Float) {
            const float* data = tensor.const_data_ptr<float>();
            std::copy(data, data + numel, outputDataVector.begin());
            ALOGI("  -> Selected Output %zu (Float32)", i);
          } else if (scalarType == executorch::aten::ScalarType::Half) {
            const _Float16* halfPtr = reinterpret_cast<const _Float16*>(tensor.const_data_ptr<uint16_t>());
            for (size_t k = 0; k < numel; ++k) {
              outputDataVector[k] = static_cast<float>(halfPtr[k]);
            }
            ALOGI("  -> Selected Output %zu (Converted Half->Float32)", i);
          }
        }
      }
    }

    if (!found && !outputs.empty() && outputs[0].isTensor()) {
      const auto& outputTensor = outputs[0].toTensor();
      const float* outputData = outputTensor.const_data_ptr<float>();
      const size_t outputSize = static_cast<size_t>(outputTensor.numel());

      if (outputSize > 0 && outputSize % targetDim == 0) {
        size_t numTokens = outputSize / targetDim;
        outputDataVector.assign(targetDim, 0.0f);
        
        ALOGI("Text inference: Fallback GAP pooling over %zu tokens", numTokens);
        
        for (size_t k = 0; k < numTokens; ++k) {
          for (size_t j = 0; j < targetDim; ++j) {
            outputDataVector[j] += outputData[k * targetDim + j];
          }
        }
        for (size_t j = 0; j < targetDim; ++j) {
          outputDataVector[j] /= static_cast<float>(numTokens);
        }
        found = true;
      }
    }

    if (!found) {
      throw SigLIPException(
          SigLIPErrorCode::Inference,
          "Text inference failed: Model returned no applicable tensor outputs.");
    }

    uint8_t* byteData = reinterpret_cast<uint8_t*>(outputDataVector.data());
    size_t byteLength = outputDataVector.size() * sizeof(float);
    
    auto arrayBuffer = ArrayBuffer::allocate(byteLength);
    std::memcpy(arrayBuffer->data(), byteData, byteLength);
    promise->resolve(arrayBuffer);

#else
    throw SigLIPException(
        SigLIPErrorCode::Inference,
        "Inference unavailable because ExecuTorch is not linked.");
#endif
  } catch (const std::exception& e) {
    promise->reject(std::current_exception());
  }

  return promise;
}

void SigLIPHybridObject::unloadModels() {
  std::lock_guard<std::mutex> lock(mutex_);
  try {
#if RNSIGLIP_HAS_EXECUTORCH
    pimpl_->visionModule.reset();
    pimpl_->textModule.reset();
#endif
    modelInfo_.reset();
  } catch (const std::exception& e) {
    throw SigLIPException(
        SigLIPErrorCode::ModelUnload,
        std::string("Failed to unload models: ") + e.what());
  }
}

std::variant<nitro::NullType, SigLIPModelInfo> SigLIPHybridObject::getModelInfo() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (modelInfo_) {
    return *modelInfo_;
  }
  return nitro::NullType();
}

}  // namespace margelo::nitro::rnsiglip
