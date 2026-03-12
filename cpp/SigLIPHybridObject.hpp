#pragma once

#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include <NitroModules/Promise.hpp>
#include <NitroModules/ArrayBuffer.hpp>

#include "HybridSigLIPSpec.hpp"
#include "SigLIPError.hpp"

namespace margelo::nitro::rnsiglip {

using namespace margelo::nitro;

/**
 * Core native implementation. Wire this class into Nitro-generated bindings.
 */
class SigLIPHybridObject : public HybridSigLIPSpec {
 public:
  SigLIPHybridObject();
  ~SigLIPHybridObject() override;

  std::shared_ptr<Promise<void>> loadVisionModel(const std::string& path) override;
  std::shared_ptr<Promise<void>> loadTextModel(const std::string& path) override;
  std::shared_ptr<Promise<std::shared_ptr<ArrayBuffer>>> getImageEmbedding(const std::string& imageUri) override;
  std::shared_ptr<Promise<std::shared_ptr<ArrayBuffer>>> getTextEmbedding(const std::shared_ptr<ArrayBuffer>& tokens) override;
  void unloadModels() override;
  std::variant<nitro::NullType, SigLIPModelInfo> getModelInfo() override;

 protected:
  size_t getExternalMemorySize() noexcept override;

 private:
  std::string normalizePath(const std::string& path) const;
  void validateReadableFile(
      const std::string& absolutePath,
      ::rnsiglip::SigLIPErrorCode notFoundCode) const;

  mutable std::mutex mutex_;
  struct Impl;
  std::unique_ptr<Impl> pimpl_;

  std::optional<SigLIPModelInfo> modelInfo_;
};

}  // namespace margelo::nitro::rnsiglip
