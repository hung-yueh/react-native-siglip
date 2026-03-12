#pragma once

#include <stdexcept>
#include <string>

namespace rnsiglip {

enum class SigLIPErrorCode {
  ModelNotFound,
  ModelLoad,
  ModelUnload,
  ImageNotFound,
  ImageDecode,
  Inference,
  NotLoaded,
  InvalidArgument,
};

inline const char* toErrorCodeString(SigLIPErrorCode code) {
  switch (code) {
    case SigLIPErrorCode::ModelNotFound:
      return "ERR_MODEL_NOT_FOUND";
    case SigLIPErrorCode::ModelLoad:
      return "ERR_MODEL_LOAD";
    case SigLIPErrorCode::ModelUnload:
      return "ERR_MODEL_UNLOAD";
    case SigLIPErrorCode::ImageNotFound:
      return "ERR_IMAGE_NOT_FOUND";
    case SigLIPErrorCode::ImageDecode:
      return "ERR_IMAGE_DECODE";
    case SigLIPErrorCode::Inference:
      return "ERR_INFERENCE";
    case SigLIPErrorCode::NotLoaded:
      return "ERR_NOT_LOADED";
    case SigLIPErrorCode::InvalidArgument:
      return "ERR_INVALID_ARGUMENT";
  }
  return "ERR_INFERENCE";
}

class SigLIPException : public std::runtime_error {
 public:
  SigLIPException(SigLIPErrorCode code, std::string message)
      : std::runtime_error(std::move(message)), code_(code) {}

  SigLIPErrorCode code() const { return code_; }
  const char* codeString() const { return toErrorCodeString(code_); }

 private:
  SigLIPErrorCode code_;
};

}  // namespace rnsiglip
