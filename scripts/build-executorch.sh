#!/usr/bin/env bash
# ============================================================================
# build-executorch.sh
# Build ExecuTorch xcframework (iOS) and AAR (Android) with delegate support.
# 
# Usage:
#   ./scripts/build-executorch.sh ios      # Build iOS xcframework
#   ./scripts/build-executorch.sh android  # Build Android AAR (XNNPACK)
#   ./scripts/build-executorch.sh all      # Build both
#
# Environment Variables:
#   EXECUTORCH_VERSION  - Git tag or branch (default: v1.1.0)
#   ANDROID_HOME        - Android SDK path (required for Android)
#   ANDROID_NDK         - Android NDK path (required for Android, r28c+ recommended)
#   BUILD_QNN           - Set to "1" to include QNN delegate (requires Hexagon SDK)
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="$PROJECT_ROOT/.executorch-build"
DIST_DIR="$PROJECT_ROOT/dist"

# Configuration
EXECUTORCH_VERSION="${EXECUTORCH_VERSION:-v1.1.0}"
EXECUTORCH_REPO="https://github.com/pytorch/executorch.git"
BUILD_QNN="${BUILD_QNN:-0}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1" >&2; }

# ============================================================================
# Setup Functions
# ============================================================================

setup_executorch() {
  log_info "Setting up ExecuTorch $EXECUTORCH_VERSION..."
  
  if [ -d "$BUILD_DIR/executorch" ]; then
    log_info "ExecuTorch directory exists, updating..."
    cd "$BUILD_DIR/executorch"
    # Fetch specific tag to handle shallow clones
    log_info "Fetching $EXECUTORCH_VERSION..."
    git fetch origin tag "$EXECUTORCH_VERSION" --no-tags
    git reset --hard
    git clean -fd
    git checkout "$EXECUTORCH_VERSION"
    git submodule sync
    git submodule update --init --recursive
  else
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"
    log_info "Cloning ExecuTorch..."
    git clone --depth 1 --branch "$EXECUTORCH_VERSION" "$EXECUTORCH_REPO"
    cd executorch
    git submodule sync
    git submodule update --init --recursive
  fi
  
  # Install Python dependencies
  log_info "Installing Python dependencies..."
  
  if [ ! -d ".venv" ]; then
      log_info "Creating virtual environment with python3.11..."
      /opt/homebrew/bin/python3.11 -m venv .venv
  fi
  source .venv/bin/activate
  
  # Upgrade pip just in case
  python3 -m pip install --upgrade pip
  
  python3 -m pip install -q cmake torch torchvision
  python3 -m pip install -q PyYAML certifi requests zstd
  # python3 -m pip install -q -e ".[devtools]" --no-build-isolation
  
  log_info "ExecuTorch setup complete."
  
  # Discover or build host flatc
  log_info "Setting up flatc..."
  cd "$BUILD_DIR/executorch/third-party/flatbuffers"
  if [ ! -f "flatc" ] && [ ! -f "build-host/flatc" ]; then
    cmake -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release
    make -j8
  fi
  
  FLATC_PATH="$BUILD_DIR/executorch/third-party/flatbuffers/flatc"
  if [ ! -f "$FLATC_PATH" ]; then
      FLATC_PATH="$BUILD_DIR/executorch/third-party/flatbuffers/build-host/flatc"
  fi
  log_info "Using flatc at: $FLATC_PATH"
  export FLATBUFFERS_FLATC_EXECUTABLE="$FLATC_PATH"
}

# ============================================================================
# iOS Build
# ============================================================================

build_ios() {
  log_info "Building iOS xcframework..."
  
  if ! command -v xcodebuild &> /dev/null; then
    log_error "xcodebuild not found. This command must be run on macOS."
    exit 1
  fi
  
  cd "$BUILD_DIR/executorch"
  
  # Build CoreML and MPS backends for iOS
  log_info "Building iOS frameworks with CoreML and MPS delegates..."
  
  # Use the official iOS build script if available
  if [ -f "scripts/build_apple_frameworks.sh" ]; then
    # Patch build_apple_frameworks.sh to accept CMAKE_ARGS
    if grep -q "CMAKE_ARGS" ./scripts/build_apple_frameworks.sh; then
        log_info "build_apple_frameworks.sh already patched."
    else
        log_info "Patching build_apple_frameworks.sh to accept CMAKE_ARGS..."
        sed -i '' 's/${CMAKE_OPTIONS_OVERRIDE\[@\]:-}/${CMAKE_OPTIONS_OVERRIDE[@]:-} ${CMAKE_ARGS}/' ./scripts/build_apple_frameworks.sh
    fi

    # Patch backends/apple/mps/CMakeLists.txt to use host flatc
    MPS_CMAKE="$PROJECT_ROOT/.executorch-build/executorch/backends/apple/mps/CMakeLists.txt"
    if [ -f "$MPS_CMAKE" ]; then
        log_info "Patching MPS CMakeLists.txt..."
        # Replace 'flatc --cpp' with '${FLATBUFFERS_FLATC_EXECUTABLE} --cpp'
        sed -i '' 's/flatc --cpp/${FLATBUFFERS_FLATC_EXECUTABLE} --cpp/' "$MPS_CMAKE"
        # Remove 'DEPENDS flatc' to avoid building/depending on target flatc
        sed -i '' '/DEPENDS flatc/d' "$MPS_CMAKE"
    fi
    
    # Patch backends/xnnpack/CMakeLists.txt to use host flatc
    XNNPACK_CMAKE="$PROJECT_ROOT/.executorch-build/executorch/backends/xnnpack/CMakeLists.txt"
    if [ -f "$XNNPACK_CMAKE" ]; then
        log_info "Patching XNNPACK CMakeLists.txt..."
        # Replace 'flatc --cpp' with '${FLATBUFFERS_FLATC_EXECUTABLE} --cpp'
        sed -i '' 's/flatc --cpp/${FLATBUFFERS_FLATC_EXECUTABLE} --cpp/' "$XNNPACK_CMAKE"
        # Remove 'DEPENDS flatc' to avoid building/depending on target flatc
        sed -i '' '/DEPENDS flatc/d' "$XNNPACK_CMAKE"
    fi
    
    # Patch schema/CMakeLists.txt to use host flatc
    SCHEMA_CMAKE="$PROJECT_ROOT/.executorch-build/executorch/schema/CMakeLists.txt"
    if [ -f "$SCHEMA_CMAKE" ]; then
        log_info "Patching schema/CMakeLists.txt..."
        # Replace 'flatc --cpp' with '${FLATBUFFERS_FLATC_EXECUTABLE} --cpp'
        sed -i '' 's/flatc --cpp/${FLATBUFFERS_FLATC_EXECUTABLE} --cpp/' "$SCHEMA_CMAKE"
        # Remove 'DEPENDS flatc' to avoid building/depending on target flatc
        sed -i '' '/DEPENDS flatc/d' "$SCHEMA_CMAKE"
    fi

    # Patch extension/flat_tensor/serialize/CMakeLists.txt to use host flatc
    FLAT_TENSOR_CMAKE="$PROJECT_ROOT/.executorch-build/executorch/extension/flat_tensor/serialize/CMakeLists.txt"
    if [ -f "$FLAT_TENSOR_CMAKE" ]; then
        log_info "Patching extension/flat_tensor/serialize/CMakeLists.txt..."
        # Replace 'flatc --cpp' with '${FLATBUFFERS_FLATC_EXECUTABLE} --cpp'
        sed -i '' 's/flatc --cpp/${FLATBUFFERS_FLATC_EXECUTABLE} --cpp/' "$FLAT_TENSOR_CMAKE"
        # Remove 'DEPENDS flatc' to avoid building/depending on target flatc
        sed -i '' '/DEPENDS flatc/d' "$FLAT_TENSOR_CMAKE"
    fi
    
    COREML=ON MPS=ON CMAKE_ARGS="-DFLATBUFFERS_FLATC_EXECUTABLE=$FLATC_PATH" ./scripts/build_apple_frameworks.sh
  else
    # Manual CMake build for older versions
    log_info "Using manual CMake build..."
    
    # Configure for iOS device (arm64)
    cmake -S . -B cmake-out-ios \
      -G Xcode \
      -DCMAKE_TOOLCHAIN_FILE=third-party/ios-cmake/ios.toolchain.cmake \
      -DPLATFORM=OS64 \
      -DEXECUTORCH_BUILD_COREML=ON \
      -DEXECUTORCH_BUILD_MPS=ON \
      -DEXECUTORCH_BUILD_XNNPACK=ON \
      -DFLATBUFFERS_FLATC_EXECUTABLE=$(which flatc) \
      -DCMAKE_BUILD_TYPE=Release
    
    # Build for device
    cmake --build cmake-out-ios --config Release -- -sdk iphoneos
    
    # Configure for iOS simulator (arm64)
    cmake -S . -B cmake-out-ios-sim \
      -G Xcode \
      -DCMAKE_TOOLCHAIN_FILE=third-party/ios-cmake/ios.toolchain.cmake \
      -DPLATFORM=SIMULATORARM64 \
      -DEXECUTORCH_BUILD_COREML=ON \
      -DEXECUTORCH_BUILD_MPS=ON \
      -DEXECUTORCH_BUILD_XNNPACK=ON \
      -DFLATBUFFERS_FLATC_EXECUTABLE=$(which flatc) \
      -DCMAKE_BUILD_TYPE=Release
    
    # Build for simulator
    cmake --build cmake-out-ios-sim --config Release -- -sdk iphonesimulator
    
    # Create xcframework
    log_info "Creating xcframework..."
    mkdir -p "$DIST_DIR/ios"
    
    # This is a simplified version - actual implementation depends on 
    # ExecuTorch's output structure
    xcodebuild -create-xcframework \
      -library cmake-out-ios/Release-iphoneos/libexecutorch.a \
      -library cmake-out-ios-sim/Release-iphonesimulator/libexecutorch.a \
      -output "$DIST_DIR/ios/executorch.xcframework"
  fi
  
  # Copy headers
echo "[INFO] Copying headers..."
rm -rf "$PROJECT_ROOT/ios/Frameworks/include"
mkdir -p "$PROJECT_ROOT/ios/Frameworks/include"

# Copy standard ExecuTorch headers
cp -R "$PROJECT_ROOT/.executorch-build/executorch/cmake-out/executorch.xcframework/ios-arm64/Headers/"* "$PROJECT_ROOT/ios/Frameworks/include/"

# Copy c10 and torch headers (required for missing irange.h, macros, etc.)
echo "[INFO] Copying c10 and torch headers..."
cp -R "$PROJECT_ROOT/.executorch-build/executorch/runtime/core/portable_type/c10/c10" "$PROJECT_ROOT/ios/Frameworks/include/"
cp -R "$PROJECT_ROOT/.executorch-build/executorch/runtime/core/portable_type/c10/torch" "$PROJECT_ROOT/ios/Frameworks/include/"
# Copy cmake_macros.h from python environment
cp "$PROJECT_ROOT/.executorch-build/executorch/.venv/lib/python3.11/site-packages/torch/include/torch/headeronly/macros/cmake_macros.h" "$PROJECT_ROOT/ios/Frameworks/include/torch/headeronly/macros/"

# Copy generated flatbuffer headers
find "$PROJECT_ROOT/.executorch-build/executorch" -name "*_generated.h" -exec cp {} "$PROJECT_ROOT/ios/Frameworks/include/" \;

# Copy third-party flatbuffers
cp -R "$PROJECT_ROOT/.executorch-build/executorch/third-party/flatbuffers/include/flatbuffers" "$PROJECT_ROOT/ios/Frameworks/include/"

echo "[INFO] Copying XCFrameworks to dist/ios..."
rm -rf "$PROJECT_ROOT/dist/ios"
mkdir -p "$PROJECT_ROOT/dist/ios"

# Copy XCFrameworks
find "$PROJECT_ROOT/.executorch-build/executorch/cmake-out" -name "*.xcframework" -exec cp -R {} "$PROJECT_ROOT/dist/ios/" \;

# Copy static libraries (Release only) to avoid duplicate symbols
echo "[INFO] Copying static libraries (Release only)..."
find "$PROJECT_ROOT/.executorch-build/executorch/cmake-out" -name "libbackend_*.a" ! -name "*_debug.a" -exec cp {} "$PROJECT_ROOT/dist/ios/" \;

# Run the fix script
python3 "$PROJECT_ROOT/scripts/fix_xcframeworks.py"

echo "[SUCCESS] ExecuTorch iOS build complete. Output: $DIST_DIR/ios/"
}

# ============================================================================
# Android Build
# ============================================================================

build_android() {
  log_info "Building Android AAR..."
  
  # Validate environment
  if [ -z "${ANDROID_HOME:-}" ]; then
    log_error "ANDROID_HOME is not set. Please set it to your Android SDK path."
    exit 1
  fi
  
  if [ -z "${ANDROID_NDK:-}" ]; then
    # Try to find NDK in SDK
    if [ -d "$ANDROID_HOME/ndk" ]; then
      ANDROID_NDK=$(ls -d "$ANDROID_HOME/ndk/"* 2>/dev/null | head -1)
      log_warn "ANDROID_NDK not set, using: $ANDROID_NDK"
    else
      log_error "ANDROID_NDK is not set. Please set it to your Android NDK path."
      exit 1
    fi
  fi
  
  export ANDROID_HOME
  export ANDROID_NDK
  export ANDROID_SDK=$ANDROID_HOME
  
  cd "$BUILD_DIR/executorch"
  
  # Build using official script (includes XNNPACK by default)
  log_info "Building Android AAR with XNNPACK backend..."
  
  if [ -f "scripts/build_android_library.sh" ]; then
    # Patch build_android_library.sh to accept FLATBUFFERS_FLATC_EXECUTABLE
    if grep -q "FLATBUFFERS_FLATC_EXECUTABLE" ./scripts/build_android_library.sh; then
        log_info "build_android_library.sh already patched."
    else
        log_info "Patching build_android_library.sh for flatc..."
        sed -i '' "s|cmake \.|cmake . -DFLATBUFFERS_FLATC_EXECUTABLE=$FLATC_PATH |" ./scripts/build_android_library.sh
    fi
    
    # Official script handles everything
    sh scripts/build_android_library.sh
  else
    log_error "build_android_library.sh not found in ExecuTorch"
    exit 1
  fi
  
  # Copy AAR to dist
  mkdir -p "$DIST_DIR/android"
  
  # Find and copy the AAR file
  AAR_FILE=$(find . -name "*.aar" -path "*android*" | head -1)
  if [ -n "$AAR_FILE" ]; then
    cp "$AAR_FILE" "$DIST_DIR/android/executorch.aar"
    log_info "Android AAR copied to: $DIST_DIR/android/executorch.aar"
  else
    log_warn "AAR file not found. Check build output."
  fi
  
  # Copy static libraries for each ABI
  for abi in "arm64-v8a" "x86_64"; do
      ABI_DIR="cmake-out-android-$abi"
      if [ -d "$ABI_DIR" ]; then
          mkdir -p "$DIST_DIR/android/libs/$abi"
          find "$ABI_DIR" -name "*.a" -exec cp {} "$DIST_DIR/android/libs/$abi/" \;
          log_info "Copied static libraries for $abi to $DIST_DIR/android/libs/$abi"
      fi
  done
  
  # QNN build (optional, requires Hexagon SDK)
  if [ "$BUILD_QNN" = "1" ]; then
    log_warn "QNN build requested. This requires Qualcomm Hexagon SDK."
    log_warn "QNN build is not automated - please build manually with SDK installed."
  fi
  
  log_info "Android build complete. Output: $DIST_DIR/android/"
}

# ============================================================================
# Main
# ============================================================================

main() {
  local target="${1:-all}"
  
  log_info "========================================"
  log_info "react-native-siglip ExecuTorch Builder"
  log_info "Version: $EXECUTORCH_VERSION"
  log_info "Target: $target"
  log_info "========================================"
  
  mkdir -p "$DIST_DIR"
  
  setup_executorch
  
  # Patch XNNCompiler.cpp to avoid duplicate case label error for Sin/Cos
  cd "$BUILD_DIR/executorch"
  XNN_COMPILER_CPP="backends/xnnpack/runtime/XNNCompiler.cpp"
  if [ -f "$XNN_COMPILER_CPP" ]; then
    if grep -q "patched by patch_xnnpack.py" "$XNN_COMPILER_CPP" 2>/dev/null; then
        log_info "XNNCompiler.cpp already patched."
    else
        log_info "Patching XNNCompiler.cpp using robust Python script..."
        python3 "$PROJECT_ROOT/scripts/patch_xnnpack.py" "$XNN_COMPILER_CPP"
        echo "// patched by patch_xnnpack.py" >> "$XNN_COMPILER_CPP"
    fi
  fi
  cd "$PROJECT_ROOT"
  
  case "$target" in
    ios)
      build_ios
      ;;
    android)
      build_android
      ;;
    all)
      # Build iOS first (requires macOS)
      if [[ "$(uname)" == "Darwin" ]]; then
        build_ios
      else
        log_warn "Skipping iOS build (not on macOS)"
      fi
      build_android
      ;;
    *)
      log_error "Unknown target: $target"
      echo "Usage: $0 [ios|android|all]"
      exit 1
      ;;
  esac
  
  log_info "========================================"
  log_info "Build complete!"
  log_info "Outputs: $DIST_DIR"
  log_info "========================================"
}

main "$@"
