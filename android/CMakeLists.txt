cmake_minimum_required(VERSION 3.13)
project(NitroSigLIP)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(
  NitroSigLIP
  SHARED
  ../cpp/SigLIPHybridObject.cpp
  ../cpp/image/ImagePreprocessor.cpp
  ../cpp/image/ImageDecoder.cpp
  src/main/cpp/SigLIPImageDecoder.cpp
  src/main/cpp/cpp-adapter.cpp
)

target_include_directories(
  NitroSigLIP
  PRIVATE
  ../cpp
  ..
)

find_library(log-lib log)
find_library(android-lib android)

# Link ExecuTorch using extracted AAR libraries
set(EXECUTORCH_BUILD_ROOT "${CMAKE_SOURCE_DIR}/../.executorch-build/executorch")
set(EXECUTORCH_ABI_ROOT "${EXECUTORCH_BUILD_ROOT}/cmake-out-android-${ANDROID_ABI}")

# Add headers from build tree
target_include_directories(
  NitroSigLIP
  PRIVATE
  "${EXECUTORCH_ABI_ROOT}/include"
  "${EXECUTORCH_ABI_ROOT}/include/executorch/runtime/core/portable_type/c10"
  "${EXECUTORCH_BUILD_ROOT}"
)

# Use static libraries from our dist folder instead of stripped JNI .so
set(EXECUTORCH_LIBS_DIR "${CMAKE_SOURCE_DIR}/../dist/android/libs/${ANDROID_ABI}")
file(GLOB EXECUTORCH_A_LIBS "${EXECUTORCH_LIBS_DIR}/*.a")

# Exclude host libraries (like flatcc) which cause ET_REL errors during Android linking
list(FILTER EXECUTORCH_A_LIBS EXCLUDE REGEX "libflatcc.*\\.a$")

# Libraries that require static initialization (ops and backends)
set(EXECUTORCH_WHOLE_LIBS "")

# Prefer optimized ops if available, otherwise fallback to portable
if(EXISTS "${EXECUTORCH_LIBS_DIR}/liboptimized_native_cpu_ops_lib.a")
  list(APPEND EXECUTORCH_WHOLE_LIBS "${EXECUTORCH_LIBS_DIR}/liboptimized_native_cpu_ops_lib.a")
elseif(EXISTS "${EXECUTORCH_LIBS_DIR}/libportable_ops_lib.a")
  list(APPEND EXECUTORCH_WHOLE_LIBS "${EXECUTORCH_LIBS_DIR}/libportable_ops_lib.a")
endif()

if(EXISTS "${EXECUTORCH_LIBS_DIR}/libquantized_ops_lib.a")
  list(APPEND EXECUTORCH_WHOLE_LIBS "${EXECUTORCH_LIBS_DIR}/libquantized_ops_lib.a")
endif()

if(EXISTS "${EXECUTORCH_LIBS_DIR}/libcustom_ops.a")
  list(APPEND EXECUTORCH_WHOLE_LIBS "${EXECUTORCH_LIBS_DIR}/libcustom_ops.a")
endif()

file(GLOB EXECUTORCH_BACKEND_LIBS "${EXECUTORCH_LIBS_DIR}/lib*backend.a")
list(APPEND EXECUTORCH_WHOLE_LIBS ${EXECUTORCH_BACKEND_LIBS})

# Remove them from the general list to avoid passing twice
foreach(lib IN LISTS EXECUTORCH_WHOLE_LIBS)
  list(REMOVE_ITEM EXECUTORCH_A_LIBS ${lib})
endforeach()

target_link_libraries(
  NitroSigLIP
  ${log-lib}
  ${android-lib}
  jnigraphics
  -Wl,--start-group
  -Wl,--whole-archive
  ${EXECUTORCH_WHOLE_LIBS}
  -Wl,--no-whole-archive
  ${EXECUTORCH_A_LIBS}
  -Wl,--end-group
)

target_compile_definitions(NitroSigLIP PUBLIC RNSIGLIP_HAS_EXECUTORCH=1 C10_USING_CUSTOM_GENERATED_MACROS)

# Include Nitrogen-generated autolinking (adds specs, OnLoad, and NitroModules deps)
include(${CMAKE_SOURCE_DIR}/../nitrogen/generated/android/NitroSigLIP+autolinking.cmake)
