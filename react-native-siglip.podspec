Pod::Spec.new do |s|
  package = JSON.parse(File.read(File.join(__dir__, "package.json")))

  s.name         = "react-native-siglip"
  s.version      = package["version"]
  s.summary      = "General-purpose SigLIP embeddings for React Native via Nitro + ExecuTorch."
  s.homepage     = "https://github.com/hugh/react-native-siglip"
  s.license      = { :type => "MIT" }
  s.author       = { "Hugh" => "hugh@litert.dev" }
  s.platforms    = { :ios => "14.0" }
  s.source       = { :git => "https://github.com/hugh/react-native-siglip.git", :tag => s.version.to_s }

  s.source_files = [
    "cpp/**/*.{h,hpp,c,cc,cpp}",
    "ios/*.{h,m,mm,cpp}",
    "nitrogen/generated/shared/**/*.{h,hpp,c,cpp,swift}",
    "nitrogen/generated/ios/**/*.{h,hpp,c,cpp,mm,swift}"
  ]

  s.public_header_files = [
    "nitrogen/generated/shared/**/*.{h,hpp}",
    "nitrogen/generated/ios/NitroSigLIP-Swift-Cxx-Bridge.hpp"
  ]

  s.private_header_files = [
    "cpp/**/*.hpp",
    "nitrogen/generated/ios/c++/**/*.{h,hpp}"
  ]

  s.pod_target_xcconfig = {
    "CLANG_CXX_LANGUAGE_STANDARD" => "c++20",
    "OTHER_LDFLAGS" => [
      '$(inherited)',
      '-ObjC',
      '-all_load',
      "-force_load \"$(PODS_TARGET_SRCROOT)/dist/ios/executorch.xcframework/ios-arm64/libexecutorch.a\"",
      "-force_load \"$(PODS_TARGET_SRCROOT)/dist/ios/backend_coreml.xcframework/ios-arm64/libbackend_coreml.a\"",
      "-force_load \"$(PODS_TARGET_SRCROOT)/dist/ios/backend_mps.xcframework/ios-arm64/libbackend_mps.a\"",
      "-force_load \"$(PODS_TARGET_SRCROOT)/dist/ios/backend_xnnpack.xcframework/ios-arm64/libbackend_xnnpack.a\"",
      "-force_load \"$(PODS_TARGET_SRCROOT)/dist/ios/kernels_optimized.xcframework/ios-arm64/libkernels_optimized.a\"",
      "-force_load \"$(PODS_TARGET_SRCROOT)/dist/ios/kernels_quantized.xcframework/ios-arm64/libkernels_quantized.a\"",
      "-force_load \"$(PODS_TARGET_SRCROOT)/dist/ios/kernels_torchao.xcframework/ios-arm64/libkernels_torchao.a\""
    ].join(' '),
    "SWIFT_OBJC_INTEROP_MODE" => "objcxx",
    "DEFINES_MODULE" => "NO",
    "HEADER_SEARCH_PATHS" => [
      '"$(inherited)"',
      '"$(PODS_TARGET_SRCROOT)"',
      '"$(PODS_TARGET_SRCROOT)/cpp"',
      '"$(PODS_TARGET_SRCROOT)/nitrogen/generated/shared/c++"',
      '"$(PODS_TARGET_SRCROOT)/nitrogen/generated/ios"'
    ].join(' '),
    "GCC_PREPROCESSOR_DEFINITIONS" => '$(inherited) RNSIGLIP_HAS_EXECUTORCH=1'
  }

  s.dependency "NitroModules"

  # ExecuTorch Binaries (XCframeworks)
  s.vendored_frameworks = [
    "dist/ios/executorch.xcframework",
    "dist/ios/backend_coreml.xcframework",
    "dist/ios/backend_mps.xcframework",
    "dist/ios/backend_xnnpack.xcframework",
    "dist/ios/kernels_optimized.xcframework",
    "dist/ios/kernels_quantized.xcframework",
    "dist/ios/kernels_torchao.xcframework",
    "dist/ios/threadpool.xcframework"
  ]

  s.frameworks = ["CoreML", "Accelerate", "UIKit", "Foundation", "CoreGraphics", "Metal", "MetalPerformanceShaders", "MetalPerformanceShadersGraph"]
  s.libraries = ["c++", "sqlite3"]
end
