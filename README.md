# react-native-siglip

On-device [SigLIP 2](https://huggingface.co/google/siglip2-base-patch16-224) image & text embedding inference for React Native, powered by [ExecuTorch](https://pytorch.org/executorch/) and [Nitro Modules](https://github.com/nicklockwood/NitroModules).

Compute 768-dimensional vision and text embeddings entirely on-device — no server, no network required at inference time. Compare images to text prompts via cosine similarity for zero-shot classification, visual search, and more.

## Features

- **Dual encoder**: separate vision and text models, load independently
- **Zero-copy native buffers**: embeddings returned as raw `ArrayBuffer` via Nitro Modules (no serialization overhead)
- **Platform-optimized backends**:
  - **iOS**: CoreML delegate (GPU/ANE, FP16)
  - **Android**: XNNPACK delegate (CPU, FP32 with ARM NEON)
  - Qualcomm QNN (Hexagon NPU) support scaffolded
- **On-device preprocessing**: bilinear resize, center crop, CHW normalization — all in C++
- **768-d embeddings** from `google/siglip2-base-patch16-224`
- **Memory-mapped model loading** (`Mmap` mode) — no full model copy into RAM

## Requirements

| Platform | Minimum | Architecture |
|----------|---------|-------------|
| iOS | 14.0 | arm64 |
| Android | SDK 24 (Android 7.0) | arm64-v8a |
| React Native | 0.76+ | New Architecture |

**Peer dependencies**: `react`, `react-native`, `react-native-nitro-modules@^0.34.0`

## Installation

```sh
npm install react-native-siglip
# or
yarn add react-native-siglip
```

A `postinstall` script automatically downloads pre-compiled ExecuTorch binaries (~xcframeworks for iOS, JNI shared libraries for Android) from GitHub Releases into `node_modules/react-native-siglip/dist/`.

> Set `SIGLIP_SKIP_BINARY_DOWNLOAD=1` to skip (e.g. when building binaries locally).

### iOS Setup (without Expo)

```sh
cd ios && pod install
```

The podspec automatically links all required ExecuTorch xcframeworks and system frameworks:

- **Vendored**: `executorch`, `backend_coreml`, `backend_mps`, `backend_xnnpack`, `kernels_optimized`, `kernels_quantized`, `kernels_torchao`, `threadpool`
- **System frameworks**: CoreML, Accelerate, Metal, MetalPerformanceShaders, MetalPerformanceShadersGraph, UIKit, Foundation, CoreGraphics
- **System libraries**: libc++, libsqlite3

No additional Xcode configuration is needed.

### Android Setup (without Expo)

Add the library to your app's `settings.gradle` (React Native CLI autolinking handles this):

```groovy
// settings.gradle — typically auto-configured by RN CLI
include ':react-native-siglip'
project(':react-native-siglip').projectDir = new File(rootProject.projectDir, '../node_modules/react-native-siglip/android')
```

Your app's `build.gradle` should include:

```groovy
dependencies {
    implementation project(':react-native-siglip')
}
```

Ensure your app targets **arm64-v8a** (required for ExecuTorch):

```groovy
// android/app/build.gradle
android {
    defaultConfig {
        ndk {
            abiFilters "arm64-v8a"
        }
    }
    packagingOptions {
        pickFirst "**/libc++_shared.so"
        pickFirst "**/libfbjni.so"
    }
}
```

The library's CMake build automatically links ExecuTorch libraries from `dist/android/libs/arm64-v8a/`.

### Expo Setup

Add the config plugin to `app.json`:

```json
{
  "expo": {
    "plugins": [
      [
        "react-native-siglip",
        {
          "enableCoreML": true,
          "enableQNN": false,
          "modelsDirectory": "siglip_models"
        }
      ]
    ]
  }
}
```

Then run `npx expo prebuild` to generate native projects.

## API

### `createSigLIP()`

Creates a SigLIP hybrid object instance. This is the main entry point.

```typescript
import { createSigLIP } from "react-native-siglip";

const siglip = createSigLIP();
```

### `loadVisionModel(path: string): Promise<void>`

Loads a `.pte` vision model from an absolute local filesystem path or `file://` URI.

```typescript
await siglip.loadVisionModel("/path/to/siglip2_vision_coreml_fp16.pte");
```

### `loadTextModel(path: string): Promise<void>`

Loads a `.pte` text model from an absolute local filesystem path or `file://` URI.

```typescript
await siglip.loadTextModel("/path/to/siglip2_text_coreml_fp16.pte");
```

### `getImageEmbedding(imageUri: string): Promise<ArrayBuffer>`

Runs vision inference on a local image file. Returns a 768-d embedding as a raw `ArrayBuffer` (3072 bytes = 768 × float32).

```typescript
const buffer = await siglip.getImageEmbedding("/path/to/photo.jpg");
const embedding = new Float32Array(buffer); // 768 values
```

The native layer handles all preprocessing:
1. Decode image (platform-native: UIImage on iOS, BitmapFactory on Android)
2. Center crop to 1:1 aspect ratio
3. Bilinear resize to 224×224
4. Normalize: `(pixel / 255.0 - 0.5) / 0.5` → range `[-1, 1]`
5. Convert to CHW tensor layout `[1, 3, 224, 224]`

### `getTextEmbedding(tokens: ArrayBuffer): Promise<ArrayBuffer>`

Runs text inference on pre-tokenized input. Tokens must be packed as `BigInt64Array` (int64 per token, padded to sequence length 64).

```typescript
import { NitroModules } from "react-native-nitro-modules";

// Tokenize (you provide tokenizer — see "Text Tokenization" below)
const tokenIds = tokenizer.encode("a photo of a cat"); // number[]

// Pack into native buffer: 64 tokens × 8 bytes = 512 bytes
const buffer = NitroModules.createNativeArrayBuffer(tokenIds.length * 8);
const view = new BigInt64Array(buffer);
for (let i = 0; i < tokenIds.length; i++) {
  view[i] = BigInt(tokenIds[i]);
}

const result = await siglip.getTextEmbedding(buffer);
const embedding = new Float32Array(result); // 768 values
```

### `unloadModels(): void`

Unloads both vision and text models, freeing native memory.

```typescript
siglip.unloadModels();
```

### `modelInfo: SigLIPModelInfo | null` (readonly)

Returns metadata about the currently loaded model, or `null` if no model is loaded.

```typescript
const info = siglip.modelInfo;
// { width: 224, height: 224, dimension: 768, backend: "coreml",
//   visionModelPath: "...", textModelPath: "..." }
```

## Types

```typescript
type SigLIPBackend = "coreml" | "qnn" | "xnnpack" | "cpu";

interface SigLIPModelInfo {
  width: number;        // Input resolution (224)
  height: number;       // Input resolution (224)
  dimension: number;    // Embedding size (768)
  backend: SigLIPBackend;
  visionModelPath?: string;
  textModelPath?: string;
}

type SigLIPErrorCode =
  | "ERR_MODEL_NOT_FOUND"    // .pte file doesn't exist
  | "ERR_MODEL_LOAD"         // ExecuTorch module load failure
  | "ERR_MODEL_UNLOAD"       // Error freeing resources
  | "ERR_IMAGE_NOT_FOUND"    // Image file doesn't exist
  | "ERR_IMAGE_DECODE"       // Platform decode/preprocessing failure
  | "ERR_INFERENCE"          // forward() call failure
  | "ERR_NOT_LOADED"         // Method called before model loaded
  | "ERR_INVALID_ARGUMENT";  // Empty path, etc.
```

## Model Files

This library loads pre-exported ExecuTorch `.pte` files. **Models are not bundled** — you must download them to the device filesystem before calling `loadVisionModel`/`loadTextModel`.

### Platform-Specific Models

| Platform | Vision Model | Text Model | Backend | Precision |
|----------|-------------|------------|---------|-----------|
| iOS | `siglip2_vision_coreml_fp16.pte` | `siglip2_text_coreml_fp16.pte` | CoreML (GPU) | FP16 mixed |
| Android | `siglip2_vision_xnnpack_fp32.pte` | `siglip2_text_xnnpack_fp32.pte` | XNNPACK (CPU) | FP32 |

> iOS models use a mixed-precision strategy: MatMul ops run in FP16 on the GPU, while `layer_norm` and `softmax` stay in FP32 to avoid ANE subnormal underflow.

### Downloading Models at Runtime

Models should be downloaded to the app's document directory before loading. Example using `expo-file-system` (or any equivalent like `react-native-fs`):

```typescript
import { Platform } from "react-native";
import RNFS from "react-native-fs";

const MODEL_DIR = RNFS.DocumentDirectoryPath + "/siglip_models/";

const config = Platform.select({
  ios: {
    visionUrl: "https://your-cdn.com/siglip2_vision_coreml_fp16.pte",
    textUrl: "https://your-cdn.com/siglip2_text_coreml_fp16.pte",
    visionFile: "siglip2_vision_coreml_fp16.pte",
    textFile: "siglip2_text_coreml_fp16.pte",
  },
  android: {
    visionUrl: "https://your-cdn.com/siglip2_vision_xnnpack_fp32.pte",
    textUrl: "https://your-cdn.com/siglip2_text_xnnpack_fp32.pte",
    visionFile: "siglip2_vision_xnnpack_fp32.pte",
    textFile: "siglip2_text_xnnpack_fp32.pte",
  },
})!;

// Download if not present
await RNFS.mkdir(MODEL_DIR);
const visionPath = MODEL_DIR + config.visionFile;
if (!(await RNFS.exists(visionPath))) {
  await RNFS.downloadFile({ fromUrl: config.visionUrl, toFile: visionPath }).promise;
}
const textPath = MODEL_DIR + config.textFile;
if (!(await RNFS.exists(textPath))) {
  await RNFS.downloadFile({ fromUrl: config.textUrl, toFile: textPath }).promise;
}

// Load
const siglip = createSigLIP();
await siglip.loadVisionModel(visionPath);
await siglip.loadTextModel(textPath);
```

## Text Tokenization

SigLIP 2 uses a SentencePiece tokenizer (Gemma-based, 256K vocabulary). The tokenizer runs in JavaScript — this library only handles the embedding inference.

A reference tokenizer implementation is included in the example app at `example/SiglipTokenizer.ts`. Key details:

- **Vocabulary**: 256,128 tokens loaded from `siglip2_vocab.json` (exported from the HuggingFace model)
- **Algorithm**: Greedy maximal-munch (longest prefix match)
- **EOS token**: id `1`, appended at the end
- **Padding**: Right-pad to length **64** with token id `0`
- **Output**: `number[]` of integer token IDs

```typescript
import { SiglipTokenizer } from "./SiglipTokenizer";

const tokenizer = new SiglipTokenizer();
const tokens = tokenizer.encode("a photo of a cat");
// → [49, 3366, 302, 49, 5. 1, 0, 0, ...] (length 64)
```

> **Tip**: For best zero-shot classification results, wrap labels in `"a photo of a {label}"` before tokenizing.

## Computing Similarity

After obtaining both embeddings, compute cosine similarity:

```typescript
function l2Normalize(v: Float32Array): Float32Array {
  let sumSq = 0;
  for (let i = 0; i < v.length; i++) sumSq += v[i] * v[i];
  const norm = Math.sqrt(Math.max(sumSq, 1e-12));
  const out = new Float32Array(v.length);
  for (let i = 0; i < v.length; i++) out[i] = v[i] / norm;
  return out;
}

function cosineSimilarity(a: Float32Array, b: Float32Array): number {
  let dot = 0, normA = 0, normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

// SigLIP sigmoid scoring (learned logit scale + bias)
function siglipProbability(cosineSim: number): number {
  const t = 112.67;   // exp(4.7245) — model's logit_scale
  const b = -12.0;    // empirically calibrated bias for pooler_output
  return 1 / (1 + Math.exp(-(cosineSim * t + b)));
}

// Usage
const visionEmb = l2Normalize(new Float32Array(await siglip.getImageEmbedding(imagePath)));
const textEmb = l2Normalize(new Float32Array(await siglip.getTextEmbedding(tokenBuffer)));

const sim = cosineSimilarity(visionEmb, textEmb);
const prob = siglipProbability(sim);
console.log(`Match: ${(prob * 100).toFixed(1)}%`);
```

## Model Export

Models are exported from Python using ExecuTorch. Two export notebooks are provided in `scripts/`:

| Script | Target | Backend |
|--------|--------|---------|
| `scripts/export_ios_siglip.ipynb` | iOS | CoreML (FP16, GPU) |
| `scripts/export_android_siglip.ipynb` | Android | XNNPACK (FP32), QNN (FP16, optional) |

Both notebooks:
1. Load `google/siglip2-base-patch16-224` from HuggingFace
2. Wrap vision/text encoders to output `pooler_output` directly
3. Run `torch.export` → ExecuTorch `to_edge` → backend partitioner → `.pte`
4. Generate `assets/validation.json` with reference embeddings for on-device verification

### iOS Export Notes

The iOS CoreML export uses a monkey-patch to inject mixed-precision rules into `coremltools.convert()`:
- **FP16**: All ops except `layer_norm` and `softmax` (runs on GPU via Metal)
- **FP32**: `layer_norm` and `softmax` (prevents ANE subnormal underflow at `~6×10⁻⁵`)

### Android Export Notes

- **XNNPACK FP32**: Universal, runs on all Android ARM64 devices. No precision issues.
- **XNNPACK int8**: Optional dynamic quantization (~2-4x smaller, requires calibration).
- **QNN FP16**: Qualcomm Hexagon NPU. Requires Qualcomm AI Engine Direct SDK.

## Building ExecuTorch From Source

If you need to build ExecuTorch binaries locally (instead of using the pre-built ones from the postinstall script):

```sh
# Build iOS xcframeworks
./scripts/build-executorch.sh ios

# Build Android static libraries
./scripts/build-executorch.sh android

# Build both
./scripts/build-executorch.sh all
```

**Environment variables**:

| Variable | Default | Description |
|----------|---------|-------------|
| `EXECUTORCH_VERSION` | `v1.1.0` | Git tag to build |
| `ANDROID_HOME` | — | Android SDK path (required for Android) |
| `ANDROID_NDK` | — | Android NDK path (r28c+ recommended) |
| `BUILD_QNN` | `0` | Set to `1` to include QNN delegate |

Output goes to `dist/ios/` and `dist/android/`.

## Architecture

```
┌─────────────────────────────────────────────┐
│  JavaScript                                 │
│  createSigLIP() → loadModel → getEmbedding  │
│  Tokenizer (SentencePiece, JS)              │
└──────────────────┬──────────────────────────┘
                   │ Nitro Modules (zero-copy ArrayBuffer)
┌──────────────────▼──────────────────────────┐
│  C++ Core (shared)                          │
│  SigLIPHybridObject.cpp                     │
│  ├── loadVisionModel / loadTextModel        │
│  │   └── executorch::extension::Module      │
│  │       (Mmap load, no full copy)          │
│  ├── getImageEmbedding                      │
│  │   ├── Platform decode (UIImage / Bitmap) │
│  │   ├── Center crop + bilinear resize      │
│  │   ├── Normalize [-1,1] → CHW tensor      │
│  │   └── module->forward() → 768-d float[]  │
│  └── getTextEmbedding                       │
│      ├── Cast ArrayBuffer → int64_t*        │
│      ├── Build tensor [1, 64]               │
│      └── module->forward() → 768-d float[]  │
└──────────────────┬──────────────────────────┘
                   │
┌──────────────────▼──────────────────────────┐
│  ExecuTorch Runtime                         │
│  ├── iOS:  CoreML delegate (GPU/ANE, FP16)  │
│  ├── Android: XNNPACK delegate (CPU, FP32)  │
│  └── Fallback: portable CPU ops             │
└─────────────────────────────────────────────┘
```

### ExecuTorch Build Flow

```
┌─────────────────────────────────────────────────────────┐
│  STEP 1: Model Export  (offline, developer machine)     │
│  scripts/export_ios_siglip.ipynb                        │
│                                                         │
│  Python → torch.export → ExecuTorch to_edge() →        │
│  CoreML / XNNPACK partitioner → .pte files              │
└──────────────────────────┬──────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────┐
│  STEP 2: ExecuTorch Binary Build  (CI or local)         │
│  scripts/build-executorch.sh ios|android                │
│                                                         │
│  Triggered by: git push v* tag → GitHub Actions         │
│  Or manually: ./scripts/build-executorch.sh all         │
│                                                         │
│  iOS   → dist/ios/*.xcframework                         │
│  Android → dist/android/libs/arm64-v8a/*.so             │
│                                                         │
│  Uploaded as GitHub Release assets (tar.gz)             │
└──────────────────────────┬──────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────┐
│  STEP 3: npm install postinstall                        │
│  scripts/setup-binaries.js                              │
│                                                         │
│  Downloads matching release tarball from GitHub →       │
│    dist/ios/    (xcframeworks)                          │
│    dist/android/ (JNI shared libs)                      │
│                                                         │
│  Skip with: SIGLIP_SKIP_BINARY_DOWNLOAD=1               │
└──────────────────────────┬──────────────────────────────┘
                           │
            ┌──────────────┴──────────────┐
            ▼                             ▼
┌───────────────────────┐    ┌────────────────────────────┐
│  iOS: pod install     │    │  Android: Gradle build     │
│  podspec links        │    │  CMakeLists.txt links      │
│  dist/ios/*.xcfmwk    │    │  dist/android/libs/*.so    │
│  as vendored fmwks    │    │  via target_link_libraries │
└───────────────────────┘    └────────────────────────────┘
```

## Example App

The `example/` directory contains a full working app demonstrating:

- Platform-aware model download and loading
- Image picking and vision embedding
- Text tokenization and text embedding
- Cosine similarity + SigLIP sigmoid scoring
- Numerical validation against Python reference embeddings
- Native memory usage tracking

```sh
cd example
npm install
# iOS
cd ios && pod install && cd ..
npx react-native run-ios
# Android
npx react-native run-android
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `SIGLIP_SKIP_BINARY_DOWNLOAD` | Set to `1` to skip postinstall binary download |
| `SIGLIP_BINARY_VERSION` | Override binary version (defaults to `package.json` version) |

## AI Model Disclaimer

This library provides a **runtime bridge** to run SigLIP 2 vision-language models on-device. Please be aware of the following:

**Model Ownership & License**
The SigLIP 2 model weights are developed by Google and released under the [Apache 2.0 License](https://huggingface.co/google/siglip2-base-patch16-224). This library does not redistribute model weights — you are responsible for complying with the model's license when embedding or distributing `.pte` files in your application.

**Output Reliability**
Embedding similarity scores are statistical estimates, not ground truth. The `computeSiglipProbability` output is empirically calibrated and may produce incorrect results on:
- Out-of-distribution images (medical, satellite, microscopy, etc.)
- Languages or scripts underrepresented in SigLIP's training data
- Adversarially crafted inputs

Do not use embedding similarity as the sole decision signal in safety-critical applications.

**Prohibited Uses**
Do not use this library to build systems that:
- Profile, surveil, or identify individuals without their explicit consent
- Make automated decisions with legal or similarly significant effects on people
- Discriminate based on protected characteristics

**No Warranty**
This library is provided as-is. Inference accuracy, latency, and memory usage will vary by device, OS version, and model.

## License

MIT — see [LICENSE](./LICENSE)
