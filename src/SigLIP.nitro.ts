import type { HybridObject } from "react-native-nitro-modules";

export type SigLIPBackend = "coreml" | "qnn" | "xnnpack" | "cpu";

export type SigLIPErrorCode =
  | "ERR_MODEL_NOT_FOUND"
  | "ERR_MODEL_LOAD"
  | "ERR_MODEL_UNLOAD"
  | "ERR_IMAGE_NOT_FOUND"
  | "ERR_IMAGE_DECODE"
  | "ERR_INFERENCE"
  | "ERR_NOT_LOADED"
  | "ERR_INVALID_ARGUMENT";

export interface SigLIPModelInfo {
  width: number;
  height: number;
  dimension: number;
  backend: SigLIPBackend;
  visionModelPath?: string;
  textModelPath?: string;
}

/**
 * React Native Nitro interface for SigLIP image embeddings.
 *
 * Error handling contract:
 * Promise rejections should include one of SigLIPErrorCode values in `code`.
 */
export interface SigLIP extends HybridObject<{ ios: "c++"; android: "c++" }> {
  /**
   * Loads a .pte vision model from an absolute local filesystem path.
   * Accepts either an absolute path or a file:// URI.
   */
  loadVisionModel(path: string): Promise<void>;

  /**
   * Loads a .pte text model from an absolute local filesystem path.
   * Accepts either an absolute path or a file:// URI.
   */
  loadTextModel(path: string): Promise<void>;

  /**
   * Runs image embedding inference for a local image file.
   * Accepts either an absolute path or a file:// URI.
   */
  getImageEmbedding(imageUri: string): Promise<ArrayBuffer>;

  /**
   * Runs text embedding inference for an array of tokens.
   */
  getTextEmbedding(tokens: ArrayBuffer): Promise<ArrayBuffer>;

  /**
   * Unloads the active models and frees runtime resources.
   */
  unloadModels(): void;

  /**
   * Current model metadata. Null when no model is loaded.
   */
  readonly modelInfo: SigLIPModelInfo | null;
}
