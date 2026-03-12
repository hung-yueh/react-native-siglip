import type { SigLIP, SigLIPErrorCode, SigLIPBackend, SigLIPModelInfo } from "../src/SigLIP.nitro";

// ============================================================================
// Utility functions (same logic used in example app)
// ============================================================================

function l2Normalize(input: Float32Array): Float32Array {
  let sumSq = 0;
  for (let i = 0; i < input.length; i++) {
    sumSq += input[i] * input[i];
  }
  const norm = Math.sqrt(Math.max(sumSq, 1e-12));
  const output = new Float32Array(input.length);
  for (let i = 0; i < input.length; i++) {
    output[i] = input[i] / norm;
  }
  return output;
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

function siglipProbability(cosineSim: number): number {
  const t = 112.67;
  const b = -12.0;
  return 1 / (1 + Math.exp(-(cosineSim * t + b)));
}

// ============================================================================
// Contract Tests
// ============================================================================

describe("SigLIP type contract", () => {
  it("exposes all 8 error codes", () => {
    const codes: SigLIPErrorCode[] = [
      "ERR_MODEL_NOT_FOUND",
      "ERR_MODEL_LOAD",
      "ERR_MODEL_UNLOAD",
      "ERR_IMAGE_NOT_FOUND",
      "ERR_IMAGE_DECODE",
      "ERR_INFERENCE",
      "ERR_NOT_LOADED",
      "ERR_INVALID_ARGUMENT",
    ];
    expect(codes).toHaveLength(8);
  });

  it("exposes all 4 backends", () => {
    const backends: SigLIPBackend[] = ["coreml", "qnn", "xnnpack", "cpu"];
    expect(backends).toHaveLength(4);
  });

  it("SigLIPModelInfo has required fields", () => {
    const info: SigLIPModelInfo = {
      width: 224,
      height: 224,
      dimension: 768,
      backend: "coreml",
    };
    expect(info.width).toBe(224);
    expect(info.height).toBe(224);
    expect(info.dimension).toBe(768);
    expect(info.backend).toBe("coreml");
    expect(info.visionModelPath).toBeUndefined();
    expect(info.textModelPath).toBeUndefined();
  });
});

// ============================================================================
// Unit Tests — l2Normalize
// ============================================================================

describe("l2Normalize", () => {
  it("normalizes a vector to unit length", () => {
    const input = new Float32Array([3, 4]);
    const result = l2Normalize(input);
    expect(result[0]).toBeCloseTo(0.6, 5);
    expect(result[1]).toBeCloseTo(0.8, 5);
    // Verify unit-length
    const norm = Math.sqrt(result[0] ** 2 + result[1] ** 2);
    expect(norm).toBeCloseTo(1.0, 6);
  });

  it("handles zero vector gracefully", () => {
    const input = new Float32Array([0, 0, 0]);
    const result = l2Normalize(input);
    expect(result[0]).toBe(0);
    expect(result[1]).toBe(0);
    expect(result[2]).toBe(0);
  });

  it("is idempotent (normalizing twice yields same result)", () => {
    const input = new Float32Array([1, 2, 3, 4, 5]);
    const once = l2Normalize(input);
    const twice = l2Normalize(once);
    for (let i = 0; i < once.length; i++) {
      expect(twice[i]).toBeCloseTo(once[i], 6);
    }
  });
});

// ============================================================================
// Unit Tests — cosineSimilarity
// ============================================================================

describe("cosineSimilarity", () => {
  it("returns 1 for identical vectors", () => {
    const v = new Float32Array([1, 2, 3]);
    expect(cosineSimilarity(v, v)).toBeCloseTo(1.0, 6);
  });

  it("returns -1 for opposite vectors", () => {
    const a = new Float32Array([1, 0, 0]);
    const b = new Float32Array([-1, 0, 0]);
    expect(cosineSimilarity(a, b)).toBeCloseTo(-1.0, 6);
  });

  it("returns 0 for orthogonal vectors", () => {
    const a = new Float32Array([1, 0]);
    const b = new Float32Array([0, 1]);
    expect(cosineSimilarity(a, b)).toBeCloseTo(0.0, 6);
  });

  it("is scale-invariant", () => {
    const a = new Float32Array([1, 2, 3]);
    const b = new Float32Array([2, 4, 6]);
    expect(cosineSimilarity(a, b)).toBeCloseTo(1.0, 6);
  });
});

// ============================================================================
// Unit Tests — siglipProbability
// ============================================================================

describe("siglipProbability", () => {
  it("returns ~50% at cosine ≈ 0.107 (calibration point)", () => {
    // t * 0.1065 + b ≈ 0 → probability ≈ 0.5
    const sim = 12.0 / 112.67; // ≈ 0.1065
    const prob = siglipProbability(sim);
    expect(prob).toBeCloseTo(0.5, 1);
  });

  it("returns close to 0 for low cosine similarity", () => {
    const prob = siglipProbability(-0.1);
    expect(prob).toBeLessThan(0.001);
  });

  it("returns close to 1 for high cosine similarity", () => {
    const prob = siglipProbability(0.3);
    expect(prob).toBeGreaterThan(0.99);
  });

  it("is monotonically increasing", () => {
    const p1 = siglipProbability(0.0);
    const p2 = siglipProbability(0.1);
    const p3 = siglipProbability(0.2);
    expect(p2).toBeGreaterThan(p1);
    expect(p3).toBeGreaterThan(p2);
  });
});
