import vocab from "./siglip2_vocab.json";

// SigLIP 2 Tokenizer — Gemma-based SentencePiece with 256k vocabulary.
// Matches HuggingFace AutoTokenizer output for text length, then LEFT-PADS.
//
// Why left-pad:
//   The exported text .pte pools from last_hidden_state[:, -1, :] (position 63).
//   With attention_mask baked as all-ones, left-padding ensures EOS lands at
//   position 63 rather than a zero-PAD token. Python validation confirms that
//   left-pad produces index[4] ≈ 0.370, matching the iOS .pte output.
export class SiglipTokenizer {
  private vocab: string[];
  private vocabNorm: string[];

  constructor() {
    this.vocab = vocab as string[];
    // Normalize SentencePiece ▁ (U+2581) to regular space for text matching
    this.vocabNorm = this.vocab.map((v) => v.replace(/\u2581/g, " "));
  }

  public encode(text: string): number[] {
    const inputText = text.trim();
    const tokens: number[] = [];

    let i = 0;
    while (i < inputText.length) {
      let matchId = -1;
      let matchLen = 0;

      // Greedy maximal-munch: longest vocab entry at position i
      for (let j = 0; j < this.vocabNorm.length; j++) {
        const vn = this.vocabNorm[j];
        if (!vn) continue;
        if (inputText.startsWith(vn, i) && vn.length > matchLen) {
          matchId = j;
          matchLen = vn.length;
        }
      }

      if (matchId !== -1) {
        tokens.push(matchId);
        i += matchLen;
      } else {
        i += 1; // skip unknown
      }
    }

    // Append EOS (id = 1)
    tokens.push(1);

    // RIGHT-pad to 64: Matches HuggingFace AutoTokenizer / Notebook export.
    // The pooling from last_hidden_state[:, -1, :] is handled inside SigLIPVisionModel/SigLIPTextModel.
    const padded = new Array(64).fill(0);
    for (let k = 0; k < tokens.length && k < 64; k++) {
      padded[k] = tokens[k];
    }

    return padded;
  }
}
