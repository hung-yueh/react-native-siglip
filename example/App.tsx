import React, { useCallback, useState, useRef, useMemo } from "react";
import {
  Alert,
  Image,
  Platform,
  ScrollView,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
  ActivityIndicator,
  TextInput,
} from "react-native";
import { SafeAreaProvider, SafeAreaView } from "react-native-safe-area-context";
import * as FileSystem from "expo-file-system/legacy";
import * as ImagePicker from "expo-image-picker";
import * as Device from "expo-device";
import { createSigLIP, type SigLIP } from "react-native-siglip";
import { NitroModules } from "react-native-nitro-modules";
import { SiglipTokenizer } from "./SiglipTokenizer";

// First 100 values of reference embedding for white.png from validation.json
const WHITE_PNG_REF_100 = new Float32Array([
  0.017563581466674805, 0.20871686935424805, -0.12740153074264526,
  -0.2537783980369568, -0.06919944286346436, 0.12863582372665405,
  0.15293198823928833, -0.0619317889213562, -0.27928096055984497,
  0.3205338418483734, -0.055160343647003174, -0.08668568730354309,
  -0.4323456287384033, -0.11618003249168396, -0.43826204538345337,
  -0.44352516531944275, 0.12456744909286499, 0.31167691946029663,
  -0.0054749250411987305, -0.34170228242874146, -0.010716855525970459,
  -0.4029422998428345, -0.09554360061883926, -0.20511074364185333,
  0.17396926879882812, -0.2698221504688263, 0.31954947113990784,
  0.47582298517227173, 0.4412340223789215, -0.32154732942581177,
  -0.18164777755737305, 0.4322834014892578, 0.08667165040969849,
  -0.16782152652740479, 0.26011908054351807, 0.07817165553569794,
  -0.13670766353607178, -0.1700648069381714, -0.026773542165756226,
  -0.2040794938802719, -0.11522617936134338, 0.30947959423065186,
  -0.4429200291633606, -1.113905668258667, -0.27410706877708435,
  0.15848377346992493, -0.006439805030822754, 0.10707318782806396,
  -0.045997291803359985, 0.6839280128479004, 0.05995717644691467,
  -0.10001412034034729, 0.19491416215896606, -0.18063297867774963,
  -0.15058529376983643, -0.17560738325119019, 0.304866224527359,
  0.1449602246284485, -0.30919963121414185, 0.26652565598487854,
  0.30277132987976074, 0.3437943458557129, 0.34558096528053284,
  -0.22742950916290283, -0.3516165018081665, -0.8827477693557739,
  -0.4904223680496216, -0.054651618003845215, -0.10404680669307709,
  0.22923976182937622, 0.23035728931427002, 0.8774290084838867,
  0.10046696662902832, 0.5314559936523438, -0.37898939847946167,
  -0.2370007187128067, -0.27904218435287476, 0.032147735357284546,
  -0.14864009618759155, -0.07219117879867554, 0.035808831453323364,
  0.3175852298736572, -0.13566327095031738, -0.12792593240737915,
  0.3154521584510803, 0.35030055046081543, -0.12746387720108032,
  -0.14578884840011597, -0.20537149906158447, 0.016007155179977417,
  0.10554349422454834, -0.01573079824447632, -0.030004918575286865,
  -0.022910863161087036, -0.1346445083618164, -0.017751097679138184,
  -0.28903496265411377, -0.3197869062423706, 0.28211718797683716,
  0.28904688358306885,
]);

console.log("[App] Loading App.tsx...");

// Platform-specific model configuration
// iOS: CoreML backend (.pte with CoreML delegate, FP16, GPU)
// Android: XNNPACK backend (.pte with XNNPACK delegate, FP32, CPU)
const MODEL_CONFIG = Platform.select({
  ios: {
    visionFilename: "siglip2_vision_coreml_fp16.pte",
    visionUrl: "https://litert.dev/siglip2_vision_coreml_fp16.pte",
    textFilename: "siglip2_text_coreml_fp16.pte",
    textUrl: "https://litert.dev/siglip2_text_coreml_fp16.pte",
    label: "CoreML FP16 (GPU)",
  },
  android: {
    visionFilename: "siglip2_vision_xnnpack_fp32.pte",
    visionUrl: "https://litert.dev/siglip2_vision_xnnpack_fp32.pte",
    textFilename: "siglip2_text_xnnpack_fp32.pte",
    textUrl: "https://litert.dev/siglip2_text_xnnpack_fp32.pte",
    label: "XNNPACK FP32 (CPU)",
  },
  default: {
    visionFilename: "siglip2_vision_xnnpack_fp32.pte",
    visionUrl: "https://litert.dev/siglip2_vision_xnnpack_fp32.pte",
    textFilename: "siglip2_text_xnnpack_fp32.pte",
    textUrl: "https://litert.dev/siglip2_text_xnnpack_fp32.pte",
    label: "XNNPACK FP32 (CPU)",
  },
})!;

const THEME = {
  bg: "#050505",
  card: "#121212",
  accent: "#3B82F6", // Blue
  success: "#10B981", // Emerald
  warning: "#F59E0B", // Amber
  error: "#EF4444", // Red
  text: "#F9FAF7",
  textDim: "#9CA3AF",
  border: "#262626",
};

const getModelDirectory = () => FileSystem.documentDirectory + "siglip_models/";
const getVisionModelPath = () =>
  getModelDirectory() + MODEL_CONFIG.visionFilename;
const getTextModelPath = () =>
  getModelDirectory() + MODEL_CONFIG.textFilename;

function cosineSimilarity(a: Float32Array, b: Float32Array): number {
  let dotProduct = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < Math.min(a.length, b.length); i++) {
    dotProduct += a[i] * b[i];
    normA += Math.pow(a[i], 2);
    normB += Math.pow(b[i], 2);
  }
  return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
}

export function l2Normalize(input: Float32Array): Float32Array {
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

// SigLIP models logit scale transformation
function computeSiglipProbability(cosineSim: number): number {
  // SigLIP uses a learned temperature scale (t) and bias (b) for sigmoid scoring.
  // logit_scale = exp(4.7245) = 112.67 from google/siglip2-base-patch16-224
  // logit_bias = -16.77 (model-trained value) — but this was calibrated for
  // large-batch contrastive training where cosine sims are ~0.20+.
  // At inference with raw pooler_output, cosines are ~0.05–0.15, so we
  // use an empirically adjusted bias that gives 50% at cos≈0.107.
  const t = 112.67;
  const b = -12.0; // adjusted: empirically calibrated for pooler_output scale

  const logit = cosineSim * t + b;
  return 1 / (1 + Math.exp(-logit));
}

type LogEntry = {
  timestamp: number;
  message: string;
  type: "info" | "success" | "error";
};

type MemoryUsage = {
  visionModelSize: number; // bytes on disk
  textModelSize: number;
  visionEmbeddingSize: number; // native ArrayBuffer byteLength
  textTokenInputSize: number;
  textEmbeddingSize: number;
};

const EMPTY_MEMORY: MemoryUsage = {
  visionModelSize: 0,
  textModelSize: 0,
  visionEmbeddingSize: 0,
  textTokenInputSize: 0,
  textEmbeddingSize: 0,
};

function formatBytes(bytes: number): string {
  if (bytes === 0) return "0 B";
  const units = ["B", "KB", "MB", "GB"];
  const i = Math.floor(Math.log(bytes) / Math.log(1024));
  const value = bytes / Math.pow(1024, i);
  return `${value.toFixed(i > 1 ? 1 : 0)} ${units[i]}`;
}

export default function App() {
  return (
    <SafeAreaProvider>
      <Main />
    </SafeAreaProvider>
  );
}

function Main() {
  const [visionLoaded, setVisionLoaded] = useState(false);
  const [textLoaded, setTextLoaded] = useState(false);
  const [visionModelPath, setVisionModelPath] = useState<string | null>(null);
  const [textModelPath, setTextModelPath] = useState<string | null>(null);
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [textPrompt, setTextPrompt] = useState<string>("");
  const [visionEmbedding, setVisionEmbedding] = useState<Float32Array | null>(
    null,
  );
  const [textEmbedding, setTextEmbedding] = useState<Float32Array | null>(null);
  const [similarity, setSimilarity] = useState<number | null>(null);
  const [latency, setLatency] = useState<number | null>(null);
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [busy, setBusy] = useState(false);
  const [memory, setMemory] = useState<MemoryUsage>(EMPTY_MEMORY);

  const siglipRef = useRef<SigLIP | null>(null);
  const tokenizerRef = useRef<SiglipTokenizer | null>(null);

  const log = useCallback(
    (message: string, type: LogEntry["type"] = "info") => {
      console.log(`[App] ${message}`);
      setLogs((prev) => [
        { timestamp: Date.now(), message, type },
        ...prev.slice(0, 49),
      ]);
    },
    [],
  );

  const getSigLIP = useCallback(() => {
    if (siglipRef.current) return siglipRef.current;
    try {
      siglipRef.current = createSigLIP();
      return siglipRef.current;
    } catch (e: any) {
      log(`SigLIP init failed: ${e.message}`, "error");
      return null;
    }
  }, [log]);

  const setupModels = useCallback(async () => {
    const destDir = getModelDirectory();
    const visPath = getVisionModelPath();
    const txtPath = getTextModelPath();
    try {
      await FileSystem.makeDirectoryAsync(destDir, { intermediates: true });
      const visInfo = await FileSystem.getInfoAsync(visPath);
      const txtInfo = await FileSystem.getInfoAsync(txtPath);

      let paths = {
        vision: null as string | null,
        text: null as string | null,
      };
      if (visInfo.exists) {
        setVisionModelPath(visPath);
        paths.vision = visPath;
      }
      if (txtInfo.exists) {
        setTextModelPath(txtPath);
        paths.text = txtPath;
      }
      return paths;
    } catch (error: any) {
      log(`Setup failed: ${error.message}`, "error");
      return { vision: null, text: null };
    }
  }, [log]);

  const loadModels = useCallback(
    async (visPathOverride?: string, txtPathOverride?: string) => {
      const vPath = visPathOverride || visionModelPath;
      const tPath = txtPathOverride || textModelPath;
      const instance = getSigLIP();
      if (!instance) return;

      setBusy(true);
      try {
        if (vPath && !visionLoaded) {
          log("Loading Vision Model...");
          const vInfo = await FileSystem.getInfoAsync(vPath);
          await instance.loadVisionModel(vPath);
          setVisionLoaded(true);
          const vSize = (vInfo as any).size ?? 0;
          setMemory((prev) => ({ ...prev, visionModelSize: vSize }));
          log(`Vision Model Loaded (${formatBytes(vSize)}).`, "success");
        }
        if (tPath && !textLoaded) {
          log("Loading Text Model...");
          const tInfo = await FileSystem.getInfoAsync(tPath);
          await instance.loadTextModel(tPath);
          setTextLoaded(true);
          const tSize = (tInfo as any).size ?? 0;
          setMemory((prev) => ({ ...prev, textModelSize: tSize }));
          log(`Text Model Loaded (${formatBytes(tSize)}).`, "success");
        }
      } catch (error: any) {
        log(`Load failed: ${error.message}`, "error");
      } finally {
        setBusy(false);
      }
    },
    [visionModelPath, textModelPath, visionLoaded, textLoaded, log, getSigLIP],
  );

  const downloadModels = useCallback(async () => {
    const visPath = getVisionModelPath();
    const txtPath = getTextModelPath();
    setBusy(true);
    log(`Downloading ${MODEL_CONFIG.label} models from litert.dev...`);
    try {
      if (!(await FileSystem.getInfoAsync(visPath)).exists) {
        log(`Downloading vision model: ${MODEL_CONFIG.visionFilename}...`);
        await FileSystem.downloadAsync(MODEL_CONFIG.visionUrl, visPath);
        setVisionModelPath(visPath);
      }
      if (!(await FileSystem.getInfoAsync(txtPath)).exists) {
        log(`Downloading text model: ${MODEL_CONFIG.textFilename}...`);
        await FileSystem.downloadAsync(MODEL_CONFIG.textUrl, txtPath);
        setTextModelPath(txtPath);
      }
      log(`Downloads complete (${MODEL_CONFIG.label}).`, "success");
      return { vision: visPath, text: txtPath };
    } catch (error: any) {
      log(`Download failed: ${error.message}`, "error");
      return null;
    } finally {
      setBusy(false);
    }
  }, [log]);

  const refreshModels = useCallback(async () => {
    log("Scanning for model files...");
    const paths = await setupModels();
    if (paths.vision || paths.text) {
      await loadModels(paths.vision || undefined, paths.text || undefined);
    } else {
      Alert.alert(
        "Models Not Found",
        `Please download the models or push them to: \n${getModelDirectory()}`,
      );
    }
  }, [setupModels, loadModels, log]);

  React.useEffect(() => {
    (async () => {
      const paths = await setupModels();
      await loadModels(paths.vision || undefined, paths.text || undefined);
      // Load white.png as the default reference image
      try {
        const src = require("./assets/white.png");
        // eslint-disable-next-line @typescript-eslint/no-unsafe-argument
        const resolved = Image.resolveAssetSource(src);
        if (resolved?.uri) setSelectedImage(resolved.uri);
      } catch (e) {
        // white.png not found — user will pick manually
      }
    })();
  }, [setupModels, loadModels]);

  const runValidationTest = useCallback(async () => {
    const instance = getSigLIP();
    if (!instance) {
      log("Models not loaded", "error");
      return;
    }

    setBusy(true);
    log("[Validation] Loading white.png reference image...");
    try {
      const src = require("./assets/white.png");
      // eslint-disable-next-line @typescript-eslint/no-unsafe-argument
      const resolved = Image.resolveAssetSource(src);
      if (!resolved?.uri) throw new Error("Could not resolve white.png URI");

      const whiteLocalPath =
        FileSystem.cacheDirectory + "white_verification.png";
      log("[Validation] Caching asset to: " + whiteLocalPath);
      await FileSystem.downloadAsync(resolved.uri, whiteLocalPath);

      log("[Validation] Running vision inference...");
      const vResult = await instance.getImageEmbedding(whiteLocalPath);
      let vRaw = new Float32Array(vResult as any);

      // We discovered the pooler_output from Hugging Face is already
      // scaled by the attention head. A manual LayerNorm here corrupts it!
      // (Removed layerNorm call)

      // Compare raw first 100 values (now post-LayerNorm) directly to high precision reference
      const refRaw = WHITE_PNG_REF_100;
      const sim = cosineSimilarity(vRaw.slice(0, 100), refRaw);
      const pass = sim > 0.95;

      console.log(
        "[Validation] iOS raw first 10:",
        Array.from(vRaw.slice(0, 10)),
      );
      console.log(
        "[Validation] Ref raw first 10:",
        Array.from(refRaw.slice(0, 10)),
      );
      console.log(
        `[Validation] Raw cosine similarity vs reference: ${sim.toFixed(6)}`,
      );

      log(
        `[Validation] Raw cosine vs reference: ${sim.toFixed(4)} — ${pass ? "✅ PASS" : "❌ FAIL (expected > 0.95)"}`,
        pass ? "success" : "error",
      );

      setSelectedImage(whiteLocalPath);
      setVisionEmbedding(l2Normalize(vRaw) as any);
    } catch (e: any) {
      log(`[Validation] Failed: ${e.message}`, "error");
    } finally {
      setBusy(false);
    }
  }, [getSigLIP, log]);

  const pickImage = useCallback(async () => {
    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ["images"],
      quality: 1,
    });
    if (!result.canceled && result.assets[0]) {
      const uri = result.assets[0].uri;
      setSelectedImage(uri);
      setVisionEmbedding(null);
      setSimilarity(null);
      setLatency(null);
    }
  }, []);

  const runInference = useCallback(async () => {
    const instance = getSigLIP();
    if (!instance || !selectedImage) return;

    setBusy(true);
    try {
      const start = performance.now();

      // 1. Get Vision Embedding
      setSimilarity(null); // Clear stale match UI
      log("Generating vision embedding...");
      const vResult = await instance.getImageEmbedding(selectedImage);
      const visionBufSize = (vResult as ArrayBuffer).byteLength;
      setMemory((prev) => ({ ...prev, visionEmbeddingSize: visionBufSize }));
      log(`Vision native buffer: ${formatBytes(visionBufSize)}`);
      let vRaw = new Float32Array(vResult as any);

      // Final L2 Normalization
      let vArray = l2Normalize(vRaw);

      console.log(
        `[Native ArrayBuffer] First 10 Vision Embeddings:`,
        Array.from(vArray.slice(0, 10)),
      );
      setVisionEmbedding(vArray as any);

      let tArray: Float32Array | null = null;

      // 2. Get Text Embedding (if prompt exists and tokenizer is ready)
      if (textPrompt.trim() !== "") {
        if (!tokenizerRef.current) {
          log("Instantiating SigLIP 2 SentencePiece Tokenizer...", "info");
          tokenizerRef.current = new SiglipTokenizer();
        }
        log("Tokenizing prompt...");
        // SigLIP2 was trained with classification prompt template.
        // Wrap user input in "a photo of a {label}" for best zero-shot similarity.
        const rawLabel = textPrompt.trim();
        const formattedPrompt = rawLabel.toLowerCase().startsWith("a ")
          ? rawLabel.toLowerCase() // user typed a full sentence — lowercase it
          : `a photo of a ${rawLabel.toLowerCase()}`;
        console.log(
          `[Prompt] Raw: "${rawLabel}" → Formatted: "${formattedPrompt}"`,
        );
        const tokenIds = tokenizerRef.current.encode(formattedPrompt);

        console.log(`[Token Debug] First 15 tokens:`, tokenIds.slice(0, 15));

        // ... inside runInference ...
        log(
          `Generating text embedding for ${tokenIds.length} tokens (Zero-Copy Native Buffer)...`,
        );

        // Allocate raw memory for tokens: Int64 = 8 bytes per token
        const byteLength = tokenIds.length * 8;

        // Create a BigInt64Array view over the raw memory
        // @ts-ignore: New Nitro feature in v0.34.0, types might be lagging locally
        const nativeBuffer = NitroModules.createNativeArrayBuffer(byteLength);
        setMemory((prev) => ({ ...prev, textTokenInputSize: nativeBuffer.byteLength }));
        log(`Token input native buffer: ${formatBytes(nativeBuffer.byteLength)}`);

        // Create a BigInt64Array view over the raw memory
        const tokenView = new BigInt64Array(nativeBuffer);
        for (let i = 0; i < tokenIds.length; i++) {
          tokenView[i] = BigInt(tokenIds[i]);
        }

        const tResult = await instance.getTextEmbedding(nativeBuffer);
        const textBufSize = (tResult as ArrayBuffer).byteLength;
        setMemory((prev) => ({ ...prev, textEmbeddingSize: textBufSize }));
        log(`Text embedding native buffer: ${formatBytes(textBufSize)}`);
        tArray = new Float32Array(tResult);

        // Text Model is exported with LayerNorm intact, only standard L2 is needed
        tArray = l2Normalize(tArray);

        console.log(
          `[Native ArrayBuffer] First 10 Text Embeddings:`,
          Array.from(tArray.slice(0, 10)),
        );
        setTextEmbedding(tArray as any);

        // 3. Calculate Similarity Match
        const rawSim = cosineSimilarity(vArray, tArray);
        const prob = computeSiglipProbability(rawSim);
        setSimilarity(prob);
        log(
          `Match Probability: ${(prob * 100).toFixed(1)}% (raw: ${rawSim.toFixed(3)})`,
          "success",
        );
      }

      const elapsed = performance.now() - start;
      setLatency(elapsed);
      log(`Total Inference successful: ${elapsed.toFixed(0)}ms`, "success");
    } catch (error: any) {
      log(`Inference failed: ${error.message}`, "error");
      Alert.alert("Inference Error", error.message);
    } finally {
      setBusy(false);
    }
  }, [selectedImage, textPrompt, log, getSigLIP]);

  const backend = siglipRef.current?.modelInfo?.backend || "none";

  return (
    <SafeAreaView style={styles.container}>
      <ScrollView contentContainerStyle={styles.scrollContent}>
        <Header />

        <View style={styles.grid}>
          <DiagnosticCard
            label="Backend"
            value={backend.toUpperCase()}
            icon="⚡"
            color={backend === "cpu" ? THEME.warning : THEME.success}
          />
          <DiagnosticCard
            label="Latency"
            value={latency ? `${latency.toFixed(0)}ms` : "--"}
            icon="⏱️"
          />
        </View>

        {similarity !== null && (
          <View style={[styles.grid, { marginTop: 0 }]}>
            <DiagnosticCard
              label="Match Similarity"
              value={`${(similarity * 100).toFixed(1)}%`}
              icon="🎯"
              color={similarity > 0.2 ? THEME.success : THEME.textDim}
            />
          </View>
        )}

        <Section title="Device Compatibility">
          <InfoRow
            label="Device"
            value={`${Device.brand} ${Device.modelName}`}
          />
          <InfoRow
            label="Arch"
            value={Device.supportedCpuArchitectures?.[0] ?? "unknown"}
          />
          <InfoRow
            label="NPU"
            value={Platform.OS === "ios" ? "ANE (CoreML)" : "XNNPACK (CPU)"}
          />
          <InfoRow
            label="Model Format"
            value={MODEL_CONFIG.label}
          />
        </Section>

        <Section title="SigLIP Models">
          {!visionModelPath || !textModelPath ? (
            <View style={styles.warningBox}>
              <Text style={styles.warningText}>Models Missing</Text>
              <Text style={styles.instructionText}>
                Download SigLIP 2 Vision & Text or push manually to models/
                directory.
              </Text>
              <View style={styles.buttonGroup}>
                <TouchableOpacity
                  style={styles.primaryButton}
                  onPress={downloadModels}
                  disabled={busy}
                >
                  {busy ? (
                    <ActivityIndicator color="#fff" />
                  ) : (
                    <Text style={styles.buttonText}>{`Download ${MODEL_CONFIG.label}`}</Text>
                  )}
                </TouchableOpacity>
                <TouchableOpacity
                  style={styles.secondaryButton}
                  onPress={refreshModels}
                  disabled={busy}
                >
                  <Text style={styles.buttonText}>Scan</Text>
                </TouchableOpacity>
              </View>
            </View>
          ) : !visionLoaded || !textLoaded ? (
            <TouchableOpacity
              style={styles.primaryButton}
              onPress={() => loadModels()}
              disabled={busy}
            >
              {busy ? (
                <ActivityIndicator color="#fff" />
              ) : (
                <Text style={styles.buttonText}>Load Models</Text>
              )}
            </TouchableOpacity>
          ) : (
            <View style={styles.successBox}>
              <Text style={styles.successText}>✓ Both Models Ready (768d)</Text>
            </View>
          )}
        </Section>

        <Section title="Inference Test">
          <View style={styles.imageContainer}>
            {selectedImage ? (
              <Image source={{ uri: selectedImage }} style={styles.image} />
            ) : (
              <View style={styles.imagePlaceholder}>
                <Text style={styles.placeholderText}>No Image Selected</Text>
              </View>
            )}
          </View>
          {selectedImage && (
            <Text
              style={[
                styles.infoLabel,
                { marginTop: 8, fontSize: 12, textAlign: "center" },
              ]}
            >
              Loaded Image: {selectedImage.split("/").pop()}
            </Text>
          )}

          <View style={{ marginTop: 12 }}>
            <TextInput
              style={styles.input}
              placeholder="Optional: Enter a text prompt to match"
              placeholderTextColor={THEME.textDim}
              value={textPrompt}
              onChangeText={setTextPrompt}
              editable={!busy}
            />
          </View>

          <View style={styles.buttonGroup}>
            <TouchableOpacity
              style={styles.secondaryButton}
              onPress={pickImage}
            >
              <Text style={styles.buttonText}>Pick Image</Text>
            </TouchableOpacity>
            <TouchableOpacity
              style={[
                styles.secondaryButton,
                { backgroundColor: "#7C3AED" },
                !visionLoaded && styles.disabledButton,
              ]}
              onPress={runValidationTest}
              disabled={busy || !visionLoaded}
            >
              <Text style={styles.buttonText}>🔬 Validate</Text>
            </TouchableOpacity>
            <TouchableOpacity
              style={[
                styles.primaryButton,
                (!visionLoaded || !selectedImage) && styles.disabledButton,
              ]}
              onPress={runInference}
              disabled={busy || !visionLoaded || !selectedImage}
            >
              {busy ? (
                <ActivityIndicator color="#fff" />
              ) : (
                <Text style={styles.buttonText}>Run</Text>
              )}
            </TouchableOpacity>
          </View>
        </Section>

        {visionEmbedding && (
          <Section title="Vision Tensor Output (768d)">
            <Text style={styles.monoText}>
              [
              {Array.from(visionEmbedding.slice(0, 6))
                .map((v) => v.toFixed(4))
                .join(", ")}
              , ...]
            </Text>
          </Section>
        )}

        {textEmbedding && (
          <Section title="Text Tensor Output (768d)">
            <Text style={styles.monoText}>
              [
              {Array.from(textEmbedding.slice(0, 6))
                .map((v) => v.toFixed(4))
                .join(", ")}
              , ...]
            </Text>
          </Section>
        )}

        {(memory.visionModelSize > 0 || memory.visionEmbeddingSize > 0) && (
          <Section title="Native Memory Usage">
            <View style={styles.memoryContainer}>
              {memory.visionModelSize > 0 && (
                <>
                  <Text style={styles.memoryHeader}>Model Loading</Text>
                  <InfoRow
                    label="Vision Model"
                    value={formatBytes(memory.visionModelSize)}
                  />
                  {memory.textModelSize > 0 && (
                    <InfoRow
                      label="Text Model"
                      value={formatBytes(memory.textModelSize)}
                    />
                  )}
                  <InfoRow
                    label="Total Models"
                    value={formatBytes(
                      memory.visionModelSize + memory.textModelSize,
                    )}
                  />
                </>
              )}
              {memory.visionEmbeddingSize > 0 && (
                <>
                  <Text style={[styles.memoryHeader, { marginTop: 12 }]}>
                    Inference Buffers
                  </Text>
                  <InfoRow
                    label="Vision Embedding"
                    value={formatBytes(memory.visionEmbeddingSize)}
                  />
                  {memory.textTokenInputSize > 0 && (
                    <InfoRow
                      label="Text Token Input"
                      value={formatBytes(memory.textTokenInputSize)}
                    />
                  )}
                  {memory.textEmbeddingSize > 0 && (
                    <InfoRow
                      label="Text Embedding"
                      value={formatBytes(memory.textEmbeddingSize)}
                    />
                  )}
                  <InfoRow
                    label="Total Buffers"
                    value={formatBytes(
                      memory.visionEmbeddingSize +
                        memory.textTokenInputSize +
                        memory.textEmbeddingSize,
                    )}
                  />
                </>
              )}
              <View style={styles.memoryTotalRow}>
                <Text style={styles.memoryTotalLabel}>Total Native Memory</Text>
                <Text style={styles.memoryTotalValue}>
                  {formatBytes(
                    memory.visionModelSize +
                      memory.textModelSize +
                      memory.visionEmbeddingSize +
                      memory.textTokenInputSize +
                      memory.textEmbeddingSize,
                  )}
                </Text>
              </View>
            </View>
          </Section>
        )}

        <Section title="System Logs">
          {logs.map((log, i) => (
            <Text
              key={i}
              style={[
                styles.logText,
                log.type === "error" && { color: THEME.error },
                log.type === "success" && { color: THEME.success },
              ]}
            >
              • {log.message}
            </Text>
          ))}
        </Section>
      </ScrollView>
    </SafeAreaView>
  );
}

function Header() {
  return (
    <View style={styles.header}>
      <Text style={styles.title}>
        SigLIP <Text style={{ color: THEME.accent }}>Vision</Text>
      </Text>
      <Text style={styles.subtitle}>On-Device AI Embedding Engine</Text>
    </View>
  );
}

function DiagnosticCard({ label, value, icon, color }: any) {
  return (
    <View style={styles.card}>
      <Text style={styles.cardIcon}>{icon}</Text>
      <Text style={styles.cardLabel}>{label}</Text>
      <Text style={[styles.cardValue, color && { color }]}>{value}</Text>
    </View>
  );
}

function Section({ title, children }: any) {
  return (
    <View style={styles.section}>
      <Text style={styles.sectionTitle}>{title}</Text>
      {children}
    </View>
  );
}

function InfoRow({ label, value }: any) {
  return (
    <View style={styles.infoRow}>
      <Text style={styles.infoLabel}>{label}</Text>
      <Text style={styles.infoValue}>{value}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: THEME.bg },
  scrollContent: { padding: 20 },
  header: { marginBottom: 24, marginTop: 10 },
  title: {
    fontSize: 32,
    fontWeight: "900",
    color: THEME.text,
    letterSpacing: -1,
  },
  subtitle: { fontSize: 14, color: THEME.textDim, fontWeight: "500" },
  grid: { flexDirection: "row", gap: 12, marginBottom: 20 },
  card: {
    flex: 1,
    backgroundColor: THEME.card,
    borderRadius: 16,
    padding: 16,
    borderWidth: 1,
    borderColor: THEME.border,
  },
  cardIcon: { fontSize: 20, marginBottom: 8 },
  cardLabel: {
    fontSize: 12,
    color: THEME.textDim,
    fontWeight: "600",
    textTransform: "uppercase",
  },
  cardValue: {
    fontSize: 20,
    fontWeight: "800",
    color: THEME.text,
    marginTop: 2,
  },
  section: { marginBottom: 24 },
  sectionTitle: {
    fontSize: 13,
    fontWeight: "700",
    color: THEME.textDim,
    textTransform: "uppercase",
    letterSpacing: 1,
    marginBottom: 12,
  },
  infoRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    marginBottom: 8,
    paddingBottom: 8,
    borderBottomWidth: 1,
    borderBottomColor: THEME.border,
  },
  infoLabel: { fontSize: 14, color: THEME.textDim },
  infoValue: { fontSize: 14, color: THEME.text, fontWeight: "600" },
  primaryButton: {
    backgroundColor: THEME.accent,
    paddingVertical: 14,
    borderRadius: 12,
    alignItems: "center",
    minHeight: 48,
    minWidth: 80,
  },
  secondaryButton: {
    backgroundColor: THEME.card,
    paddingVertical: 14,
    borderRadius: 12,
    alignItems: "center",
    flex: 1,
    borderWidth: 1,
    borderColor: THEME.border,
  },
  buttonGroup: { flexDirection: "row", gap: 12, marginTop: 16 },
  buttonText: {
    color: "#fff",
    fontWeight: "700",
    fontSize: 15,
    textAlign: "center",
  },
  imageContainer: {
    borderRadius: 16,
    overflow: "hidden",
    backgroundColor: THEME.card,
    borderWidth: 1,
    borderColor: THEME.border,
  },
  image: { width: "100%", height: 240, resizeMode: "cover" },
  imagePlaceholder: {
    width: "100%",
    height: 240,
    justifyContent: "center",
    alignItems: "center",
  },
  placeholderText: { color: THEME.textDim, fontWeight: "600" },
  disabledButton: { opacity: 0.5 },
  successBox: {
    backgroundColor: "rgba(16, 185, 129, 0.1)",
    padding: 14,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: THEME.success,
  },
  successText: { color: THEME.success, fontWeight: "700", textAlign: "center" },
  monoText: {
    fontFamily: Platform.OS === "ios" ? "Menlo" : "monospace",
    color: THEME.accent,
    fontSize: 12,
    backgroundColor: THEME.card,
    padding: 12,
    borderRadius: 8,
  },
  logText: {
    fontSize: 12,
    color: THEME.textDim,
    fontFamily: Platform.OS === "ios" ? "Menlo" : "monospace",
    marginBottom: 4,
  },
  warningBox: {
    backgroundColor: "rgba(245, 158, 11, 0.1)",
    padding: 16,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: THEME.warning,
    gap: 10,
  },
  warningText: {
    color: THEME.warning,
    fontWeight: "800",
    fontSize: 16,
    textAlign: "center",
  },
  instructionText: {
    color: THEME.textDim,
    fontSize: 13,
    textAlign: "center",
    lineHeight: 18,
  },
  pathText: {
    color: THEME.accent,
    fontSize: 11,
    fontFamily: Platform.OS === "ios" ? "Menlo" : "monospace",
    backgroundColor: "rgba(0,0,0,0.3)",
    padding: 8,
    borderRadius: 6,
    textAlign: "center",
  },
  input: {
    backgroundColor: THEME.card,
    borderWidth: 1,
    borderColor: THEME.border,
    borderRadius: 12,
    color: THEME.text,
    padding: 14,
    marginTop: 12,
    fontSize: 15,
  },
  memoryContainer: {
    backgroundColor: THEME.card,
    borderRadius: 12,
    padding: 14,
    borderWidth: 1,
    borderColor: THEME.border,
  },
  memoryHeader: {
    fontSize: 12,
    fontWeight: "700",
    color: THEME.accent,
    textTransform: "uppercase",
    letterSpacing: 0.5,
    marginBottom: 8,
  },
  memoryTotalRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    marginTop: 12,
    paddingTop: 10,
    borderTopWidth: 1,
    borderTopColor: THEME.accent,
  },
  memoryTotalLabel: {
    fontSize: 14,
    color: THEME.text,
    fontWeight: "800",
  },
  memoryTotalValue: {
    fontSize: 14,
    color: THEME.accent,
    fontWeight: "800",
    fontFamily: Platform.OS === "ios" ? "Menlo" : "monospace",
  },
});
