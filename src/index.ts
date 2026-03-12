import { NitroModules } from "react-native-nitro-modules";
import type { SigLIP as SigLIPSpec } from "./SigLIP.nitro";

export type {
  SigLIP,
  SigLIPBackend,
  SigLIPErrorCode,
  SigLIPModelInfo,
} from "./SigLIP.nitro";

/**
 * Create a SigLIP hybrid object instance.
 * This is the main entry point for the library.
 */
export function createSigLIP(): SigLIPSpec {
  return NitroModules.createHybridObject<SigLIPSpec>("SigLIP");
}
