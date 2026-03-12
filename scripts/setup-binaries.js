#!/usr/bin/env node
/**
 * setup-binaries.js - Download pre-compiled ExecuTorch binaries
 *
 * This postinstall script downloads ExecuTorch xcframework (iOS) and AAR (Android)
 * from GitHub Releases to enable building react-native-siglip.
 *
 * Usage:
 *   npm install  # Runs automatically as postinstall
 *   node scripts/setup-binaries.js  # Manual run
 *
 * Environment Variables:
 *   SIGLIP_SKIP_BINARY_DOWNLOAD - Set to "1" to skip download
 *   SIGLIP_BINARY_VERSION - Override binary version (default: matches package version)
 */

const fs = require("fs");
const https = require("https");
const path = require("path");
const { execSync } = require("child_process");

// ============================================================================
// Configuration
// ============================================================================

const GITHUB_REPO = "hugh/react-native-siglip";
const BINARY_VERSION = process.env.SIGLIP_BINARY_VERSION || getPackageVersion();

const TARGETS = {
  ios: {
    artifact: "executorch-ios.tar.gz",
    destDir: "dist/ios",
    platforms: ["darwin"],
  },
  android: {
    artifact: "executorch-android.tar.gz",
    destDir: "dist/android",
    platforms: ["darwin", "linux", "win32"],
  },
};

// ============================================================================
// Helpers
// ============================================================================

function getPackageVersion() {
  try {
    const pkg = require("../package.json");
    return `v${pkg.version}`;
  } catch {
    return "v0.1.0";
  }
}

function log(message) {
  console.log(`[react-native-siglip] ${message}`);
}

function logError(message) {
  console.error(`[react-native-siglip] ERROR: ${message}`);
}

function shouldSkipDownload() {
  return process.env.SIGLIP_SKIP_BINARY_DOWNLOAD === "1";
}

function isCI() {
  return process.env.CI === "true" || process.env.GITHUB_ACTIONS === "true";
}

async function downloadFile(url, destPath) {
  return new Promise((resolve, reject) => {
    log(`Downloading: ${url}`);

    const file = fs.createWriteStream(destPath);

    const request = https.get(
      url,
      {
        headers: { "User-Agent": "react-native-siglip" },
      },
      (response) => {
        // Handle redirects (GitHub releases use these)
        if (response.statusCode === 302 || response.statusCode === 301) {
          file.close();
          fs.unlinkSync(destPath);
          return downloadFile(response.headers.location, destPath)
            .then(resolve)
            .catch(reject);
        }

        if (response.statusCode !== 200) {
          file.close();
          fs.unlinkSync(destPath);
          return reject(new Error(`HTTP ${response.statusCode}: ${url}`));
        }

        const totalBytes = parseInt(response.headers["content-length"], 10);
        let downloadedBytes = 0;

        response.on("data", (chunk) => {
          downloadedBytes += chunk.length;
          if (totalBytes && !isCI()) {
            const percent = ((downloadedBytes / totalBytes) * 100).toFixed(1);
            process.stdout.write(
              `\r[react-native-siglip] Progress: ${percent}%  `,
            );
          }
        });

        response.pipe(file);

        file.on("finish", () => {
          file.close();
          if (!isCI()) process.stdout.write("\n");
          resolve();
        });
      },
    );

    request.on("error", (err) => {
      file.close();
      fs.unlinkSync(destPath);
      reject(err);
    });
  });
}

async function extractTarGz(tarPath, destDir) {
  log(`Extracting to: ${destDir}`);
  fs.mkdirSync(destDir, { recursive: true });

  // Use system tar command (available on macOS, Linux, and most CI)
  const { execSync } = require("child_process");
  execSync(`tar -xzf "${tarPath}" --strip-components=1 -C "${destDir}"`);
}

function getReleaseUrl(artifact) {
  return `https://github.com/${GITHUB_REPO}/releases/download/${BINARY_VERSION}/${artifact}`;
}

async function downloadAndExtract(target, config) {
  const projectRoot = path.resolve(__dirname, "..");
  const destDir = path.join(projectRoot, config.destDir);
  const tempPath = path.join(projectRoot, ".tmp", config.artifact);

  // Check if platform is supported
  if (!config.platforms.includes(process.platform)) {
    log(`Skipping ${target} (not supported on ${process.platform})`);
    return;
  }

  // Check if already downloaded
  if (fs.existsSync(destDir) && fs.readdirSync(destDir).length > 0) {
    log(`${target}: Already exists at ${config.destDir}`);
    return;
  }

  try {
    // Create temp directory
    fs.mkdirSync(path.dirname(tempPath), { recursive: true });

    // Download
    const url = getReleaseUrl(config.artifact);
    await downloadFile(url, tempPath);

    // Extract
    await extractTarGz(tempPath, destDir);

    // Cleanup
    fs.unlinkSync(tempPath);

    log(`${target}: Installed to ${config.destDir}`);
  } catch (err) {
    logError(`Failed to download ${target}: ${err.message}`);
    log(
      `You may need to build ExecuTorch manually. See scripts/build-executorch.sh`,
    );

    // Don't fail the install - user might have prebuilt binaries
    if (!isCI()) {
      return;
    }
    throw err;
  }
}

// ============================================================================
// Main
// ============================================================================

async function main() {
  log("Setting up ExecuTorch binaries...");
  log(`Version: ${BINARY_VERSION}`);

  if (shouldSkipDownload()) {
    log("Skipping download (SIGLIP_SKIP_BINARY_DOWNLOAD=1)");
    return;
  }



  // Download each target
  for (const [target, config] of Object.entries(TARGETS)) {
    try {
      await downloadAndExtract(target, config);
    } catch (err) {
      // Log but don't fail - allows manual binary setup
      logError(`${target} setup failed: ${err.message}`);
    }
  }

  // Cleanup temp directory
  const tmpDir = path.join(__dirname, "..", ".tmp");
  if (fs.existsSync(tmpDir)) {
    fs.rmSync(tmpDir, { recursive: true });
  }

  log("Binary setup complete!");
}

main().catch((err) => {
  logError(err.message);
  log("Continuing without prebuilt binaries. Build from source if needed.");
  // Exit 0 to not block npm install
  process.exit(0);
});
