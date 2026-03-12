const {
  createRunOncePlugin,
  withAppBuildGradle,
  withDangerousMod,
  withXcodeProject,
} = require("@expo/config-plugins");
const fs = require("fs");
const path = require("path");

const pkg = require("./package.json");

const withSigLIP = (config, props = {}) => {
  const enableCoreML = props.enableCoreML ?? true;
  const enableQNN = props.enableQNN ?? false;
  const modelsDirectory = props.modelsDirectory ?? "siglip_models";
  const verboseLogs = props.verboseLogs ?? false;

  config = withXcodeProject(config, (modConfig) => {
    const project = modConfig.modResults;
    const target = project.getFirstTarget().uuid;
    if (enableCoreML) {
      project.addFramework("CoreML.framework", { target });
      project.addFramework("Accelerate.framework", { target });
    }
    project.addBuildProperty("OTHER_LDFLAGS", '"$(inherited) -ObjC -all_load"');
    return modConfig;
  });

  config = withAppBuildGradle(config, (modConfig) => {
    const qnnLine = `buildConfigField "boolean", "SIGLIP_ENABLE_QNN", "${enableQNN ? "true" : "false"}"`;
    if (!modConfig.modResults.contents.includes("SIGLIP_ENABLE_QNN")) {
      modConfig.modResults.contents = modConfig.modResults.contents.replace(
        /defaultConfig\s*\{/,
        (m) => `${m}\n    ${qnnLine}`,
      );
    }
    if (!modConfig.modResults.contents.includes("abiFilters")) {
      modConfig.modResults.contents = modConfig.modResults.contents.replace(
        /defaultConfig\s*\{/,
        (m) => `${m}\n        ndk {\n            abiFilters 'arm64-v8a'\n        }`,
      );
    }
    if (!modConfig.modResults.contents.includes('pickFirst "**/libjsi.so"')) {
      modConfig.modResults.contents = modConfig.modResults.contents.replace(
        /android\s*\{/,
        (m) =>
          `${m}
  packagingOptions {
    pickFirst "**/libc++_shared.so"
    pickFirst "**/libfbjni.so"
    pickFirst "**/libjsi.so"
  }`,
      );
    }
    return modConfig;
  });

  config = withDangerousMod(config, [
    "android",
    async (modConfig) => {
      const dir = path.join(
        modConfig.modRequest.projectRoot,
        "android",
        "app",
        "src",
        "main",
        "assets",
        modelsDirectory,
      );
      fs.mkdirSync(dir, { recursive: true });
      const noMedia = path.join(dir, ".nomedia");
      if (!fs.existsSync(noMedia)) {
        fs.writeFileSync(noMedia, "");
        if (verboseLogs) {
          // eslint-disable-next-line no-console
          console.log(`[react-native-siglip] Created ${noMedia}`);
        }
      }
      return modConfig;
    },
  ]);

  config.extra = config.extra || {};
  config.extra.siglip = {
    ...(config.extra.siglip || {}),
    enableCoreML,
    enableQNN,
    modelsDirectory,
    verboseLogs,
  };

  return config;
};

module.exports = createRunOncePlugin(
  withSigLIP,
  "react-native-siglip",
  pkg.version,
);
