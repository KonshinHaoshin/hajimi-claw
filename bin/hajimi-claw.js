#!/usr/bin/env node

const fs = require("fs");
const path = require("path");
const { spawn } = require("child_process");

const root = path.resolve(__dirname, "..");
const binaryName = process.platform === "win32" ? "hajimi-claw.exe" : "hajimi-claw";

function detectLinuxLibc() {
  if (process.platform !== "linux") {
    return null;
  }
  const report = process.report && typeof process.report.getReport === "function"
    ? process.report.getReport()
    : null;
  const glibc = report && report.header && report.header.glibcVersionRuntime;
  return glibc ? "gnu" : "musl";
}

function platformPackageName() {
  if (process.platform === "win32" && process.arch === "x64") {
    return "hajimi-claw-win32-x64-msvc";
  }
  if (process.platform === "win32" && process.arch === "arm64") {
    return "hajimi-claw-win32-arm64-msvc";
  }
  const libc = detectLinuxLibc();
  if (process.platform === "linux" && process.arch === "x64" && libc === "gnu") {
    return "hajimi-claw-linux-x64-gnu";
  }
  if (process.platform === "linux" && process.arch === "arm64" && libc === "gnu") {
    return "hajimi-claw-linux-arm64-gnu";
  }
  return null;
}

function resolveInstalledBinary() {
  const packageName = platformPackageName();
  if (!packageName) {
    return null;
  }
  try {
    const entry = require.resolve(packageName, { paths: [root] });
    const binaryPath = require(entry);
    if (typeof binaryPath === "string" && fs.existsSync(binaryPath)) {
      return binaryPath;
    }
  } catch (_) {
    return null;
  }
  return null;
}

function resolveLocalFallback() {
  const localDirMap = {
    "hajimi-claw-win32-x64-msvc": "win32-x64-msvc",
    "hajimi-claw-win32-arm64-msvc": "win32-arm64-msvc",
    "hajimi-claw-linux-x64-gnu": "linux-x64-gnu",
    "hajimi-claw-linux-arm64-gnu": "linux-arm64-gnu",
  };
  const packageName = platformPackageName();
  const candidates = [];
  if (process.env.HAJIMI_CLAW_BINARY_PATH) {
    candidates.push(process.env.HAJIMI_CLAW_BINARY_PATH);
  }
  if (packageName && localDirMap[packageName]) {
    candidates.push(
      path.join(root, "npm", "platforms", localDirMap[packageName], "bin", binaryName)
    );
  }
  candidates.push(path.join(root, "npm-dist", binaryName));
  candidates.push(path.join(root, "target", "release", binaryName));
  return candidates.find((candidate) => fs.existsSync(candidate)) || null;
}

const binaryPath = resolveInstalledBinary() || resolveLocalFallback();

if (!binaryPath) {
  const packageName = platformPackageName();
  const targetLabel = packageName || `${process.platform}-${process.arch}`;
  console.error(
    [
      `No prebuilt hajimi binary is available for ${targetLabel}.`,
      "Install a published platform package, or stage a local binary with `npm run stage:binary`.",
      "If you are publishing releases, publish the matching platform package before the root package."
    ].join("\n")
  );
  process.exit(1);
}

const child = spawn(binaryPath, process.argv.slice(2), {
  stdio: "inherit",
});

child.on("exit", (code, signal) => {
  if (signal) {
    process.kill(process.pid, signal);
    return;
  }
  process.exit(code ?? 0);
});
