#!/usr/bin/env node

const fs = require("fs");
const path = require("path");

const root = path.resolve(__dirname, "..");

const TARGETS = {
  "win32-x64-msvc": {
    packageName: "hajimi-claw-win32-x64-msvc",
    dir: path.join(root, "npm", "platforms", "win32-x64-msvc"),
    binaryName: "hajimi-claw.exe",
  },
  "win32-arm64-msvc": {
    packageName: "hajimi-claw-win32-arm64-msvc",
    dir: path.join(root, "npm", "platforms", "win32-arm64-msvc"),
    binaryName: "hajimi-claw.exe",
  },
  "linux-x64-gnu": {
    packageName: "hajimi-claw-linux-x64-gnu",
    dir: path.join(root, "npm", "platforms", "linux-x64-gnu"),
    binaryName: "hajimi-claw",
  },
  "linux-arm64-gnu": {
    packageName: "hajimi-claw-linux-arm64-gnu",
    dir: path.join(root, "npm", "platforms", "linux-arm64-gnu"),
    binaryName: "hajimi-claw",
  },
};

function fail(message) {
  console.error(message);
  process.exit(1);
}

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

function currentTarget() {
  if (process.platform === "win32" && process.arch === "x64") {
    return "win32-x64-msvc";
  }
  if (process.platform === "win32" && process.arch === "arm64") {
    return "win32-arm64-msvc";
  }
  const libc = detectLinuxLibc();
  if (process.platform === "linux" && process.arch === "x64" && libc === "gnu") {
    return "linux-x64-gnu";
  }
  if (process.platform === "linux" && process.arch === "arm64" && libc === "gnu") {
    return "linux-arm64-gnu";
  }
  return null;
}

const targetId = process.argv[2] || process.env.HAJIMI_NPM_TARGET || currentTarget();
if (!targetId || !TARGETS[targetId]) {
  fail(
    `unsupported or missing target. Supported targets: ${Object.keys(TARGETS).join(", ")}`
  );
}

const target = TARGETS[targetId];
const sourceBinary =
  process.argv[3] ||
  process.env.HAJIMI_BINARY_PATH ||
  path.join(root, "target", "release", target.binaryName);

if (!fs.existsSync(sourceBinary)) {
  fail(`source binary not found: ${sourceBinary}`);
}

const outputDir = path.join(target.dir, "bin");
const outputBinary = path.join(outputDir, target.binaryName);
fs.mkdirSync(outputDir, { recursive: true });
fs.copyFileSync(sourceBinary, outputBinary);
if (target.binaryName.endsWith(".exe") === false) {
  fs.chmodSync(outputBinary, 0o755);
}

console.log(`staged ${target.packageName}`);
console.log(`source: ${sourceBinary}`);
console.log(`dest:   ${outputBinary}`);
