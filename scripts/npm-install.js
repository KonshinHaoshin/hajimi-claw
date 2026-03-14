#!/usr/bin/env node

const fs = require("fs");
const path = require("path");
const { spawnSync } = require("child_process");

const root = path.resolve(__dirname, "..");
const cargo = process.platform === "win32" ? "cargo.exe" : "cargo";
const binaryName = process.platform === "win32" ? "hajimi-claw.exe" : "hajimi-claw";
const sourceBinary = path.join(root, "target", "release", binaryName);
const outputDir = path.join(root, "npm-dist");
const outputBinary = path.join(outputDir, binaryName);

function fail(message) {
  console.error(message);
  process.exit(1);
}

const cargoCheck = spawnSync(cargo, ["--version"], { cwd: root, stdio: "pipe" });
if (cargoCheck.error || cargoCheck.status !== 0) {
  fail("cargo is required to install the npm package for hajimi-claw.");
}

const build = spawnSync(
  cargo,
  ["build", "--release", "--manifest-path", path.join(root, "Cargo.toml")],
  {
    cwd: root,
    stdio: "inherit",
  }
);

if (build.status !== 0) {
  fail("cargo build --release failed during npm install.");
}

if (!fs.existsSync(sourceBinary)) {
  fail(`release binary not found at ${sourceBinary}`);
}

fs.mkdirSync(outputDir, { recursive: true });
fs.copyFileSync(sourceBinary, outputBinary);
if (process.platform !== "win32") {
  fs.chmodSync(outputBinary, 0o755);
}
