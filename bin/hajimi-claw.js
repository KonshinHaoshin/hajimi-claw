#!/usr/bin/env node

const fs = require("fs");
const path = require("path");
const { spawn } = require("child_process");

const root = path.resolve(__dirname, "..");
const binaryName = process.platform === "win32" ? "hajimi-claw.exe" : "hajimi-claw";
const binaryPath = path.join(root, "npm-dist", binaryName);

if (!fs.existsSync(binaryPath)) {
  console.error(
    "hajimi-claw binary is not installed. Reinstall the npm package or run `npm rebuild hajimi-claw`."
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
