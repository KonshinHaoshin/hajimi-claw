#!/usr/bin/env node

const fs = require("fs");
const path = require("path");

const root = path.resolve(__dirname, "..");
const rootPackagePath = path.join(root, "package.json");
const platformsRoot = path.join(root, "npm", "platforms");

function readJson(filePath) {
  return JSON.parse(fs.readFileSync(filePath, "utf8"));
}

function writeJson(filePath, value) {
  fs.writeFileSync(filePath, `${JSON.stringify(value, null, 2)}\n`);
}

const rootPackage = readJson(rootPackagePath);
const version = rootPackage.version;

const platformDirs = fs
  .readdirSync(platformsRoot, { withFileTypes: true })
  .filter((entry) => entry.isDirectory())
  .map((entry) => path.join(platformsRoot, entry.name));

const optionalDependencies = {
  ...(rootPackage.optionalDependencies || {}),
};

for (const dir of platformDirs) {
  const packagePath = path.join(dir, "package.json");
  const manifest = readJson(packagePath);
  manifest.version = version;
  writeJson(packagePath, manifest);
  optionalDependencies[manifest.name] = version;
}

rootPackage.optionalDependencies = Object.fromEntries(
  Object.entries(optionalDependencies).sort(([left], [right]) =>
    left.localeCompare(right)
  )
);

writeJson(rootPackagePath, rootPackage);

console.log(`Synced npm package versions to ${version}`);
