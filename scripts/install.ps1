param(
    [string]$InstallDir,
    [switch]$System,
    [switch]$NoBuild
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
$BinaryName = "hajimi-claw.exe"
$AliasBinaryName = "hajimi.exe"
$ReleaseBinary = Join-Path $RepoRoot "target\release\$BinaryName"

if ([string]::IsNullOrWhiteSpace($InstallDir)) {
    if ($System) {
        $InstallDir = Join-Path ${env:ProgramFiles} "hajimi-claw\bin"
    } else {
        $InstallDir = Join-Path ${env:LOCALAPPDATA} "Programs\hajimi-claw\bin"
    }
}

$ConfigDir = Split-Path -Parent $InstallDir
$ConfigExampleSource = Join-Path $RepoRoot "config.example.toml"
$ConfigExampleTarget = Join-Path $ConfigDir "config.example.toml"

if (-not $NoBuild) {
    Write-Host "Building release binary..."
    cargo build --release --manifest-path (Join-Path $RepoRoot "Cargo.toml")
    if ($LASTEXITCODE -ne 0) {
        throw "cargo build --release failed"
    }
}

if (-not (Test-Path $ReleaseBinary)) {
    throw "Release binary not found at $ReleaseBinary"
}

New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null
Copy-Item -Force $ReleaseBinary (Join-Path $InstallDir $BinaryName)
Copy-Item -Force $ReleaseBinary (Join-Path $InstallDir $AliasBinaryName)
Copy-Item -Force $ConfigExampleSource $ConfigExampleTarget

$PathScope = if ($System) { "Machine" } else { "User" }
$CurrentPath = [Environment]::GetEnvironmentVariable("Path", $PathScope)
$NormalizedPath = @($CurrentPath -split ';') | Where-Object { $_ -and $_.Trim() -ne "" }

if ($NormalizedPath -notcontains $InstallDir) {
    $UpdatedPath = (($NormalizedPath + $InstallDir) -join ';')
    [Environment]::SetEnvironmentVariable("Path", $UpdatedPath, $PathScope)
    Write-Host "Updated $PathScope PATH with $InstallDir"
} else {
    Write-Host "$InstallDir is already present in $PathScope PATH"
}

Write-Host ""
Write-Host "Installed $BinaryName and $AliasBinaryName to $InstallDir"
Write-Host "Config example copied to $ConfigExampleTarget"
Write-Host "Run `hajimi onboard` to create your config, then use `hajimi`."
