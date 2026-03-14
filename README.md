# hajimi-claw

Single-user Telegram-first ops agent in Rust.

## Current scope

- Telegram channel -> gateway -> runtime command flow
- Telegram long polling command surface
- Single active task gate
- Structured tools for file access, Docker, and systemd
- Guarded local command execution with approvals and short-lived elevated lease
- SQLite audit/task/session persistence
- Windows-safe execution mode with allowlist checks and Job Object cleanup
- Telegram onboarding flow for custom providers and chat-level provider binding

## Running

1. Run `cargo run -- onboard` or `hajimi onboard`.
2. Fill in the Telegram bot token, admin ids, and provider details.
3. Start the daemon with `cargo run` or `hajimi`.
4. In Telegram, use `/onboard` to add or switch providers interactively.

Set `HAJIMI_CLAW_CONFIG` if you want to load a different config path.

## Install

### Windows

- Build and install to a user directory in `PATH`:
  `powershell -ExecutionPolicy Bypass -File .\scripts\install.ps1`
- System-wide install:
  `powershell -ExecutionPolicy Bypass -File .\scripts\install.ps1 -System`

The installer copies `hajimi-claw.exe`, creates the `hajimi.exe` alias, copies `config.example.toml`, and updates `PATH`.

### Linux

- User install:
  `sh ./scripts/install.sh`
- System-wide install:
  `sudo sh ./scripts/install.sh --system`
- System-wide install with systemd unit:
  `sudo sh ./scripts/install.sh --system --install-service`

The Linux installer copies the binary to `PREFIX/bin/hajimi-claw`, creates the `PREFIX/bin/hajimi` alias, and copies `config.example.toml` to `PREFIX/share/hajimi-claw`.

### npm

- Global install from the package:
  `npm install -g hajimi-claw`
- Global install from the repo:
  `npm install -g .`

The npm package exposes both `hajimi` and `hajimi-claw`, and builds the Rust binary locally during `postinstall`, so `cargo` must already be installed on the target machine.

## CLI

- `hajimi`
- `hajimi daemon`
- `hajimi onboard`
- `hajimi models [provider-id]`
- `hajimi restart`
- `hajimi help`

## Telegram commands

- `/onboard`
- `/onboard cancel`
- `/provider list`
- `/provider current`
- `/provider use <id>`
- `/provider bind <id>`
- `/provider test [id]`
- `/provider models [id]`
- `/ask <text>`
- `/shell open [name]`
- `/shell exec <cmd>`
- `/shell close`
- `/status`
