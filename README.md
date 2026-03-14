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
2. `hajimi onboard` verifies the Telegram bot token, can pair the admin user/chat automatically with a one-time pairing code sent to the bot, and interactively guides provider model selection.
3. Start the daemon with `cargo run` or `hajimi`.
4. In Telegram, use `/onboard` to add or switch providers interactively.

Set `HAJIMI_CLAW_CONFIG` if you want to load a different config path.

For background use without keeping a terminal open:

- `hajimi launch`
- `hajimi status`
- `hajimi stop`

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
- `hajimi launch`
- `hajimi stop`
- `hajimi status`
- `hajimi onboard`
- `hajimi providers`
- `hajimi provider current`
- `hajimi provider use <provider-id>`
- `hajimi provider models [provider-id]`
- `hajimi provider set-model <provider-id> <model>`
- `hajimi model current`
- `hajimi model use <model>`
- `hajimi models [provider-id]`
- `hajimi restart`
- `hajimi help`

## Persona Files

`hajimi onboard` now creates empty persona files in `~/.hajimi/`:

- `soul.md`
- `agents.md`
- `tools.md`
- `skills.md`

`hajimi` reloads these files on each request, so Telegram edits take effect on the next `/ask`.

- Auto-discovered files: `soul.md`, `agents.md`, `AGENTS.md`, `tools.md`, `skills.md`
- Search roots: the current working directory, the config directory, and `~/.hajimi`
- Optional explicit list: set `[persona].prompt_files` in `config.toml`

Use these files for:

- `soul.md`: tone, temperament, and high-level persona
- `agents.md` or `AGENTS.md`: repo or operator instructions
- `tools.md`: tool-use policy and operational preferences
- `skills.md`: extra habits, playbooks, and skill-selection hints

## Telegram commands

- `/onboard`
- `/onboard cancel`
- `/provider list`
- `/provider add`
- `/provider current`
- `/provider use <id>`
- `/provider bind <id>`
- `/provider test [id]`
- `/provider models [id]`
- `/provider set-model <provider-id> <model>`
- `/model current`
- `/model use <model>`
- `/persona list`
- `/persona read <soul|agents|tools|skills>`
- `/persona write <file> <content>`
- `/persona append <file> <content>`
- `/ask <text>`
- `/shell open [name]`
- `/shell exec <cmd>`
- `/shell close`
- `/status`

Plain Telegram text now defaults to a natural-language task, so `/ask` is optional for normal requests.

## Provider And Model Switching

- Add a new provider: `hajimi onboard` or Telegram `/provider add`
- List configured providers: `hajimi providers` or Telegram `/provider list`
- Switch the default provider: `hajimi provider use <provider-id>` or Telegram `/provider use <id>`
- See the current model: `hajimi model current` or Telegram `/model current`
- Switch the current model on the active provider: `hajimi model use <model>` or Telegram `/model use <model>`
- Switch a specific provider to a specific model: `hajimi provider set-model <provider-id> <model>` or Telegram `/provider set-model <provider-id> <model>`
