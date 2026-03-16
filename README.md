# hajimi-claw

Single-user Telegram/Feishu-first ops agent in Rust.

## Current scope

- Telegram channel -> gateway -> runtime command flow
- Feishu webhook channel -> gateway -> runtime command flow
- Telegram long polling command surface
- Single active task gate
- Structured tools for file access, Docker, and systemd
- Guarded local command execution with approval mode, short-lived elevated mode, and full elevated mode
- SQLite audit/task/session persistence
- Windows-safe execution mode with allowlist checks and Job Object cleanup
- Channel-aware onboarding for Telegram or Feishu plus provider/model setup

## Running

1. Run `cargo run -- onboard` or `hajimi onboard`.
2. `hajimi onboard` lets you choose `telegram` or `feishu`, verifies the channel credentials, and then interactively guides provider model selection.
3. Start the daemon with `cargo run` or `hajimi`.
4. In Telegram, use `/menu` for the main quick-action panel, or just send plain text and let hajimi reply in natural language.
5. In Feishu mode, configure the app's event subscription URL to point at `http(s)://<your-host><event_path>` for the local webhook listener.

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

## Channels

- `telegram`
- `feishu`

`hajimi onboard` now lets you pick the primary channel:

- `telegram`: verifies the bot token and can auto-pair the admin user/chat
- `feishu`: asks for `app_id` and `app_secret`, verifies them by requesting a tenant access token, and starts a webhook listener on `listen_addr + event_path`

Current Feishu limitations:

- You still need to configure the Feishu event subscription URL manually in the Feishu developer console
- The Feishu adapter replies with text only; Telegram inline buttons are flattened into text actions

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
- `hajimi model use [model]`
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
- `/model use [model]`
- `/persona list`
- `/persona read <soul|agents|tools|skills>`
- `/persona write <file> <content>`
- `/persona append <file> <content>`
- `/ask <text>`
- `/shell open [name]`
- `/shell exec <cmd>`
- `/shell close`
- `/status`
- `/menu`
- `/elevated on`
- `/elevated off`
- `/elevated ask`
- `/elevated full`

Plain Telegram text now defaults to a natural-language task, so `/ask` is optional for normal requests. hajimi sends a Telegram `typing` action, posts a short placeholder, and then edits that message with the final answer.

The bot also registers Telegram slash commands with `setMyCommands` on startup and exposes an inline quick-action menu via `/menu` and `/help`. `/provider current` and `/model current` now return inline buttons so you can switch provider/model directly from Telegram without typing full commands.

## Provider And Model Switching

- Add a new provider: `hajimi onboard` or Telegram `/provider add`
- List configured providers: `hajimi providers` or Telegram `/provider list`
- Switch the default provider: `hajimi provider use <provider-id>` or Telegram `/provider use <id>`
- See the current model: `hajimi model current` or Telegram `/model current`
- Switch the current model on the active provider: `hajimi model use [model]` or Telegram `/model use [model]`
- Open the model picker with inline buttons: `hajimi model use` or Telegram `/model use`
- Switch a specific provider to a specific model: `hajimi provider set-model <provider-id> <model>` or Telegram `/provider set-model <provider-id> <model>`
