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
- Configurable multi-agent orchestration with coordinator/worker/integrator flow

## Running

1. Run `cargo run -- onboard` or `hajimi onboard`.
2. `hajimi onboard` lets you choose `telegram`, `feishu`, or `skip`, verifies the channel credentials when configured, and then interactively guides provider model selection.
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

The npm package exposes both `hajimi` and `hajimi-claw`, but it no longer compiles Rust during
install. Instead:

- The root package is a small launcher package
- Platform-specific binaries are published as optional npm packages
- `npm install -g hajimi-claw` installs the matching prebuilt binary package when one is available

Supported prebuilt package names:

- `hajimi-claw-win32-x64-msvc`
- `hajimi-claw-win32-arm64-msvc`
- `hajimi-claw-linux-x64-gnu`
- `hajimi-claw-linux-arm64-gnu`

For local testing from the repo, stage a binary into the local platform package first:

```bash
cargo build --release
npm run stage:binary
npm install -g .
```

For CI or release staging, you can target a specific package explicitly:

```bash
node scripts/stage-prebuilt.js win32-x64-msvc path/to/hajimi-claw.exe
node scripts/stage-prebuilt.js linux-x64-gnu path/to/hajimi-claw
```

Recommended publish order:

1. Build each platform binary in CI
2. Stage it into the matching `npm/platforms/<target>/bin/`
3. Publish each platform package
4. Publish the root `hajimi-claw` package last

Current GitHub Actions automation is wired for:

- `hajimi-claw-win32-x64-msvc`
- `hajimi-claw-linux-x64-gnu`

The arm64 package skeletons are present in the repo, but they are not yet part of the automated
publish matrix.

## npm Versioning

For npm releases, the root `package.json` is the single version source.

When you change `package.json.version`, run:

```bash
npm run sync:npm-versions
```

That script updates:

- the root package `optionalDependencies`
- every platform package version under `npm/platforms/*/package.json`

The release workflow also runs this sync step automatically, so you do not need to edit platform
package versions by hand.

## Channels

- `telegram`
- `feishu`

`hajimi onboard` now lets you pick the primary channel:

- `telegram`: verifies the bot token and can auto-pair the admin user/chat
- `feishu`: asks for `app_id` and `app_secret`, verifies them by requesting a tenant access token, and supports both `webhook` and `long-connection` modes
- `skip`: leaves channel setup empty for now so you can finish provider/model setup first

Current Feishu limitations:

- In `webhook` mode, you still need to configure the Feishu event subscription URL manually in the Feishu developer console
- In `long-connection` mode, message receive events can come through the WebSocket connection, but card button callbacks still need the lightweight HTTP callback path
- Feishu card support currently targets `menu/provider/model` style button flows; it is not a full parity implementation of Telegram inline interactions

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

## Multi-Agent

`hajimi` can split one natural-language request into multiple sub-agents. This is configured in
`config.toml` under `[multi_agent]`:

```toml
[multi_agent]
enabled = true
auto_delegate = false
default_workers = 3
max_workers = 8
worker_timeout_secs = 90
max_context_chars_per_worker = 24000
```

Natural-language triggers:

- `use 4 agents to analyze this`
- `开 6 个 agent 帮我排查`
- `please use sub agents for this`

Behavior:

- If the prompt explicitly asks for `N agents` or `N workers`, hajimi uses that count up to
  `max_workers`
- If the prompt says `sub agent` / `multi agent`, hajimi uses `default_workers`
- `auto_delegate = true` enables light automatic delegation for obviously parallel analysis tasks
- Worker agents only perform parallel reasoning in v1; local command execution still stays behind
  the existing single runtime/policy path
- Session-level controls are available in chat:
  - `/agents on`
  - `/agents off`
  - `/agents auto`
  - `/agents status`
- When a natural-language task is about to run in multi-agent mode, Telegram now shows a temporary
  progress card and Feishu sends a status card before the final reply

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
- `hajimi onboard` now supports optional fallback models per provider; if the primary model fails, hajimi will retry the provider using the configured fallback models in order
