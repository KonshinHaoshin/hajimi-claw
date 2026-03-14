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

1. Copy `config.example.toml` to `config.toml`.
2. Fill in the Telegram bot token and admin ids.
3. Set the provider secret key env var before starting:
   `set HAJIMI_CLAW_MASTER_KEY=replace-me-with-a-long-random-string`
4. Optional: fill in the `llm` section to bootstrap a default provider on first start.
5. Run `cargo run`.
6. In Telegram, use `/onboard` to add or switch providers interactively.

Set `HAJIMI_CLAW_CONFIG` if you want to load a different config path.

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
