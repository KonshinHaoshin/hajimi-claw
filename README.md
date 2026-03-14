# hajimi-claw

Single-user Telegram-first ops agent in Rust.

## Current scope

- Telegram long polling command surface
- Single active task gate
- Structured tools for file access, Docker, and systemd
- Guarded local command execution with approvals and short-lived elevated lease
- SQLite audit/task/session persistence
- Windows-safe execution mode with allowlist checks and Job Object cleanup

## Running

1. Copy `config.example.toml` to `config.toml`.
2. Fill in the Telegram bot token, admin ids, and LLM settings.
3. Run `cargo run`.

Set `HAJIMI_CLAW_CONFIG` if you want to load a different config path.
