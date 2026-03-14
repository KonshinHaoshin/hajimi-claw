use std::sync::Arc;

use anyhow::{Context, Result};
use hajimi_claw_agent::AgentRuntime;
use serde::Deserialize;
use tokio::sync::Mutex;
use tracing::{error, info, warn};

#[derive(Debug, Clone)]
pub struct TelegramConfig {
    pub token: String,
    pub poll_timeout_secs: u64,
    pub admin_user_id: i64,
    pub admin_chat_id: i64,
}

pub struct TelegramBot {
    client: reqwest::Client,
    config: TelegramConfig,
    runtime: Arc<AgentRuntime>,
    current_session: Mutex<Option<String>>,
}

impl TelegramBot {
    pub fn new(config: TelegramConfig, runtime: Arc<AgentRuntime>) -> Self {
        Self {
            client: reqwest::Client::new(),
            config,
            runtime,
            current_session: Mutex::new(None),
        }
    }

    pub async fn run(&self) -> Result<()> {
        let mut offset = 0_i64;
        loop {
            match self.get_updates(offset).await {
                Ok(updates) => {
                    for update in updates {
                        offset = update.update_id + 1;
                        if let Err(err) = self.handle_update(update).await {
                            error!(error = %err, "failed to handle telegram update");
                        }
                    }
                }
                Err(err) => {
                    warn!(error = %err, "telegram polling failed");
                    tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;
                }
            }
        }
    }

    async fn handle_update(&self, update: Update) -> Result<()> {
        let Some(message) = update.message else {
            return Ok(());
        };
        if message.chat.id != self.config.admin_chat_id
            || message
                .from
                .as_ref()
                .map(|user| user.id)
                .unwrap_or_default()
                != self.config.admin_user_id
        {
            warn!("ignored telegram message from unauthorized actor");
            return Ok(());
        }

        let Some(text) = message.text else {
            return Ok(());
        };
        let reply = self.dispatch_command(&text).await;
        self.send_message(message.chat.id, &reply).await?;
        Ok(())
    }

    async fn dispatch_command(&self, text: &str) -> String {
        match Command::parse(text) {
            Command::Ask(prompt) => self
                .runtime
                .ask(&prompt, None)
                .await
                .unwrap_or_else(|err| format!("error: {err}")),
            Command::ShellOpen(name) => match self.runtime.shell_open(name, None).await {
                Ok(reply) => {
                    if let Some(session_id) = reply.split_whitespace().nth(2) {
                        *self.current_session.lock().await = Some(session_id.to_string());
                    }
                    reply
                }
                Err(err) => format!("error: {err}"),
            },
            Command::ShellExec(command) => {
                let session_id = self.current_session.lock().await.clone();
                match session_id {
                    Some(session_id) => self
                        .runtime
                        .shell_exec(&session_id, &command)
                        .await
                        .unwrap_or_else(|err| format!("error: {err}")),
                    None => "error: no active session, use /shell open first".into(),
                }
            }
            Command::ShellClose => {
                let session_id = self.current_session.lock().await.clone();
                match session_id {
                    Some(session_id) => {
                        let reply = self
                            .runtime
                            .shell_close(&session_id)
                            .await
                            .unwrap_or_else(|err| format!("error: {err}"));
                        *self.current_session.lock().await = None;
                        reply
                    }
                    None => "no active session".into(),
                }
            }
            Command::Status => self
                .runtime
                .status()
                .unwrap_or_else(|err| format!("error: {err}")),
            Command::Approve(request_id) => self
                .runtime
                .approve(&request_id)
                .unwrap_or_else(|err| format!("error: {err}")),
            Command::ElevatedStart { minutes, reason } => {
                self.runtime.request_elevated(minutes, reason)
            }
            Command::ElevatedStop => self.runtime.stop_elevated(),
            Command::Cancel(task_id) => format!("cancel is not implemented yet for task {task_id}"),
            Command::Help => help_text(),
            Command::Unknown(raw) => format!("unrecognized command: {raw}\n\n{}", help_text()),
        }
    }

    async fn get_updates(&self, offset: i64) -> Result<Vec<Update>> {
        let response = self
            .client
            .post(self.api_url("getUpdates"))
            .json(&serde_json::json!({
                "timeout": self.config.poll_timeout_secs,
                "offset": offset,
            }))
            .send()
            .await
            .context("getUpdates request")?;

        let payload: TelegramEnvelope<Vec<Update>> =
            response.json().await.context("decode updates")?;
        if !payload.ok {
            anyhow::bail!("telegram getUpdates returned ok=false");
        }
        Ok(payload.result)
    }

    async fn send_message(&self, chat_id: i64, text: &str) -> Result<()> {
        let response = self
            .client
            .post(self.api_url("sendMessage"))
            .json(&serde_json::json!({
                "chat_id": chat_id,
                "text": clamp_text(text),
            }))
            .send()
            .await
            .context("sendMessage request")?;
        let payload: TelegramEnvelope<serde_json::Value> =
            response.json().await.context("decode sendMessage")?;
        if !payload.ok {
            anyhow::bail!("telegram sendMessage returned ok=false");
        }
        info!("sent telegram message");
        Ok(())
    }

    fn api_url(&self, method: &str) -> String {
        format!(
            "https://api.telegram.org/bot{}/{}",
            self.config.token, method
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Command {
    Ask(String),
    ShellOpen(Option<String>),
    ShellExec(String),
    ShellClose,
    Status,
    Approve(String),
    ElevatedStart { minutes: i64, reason: String },
    ElevatedStop,
    Cancel(String),
    Help,
    Unknown(String),
}

impl Command {
    fn parse(text: &str) -> Self {
        let trimmed = text.trim();
        if let Some(rest) = trimmed.strip_prefix("/ask ") {
            return Self::Ask(rest.trim().into());
        }
        if trimmed == "/status" {
            return Self::Status;
        }
        if let Some(rest) = trimmed.strip_prefix("/approve ") {
            return Self::Approve(rest.trim().into());
        }
        if trimmed == "/elevated stop" {
            return Self::ElevatedStop;
        }
        if let Some(rest) = trimmed.strip_prefix("/elevated start ") {
            let mut parts = rest.trim().splitn(2, ' ');
            let minutes = parts
                .next()
                .and_then(|value| value.parse::<i64>().ok())
                .unwrap_or(10);
            let reason = parts.next().unwrap_or("manual request").trim().to_string();
            return Self::ElevatedStart { minutes, reason };
        }
        if trimmed == "/shell close" {
            return Self::ShellClose;
        }
        if let Some(rest) = trimmed.strip_prefix("/shell open") {
            let name = rest.trim();
            return Self::ShellOpen((!name.is_empty()).then(|| name.to_string()));
        }
        if let Some(rest) = trimmed.strip_prefix("/shell exec ") {
            return Self::ShellExec(rest.trim().into());
        }
        if let Some(rest) = trimmed.strip_prefix("/cancel ") {
            return Self::Cancel(rest.trim().into());
        }
        if trimmed == "/help" || trimmed == "/start" {
            return Self::Help;
        }
        Self::Unknown(trimmed.into())
    }
}

#[derive(Debug, Deserialize)]
struct TelegramEnvelope<T> {
    ok: bool,
    result: T,
}

#[derive(Debug, Deserialize)]
struct Update {
    update_id: i64,
    message: Option<Message>,
}

#[derive(Debug, Deserialize)]
struct Message {
    chat: Chat,
    from: Option<User>,
    text: Option<String>,
}

#[derive(Debug, Deserialize)]
struct Chat {
    id: i64,
}

#[derive(Debug, Deserialize)]
struct User {
    id: i64,
}

fn clamp_text(text: &str) -> String {
    const MAX: usize = 3500;
    if text.len() <= MAX {
        return text.to_string();
    }
    format!("{}...\n[truncated]", &text[..MAX])
}

fn help_text() -> String {
    [
        "/ask <text>",
        "/shell open [name]",
        "/shell exec <cmd>",
        "/shell close",
        "/status",
        "/approve <request-id>",
        "/elevated start <minutes> <reason>",
        "/elevated stop",
        "/cancel <task-id>",
    ]
    .join("\n")
}

#[cfg(test)]
mod tests {
    use super::Command;

    #[test]
    fn parses_elevated_start() {
        assert_eq!(
            Command::parse("/elevated start 15 maintenance"),
            Command::ElevatedStart {
                minutes: 15,
                reason: "maintenance".into(),
            }
        );
    }
}
