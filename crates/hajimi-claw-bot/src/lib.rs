use std::sync::Arc;

use anyhow::{Context, Result};
use hajimi_claw_gateway::{
    Gateway, GatewayRequest, InlineKeyboard, SessionDirective, parse_gateway_command,
};
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
    gateway: Arc<dyn Gateway>,
    current_session: Mutex<Option<String>>,
}

impl TelegramBot {
    pub fn new(config: TelegramConfig, gateway: Arc<dyn Gateway>) -> Self {
        Self {
            client: reqwest::Client::new(),
            config,
            gateway,
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
        if let Some(message) = update.message {
            if !self.is_authorized(
                message.chat.id,
                message
                    .from
                    .as_ref()
                    .map(|user| user.id)
                    .unwrap_or_default(),
            ) {
                warn!("ignored telegram message from unauthorized actor");
                return Ok(());
            }

            let Some(text) = message.text else {
                return Ok(());
            };
            let reply = self.dispatch_command(&text).await;
            self.send_message(message.chat.id, &reply.text, reply.keyboard)
                .await?;
            return Ok(());
        }

        if let Some(callback_query) = update.callback_query {
            let chat_id = callback_query
                .message
                .as_ref()
                .map(|message| message.chat.id)
                .unwrap_or(self.config.admin_chat_id);
            if !self.is_authorized(chat_id, callback_query.from.id) {
                warn!("ignored telegram callback from unauthorized actor");
                return Ok(());
            }

            if let Some(data) = callback_query.data {
                let reply = self.dispatch_command(&data).await;
                self.send_message(chat_id, &reply.text, reply.keyboard)
                    .await?;
            }
            self.answer_callback_query(&callback_query.id).await?;
        }

        Ok(())
    }

    async fn dispatch_command(&self, text: &str) -> BotReply {
        let current_session_id = self.current_session.lock().await.clone();
        match self
            .gateway
            .handle(GatewayRequest {
                actor_user_id: self.config.admin_user_id,
                actor_chat_id: self.config.admin_chat_id,
                raw_text: text.to_string(),
                command: parse_gateway_command(text),
                current_session_id,
            })
            .await
        {
            Ok(response) => {
                match response.session {
                    SessionDirective::Keep => {}
                    SessionDirective::Set(session_id) => {
                        *self.current_session.lock().await = Some(session_id);
                    }
                    SessionDirective::Clear => {
                        *self.current_session.lock().await = None;
                    }
                }
                BotReply {
                    text: response.text,
                    keyboard: response.keyboard,
                }
            }
            Err(err) => BotReply {
                text: format!("error: {err}"),
                keyboard: None,
            },
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

    async fn send_message(
        &self,
        chat_id: i64,
        text: &str,
        keyboard: Option<InlineKeyboard>,
    ) -> Result<()> {
        let payload = if let Some(keyboard) = keyboard {
            serde_json::json!({
                "chat_id": chat_id,
                "text": clamp_text(text),
                "reply_markup": to_reply_markup(keyboard),
            })
        } else {
            serde_json::json!({
                "chat_id": chat_id,
                "text": clamp_text(text),
            })
        };
        let response = self
            .client
            .post(self.api_url("sendMessage"))
            .json(&payload)
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

    async fn answer_callback_query(&self, callback_query_id: &str) -> Result<()> {
        let response = self
            .client
            .post(self.api_url("answerCallbackQuery"))
            .json(&serde_json::json!({
                "callback_query_id": callback_query_id,
            }))
            .send()
            .await
            .context("answerCallbackQuery request")?;
        let payload: TelegramEnvelope<serde_json::Value> = response
            .json()
            .await
            .context("decode answerCallbackQuery")?;
        if !payload.ok {
            anyhow::bail!("telegram answerCallbackQuery returned ok=false");
        }
        Ok(())
    }

    fn api_url(&self, method: &str) -> String {
        format!(
            "https://api.telegram.org/bot{}/{}",
            self.config.token, method
        )
    }

    fn is_authorized(&self, chat_id: i64, user_id: i64) -> bool {
        chat_id == self.config.admin_chat_id && user_id == self.config.admin_user_id
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
    callback_query: Option<CallbackQuery>,
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

#[derive(Debug, Deserialize)]
struct CallbackQuery {
    id: String,
    from: User,
    data: Option<String>,
    message: Option<Message>,
}

struct BotReply {
    text: String,
    keyboard: Option<InlineKeyboard>,
}

fn clamp_text(text: &str) -> String {
    const MAX: usize = 3500;
    if text.len() <= MAX {
        return text.to_string();
    }
    format!("{}...\n[truncated]", &text[..MAX])
}

fn to_reply_markup(keyboard: InlineKeyboard) -> serde_json::Value {
    let rows = keyboard
        .rows
        .into_iter()
        .map(|row| {
            row.into_iter()
                .map(|button| {
                    serde_json::json!({
                        "text": button.text,
                        "callback_data": button.data,
                    })
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    serde_json::json!({ "inline_keyboard": rows })
}

#[cfg(test)]
mod tests {
    use hajimi_claw_gateway::{GatewayCommand, parse_gateway_command};

    #[test]
    fn parses_elevated_start() {
        assert_eq!(
            parse_gateway_command("/elevated start 15 maintenance"),
            GatewayCommand::ElevatedStart {
                minutes: 15,
                reason: "maintenance".into(),
            }
        );
    }
}
