use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use axum::extract::State;
use axum::routing::post;
use axum::{Json, Router};
use hajimi_claw_gateway::{
    Gateway, GatewayRequest, InlineKeyboard, SessionDirective, parse_gateway_command,
};
use hajimi_claw_types::ClawError;
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

#[derive(Debug, Clone)]
pub struct FeishuConfig {
    pub app_id: String,
    pub app_secret: String,
    pub listen_addr: String,
    pub event_path: String,
    pub admin_user_id: i64,
    pub admin_chat_id: i64,
}

pub struct TelegramBot {
    client: reqwest::Client,
    config: TelegramConfig,
    gateway: Arc<dyn Gateway>,
    current_session: Mutex<Option<String>>,
}

pub struct FeishuBot {
    client: reqwest::Client,
    config: FeishuConfig,
    gateway: Arc<dyn Gateway>,
    current_session: Mutex<Option<String>>,
    token_cache: Mutex<Option<CachedTenantToken>>,
}

#[derive(Debug, Clone)]
struct CachedTenantToken {
    value: String,
    expires_at: Instant,
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
        self.set_my_commands().await?;
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
            if is_natural_language(&text) {
                self.send_chat_action(message.chat.id, "typing").await?;
                let placeholder_id = self
                    .send_message(message.chat.id, "Processing your request...", None)
                    .await?;
                let reply = self.dispatch_command(&text).await;
                self.edit_message(message.chat.id, placeholder_id, &reply.text, reply.keyboard)
                    .await?;
            } else {
                let reply = self.dispatch_command(&text).await;
                self.send_message(message.chat.id, &reply.text, reply.keyboard)
                    .await?;
            }
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
                if let Some(message) = callback_query.message {
                    self.edit_message(chat_id, message.message_id, &reply.text, reply.keyboard)
                        .await?;
                } else {
                    self.send_message(chat_id, &reply.text, reply.keyboard)
                        .await?;
                }
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
                text: render_user_error(&err),
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
    ) -> Result<i64> {
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
        let payload: TelegramEnvelope<TelegramMessagePayload> =
            response.json().await.context("decode sendMessage")?;
        if !payload.ok {
            anyhow::bail!("telegram sendMessage returned ok=false");
        }
        info!("sent telegram message");
        Ok(payload.result.message_id)
    }

    async fn edit_message(
        &self,
        chat_id: i64,
        message_id: i64,
        text: &str,
        keyboard: Option<InlineKeyboard>,
    ) -> Result<()> {
        let payload = if let Some(keyboard) = keyboard {
            serde_json::json!({
                "chat_id": chat_id,
                "message_id": message_id,
                "text": clamp_text(text),
                "reply_markup": to_reply_markup(keyboard),
            })
        } else {
            serde_json::json!({
                "chat_id": chat_id,
                "message_id": message_id,
                "text": clamp_text(text),
            })
        };
        let response = self
            .client
            .post(self.api_url("editMessageText"))
            .json(&payload)
            .send()
            .await
            .context("editMessageText request")?;
        let payload: TelegramEnvelope<serde_json::Value> =
            response.json().await.context("decode editMessageText")?;
        if !payload.ok {
            anyhow::bail!("telegram editMessageText returned ok=false");
        }
        Ok(())
    }

    async fn send_chat_action(&self, chat_id: i64, action: &str) -> Result<()> {
        let response = self
            .client
            .post(self.api_url("sendChatAction"))
            .json(&serde_json::json!({
                "chat_id": chat_id,
                "action": action,
            }))
            .send()
            .await
            .context("sendChatAction request")?;
        let payload: TelegramEnvelope<serde_json::Value> =
            response.json().await.context("decode sendChatAction")?;
        if !payload.ok {
            anyhow::bail!("telegram sendChatAction returned ok=false");
        }
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

    async fn set_my_commands(&self) -> Result<()> {
        let response = self
            .client
            .post(self.api_url("setMyCommands"))
            .json(&serde_json::json!({
                "commands": [
                    { "command": "menu", "description": "Open quick actions" },
                    { "command": "status", "description": "Show task and policy status" },
                    { "command": "onboard", "description": "Add a provider" },
                    { "command": "provider", "description": "Manage providers" },
                    { "command": "model", "description": "Manage the active model" },
                    { "command": "elevated", "description": "Switch elevation mode" },
                    { "command": "persona", "description": "Manage persona files" },
                    { "command": "shell", "description": "Open or control shell sessions" },
                    { "command": "help", "description": "Show bot help" }
                ]
            }))
            .send()
            .await
            .context("setMyCommands request")?;
        let payload: TelegramEnvelope<serde_json::Value> =
            response.json().await.context("decode setMyCommands")?;
        if !payload.ok {
            anyhow::bail!("telegram setMyCommands returned ok=false");
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

impl FeishuBot {
    pub fn new(config: FeishuConfig, gateway: Arc<dyn Gateway>) -> Self {
        Self {
            client: reqwest::Client::new(),
            config,
            gateway,
            current_session: Mutex::new(None),
            token_cache: Mutex::new(None),
        }
    }

    pub async fn run(self: Arc<Self>) -> Result<()> {
        let addr: SocketAddr = self.config.listen_addr.parse().with_context(|| {
            format!("parse Feishu listen address `{}`", self.config.listen_addr)
        })?;
        let app = Router::new()
            .route(&self.config.event_path, post(feishu_event_handler))
            .with_state(self.clone());
        let listener = tokio::net::TcpListener::bind(addr)
            .await
            .with_context(|| format!("bind Feishu listener {}", addr))?;
        info!(
            listen_addr = %addr,
            event_path = %self.config.event_path,
            "listening for Feishu events"
        );
        axum::serve(listener, app)
            .await
            .context("serve Feishu webhook")?;
        Ok(())
    }

    async fn handle_event(&self, payload: serde_json::Value) -> Result<serde_json::Value> {
        if payload
            .get("type")
            .and_then(|value| value.as_str())
            .is_some_and(|value| value == "url_verification")
        {
            let challenge = payload
                .get("challenge")
                .and_then(|value| value.as_str())
                .context("Feishu challenge is missing")?;
            return Ok(serde_json::json!({ "challenge": challenge }));
        }

        let event_type = payload
            .pointer("/header/event_type")
            .and_then(|value| value.as_str())
            .unwrap_or_default();
        if event_type != "im.message.receive_v1" {
            return Ok(serde_json::json!({ "code": 0 }));
        }

        let message_type = payload
            .pointer("/event/message/message_type")
            .and_then(|value| value.as_str())
            .unwrap_or_default();
        if message_type != "text" {
            return Ok(serde_json::json!({ "code": 0 }));
        }

        let content = payload
            .pointer("/event/message/content")
            .and_then(|value| value.as_str())
            .context("Feishu text content missing")?;
        let text = serde_json::from_str::<serde_json::Value>(content)
            .ok()
            .and_then(|value| {
                value
                    .get("text")
                    .and_then(|text| text.as_str())
                    .map(str::to_string)
            })
            .unwrap_or_default();
        if text.trim().is_empty() {
            return Ok(serde_json::json!({ "code": 0 }));
        }

        let sender_id = payload
            .pointer("/event/sender/sender_id/open_id")
            .and_then(|value| value.as_str())
            .or_else(|| {
                payload
                    .pointer("/event/sender/sender_id/user_id")
                    .and_then(|value| value.as_str())
            })
            .unwrap_or_default();
        let chat_id = payload
            .pointer("/event/message/chat_id")
            .and_then(|value| value.as_str())
            .context("Feishu chat_id missing")?;

        let (actor_user_id, actor_chat_id) = self.resolve_actor_ids(sender_id, chat_id);
        let reply = self
            .dispatch_command(&text, actor_user_id, actor_chat_id)
            .await;
        self.send_message(chat_id, &render_feishu_text(&reply))
            .await?;
        Ok(serde_json::json!({ "code": 0 }))
    }

    async fn dispatch_command(
        &self,
        text: &str,
        actor_user_id: i64,
        actor_chat_id: i64,
    ) -> BotReply {
        let current_session_id = self.current_session.lock().await.clone();
        match self
            .gateway
            .handle(GatewayRequest {
                actor_user_id,
                actor_chat_id,
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
                text: render_user_error(&err),
                keyboard: None,
            },
        }
    }

    async fn send_message(&self, chat_id: &str, text: &str) -> Result<()> {
        let token = self.tenant_access_token().await?;
        let content = serde_json::to_string(&serde_json::json!({
            "text": clamp_text(text),
        }))
        .context("serialize Feishu text payload")?;
        let response = self
            .client
            .post("https://open.feishu.cn/open-apis/im/v1/messages?receive_id_type=chat_id")
            .bearer_auth(token)
            .json(&serde_json::json!({
                "receive_id": chat_id,
                "msg_type": "text",
                "content": content,
            }))
            .send()
            .await
            .context("send Feishu message request")?;
        let payload: FeishuEnvelope<serde_json::Value> =
            response.json().await.context("decode Feishu sendMessage")?;
        if payload.code != 0 {
            anyhow::bail!(
                "Feishu sendMessage failed: code={} msg={}",
                payload.code,
                payload.msg.unwrap_or_else(|| "unknown".into())
            );
        }
        Ok(())
    }

    async fn tenant_access_token(&self) -> Result<String> {
        let mut cache = self.token_cache.lock().await;
        if let Some(token) = cache.as_ref() {
            if token.expires_at > Instant::now() + Duration::from_secs(30) {
                return Ok(token.value.clone());
            }
        }
        let response = self
            .client
            .post("https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal")
            .json(&serde_json::json!({
                "app_id": self.config.app_id,
                "app_secret": self.config.app_secret,
            }))
            .send()
            .await
            .context("request Feishu tenant_access_token")?;
        let payload: FeishuTokenResponse = response
            .json()
            .await
            .context("decode Feishu tenant_access_token")?;
        if payload.code != 0 {
            anyhow::bail!(
                "Feishu tenant_access_token failed: code={} msg={}",
                payload.code,
                payload.msg.unwrap_or_else(|| "unknown".into())
            );
        }
        let token = payload
            .tenant_access_token
            .context("Feishu tenant_access_token missing")?;
        let expires_in = payload.expire.unwrap_or(7200).max(60) as u64;
        *cache = Some(CachedTenantToken {
            value: token.clone(),
            expires_at: Instant::now() + Duration::from_secs(expires_in.saturating_sub(60)),
        });
        Ok(token)
    }

    fn resolve_actor_ids(&self, raw_user_id: &str, raw_chat_id: &str) -> (i64, i64) {
        let user_id = if self.config.admin_user_id == 0 {
            0
        } else {
            stable_channel_id(raw_user_id)
        };
        let chat_id = if self.config.admin_chat_id == 0 {
            0
        } else {
            stable_channel_id(raw_chat_id)
        };
        (user_id, chat_id)
    }
}

async fn feishu_event_handler(
    State(bot): State<Arc<FeishuBot>>,
    Json(payload): Json<serde_json::Value>,
) -> Json<serde_json::Value> {
    match bot.handle_event(payload).await {
        Ok(response) => Json(response),
        Err(err) => {
            error!(error = %err, "failed to handle Feishu event");
            Json(serde_json::json!({
                "code": 0,
            }))
        }
    }
}

#[derive(Debug, Deserialize)]
struct TelegramEnvelope<T> {
    ok: bool,
    result: T,
}

#[derive(Debug, Deserialize)]
struct FeishuEnvelope<T> {
    code: i64,
    msg: Option<String>,
    #[serde(rename = "data")]
    _data: Option<T>,
}

#[derive(Debug, Deserialize)]
struct FeishuTokenResponse {
    code: i64,
    msg: Option<String>,
    tenant_access_token: Option<String>,
    expire: Option<i64>,
}

#[derive(Debug, Deserialize)]
struct Update {
    update_id: i64,
    message: Option<Message>,
    callback_query: Option<CallbackQuery>,
}

#[derive(Debug, Deserialize)]
struct Message {
    message_id: i64,
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

#[derive(Debug, Deserialize)]
struct TelegramMessagePayload {
    message_id: i64,
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

fn is_natural_language(text: &str) -> bool {
    !text.trim_start().starts_with('/')
}

fn render_user_error(err: &ClawError) -> String {
    match err {
        ClawError::AccessDenied(_) => {
            "This Telegram chat is not allowed to control hajimi.".to_string()
        }
        ClawError::ApprovalRequired(reason) => format!(
            "This action needs approval before it can run.\n{}",
            reason.trim()
        ),
        ClawError::InvalidRequest(message) if message.contains("no active session") => {
            "There is no active shell yet. Use /shell open first.".to_string()
        }
        ClawError::NotFound(message) if message.contains("no provider configured") => {
            "No provider is configured yet. Use /onboard or /provider add first.".to_string()
        }
        ClawError::NotFound(message) if message.contains("provider not found") => {
            "That provider does not exist. Use /provider list to pick one.".to_string()
        }
        ClawError::Backend(message) if message.contains("LLM backend not configured") => {
            "No active model is available yet. Open /provider current and choose a provider and model.".to_string()
        }
        ClawError::Backend(message)
            if message.contains("timed out")
                || message.contains("timeout")
                || message.contains("Connect")
                || message.contains("connection")
                || message.contains("tcp connect") =>
        {
            "The provider request timed out or the network is unreachable. Try again later or switch provider/model.".to_string()
        }
        ClawError::Backend(message) if message.contains("429") => {
            "The provider is rate-limiting this request. Try again later or switch provider/model.".to_string()
        }
        _ => format!("Request failed: {err}"),
    }
}

fn render_feishu_text(reply: &BotReply) -> String {
    let mut text = reply.text.clone();
    if let Some(keyboard) = &reply.keyboard {
        let actions = keyboard
            .rows
            .iter()
            .flat_map(|row| row.iter())
            .map(|button| format!("{} -> {}", button.text, button.data))
            .collect::<Vec<_>>();
        if !actions.is_empty() {
            text.push_str("\n\nActions:\n");
            text.push_str(&actions.join("\n"));
        }
    }
    text
}

fn stable_channel_id(value: &str) -> i64 {
    use std::hash::{Hash, Hasher};

    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    value.hash(&mut hasher);
    (hasher.finish() & (i64::MAX as u64)) as i64
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
    fn parses_elevated_on() {
        assert_eq!(
            parse_gateway_command("/elevated on"),
            GatewayCommand::ElevatedOn
        );
    }
}
