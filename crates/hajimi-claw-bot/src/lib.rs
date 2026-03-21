use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use axum::extract::State;
use axum::routing::post;
use axum::{Json, Router};
use futures::{SinkExt, StreamExt};
use hajimi_claw_gateway::{
    Gateway, GatewayPreview, GatewayRequest, InlineKeyboard, SessionDirective,
    parse_gateway_command,
};
use hajimi_claw_types::ClawError;
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;
use tokio_tungstenite::{connect_async, tungstenite::Message as WsMessage};
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
    pub card_callback_path: String,
    pub mode: String,
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
    ws_state: Mutex<FeishuWsState>,
}

#[derive(Debug, Clone)]
struct CachedTenantToken {
    value: String,
    expires_at: Instant,
}

#[derive(Debug, Clone, Default)]
struct FeishuWsState {
    service_id: i32,
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
            let actor_user_id = message
                .from
                .as_ref()
                .map(|user| user.id)
                .unwrap_or_default();
            let actor_chat_id = message.chat.id;
            if is_natural_language(&text) {
                self.send_chat_action(message.chat.id, "typing").await?;
                let preview = self
                    .preview_command(&text, actor_user_id, actor_chat_id)
                    .await;
                let placeholder_text = preview
                    .as_ref()
                    .map(|item| item.text.as_str())
                    .unwrap_or("Processing your request...");
                let placeholder_keyboard = preview.clone().and_then(|item| item.keyboard);
                let placeholder_id = self
                    .send_message(message.chat.id, placeholder_text, placeholder_keyboard)
                    .await?;
                let reply = self
                    .dispatch_command(&text, actor_user_id, actor_chat_id)
                    .await;
                self.edit_message(message.chat.id, placeholder_id, &reply.text, reply.keyboard)
                    .await?;
            } else {
                let reply = self
                    .dispatch_command(&text, actor_user_id, actor_chat_id)
                    .await;
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
                let reply = self
                    .dispatch_command(&data, callback_query.from.id, chat_id)
                    .await;
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

    async fn preview_command(
        &self,
        text: &str,
        actor_user_id: i64,
        actor_chat_id: i64,
    ) -> Option<GatewayPreview> {
        let current_session_id = self.current_session.lock().await.clone();
        self.gateway
            .preview(GatewayRequest {
                actor_user_id,
                actor_chat_id,
                raw_text: text.to_string(),
                command: parse_gateway_command(text),
                current_session_id,
            })
            .await
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
                    { "command": "agents", "description": "Manage multi-agent mode" },
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
        let chat_allowed = actor_matches(self.config.admin_chat_id, chat_id);
        let user_allowed = actor_matches(self.config.admin_user_id, user_id);
        chat_allowed && user_allowed
    }
}

fn actor_matches(configured: i64, actual: i64) -> bool {
    configured == 0 || configured == actual
}

impl FeishuBot {
    pub fn new(config: FeishuConfig, gateway: Arc<dyn Gateway>) -> Self {
        Self {
            client: reqwest::Client::new(),
            config,
            gateway,
            current_session: Mutex::new(None),
            token_cache: Mutex::new(None),
            ws_state: Mutex::new(FeishuWsState::default()),
        }
    }

    pub async fn run(self: Arc<Self>) -> Result<()> {
        let mode = self.config.mode.trim().to_ascii_lowercase();
        if mode == "webhook" {
            self.run_webhook_server().await
        } else {
            self.run_long_connection().await
        }
    }

    async fn run_webhook_server(self: Arc<Self>) -> Result<()> {
        let addr: SocketAddr = self.config.listen_addr.parse().with_context(|| {
            format!("parse Feishu listen address `{}`", self.config.listen_addr)
        })?;
        let app = Router::new()
            .route(&self.config.event_path, post(feishu_event_handler))
            .route(
                &self.config.card_callback_path,
                post(feishu_card_callback_handler),
            )
            .with_state(self.clone());
        let listener = tokio::net::TcpListener::bind(addr)
            .await
            .with_context(|| format!("bind Feishu listener {}", addr))?;
        info!(
            listen_addr = %addr,
            event_path = %self.config.event_path,
            card_callback_path = %self.config.card_callback_path,
            "listening for Feishu webhook events"
        );
        axum::serve(listener, app)
            .await
            .context("serve Feishu webhook")?;
        Ok(())
    }

    async fn run_long_connection(self: Arc<Self>) -> Result<()> {
        let webhook = self.clone();
        let webhook_task = if !self.config.card_callback_path.trim().is_empty() {
            Some(tokio::spawn(async move {
                webhook.run_card_callback_server().await
            }))
        } else {
            None
        };

        let ws_result = self.long_connection_loop().await;
        if let Some(task) = webhook_task {
            task.abort();
        }
        ws_result
    }

    async fn run_card_callback_server(self: Arc<Self>) -> Result<()> {
        let addr: SocketAddr = self.config.listen_addr.parse().with_context(|| {
            format!("parse Feishu listen address `{}`", self.config.listen_addr)
        })?;
        let app = Router::new()
            .route(
                &self.config.card_callback_path,
                post(feishu_card_callback_handler),
            )
            .with_state(self.clone());
        let listener = tokio::net::TcpListener::bind(addr)
            .await
            .with_context(|| format!("bind Feishu card callback listener {}", addr))?;
        info!(
            listen_addr = %addr,
            card_callback_path = %self.config.card_callback_path,
            "listening for Feishu card callbacks"
        );
        axum::serve(listener, app)
            .await
            .context("serve Feishu card callback webhook")?;
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
        self.process_message_event(payload).await?;
        Ok(serde_json::json!({ "code": 0 }))
    }

    async fn process_message_event(&self, payload: serde_json::Value) -> Result<()> {
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
            return Ok(());
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
        if is_natural_language(&text) {
            if let Some(preview) = self
                .preview_command(&text, actor_user_id, actor_chat_id)
                .await
            {
                self.send_reply(
                    chat_id,
                    &BotReply {
                        text: preview.text,
                        keyboard: preview.keyboard,
                    },
                )
                .await?;
            }
        }
        let reply = self
            .dispatch_command(&text, actor_user_id, actor_chat_id)
            .await;
        self.send_reply(chat_id, &reply).await?;
        Ok(())
    }

    async fn handle_card_callback(&self, payload: serde_json::Value) -> Result<serde_json::Value> {
        if payload
            .get("type")
            .and_then(|value| value.as_str())
            .is_some_and(|value| value == "url_verification")
        {
            let challenge = payload
                .get("challenge")
                .and_then(|value| value.as_str())
                .context("Feishu card callback challenge missing")?;
            return Ok(serde_json::json!({ "challenge": challenge }));
        }

        let action = payload
            .pointer("/action/value/command")
            .and_then(|value| value.as_str())
            .context("Feishu card callback command missing")?;
        let sender_id = payload
            .pointer("/open_id")
            .and_then(|value| value.as_str())
            .or_else(|| payload.pointer("/user_id").and_then(|value| value.as_str()))
            .unwrap_or_default();
        let chat_id = payload
            .pointer("/action/value/chat_id")
            .and_then(|value| value.as_str())
            .unwrap_or("card");
        let (actor_user_id, actor_chat_id) = self.resolve_actor_ids(sender_id, chat_id);
        let reply = self
            .dispatch_command(action, actor_user_id, actor_chat_id)
            .await;
        Ok(serde_json::json!({
            "toast": {
                "type": "info",
                "content": clamp_text("Updated"),
            },
            "card": build_feishu_card(&reply, Some(action), chat_id),
        }))
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

    async fn preview_command(
        &self,
        text: &str,
        actor_user_id: i64,
        actor_chat_id: i64,
    ) -> Option<GatewayPreview> {
        let current_session_id = self.current_session.lock().await.clone();
        self.gateway
            .preview(GatewayRequest {
                actor_user_id,
                actor_chat_id,
                raw_text: text.to_string(),
                command: parse_gateway_command(text),
                current_session_id,
            })
            .await
    }

    async fn long_connection_loop(&self) -> Result<()> {
        loop {
            let endpoint = self.fetch_ws_endpoint().await?;
            let service_id = endpoint.service_id;
            {
                let mut state = self.ws_state.lock().await;
                state.service_id = service_id;
            }
            info!(service_id, "connecting to Feishu long connection");
            let (stream, _) = connect_async(&endpoint.url)
                .await
                .context("connect to Feishu websocket")?;
            let (mut write, mut read) = stream.split();
            let mut heartbeat = tokio::time::interval(Duration::from_secs(50));

            loop {
                tokio::select! {
                    _ = heartbeat.tick() => {
                        let ping = FeishuWsFrame {
                            seq_id: 1,
                            log_id: 1,
                            service: service_id,
                            method: 0,
                            headers: vec![FeishuWsHeader {
                                key: "type".into(),
                                value: "ping".into(),
                            }],
                            payload_encoding: String::new(),
                            payload_type: String::new(),
                            payload: Vec::new(),
                            log_id_new: String::new(),
                        }.encode()?;
                        write
                            .send(WsMessage::Binary(ping.into()))
                            .await
                            .context("write Feishu websocket ping")?;
                    }
                    message = read.next() => {
                        let Some(message) = message else {
                            warn!("Feishu websocket stream ended; reconnecting");
                            break;
                        };
                        let message = message.context("read Feishu websocket message")?;
                        match message {
                            WsMessage::Binary(payload) => {
                                if let Some(response) = self.handle_ws_frame(payload.to_vec()).await? {
                                    write
                                        .send(WsMessage::Binary(response.into()))
                                        .await
                                        .context("write Feishu websocket frame")?;
                                }
                            }
                            WsMessage::Ping(payload) => {
                                write
                                    .send(WsMessage::Pong(payload))
                                    .await
                                    .context("reply Feishu websocket pong")?;
                            }
                            WsMessage::Close(frame) => {
                                warn!(?frame, "Feishu websocket closed; reconnecting");
                                break;
                            }
                            _ => {}
                        }
                    }
                }
            }
            tokio::time::sleep(Duration::from_secs(2)).await;
        }
    }

    async fn fetch_ws_endpoint(&self) -> Result<FeishuWsEndpoint> {
        let response = self
            .client
            .post("https://open.feishu.cn/callback/ws/endpoint")
            .header("locale", "zh")
            .json(&serde_json::json!({
                "AppID": self.config.app_id,
                "AppSecret": self.config.app_secret,
            }))
            .send()
            .await
            .context("request Feishu websocket endpoint")?;
        let payload: FeishuWsEndpointResponse = response
            .json()
            .await
            .context("decode Feishu websocket endpoint")?;
        if payload.code != 0 {
            anyhow::bail!(
                "Feishu websocket endpoint failed: code={} msg={}",
                payload.code,
                payload.msg.unwrap_or_else(|| "unknown".into())
            );
        }
        let url = payload
            .data
            .as_ref()
            .and_then(|data| data.url.clone())
            .context("Feishu websocket endpoint URL missing")?;
        let service_id = url::Url::parse(&url)
            .ok()
            .and_then(|parsed| {
                parsed
                    .query_pairs()
                    .find_map(|(key, value)| (key == "service_id").then_some(value.into_owned()))
            })
            .and_then(|value| value.parse::<i32>().ok())
            .unwrap_or_default();
        Ok(FeishuWsEndpoint { url, service_id })
    }

    async fn handle_ws_frame(&self, payload: Vec<u8>) -> Result<Option<Vec<u8>>> {
        let frame = FeishuWsFrame::decode(&payload)?;
        let message_type = frame.header("type").unwrap_or_default();
        if frame.method == 0 {
            return Ok(None);
        }

        if message_type == "card" {
            return Ok(Some(frame.response_payload(200, None)?));
        }

        if message_type != "event" {
            return Ok(Some(frame.response_payload(200, None)?));
        }

        let event: serde_json::Value =
            serde_json::from_slice(&frame.payload).context("decode Feishu event payload")?;
        self.process_message_event(event).await?;
        Ok(Some(frame.response_payload(200, None)?))
    }

    async fn send_reply(&self, chat_id: &str, reply: &BotReply) -> Result<()> {
        let token = self.tenant_access_token().await?;
        let (msg_type, content) = if reply.keyboard.is_some() {
            (
                "interactive",
                serde_json::to_string(&build_feishu_card(reply, None, chat_id))
                    .context("serialize Feishu card payload")?,
            )
        } else {
            (
                "text",
                serde_json::to_string(&serde_json::json!({
                    "text": clamp_text(&reply.text),
                }))
                .context("serialize Feishu text payload")?,
            )
        };
        let response = self
            .client
            .post("https://open.feishu.cn/open-apis/im/v1/messages?receive_id_type=chat_id")
            .bearer_auth(token)
            .json(&serde_json::json!({
                "receive_id": chat_id,
                "msg_type": msg_type,
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
            stable_channel_id(raw_user_id)
        } else {
            stable_channel_id(raw_user_id)
        };
        let chat_id = if self.config.admin_chat_id == 0 {
            stable_channel_id(raw_chat_id)
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

async fn feishu_card_callback_handler(
    State(bot): State<Arc<FeishuBot>>,
    Json(payload): Json<serde_json::Value>,
) -> Json<serde_json::Value> {
    match bot.handle_card_callback(payload).await {
        Ok(response) => Json(response),
        Err(err) => {
            error!(error = %err, "failed to handle Feishu card callback");
            Json(serde_json::json!({
                "toast": {
                    "type": "error",
                    "content": clamp_text(&format!("Request failed: {err}")),
                }
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
        ClawError::AccessDenied(message) if message.contains("telegram actor is not authorized") => {
            "This Telegram chat is not allowed to control hajimi.".to_string()
        }
        ClawError::AccessDenied(message) => {
            format!("Access denied.\n{}", message.trim())
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

fn build_feishu_card(
    reply: &BotReply,
    current_command: Option<&str>,
    chat_id: &str,
) -> serde_json::Value {
    let mut elements = vec![serde_json::json!({
        "tag": "markdown",
        "content": clamp_text(&reply.text),
    })];

    if let Some(keyboard) = &reply.keyboard {
        let mut rows = Vec::new();
        for row in &keyboard.rows {
            let actions = row
                .iter()
                .map(|button| {
                    serde_json::json!({
                        "tag": "button",
                        "type": if current_command.is_some_and(|command| command == button.data) {
                            "primary"
                        } else {
                            "default"
                        },
                        "text": {
                            "tag": "plain_text",
                            "content": button.text,
                        },
                        "value": {
                            "command": button.data,
                            "chat_id": chat_id,
                        }
                    })
                })
                .collect::<Vec<_>>();
            rows.push(serde_json::json!({
                "tag": "action",
                "actions": actions,
            }));
        }
        elements.extend(rows);
    }

    serde_json::json!({
        "config": {
            "wide_screen_mode": true,
        },
        "header": {
            "title": {
                "tag": "plain_text",
                "content": "hajimi"
            },
            "template": "blue"
        },
        "elements": elements,
    })
}

#[derive(Debug, Deserialize)]
struct FeishuWsEndpointResponse {
    code: i64,
    msg: Option<String>,
    data: Option<FeishuWsEndpointData>,
}

#[derive(Debug, Deserialize)]
struct FeishuWsEndpointData {
    #[serde(rename = "URL")]
    url: Option<String>,
}

#[derive(Debug)]
struct FeishuWsEndpoint {
    url: String,
    service_id: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct FeishuWsHeader {
    key: String,
    value: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct FeishuWsResponsePayload {
    code: i32,
    #[serde(default)]
    headers: std::collections::BTreeMap<String, String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    data: Option<serde_json::Value>,
}

#[derive(Debug, Clone)]
struct FeishuWsFrame {
    seq_id: u64,
    log_id: u64,
    service: i32,
    method: i32,
    headers: Vec<FeishuWsHeader>,
    payload_encoding: String,
    payload_type: String,
    payload: Vec<u8>,
    log_id_new: String,
}

impl FeishuWsFrame {
    fn decode(bytes: &[u8]) -> Result<Self> {
        let mut index = 0;
        let mut frame = Self {
            seq_id: 0,
            log_id: 0,
            service: 0,
            method: 0,
            headers: Vec::new(),
            payload_encoding: String::new(),
            payload_type: String::new(),
            payload: Vec::new(),
            log_id_new: String::new(),
        };
        while index < bytes.len() {
            let tag = read_varint(bytes, &mut index)?;
            let field = tag >> 3;
            let wire = tag & 0x7;
            match (field, wire) {
                (1, 0) => frame.seq_id = read_varint(bytes, &mut index)?,
                (2, 0) => frame.log_id = read_varint(bytes, &mut index)?,
                (3, 0) => frame.service = read_varint(bytes, &mut index)? as i32,
                (4, 0) => frame.method = read_varint(bytes, &mut index)? as i32,
                (5, 2) => {
                    let len = read_varint(bytes, &mut index)? as usize;
                    let end = index + len;
                    let mut inner = index;
                    let mut key = String::new();
                    let mut value = String::new();
                    while inner < end {
                        let inner_tag = read_varint(bytes, &mut inner)?;
                        match (inner_tag >> 3, inner_tag & 0x7) {
                            (1, 2) => key = read_string(bytes, &mut inner)?,
                            (2, 2) => value = read_string(bytes, &mut inner)?,
                            _ => skip_wire(bytes, &mut inner, inner_tag & 0x7)?,
                        }
                    }
                    index = end;
                    frame.headers.push(FeishuWsHeader { key, value });
                }
                (6, 2) => frame.payload_encoding = read_string(bytes, &mut index)?,
                (7, 2) => frame.payload_type = read_string(bytes, &mut index)?,
                (8, 2) => frame.payload = read_bytes(bytes, &mut index)?,
                (9, 2) => frame.log_id_new = read_string(bytes, &mut index)?,
                _ => skip_wire(bytes, &mut index, wire)?,
            }
        }
        Ok(frame)
    }

    fn response_payload(&self, code: i32, data: Option<serde_json::Value>) -> Result<Vec<u8>> {
        let mut headers = self.headers.clone();
        headers.push(FeishuWsHeader {
            key: "biz_rt".into(),
            value: "0".into(),
        });
        let payload = serde_json::to_vec(&FeishuWsResponsePayload {
            code,
            headers: std::collections::BTreeMap::new(),
            data,
        })?;
        FeishuWsFrame {
            seq_id: self.seq_id,
            log_id: self.log_id,
            service: self.service,
            method: self.method,
            headers,
            payload_encoding: self.payload_encoding.clone(),
            payload_type: self.payload_type.clone(),
            payload,
            log_id_new: self.log_id_new.clone(),
        }
        .encode()
    }

    fn encode(&self) -> Result<Vec<u8>> {
        let mut out = Vec::new();
        write_varint_field(&mut out, 1, self.seq_id);
        write_varint_field(&mut out, 2, self.log_id);
        write_varint_field(&mut out, 3, self.service as u64);
        write_varint_field(&mut out, 4, self.method as u64);
        for header in &self.headers {
            let mut nested = Vec::new();
            write_len_field(&mut nested, 1, header.key.as_bytes());
            write_len_field(&mut nested, 2, header.value.as_bytes());
            write_len_field(&mut out, 5, &nested);
        }
        write_len_field(&mut out, 6, self.payload_encoding.as_bytes());
        write_len_field(&mut out, 7, self.payload_type.as_bytes());
        if !self.payload.is_empty() {
            write_len_field(&mut out, 8, &self.payload);
        }
        if !self.log_id_new.is_empty() {
            write_len_field(&mut out, 9, self.log_id_new.as_bytes());
        }
        Ok(out)
    }

    fn header(&self, key: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|header| header.key == key)
            .map(|header| header.value.as_str())
    }
}

fn stable_channel_id(value: &str) -> i64 {
    use std::hash::{Hash, Hasher};

    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    value.hash(&mut hasher);
    (hasher.finish() & (i64::MAX as u64)) as i64
}

fn read_varint(bytes: &[u8], index: &mut usize) -> Result<u64> {
    let mut shift = 0;
    let mut value = 0_u64;
    loop {
        if *index >= bytes.len() {
            anyhow::bail!("unexpected eof while decoding varint");
        }
        let byte = bytes[*index];
        *index += 1;
        value |= ((byte & 0x7f) as u64) << shift;
        if byte < 0x80 {
            return Ok(value);
        }
        shift += 7;
        if shift >= 64 {
            anyhow::bail!("varint overflow");
        }
    }
}

fn read_bytes(bytes: &[u8], index: &mut usize) -> Result<Vec<u8>> {
    let len = read_varint(bytes, index)? as usize;
    if *index + len > bytes.len() {
        anyhow::bail!("unexpected eof while decoding bytes");
    }
    let value = bytes[*index..*index + len].to_vec();
    *index += len;
    Ok(value)
}

fn read_string(bytes: &[u8], index: &mut usize) -> Result<String> {
    String::from_utf8(read_bytes(bytes, index)?).context("decode utf-8 string")
}

fn skip_wire(bytes: &[u8], index: &mut usize, wire: u64) -> Result<()> {
    match wire {
        0 => {
            let _ = read_varint(bytes, index)?;
        }
        2 => {
            let len = read_varint(bytes, index)? as usize;
            if *index + len > bytes.len() {
                anyhow::bail!("unexpected eof while skipping bytes");
            }
            *index += len;
        }
        _ => anyhow::bail!("unsupported protobuf wire type {wire}"),
    }
    Ok(())
}

fn write_varint_field(out: &mut Vec<u8>, field: u64, value: u64) {
    write_varint(out, field << 3);
    write_varint(out, value);
}

fn write_len_field(out: &mut Vec<u8>, field: u64, value: &[u8]) {
    write_varint(out, (field << 3) | 2);
    write_varint(out, value.len() as u64);
    out.extend_from_slice(value);
}

fn write_varint(out: &mut Vec<u8>, mut value: u64) {
    while value >= 0x80 {
        out.push(((value as u8) & 0x7f) | 0x80);
        value >>= 7;
    }
    out.push(value as u8);
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
    use crate::{actor_matches, render_user_error};
    use hajimi_claw_types::ClawError;

    use hajimi_claw_gateway::{GatewayCommand, parse_gateway_command};

    #[test]
    fn parses_elevated_on() {
        assert_eq!(
            parse_gateway_command("/elevated on"),
            GatewayCommand::ElevatedOn
        );
    }

    #[test]
    fn access_denied_renders_actual_reason_for_non_auth_failures() {
        let rendered = render_user_error(&ClawError::AccessDenied(
            "working directory is not allowed: C:/Windows/System32".into(),
        ));
        assert!(rendered.contains("Access denied."));
        assert!(rendered.contains("working directory is not allowed"));
    }

    #[test]
    fn zero_actor_config_is_treated_as_wildcard() {
        assert!(actor_matches(0, 123));
        assert!(actor_matches(456, 456));
        assert!(!actor_matches(456, 123));
    }
}
