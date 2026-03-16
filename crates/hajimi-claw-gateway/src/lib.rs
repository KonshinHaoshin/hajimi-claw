use std::sync::Arc;

use async_trait::async_trait;
use chrono::Utc;
use hajimi_claw_agent::AgentRuntime;
use hajimi_claw_llm::{list_models, test_provider};
use hajimi_claw_policy::PolicyEngine;
use hajimi_claw_store::Store;
use hajimi_claw_types::{
    ClawError, ClawResult, OnboardingSession, OnboardingStep, ProviderConfig, ProviderDraft,
    ProviderKind, ProviderRecord,
};
use reqwest::Client;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum GatewayCommand {
    Ask(String),
    ShellOpen(Option<String>),
    ShellExec(String),
    ShellClose,
    Status,
    Approve(String),
    ElevatedOn,
    ElevatedOff,
    ElevatedAsk,
    ElevatedFull,
    Cancel(String),
    Onboard,
    OnboardCancel,
    ProviderList,
    ProviderUse(String),
    ProviderBind(String),
    ProviderCurrent,
    ProviderTest(Option<String>),
    ProviderModels(Option<String>),
    ProviderAdd,
    ProviderSetModel { provider_id: String, model: String },
    ModelCurrent,
    ModelPicker,
    ModelUse(String),
    PersonaList,
    PersonaRead(String),
    PersonaWrite { file: String, content: String },
    PersonaAppend { file: String, content: String },
    Help,
    Menu,
    Unknown(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GatewayRequest {
    pub actor_user_id: i64,
    pub actor_chat_id: i64,
    pub raw_text: String,
    pub command: GatewayCommand,
    pub current_session_id: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SessionDirective {
    Keep,
    Set(String),
    Clear,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GatewayResponse {
    pub text: String,
    pub session: SessionDirective,
    pub keyboard: Option<InlineKeyboard>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InlineKeyboard {
    pub rows: Vec<Vec<InlineButton>>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InlineButton {
    pub text: String,
    pub data: String,
}

#[async_trait]
pub trait Gateway: Send + Sync {
    async fn handle(&self, request: GatewayRequest) -> ClawResult<GatewayResponse>;
}

pub struct InProcessGateway {
    runtime: Arc<AgentRuntime>,
    policy: Arc<PolicyEngine>,
    store: Arc<Store>,
    client: Client,
}

impl InProcessGateway {
    pub fn new(runtime: Arc<AgentRuntime>, policy: Arc<PolicyEngine>, store: Arc<Store>) -> Self {
        Self {
            runtime,
            policy,
            store,
            client: Client::new(),
        }
    }
}

#[async_trait]
impl Gateway for InProcessGateway {
    async fn handle(&self, request: GatewayRequest) -> ClawResult<GatewayResponse> {
        if !self
            .policy
            .authorize_telegram_actor(request.actor_user_id, request.actor_chat_id)
        {
            return Err(ClawError::AccessDenied(
                "telegram actor is not authorized".into(),
            ));
        }

        if matches!(request.command, GatewayCommand::OnboardCancel) {
            self.store
                .clear_onboarding_session(request.actor_chat_id, request.actor_user_id)
                .map_err(store_error)?;
            return Ok(text_response("onboarding cancelled"));
        }

        if let Some(session) = self
            .store
            .load_onboarding_session(request.actor_chat_id, request.actor_user_id)
            .map_err(store_error)?
        {
            if !request.raw_text.trim_start().starts_with('/') {
                return self.continue_onboarding(session, request).await;
            }
        }

        match request.command {
            GatewayCommand::Ask(prompt) => {
                let provider_id = self
                    .store
                    .resolve_provider_for_chat(request.actor_chat_id)
                    .map_err(store_error)?
                    .map(|record| record.config.id);
                Ok(GatewayResponse {
                    text: self
                        .runtime
                        .ask_with_provider(&prompt, None, provider_id)
                        .await?,
                    session: SessionDirective::Keep,
                    keyboard: None,
                })
            }
            GatewayCommand::ShellOpen(name) => {
                let reply = self.runtime.shell_open(name, None).await?;
                Ok(GatewayResponse {
                    text: reply.message,
                    session: SessionDirective::Set(reply.session_id),
                    keyboard: None,
                })
            }
            GatewayCommand::ShellExec(command) => {
                let session_id = request.current_session_id.ok_or_else(|| {
                    ClawError::InvalidRequest("no active session, use /shell open first".into())
                })?;
                Ok(GatewayResponse {
                    text: self.runtime.shell_exec(&session_id, &command).await?,
                    session: SessionDirective::Keep,
                    keyboard: None,
                })
            }
            GatewayCommand::ShellClose => {
                let session_id = request.current_session_id.ok_or_else(|| {
                    ClawError::InvalidRequest("no active session, use /shell open first".into())
                })?;
                Ok(GatewayResponse {
                    text: self.runtime.shell_close(&session_id).await?,
                    session: SessionDirective::Clear,
                    keyboard: None,
                })
            }
            GatewayCommand::Status => Ok(text_response(&self.runtime.status()?)),
            GatewayCommand::Approve(request_id) => {
                Ok(text_response(&self.runtime.approve(&request_id)?))
            }
            GatewayCommand::ElevatedOn => Ok(text_response(&self.runtime.enable_elevated())),
            GatewayCommand::ElevatedOff => Ok(text_response(&self.runtime.stop_elevated())),
            GatewayCommand::ElevatedAsk => Ok(text_response(&self.runtime.enable_approval_mode())),
            GatewayCommand::ElevatedFull => Ok(text_response(&self.runtime.enable_full_elevated())),
            GatewayCommand::Cancel(task_id) => Ok(text_response(&format!(
                "cancel is not implemented yet for task {task_id}"
            ))),
            GatewayCommand::Onboard => self.start_onboarding(request).await,
            GatewayCommand::ProviderAdd => self.start_onboarding(request).await,
            GatewayCommand::ProviderList => self.provider_list().await,
            GatewayCommand::ProviderUse(provider_id) => {
                self.provider_use(request.actor_chat_id, &provider_id).await
            }
            GatewayCommand::ProviderBind(provider_id) => {
                self.provider_bind(request.actor_chat_id, &provider_id)
                    .await
            }
            GatewayCommand::ProviderCurrent => self.provider_current(request.actor_chat_id).await,
            GatewayCommand::ProviderSetModel { provider_id, model } => {
                self.provider_set_model(&provider_id, &model).await
            }
            GatewayCommand::ProviderTest(provider_id) => {
                self.provider_test(request.actor_chat_id, provider_id.as_deref())
                    .await
            }
            GatewayCommand::ProviderModels(provider_id) => {
                self.provider_models(request.actor_chat_id, provider_id.as_deref())
                    .await
            }
            GatewayCommand::ModelCurrent => self.model_current(request.actor_chat_id).await,
            GatewayCommand::ModelPicker => self.model_picker(request.actor_chat_id).await,
            GatewayCommand::ModelUse(model) => self.model_use(request.actor_chat_id, &model).await,
            GatewayCommand::PersonaList => Ok(text_response(&self.runtime.persona_list().await?)),
            GatewayCommand::PersonaRead(file) => {
                Ok(text_response(&self.runtime.persona_read(&file).await?))
            }
            GatewayCommand::PersonaWrite { file, content } => Ok(text_response(
                &self.runtime.persona_write(&file, &content).await?,
            )),
            GatewayCommand::PersonaAppend { file, content } => Ok(text_response(
                &self.runtime.persona_append(&file, &content).await?,
            )),
            GatewayCommand::Help | GatewayCommand::Menu => Ok(text_response_with_keyboard(
                &help_text(),
                Some(main_menu_keyboard()),
            )),
            GatewayCommand::Unknown(raw) => Ok(text_response(&format!(
                "unrecognized command: {raw}\n\n{}",
                help_text()
            ))),
            GatewayCommand::OnboardCancel => Ok(text_response("onboarding cancelled")),
        }
    }
}

impl InProcessGateway {
    async fn start_onboarding(&self, request: GatewayRequest) -> ClawResult<GatewayResponse> {
        let session = OnboardingSession {
            user_id: request.actor_user_id,
            chat_id: request.actor_chat_id,
            step: OnboardingStep::ProviderLabel,
            draft: ProviderDraft::default(),
            updated_at: Utc::now(),
        };
        self.store
            .save_onboarding_session(&session)
            .map_err(store_error)?;
        Ok(text_response_with_keyboard(
            "hajimi onboard started.\nStep 1/5: send a short provider label, for example `OpenAI` or `Moonshot`.\nSend `/onboard cancel` to stop.",
            Some(cancel_keyboard()),
        ))
    }

    async fn continue_onboarding(
        &self,
        mut session: OnboardingSession,
        request: GatewayRequest,
    ) -> ClawResult<GatewayResponse> {
        let input = request.raw_text.trim();
        match session.step {
            OnboardingStep::ProviderLabel => {
                session.draft.label = Some(input.to_string());
                session.step = OnboardingStep::ProviderKind;
                session.updated_at = Utc::now();
                self.store
                    .save_onboarding_session(&session)
                    .map_err(store_error)?;
                Ok(text_response_with_keyboard(
                    "Step 2/5: choose provider kind below or send it manually: `openai-compatible` or `custom-chat-completions`.",
                    Some(provider_kind_keyboard()),
                ))
            }
            OnboardingStep::ProviderKind => {
                session.draft.kind = Some(parse_provider_kind(input)?);
                session.step = OnboardingStep::ProviderBaseUrl;
                session.updated_at = Utc::now();
                self.store
                    .save_onboarding_session(&session)
                    .map_err(store_error)?;
                Ok(text_response(
                    "Step 3/5: send the API base URL, for example `https://api.openai.com/v1`.",
                ))
            }
            OnboardingStep::ProviderBaseUrl => {
                session.draft.base_url = Some(normalize_base_url(input));
                session.step = OnboardingStep::ProviderApiKey;
                session.updated_at = Utc::now();
                self.store
                    .save_onboarding_session(&session)
                    .map_err(store_error)?;
                Ok(text_response(
                    "Step 4/5: send the API key. It will be encrypted before being stored in SQLite.",
                ))
            }
            OnboardingStep::ProviderApiKey => {
                session.draft.api_key = Some(input.to_string());
                session.step = OnboardingStep::ProviderModel;
                session.updated_at = Utc::now();
                self.store
                    .save_onboarding_session(&session)
                    .map_err(store_error)?;
                Ok(text_response(
                    "Step 5/5: send the default model name, for example `gpt-4.1-mini`.",
                ))
            }
            OnboardingStep::ProviderModel => {
                session.draft.model = Some(input.to_string());
                let record = finalize_provider(session.draft.clone())?;
                let make_default = self
                    .store
                    .get_default_provider()
                    .map_err(store_error)?
                    .is_none();
                self.store
                    .upsert_provider(&ProviderRecord {
                        config: record.clone(),
                        is_default: make_default,
                    })
                    .map_err(store_error)?;
                self.store
                    .bind_provider_to_chat(request.actor_chat_id, &record.id)
                    .map_err(store_error)?;
                self.store
                    .clear_onboarding_session(request.actor_chat_id, request.actor_user_id)
                    .map_err(store_error)?;

                let health = test_provider(&self.client, &record).await?;
                Ok(text_response(&format!(
                    "onboarding complete.\nprovider=`{}` id=`{}`{}\nchat_binding=enabled\nhealth={}\n{}",
                    record.label,
                    record.id,
                    if make_default { " default=yes" } else { "" },
                    if health.ok { "ok" } else { "failed" },
                    health.message
                )))
            }
            OnboardingStep::Completed => Ok(text_response("onboarding already completed")),
        }
    }

    async fn provider_list(&self) -> ClawResult<GatewayResponse> {
        let providers = self.store.list_providers().map_err(store_error)?;
        if providers.is_empty() {
            return Ok(text_response(
                "no providers configured. Use `/onboard` to add one.",
            ));
        }
        let text = providers
            .iter()
            .map(|provider| {
                format!(
                    "`{}` {} kind={} model={}{}",
                    provider.config.id,
                    provider.config.label,
                    provider.config.kind.as_str(),
                    provider.config.model,
                    if provider.is_default { " default" } else { "" }
                )
            })
            .collect::<Vec<_>>()
            .join("\n");
        Ok(text_response_with_keyboard(
            &text,
            Some(provider_list_keyboard(&providers)),
        ))
    }

    async fn provider_use(&self, chat_id: i64, provider_id: &str) -> ClawResult<GatewayResponse> {
        ensure_provider_exists(&self.store, provider_id)?;
        self.store
            .set_default_provider(provider_id)
            .map_err(store_error)?;
        self.provider_current(chat_id).await
    }

    async fn provider_bind(&self, chat_id: i64, provider_id: &str) -> ClawResult<GatewayResponse> {
        ensure_provider_exists(&self.store, provider_id)?;
        self.store
            .bind_provider_to_chat(chat_id, provider_id)
            .map_err(store_error)?;
        self.provider_current(chat_id).await
    }

    async fn provider_current(&self, chat_id: i64) -> ClawResult<GatewayResponse> {
        let bound = self
            .store
            .get_bound_provider_id(chat_id)
            .map_err(store_error)?;
        let resolved = self
            .store
            .resolve_provider_for_chat(chat_id)
            .map_err(store_error)?;
        match resolved {
            Some(provider) => Ok(text_response_with_keyboard(
                &format!(
                    "current provider for chat `{chat_id}`: `{}` ({}){}\nmodel={}\ndefault={}",
                    provider.config.id,
                    provider.config.label,
                    if bound.is_some() {
                        " bound=yes"
                    } else {
                        " bound=no"
                    },
                    provider.config.model,
                    provider.is_default
                ),
                Some(provider_current_keyboard(&provider.config.id)),
            )),
            None => Ok(text_response("no provider configured")),
        }
    }

    async fn provider_set_model(
        &self,
        provider_id: &str,
        model: &str,
    ) -> ClawResult<GatewayResponse> {
        ensure_provider_exists(&self.store, provider_id)?;
        self.store
            .update_provider_model(provider_id, model)
            .map_err(store_error)?;
        let provider = self
            .store
            .get_provider(provider_id)
            .map_err(store_error)?
            .ok_or_else(|| ClawError::NotFound(format!("provider not found: {provider_id}")))?;
        Ok(text_response_with_keyboard(
            &format!(
                "provider `{}` now uses model `{}`",
                provider_id, provider.config.model
            ),
            Some(model_picker_keyboard(
                &provider.config.id,
                &provider.config.model,
                &[],
            )),
        ))
    }

    async fn provider_test(
        &self,
        chat_id: i64,
        provider_id: Option<&str>,
    ) -> ClawResult<GatewayResponse> {
        let provider = if let Some(provider_id) = provider_id {
            self.store
                .get_provider(provider_id)
                .map_err(store_error)?
                .ok_or_else(|| ClawError::NotFound(format!("provider not found: {provider_id}")))?
        } else {
            self.store
                .resolve_provider_for_chat(chat_id)
                .map_err(store_error)?
                .ok_or_else(|| ClawError::NotFound("no provider configured".into()))?
        };
        let health = test_provider(&self.client, &provider.config).await?;
        let models = if health.suggested_models.is_empty() {
            String::from("none")
        } else {
            health.suggested_models.join(", ")
        };
        Ok(text_response(&format!(
            "provider `{}` test={}\n{}\nmodels={}",
            provider.config.id,
            if health.ok { "ok" } else { "failed" },
            health.message,
            models
        )))
    }

    async fn provider_models(
        &self,
        chat_id: i64,
        provider_id: Option<&str>,
    ) -> ClawResult<GatewayResponse> {
        let provider = if let Some(provider_id) = provider_id {
            self.store
                .get_provider(provider_id)
                .map_err(store_error)?
                .ok_or_else(|| ClawError::NotFound(format!("provider not found: {provider_id}")))?
        } else {
            self.store
                .resolve_provider_for_chat(chat_id)
                .map_err(store_error)?
                .ok_or_else(|| ClawError::NotFound("no provider configured".into()))?
        };
        let models = list_models(&self.client, &provider.config).await?;
        if models.is_empty() {
            return Ok(text_response("provider returned no models"));
        }
        Ok(text_response_with_keyboard(
            &format!(
                "models for `{}`:\n{}",
                provider.config.id,
                models.join("\n")
            ),
            Some(model_picker_keyboard(
                &provider.config.id,
                &provider.config.model,
                &models,
            )),
        ))
    }

    async fn model_current(&self, chat_id: i64) -> ClawResult<GatewayResponse> {
        let provider = self
            .store
            .resolve_provider_for_chat(chat_id)
            .map_err(store_error)?
            .ok_or_else(|| ClawError::NotFound("no provider configured".into()))?;
        Ok(text_response_with_keyboard(
            &format!(
                "current provider=`{}`\ncurrent model=`{}`",
                provider.config.id, provider.config.model
            ),
            Some(model_picker_keyboard(
                &provider.config.id,
                &provider.config.model,
                &[],
            )),
        ))
    }

    async fn model_picker(&self, chat_id: i64) -> ClawResult<GatewayResponse> {
        self.provider_models(chat_id, None).await
    }

    async fn model_use(&self, chat_id: i64, model: &str) -> ClawResult<GatewayResponse> {
        let provider = self
            .store
            .resolve_provider_for_chat(chat_id)
            .map_err(store_error)?
            .ok_or_else(|| ClawError::NotFound("no provider configured".into()))?;
        self.store
            .update_provider_model(&provider.config.id, model)
            .map_err(store_error)?;
        self.model_current(chat_id).await
    }
}

pub fn parse_gateway_command(text: &str) -> GatewayCommand {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return GatewayCommand::Help;
    }
    if let Some(rest) = trimmed.strip_prefix("/ask ") {
        return GatewayCommand::Ask(rest.trim().into());
    }
    if trimmed == "/status" {
        return GatewayCommand::Status;
    }
    if trimmed == "/menu" {
        return GatewayCommand::Menu;
    }
    if trimmed == "/onboard" {
        return GatewayCommand::Onboard;
    }
    if trimmed == "/onboard cancel" {
        return GatewayCommand::OnboardCancel;
    }
    if trimmed == "/provider list" {
        return GatewayCommand::ProviderList;
    }
    if trimmed == "/provider add" {
        return GatewayCommand::ProviderAdd;
    }
    if trimmed == "/provider current" {
        return GatewayCommand::ProviderCurrent;
    }
    if trimmed == "/model current" {
        return GatewayCommand::ModelCurrent;
    }
    if trimmed == "/model use" {
        return GatewayCommand::ModelPicker;
    }
    if let Some(rest) = trimmed.strip_prefix("/model use ") {
        return GatewayCommand::ModelUse(rest.trim().into());
    }
    if trimmed == "/persona list" {
        return GatewayCommand::PersonaList;
    }
    if let Some(rest) = trimmed.strip_prefix("/persona read ") {
        return GatewayCommand::PersonaRead(rest.trim().into());
    }
    if let Some(rest) = trimmed.strip_prefix("/persona write ") {
        if let Some((file, content)) = split_file_and_content(rest.trim()) {
            return GatewayCommand::PersonaWrite { file, content };
        }
    }
    if let Some(rest) = trimmed.strip_prefix("/persona append ") {
        if let Some((file, content)) = split_file_and_content(rest.trim()) {
            return GatewayCommand::PersonaAppend { file, content };
        }
    }
    if let Some(rest) = trimmed.strip_prefix("/provider models") {
        let value = rest.trim();
        return GatewayCommand::ProviderModels((!value.is_empty()).then(|| value.to_string()));
    }
    if let Some(rest) = trimmed.strip_prefix("/provider set-model ") {
        if let Some((provider_id, model)) = split_file_and_content(rest.trim()) {
            return GatewayCommand::ProviderSetModel { provider_id, model };
        }
    }
    if let Some(rest) = trimmed.strip_prefix("/provider use ") {
        return GatewayCommand::ProviderUse(rest.trim().into());
    }
    if let Some(rest) = trimmed.strip_prefix("/provider bind ") {
        return GatewayCommand::ProviderBind(rest.trim().into());
    }
    if let Some(rest) = trimmed.strip_prefix("/provider test") {
        let value = rest.trim();
        return GatewayCommand::ProviderTest((!value.is_empty()).then(|| value.to_string()));
    }
    if let Some(rest) = trimmed.strip_prefix("/approve ") {
        return GatewayCommand::Approve(rest.trim().into());
    }
    if trimmed == "/elevated on" {
        return GatewayCommand::ElevatedOn;
    }
    if trimmed == "/elevated off" {
        return GatewayCommand::ElevatedOff;
    }
    if trimmed == "/elevated ask" {
        return GatewayCommand::ElevatedAsk;
    }
    if trimmed == "/elevated full" {
        return GatewayCommand::ElevatedFull;
    }
    if trimmed == "/shell close" {
        return GatewayCommand::ShellClose;
    }
    if let Some(rest) = trimmed.strip_prefix("/shell open") {
        let name = rest.trim();
        return GatewayCommand::ShellOpen((!name.is_empty()).then(|| name.to_string()));
    }
    if let Some(rest) = trimmed.strip_prefix("/shell exec ") {
        return GatewayCommand::ShellExec(rest.trim().into());
    }
    if let Some(rest) = trimmed.strip_prefix("/cancel ") {
        return GatewayCommand::Cancel(rest.trim().into());
    }
    if trimmed == "/help" || trimmed == "/start" {
        return GatewayCommand::Help;
    }
    if !trimmed.starts_with('/') {
        return GatewayCommand::Ask(trimmed.into());
    }
    GatewayCommand::Unknown(trimmed.into())
}

pub fn help_text() -> String {
    [
        "/ask <text>",
        "/onboard",
        "/onboard cancel",
        "/provider list",
        "/provider add",
        "/provider current",
        "/provider use <id>",
        "/provider bind <id>",
        "/provider test [id]",
        "/provider models [id]",
        "/provider set-model <provider-id> <model>",
        "/model current",
        "/model use [model]",
        "/persona list",
        "/persona read <soul|agents|tools|skills>",
        "/persona write <file> <content>",
        "/persona append <file> <content>",
        "/shell open [name]",
        "/shell exec <cmd>",
        "/shell close",
        "/status",
        "/menu",
        "/approve <request-id>",
        "/elevated on",
        "/elevated off",
        "/elevated ask",
        "/elevated full",
        "/cancel <task-id>",
        "plain text = natural-language task",
    ]
    .join("\n")
}

fn ensure_provider_exists(store: &Store, provider_id: &str) -> ClawResult<()> {
    if store
        .get_provider(provider_id)
        .map_err(store_error)?
        .is_none()
    {
        return Err(ClawError::NotFound(format!(
            "provider not found: {provider_id}"
        )));
    }
    Ok(())
}

fn finalize_provider(draft: ProviderDraft) -> ClawResult<ProviderConfig> {
    let label = draft
        .label
        .ok_or_else(|| ClawError::InvalidRequest("missing provider label".into()))?;
    let kind = draft
        .kind
        .ok_or_else(|| ClawError::InvalidRequest("missing provider kind".into()))?;
    let base_url = draft
        .base_url
        .ok_or_else(|| ClawError::InvalidRequest("missing provider base_url".into()))?;
    let api_key = draft
        .api_key
        .ok_or_else(|| ClawError::InvalidRequest("missing provider api_key".into()))?;
    let model = draft
        .model
        .ok_or_else(|| ClawError::InvalidRequest("missing provider model".into()))?;

    Ok(ProviderConfig {
        id: slugify(&label),
        label,
        kind,
        base_url,
        api_key,
        model,
        enabled: true,
        extra_headers: vec![],
        created_at: Utc::now(),
    })
}

fn parse_provider_kind(raw: &str) -> ClawResult<ProviderKind> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "openai-compatible" | "openai" => Ok(ProviderKind::OpenAiCompatible),
        "custom-chat-completions" | "custom" => Ok(ProviderKind::CustomChatCompletions),
        _ => Err(ClawError::InvalidRequest(
            "provider kind must be `openai-compatible` or `custom-chat-completions`".into(),
        )),
    }
}

fn normalize_base_url(raw: &str) -> String {
    raw.trim().trim_end_matches('/').to_string()
}

fn split_file_and_content(raw: &str) -> Option<(String, String)> {
    let mut parts = raw.splitn(2, ' ');
    let file = parts.next()?.trim();
    let content = parts.next()?.trim();
    if file.is_empty() || content.is_empty() {
        return None;
    }
    Some((file.to_string(), content.to_string()))
}

fn slugify(value: &str) -> String {
    let mut slug = value
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() {
                ch.to_ascii_lowercase()
            } else {
                '-'
            }
        })
        .collect::<String>();
    while slug.contains("--") {
        slug = slug.replace("--", "-");
    }
    slug.trim_matches('-').to_string()
}

fn store_error(err: anyhow::Error) -> ClawError {
    ClawError::Backend(err.to_string())
}

fn text_response(text: &str) -> GatewayResponse {
    GatewayResponse {
        text: text.to_string(),
        session: SessionDirective::Keep,
        keyboard: None,
    }
}

fn text_response_with_keyboard(text: &str, keyboard: Option<InlineKeyboard>) -> GatewayResponse {
    GatewayResponse {
        text: text.to_string(),
        session: SessionDirective::Keep,
        keyboard,
    }
}

fn provider_kind_keyboard() -> InlineKeyboard {
    InlineKeyboard {
        rows: vec![
            vec![
                InlineButton {
                    text: "OpenAI-compatible".into(),
                    data: "openai-compatible".into(),
                },
                InlineButton {
                    text: "Custom Chat".into(),
                    data: "custom-chat-completions".into(),
                },
            ],
            vec![InlineButton {
                text: "Cancel onboarding".into(),
                data: "/onboard cancel".into(),
            }],
        ],
    }
}

fn cancel_keyboard() -> InlineKeyboard {
    InlineKeyboard {
        rows: vec![vec![InlineButton {
            text: "Cancel onboarding".into(),
            data: "/onboard cancel".into(),
        }]],
    }
}

fn main_menu_keyboard() -> InlineKeyboard {
    InlineKeyboard {
        rows: vec![
            vec![
                InlineButton {
                    text: "Status".into(),
                    data: "/status".into(),
                },
                InlineButton {
                    text: "Provider".into(),
                    data: "/provider current".into(),
                },
                InlineButton {
                    text: "Model".into(),
                    data: "/model current".into(),
                },
            ],
            vec![
                InlineButton {
                    text: "Providers".into(),
                    data: "/provider list".into(),
                },
                InlineButton {
                    text: "Persona".into(),
                    data: "/persona list".into(),
                },
            ],
        ],
    }
}

fn provider_current_keyboard(provider_id: &str) -> InlineKeyboard {
    InlineKeyboard {
        rows: vec![
            vec![
                InlineButton {
                    text: "Switch provider".into(),
                    data: "/provider list".into(),
                },
                InlineButton {
                    text: "Switch model".into(),
                    data: format!("/provider models {provider_id}"),
                },
            ],
            vec![
                InlineButton {
                    text: "Test provider".into(),
                    data: format!("/provider test {provider_id}"),
                },
                InlineButton {
                    text: "Bind this chat".into(),
                    data: format!("/provider bind {provider_id}"),
                },
            ],
            vec![InlineButton {
                text: "Back to menu".into(),
                data: "/menu".into(),
            }],
        ],
    }
}

fn provider_list_keyboard(providers: &[ProviderRecord]) -> InlineKeyboard {
    let mut rows = providers
        .iter()
        .map(|provider| {
            vec![InlineButton {
                text: if provider.is_default {
                    format!("* {} ({})", provider.config.label, provider.config.id)
                } else {
                    format!("{} ({})", provider.config.label, provider.config.id)
                },
                data: format!("/provider use {}", provider.config.id),
            }]
        })
        .collect::<Vec<_>>();
    rows.push(vec![InlineButton {
        text: "Back to provider".into(),
        data: "/provider current".into(),
    }]);
    rows.push(vec![InlineButton {
        text: "Add provider".into(),
        data: "/provider add".into(),
    }]);
    InlineKeyboard { rows }
}

fn model_picker_keyboard(
    provider_id: &str,
    current_model: &str,
    models: &[String],
) -> InlineKeyboard {
    let mut rows = Vec::new();
    for model in models.iter().take(12) {
        rows.push(vec![InlineButton {
            text: if model == current_model {
                format!("* {model}")
            } else {
                model.clone()
            },
            data: format!("/provider set-model {provider_id} {model}"),
        }]);
    }
    if rows.is_empty() {
        rows.push(vec![InlineButton {
            text: "Open model list".into(),
            data: format!("/provider models {provider_id}"),
        }]);
    }
    rows.push(vec![
        InlineButton {
            text: "Refresh models".into(),
            data: format!("/provider models {provider_id}"),
        },
        InlineButton {
            text: "Provider card".into(),
            data: "/provider current".into(),
        },
    ]);
    rows.push(vec![InlineButton {
        text: "Back to menu".into(),
        data: "/menu".into(),
    }]);
    InlineKeyboard { rows }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use anyhow::Result;
    use hajimi_claw_agent::AgentRuntime;
    use hajimi_claw_exec::{LocalExecutor, PlatformMode};
    use hajimi_claw_policy::PolicyEngine;
    use hajimi_claw_store::Store;
    use hajimi_claw_tools::ToolRegistry;
    use tempfile::tempdir;

    use super::{
        Gateway, GatewayCommand, GatewayRequest, InProcessGateway, SessionDirective,
        parse_gateway_command,
    };

    #[test]
    fn parses_provider_test_command() {
        assert_eq!(
            parse_gateway_command("/provider test moonshot"),
            GatewayCommand::ProviderTest(Some("moonshot".into()))
        );
    }

    #[test]
    fn parses_model_use_command() {
        assert_eq!(
            parse_gateway_command("/model use gpt-5.1"),
            GatewayCommand::ModelUse("gpt-5.1".into())
        );
    }

    #[test]
    fn parses_model_picker_command() {
        assert_eq!(
            parse_gateway_command("/model use"),
            GatewayCommand::ModelPicker
        );
    }

    #[test]
    fn parses_plain_text_as_ask() {
        assert_eq!(
            parse_gateway_command("帮我检查 docker 日志"),
            GatewayCommand::Ask("帮我检查 docker 日志".into())
        );
    }

    #[test]
    fn parses_persona_append_command() {
        assert_eq!(
            parse_gateway_command("/persona append soul Be terse."),
            GatewayCommand::PersonaAppend {
                file: "soul".into(),
                content: "Be terse.".into(),
            }
        );
    }

    #[tokio::test]
    async fn gateway_opens_session_and_sets_channel_state() -> Result<()> {
        let dir = tempdir()?;
        let mut config = hajimi_claw_policy::PolicyConfig::default();
        config.allowed_workdirs = vec![dir.path().to_path_buf(), std::env::current_dir()?];
        config.admin_user_id = 1;
        config.admin_chat_id = 2;
        let policy = Arc::new(PolicyEngine::new(config));
        let executor = Arc::new(LocalExecutor::new(
            policy.clone(),
            PlatformMode::WindowsSafe,
        ));
        let tools = Arc::new(ToolRegistry::default(executor, policy.clone()));
        let store = Arc::new(Store::open_in_memory()?);
        let runtime = Arc::new(AgentRuntime::for_tests(
            tools,
            store.clone(),
            policy.clone(),
        ));
        let gateway = InProcessGateway::new(runtime, policy, store);

        let response = gateway
            .handle(GatewayRequest {
                actor_user_id: 1,
                actor_chat_id: 2,
                raw_text: "/shell open ops".into(),
                command: GatewayCommand::ShellOpen(Some("ops".into())),
                current_session_id: None,
            })
            .await?;
        assert!(matches!(response.session, SessionDirective::Set(_)));
        Ok(())
    }

    #[tokio::test]
    async fn onboarding_session_starts() -> Result<()> {
        let dir = tempdir()?;
        let mut config = hajimi_claw_policy::PolicyConfig::default();
        config.allowed_workdirs = vec![dir.path().to_path_buf(), std::env::current_dir()?];
        config.admin_user_id = 1;
        config.admin_chat_id = 2;
        let policy = Arc::new(PolicyEngine::new(config));
        let executor = Arc::new(LocalExecutor::new(
            policy.clone(),
            PlatformMode::WindowsSafe,
        ));
        let tools = Arc::new(ToolRegistry::default(executor, policy.clone()));
        let store = Arc::new(Store::open_in_memory()?);
        let runtime = Arc::new(AgentRuntime::for_tests(
            tools,
            store.clone(),
            policy.clone(),
        ));
        let gateway = InProcessGateway::new(runtime, policy, store.clone());

        let response = gateway
            .handle(GatewayRequest {
                actor_user_id: 1,
                actor_chat_id: 2,
                raw_text: "/onboard".into(),
                command: GatewayCommand::Onboard,
                current_session_id: None,
            })
            .await?;
        assert!(response.text.contains("Step 1/5"));
        assert!(store.load_onboarding_session(2, 1)?.is_some());
        Ok(())
    }

    #[tokio::test]
    async fn provider_current_returns_keyboard() -> Result<()> {
        let dir = tempdir()?;
        let mut config = hajimi_claw_policy::PolicyConfig::default();
        config.allowed_workdirs = vec![dir.path().to_path_buf(), std::env::current_dir()?];
        config.admin_user_id = 1;
        config.admin_chat_id = 2;
        let policy = Arc::new(PolicyEngine::new(config));
        let executor = Arc::new(LocalExecutor::new(
            policy.clone(),
            PlatformMode::WindowsSafe,
        ));
        let tools = Arc::new(ToolRegistry::default(executor, policy.clone()));
        let store = Arc::new(Store::open_in_memory()?);
        store.upsert_provider(&hajimi_claw_types::ProviderRecord {
            config: hajimi_claw_types::ProviderConfig {
                id: "openai".into(),
                label: "OpenAI".into(),
                kind: hajimi_claw_types::ProviderKind::OpenAiCompatible,
                base_url: "https://example.com/v1".into(),
                api_key: "secret".into(),
                model: "gpt-5.1".into(),
                enabled: true,
                extra_headers: vec![],
                created_at: chrono::Utc::now(),
            },
            is_default: true,
        })?;
        let runtime = Arc::new(AgentRuntime::for_tests(
            tools,
            store.clone(),
            policy.clone(),
        ));
        let gateway = InProcessGateway::new(runtime, policy, store);

        let response = gateway
            .handle(GatewayRequest {
                actor_user_id: 1,
                actor_chat_id: 2,
                raw_text: "/provider current".into(),
                command: GatewayCommand::ProviderCurrent,
                current_session_id: None,
            })
            .await?;
        assert!(response.keyboard.is_some());
        Ok(())
    }
}
