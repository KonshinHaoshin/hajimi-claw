use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use chrono::Utc;
use hajimi_claw_agent::{AgentRuntime, MultiAgentPreference};
use hajimi_claw_llm::{list_models, test_provider};
use hajimi_claw_policy::PolicyEngine;
use hajimi_claw_store::Store;
use hajimi_claw_types::{
    ClawError, ClawResult, ConversationId, OnboardingSession, OnboardingStep, ProviderCapabilities,
    ProviderConfig, ProviderDraft, ProviderKind, ProviderRecord,
};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::sync::RwLock;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum GatewayCommand {
    Ask(String),
    ShellOpen(Option<String>),
    ShellStatus,
    ShellExec(String),
    ShellClose,
    Status,
    NewConversation,
    Approve(String),
    ElevatedMenu,
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
    AgentsOn,
    AgentsOff,
    AgentsAuto,
    AgentsStatus,
    Capabilities,
    Skills,
    SkillRun { name: String, input: String },
    Mcp,
    McpTools(Option<String>),
    PersonaGuide,
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
    pub current_conversation_id: Option<ConversationId>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SessionDirective {
    Keep,
    Set(String),
    Clear,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConversationDirective {
    Keep,
    Set(ConversationId),
    Clear,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GatewayResponse {
    pub text: String,
    pub session: SessionDirective,
    pub conversation: ConversationDirective,
    pub keyboard: Option<InlineKeyboard>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GatewayPreview {
    pub text: String,
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

    async fn preview(&self, _request: GatewayRequest) -> Option<GatewayPreview> {
        None
    }
}

pub struct InProcessGateway {
    runtime: Arc<AgentRuntime>,
    policy: Arc<PolicyEngine>,
    store: Arc<Store>,
    client: Client,
    multi_agent_modes: RwLock<HashMap<i64, SessionMultiAgentMode>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SessionMultiAgentMode {
    Auto,
    On,
    Off,
}

impl InProcessGateway {
    pub fn new(runtime: Arc<AgentRuntime>, policy: Arc<PolicyEngine>, store: Arc<Store>) -> Self {
        Self {
            runtime,
            policy,
            store,
            client: Client::new(),
            multi_agent_modes: RwLock::new(HashMap::new()),
        }
    }
}

#[async_trait]
impl Gateway for InProcessGateway {
    async fn preview(&self, request: GatewayRequest) -> Option<GatewayPreview> {
        if !self
            .policy
            .authorize_telegram_actor(request.actor_user_id, request.actor_chat_id)
        {
            return None;
        }
        match request.command {
            GatewayCommand::Ask(prompt) => {
                let preference = self.multi_agent_preference(request.actor_chat_id).await;
                let preview = self
                    .runtime
                    .preview_multi_agent_request(&prompt, preference)?;
                Some(GatewayPreview {
                    text: format!("已拆成 {} 个 agents，正在汇总...", preview.worker_count),
                    keyboard: Some(agents_keyboard(
                        self.multi_agent_mode(request.actor_chat_id).await,
                    )),
                })
            }
            _ => None,
        }
    }

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
                let preference = self.multi_agent_preference(request.actor_chat_id).await;
                let provider_id = self
                    .store
                    .resolve_provider_for_chat(request.actor_chat_id)
                    .map_err(store_error)?
                    .map(|record| record.config.id);
                let reply = self
                    .runtime
                    .ask_with_provider_and_preference_and_session_and_conversation(
                        &prompt,
                        None,
                        provider_id,
                        preference,
                        request.current_conversation_id,
                        request.current_session_id.clone(),
                    )
                    .await?;
                Ok(GatewayResponse {
                    text: reply.message,
                    session: SessionDirective::Keep,
                    conversation: ConversationDirective::Set(reply.conversation_id),
                    keyboard: None,
                })
            }
            GatewayCommand::ShellOpen(name) => {
                let reply = self.runtime.shell_open(name, None).await?;
                Ok(GatewayResponse {
                    text: reply.message,
                    session: SessionDirective::Set(reply.session_id),
                    conversation: ConversationDirective::Keep,
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
                    conversation: ConversationDirective::Keep,
                    keyboard: None,
                })
            }
            GatewayCommand::ShellStatus => {
                let session_id = request.current_session_id.ok_or_else(|| {
                    ClawError::InvalidRequest("no active session, use /shell open first".into())
                })?;
                Ok(GatewayResponse {
                    text: self.runtime.shell_status(&session_id).await?,
                    session: SessionDirective::Keep,
                    conversation: ConversationDirective::Keep,
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
                    conversation: ConversationDirective::Keep,
                    keyboard: None,
                })
            }
            GatewayCommand::NewConversation => Ok(GatewayResponse {
                text: "started a new conversation".into(),
                session: SessionDirective::Keep,
                conversation: ConversationDirective::Clear,
                keyboard: None,
            }),
            GatewayCommand::Status => Ok(text_response(&self.runtime.status()?)),
            GatewayCommand::Approve(request_id) => {
                Ok(text_response(&self.runtime.approve(&request_id).await?))
            }
            GatewayCommand::ElevatedMenu => Ok(text_response_with_keyboard(
                &elevated_menu_text(),
                Some(elevated_keyboard()),
            )),
            GatewayCommand::ElevatedOn => Ok(text_response_with_keyboard(
                &self.runtime.enable_elevated(),
                Some(elevated_keyboard()),
            )),
            GatewayCommand::ElevatedOff => Ok(text_response_with_keyboard(
                &self.runtime.stop_elevated(),
                Some(elevated_keyboard()),
            )),
            GatewayCommand::ElevatedAsk => Ok(text_response_with_keyboard(
                &self.runtime.enable_approval_mode(),
                Some(elevated_keyboard()),
            )),
            GatewayCommand::ElevatedFull => Ok(text_response_with_keyboard(
                &self.runtime.enable_full_elevated(),
                Some(elevated_keyboard()),
            )),
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
            GatewayCommand::AgentsOn => {
                self.agents_set_mode(request.actor_chat_id, SessionMultiAgentMode::On)
                    .await
            }
            GatewayCommand::AgentsOff => {
                self.agents_set_mode(request.actor_chat_id, SessionMultiAgentMode::Off)
                    .await
            }
            GatewayCommand::AgentsAuto => {
                self.agents_set_mode(request.actor_chat_id, SessionMultiAgentMode::Auto)
                    .await
            }
            GatewayCommand::AgentsStatus => self.agents_status(request.actor_chat_id).await,
            GatewayCommand::Capabilities => Ok(text_response_with_keyboard(
                &self.runtime.render_capability_inventory(),
                Some(capabilities_keyboard()),
            )),
            GatewayCommand::Skills => Ok(text_response_with_keyboard(
                &self.runtime.render_skill_inventory(),
                Some(skills_keyboard()),
            )),
            GatewayCommand::SkillRun { name, input } => {
                let parsed = parse_skill_input(&input)?;
                let reply = self
                    .runtime
                    .invoke_skill(
                        &name,
                        parsed,
                        None,
                        request.current_conversation_id,
                        request.current_session_id.clone(),
                    )
                    .await?;
                Ok(GatewayResponse {
                    text: reply.message,
                    session: SessionDirective::Keep,
                    conversation: ConversationDirective::Set(reply.conversation_id),
                    keyboard: None,
                })
            }
            GatewayCommand::Mcp => Ok(text_response_with_keyboard(
                &self.runtime.render_mcp_inventory(),
                Some(mcp_keyboard()),
            )),
            GatewayCommand::McpTools(server) => Ok(text_response_with_keyboard(
                &self.runtime.render_mcp_tool_inventory(server.as_deref()),
                Some(mcp_keyboard()),
            )),
            GatewayCommand::PersonaGuide => Ok(text_response_with_keyboard(
                &persona_guide_text(),
                Some(persona_guide_keyboard()),
            )),
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
                    "onboarding complete.\nprovider=`{}` id=`{}`{}\nchat_binding=enabled\nhealth={}\n{}\n\nNext: open `/persona guide` to configure identity, soul, and heartbeat from chat.",
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

    async fn multi_agent_mode(&self, chat_id: i64) -> SessionMultiAgentMode {
        self.multi_agent_modes
            .read()
            .await
            .get(&chat_id)
            .copied()
            .unwrap_or(SessionMultiAgentMode::Auto)
    }

    async fn multi_agent_preference(&self, chat_id: i64) -> MultiAgentPreference {
        match self.multi_agent_mode(chat_id).await {
            SessionMultiAgentMode::Auto => MultiAgentPreference::Auto,
            SessionMultiAgentMode::On => MultiAgentPreference::ForceOn,
            SessionMultiAgentMode::Off => MultiAgentPreference::ForceOff,
        }
    }

    async fn agents_set_mode(
        &self,
        chat_id: i64,
        mode: SessionMultiAgentMode,
    ) -> ClawResult<GatewayResponse> {
        self.multi_agent_modes.write().await.insert(chat_id, mode);
        self.agents_status(chat_id).await
    }

    async fn agents_status(&self, chat_id: i64) -> ClawResult<GatewayResponse> {
        let mode = self.multi_agent_mode(chat_id).await;
        let description = match mode {
            SessionMultiAgentMode::Auto => {
                "mode=auto\nmulti-agent follows prompt keywords or explicit `N agents`."
            }
            SessionMultiAgentMode::On => {
                "mode=on\nnatural-language asks will force multi-agent execution."
            }
            SessionMultiAgentMode::Off => {
                "mode=off\nnatural-language asks stay single-agent unless you change the mode."
            }
        };
        Ok(text_response_with_keyboard(
            &format!("multi-agent for chat `{chat_id}`\n{description}"),
            Some(agents_keyboard(mode)),
        ))
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
    if trimmed == "/agents on" {
        return GatewayCommand::AgentsOn;
    }
    if trimmed == "/agents off" {
        return GatewayCommand::AgentsOff;
    }
    if trimmed == "/agents auto" {
        return GatewayCommand::AgentsAuto;
    }
    if trimmed == "/agents status" || trimmed == "/agents" {
        return GatewayCommand::AgentsStatus;
    }
    if trimmed == "/model use" {
        return GatewayCommand::ModelPicker;
    }
    if trimmed == "/capabilities" {
        return GatewayCommand::Capabilities;
    }
    if trimmed == "/skills" {
        return GatewayCommand::Skills;
    }
    if trimmed == "/mcp" {
        return GatewayCommand::Mcp;
    }
    if let Some(rest) = trimmed.strip_prefix("/model use ") {
        return GatewayCommand::ModelUse(rest.trim().into());
    }
    if let Some(rest) = trimmed.strip_prefix("/skill run ") {
        if let Some((name, input)) = split_file_and_content(rest.trim()) {
            return GatewayCommand::SkillRun { name, input };
        }
    }
    if trimmed == "/mcp tools" {
        return GatewayCommand::McpTools(None);
    }
    if let Some(rest) = trimmed.strip_prefix("/mcp tools ") {
        let server = rest.trim();
        return GatewayCommand::McpTools((!server.is_empty()).then(|| server.to_string()));
    }
    if trimmed == "/persona list" {
        return GatewayCommand::PersonaList;
    }
    if trimmed == "/persona guide" {
        return GatewayCommand::PersonaGuide;
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
    if trimmed == "/elevated" {
        return GatewayCommand::ElevatedMenu;
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
    if trimmed == "/new" {
        return GatewayCommand::NewConversation;
    }
    if trimmed == "/shell status" {
        return GatewayCommand::ShellStatus;
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
        "/capabilities",
        "/skills",
        "/skill run <name> <json-or-text>",
        "/mcp",
        "/mcp tools [server]",
        "/agents on",
        "/agents off",
        "/agents auto",
        "/agents status",
        "/persona guide",
        "/persona list",
        "/persona read <identity|soul|agents|tools|skills|heartbeat>",
        "/persona write <file> <content>",
        "/persona append <file> <content>",
        "/shell open [name]",
        "/shell status",
        "/shell exec <cmd>",
        "/shell close",
        "/new",
        "/status",
        "/menu",
        "/approve <request-id>",
        "/elevated",
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

fn parse_skill_input(raw: &str) -> ClawResult<Value> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Ok(serde_json::json!({}));
    }
    serde_json::from_str(trimmed).or_else(|_| Ok(serde_json::json!({ "input": trimmed })))
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
        fallback_models: draft.fallback_models.unwrap_or_default(),
        capabilities: default_provider_capabilities(),
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

fn default_provider_capabilities() -> ProviderCapabilities {
    ProviderCapabilities {
        tool_calling: true,
        streaming: false,
        json_mode: false,
        max_context_chars: Some(24_000),
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
        conversation: ConversationDirective::Keep,
        keyboard: None,
    }
}

fn text_response_with_keyboard(text: &str, keyboard: Option<InlineKeyboard>) -> GatewayResponse {
    GatewayResponse {
        text: text.to_string(),
        session: SessionDirective::Keep,
        conversation: ConversationDirective::Keep,
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
                InlineButton {
                    text: "Agents".into(),
                    data: "/agents status".into(),
                },
            ],
            vec![
                InlineButton {
                    text: "Providers".into(),
                    data: "/provider list".into(),
                },
                InlineButton {
                    text: "Skills".into(),
                    data: "/skills".into(),
                },
                InlineButton {
                    text: "MCP".into(),
                    data: "/mcp".into(),
                },
                InlineButton {
                    text: "Capabilities".into(),
                    data: "/capabilities".into(),
                },
            ],
            vec![
                InlineButton {
                    text: "New".into(),
                    data: "/new".into(),
                },
                InlineButton {
                    text: "Elevated".into(),
                    data: "/elevated".into(),
                },
                InlineButton {
                    text: "Persona".into(),
                    data: "/persona guide".into(),
                },
            ],
        ],
    }
}

fn elevated_menu_text() -> String {
    [
        "elevated controls",
        "",
        "`/elevated ask` = default guarded mode. Sensitive actions stop for approval.",
        "`/elevated on` = temporary elevated lease for guarded and dangerous commands.",
        "`/elevated full` = full local bypass for workdir, writable path, Windows safe allowlist, and sensitive env checks.",
        "`/elevated off` = disable any active elevated lease.",
    ]
    .join("\n")
}

fn capabilities_keyboard() -> InlineKeyboard {
    InlineKeyboard {
        rows: vec![
            vec![
                InlineButton {
                    text: "Skills".into(),
                    data: "/skills".into(),
                },
                InlineButton {
                    text: "MCP".into(),
                    data: "/mcp".into(),
                },
            ],
            vec![InlineButton {
                text: "Back to menu".into(),
                data: "/menu".into(),
            }],
        ],
    }
}

fn skills_keyboard() -> InlineKeyboard {
    InlineKeyboard {
        rows: vec![
            vec![
                InlineButton {
                    text: "Capabilities".into(),
                    data: "/capabilities".into(),
                },
                InlineButton {
                    text: "MCP tools".into(),
                    data: "/mcp tools".into(),
                },
            ],
            vec![InlineButton {
                text: "Back to menu".into(),
                data: "/menu".into(),
            }],
        ],
    }
}

fn mcp_keyboard() -> InlineKeyboard {
    InlineKeyboard {
        rows: vec![
            vec![
                InlineButton {
                    text: "MCP tools".into(),
                    data: "/mcp tools".into(),
                },
                InlineButton {
                    text: "Capabilities".into(),
                    data: "/capabilities".into(),
                },
            ],
            vec![InlineButton {
                text: "Back to menu".into(),
                data: "/menu".into(),
            }],
        ],
    }
}

fn elevated_keyboard() -> InlineKeyboard {
    InlineKeyboard {
        rows: vec![
            vec![
                InlineButton {
                    text: "Ask".into(),
                    data: "/elevated ask".into(),
                },
                InlineButton {
                    text: "On".into(),
                    data: "/elevated on".into(),
                },
            ],
            vec![
                InlineButton {
                    text: "Full".into(),
                    data: "/elevated full".into(),
                },
                InlineButton {
                    text: "Off".into(),
                    data: "/elevated off".into(),
                },
            ],
        ],
    }
}

fn persona_guide_text() -> String {
    [
        "persona guide",
        "",
        "Layer model:",
        "1. base system prompt",
        "2. `identity.md` = who the user is, owned systems, environments, durable preferences, and hard constraints",
        "3. `soul.md` = Hajimi's stable role, tone, style, and behavioral stance",
        "4. `agents.md` / `AGENTS.md` / `tools.md` / `skills.md` = operational extensions for delegation, tools, workflows, and repo guidance",
        "5. runtime overlays like shell-session metadata and multi-agent role instructions",
        "",
        "Precedence:",
        "- auto-discovery loads `persona.directory`, then the config directory, then the current working directory",
        "- higher-precedence files override structured `identity.md` / `soul.md` fields while preserving accumulated notes",
        "- extensions stay additive in precedence order",
        "- if `[persona].prompt_files` is set, that explicit list is used instead",
        "",
        "Parsing:",
        "- `identity.md` and `soul.md` support optional front matter, but plain markdown still works",
        "- malformed front matter safely falls back to legacy markdown behavior",
        "- `heartbeat.md` is runtime config only and never enters the prompt",
        "",
        "Heartbeat format example:",
        "`enabled: true`",
        "`interval_secs: 30`",
        "",
        "Useful commands:",
        "`/persona list`",
        "`/persona read identity`",
        "`/persona read soul`",
        "`/persona write identity You are helping Alice maintain two Linux VPS nodes.`",
        "`/persona write soul You are Hajimi, a calm cat AI ops assistant.`",
        "`/persona write heartbeat enabled: true`",
        "`/persona append heartbeat interval_secs: 15`",
    ]
    .join("\n")
}

fn persona_guide_keyboard() -> InlineKeyboard {
    InlineKeyboard {
        rows: vec![
            vec![
                InlineButton {
                    text: "Read soul".into(),
                    data: "/persona read soul".into(),
                },
                InlineButton {
                    text: "Read identity".into(),
                    data: "/persona read identity".into(),
                },
            ],
            vec![
                InlineButton {
                    text: "Read heartbeat".into(),
                    data: "/persona read heartbeat".into(),
                },
                InlineButton {
                    text: "List persona files".into(),
                    data: "/persona list".into(),
                },
            ],
            vec![InlineButton {
                text: "Back to menu".into(),
                data: "/menu".into(),
            }],
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

fn agents_keyboard(mode: SessionMultiAgentMode) -> InlineKeyboard {
    InlineKeyboard {
        rows: vec![
            vec![
                InlineButton {
                    text: if mode == SessionMultiAgentMode::On {
                        "* Agents on".into()
                    } else {
                        "Agents on".into()
                    },
                    data: "/agents on".into(),
                },
                InlineButton {
                    text: if mode == SessionMultiAgentMode::Off {
                        "* Agents off".into()
                    } else {
                        "Agents off".into()
                    },
                    data: "/agents off".into(),
                },
                InlineButton {
                    text: if mode == SessionMultiAgentMode::Auto {
                        "* Agents auto".into()
                    } else {
                        "Agents auto".into()
                    },
                    data: "/agents auto".into(),
                },
            ],
            vec![InlineButton {
                text: "Back to menu".into(),
                data: "/menu".into(),
            }],
        ],
    }
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
    fn parses_agents_commands() {
        assert_eq!(
            parse_gateway_command("/agents status"),
            GatewayCommand::AgentsStatus
        );
        assert_eq!(
            parse_gateway_command("/agents on"),
            GatewayCommand::AgentsOn
        );
        assert_eq!(
            parse_gateway_command("/agents off"),
            GatewayCommand::AgentsOff
        );
        assert_eq!(
            parse_gateway_command("/agents auto"),
            GatewayCommand::AgentsAuto
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

    #[test]
    fn parses_persona_guide_command() {
        assert_eq!(
            parse_gateway_command("/persona guide"),
            GatewayCommand::PersonaGuide
        );
    }

    #[test]
    fn parses_capability_commands() {
        assert_eq!(
            parse_gateway_command("/capabilities"),
            GatewayCommand::Capabilities
        );
        assert_eq!(parse_gateway_command("/skills"), GatewayCommand::Skills);
        assert_eq!(parse_gateway_command("/mcp"), GatewayCommand::Mcp);
        assert_eq!(
            parse_gateway_command("/mcp tools demo"),
            GatewayCommand::McpTools(Some("demo".into()))
        );
        assert_eq!(
            parse_gateway_command("/skill run skill.deploy {\"service\":\"api\"}"),
            GatewayCommand::SkillRun {
                name: "skill.deploy".into(),
                input: "{\"service\":\"api\"}".into(),
            }
        );
    }

    #[test]
    fn persona_guide_mentions_layered_persona_model() {
        let guide = super::persona_guide_text();
        assert!(guide.contains("Layer model:"));
        assert!(guide.contains("`identity.md`"));
        assert!(guide.contains("`soul.md`"));
        assert!(
            guide.contains("`heartbeat.md` is runtime config only and never enters the prompt")
        );
        assert!(guide.contains("`/persona list`"));
    }

    #[test]
    fn help_text_mentions_capability_commands() {
        let help = super::help_text();
        assert!(help.contains("/capabilities"));
        assert!(help.contains("/skills"));
        assert!(help.contains("/skill run <name> <json-or-text>"));
        assert!(help.contains("/mcp"));
        assert!(help.contains("/mcp tools [server]"));
    }

    #[test]
    fn parses_shell_status_command() {
        assert_eq!(
            parse_gateway_command("/shell status"),
            GatewayCommand::ShellStatus
        );
    }

    #[test]
    fn parses_elevated_menu_command() {
        assert_eq!(
            parse_gateway_command("/elevated"),
            GatewayCommand::ElevatedMenu
        );
    }

    #[test]
    fn parses_new_conversation_command() {
        assert_eq!(
            parse_gateway_command("/new"),
            GatewayCommand::NewConversation
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
                current_conversation_id: None,
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
                current_conversation_id: None,
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
                fallback_models: vec![],
                capabilities: hajimi_claw_types::ProviderCapabilities {
                    tool_calling: true,
                    streaming: false,
                    json_mode: false,
                    max_context_chars: Some(24_000),
                },
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
                current_conversation_id: None,
            })
            .await?;
        assert!(response.keyboard.is_some());
        Ok(())
    }

    #[tokio::test]
    async fn elevated_menu_returns_keyboard() -> Result<()> {
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
                raw_text: "/elevated".into(),
                command: GatewayCommand::ElevatedMenu,
                current_session_id: None,
                current_conversation_id: None,
            })
            .await?;
        assert!(response.text.contains("elevated controls"));
        assert!(response.keyboard.is_some());
        Ok(())
    }

    #[tokio::test]
    async fn agents_on_enables_multi_agent_preview() -> Result<()> {
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

        let _ = gateway
            .handle(GatewayRequest {
                actor_user_id: 1,
                actor_chat_id: 2,
                raw_text: "/agents on".into(),
                command: GatewayCommand::AgentsOn,
                current_session_id: None,
                current_conversation_id: None,
            })
            .await?;

        let preview = gateway
            .preview(GatewayRequest {
                actor_user_id: 1,
                actor_chat_id: 2,
                raw_text: "你好".into(),
                command: GatewayCommand::Ask("你好".into()),
                current_session_id: None,
                current_conversation_id: None,
            })
            .await
            .expect("preview should exist");
        assert!(preview.text.contains("agents"));
        assert!(preview.keyboard.is_some());
        Ok(())
    }
}
