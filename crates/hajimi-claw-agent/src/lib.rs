use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use std::{fs, path::Path};

use chrono::Utc;
use futures::TryStreamExt;
use hajimi_claw_llm::StaticBackend;
use hajimi_claw_policy::PolicyEngine;
use hajimi_claw_store::Store;
use hajimi_claw_tools::ToolRegistry;
use hajimi_claw_types::{
    AgentEvent, AgentRequest, ApprovalId, ApprovalRecord, ClawError, ClawResult, ConversationId,
    ConversationMessage, LlmBackend, MessageRole, PolicyMode, ProviderRecord, TaskId, TaskKind,
    TaskRunState, TaskStatus, ToolCallRecord, ToolContext, ToolExchange, ToolInvocationRecord,
    ToolInvocationStatus, ToolResultRecord,
};
use regex::Regex;
use serde_json::json;
use tokio::sync::Semaphore;
use tokio::task::JoinSet;
use tokio::time;

pub struct AgentRuntime {
    llm: Arc<dyn LlmBackend>,
    tools: Arc<ToolRegistry>,
    store: Arc<Store>,
    policy: Arc<PolicyEngine>,
    task_gate: Arc<Semaphore>,
    prompt_source: Arc<dyn SystemPromptSource>,
    persona_dir: PathBuf,
    multi_agent: MultiAgentConfig,
}

#[derive(Debug, Clone)]
pub struct ShellOpenReply {
    pub session_id: String,
    pub message: String,
}

#[derive(Debug, Clone)]
pub struct AskReply {
    pub conversation_id: ConversationId,
    pub message: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MultiAgentPreference {
    Auto,
    ForceOn,
    ForceOff,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MultiAgentPreview {
    pub worker_count: usize,
}

#[derive(Debug, Clone)]
pub struct MultiAgentConfig {
    pub enabled: bool,
    pub auto_delegate: bool,
    pub default_workers: usize,
    pub max_workers: usize,
    pub worker_timeout_secs: u64,
    pub max_context_chars_per_worker: usize,
}

impl Default for MultiAgentConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            auto_delegate: false,
            default_workers: 3,
            max_workers: 8,
            worker_timeout_secs: 90,
            max_context_chars_per_worker: 24_000,
        }
    }
}

#[derive(Debug, Clone)]
struct TaskExecution {
    task_id: TaskId,
    conversation_id: ConversationId,
    prompt: String,
    cwd: Option<PathBuf>,
    provider_id: Option<String>,
    current_session_id: Option<String>,
}

enum ToolCallOutcome {
    Completed(ToolExchange),
    Blocked(ClawError),
}

impl MultiAgentConfig {
    fn normalized(&self) -> Self {
        let max_workers = self.max_workers.max(1);
        let default_workers = self.default_workers.max(1).min(max_workers);
        let worker_timeout_secs = self.worker_timeout_secs.max(5);
        let max_context_chars_per_worker = self.max_context_chars_per_worker.max(500);
        Self {
            enabled: self.enabled,
            auto_delegate: self.auto_delegate,
            default_workers,
            max_workers,
            worker_timeout_secs,
            max_context_chars_per_worker,
        }
    }
}

impl AgentRuntime {
    pub fn new(
        llm: Arc<dyn LlmBackend>,
        tools: Arc<ToolRegistry>,
        store: Arc<Store>,
        policy: Arc<PolicyEngine>,
        prompt_source: Arc<dyn SystemPromptSource>,
        persona_dir: PathBuf,
        multi_agent: MultiAgentConfig,
    ) -> Self {
        Self {
            llm,
            tools,
            store,
            policy,
            task_gate: Arc::new(Semaphore::new(1)),
            prompt_source,
            persona_dir,
            multi_agent: multi_agent.normalized(),
        }
    }

    pub fn for_tests(
        tools: Arc<ToolRegistry>,
        store: Arc<Store>,
        policy: Arc<PolicyEngine>,
    ) -> Self {
        Self::new(
            Arc::new(StaticBackend::new("fallback")),
            tools,
            store,
            policy,
            Arc::new(StaticSystemPrompt::new(default_system_prompt())),
            std::env::temp_dir(),
            MultiAgentConfig::default(),
        )
    }

    pub async fn ask(&self, prompt: &str, cwd: Option<PathBuf>) -> ClawResult<String> {
        Ok(self
            .ask_with_provider_and_preference_and_session(
                prompt,
                cwd,
                None,
                MultiAgentPreference::Auto,
                None,
            )
            .await?
            .message)
    }

    pub async fn ask_with_provider(
        &self,
        prompt: &str,
        cwd: Option<PathBuf>,
        provider_id: Option<String>,
    ) -> ClawResult<String> {
        Ok(self
            .ask_with_provider_and_preference_and_session(
                prompt,
                cwd,
                provider_id,
                MultiAgentPreference::Auto,
                None,
            )
            .await?
            .message)
    }

    pub fn preview_multi_agent_request(
        &self,
        prompt: &str,
        preference: MultiAgentPreference,
    ) -> Option<MultiAgentPreview> {
        should_delegate_multi_agent(prompt, &self.multi_agent, preference).then(|| {
            MultiAgentPreview {
                worker_count: resolve_worker_count(prompt, &self.multi_agent, preference),
            }
        })
    }

    pub async fn ask_with_provider_and_preference(
        &self,
        prompt: &str,
        cwd: Option<PathBuf>,
        provider_id: Option<String>,
        preference: MultiAgentPreference,
    ) -> ClawResult<String> {
        Ok(self
            .ask_with_provider_and_preference_and_session(
                prompt,
                cwd,
                provider_id,
                preference,
                None,
            )
            .await?
            .message)
    }

    pub async fn ask_with_provider_and_preference_and_session(
        &self,
        prompt: &str,
        cwd: Option<PathBuf>,
        provider_id: Option<String>,
        preference: MultiAgentPreference,
        current_session_id: Option<String>,
    ) -> ClawResult<AskReply> {
        self.ask_with_provider_and_preference_and_session_and_conversation(
            prompt,
            cwd,
            provider_id,
            preference,
            None,
            current_session_id,
        )
        .await
    }

    pub async fn ask_with_provider_and_preference_and_session_and_conversation(
        &self,
        prompt: &str,
        cwd: Option<PathBuf>,
        provider_id: Option<String>,
        preference: MultiAgentPreference,
        conversation_id: Option<ConversationId>,
        current_session_id: Option<String>,
    ) -> ClawResult<AskReply> {
        let _permit = self
            .task_gate
            .acquire()
            .await
            .map_err(|_| ClawError::Backend("task gate closed".into()))?;

        let task_id = TaskId::new();
        let conversation_id = conversation_id.unwrap_or_default();
        let mut status = self.new_task_status(
            task_id,
            conversation_id,
            TaskKind::EphemeralAgentTask,
            prompt,
            cwd.clone(),
            provider_id.clone(),
            current_session_id.clone(),
        );
        self.store.upsert_task(&status).map_err(store_error)?;

        self.store
            .save_message(
                conversation_id,
                &ConversationMessage {
                    role: MessageRole::User,
                    content: prompt.into(),
                    created_at: Utc::now(),
                },
            )
            .map_err(store_error)?;

        let execution = TaskExecution {
            task_id,
            conversation_id,
            prompt: prompt.into(),
            cwd: cwd.clone(),
            provider_id: provider_id.clone(),
            current_session_id,
        };

        let result = if let Some((tool, input)) = select_tool(prompt) {
            match self
                .execute_tool_call(
                    &execution,
                    None,
                    tool,
                    input,
                    status.current_session_id.clone(),
                    1,
                )
                .await?
            {
                ToolCallOutcome::Completed(exchange) => Ok(exchange.result.content),
                ToolCallOutcome::Blocked(err) => Err(err),
            }
        } else if should_delegate_multi_agent(prompt, &self.multi_agent, preference) {
            self.run_multi_agent(conversation_id, prompt, provider_id, preference)
                .await
        } else {
            self.run_llm(execution).await
        };

        match &result {
            Ok(content) => {
                status.running = false;
                status.state = TaskRunState::Completed;
                status.finished_at = Some(Utc::now());
                status.result_preview = Some(clamp_chars(content, 400));
                status.error = None;
                status.blocked_approval_id = None;
                self.store.upsert_task(&status).map_err(store_error)?;
                self.store
                    .save_message(
                        conversation_id,
                        &ConversationMessage {
                            role: MessageRole::Assistant,
                            content: content.clone(),
                            created_at: Utc::now(),
                        },
                    )
                    .map_err(store_error)?;
            }
            Err(ClawError::ApprovalRequired(_)) => {}
            Err(err) => {
                status.running = false;
                status.state = TaskRunState::Failed;
                status.finished_at = Some(Utc::now());
                status.error = Some(err.to_string());
                self.store.upsert_task(&status).map_err(store_error)?;
            }
        }

        result.map(|message| AskReply {
            conversation_id,
            message,
        })
    }

    pub async fn shell_open(
        &self,
        name: Option<String>,
        cwd: Option<PathBuf>,
    ) -> ClawResult<ShellOpenReply> {
        let output = self
            .tools
            .call(
                "session_open",
                ToolContext {
                    conversation_id: ConversationId::new(),
                    working_directory: cwd,
                    elevated: self.policy.is_elevated(),
                },
                json!({ "name": name }),
            )
            .await?;
        let session_id = output
            .structured
            .as_ref()
            .and_then(|value| value.get("session_id"))
            .and_then(|value| value.as_str())
            .ok_or_else(|| ClawError::Backend("session_open did not return session_id".into()))?;
        let handle = self.fetch_session_handle(session_id).await?;
        self.store
            .upsert_session(&handle, true)
            .map_err(store_error)?;
        Ok(ShellOpenReply {
            session_id: session_id.to_string(),
            message: output.content,
        })
    }

    pub async fn shell_exec(&self, session_id: &str, command: &str) -> ClawResult<String> {
        let output = self
            .tools
            .call(
                "session_exec",
                ToolContext {
                    conversation_id: ConversationId::new(),
                    working_directory: None,
                    elevated: self.policy.is_elevated(),
                },
                json!({ "session_id": session_id, "command": command }),
            )
            .await?;
        let handle = self.fetch_session_handle(session_id).await?;
        self.store
            .upsert_session(&handle, true)
            .map_err(store_error)?;
        if let Some(structured) = &output.structured {
            self.store
                .append_command_audit(
                    None,
                    Some(session_id.to_string()),
                    command,
                    structured
                        .get("exit_code")
                        .and_then(|value| value.as_i64())
                        .map(|value| value as i32),
                    structured
                        .get("duration_ms")
                        .and_then(|value| value.as_u64())
                        .unwrap_or_default() as u128,
                )
                .map_err(store_error)?;
        }
        Ok(output.content)
    }

    pub async fn shell_status(&self, session_id: &str) -> ClawResult<String> {
        let output = self
            .tools
            .call(
                "session_status",
                ToolContext {
                    conversation_id: ConversationId::new(),
                    working_directory: None,
                    elevated: self.policy.is_elevated(),
                },
                json!({ "session_id": session_id }),
            )
            .await?;
        let handle = self.fetch_session_handle(session_id).await?;
        self.store
            .upsert_session(&handle, true)
            .map_err(store_error)?;
        Ok(output.content)
    }

    pub async fn shell_close(&self, session_id: &str) -> ClawResult<String> {
        let handle = self.fetch_session_handle(session_id).await?;
        let output = self
            .tools
            .call(
                "session_close",
                ToolContext {
                    conversation_id: ConversationId::new(),
                    working_directory: None,
                    elevated: self.policy.is_elevated(),
                },
                json!({ "session_id": session_id }),
            )
            .await?;
        self.store
            .upsert_session(&handle, false)
            .map_err(store_error)?;
        Ok(output.content)
    }

    pub async fn persona_list(&self) -> ClawResult<String> {
        if let Some(report) = self.prompt_source.describe()? {
            let heartbeat_path = self.persona_dir.join("heartbeat.md");
            return Ok(report
                .with_runtime_entry(heartbeat_path)
                .render_persona_list());
        }

        let mut lines = Vec::new();
        for file in persona_file_names() {
            let path = self.persona_dir.join(file);
            let exists = fs::metadata(&path).is_ok();
            lines.push(format!(
                "{}\t{}",
                if exists { "present" } else { "missing" },
                path.display()
            ));
        }
        Ok(lines.join("\n"))
    }

    pub async fn persona_read(&self, name: &str) -> ClawResult<String> {
        let path = self.resolve_persona_file(name)?;
        self.tools
            .call(
                "read_file",
                ToolContext {
                    conversation_id: ConversationId::new(),
                    working_directory: Some(self.persona_dir.clone()),
                    elevated: self.policy.is_elevated(),
                },
                json!({ "path": path, "max_bytes": 12 * 1024 }),
            )
            .await
            .map(|output| output.content)
    }

    pub async fn persona_write(&self, name: &str, content: &str) -> ClawResult<String> {
        let path = self.resolve_persona_file(name)?;
        self.tools
            .call(
                "write_file",
                ToolContext {
                    conversation_id: ConversationId::new(),
                    working_directory: Some(self.persona_dir.clone()),
                    elevated: self.policy.is_elevated(),
                },
                json!({ "path": path, "content": content }),
            )
            .await
            .map(|output| output.content)
    }

    pub async fn persona_append(&self, name: &str, content: &str) -> ClawResult<String> {
        let path = self.resolve_persona_file(name)?;
        self.tools
            .call(
                "append_file",
                ToolContext {
                    conversation_id: ConversationId::new(),
                    working_directory: Some(self.persona_dir.clone()),
                    elevated: self.policy.is_elevated(),
                },
                json!({ "path": path, "content": content }),
            )
            .await
            .map(|output| output.content)
    }

    pub fn enable_elevated(&self) -> String {
        self.policy
            .enable_elevation(10, "manual elevated on".into());
        "elevated mode enabled for 10 minutes".into()
    }

    pub fn enable_full_elevated(&self) -> String {
        self.policy
            .enable_full_elevation("manual elevated full".into());
        "full elevated mode enabled until /elevated off".into()
    }

    pub fn enable_approval_mode(&self) -> String {
        self.policy.stop_elevation();
        "approval mode enabled. Guarded and dangerous commands will ask before running.".into()
    }

    pub async fn approve(&self, request_id: &str) -> ClawResult<String> {
        let approval_id = ApprovalId(
            uuid::Uuid::parse_str(request_id)
                .map_err(|err| ClawError::InvalidRequest(err.to_string()))?,
        );
        let approval = self
            .policy
            .approve(approval_id)
            .map(|request| ApprovalRecord {
                request,
                approved: Some(true),
                task_id: None,
                tool_name: None,
            })
            .or_else(|| self.store.get_approval_record(request_id).ok().flatten())
            .ok_or_else(|| ClawError::NotFound(format!("approval not found: {request_id}")))?;
        let approval = ApprovalRecord {
            approved: Some(true),
            ..approval
        };
        self.store
            .save_approval_record(&approval)
            .map_err(store_error)?;
        self.policy
            .enable_elevation(1, format!("approved {}", approval.request.request_id));
        let resumed = if let Some(task_id) = approval.task_id {
            Some(self.resume_task(task_id).await?)
        } else {
            None
        };
        self.policy.stop_elevation();
        let mut response = format!(
            "approved {} ({})",
            approval.request.command_preview, approval.request.request_id
        );
        if let Some(resumed) = resumed {
            response.push_str("\n\nresumed task result:\n");
            response.push_str(&resumed);
        }
        Ok(response)
    }

    pub fn stop_elevated(&self) -> String {
        self.policy.stop_elevation();
        "elevated mode disabled".into()
    }

    pub fn list_tasks(&self) -> ClawResult<Vec<TaskStatus>> {
        self.store.list_tasks().map_err(store_error)
    }

    pub fn list_pending_approvals(&self) -> ClawResult<Vec<ApprovalRecord>> {
        self.store.list_pending_approvals().map_err(store_error)
    }

    pub fn status(&self) -> ClawResult<String> {
        let tasks = self.store.list_tasks().map_err(store_error)?;
        let task_lines = if tasks.is_empty() {
            "no tasks".into()
        } else {
            tasks
                .into_iter()
                .take(5)
                .map(|task| {
                    format!(
                        "{} [{}] state={:?} running={} queued_at={}",
                        task.id, task.description, task.state, task.running, task.queued_at
                    )
                })
                .collect::<Vec<_>>()
                .join("\n")
        };
        let mode = match self.policy.current_mode() {
            PolicyMode::Normal => "normal",
            PolicyMode::ApprovalPending => "approval_pending",
            PolicyMode::ElevatedLease => "elevated",
        };
        let heartbeat_line = self
            .store
            .get_heartbeat()
            .map_err(store_error)?
            .map(|heartbeat| {
                let age = (Utc::now() - heartbeat.last_seen_at).num_seconds().max(0);
                format!(
                    "heartbeat_last_seen_at={}\nheartbeat_age_secs={}\nheartbeat_pid={}\nheartbeat_channel={}",
                    heartbeat.last_seen_at,
                    age,
                    heartbeat
                        .pid
                        .map(|pid| pid.to_string())
                        .unwrap_or_else(|| "unknown".into()),
                    heartbeat.channel.unwrap_or_else(|| "unknown".into())
                )
            })
            .unwrap_or_else(|| "heartbeat_last_seen_at=unknown".into());
        Ok(format!(
            "policy_mode={mode}\n{heartbeat_line}\n{task_lines}"
        ))
    }

    fn new_task_status(
        &self,
        task_id: TaskId,
        conversation_id: ConversationId,
        kind: TaskKind,
        description: &str,
        cwd: Option<PathBuf>,
        provider_id: Option<String>,
        current_session_id: Option<String>,
    ) -> TaskStatus {
        TaskStatus {
            id: task_id,
            conversation_id,
            kind,
            description: description.into(),
            queued_at: Utc::now(),
            started_at: Some(Utc::now()),
            finished_at: None,
            state: TaskRunState::Running,
            running: true,
            cwd,
            provider_id,
            current_session_id,
            result_preview: None,
            error: None,
            blocked_approval_id: None,
        }
    }

    fn ensure_tool_capable_provider(
        &self,
        provider_id: Option<&str>,
    ) -> ClawResult<ProviderRecord> {
        let provider = match provider_id {
            Some(provider_id) => self
                .store
                .get_provider(provider_id)
                .map_err(store_error)?
                .ok_or_else(|| ClawError::NotFound(format!("provider not found: {provider_id}")))?,
            None => self
                .store
                .get_default_provider()
                .map_err(store_error)?
                .or_else(|| self.store.get_first_provider().ok().flatten())
                .ok_or_else(|| ClawError::NotFound("no provider configured".into()))?,
        };
        if !provider.config.capabilities.tool_calling {
            return Err(ClawError::Backend(format!(
                "provider `{}` does not declare tool_calling capability; update the provider capabilities before using agent execution",
                provider.config.id
            )));
        }
        Ok(provider)
    }

    async fn execute_tool_call(
        &self,
        execution: &TaskExecution,
        call_id: Option<String>,
        tool_name: &str,
        arguments: serde_json::Value,
        working_session_id: Option<String>,
        sequence: i64,
    ) -> ClawResult<ToolCallOutcome> {
        let started_at = Utc::now();
        let tool_result = self
            .tools
            .call(
                tool_name,
                ToolContext {
                    conversation_id: execution.conversation_id,
                    working_directory: execution.cwd.clone(),
                    elevated: self.policy.is_elevated(),
                },
                arguments.clone(),
            )
            .await;

        match tool_result {
            Ok(result) => {
                let exchange = ToolExchange {
                    call: ToolCallRecord {
                        id: call_id.clone(),
                        name: tool_name.into(),
                        arguments: arguments.clone(),
                    },
                    result: ToolResultRecord {
                        call_id: call_id.clone(),
                        name: tool_name.into(),
                        content: render_tool_output(&result),
                        structured: result.structured.clone(),
                    },
                };
                self.store
                    .save_tool_invocation(&ToolInvocationRecord {
                        task_id: execution.task_id,
                        conversation_id: execution.conversation_id,
                        call_id,
                        tool_name: tool_name.into(),
                        arguments: arguments.clone(),
                        status: ToolInvocationStatus::Completed,
                        output_content: Some(exchange.result.content.clone()),
                        output_structured: result.structured.clone(),
                        error: None,
                        approval_id: None,
                        sequence,
                        created_at: started_at,
                        updated_at: Utc::now(),
                    })
                    .map_err(store_error)?;
                self.persist_session_and_audit(
                    execution.task_id,
                    tool_name,
                    &arguments,
                    &result,
                    working_session_id,
                )
                .await?;
                Ok(ToolCallOutcome::Completed(exchange))
            }
            Err(err @ ClawError::ApprovalRequired(_)) => {
                let approval_id = extract_approval_id(&err);
                let approval_record =
                    approval_id
                        .and_then(|id| self.policy.get_approval(id))
                        .map(|request| ApprovalRecord {
                            request,
                            approved: None,
                            task_id: Some(execution.task_id),
                            tool_name: Some(tool_name.into()),
                        });
                if let Some(approval_record) = approval_record.clone() {
                    self.store
                        .save_approval_record(&approval_record)
                        .map_err(store_error)?;
                }
                let blocked_approval_id = approval_record
                    .as_ref()
                    .map(|record| record.request.request_id);
                self.store
                    .save_tool_invocation(&ToolInvocationRecord {
                        task_id: execution.task_id,
                        conversation_id: execution.conversation_id,
                        call_id,
                        tool_name: tool_name.into(),
                        arguments,
                        status: ToolInvocationStatus::BlockedApproval,
                        output_content: None,
                        output_structured: None,
                        error: Some(err.to_string()),
                        approval_id: blocked_approval_id,
                        sequence,
                        created_at: started_at,
                        updated_at: Utc::now(),
                    })
                    .map_err(store_error)?;
                let mut status = self
                    .store
                    .get_task(execution.task_id)
                    .map_err(store_error)?
                    .ok_or_else(|| {
                        ClawError::NotFound(format!("task not found: {}", execution.task_id))
                    })?;
                status.running = false;
                status.state = TaskRunState::BlockedApproval;
                status.finished_at = None;
                status.blocked_approval_id = blocked_approval_id;
                status.error = None;
                self.store.upsert_task(&status).map_err(store_error)?;
                Ok(ToolCallOutcome::Blocked(err))
            }
            Err(err) => {
                self.store
                    .save_tool_invocation(&ToolInvocationRecord {
                        task_id: execution.task_id,
                        conversation_id: execution.conversation_id,
                        call_id,
                        tool_name: tool_name.into(),
                        arguments,
                        status: ToolInvocationStatus::Failed,
                        output_content: None,
                        output_structured: None,
                        error: Some(err.to_string()),
                        approval_id: None,
                        sequence,
                        created_at: started_at,
                        updated_at: Utc::now(),
                    })
                    .map_err(store_error)?;
                Err(err)
            }
        }
    }

    async fn persist_session_and_audit(
        &self,
        task_id: TaskId,
        tool_name: &str,
        arguments: &serde_json::Value,
        result: &hajimi_claw_types::ToolOutput,
        session_id_hint: Option<String>,
    ) -> ClawResult<()> {
        if let Some(session_id) = result
            .structured
            .as_ref()
            .and_then(|value| value.get("session_id"))
            .and_then(|value| value.as_str())
            .map(str::to_string)
            .or(session_id_hint)
        {
            if let Ok(handle) = self.fetch_session_handle(&session_id).await {
                self.store
                    .upsert_session(&handle, true)
                    .map_err(store_error)?;
                let mut task = self
                    .store
                    .get_task(task_id)
                    .map_err(store_error)?
                    .ok_or_else(|| ClawError::NotFound(format!("task not found: {task_id}")))?;
                task.current_session_id = Some(session_id);
                task.cwd = Some(handle.cwd);
                self.store.upsert_task(&task).map_err(store_error)?;
            }
        }

        if let Some(structured) = &result.structured {
            if let Some(exit_code) = structured.get("exit_code").and_then(|value| value.as_i64()) {
                let duration_ms = structured
                    .get("duration_ms")
                    .and_then(|value| value.as_u64())
                    .unwrap_or_default() as u128;
                self.store
                    .append_command_audit(
                        Some(task_id),
                        structured
                            .get("session_id")
                            .and_then(|value| value.as_str())
                            .map(str::to_string),
                        &render_command_preview(tool_name, arguments),
                        Some(exit_code as i32),
                        duration_ms,
                    )
                    .map_err(store_error)?;
            }
        }
        Ok(())
    }

    async fn fetch_session_handle(
        &self,
        session_id: &str,
    ) -> ClawResult<hajimi_claw_types::SessionHandle> {
        let output = self
            .tools
            .call(
                "session_status",
                ToolContext {
                    conversation_id: ConversationId::new(),
                    working_directory: None,
                    elevated: self.policy.is_elevated(),
                },
                json!({ "session_id": session_id }),
            )
            .await?;
        let structured = output
            .structured
            .ok_or_else(|| ClawError::Backend("session_status missing structured data".into()))?;
        let session_uuid = structured
            .get("session_id")
            .and_then(|value| value.as_str())
            .ok_or_else(|| ClawError::Backend("session_status missing session_id".into()))?;
        Ok(hajimi_claw_types::SessionHandle {
            id: hajimi_claw_types::SessionId(
                uuid::Uuid::parse_str(session_uuid)
                    .map_err(|err| ClawError::Backend(err.to_string()))?,
            ),
            name: structured
                .get("name")
                .and_then(|value| value.as_str())
                .unwrap_or("shell")
                .to_string(),
            cwd: structured
                .get("cwd")
                .and_then(|value| value.as_str())
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from(".")),
            created_at: structured
                .get("created_at")
                .and_then(|value| value.as_str())
                .map(parse_rfc3339)
                .transpose()?
                .unwrap_or_else(Utc::now),
            last_used_at: structured
                .get("last_used_at")
                .and_then(|value| value.as_str())
                .map(parse_rfc3339)
                .transpose()?
                .unwrap_or_else(Utc::now),
        })
    }

    async fn run_llm(&self, execution: TaskExecution) -> ClawResult<String> {
        let provider = self.ensure_tool_capable_provider(execution.provider_id.as_deref())?;
        let mut system_prompt = self.prompt_source.load()?;
        if let Some(session_id) = &execution.current_session_id {
            if let Ok(handle) = self.fetch_session_handle(session_id).await {
                system_prompt.push_str(&format!(
                    "\n\nCurrent shell session:\n- session_id={}\n- cwd={}",
                    session_id,
                    handle.cwd.display()
                ));
            }
        }
        let mut tool_history = self.load_completed_tool_history(execution.task_id)?;
        let blocked = self.load_blocked_tool_invocation(execution.task_id)?;
        let all_messages = self
            .store
            .list_messages(execution.conversation_id, 10_000)
            .map_err(store_error)?;

        for _ in 0..8 {
            let messages = trim_messages_to_context_limit(
                all_messages.clone(),
                provider.config.capabilities.max_context_chars,
                system_prompt.len(),
                &tool_history,
            );
            let should_resume_blocked = blocked.as_ref().is_some_and(|blocked_record| {
                !tool_history.iter().any(|item| {
                    item.call.id == blocked_record.call_id
                        && item.call.name == blocked_record.tool_name
                })
            });
            if let Some(blocked) = blocked.as_ref().filter(|_| should_resume_blocked) {
                match self
                    .execute_tool_call(
                        &execution,
                        blocked.call_id.clone(),
                        &blocked.tool_name,
                        blocked.arguments.clone(),
                        execution.current_session_id.clone(),
                        blocked.sequence,
                    )
                    .await?
                {
                    ToolCallOutcome::Completed(exchange) => {
                        tool_history.push(exchange);
                        continue;
                    }
                    ToolCallOutcome::Blocked(err) => return Err(err),
                }
            }
            let request = AgentRequest {
                conversation_id: execution.conversation_id,
                provider_id: Some(provider.config.id.clone()),
                system_prompt: system_prompt.clone(),
                messages: messages.clone(),
                tool_specs: self.tools.specs(),
                tool_history: tool_history.clone(),
            };
            let llm_output = collect_response_events(self.llm.clone(), request).await?;
            if llm_output.tool_calls.is_empty() {
                return Ok(llm_output.text);
            }

            for call in llm_output.tool_calls {
                match self
                    .execute_tool_call(
                        &execution,
                        call.id.clone(),
                        &call.tool,
                        call.input.clone(),
                        execution.current_session_id.clone(),
                        tool_history.len() as i64 + 1,
                    )
                    .await?
                {
                    ToolCallOutcome::Completed(exchange) => tool_history.push(exchange),
                    ToolCallOutcome::Blocked(err) => return Err(err),
                }
            }
        }

        Err(ClawError::Backend(
            "agent exceeded tool-calling iteration limit".into(),
        ))
    }

    fn load_completed_tool_history(&self, task_id: TaskId) -> ClawResult<Vec<ToolExchange>> {
        let invocations = self
            .store
            .list_tool_invocations(task_id)
            .map_err(store_error)?;
        let latest = latest_tool_invocations(invocations);
        Ok(latest
            .into_iter()
            .filter(|record| matches!(record.status, ToolInvocationStatus::Completed))
            .map(|record| ToolExchange {
                call: ToolCallRecord {
                    id: record.call_id.clone(),
                    name: record.tool_name.clone(),
                    arguments: record.arguments.clone(),
                },
                result: ToolResultRecord {
                    call_id: record.call_id.clone(),
                    name: record.tool_name,
                    content: record.output_content.unwrap_or_default(),
                    structured: record.output_structured,
                },
            })
            .collect())
    }

    fn load_blocked_tool_invocation(
        &self,
        task_id: TaskId,
    ) -> ClawResult<Option<ToolInvocationRecord>> {
        let invocations = self
            .store
            .list_tool_invocations(task_id)
            .map_err(store_error)?;
        Ok(latest_tool_invocations(invocations)
            .into_iter()
            .find(|record| matches!(record.status, ToolInvocationStatus::BlockedApproval)))
    }

    async fn resume_task(&self, task_id: TaskId) -> ClawResult<String> {
        let mut task = self
            .store
            .get_task(task_id)
            .map_err(store_error)?
            .ok_or_else(|| ClawError::NotFound(format!("task not found: {task_id}")))?;
        task.running = true;
        task.state = TaskRunState::Running;
        task.error = None;
        task.blocked_approval_id = None;
        self.store.upsert_task(&task).map_err(store_error)?;
        let execution = TaskExecution {
            task_id: task.id,
            conversation_id: task.conversation_id,
            prompt: task.description.clone(),
            cwd: task.cwd.clone(),
            provider_id: task.provider_id.clone(),
            current_session_id: task.current_session_id.clone(),
        };

        let result = if execution.provider_id.is_none() && select_tool(&execution.prompt).is_some()
        {
            let blocked = self.load_blocked_tool_invocation(task_id)?.ok_or_else(|| {
                ClawError::NotFound(format!(
                    "blocked tool invocation not found for task {task_id}"
                ))
            })?;
            match self
                .execute_tool_call(
                    &execution,
                    blocked.call_id.clone(),
                    &blocked.tool_name,
                    blocked.arguments.clone(),
                    execution.current_session_id.clone(),
                    blocked.sequence,
                )
                .await?
            {
                ToolCallOutcome::Completed(exchange) => Ok(exchange.result.content),
                ToolCallOutcome::Blocked(err) => Err(err),
            }
        } else {
            self.run_llm(execution).await
        };

        match &result {
            Ok(content) => {
                task.running = false;
                task.state = TaskRunState::Completed;
                task.finished_at = Some(Utc::now());
                task.result_preview = Some(clamp_chars(content, 400));
                task.error = None;
                task.blocked_approval_id = None;
                self.store.upsert_task(&task).map_err(store_error)?;
                self.store
                    .save_message(
                        task.conversation_id,
                        &ConversationMessage {
                            role: MessageRole::Assistant,
                            content: content.clone(),
                            created_at: Utc::now(),
                        },
                    )
                    .map_err(store_error)?;
            }
            Err(err) => {
                task.running = false;
                task.state = TaskRunState::Failed;
                task.finished_at = Some(Utc::now());
                task.error = Some(err.to_string());
                self.store.upsert_task(&task).map_err(store_error)?;
            }
        }
        result
    }

    async fn run_multi_agent(
        &self,
        conversation_id: ConversationId,
        prompt: &str,
        provider_id: Option<String>,
        preference: MultiAgentPreference,
    ) -> ClawResult<String> {
        let config = self.multi_agent.normalized();
        let worker_count = resolve_worker_count(prompt, &config, preference);
        let base_prompt = self.prompt_source.load()?;
        let clamped_prompt = clamp_chars(prompt, config.max_context_chars_per_worker);
        let coordinator_plan = match self
            .run_llm_request(AgentRequest {
                conversation_id,
                provider_id: provider_id.clone(),
                system_prompt: format!(
                    "{base_prompt}\n\nYou are the coordinator in multi-agent mode. Break the user request into {worker_count} complementary worker assignments. Return exactly {worker_count} lines. Each line must start with `WORKER <n>:` and contain only that worker's assignment."
                ),
                messages: vec![ConversationMessage {
                    role: MessageRole::User,
                    content: clamped_prompt.clone(),
                    created_at: Utc::now(),
                }],
                tool_specs: vec![],
                tool_history: vec![],
            })
            .await
        {
            Ok(plan) if !plan.trim().is_empty() => plan,
            Ok(_) => generic_coordinator_plan(prompt, worker_count),
            Err(_) => generic_coordinator_plan(prompt, worker_count),
        };
        let worker_briefs = parse_worker_briefs(&coordinator_plan, worker_count)
            .unwrap_or_else(|| generic_worker_briefs(prompt, worker_count));

        let mut join_set = JoinSet::new();
        for (index, brief) in worker_briefs.iter().enumerate() {
            let llm = self.llm.clone();
            let provider_id = provider_id.clone();
            let brief = brief.clone();
            let original_prompt = clamped_prompt.clone();
            let system_prompt = format!(
                "{base_prompt}\n\nYou are worker {}/{} in multi-agent mode. Focus only on your assignment. Do not assume tool access. Produce a concise analysis with findings, risks, and next actions.",
                index + 1,
                worker_count
            );
            let timeout = Duration::from_secs(config.worker_timeout_secs);
            join_set.spawn(async move {
                let request = AgentRequest {
                    conversation_id: ConversationId::new(),
                    provider_id,
                    system_prompt,
                    messages: vec![ConversationMessage {
                        role: MessageRole::User,
                        content: format!(
                            "Original user request:\n{original_prompt}\n\nCoordinator assignment:\n{brief}"
                        ),
                        created_at: Utc::now(),
                    }],
                    tool_specs: vec![],
                    tool_history: vec![],
                };
                let result = time::timeout(timeout, collect_text_response(llm, request)).await;
                match result {
                    Ok(Ok(text)) => Ok((index, brief, text)),
                    Ok(Err(err)) => Ok((
                        index,
                        brief,
                        format!("worker {} failed: {}", index + 1, err),
                    )),
                    Err(_) => Ok((
                        index,
                        brief,
                        format!(
                            "worker {} timed out after {}s",
                            index + 1,
                            timeout.as_secs()
                        ),
                    )),
                }
            });
        }

        let mut worker_sections = vec![String::new(); worker_count];
        while let Some(result) = join_set.join_next().await {
            let (index, brief, text) = result
                .map_err(|err| ClawError::Backend(err.to_string()))?
                .map_err(|err: ClawError| err)?;
            worker_sections[index] = format!(
                "Worker {} assignment:\n{}\n\nWorker {} output:\n{}",
                index + 1,
                brief,
                index + 1,
                clamp_chars(&text, config.max_context_chars_per_worker)
            );
        }

        let summary_block = worker_sections.join("\n\n");
        match self
            .run_llm_request(AgentRequest {
                conversation_id,
                provider_id,
                system_prompt: format!(
                    "{base_prompt}\n\nYou are the integrator in multi-agent mode. Merge the worker outputs into one clear answer for the user. Be direct. Mention that multi-agent mode was used only when it helps clarity."
                ),
                messages: vec![ConversationMessage {
                    role: MessageRole::User,
                    content: format!(
                        "Original user request:\n{}\n\nCoordinator plan:\n{}\n\nWorker results:\n{}",
                        clamped_prompt,
                        coordinator_plan,
                        clamp_chars(&summary_block, config.max_context_chars_per_worker)
                    ),
                    created_at: Utc::now(),
                }],
                tool_specs: vec![],
                tool_history: vec![],
            })
            .await
        {
            Ok(text) if !text.trim().is_empty() => Ok(text),
            Ok(_) | Err(_) => Ok(format!(
                "I used {worker_count} sub-agents for this request.\n\nCoordinator plan:\n{coordinator_plan}\n\n{summary_block}"
            )),
        }
    }

    async fn run_llm_request(&self, request: AgentRequest) -> ClawResult<String> {
        collect_text_response(self.llm.clone(), request).await
    }
}

fn trim_messages_to_context_limit(
    messages: Vec<ConversationMessage>,
    max_context_chars: Option<usize>,
    system_prompt_chars: usize,
    tool_history: &[ToolExchange],
) -> Vec<ConversationMessage> {
    let Some(max_context_chars) = max_context_chars else {
        return messages;
    };
    let reserved = system_prompt_chars
        .saturating_add(approx_tool_history_chars(tool_history))
        .saturating_add(1024);
    let budget = max_context_chars.saturating_sub(reserved);
    if budget == 0 {
        return messages
            .into_iter()
            .rev()
            .take(1)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect();
    }

    let mut kept = Vec::new();
    let mut used = 0usize;
    for message in messages.into_iter().rev() {
        let message_chars = message.content.chars().count().saturating_add(32);
        if !kept.is_empty() && used.saturating_add(message_chars) > budget {
            break;
        }
        used = used.saturating_add(message_chars);
        kept.push(message);
    }
    kept.reverse();
    kept
}

fn approx_tool_history_chars(tool_history: &[ToolExchange]) -> usize {
    tool_history
        .iter()
        .map(|exchange| {
            exchange.call.name.chars().count()
                + exchange.result.content.chars().count()
                + exchange.call.arguments.to_string().chars().count()
                + 128
        })
        .sum()
}

async fn collect_text_response(
    llm: Arc<dyn LlmBackend>,
    request: AgentRequest,
) -> ClawResult<String> {
    Ok(collect_response_events(llm, request).await?.text)
}

struct LlmResponseEvents {
    text: String,
    tool_calls: Vec<ToolCallEvent>,
}

struct ToolCallEvent {
    id: Option<String>,
    tool: String,
    input: serde_json::Value,
}

async fn collect_response_events(
    llm: Arc<dyn LlmBackend>,
    request: AgentRequest,
) -> ClawResult<LlmResponseEvents> {
    let stream = llm.respond(request).await?;
    let events = stream.try_collect::<Vec<_>>().await?;
    let mut text = String::new();
    let mut tool_calls = Vec::new();
    for event in events {
        match event {
            AgentEvent::TextDelta(delta) => text.push_str(&delta),
            AgentEvent::ToolCall { id, tool, input } => {
                tool_calls.push(ToolCallEvent { id, tool, input });
            }
            AgentEvent::Finished => {}
        }
    }
    Ok(LlmResponseEvents { text, tool_calls })
}

fn render_tool_output(output: &hajimi_claw_types::ToolOutput) -> String {
    output
        .structured
        .as_ref()
        .map(|value| serde_json::to_string(value).unwrap_or_else(|_| output.content.clone()))
        .unwrap_or_else(|| output.content.clone())
}

fn latest_tool_invocations(records: Vec<ToolInvocationRecord>) -> Vec<ToolInvocationRecord> {
    let mut latest = BTreeMap::new();
    for record in records {
        latest.insert(record.sequence, record);
    }
    latest.into_values().collect()
}

fn extract_approval_id(err: &ClawError) -> Option<ApprovalId> {
    let ClawError::ApprovalRequired(message) = err else {
        return None;
    };
    let start = message.rfind('[')? + 1;
    let end = message.rfind(']')?;
    if end <= start {
        return None;
    }
    uuid::Uuid::parse_str(&message[start..end])
        .ok()
        .map(ApprovalId)
}

fn render_command_preview(tool_name: &str, arguments: &serde_json::Value) -> String {
    if let Some(command) = arguments.get("command").and_then(|value| value.as_str()) {
        return command.to_string();
    }
    if let Some(command) = arguments.get("service").and_then(|value| value.as_str()) {
        return format!("{tool_name} {command}");
    }
    if let Some(command) = arguments.get("container").and_then(|value| value.as_str()) {
        return format!("{tool_name} {command}");
    }
    tool_name.to_string()
}

fn parse_rfc3339(raw: &str) -> ClawResult<chrono::DateTime<Utc>> {
    chrono::DateTime::parse_from_rfc3339(raw)
        .map(|value| value.with_timezone(&Utc))
        .map_err(|err| ClawError::Backend(err.to_string()))
}

fn store_error(err: anyhow::Error) -> ClawError {
    ClawError::Backend(err.to_string())
}

impl AgentRuntime {
    fn resolve_persona_file(&self, raw: &str) -> ClawResult<PathBuf> {
        let trimmed = raw.trim().trim_matches('`');
        let canonical = match trimmed.to_ascii_lowercase().as_str() {
            "heartbeat" | "heartbeat.md" => "heartbeat.md",
            "soul" | "soul.md" => "soul.md",
            "agents" | "agents.md" => "agents.md",
            "tools" | "tools.md" => "tools.md",
            "skills" | "skills.md" => "skills.md",
            "identity" | "identity.md" => "identity.md",
            _ => {
                return Err(ClawError::InvalidRequest(
                    "persona file must be one of: identity, heartbeat, soul, agents, tools, skills"
                        .into(),
                ));
            }
        };
        Ok(self.persona_dir.join(canonical))
    }
}

pub fn default_system_prompt() -> &'static str {
    "You are hajimi-claw, a single-user execution agent with terminal and network tools.

Rules:
- If the user asks to inspect, test, verify, run, diagnose, fetch, restart, or check something, prefer tools over speculative prose.
- Do not claim you cannot access the machine or network when a relevant tool exists.
- For shell work, prefer `exec_once` or the shell session tools.
- For web and network diagnostics, prefer `http_probe`, `curl_request`, `dns_lookup`, `port_check`, `tls_check`, and `ping_host`.
- Use structured tools before raw shell when a structured tool fits.
- Be concise, factual, and action-oriented.
- When a tool result is enough to answer, summarize the result directly instead of narrating your limitations."
}

pub trait SystemPromptSource: Send + Sync {
    fn load(&self) -> ClawResult<String>;

    fn describe(&self) -> ClawResult<Option<PromptSourceReport>> {
        Ok(None)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PromptSourceMode {
    AutoDiscovery,
    ExplicitList,
}

impl PromptSourceMode {
    fn label(self) -> &'static str {
        match self {
            Self::AutoDiscovery => "auto-discovery",
            Self::ExplicitList => "explicit [persona].prompt_files",
        }
    }

    fn precedence_label(self) -> &'static str {
        match self {
            Self::AutoDiscovery => {
                "persona.directory -> config directory -> current working directory"
            }
            Self::ExplicitList => "uses the configured prompt_files order",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum PromptLayer {
    Identity,
    Soul,
    OperationalExtensions,
    RuntimeConfig,
}

impl PromptLayer {
    fn heading(self) -> &'static str {
        match self {
            Self::Identity => "identity layer",
            Self::Soul => "soul layer",
            Self::OperationalExtensions => "operational extensions",
            Self::RuntimeConfig => "runtime config",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PromptParseMode {
    Structured,
    Legacy,
    Markdown,
    Missing,
    Empty,
    RuntimeOnly,
}

impl PromptParseMode {
    fn label(self) -> &'static str {
        match self {
            Self::Structured => "structured",
            Self::Legacy => "legacy",
            Self::Markdown => "markdown",
            Self::Missing => "missing",
            Self::Empty => "empty",
            Self::RuntimeOnly => "runtime-only",
        }
    }
}

#[derive(Debug, Clone)]
pub struct PromptSourceEntry {
    pub path: PathBuf,
    pub layer: PromptLayer,
    pub label: String,
    pub included: bool,
    pub parse_mode: PromptParseMode,
}

impl PromptSourceEntry {
    fn new(
        path: PathBuf,
        layer: PromptLayer,
        label: String,
        included: bool,
        parse_mode: PromptParseMode,
    ) -> Self {
        Self {
            path,
            layer,
            label,
            included,
            parse_mode,
        }
    }

    fn runtime_only(path: PathBuf) -> Self {
        Self::new(
            path.clone(),
            PromptLayer::RuntimeConfig,
            file_label(&path),
            false,
            PromptParseMode::RuntimeOnly,
        )
    }
}

#[derive(Debug, Clone)]
pub struct PromptSourceReport {
    pub mode: PromptSourceMode,
    pub entries: Vec<PromptSourceEntry>,
}

impl PromptSourceReport {
    pub fn contains_path(&self, path: &Path) -> bool {
        self.entries.iter().any(|entry| entry.path == path)
    }

    pub fn with_runtime_entry(mut self, path: PathBuf) -> Self {
        if !self.contains_path(&path) {
            self.entries.push(PromptSourceEntry::runtime_only(path));
        }
        self
    }

    pub fn render_persona_list(&self) -> String {
        let mut lines = vec![
            "persona effective view".to_string(),
            format!("source mode: {}", self.mode.label()),
            format!("precedence: {}", self.mode.precedence_label()),
            "prompt order: base system prompt -> identity layer -> soul layer -> operational extensions -> runtime overlays".to_string(),
        ];
        for layer in [
            PromptLayer::Identity,
            PromptLayer::Soul,
            PromptLayer::OperationalExtensions,
            PromptLayer::RuntimeConfig,
        ] {
            let entries = self
                .entries
                .iter()
                .filter(|entry| entry.layer == layer)
                .collect::<Vec<_>>();
            if entries.is_empty() {
                continue;
            }
            lines.push(String::new());
            lines.push(format!("[{}]", layer.heading()));
            for entry in entries {
                let status = if entry.included {
                    "included"
                } else {
                    "excluded"
                };
                lines.push(format!(
                    "{}\t{}\t{}\t{}",
                    status,
                    entry.label,
                    entry.parse_mode.label(),
                    entry.path.display()
                ));
            }
        }
        lines.join("\n")
    }
}

pub struct StaticSystemPrompt {
    prompt: String,
}

impl StaticSystemPrompt {
    pub fn new(prompt: impl Into<String>) -> Self {
        Self {
            prompt: prompt.into(),
        }
    }
}

impl SystemPromptSource for StaticSystemPrompt {
    fn load(&self) -> ClawResult<String> {
        Ok(self.prompt.clone())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PersonaFileKind {
    Identity,
    Soul,
    Agents,
    Tools,
    Skills,
    Heartbeat,
    Additional,
}

impl PersonaFileKind {
    fn layer(self) -> PromptLayer {
        match self {
            Self::Identity => PromptLayer::Identity,
            Self::Soul => PromptLayer::Soul,
            Self::Heartbeat => PromptLayer::RuntimeConfig,
            Self::Agents | Self::Tools | Self::Skills | Self::Additional => {
                PromptLayer::OperationalExtensions
            }
        }
    }
}

#[derive(Debug, Clone, Default)]
struct StructuredPersonaLayer {
    fields: BTreeMap<String, Vec<String>>,
    notes: Vec<String>,
}

impl StructuredPersonaLayer {
    fn is_empty(&self) -> bool {
        self.fields.values().all(Vec::is_empty) && self.notes.is_empty()
    }
}

#[derive(Debug, Clone)]
enum ParsedPersonaLayer {
    Structured {
        fields: BTreeMap<String, Vec<String>>,
        body: String,
    },
    Legacy(String),
}

#[derive(Debug, Clone)]
struct ExtensionBlock {
    title: &'static str,
    content: String,
}

impl ExtensionBlock {
    fn new(kind: PersonaFileKind, content: String) -> Self {
        let title = match kind {
            PersonaFileKind::Agents => "Agents and delegation",
            PersonaFileKind::Tools => "Tools and safety",
            PersonaFileKind::Skills => "Skills and workflows",
            PersonaFileKind::Additional => "Additional guidance",
            _ => "Additional guidance",
        };
        Self { title, content }
    }
}

#[derive(Debug, Clone)]
struct CompiledPromptSource {
    report: PromptSourceReport,
    identity: StructuredPersonaLayer,
    soul: StructuredPersonaLayer,
    extensions: Vec<ExtensionBlock>,
}

pub struct MarkdownPromptSource {
    base_prompt: String,
    files: Vec<PathBuf>,
    mode: PromptSourceMode,
}

impl MarkdownPromptSource {
    pub fn new(base_prompt: impl Into<String>, files: Vec<PathBuf>) -> Self {
        Self::with_mode(base_prompt, files, PromptSourceMode::AutoDiscovery)
    }

    pub fn with_mode(
        base_prompt: impl Into<String>,
        files: Vec<PathBuf>,
        mode: PromptSourceMode,
    ) -> Self {
        Self {
            base_prompt: base_prompt.into(),
            files,
            mode,
        }
    }

    fn compile(&self) -> ClawResult<CompiledPromptSource> {
        let mut entries = Vec::new();
        let mut identity = StructuredPersonaLayer::default();
        let mut soul = StructuredPersonaLayer::default();
        let mut extensions = Vec::new();

        for path in &self.files {
            let label = file_label(path);
            let kind = classify_persona_file(path);
            if matches!(kind, PersonaFileKind::Heartbeat) {
                entries.push(PromptSourceEntry::new(
                    path.clone(),
                    kind.layer(),
                    label,
                    false,
                    PromptParseMode::RuntimeOnly,
                ));
                continue;
            }

            let content = match fs::read_to_string(path) {
                Ok(content) => content,
                Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
                    entries.push(PromptSourceEntry::new(
                        path.clone(),
                        kind.layer(),
                        label,
                        false,
                        PromptParseMode::Missing,
                    ));
                    continue;
                }
                Err(err) => return Err(ClawError::Backend(err.to_string())),
            };
            let trimmed = content.trim();
            if trimmed.is_empty() {
                entries.push(PromptSourceEntry::new(
                    path.clone(),
                    kind.layer(),
                    label,
                    false,
                    PromptParseMode::Empty,
                ));
                continue;
            }

            match kind {
                PersonaFileKind::Identity => {
                    let parse_mode =
                        merge_structured_layer(&mut identity, parse_persona_layer(trimmed));
                    entries.push(PromptSourceEntry::new(
                        path.clone(),
                        kind.layer(),
                        label,
                        true,
                        parse_mode,
                    ));
                }
                PersonaFileKind::Soul => {
                    let parse_mode =
                        merge_structured_layer(&mut soul, parse_persona_layer(trimmed));
                    entries.push(PromptSourceEntry::new(
                        path.clone(),
                        kind.layer(),
                        label,
                        true,
                        parse_mode,
                    ));
                }
                PersonaFileKind::Agents
                | PersonaFileKind::Tools
                | PersonaFileKind::Skills
                | PersonaFileKind::Additional => {
                    extensions.push(ExtensionBlock::new(kind, clamp_prompt_content(trimmed)));
                    entries.push(PromptSourceEntry::new(
                        path.clone(),
                        kind.layer(),
                        label,
                        true,
                        PromptParseMode::Markdown,
                    ));
                }
                PersonaFileKind::Heartbeat => unreachable!(),
            }
        }

        Ok(CompiledPromptSource {
            report: PromptSourceReport {
                mode: self.mode,
                entries,
            },
            identity,
            soul,
            extensions,
        })
    }
}

impl SystemPromptSource for MarkdownPromptSource {
    fn load(&self) -> ClawResult<String> {
        let compiled = self.compile()?;
        let mut prompt = self.base_prompt.clone();
        let mut sections = Vec::new();
        if let Some(section) = render_structured_persona_layer(
            "Identity layer",
            &[
                "summary",
                "owned_systems",
                "environments",
                "standing_preferences",
                "hard_constraints",
            ],
            &compiled.identity,
        ) {
            sections.push(section);
        }
        if let Some(section) = render_structured_persona_layer(
            "Soul layer",
            &[
                "name",
                "role",
                "tone",
                "style",
                "non_goals",
                "behavioral_rules",
            ],
            &compiled.soul,
        ) {
            sections.push(section);
        }
        if let Some(section) = render_operational_extensions(&compiled.extensions) {
            sections.push(section);
        }
        if !sections.is_empty() {
            prompt.push_str(
                "\n\nApply the layered persona context below after the base system rules. Identity captures the user and durable environment. Soul defines Hajimi's stable character. Operational extensions add workflow, delegation, and tool guidance.\n\n",
            );
            prompt.push_str(&sections.join("\n\n"));
        }
        Ok(prompt)
    }

    fn describe(&self) -> ClawResult<Option<PromptSourceReport>> {
        Ok(Some(self.compile()?.report))
    }
}

fn classify_persona_file(path: &Path) -> PersonaFileKind {
    match file_label(path).to_ascii_lowercase().as_str() {
        "identity.md" => PersonaFileKind::Identity,
        "soul.md" => PersonaFileKind::Soul,
        "agents.md" => PersonaFileKind::Agents,
        "tools.md" => PersonaFileKind::Tools,
        "skills.md" => PersonaFileKind::Skills,
        "heartbeat.md" => PersonaFileKind::Heartbeat,
        _ => PersonaFileKind::Additional,
    }
}

fn merge_structured_layer(
    target: &mut StructuredPersonaLayer,
    parsed: ParsedPersonaLayer,
) -> PromptParseMode {
    match parsed {
        ParsedPersonaLayer::Structured { fields, body } => {
            for (key, value) in fields {
                target.fields.insert(key, value);
            }
            let body = clamp_prompt_content(body.trim());
            if !body.is_empty() {
                target.notes.push(body);
            }
            PromptParseMode::Structured
        }
        ParsedPersonaLayer::Legacy(body) => {
            let body = clamp_prompt_content(body.trim());
            if !body.is_empty() {
                target.notes.push(body);
            }
            PromptParseMode::Legacy
        }
    }
}

fn parse_persona_layer(content: &str) -> ParsedPersonaLayer {
    let trimmed = content.trim();
    let Some((front_matter, body)) = split_front_matter(trimmed) else {
        return ParsedPersonaLayer::Legacy(trimmed.to_string());
    };
    let Some(fields) = parse_front_matter_map(front_matter) else {
        return ParsedPersonaLayer::Legacy(trimmed.to_string());
    };
    ParsedPersonaLayer::Structured {
        fields,
        body: body.trim().to_string(),
    }
}

fn split_front_matter(content: &str) -> Option<(&str, &str)> {
    static FRONT_MATTER_RE: std::sync::OnceLock<Regex> = std::sync::OnceLock::new();
    let regex = FRONT_MATTER_RE.get_or_init(|| {
        Regex::new(r"\A---\r?\n(?P<front>[\s\S]*?)\r?\n---(?:\r?\n(?P<body>[\s\S]*))?\z")
            .expect("valid front matter regex")
    });
    let captures = regex.captures(content)?;
    Some((
        captures.name("front")?.as_str(),
        captures
            .name("body")
            .map(|value| value.as_str())
            .unwrap_or(""),
    ))
}

fn parse_front_matter_map(front_matter: &str) -> Option<BTreeMap<String, Vec<String>>> {
    let lines = front_matter.lines().collect::<Vec<_>>();
    let mut index = 0usize;
    let mut fields = BTreeMap::new();
    while index < lines.len() {
        let line = lines[index].trim_end();
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            index += 1;
            continue;
        }
        if line.starts_with(' ') || line.starts_with('\t') {
            return None;
        }
        let (raw_key, raw_value) = trimmed.split_once(':')?;
        let key = normalize_front_matter_key(raw_key);
        if key.is_empty() {
            return None;
        }
        let value = raw_value.trim();
        if !value.is_empty() {
            fields.insert(key, vec![normalize_front_matter_value(value)]);
            index += 1;
            continue;
        }

        index += 1;
        let mut values = Vec::new();
        while index < lines.len() {
            let next = lines[index];
            let next_trimmed = next.trim();
            if next_trimmed.is_empty() {
                index += 1;
                continue;
            }
            if !next.starts_with(' ') && !next.starts_with('\t') {
                break;
            }
            if let Some(item) = next_trimmed.strip_prefix("- ") {
                values.push(normalize_front_matter_value(item.trim()));
            } else {
                values.push(normalize_front_matter_value(next_trimmed));
            }
            index += 1;
        }
        fields.insert(key, values);
    }
    Some(
        fields
            .into_iter()
            .map(|(key, values)| {
                (
                    key,
                    values
                        .into_iter()
                        .map(|value| clamp_prompt_content(value.trim()))
                        .collect(),
                )
            })
            .collect(),
    )
}

fn normalize_front_matter_key(value: &str) -> String {
    let mut normalized = String::new();
    let mut last_was_underscore = false;
    for ch in value.trim().chars() {
        if ch.is_ascii_alphanumeric() {
            normalized.push(ch.to_ascii_lowercase());
            last_was_underscore = false;
        } else if !last_was_underscore {
            normalized.push('_');
            last_was_underscore = true;
        }
    }
    normalized.trim_matches('_').to_string()
}

fn normalize_front_matter_value(value: &str) -> String {
    let trimmed = value.trim();
    if trimmed.len() >= 2 {
        let first = trimmed.chars().next().unwrap_or_default();
        let last = trimmed.chars().last().unwrap_or_default();
        if (first == '"' && last == '"') || (first == '\'' && last == '\'') {
            return trimmed[1..trimmed.len() - 1].trim().to_string();
        }
    }
    trimmed.to_string()
}

fn render_structured_persona_layer(
    title: &str,
    preferred_keys: &[&str],
    layer: &StructuredPersonaLayer,
) -> Option<String> {
    if layer.is_empty() {
        return None;
    }
    let mut lines = vec![format!("## {title}")];
    let mut emitted = Vec::new();
    for key in preferred_keys {
        if let Some(values) = layer.fields.get(*key) {
            render_front_matter_field(&mut lines, key, values);
            emitted.push((*key).to_string());
        }
    }
    for (key, values) in &layer.fields {
        if emitted.iter().any(|item| item == key) {
            continue;
        }
        render_front_matter_field(&mut lines, key, values);
    }
    if !layer.notes.is_empty() {
        lines.push("### Notes".into());
        for note in &layer.notes {
            lines.push(note.clone());
        }
    }
    Some(lines.join("\n"))
}

fn render_front_matter_field(lines: &mut Vec<String>, key: &str, values: &[String]) {
    let values = values
        .iter()
        .map(|value| value.trim())
        .filter(|value| !value.is_empty())
        .collect::<Vec<_>>();
    if values.is_empty() {
        return;
    }
    let label = humanize_front_matter_key(key);
    if values.len() == 1 {
        lines.push(format!("- {}: {}", label, values[0]));
        return;
    }
    lines.push(format!("- {label}:"));
    for value in values {
        lines.push(format!("  - {value}"));
    }
}

fn humanize_front_matter_key(key: &str) -> String {
    match key {
        "summary" => "Summary".into(),
        "owned_systems" => "Owned systems".into(),
        "environments" => "Environments".into(),
        "standing_preferences" => "Standing preferences".into(),
        "hard_constraints" => "Hard constraints".into(),
        "name" => "Name".into(),
        "role" => "Role".into(),
        "tone" => "Tone".into(),
        "style" => "Style".into(),
        "non_goals" => "Non-goals".into(),
        "behavioral_rules" => "Behavioral rules".into(),
        _ => key
            .split('_')
            .filter(|part| !part.is_empty())
            .map(|part| {
                let mut chars = part.chars();
                match chars.next() {
                    Some(first) => format!("{}{}", first.to_ascii_uppercase(), chars.as_str()),
                    None => String::new(),
                }
            })
            .collect::<Vec<_>>()
            .join(" "),
    }
}

fn render_operational_extensions(blocks: &[ExtensionBlock]) -> Option<String> {
    let blocks = blocks
        .iter()
        .filter(|block| !block.content.trim().is_empty())
        .collect::<Vec<_>>();
    if blocks.is_empty() {
        return None;
    }
    let mut lines = vec!["## Operational extensions".to_string()];
    for block in blocks {
        lines.push(format!("### {}", block.title));
        lines.push(block.content.clone());
    }
    Some(lines.join("\n"))
}

fn clamp_prompt_content(content: &str) -> String {
    const MAX_BYTES: usize = 12_000;
    if content.len() <= MAX_BYTES {
        return content.to_string();
    }
    let mut end = MAX_BYTES;
    while !content.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}\n[truncated]", &content[..end])
}

fn file_label(path: &Path) -> String {
    path.file_name()
        .map(|value| value.to_string_lossy().to_string())
        .unwrap_or_else(|| path.display().to_string())
}

fn persona_file_names() -> &'static [&'static str] {
    &[
        "identity.md",
        "heartbeat.md",
        "soul.md",
        "agents.md",
        "tools.md",
        "skills.md",
    ]
}

fn should_delegate_multi_agent(
    prompt: &str,
    config: &MultiAgentConfig,
    preference: MultiAgentPreference,
) -> bool {
    if !config.enabled {
        return false;
    }
    match preference {
        MultiAgentPreference::ForceOff => return false,
        MultiAgentPreference::ForceOn => return true,
        MultiAgentPreference::Auto => {}
    }
    if requested_worker_count(prompt).is_some() {
        return true;
    }
    let normalized = prompt.to_ascii_lowercase();
    let explicit = [
        "sub agent",
        "sub-agent",
        "sub agents",
        "sub-agents",
        "multi agent",
        "multi-agent",
        "multi agents",
        "multi-agents",
        "multiple agents",
        "多个agent",
        "多个agents",
        "多个worker",
        "多个workers",
        "多agent",
        "多agents",
        "多worker",
        "多workers",
    ];
    if explicit.iter().any(|needle| normalized.contains(needle)) {
        return true;
    }
    config.auto_delegate
        && [
            "同时",
            "分别",
            "并行",
            "parallel",
            "compare",
            "split this task",
        ]
        .iter()
        .any(|needle| prompt.contains(needle) || normalized.contains(needle))
}

fn resolve_worker_count(
    prompt: &str,
    config: &MultiAgentConfig,
    preference: MultiAgentPreference,
) -> usize {
    requested_worker_count(prompt)
        .or(match preference {
            MultiAgentPreference::ForceOn => Some(config.default_workers),
            _ => None,
        })
        .unwrap_or(config.default_workers)
        .max(1)
        .min(config.max_workers.max(1))
}

fn requested_worker_count(prompt: &str) -> Option<usize> {
    static EN_RE: std::sync::OnceLock<Regex> = std::sync::OnceLock::new();
    static ZH_RE: std::sync::OnceLock<Regex> = std::sync::OnceLock::new();
    let english = EN_RE.get_or_init(|| {
        Regex::new(r"(?i)\b(\d+)\s*(?:sub[\s-]*)?(agents?|workers?)\b")
            .expect("valid multi-agent regex")
    });
    if let Some(captures) = english.captures(prompt) {
        return captures.get(1)?.as_str().parse::<usize>().ok();
    }

    let chinese = ZH_RE.get_or_init(|| {
        Regex::new(r"(?i)(\d+)\s*个?\s*(?:sub[\s-]*)?(agents?|workers?)")
            .expect("valid multi-agent regex")
    });
    chinese
        .captures(prompt)
        .and_then(|captures| captures.get(1))
        .and_then(|value| value.as_str().parse::<usize>().ok())
}

fn parse_worker_briefs(plan: &str, worker_count: usize) -> Option<Vec<String>> {
    let mut briefs = vec![String::new(); worker_count];
    let mut seen = 0usize;
    for line in plan.lines() {
        let trimmed = line.trim();
        if !trimmed.to_ascii_uppercase().starts_with("WORKER ") {
            continue;
        }
        let mut parts = trimmed.splitn(2, ':');
        let prefix = parts.next()?.trim();
        let body = parts.next()?.trim();
        if body.is_empty() {
            continue;
        }
        let index = prefix
            .trim_start_matches("WORKER ")
            .trim()
            .parse::<usize>()
            .ok()?;
        if !(1..=worker_count).contains(&index) {
            continue;
        }
        if briefs[index - 1].is_empty() {
            seen += 1;
        }
        briefs[index - 1] = body.to_string();
    }
    if seen == worker_count && briefs.iter().all(|item| !item.is_empty()) {
        Some(briefs)
    } else {
        None
    }
}

fn generic_coordinator_plan(prompt: &str, worker_count: usize) -> String {
    generic_worker_briefs(prompt, worker_count)
        .into_iter()
        .enumerate()
        .map(|(index, brief)| format!("WORKER {}: {}", index + 1, brief))
        .collect::<Vec<_>>()
        .join("\n")
}

fn generic_worker_briefs(prompt: &str, worker_count: usize) -> Vec<String> {
    (0..worker_count)
        .map(|index| {
            format!(
                "Investigate perspective {} of {} for this request: {}",
                index + 1,
                worker_count,
                clamp_chars(prompt, 300)
            )
        })
        .collect()
}

fn clamp_chars(content: &str, max_chars: usize) -> String {
    if content.chars().count() <= max_chars {
        return content.to_string();
    }
    let truncated = content.chars().take(max_chars).collect::<String>();
    format!("{truncated}\n[truncated]")
}

fn select_tool(prompt: &str) -> Option<(&'static str, serde_json::Value)> {
    let trimmed = prompt.trim();

    if trimmed.eq_ignore_ascii_case("docker ps") {
        return Some(("docker_ps", json!({})));
    }
    if let Some(service) = trimmed.strip_prefix("systemctl status ") {
        return Some(("systemd_status", json!({ "service": service.trim() })));
    }
    if let Some(service) = trimmed.strip_prefix("systemctl restart ") {
        return Some(("systemd_restart", json!({ "service": service.trim() })));
    }
    if let Some(container) = trimmed.strip_prefix("docker logs ") {
        return Some(("docker_logs", json!({ "container": container.trim() })));
    }
    if let Some(container) = trimmed.strip_prefix("docker restart ") {
        return Some(("docker_restart", json!({ "container": container.trim() })));
    }
    if let Some(path) = trimmed.strip_prefix("read ") {
        return Some(("read_file", json!({ "path": path.trim() })));
    }
    None
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::PathBuf;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use anyhow::Result;
    use chrono::Utc;
    use hajimi_claw_exec::{LocalExecutor, PlatformMode};
    use hajimi_claw_llm::StaticBackend;
    use hajimi_claw_policy::{PolicyConfig, PolicyEngine};
    use hajimi_claw_store::Store;
    use hajimi_claw_tools::ToolRegistry;
    use hajimi_claw_types::{
        AgentEvent, AgentRequest, AgentStream, ClawResult, LlmBackend, ProviderCapabilities,
        ProviderConfig, ProviderKind, ProviderRecord,
    };
    use tempfile::tempdir;
    use tokio::sync::Mutex;

    use super::{
        AgentRuntime, MarkdownPromptSource, MultiAgentConfig, MultiAgentPreference,
        PromptParseMode, PromptSourceMode, StaticSystemPrompt, SystemPromptSource,
        parse_persona_layer, parse_worker_briefs, requested_worker_count, resolve_worker_count,
        should_delegate_multi_agent,
    };

    #[tokio::test]
    async fn routes_file_read_without_llm() -> Result<()> {
        let dir = tempdir()?;
        let path = dir.path().join("notes.txt");
        tokio::fs::write(&path, "hello from tool").await?;

        let mut config = PolicyConfig::default();
        config.allowed_workdirs = vec![dir.path().to_path_buf()];
        let policy = Arc::new(PolicyEngine::new(config));
        let executor = Arc::new(LocalExecutor::new(
            policy.clone(),
            PlatformMode::WindowsSafe,
        ));
        let tools = Arc::new(hajimi_claw_tools::ToolRegistry::default(
            executor,
            policy.clone(),
        ));
        let store = Arc::new(Store::open_in_memory()?);
        let agent = AgentRuntime::for_tests(tools, store, policy);

        let response = agent
            .ask(&format!("read {}", path.display()), None)
            .await
            .expect("agent response");
        assert_eq!(response, "hello from tool");
        Ok(())
    }

    struct ScriptedBackend {
        calls: Mutex<usize>,
    }

    struct CountingPromptSource {
        loads: Arc<AtomicUsize>,
        prompt: String,
    }

    impl CountingPromptSource {
        fn new(loads: Arc<AtomicUsize>, prompt: impl Into<String>) -> Self {
            Self {
                loads,
                prompt: prompt.into(),
            }
        }
    }

    impl SystemPromptSource for CountingPromptSource {
        fn load(&self) -> ClawResult<String> {
            self.loads.fetch_add(1, Ordering::SeqCst);
            Ok(self.prompt.clone())
        }
    }

    #[async_trait::async_trait]
    impl LlmBackend for ScriptedBackend {
        async fn respond(&self, request: AgentRequest) -> ClawResult<AgentStream> {
            let mut calls = self.calls.lock().await;
            let step = *calls;
            *calls += 1;

            let events = if step == 0 {
                assert!(request.tool_history.is_empty());
                vec![
                    Ok(AgentEvent::ToolCall {
                        id: Some("call_read".into()),
                        tool: "read_file".into(),
                        input: serde_json::json!({
                            "path": request.messages[0]
                                .content
                                .rsplit_once(' ')
                                .map(|(_, path)| path)
                                .unwrap_or(""),
                        }),
                    }),
                    Ok(AgentEvent::Finished),
                ]
            } else {
                assert_eq!(request.tool_history.len(), 1);
                vec![
                    Ok(AgentEvent::TextDelta("tool loop ok".into())),
                    Ok(AgentEvent::Finished),
                ]
            };
            Ok(Box::pin(futures::stream::iter(events)))
        }
    }

    #[tokio::test]
    async fn tool_calling_loop_executes_tool_and_returns_final_text() -> Result<()> {
        let dir = tempdir()?;
        let path = dir.path().join("notes.txt");
        tokio::fs::write(&path, "hello from tool loop").await?;

        let mut config = PolicyConfig::default();
        config.allowed_workdirs = vec![dir.path().to_path_buf()];
        let policy = Arc::new(PolicyEngine::new(config));
        let executor = Arc::new(LocalExecutor::new(
            policy.clone(),
            PlatformMode::WindowsSafe,
        ));
        let tools = Arc::new(hajimi_claw_tools::ToolRegistry::default(
            executor,
            policy.clone(),
        ));
        let store = Arc::new(Store::open_in_memory()?);
        store.upsert_provider(&ProviderRecord {
            config: ProviderConfig {
                id: "test-provider".into(),
                label: "Test Provider".into(),
                kind: ProviderKind::OpenAiCompatible,
                base_url: "https://example.com/v1".into(),
                api_key: "secret".into(),
                model: "gpt-test".into(),
                fallback_models: vec![],
                capabilities: ProviderCapabilities {
                    tool_calling: true,
                    streaming: false,
                    json_mode: false,
                    max_context_chars: Some(24_000),
                },
                enabled: true,
                extra_headers: vec![],
                created_at: Utc::now(),
            },
            is_default: true,
        })?;
        let agent = AgentRuntime::new(
            Arc::new(ScriptedBackend {
                calls: Mutex::new(0),
            }),
            tools,
            store,
            policy,
            Arc::new(StaticSystemPrompt::new(super::default_system_prompt())),
            std::env::temp_dir(),
            MultiAgentConfig::default(),
        );

        let response = agent
            .ask(&format!("please inspect the file {}", path.display()), None)
            .await
            .expect("agent response");
        assert_eq!(response, "tool loop ok");
        Ok(())
    }

    fn write_persona_file(dir: &std::path::Path, name: &str, content: &str) {
        fs::write(dir.join(name), content).unwrap();
    }

    fn build_test_runtime(
        llm: Arc<dyn LlmBackend>,
        prompt_source: Arc<dyn SystemPromptSource>,
    ) -> AgentRuntime {
        build_test_runtime_with_persona_dir(llm, prompt_source, std::env::temp_dir())
    }

    fn build_test_runtime_with_persona_dir(
        llm: Arc<dyn LlmBackend>,
        prompt_source: Arc<dyn SystemPromptSource>,
        persona_dir: PathBuf,
    ) -> AgentRuntime {
        let policy = Arc::new(PolicyEngine::new(PolicyConfig::default()));
        let executor = Arc::new(LocalExecutor::new(
            policy.clone(),
            PlatformMode::WindowsSafe,
        ));
        let tools = Arc::new(ToolRegistry::default(executor, policy.clone()));
        let store = Arc::new(Store::open_in_memory().unwrap());
        store
            .upsert_provider(&ProviderRecord {
                config: ProviderConfig {
                    id: "test-provider".into(),
                    label: "Test Provider".into(),
                    kind: ProviderKind::OpenAiCompatible,
                    base_url: "https://example.com/v1".into(),
                    api_key: "secret".into(),
                    model: "gpt-test".into(),
                    fallback_models: vec![],
                    capabilities: ProviderCapabilities {
                        tool_calling: true,
                        streaming: false,
                        json_mode: false,
                        max_context_chars: Some(24_000),
                    },
                    enabled: true,
                    extra_headers: vec![],
                    created_at: Utc::now(),
                },
                is_default: true,
            })
            .unwrap();
        AgentRuntime::new(
            llm,
            tools,
            store,
            policy,
            prompt_source,
            persona_dir,
            MultiAgentConfig::default(),
        )
    }

    #[test]
    fn plain_text_identity_and_soul_render_as_legacy_layers() {
        let dir = tempdir().unwrap();
        write_persona_file(dir.path(), "identity.md", "User runs two VPS nodes.");
        write_persona_file(dir.path(), "soul.md", "Stay concise and calm.");
        let source = MarkdownPromptSource::with_mode(
            "base",
            vec![dir.path().join("identity.md"), dir.path().join("soul.md")],
            PromptSourceMode::ExplicitList,
        );

        let prompt = source.load().unwrap();
        assert!(prompt.contains("## Identity layer"));
        assert!(prompt.contains("User runs two VPS nodes."));
        assert!(prompt.contains("## Soul layer"));
        assert!(prompt.contains("Stay concise and calm."));
        assert!(!prompt.contains("## Operational extensions"));
    }

    #[test]
    fn structured_front_matter_parses_and_renders_fields() {
        let dir = tempdir().unwrap();
        write_persona_file(
            dir.path(),
            "identity.md",
            "---\nsummary: Alice on-call owner\nowned systems:\n  - prod-api\nstanding preferences:\n  - be terse\n---\n\nExtra note.",
        );
        let source = MarkdownPromptSource::with_mode(
            "base",
            vec![dir.path().join("identity.md")],
            PromptSourceMode::ExplicitList,
        );

        let prompt = source.load().unwrap();
        assert!(prompt.contains("- Summary: Alice on-call owner"));
        assert!(prompt.contains("- Owned systems: prod-api"));
        assert!(prompt.contains("- Standing preferences: be terse"));
        assert!(prompt.contains("### Notes\nExtra note."));
        let report = source.describe().unwrap().unwrap();
        assert!(matches!(
            report.entries[0].parse_mode,
            PromptParseMode::Structured
        ));
    }

    #[test]
    fn malformed_front_matter_falls_back_to_legacy() {
        let parsed = parse_persona_layer("---\nsummary\n---\nBody text");
        match parsed {
            super::ParsedPersonaLayer::Legacy(body) => {
                assert!(body.contains("Body text"));
            }
            _ => panic!("expected legacy fallback"),
        }
    }

    #[test]
    fn merge_rules_override_structured_fields_and_keep_notes() {
        let dir = tempdir().unwrap();
        let base_dir = dir.path().join("base");
        let override_dir = dir.path().join("override");
        fs::create_dir_all(&base_dir).unwrap();
        fs::create_dir_all(&override_dir).unwrap();
        write_persona_file(
            &base_dir,
            "identity.md",
            "---\nsummary: Base summary\nowned_systems:\n  - prod-api\n---\n\nBase note.",
        );
        write_persona_file(
            &override_dir,
            "identity.md",
            "---\nsummary: Override summary\nhard_constraints:\n  - never touch prod without approval\n---\n\nOverride note.",
        );
        let source = MarkdownPromptSource::with_mode(
            "base",
            vec![
                base_dir.join("identity.md"),
                override_dir.join("identity.md"),
            ],
            PromptSourceMode::ExplicitList,
        );

        let prompt = source.load().unwrap();
        assert!(prompt.contains("- Summary: Override summary"));
        assert!(prompt.contains("- Hard constraints: never touch prod without approval"));
        assert!(prompt.contains("Base note."));
        assert!(prompt.contains("Override note."));
        assert!(!prompt.contains("Base summary"));
    }

    #[test]
    fn operational_extensions_render_in_stable_order() {
        let dir = tempdir().unwrap();
        let base_dir = dir.path().join("base");
        let override_dir = dir.path().join("override");
        fs::create_dir_all(&base_dir).unwrap();
        fs::create_dir_all(&override_dir).unwrap();
        write_persona_file(&base_dir, "agents.md", "Agent policy");
        write_persona_file(&override_dir, "AGENTS.md", "Repo-local agent override");
        write_persona_file(&override_dir, "tools.md", "Tool policy");
        write_persona_file(&override_dir, "skills.md", "Skill hints");
        let source = MarkdownPromptSource::with_mode(
            "base",
            vec![
                base_dir.join("agents.md"),
                override_dir.join("AGENTS.md"),
                override_dir.join("tools.md"),
                override_dir.join("skills.md"),
            ],
            PromptSourceMode::ExplicitList,
        );

        let prompt = source.load().unwrap();
        let agent_policy = prompt.find("Agent policy").unwrap();
        let repo_override = prompt.find("Repo-local agent override").unwrap();
        let tool_policy = prompt.find("Tool policy").unwrap();
        let skill_hints = prompt.find("Skill hints").unwrap();
        assert!(agent_policy < repo_override);
        assert!(repo_override < tool_policy);
        assert!(tool_policy < skill_hints);
    }

    #[test]
    fn heartbeat_is_reported_but_excluded_from_prompt() {
        let dir = tempdir().unwrap();
        write_persona_file(dir.path(), "heartbeat.md", "enabled: false");
        let source = MarkdownPromptSource::with_mode(
            "base",
            vec![dir.path().join("heartbeat.md")],
            PromptSourceMode::ExplicitList,
        );

        let prompt = source.load().unwrap();
        assert_eq!(prompt, "base");
        let report = source.describe().unwrap().unwrap();
        assert_eq!(report.entries.len(), 1);
        assert!(!report.entries[0].included);
        assert!(matches!(
            report.entries[0].parse_mode,
            PromptParseMode::RuntimeOnly
        ));
    }

    #[test]
    fn describe_marks_missing_and_empty_files() {
        let dir = tempdir().unwrap();
        write_persona_file(dir.path(), "soul.md", "   \n\n");
        let source = MarkdownPromptSource::with_mode(
            "base",
            vec![dir.path().join("identity.md"), dir.path().join("soul.md")],
            PromptSourceMode::ExplicitList,
        );

        let report = source.describe().unwrap().unwrap();
        assert!(matches!(
            report.entries[0].parse_mode,
            PromptParseMode::Missing
        ));
        assert!(matches!(
            report.entries[1].parse_mode,
            PromptParseMode::Empty
        ));
    }

    #[tokio::test]
    async fn persona_list_renders_effective_view() {
        let dir = tempdir().unwrap();
        write_persona_file(dir.path(), "identity.md", "identity body");
        let runtime = build_test_runtime_with_persona_dir(
            Arc::new(StaticBackend::new("fallback")),
            Arc::new(MarkdownPromptSource::with_mode(
                "base",
                vec![dir.path().join("identity.md")],
                PromptSourceMode::ExplicitList,
            )),
            dir.path().to_path_buf(),
        );

        let list = runtime.persona_list().await.unwrap();
        assert!(list.contains("persona effective view"));
        assert!(list.contains("[identity layer]"));
        assert!(list.contains("identity.md"));
        assert!(list.contains("runtime-only"));
    }

    #[tokio::test]
    async fn single_agent_keeps_request_time_prompt_loading() -> Result<()> {
        let loads = Arc::new(AtomicUsize::new(0));
        let runtime = build_test_runtime(
            Arc::new(StaticBackend::new("done")),
            Arc::new(CountingPromptSource::new(loads.clone(), "prompt")),
        );
        let reply = runtime
            .ask_with_provider_and_preference("hello", None, None, MultiAgentPreference::ForceOff)
            .await?;
        assert_eq!(reply, "done");
        assert_eq!(loads.load(Ordering::SeqCst), 1);
        Ok(())
    }

    #[tokio::test]
    async fn multi_agent_keeps_request_time_prompt_loading() -> Result<()> {
        let loads = Arc::new(AtomicUsize::new(0));
        let runtime = build_test_runtime(
            Arc::new(StaticBackend::new("WORKER 1: first\nWORKER 2: second")),
            Arc::new(CountingPromptSource::new(loads.clone(), "prompt")),
        );
        let _reply = runtime
            .ask_with_provider_and_preference(
                "use 2 agents for this",
                None,
                None,
                MultiAgentPreference::ForceOn,
            )
            .await?;
        assert_eq!(loads.load(Ordering::SeqCst), 1);
        Ok(())
    }

    #[test]
    fn parses_requested_worker_count_from_english_and_chinese() {
        assert_eq!(
            requested_worker_count("use 4 agents to inspect this"),
            Some(4)
        );
        assert_eq!(requested_worker_count("开 6 个 agent 帮我排查"), Some(6));
        assert_eq!(requested_worker_count("let 3 workers handle this"), Some(3));
    }

    #[test]
    fn multi_agent_trigger_respects_keywords_and_config() {
        let config = MultiAgentConfig::default();
        assert!(should_delegate_multi_agent(
            "please use sub agents for this incident",
            &config,
            MultiAgentPreference::Auto
        ));
        assert!(should_delegate_multi_agent(
            "开4个agent来做",
            &config,
            MultiAgentPreference::Auto
        ));
        assert!(!should_delegate_multi_agent(
            "just answer normally",
            &config,
            MultiAgentPreference::Auto
        ));
        assert!(should_delegate_multi_agent(
            "just answer normally",
            &config,
            MultiAgentPreference::ForceOn
        ));
    }

    #[test]
    fn worker_count_uses_configured_maximum() {
        let config = MultiAgentConfig {
            default_workers: 5,
            max_workers: 12,
            ..MultiAgentConfig::default()
        };
        assert_eq!(
            resolve_worker_count("use 9 agents", &config, MultiAgentPreference::Auto),
            9
        );
        assert_eq!(
            resolve_worker_count("use 30 agents", &config, MultiAgentPreference::Auto),
            12
        );
        assert_eq!(
            resolve_worker_count("no explicit count", &config, MultiAgentPreference::Auto),
            5
        );
        assert_eq!(
            resolve_worker_count("no explicit count", &config, MultiAgentPreference::ForceOn),
            5
        );
    }

    #[test]
    fn parses_coordinator_worker_briefs() {
        let plan = "WORKER 1: inspect logs\nWORKER 2: inspect config";
        let briefs = parse_worker_briefs(plan, 2).expect("parsed worker briefs");
        assert_eq!(briefs[0], "inspect logs");
        assert_eq!(briefs[1], "inspect config");
    }
}
