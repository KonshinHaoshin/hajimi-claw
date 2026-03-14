use std::path::PathBuf;
use std::sync::Arc;
use std::{fs, path::Path};

use chrono::Utc;
use futures::TryStreamExt;
use hajimi_claw_llm::StaticBackend;
use hajimi_claw_policy::PolicyEngine;
use hajimi_claw_store::Store;
use hajimi_claw_tools::ToolRegistry;
use hajimi_claw_types::{
    AgentRequest, ApprovalId, ClawError, ClawResult, ConversationId, ConversationMessage,
    LlmBackend, MessageRole, PolicyMode, TaskId, TaskKind, TaskStatus, ToolContext,
};
use serde_json::json;
use tokio::sync::Semaphore;

pub struct AgentRuntime {
    llm: Arc<dyn LlmBackend>,
    tools: Arc<ToolRegistry>,
    store: Arc<Store>,
    policy: Arc<PolicyEngine>,
    task_gate: Arc<Semaphore>,
    prompt_source: Arc<dyn SystemPromptSource>,
    persona_dir: PathBuf,
}

#[derive(Debug, Clone)]
pub struct ShellOpenReply {
    pub session_id: String,
    pub message: String,
}

impl AgentRuntime {
    pub fn new(
        llm: Arc<dyn LlmBackend>,
        tools: Arc<ToolRegistry>,
        store: Arc<Store>,
        policy: Arc<PolicyEngine>,
        prompt_source: Arc<dyn SystemPromptSource>,
        persona_dir: PathBuf,
    ) -> Self {
        Self {
            llm,
            tools,
            store,
            policy,
            task_gate: Arc::new(Semaphore::new(1)),
            prompt_source,
            persona_dir,
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
        )
    }

    pub async fn ask(&self, prompt: &str, cwd: Option<PathBuf>) -> ClawResult<String> {
        self.ask_with_provider(prompt, cwd, None).await
    }

    pub async fn ask_with_provider(
        &self,
        prompt: &str,
        cwd: Option<PathBuf>,
        provider_id: Option<String>,
    ) -> ClawResult<String> {
        let _permit = self
            .task_gate
            .acquire()
            .await
            .map_err(|_| ClawError::Backend("task gate closed".into()))?;

        let task_id = TaskId::new();
        let conversation_id = ConversationId::new();
        let mut status = TaskStatus {
            id: task_id,
            kind: TaskKind::EphemeralAgentTask,
            description: prompt.into(),
            queued_at: Utc::now(),
            started_at: Some(Utc::now()),
            finished_at: None,
            running: true,
        };
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

        let result = if let Some((tool, input)) = select_tool(prompt) {
            self.tools
                .call(
                    tool,
                    ToolContext {
                        conversation_id,
                        working_directory: cwd,
                        elevated: self.policy.is_elevated(),
                    },
                    input,
                )
                .await
                .map(|output| output.content)
        } else {
            self.run_llm(conversation_id, prompt, provider_id).await
        };

        status.running = false;
        status.finished_at = Some(Utc::now());
        self.store.upsert_task(&status).map_err(store_error)?;

        if let Ok(content) = &result {
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

        result
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
        Ok(ShellOpenReply {
            session_id: session_id.to_string(),
            message: output.content,
        })
    }

    pub async fn shell_exec(&self, session_id: &str, command: &str) -> ClawResult<String> {
        self.tools
            .call(
                "session_exec",
                ToolContext {
                    conversation_id: ConversationId::new(),
                    working_directory: None,
                    elevated: self.policy.is_elevated(),
                },
                json!({ "session_id": session_id, "command": command }),
            )
            .await
            .map(|output| output.content)
    }

    pub async fn shell_close(&self, session_id: &str) -> ClawResult<String> {
        self.tools
            .call(
                "session_close",
                ToolContext {
                    conversation_id: ConversationId::new(),
                    working_directory: None,
                    elevated: self.policy.is_elevated(),
                },
                json!({ "session_id": session_id }),
            )
            .await
            .map(|output| output.content)
    }

    pub async fn persona_list(&self) -> ClawResult<String> {
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

    pub fn request_elevated(&self, minutes: i64, reason: String) -> String {
        let approval = self.policy.request_elevation(minutes, reason.clone());
        let _ = self.store.save_approval(&approval, None);
        format!(
            "approval required: {} [{}], expires at {}",
            reason, approval.request_id, approval.expires_at
        )
    }

    pub fn approve(&self, request_id: &str) -> ClawResult<String> {
        let approval_id = ApprovalId(
            uuid::Uuid::parse_str(request_id)
                .map_err(|err| ClawError::InvalidRequest(err.to_string()))?,
        );
        let approval = self
            .policy
            .approve(approval_id)
            .ok_or_else(|| ClawError::NotFound(format!("approval not found: {request_id}")))?;
        let _ = self.store.save_approval(&approval, Some(true));
        Ok(format!(
            "approved {} ({})",
            approval.command_preview, approval.request_id
        ))
    }

    pub fn stop_elevated(&self) -> String {
        self.policy.stop_elevation();
        "elevated lease stopped".into()
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
                        "{} [{}] running={} queued_at={}",
                        task.id, task.description, task.running, task.queued_at
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
        Ok(format!("policy_mode={mode}\n{task_lines}"))
    }

    async fn run_llm(
        &self,
        conversation_id: ConversationId,
        prompt: &str,
        provider_id: Option<String>,
    ) -> ClawResult<String> {
        let stream = self
            .llm
            .respond(AgentRequest {
                conversation_id,
                provider_id,
                system_prompt: self.prompt_source.load()?,
                messages: vec![ConversationMessage {
                    role: MessageRole::User,
                    content: prompt.into(),
                    created_at: Utc::now(),
                }],
                tool_specs: self.tools.specs(),
            })
            .await?;

        let events = stream.try_collect::<Vec<_>>().await?;
        let content = events
            .into_iter()
            .filter_map(|event| match event {
                hajimi_claw_types::AgentEvent::TextDelta(delta) => Some(delta),
                _ => None,
            })
            .collect::<String>();
        Ok(content)
    }
}

fn store_error(err: anyhow::Error) -> ClawError {
    ClawError::Backend(err.to_string())
}

impl AgentRuntime {
    fn resolve_persona_file(&self, raw: &str) -> ClawResult<PathBuf> {
        let trimmed = raw.trim().trim_matches('`');
        let canonical = match trimmed.to_ascii_lowercase().as_str() {
            "soul" | "soul.md" => "soul.md",
            "agents" | "agents.md" => "agents.md",
            "tools" | "tools.md" => "tools.md",
            "skills" | "skills.md" => "skills.md",
            _ => {
                return Err(ClawError::InvalidRequest(
                    "persona file must be one of: soul, agents, tools, skills".into(),
                ));
            }
        };
        Ok(self.persona_dir.join(canonical))
    }
}

pub fn default_system_prompt() -> &'static str {
    "You are hajimi-claw, a narrow single-user operations agent. Prefer concise, actionable answers. Use tools when possible."
}

pub trait SystemPromptSource: Send + Sync {
    fn load(&self) -> ClawResult<String>;
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

pub struct MarkdownPromptSource {
    base_prompt: String,
    files: Vec<PathBuf>,
}

impl MarkdownPromptSource {
    pub fn new(base_prompt: impl Into<String>, files: Vec<PathBuf>) -> Self {
        Self {
            base_prompt: base_prompt.into(),
            files,
        }
    }
}

impl SystemPromptSource for MarkdownPromptSource {
    fn load(&self) -> ClawResult<String> {
        let mut prompt = self.base_prompt.clone();
        let mut sections = Vec::new();
        for path in &self.files {
            let content = match fs::read_to_string(path) {
                Ok(content) => content,
                Err(err) if err.kind() == std::io::ErrorKind::NotFound => continue,
                Err(err) => return Err(ClawError::Backend(err.to_string())),
            };
            let trimmed = content.trim();
            if trimmed.is_empty() {
                continue;
            }
            sections.push(format!(
                "[{}]\n{}",
                file_label(path),
                clamp_prompt_content(trimmed)
            ));
        }
        if !sections.is_empty() {
            prompt.push_str(
                "\n\nLocal markdown guidance is attached below. Treat it as repo-specific operating guidance and persona shaping context.\n\n",
            );
            prompt.push_str(&sections.join("\n\n"));
        }
        Ok(prompt)
    }
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
    &["soul.md", "agents.md", "tools.md", "skills.md"]
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
    use std::sync::Arc;

    use anyhow::Result;
    use hajimi_claw_exec::{LocalExecutor, PlatformMode};
    use hajimi_claw_policy::{PolicyConfig, PolicyEngine};
    use hajimi_claw_store::Store;
    use tempfile::tempdir;

    use super::AgentRuntime;

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
}
