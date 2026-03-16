use std::fmt::{Display, Formatter};
use std::path::PathBuf;
use std::pin::Pin;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use futures::Stream;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;
use uuid::Uuid;

#[derive(Debug, Error)]
pub enum ClawError {
    #[error("access denied: {0}")]
    AccessDenied(String),
    #[error("approval required: {0}")]
    ApprovalRequired(String),
    #[error("invalid request: {0}")]
    InvalidRequest(String),
    #[error("not found: {0}")]
    NotFound(String),
    #[error("backend error: {0}")]
    Backend(String),
}

pub type ClawResult<T> = Result<T, ClawError>;

macro_rules! id_type {
    ($name:ident) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
        pub struct $name(pub Uuid);

        impl $name {
            pub fn new() -> Self {
                Self(Uuid::new_v4())
            }
        }

        impl Default for $name {
            fn default() -> Self {
                Self::new()
            }
        }

        impl Display for $name {
            fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
                write!(f, "{}", self.0)
            }
        }
    };
}

id_type!(TaskId);
id_type!(SessionId);
id_type!(ApprovalId);
id_type!(ConversationId);

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProviderKind {
    OpenAiCompatible,
    CustomChatCompletions,
}

impl ProviderKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::OpenAiCompatible => "openai-compatible",
            Self::CustomChatCompletions => "custom-chat-completions",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProviderConfig {
    pub id: String,
    pub label: String,
    pub kind: ProviderKind,
    pub base_url: String,
    pub api_key: String,
    pub model: String,
    #[serde(default)]
    pub fallback_models: Vec<String>,
    pub enabled: bool,
    pub extra_headers: Vec<(String, String)>,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProviderRecord {
    pub config: ProviderConfig,
    pub is_default: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum OnboardingStep {
    ProviderLabel,
    ProviderKind,
    ProviderBaseUrl,
    ProviderApiKey,
    ProviderModel,
    Completed,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct ProviderDraft {
    pub label: Option<String>,
    pub kind: Option<ProviderKind>,
    pub base_url: Option<String>,
    pub api_key: Option<String>,
    pub model: Option<String>,
    pub fallback_models: Option<Vec<String>>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OnboardingSession {
    pub user_id: i64,
    pub chat_id: i64,
    pub step: OnboardingStep,
    pub draft: ProviderDraft,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProviderHealth {
    pub ok: bool,
    pub message: String,
    pub suggested_models: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PolicyMode {
    Normal,
    ApprovalPending,
    ElevatedLease,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum RiskLevel {
    Safe,
    Guarded,
    Dangerous,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecRequest {
    pub command: String,
    pub args: Vec<String>,
    pub cwd: Option<PathBuf>,
    pub env_allowlist: Vec<String>,
    pub timeout_secs: u64,
    pub max_output_bytes: usize,
    pub requires_tty: bool,
}

impl ExecRequest {
    pub fn summary(&self) -> String {
        if self.args.is_empty() {
            self.command.clone()
        } else {
            format!("{} {}", self.command, self.args.join(" "))
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecResult {
    pub exit_code: Option<i32>,
    pub stdout: String,
    pub stderr: String,
    pub duration_ms: u128,
    pub truncated: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionOpenRequest {
    pub name: Option<String>,
    pub cwd: Option<PathBuf>,
    pub env_allowlist: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionHandle {
    pub id: SessionId,
    pub name: String,
    pub cwd: PathBuf,
    pub created_at: DateTime<Utc>,
    pub last_used_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionSummary {
    pub session_id: SessionId,
    pub cwd: PathBuf,
    pub recent_commands: Vec<String>,
    pub summary: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApprovalRequest {
    pub request_id: ApprovalId,
    pub reason: String,
    pub risk_level: RiskLevel,
    pub command_preview: String,
    pub cwd: Option<PathBuf>,
    pub expires_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApprovalDecision {
    pub request_id: ApprovalId,
    pub approved: bool,
    pub decided_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolSpec {
    pub name: String,
    pub description: String,
    pub requires_approval: bool,
    pub input_schema: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolOutput {
    pub content: String,
    pub structured: Option<Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolContext {
    pub conversation_id: ConversationId,
    pub working_directory: Option<PathBuf>,
    pub elevated: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MessageRole {
    System,
    User,
    Assistant,
    Tool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationMessage {
    pub role: MessageRole,
    pub content: String,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallRecord {
    pub id: Option<String>,
    pub name: String,
    pub arguments: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResultRecord {
    pub call_id: Option<String>,
    pub name: String,
    pub content: String,
    pub structured: Option<Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolExchange {
    pub call: ToolCallRecord,
    pub result: ToolResultRecord,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentRequest {
    pub conversation_id: ConversationId,
    pub provider_id: Option<String>,
    pub system_prompt: String,
    pub messages: Vec<ConversationMessage>,
    pub tool_specs: Vec<ToolSpec>,
    #[serde(default)]
    pub tool_history: Vec<ToolExchange>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AgentEvent {
    TextDelta(String),
    ToolCall {
        id: Option<String>,
        tool: String,
        input: Value,
    },
    Finished,
}

pub type AgentStream = Pin<Box<dyn Stream<Item = ClawResult<AgentEvent>> + Send>>;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TaskKind {
    EphemeralAgentTask,
    PersistentShellTask,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskStatus {
    pub id: TaskId,
    pub kind: TaskKind,
    pub description: String,
    pub queued_at: DateTime<Utc>,
    pub started_at: Option<DateTime<Utc>>,
    pub finished_at: Option<DateTime<Utc>>,
    pub running: bool,
}

#[async_trait]
pub trait LlmBackend: Send + Sync {
    async fn respond(&self, req: AgentRequest) -> ClawResult<AgentStream>;
}

#[async_trait]
pub trait Tool: Send + Sync {
    fn spec(&self) -> ToolSpec;
    async fn call(&self, ctx: ToolContext, input: Value) -> ClawResult<ToolOutput>;
}

#[async_trait]
pub trait Executor: Send + Sync {
    async fn run_once(&self, req: ExecRequest) -> ClawResult<ExecResult>;
    async fn open_session(&self, req: SessionOpenRequest) -> ClawResult<SessionHandle>;
    async fn run_in_session(&self, id: SessionId, req: ExecRequest) -> ClawResult<ExecResult>;
    async fn close_session(&self, id: SessionId) -> ClawResult<()>;
}
