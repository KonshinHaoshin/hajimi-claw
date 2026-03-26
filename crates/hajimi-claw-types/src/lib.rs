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
    #[serde(default)]
    pub capabilities: ProviderCapabilities,
    pub enabled: bool,
    pub extra_headers: Vec<(String, String)>,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProviderCapabilities {
    pub tool_calling: bool,
    pub streaming: bool,
    pub json_mode: bool,
    pub max_context_chars: Option<usize>,
}

impl Default for ProviderCapabilities {
    fn default() -> Self {
        Self {
            tool_calling: false,
            streaming: false,
            json_mode: false,
            max_context_chars: None,
        }
    }
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
    #[serde(default)]
    pub stdin: Option<String>,
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
pub struct HeartbeatStatus {
    pub last_seen_at: DateTime<Utc>,
    pub pid: Option<u32>,
    pub channel: Option<String>,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CapabilityKind {
    NativeTool,
    Skill,
    McpTool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CapabilityId {
    pub kind: CapabilityKind,
    pub name: String,
    pub source: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExecutableSkillConfig {
    pub name: String,
    pub description: String,
    pub command: String,
    #[serde(default)]
    pub args: Vec<String>,
    pub cwd: Option<PathBuf>,
    #[serde(default)]
    pub env_allowlist: Vec<String>,
    #[serde(default)]
    pub requires_approval: bool,
    pub timeout_secs: Option<u64>,
    pub max_output_bytes: Option<usize>,
    pub input_schema: Value,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct McpServerConfig {
    pub name: String,
    pub command: String,
    #[serde(default)]
    pub args: Vec<String>,
    pub cwd: Option<PathBuf>,
    #[serde(default)]
    pub env_allowlist: Vec<String>,
    pub startup_timeout_secs: Option<u64>,
    #[serde(default = "default_enabled")]
    pub enabled: bool,
    #[serde(default)]
    pub requires_approval: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CapabilitySummary {
    pub id: CapabilityId,
    pub description: String,
    pub requires_approval: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct McpServerStatus {
    pub name: String,
    pub connected: bool,
    pub tool_count: usize,
    pub message: String,
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

fn default_enabled() -> bool {
    true
}

fn default_capability_kind() -> CapabilityKind {
    CapabilityKind::NativeTool
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
    DirectToolTask,
    PersistentShellTask,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TaskRunState {
    Queued,
    Running,
    BlockedApproval,
    Completed,
    Failed,
    Cancelled,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskStatus {
    pub id: TaskId,
    pub conversation_id: ConversationId,
    pub kind: TaskKind,
    pub description: String,
    pub queued_at: DateTime<Utc>,
    pub started_at: Option<DateTime<Utc>>,
    pub finished_at: Option<DateTime<Utc>>,
    pub state: TaskRunState,
    pub running: bool,
    pub cwd: Option<PathBuf>,
    pub provider_id: Option<String>,
    pub current_session_id: Option<String>,
    pub result_preview: Option<String>,
    pub error: Option<String>,
    pub blocked_approval_id: Option<ApprovalId>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ToolInvocationStatus {
    Pending,
    Running,
    BlockedApproval,
    Completed,
    Failed,
    Cancelled,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolInvocationRecord {
    pub task_id: TaskId,
    pub conversation_id: ConversationId,
    pub call_id: Option<String>,
    pub tool_name: String,
    pub arguments: Value,
    pub status: ToolInvocationStatus,
    pub output_content: Option<String>,
    pub output_structured: Option<Value>,
    pub error: Option<String>,
    pub approval_id: Option<ApprovalId>,
    pub sequence: i64,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApprovalRecord {
    pub request: ApprovalRequest,
    pub approved: Option<bool>,
    pub task_id: Option<TaskId>,
    pub tool_name: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExecutionProfile {
    OpsSafe,
    DevAgent,
    ComputerUse,
}

impl ExecutionProfile {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::OpsSafe => "ops-safe",
            Self::DevAgent => "dev-agent",
            Self::ComputerUse => "computer-use",
        }
    }

    pub fn parse(raw: &str) -> Option<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "ops-safe" | "ops" => Some(Self::OpsSafe),
            "dev-agent" | "dev" => Some(Self::DevAgent),
            "computer-use" | "computer" => Some(Self::ComputerUse),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionBinding {
    pub task_id: TaskId,
    pub conversation_id: ConversationId,
    pub session_id: SessionId,
    pub updated_at: DateTime<Utc>,
}

#[async_trait]
pub trait LlmBackend: Send + Sync {
    async fn respond(&self, req: AgentRequest) -> ClawResult<AgentStream>;
}

#[async_trait]
pub trait Tool: Send + Sync {
    fn spec(&self) -> ToolSpec;

    fn capability_id(&self) -> CapabilityId {
        CapabilityId {
            kind: CapabilityKind::NativeTool,
            name: self.spec().name,
            source: None,
        }
    }

    fn capability_summary(&self) -> CapabilitySummary {
        let spec = self.spec();
        CapabilitySummary {
            id: self.capability_id(),
            description: spec.description,
            requires_approval: spec.requires_approval,
        }
    }

    async fn call(&self, ctx: ToolContext, input: Value) -> ClawResult<ToolOutput>;
}

#[async_trait]
pub trait Executor: Send + Sync {
    async fn run_once(&self, req: ExecRequest) -> ClawResult<ExecResult>;
    async fn open_session(&self, req: SessionOpenRequest) -> ClawResult<SessionHandle>;
    async fn run_in_session(&self, id: SessionId, req: ExecRequest) -> ClawResult<ExecResult>;
    async fn describe_session(&self, id: SessionId) -> ClawResult<SessionHandle>;
    async fn close_session(&self, id: SessionId) -> ClawResult<()>;
}
