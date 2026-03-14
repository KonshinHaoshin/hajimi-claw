use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use async_trait::async_trait;
use hajimi_claw_policy::PolicyEngine;
use hajimi_claw_types::{
    ClawError, ClawResult, ExecRequest, Executor, SessionId, SessionOpenRequest, Tool, ToolContext,
    ToolOutput, ToolSpec,
};
use regex::Regex;
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::fs;

pub struct ToolRegistry {
    tools: HashMap<String, Arc<dyn Tool>>,
}

impl ToolRegistry {
    pub fn new(tools: Vec<Arc<dyn Tool>>) -> Self {
        let tools = tools
            .into_iter()
            .map(|tool| (tool.spec().name.clone(), tool))
            .collect();
        Self { tools }
    }

    pub fn default(executor: Arc<dyn Executor>, policy: Arc<PolicyEngine>) -> Self {
        Self::new(vec![
            Arc::new(ReadFileTool::new(policy.clone())),
            Arc::new(TailFileTool::new(policy.clone())),
            Arc::new(ListDirTool::new(policy.clone())),
            Arc::new(GrepTextTool::new(policy.clone())),
            Arc::new(WriteFileTool::new(policy.clone())),
            Arc::new(AppendFileTool::new(policy.clone())),
            Arc::new(SystemdStatusTool::new(executor.clone())),
            Arc::new(SystemdRestartTool::new(executor.clone())),
            Arc::new(DockerPsTool::new(executor.clone())),
            Arc::new(DockerLogsTool::new(executor.clone())),
            Arc::new(DockerRestartTool::new(executor.clone())),
            Arc::new(RunCommandTool::new(executor.clone())),
            Arc::new(SessionOpenTool::new(executor.clone())),
            Arc::new(SessionExecTool::new(executor.clone())),
            Arc::new(SessionCloseTool::new(executor)),
        ])
    }

    pub fn specs(&self) -> Vec<ToolSpec> {
        self.tools.values().map(|tool| tool.spec()).collect()
    }

    pub async fn call(&self, name: &str, ctx: ToolContext, input: Value) -> ClawResult<ToolOutput> {
        let tool = self
            .tools
            .get(name)
            .ok_or_else(|| ClawError::NotFound(format!("tool not found: {name}")))?;
        tool.call(ctx, input).await
    }
}

struct ReadFileTool {
    policy: Arc<PolicyEngine>,
}

impl ReadFileTool {
    fn new(policy: Arc<PolicyEngine>) -> Self {
        Self { policy }
    }
}

#[async_trait]
impl Tool for ReadFileTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "read_file".into(),
            description: "Read a text file from an allowed directory.".into(),
            requires_approval: false,
        }
    }

    async fn call(&self, _ctx: ToolContext, input: Value) -> ClawResult<ToolOutput> {
        #[derive(Deserialize)]
        struct Input {
            path: PathBuf,
            max_bytes: Option<usize>,
        }
        let input: Input = serde_json::from_value(input)
            .map_err(|err| ClawError::InvalidRequest(err.to_string()))?;
        guard_path(&self.policy, &input.path)?;
        let content = fs::read_to_string(&input.path)
            .await
            .map_err(|err| ClawError::Backend(err.to_string()))?;
        let max_bytes = input.max_bytes.unwrap_or(8 * 1024);
        let content = truncate_string(content, max_bytes);
        Ok(ToolOutput {
            content,
            structured: None,
        })
    }
}

struct TailFileTool {
    policy: Arc<PolicyEngine>,
}

impl TailFileTool {
    fn new(policy: Arc<PolicyEngine>) -> Self {
        Self { policy }
    }
}

#[async_trait]
impl Tool for TailFileTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "tail_file".into(),
            description: "Read the tail of a text file.".into(),
            requires_approval: false,
        }
    }

    async fn call(&self, _ctx: ToolContext, input: Value) -> ClawResult<ToolOutput> {
        #[derive(Deserialize)]
        struct Input {
            path: PathBuf,
            lines: Option<usize>,
        }
        let input: Input = serde_json::from_value(input)
            .map_err(|err| ClawError::InvalidRequest(err.to_string()))?;
        guard_path(&self.policy, &input.path)?;
        let content = fs::read_to_string(&input.path)
            .await
            .map_err(|err| ClawError::Backend(err.to_string()))?;
        let lines = input.lines.unwrap_or(50);
        let tail = content
            .lines()
            .rev()
            .take(lines)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect::<Vec<_>>()
            .join("\n");
        Ok(ToolOutput {
            content: tail,
            structured: None,
        })
    }
}

struct ListDirTool {
    policy: Arc<PolicyEngine>,
}

impl ListDirTool {
    fn new(policy: Arc<PolicyEngine>) -> Self {
        Self { policy }
    }
}

#[async_trait]
impl Tool for ListDirTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "list_dir".into(),
            description: "List files and folders in an allowed directory.".into(),
            requires_approval: false,
        }
    }

    async fn call(&self, _ctx: ToolContext, input: Value) -> ClawResult<ToolOutput> {
        #[derive(Deserialize)]
        struct Input {
            path: PathBuf,
        }
        let input: Input = serde_json::from_value(input)
            .map_err(|err| ClawError::InvalidRequest(err.to_string()))?;
        guard_dir(&self.policy, &input.path)?;
        let mut dir = fs::read_dir(&input.path)
            .await
            .map_err(|err| ClawError::Backend(err.to_string()))?;
        let mut entries = Vec::new();
        while let Some(entry) = dir
            .next_entry()
            .await
            .map_err(|err| ClawError::Backend(err.to_string()))?
        {
            let metadata = entry
                .metadata()
                .await
                .map_err(|err| ClawError::Backend(err.to_string()))?;
            entries.push(format!(
                "{}\t{}",
                if metadata.is_dir() { "dir" } else { "file" },
                entry.path().display()
            ));
        }
        Ok(ToolOutput {
            content: entries.join("\n"),
            structured: None,
        })
    }
}

struct GrepTextTool {
    policy: Arc<PolicyEngine>,
}

impl GrepTextTool {
    fn new(policy: Arc<PolicyEngine>) -> Self {
        Self { policy }
    }
}

struct WriteFileTool {
    policy: Arc<PolicyEngine>,
}

impl WriteFileTool {
    fn new(policy: Arc<PolicyEngine>) -> Self {
        Self { policy }
    }
}

#[async_trait]
impl Tool for WriteFileTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "write_file".into(),
            description:
                "Write text to a file in a writable allowed directory, replacing existing content."
                    .into(),
            requires_approval: false,
        }
    }

    async fn call(&self, _ctx: ToolContext, input: Value) -> ClawResult<ToolOutput> {
        #[derive(Deserialize)]
        struct Input {
            path: PathBuf,
            content: String,
        }
        let input: Input = serde_json::from_value(input)
            .map_err(|err| ClawError::InvalidRequest(err.to_string()))?;
        guard_write_path(&self.policy, &input.path)?;
        if let Some(parent) = input.path.parent() {
            fs::create_dir_all(parent)
                .await
                .map_err(|err| ClawError::Backend(err.to_string()))?;
        }
        fs::write(&input.path, input.content.as_bytes())
            .await
            .map_err(|err| ClawError::Backend(err.to_string()))?;
        Ok(ToolOutput {
            content: format!("wrote {}", input.path.display()),
            structured: None,
        })
    }
}

struct AppendFileTool {
    policy: Arc<PolicyEngine>,
}

impl AppendFileTool {
    fn new(policy: Arc<PolicyEngine>) -> Self {
        Self { policy }
    }
}

#[async_trait]
impl Tool for AppendFileTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "append_file".into(),
            description: "Append text to a file in a writable allowed directory.".into(),
            requires_approval: false,
        }
    }

    async fn call(&self, _ctx: ToolContext, input: Value) -> ClawResult<ToolOutput> {
        #[derive(Deserialize)]
        struct Input {
            path: PathBuf,
            content: String,
        }
        let input: Input = serde_json::from_value(input)
            .map_err(|err| ClawError::InvalidRequest(err.to_string()))?;
        guard_write_path(&self.policy, &input.path)?;
        if let Some(parent) = input.path.parent() {
            fs::create_dir_all(parent)
                .await
                .map_err(|err| ClawError::Backend(err.to_string()))?;
        }
        let mut file = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&input.path)
            .await
            .map_err(|err| ClawError::Backend(err.to_string()))?;
        use tokio::io::AsyncWriteExt;
        file.write_all(input.content.as_bytes())
            .await
            .map_err(|err| ClawError::Backend(err.to_string()))?;
        Ok(ToolOutput {
            content: format!("appended {}", input.path.display()),
            structured: None,
        })
    }
}

#[async_trait]
impl Tool for GrepTextTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "grep_text".into(),
            description: "Search a text file with a regex pattern.".into(),
            requires_approval: false,
        }
    }

    async fn call(&self, _ctx: ToolContext, input: Value) -> ClawResult<ToolOutput> {
        #[derive(Deserialize)]
        struct Input {
            path: PathBuf,
            pattern: String,
        }
        let input: Input = serde_json::from_value(input)
            .map_err(|err| ClawError::InvalidRequest(err.to_string()))?;
        guard_path(&self.policy, &input.path)?;
        let pattern =
            Regex::new(&input.pattern).map_err(|err| ClawError::InvalidRequest(err.to_string()))?;
        let content = fs::read_to_string(&input.path)
            .await
            .map_err(|err| ClawError::Backend(err.to_string()))?;
        let matches = content
            .lines()
            .enumerate()
            .filter_map(|(index, line)| {
                pattern
                    .is_match(line)
                    .then(|| format!("{}:{}", index + 1, line))
            })
            .collect::<Vec<_>>()
            .join("\n");
        Ok(ToolOutput {
            content: matches,
            structured: None,
        })
    }
}

struct SystemdStatusTool {
    executor: Arc<dyn Executor>,
}

impl SystemdStatusTool {
    fn new(executor: Arc<dyn Executor>) -> Self {
        Self { executor }
    }
}

#[async_trait]
impl Tool for SystemdStatusTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "systemd_status".into(),
            description: "Inspect a systemd service status.".into(),
            requires_approval: false,
        }
    }

    async fn call(&self, ctx: ToolContext, input: Value) -> ClawResult<ToolOutput> {
        #[derive(Deserialize)]
        struct Input {
            service: String,
        }
        let input: Input = serde_json::from_value(input)
            .map_err(|err| ClawError::InvalidRequest(err.to_string()))?;
        let result = self
            .executor
            .run_once(ExecRequest {
                command: "systemctl".into(),
                args: vec!["status".into(), "--no-pager".into(), input.service],
                cwd: ctx.working_directory,
                env_allowlist: vec![],
                timeout_secs: 30,
                max_output_bytes: 16 * 1024,
                requires_tty: false,
            })
            .await?;
        Ok(command_output(result))
    }
}

struct SystemdRestartTool {
    executor: Arc<dyn Executor>,
}

impl SystemdRestartTool {
    fn new(executor: Arc<dyn Executor>) -> Self {
        Self { executor }
    }
}

#[async_trait]
impl Tool for SystemdRestartTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "systemd_restart".into(),
            description: "Restart a systemd service.".into(),
            requires_approval: true,
        }
    }

    async fn call(&self, ctx: ToolContext, input: Value) -> ClawResult<ToolOutput> {
        #[derive(Deserialize)]
        struct Input {
            service: String,
        }
        let input: Input = serde_json::from_value(input)
            .map_err(|err| ClawError::InvalidRequest(err.to_string()))?;
        let result = self
            .executor
            .run_once(ExecRequest {
                command: "systemctl".into(),
                args: vec!["restart".into(), input.service],
                cwd: ctx.working_directory,
                env_allowlist: vec![],
                timeout_secs: 30,
                max_output_bytes: 8 * 1024,
                requires_tty: false,
            })
            .await?;
        Ok(command_output(result))
    }
}

struct DockerPsTool {
    executor: Arc<dyn Executor>,
}

impl DockerPsTool {
    fn new(executor: Arc<dyn Executor>) -> Self {
        Self { executor }
    }
}

#[async_trait]
impl Tool for DockerPsTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "docker_ps".into(),
            description: "List running containers.".into(),
            requires_approval: false,
        }
    }

    async fn call(&self, ctx: ToolContext, _input: Value) -> ClawResult<ToolOutput> {
        let result = self
            .executor
            .run_once(ExecRequest {
                command: "docker".into(),
                args: vec!["ps".into()],
                cwd: ctx.working_directory,
                env_allowlist: vec![],
                timeout_secs: 20,
                max_output_bytes: 12 * 1024,
                requires_tty: false,
            })
            .await?;
        Ok(command_output(result))
    }
}

struct DockerLogsTool {
    executor: Arc<dyn Executor>,
}

impl DockerLogsTool {
    fn new(executor: Arc<dyn Executor>) -> Self {
        Self { executor }
    }
}

#[async_trait]
impl Tool for DockerLogsTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "docker_logs".into(),
            description: "Fetch recent logs from a container.".into(),
            requires_approval: false,
        }
    }

    async fn call(&self, ctx: ToolContext, input: Value) -> ClawResult<ToolOutput> {
        #[derive(Deserialize)]
        struct Input {
            container: String,
            tail: Option<u32>,
        }
        let input: Input = serde_json::from_value(input)
            .map_err(|err| ClawError::InvalidRequest(err.to_string()))?;
        let result = self
            .executor
            .run_once(ExecRequest {
                command: "docker".into(),
                args: vec![
                    "logs".into(),
                    "--tail".into(),
                    input.tail.unwrap_or(200).to_string(),
                    input.container,
                ],
                cwd: ctx.working_directory,
                env_allowlist: vec![],
                timeout_secs: 30,
                max_output_bytes: 16 * 1024,
                requires_tty: false,
            })
            .await?;
        Ok(command_output(result))
    }
}

struct DockerRestartTool {
    executor: Arc<dyn Executor>,
}

impl DockerRestartTool {
    fn new(executor: Arc<dyn Executor>) -> Self {
        Self { executor }
    }
}

#[async_trait]
impl Tool for DockerRestartTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "docker_restart".into(),
            description: "Restart a Docker container.".into(),
            requires_approval: true,
        }
    }

    async fn call(&self, ctx: ToolContext, input: Value) -> ClawResult<ToolOutput> {
        #[derive(Deserialize)]
        struct Input {
            container: String,
        }
        let input: Input = serde_json::from_value(input)
            .map_err(|err| ClawError::InvalidRequest(err.to_string()))?;
        let result = self
            .executor
            .run_once(ExecRequest {
                command: "docker".into(),
                args: vec!["restart".into(), input.container],
                cwd: ctx.working_directory,
                env_allowlist: vec![],
                timeout_secs: 30,
                max_output_bytes: 8 * 1024,
                requires_tty: false,
            })
            .await?;
        Ok(command_output(result))
    }
}

struct RunCommandTool {
    executor: Arc<dyn Executor>,
}

impl RunCommandTool {
    fn new(executor: Arc<dyn Executor>) -> Self {
        Self { executor }
    }
}

#[async_trait]
impl Tool for RunCommandTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "run_command".into(),
            description: "Run a guarded command once.".into(),
            requires_approval: true,
        }
    }

    async fn call(&self, ctx: ToolContext, input: Value) -> ClawResult<ToolOutput> {
        #[derive(Deserialize)]
        struct Input {
            command: String,
            args: Option<Vec<String>>,
            cwd: Option<PathBuf>,
        }
        let input: Input = serde_json::from_value(input)
            .map_err(|err| ClawError::InvalidRequest(err.to_string()))?;
        let result = self
            .executor
            .run_once(ExecRequest {
                command: input.command,
                args: input.args.unwrap_or_default(),
                cwd: input.cwd.or(ctx.working_directory),
                env_allowlist: vec![],
                timeout_secs: 60,
                max_output_bytes: 24 * 1024,
                requires_tty: false,
            })
            .await?;
        Ok(command_output(result))
    }
}

struct SessionOpenTool {
    executor: Arc<dyn Executor>,
}

impl SessionOpenTool {
    fn new(executor: Arc<dyn Executor>) -> Self {
        Self { executor }
    }
}

#[async_trait]
impl Tool for SessionOpenTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "session_open".into(),
            description: "Open a persistent shell session.".into(),
            requires_approval: false,
        }
    }

    async fn call(&self, ctx: ToolContext, input: Value) -> ClawResult<ToolOutput> {
        #[derive(Deserialize)]
        struct Input {
            name: Option<String>,
            cwd: Option<PathBuf>,
        }
        let input: Input = serde_json::from_value(input)
            .map_err(|err| ClawError::InvalidRequest(err.to_string()))?;
        let session = self
            .executor
            .open_session(SessionOpenRequest {
                name: input.name,
                cwd: input.cwd.or(ctx.working_directory),
                env_allowlist: vec![],
            })
            .await?;
        Ok(ToolOutput {
            content: format!("opened session {} at {}", session.id, session.cwd.display()),
            structured: Some(json!({
                "session_id": session.id.to_string(),
                "cwd": session.cwd,
                "name": session.name,
            })),
        })
    }
}

struct SessionExecTool {
    executor: Arc<dyn Executor>,
}

impl SessionExecTool {
    fn new(executor: Arc<dyn Executor>) -> Self {
        Self { executor }
    }
}

#[async_trait]
impl Tool for SessionExecTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "session_exec".into(),
            description: "Run a command in an existing session.".into(),
            requires_approval: true,
        }
    }

    async fn call(&self, _ctx: ToolContext, input: Value) -> ClawResult<ToolOutput> {
        #[derive(Deserialize)]
        struct Input {
            session_id: String,
            command: String,
        }
        let input: Input = serde_json::from_value(input)
            .map_err(|err| ClawError::InvalidRequest(err.to_string()))?;
        let session_id = SessionId(
            uuid::Uuid::parse_str(&input.session_id)
                .map_err(|err| ClawError::InvalidRequest(err.to_string()))?,
        );
        let (command, args) = shell_request_parts(&input.command);
        let result = self
            .executor
            .run_in_session(
                session_id,
                ExecRequest {
                    command,
                    args,
                    cwd: None,
                    env_allowlist: vec![],
                    timeout_secs: 120,
                    max_output_bytes: 24 * 1024,
                    requires_tty: false,
                },
            )
            .await?;
        Ok(command_output(result))
    }
}

struct SessionCloseTool {
    executor: Arc<dyn Executor>,
}

impl SessionCloseTool {
    fn new(executor: Arc<dyn Executor>) -> Self {
        Self { executor }
    }
}

#[async_trait]
impl Tool for SessionCloseTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "session_close".into(),
            description: "Close a shell session.".into(),
            requires_approval: false,
        }
    }

    async fn call(&self, _ctx: ToolContext, input: Value) -> ClawResult<ToolOutput> {
        #[derive(Deserialize)]
        struct Input {
            session_id: String,
        }
        let input: Input = serde_json::from_value(input)
            .map_err(|err| ClawError::InvalidRequest(err.to_string()))?;
        let session_id = SessionId(
            uuid::Uuid::parse_str(&input.session_id)
                .map_err(|err| ClawError::InvalidRequest(err.to_string()))?,
        );
        self.executor.close_session(session_id).await?;
        Ok(ToolOutput {
            content: format!("closed session {}", input.session_id),
            structured: None,
        })
    }
}

fn shell_request_parts(command: &str) -> (String, Vec<String>) {
    if cfg!(windows) {
        ("cmd".into(), vec!["/C".into(), command.into()])
    } else {
        ("sh".into(), vec!["-lc".into(), command.into()])
    }
}

fn command_output(result: hajimi_claw_types::ExecResult) -> ToolOutput {
    ToolOutput {
        content: format_output(
            &result.stdout,
            &result.stderr,
            result.exit_code,
            result.truncated,
        ),
        structured: Some(json!({
            "exit_code": result.exit_code,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "truncated": result.truncated,
            "duration_ms": result.duration_ms,
        })),
    }
}

fn format_output(stdout: &str, stderr: &str, exit_code: Option<i32>, truncated: bool) -> String {
    let mut parts = vec![format!("exit_code={}", exit_code.unwrap_or(-1))];
    if !stdout.trim().is_empty() {
        parts.push(format!("stdout:\n{}", stdout.trim()));
    }
    if !stderr.trim().is_empty() {
        parts.push(format!("stderr:\n{}", stderr.trim()));
    }
    if truncated {
        parts.push("output was truncated".into());
    }
    parts.join("\n\n")
}

fn guard_path(policy: &PolicyEngine, path: &Path) -> ClawResult<()> {
    let candidate = path.parent().unwrap_or(path);
    guard_dir(policy, candidate)
}

fn guard_dir(policy: &PolicyEngine, path: &Path) -> ClawResult<()> {
    if !policy.is_allowed_workdir(path) {
        return Err(ClawError::AccessDenied(format!(
            "path is outside allowed directories: {}",
            path.display()
        )));
    }
    Ok(())
}

fn guard_write_path(policy: &PolicyEngine, path: &Path) -> ClawResult<()> {
    let candidate = path.parent().unwrap_or(path);
    guard_dir(policy, candidate)?;
    if !policy.is_writable_workdir(candidate) {
        return Err(ClawError::AccessDenied(format!(
            "path is outside writable directories: {}",
            candidate.display()
        )));
    }
    Ok(())
}

fn truncate_string(content: String, max_bytes: usize) -> String {
    if content.len() <= max_bytes {
        return content;
    }
    let tail = &content[content.len() - max_bytes..];
    format!("...[truncated]\n{tail}")
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use anyhow::Result;
    use hajimi_claw_exec::{LocalExecutor, PlatformMode};
    use hajimi_claw_policy::{PolicyConfig, PolicyEngine};
    use hajimi_claw_types::{ConversationId, ToolContext};
    use serde_json::json;
    use tempfile::tempdir;

    use super::ToolRegistry;

    #[tokio::test]
    async fn read_file_respects_policy() -> Result<()> {
        let dir = tempdir()?;
        let path = dir.path().join("demo.txt");
        tokio::fs::write(&path, "hello").await?;

        let mut config = PolicyConfig::default();
        config.allowed_workdirs = vec![dir.path().to_path_buf()];
        let policy = Arc::new(PolicyEngine::new(config));
        let executor = Arc::new(LocalExecutor::new(
            policy.clone(),
            PlatformMode::WindowsSafe,
        ));
        let tools = ToolRegistry::default(executor, policy);

        let output = tools
            .call(
                "read_file",
                ToolContext {
                    conversation_id: ConversationId::new(),
                    working_directory: None,
                    elevated: false,
                },
                json!({ "path": path }),
            )
            .await?;
        assert_eq!(output.content, "hello");
        Ok(())
    }
}
