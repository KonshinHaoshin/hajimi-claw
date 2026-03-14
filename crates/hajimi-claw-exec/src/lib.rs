use std::collections::HashMap;
use std::path::PathBuf;
use std::process::Stdio;
use std::sync::Arc;
use std::time::Instant;

use async_trait::async_trait;
use chrono::Utc;
use hajimi_claw_policy::{PolicyDecision, PolicyEngine};
use hajimi_claw_types::{
    ClawError, ClawResult, ExecRequest, ExecResult, Executor, SessionHandle, SessionId,
    SessionOpenRequest,
};
use tokio::process::Command;
use tokio::sync::Mutex;
use tokio::time::{Duration, timeout};

#[derive(Debug, Clone, Copy)]
pub enum PlatformMode {
    Unix,
    WindowsSafe,
    WindowsElevated,
}

#[derive(Debug, Clone)]
struct SessionState {
    handle: SessionHandle,
    env_allowlist: Vec<String>,
    history: Vec<String>,
}

pub struct LocalExecutor {
    policy: Arc<PolicyEngine>,
    mode: PlatformMode,
    sessions: Mutex<HashMap<SessionId, SessionState>>,
}

impl LocalExecutor {
    pub fn new(policy: Arc<PolicyEngine>, mode: PlatformMode) -> Self {
        Self {
            policy,
            mode,
            sessions: Mutex::new(HashMap::new()),
        }
    }

    pub fn policy(&self) -> &Arc<PolicyEngine> {
        &self.policy
    }

    async fn run_checked(&self, req: ExecRequest) -> ClawResult<ExecResult> {
        match self.policy.evaluate_exec(&req) {
            PolicyDecision::Allow { .. } => self.spawn(req).await,
            PolicyDecision::RequiresApproval(approval) => Err(ClawError::ApprovalRequired(
                format!("{} [{}]", approval.reason, approval.request_id),
            )),
            PolicyDecision::Deny(reason) => Err(ClawError::AccessDenied(reason)),
        }
    }

    async fn spawn(&self, req: ExecRequest) -> ClawResult<ExecResult> {
        if matches!(self.mode, PlatformMode::WindowsSafe) && !self.policy.is_elevated() {
            self.validate_windows_safe_request(&req)?;
        }

        let start = Instant::now();
        let mut command = build_command(&req, self.mode)?;
        let child = command
            .spawn()
            .map_err(|err| ClawError::Backend(err.to_string()))?;

        #[cfg(windows)]
        let _job = windows_job::attach_kill_on_close(&child).ok();

        let output = timeout(
            Duration::from_secs(req.timeout_secs),
            child.wait_with_output(),
        )
        .await
        .map_err(|_| ClawError::Backend("command timed out".into()))?
        .map_err(|err| ClawError::Backend(err.to_string()))?;

        let (stdout, stderr, truncated) =
            truncate_output(output.stdout, output.stderr, req.max_output_bytes);

        Ok(ExecResult {
            exit_code: output.status.code(),
            stdout,
            stderr,
            duration_ms: start.elapsed().as_millis(),
            truncated,
        })
    }

    fn validate_windows_safe_request(&self, req: &ExecRequest) -> ClawResult<()> {
        if !self.policy.windows_command_allowed(&req.command) {
            return Err(ClawError::AccessDenied(format!(
                "command is not in the Windows safe allowlist: {}",
                req.command
            )));
        }
        if let Some(cwd) = &req.cwd {
            if !self.policy.is_allowed_workdir(cwd) {
                return Err(ClawError::AccessDenied(format!(
                    "working directory is not allowed: {}",
                    cwd.display()
                )));
            }
        }
        Ok(())
    }
}

#[async_trait]
impl Executor for LocalExecutor {
    async fn run_once(&self, req: ExecRequest) -> ClawResult<ExecResult> {
        self.run_checked(req).await
    }

    async fn open_session(&self, req: SessionOpenRequest) -> ClawResult<SessionHandle> {
        let cwd = req
            .cwd
            .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));
        if !self.policy.is_allowed_workdir(&cwd) {
            return Err(ClawError::AccessDenied(format!(
                "working directory is not allowed: {}",
                cwd.display()
            )));
        }

        let handle = SessionHandle {
            id: SessionId::new(),
            name: req
                .name
                .unwrap_or_else(|| format!("shell-{}", Utc::now().timestamp())),
            cwd,
            created_at: Utc::now(),
            last_used_at: Utc::now(),
        };
        let state = SessionState {
            handle: handle.clone(),
            env_allowlist: req.env_allowlist,
            history: Vec::new(),
        };
        self.sessions.lock().await.insert(handle.id, state);
        Ok(handle)
    }

    async fn run_in_session(&self, id: SessionId, mut req: ExecRequest) -> ClawResult<ExecResult> {
        let mut sessions = self.sessions.lock().await;
        let state = sessions
            .get_mut(&id)
            .ok_or_else(|| ClawError::NotFound(format!("session not found: {id}")))?;

        if req.command.eq_ignore_ascii_case("cd") {
            let target = req
                .args
                .first()
                .map(PathBuf::from)
                .ok_or_else(|| ClawError::InvalidRequest("cd requires a target path".into()))?;
            let next = if target.is_absolute() {
                target
            } else {
                state.handle.cwd.join(target)
            };
            if !self.policy.is_allowed_workdir(&next) {
                return Err(ClawError::AccessDenied(format!(
                    "working directory is not allowed: {}",
                    next.display()
                )));
            }
            state.handle.cwd = next.clone();
            state.handle.last_used_at = Utc::now();
            state.history.push(format!("cd {}", next.display()));
            return Ok(ExecResult {
                exit_code: Some(0),
                stdout: format!("cwd -> {}", next.display()),
                stderr: String::new(),
                duration_ms: 0,
                truncated: false,
            });
        }

        req.cwd = Some(state.handle.cwd.clone());
        if req.env_allowlist.is_empty() {
            req.env_allowlist = state.env_allowlist.clone();
        }
        state.history.push(req.summary());
        state.handle.last_used_at = Utc::now();
        drop(sessions);

        self.run_checked(req).await
    }

    async fn close_session(&self, id: SessionId) -> ClawResult<()> {
        self.sessions
            .lock()
            .await
            .remove(&id)
            .ok_or_else(|| ClawError::NotFound(format!("session not found: {id}")))?;
        Ok(())
    }
}

fn build_command(req: &ExecRequest, mode: PlatformMode) -> ClawResult<Command> {
    let mut command = if matches!(mode, PlatformMode::Unix) {
        let mut cmd = Command::new(&req.command);
        cmd.args(&req.args);
        cmd
    } else {
        let program = req.command.to_ascii_lowercase();
        if matches!(
            program.as_str(),
            "powershell" | "powershell.exe" | "pwsh" | "pwsh.exe"
        ) {
            let mut cmd = Command::new(&req.command);
            cmd.arg("-NoProfile").args(&req.args);
            cmd
        } else {
            let mut cmd = Command::new(&req.command);
            cmd.args(&req.args);
            cmd
        }
    };

    command.kill_on_drop(true);
    command.stdin(Stdio::null());
    command.stdout(Stdio::piped());
    command.stderr(Stdio::piped());
    command.env_clear();

    for key in &req.env_allowlist {
        if let Ok(value) = std::env::var(key) {
            command.env(key, value);
        }
    }

    if let Some(cwd) = &req.cwd {
        command.current_dir(cwd);
    }

    #[cfg(unix)]
    {
        command.env("PATH", std::env::var("PATH").unwrap_or_default());
    }
    #[cfg(windows)]
    {
        command.env("PATH", std::env::var("PATH").unwrap_or_default());
        command.creation_flags(0x08000000);
    }

    Ok(command)
}

fn truncate_output(
    stdout: Vec<u8>,
    stderr: Vec<u8>,
    max_output_bytes: usize,
) -> (String, String, bool) {
    let mut truncated = false;
    let max_each = max_output_bytes / 2;

    let stdout = if stdout.len() > max_each {
        truncated = true;
        lossy_tail(stdout, max_each)
    } else {
        String::from_utf8_lossy(&stdout).into_owned()
    };
    let stderr = if stderr.len() > max_each {
        truncated = true;
        lossy_tail(stderr, max_each)
    } else {
        String::from_utf8_lossy(&stderr).into_owned()
    };

    (stdout, stderr, truncated)
}

fn lossy_tail(bytes: Vec<u8>, limit: usize) -> String {
    let slice = if bytes.len() > limit {
        &bytes[bytes.len() - limit..]
    } else {
        &bytes[..]
    };
    format!("...[truncated]\n{}", String::from_utf8_lossy(slice))
}

#[cfg(windows)]
mod windows_job {
    use tokio::process::Child;
    use windows::Win32::Foundation::{CloseHandle, HANDLE};
    use windows::Win32::System::JobObjects::{
        AssignProcessToJobObject, CreateJobObjectW, JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
        JOBOBJECT_EXTENDED_LIMIT_INFORMATION, JobObjectExtendedLimitInformation,
        SetInformationJobObject,
    };

    pub struct KillOnCloseJob(*mut core::ffi::c_void);

    unsafe impl Send for KillOnCloseJob {}
    unsafe impl Sync for KillOnCloseJob {}

    impl Drop for KillOnCloseJob {
        fn drop(&mut self) {
            unsafe {
                let _ = CloseHandle(HANDLE(self.0));
            }
        }
    }

    pub fn attach_kill_on_close(child: &Child) -> anyhow::Result<KillOnCloseJob> {
        unsafe {
            let job = CreateJobObjectW(None, None)?;
            let mut info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION::default();
            info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
            SetInformationJobObject(
                job,
                JobObjectExtendedLimitInformation,
                &info as *const _ as *const _,
                std::mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
            )?;

            let raw = child
                .raw_handle()
                .ok_or_else(|| anyhow::anyhow!("child does not expose a raw handle"))?;
            let process = HANDLE(raw);
            AssignProcessToJobObject(job, process)?;
            Ok(KillOnCloseJob(job.0))
        }
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::sync::Arc;

    use hajimi_claw_policy::{PolicyConfig, PolicyEngine};
    use hajimi_claw_types::{ExecRequest, Executor, SessionOpenRequest};

    use super::{LocalExecutor, PlatformMode};

    fn executor() -> LocalExecutor {
        let mut config = PolicyConfig::default();
        config.allowed_workdirs = vec![std::env::current_dir().unwrap(), std::env::temp_dir()];
        LocalExecutor::new(
            Arc::new(PolicyEngine::new(config)),
            PlatformMode::WindowsSafe,
        )
    }

    #[tokio::test]
    async fn truncates_output() {
        let executor = executor();
        let req = ExecRequest {
            command: "cmd".into(),
            args: vec!["/C".into(), "for /L %i in (1,1,300) do @echo line".into()],
            cwd: Some(std::env::temp_dir()),
            env_allowlist: vec![],
            timeout_secs: 10,
            max_output_bytes: 128,
            requires_tty: false,
        };

        let result = executor.run_once(req).await.expect("command succeeds");
        assert!(result.truncated);
    }

    #[tokio::test]
    async fn session_preserves_cwd() {
        let executor = executor();
        let dir = tempfile::tempdir().unwrap();
        let session = executor
            .open_session(SessionOpenRequest {
                name: None,
                cwd: Some(PathBuf::from(dir.path())),
                env_allowlist: vec![],
            })
            .await
            .expect("session opens");
        executor
            .run_in_session(
                session.id,
                ExecRequest {
                    command: "cd".into(),
                    args: vec![".".into()],
                    cwd: None,
                    env_allowlist: vec![],
                    timeout_secs: 10,
                    max_output_bytes: 128,
                    requires_tty: false,
                },
            )
            .await
            .expect("cd succeeds");

        let output = executor
            .run_in_session(
                session.id,
                ExecRequest {
                    command: "cmd".into(),
                    args: vec!["/C".into(), "cd".into()],
                    cwd: None,
                    env_allowlist: vec![],
                    timeout_secs: 10,
                    max_output_bytes: 256,
                    requires_tty: false,
                },
            )
            .await
            .expect("pwd succeeds");
        assert!(
            output
                .stdout
                .to_ascii_lowercase()
                .contains(&dir.path().display().to_string().to_ascii_lowercase())
        );
    }
}
