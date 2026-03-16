use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::RwLock;

use chrono::{DateTime, Duration, Utc};
use hajimi_claw_types::{ApprovalId, ApprovalRequest, ExecRequest, PolicyMode, RiskLevel};
use regex::Regex;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyConfig {
    pub admin_user_id: i64,
    pub admin_chat_id: i64,
    pub allowed_workdirs: Vec<PathBuf>,
    pub writable_workdirs: Vec<PathBuf>,
    pub windows_safe_allowlist: Vec<String>,
    pub guarded_patterns: Vec<String>,
    pub dangerous_patterns: Vec<String>,
    pub max_timeout_secs: u64,
    pub max_output_bytes: usize,
    pub session_idle_timeout_secs: u64,
}

impl Default for PolicyConfig {
    fn default() -> Self {
        Self {
            admin_user_id: 0,
            admin_chat_id: 0,
            allowed_workdirs: vec![std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))],
            writable_workdirs: vec![std::env::temp_dir()],
            windows_safe_allowlist: vec![
                "cmd".into(),
                "cmd.exe".into(),
                "powershell".into(),
                "powershell.exe".into(),
                "pwsh".into(),
                "pwsh.exe".into(),
                "docker".into(),
                "docker.exe".into(),
                "systemctl".into(),
                "type".into(),
                "cat".into(),
            ],
            guarded_patterns: vec![
                r"\b(systemctl|docker)\s+(restart|stop|rm)\b".into(),
                r"\b(chmod|chown|mv|cp)\b".into(),
            ],
            dangerous_patterns: vec![
                r"\b(rm|del)\s+(-rf|/s|/q|/f)\b".into(),
                r"\b(sudo|su|passwd|shutdown|reboot)\b".into(),
                r">\s*/".into(),
                r"format\s+".into(),
            ],
            max_timeout_secs: 120,
            max_output_bytes: 32 * 1024,
            session_idle_timeout_secs: 1800,
        }
    }
}

#[derive(Debug, Clone)]
pub enum PolicyDecision {
    Allow { risk: RiskLevel, mode: PolicyMode },
    RequiresApproval(ApprovalRequest),
    Deny(String),
}

#[derive(Debug, Clone)]
pub struct ElevationLease {
    pub reason: String,
    pub expires_at: Option<DateTime<Utc>>,
}

#[derive(Debug, Default)]
struct PolicyState {
    approvals: HashMap<ApprovalId, ApprovalRequest>,
    elevated_lease: Option<ElevationLease>,
}

pub struct PolicyEngine {
    config: PolicyConfig,
    guarded_patterns: Vec<Regex>,
    dangerous_patterns: Vec<Regex>,
    state: RwLock<PolicyState>,
}

impl PolicyEngine {
    pub fn new(config: PolicyConfig) -> Self {
        Self {
            guarded_patterns: compile_patterns(&config.guarded_patterns),
            dangerous_patterns: compile_patterns(&config.dangerous_patterns),
            config,
            state: RwLock::new(PolicyState::default()),
        }
    }

    pub fn config(&self) -> &PolicyConfig {
        &self.config
    }

    pub fn authorize_telegram_actor(&self, user_id: i64, chat_id: i64) -> bool {
        let user_allowed = self.config.admin_user_id == 0 || self.config.admin_user_id == user_id;
        let chat_allowed = self.config.admin_chat_id == 0 || self.config.admin_chat_id == chat_id;
        user_allowed && chat_allowed
    }

    pub fn current_mode(&self) -> PolicyMode {
        let state = self.state.read().expect("policy state poisoned");
        match &state.elevated_lease {
            Some(lease)
                if lease
                    .expires_at
                    .is_none_or(|expires_at| expires_at > Utc::now()) =>
            {
                PolicyMode::ElevatedLease
            }
            _ if !state.approvals.is_empty() => PolicyMode::ApprovalPending,
            _ => PolicyMode::Normal,
        }
    }

    pub fn is_elevated(&self) -> bool {
        matches!(self.current_mode(), PolicyMode::ElevatedLease)
    }

    pub fn request_elevation(&self, minutes: i64, reason: String) -> ApprovalRequest {
        let request = ApprovalRequest {
            request_id: ApprovalId::new(),
            reason,
            risk_level: RiskLevel::Dangerous,
            command_preview: format!("elevated lease for {} minute(s)", minutes),
            cwd: None,
            expires_at: Utc::now() + Duration::minutes(minutes),
        };
        self.state
            .write()
            .expect("policy state poisoned")
            .approvals
            .insert(request.request_id, request.clone());
        request
    }

    pub fn approve(&self, request_id: ApprovalId) -> Option<ApprovalRequest> {
        let mut state = self.state.write().expect("policy state poisoned");
        let request = state.approvals.remove(&request_id)?;
        if request.command_preview.starts_with("elevated lease") {
            state.elevated_lease = Some(ElevationLease {
                reason: request.reason.clone(),
                expires_at: Some(request.expires_at),
            });
        }
        Some(request)
    }

    pub fn enable_elevation(&self, minutes: i64, reason: String) {
        self.state
            .write()
            .expect("policy state poisoned")
            .elevated_lease = Some(ElevationLease {
            reason,
            expires_at: Some(Utc::now() + Duration::minutes(minutes)),
        });
    }

    pub fn enable_full_elevation(&self, reason: String) {
        self.state
            .write()
            .expect("policy state poisoned")
            .elevated_lease = Some(ElevationLease {
            reason,
            expires_at: None,
        });
    }

    pub fn reject(&self, request_id: ApprovalId) -> Option<ApprovalRequest> {
        self.state
            .write()
            .expect("policy state poisoned")
            .approvals
            .remove(&request_id)
    }

    pub fn stop_elevation(&self) {
        self.state
            .write()
            .expect("policy state poisoned")
            .elevated_lease = None;
    }

    pub fn expire_lease_if_needed(&self) {
        let mut state = self.state.write().expect("policy state poisoned");
        if state.elevated_lease.as_ref().is_some_and(|lease| {
            lease
                .expires_at
                .is_some_and(|expires_at| expires_at <= Utc::now())
        }) {
            state.elevated_lease = None;
        }
    }

    pub fn evaluate_exec(&self, req: &ExecRequest) -> PolicyDecision {
        self.expire_lease_if_needed();

        if req.timeout_secs == 0 || req.timeout_secs > self.config.max_timeout_secs {
            return PolicyDecision::Deny(format!(
                "timeout must be between 1 and {} seconds",
                self.config.max_timeout_secs
            ));
        }

        if req.max_output_bytes == 0 || req.max_output_bytes > self.config.max_output_bytes {
            return PolicyDecision::Deny(format!(
                "max_output_bytes must be between 1 and {}",
                self.config.max_output_bytes
            ));
        }

        if let Some(cwd) = &req.cwd {
            if !self.is_allowed_workdir(cwd) {
                return PolicyDecision::Deny(format!(
                    "working directory is not allowed: {}",
                    cwd.display()
                ));
            }
        }

        if is_sensitive_env(&req.env_allowlist) {
            return PolicyDecision::Deny("sensitive env vars cannot be inherited".into());
        }

        let rendered = req.summary();

        if self
            .dangerous_patterns
            .iter()
            .any(|pattern| pattern.is_match(&rendered))
        {
            if self.is_elevated() {
                return PolicyDecision::Allow {
                    risk: RiskLevel::Dangerous,
                    mode: PolicyMode::ElevatedLease,
                };
            }

            let approval = ApprovalRequest {
                request_id: ApprovalId::new(),
                reason: "dangerous command requires elevated lease".into(),
                risk_level: RiskLevel::Dangerous,
                command_preview: rendered,
                cwd: req.cwd.clone(),
                expires_at: Utc::now() + Duration::minutes(10),
            };
            self.state
                .write()
                .expect("policy state poisoned")
                .approvals
                .insert(approval.request_id, approval.clone());
            return PolicyDecision::RequiresApproval(approval);
        }

        if self
            .guarded_patterns
            .iter()
            .any(|pattern| pattern.is_match(&rendered))
        {
            let approval = ApprovalRequest {
                request_id: ApprovalId::new(),
                reason: "guarded command requires explicit approval".into(),
                risk_level: RiskLevel::Guarded,
                command_preview: rendered,
                cwd: req.cwd.clone(),
                expires_at: Utc::now() + Duration::minutes(10),
            };
            self.state
                .write()
                .expect("policy state poisoned")
                .approvals
                .insert(approval.request_id, approval.clone());
            return PolicyDecision::RequiresApproval(approval);
        }

        PolicyDecision::Allow {
            risk: RiskLevel::Safe,
            mode: self.current_mode(),
        }
    }

    pub fn windows_command_allowed(&self, command: &str) -> bool {
        let allowlist: HashSet<_> = self
            .config
            .windows_safe_allowlist
            .iter()
            .map(|cmd| cmd.to_ascii_lowercase())
            .collect();
        allowlist.contains(&command.to_ascii_lowercase())
    }

    pub fn is_allowed_workdir(&self, path: &Path) -> bool {
        normalize(path).is_some_and(|candidate| {
            self.config
                .allowed_workdirs
                .iter()
                .any(|root| path_within(&candidate, root))
        })
    }

    pub fn is_writable_workdir(&self, path: &Path) -> bool {
        normalize(path).is_some_and(|candidate| {
            self.config
                .writable_workdirs
                .iter()
                .any(|root| path_within(&candidate, root))
        })
    }
}

fn normalize(path: &Path) -> Option<PathBuf> {
    path.canonicalize()
        .ok()
        .or_else(|| Some(path.to_path_buf()))
}

fn path_within(path: &Path, root: &Path) -> bool {
    let root = normalize(root).unwrap_or_else(|| root.to_path_buf());
    path.starts_with(&root)
}

fn is_sensitive_env(allowlist: &[String]) -> bool {
    static BLOCKED: &[&str] = &[
        "AWS_SECRET_ACCESS_KEY",
        "OPENAI_API_KEY",
        "TELEGRAM_BOT_TOKEN",
        "SSH_AUTH_SOCK",
    ];
    allowlist.iter().any(|entry| {
        BLOCKED
            .iter()
            .any(|blocked| blocked.eq_ignore_ascii_case(entry))
    })
}

fn compile_patterns(patterns: &[String]) -> Vec<Regex> {
    patterns
        .iter()
        .filter_map(|pattern| Regex::new(pattern).ok())
        .collect()
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use hajimi_claw_types::ExecRequest;

    use super::{PolicyConfig, PolicyDecision, PolicyEngine};

    fn sample_request(command: &str, args: &[&str]) -> ExecRequest {
        ExecRequest {
            command: command.into(),
            args: args.iter().map(|item| item.to_string()).collect(),
            cwd: Some(std::env::temp_dir()),
            env_allowlist: vec![],
            timeout_secs: 30,
            max_output_bytes: 1024,
            requires_tty: false,
        }
    }

    #[test]
    fn denies_disallowed_workdir() {
        let engine = PolicyEngine::new(PolicyConfig {
            allowed_workdirs: vec![PathBuf::from("C:/allowed")],
            ..PolicyConfig::default()
        });
        let request = sample_request("cmd", &["/C", "echo hi"]);
        assert!(matches!(
            engine.evaluate_exec(&request),
            PolicyDecision::Deny(_)
        ));
    }

    #[test]
    fn guarded_commands_require_approval() {
        let mut config = PolicyConfig::default();
        config.allowed_workdirs = vec![std::env::temp_dir()];
        let engine = PolicyEngine::new(config);
        let request = sample_request("systemctl", &["restart", "nginx"]);
        assert!(matches!(
            engine.evaluate_exec(&request),
            PolicyDecision::RequiresApproval(_)
        ));
    }

    #[test]
    fn elevation_allows_dangerous_command_after_approval() {
        let mut config = PolicyConfig::default();
        config.allowed_workdirs = vec![std::env::temp_dir()];
        let engine = PolicyEngine::new(config);
        let request = sample_request("sudo", &["shutdown", "-r", "now"]);
        engine.enable_elevation(10, "maintenance".into());
        assert!(matches!(
            engine.evaluate_exec(&request),
            PolicyDecision::Allow { .. }
        ));
    }

    #[test]
    fn zero_admin_ids_allow_any_actor() {
        let engine = PolicyEngine::new(PolicyConfig::default());
        assert!(engine.authorize_telegram_actor(123, 456));
    }

    #[test]
    fn configured_admin_ids_still_restrict_actor() {
        let engine = PolicyEngine::new(PolicyConfig {
            admin_user_id: 123,
            admin_chat_id: 456,
            ..PolicyConfig::default()
        });
        assert!(engine.authorize_telegram_actor(123, 456));
        assert!(!engine.authorize_telegram_actor(123, 999));
        assert!(!engine.authorize_telegram_actor(999, 456));
    }
}
