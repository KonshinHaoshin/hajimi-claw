use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result};
use hajimi_claw_agent::AgentRuntime;
use hajimi_claw_bot::{TelegramBot, TelegramConfig};
use hajimi_claw_exec::{LocalExecutor, PlatformMode};
use hajimi_claw_llm::{OpenAiCompatibleBackend, StaticBackend};
use hajimi_claw_policy::{PolicyConfig, PolicyEngine};
use hajimi_claw_store::Store;
use hajimi_claw_tools::ToolRegistry;
use serde::Deserialize;
use tracing_subscriber::EnvFilter;

#[derive(Debug, Clone, Deserialize)]
pub struct AppConfig {
    pub telegram: TelegramSection,
    pub llm: LlmSection,
    pub storage: StorageSection,
    pub policy: PolicyConfig,
    pub execution: ExecutionSection,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TelegramSection {
    pub bot_token: String,
    pub poll_timeout_secs: Option<u64>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct LlmSection {
    pub base_url: Option<String>,
    pub api_key: Option<String>,
    pub model: Option<String>,
    pub static_fallback_response: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct StorageSection {
    pub sqlite_path: PathBuf,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ExecutionSection {
    pub mode: Option<String>,
}

pub async fn run_from_env() -> Result<()> {
    init_tracing();
    let config_path = std::env::var("HAJIMI_CLAW_CONFIG").unwrap_or_else(|_| "config.toml".into());
    let config = load_config(PathBuf::from(config_path))?;
    run(config).await
}

pub async fn run(config: AppConfig) -> Result<()> {
    if let Some(parent) = config.storage.sqlite_path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("create storage directory {}", parent.display()))?;
    }
    let policy = Arc::new(PolicyEngine::new(config.policy.clone()));
    let store = Arc::new(Store::open(&config.storage.sqlite_path)?);
    let executor = Arc::new(LocalExecutor::new(
        policy.clone(),
        select_platform_mode(config.execution.mode.as_deref()),
    ));
    let tools = Arc::new(ToolRegistry::default(executor, policy.clone()));
    let llm: Arc<dyn hajimi_claw_types::LlmBackend> =
        if let (Some(base_url), Some(api_key), Some(model)) = (
            config.llm.base_url.clone(),
            config.llm.api_key.clone(),
            config.llm.model.clone(),
        ) {
            Arc::new(OpenAiCompatibleBackend::new(base_url, api_key, model))
        } else {
            Arc::new(StaticBackend::new(
                config
                    .llm
                    .static_fallback_response
                    .unwrap_or_else(|| "LLM backend not configured.".into()),
            ))
        };

    let runtime = Arc::new(AgentRuntime::new(llm, tools, store, policy.clone()));
    let bot = TelegramBot::new(
        TelegramConfig {
            token: config.telegram.bot_token,
            poll_timeout_secs: config.telegram.poll_timeout_secs.unwrap_or(30),
            admin_user_id: config.policy.admin_user_id,
            admin_chat_id: config.policy.admin_chat_id,
        },
        runtime,
    );

    bot.run().await
}

pub fn load_config(path: PathBuf) -> Result<AppConfig> {
    let raw = std::fs::read_to_string(&path)
        .with_context(|| format!("read config file {}", path.display()))?;
    let config: AppConfig = toml::from_str(&raw).context("parse config.toml")?;
    Ok(config)
}

fn init_tracing() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .try_init();
}

fn select_platform_mode(mode: Option<&str>) -> PlatformMode {
    match mode.unwrap_or("auto") {
        "windows-safe" => PlatformMode::WindowsSafe,
        "windows-elevated" => PlatformMode::WindowsElevated,
        "unix" => PlatformMode::Unix,
        _ if cfg!(windows) => PlatformMode::WindowsSafe,
        _ => PlatformMode::Unix,
    }
}

#[cfg(test)]
mod tests {
    use super::select_platform_mode;
    use hajimi_claw_exec::PlatformMode;

    #[test]
    fn selects_explicit_mode() {
        assert!(matches!(
            select_platform_mode(Some("windows-safe")),
            PlatformMode::WindowsSafe
        ));
    }
}
