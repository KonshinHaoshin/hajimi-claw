use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;

use anyhow::{Context, Result};
use chrono::Utc;
use hajimi_claw_agent::AgentRuntime;
use hajimi_claw_bot::{TelegramBot, TelegramConfig};
use hajimi_claw_exec::{LocalExecutor, PlatformMode};
use hajimi_claw_gateway::InProcessGateway;
use hajimi_claw_llm::{StaticBackend, StoreBackedBackend, list_models, test_provider};
use hajimi_claw_policy::{PolicyConfig, PolicyEngine};
use hajimi_claw_store::{SecretCipher, Store};
use hajimi_claw_tools::ToolRegistry;
use hajimi_claw_types::{ProviderConfig, ProviderKind, ProviderRecord};
use serde::{Deserialize, Serialize};
use tracing_subscriber::EnvFilter;
use uuid::Uuid;

const DEFAULT_MASTER_KEY_ENV: &str = "HAJIMI_CLAW_MASTER_KEY";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppConfig {
    pub telegram: TelegramSection,
    pub llm: LlmSection,
    pub storage: StorageSection,
    pub security: SecuritySection,
    pub policy: PolicyConfig,
    pub execution: ExecutionSection,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TelegramSection {
    pub bot_token: String,
    pub poll_timeout_secs: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmSection {
    pub base_url: Option<String>,
    pub api_key: Option<String>,
    pub model: Option<String>,
    pub static_fallback_response: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageSection {
    pub sqlite_path: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecuritySection {
    pub master_key_env: Option<String>,
    pub master_key_file: Option<PathBuf>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionSection {
    pub mode: Option<String>,
}

#[derive(Debug, Clone)]
pub struct LoadedConfig {
    pub path: PathBuf,
    pub config: AppConfig,
}

pub async fn entry_from_env() -> Result<()> {
    init_tracing();
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    match args.first().map(String::as_str) {
        None => run_from_env().await,
        Some("daemon" | "run" | "start") => run_from_env().await,
        Some("onboard") => {
            let path = resolve_or_default_config_path()?;
            interactive_onboard(path).await
        }
        Some("models") => {
            let provider_id = args.get(1).cloned();
            cli_models(provider_id).await
        }
        Some("restart") => cli_restart(),
        Some("help" | "--help" | "-h") => {
            print_help();
            Ok(())
        }
        Some(other) => {
            anyhow::bail!("unknown command `{other}`\n\n{}", help_text())
        }
    }
}

pub async fn run_from_env() -> Result<()> {
    let loaded = load_config_from_env_or_default()?;
    run(loaded.config).await
}

pub async fn run(config: AppConfig) -> Result<()> {
    if let Some(parent) = config.storage.sqlite_path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("create storage directory {}", parent.display()))?;
    }
    let policy = Arc::new(PolicyEngine::new(config.policy.clone()));
    let store = open_store(&config)?;
    let executor = Arc::new(LocalExecutor::new(
        policy.clone(),
        select_platform_mode(config.execution.mode.as_deref()),
    ));
    let tools = Arc::new(ToolRegistry::default(executor, policy.clone()));
    let fallback = Arc::new(StaticBackend::new(
        config
            .llm
            .static_fallback_response
            .clone()
            .unwrap_or_else(|| "LLM backend not configured.".into()),
    ));
    bootstrap_provider_if_configured(&store, &config)?;
    let llm: Arc<dyn hajimi_claw_types::LlmBackend> =
        Arc::new(StoreBackedBackend::new(store.clone(), Some(fallback)));

    let runtime = Arc::new(AgentRuntime::new(llm, tools, store.clone(), policy.clone()));
    let gateway = Arc::new(InProcessGateway::new(
        runtime,
        policy.clone(),
        store.clone(),
    ));
    let bot = TelegramBot::new(
        TelegramConfig {
            token: config.telegram.bot_token,
            poll_timeout_secs: config.telegram.poll_timeout_secs.unwrap_or(30),
            admin_user_id: config.policy.admin_user_id,
            admin_chat_id: config.policy.admin_chat_id,
        },
        gateway,
    );

    bot.run().await
}

pub fn load_config(path: PathBuf) -> Result<AppConfig> {
    let raw = fs::read_to_string(&path)
        .with_context(|| format!("read config file {}", path.display()))?;
    let mut config: AppConfig = toml::from_str(&raw).context("parse config.toml")?;
    relativize_config(&mut config, &path);
    Ok(config)
}

pub fn load_config_from_env_or_default() -> Result<LoadedConfig> {
    if let Ok(path) = std::env::var("HAJIMI_CLAW_CONFIG") {
        let path = PathBuf::from(path);
        if !path.exists() {
            anyhow::bail!(
                "config file not found at {}\nRun `hajimi onboard` to create one or update `HAJIMI_CLAW_CONFIG`.",
                path.display()
            );
        }
        return Ok(LoadedConfig {
            config: load_config(path.clone())?,
            path,
        });
    }

    let path = resolve_or_default_config_path()?;
    if !path.exists() {
        anyhow::bail!(
            "config file not found at {}\nRun `hajimi onboard` to create one.",
            path.display()
        );
    }

    Ok(LoadedConfig {
        config: load_config(path.clone())?,
        path,
    })
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

fn relativize_config(config: &mut AppConfig, config_path: &Path) {
    let base = config_path.parent().unwrap_or_else(|| Path::new("."));
    if config.storage.sqlite_path.is_relative() {
        config.storage.sqlite_path = base.join(&config.storage.sqlite_path);
    }
    if let Some(path) = config.security.master_key_file.as_mut() {
        if path.is_relative() {
            *path = base.join(&*path);
        }
    }
    config.policy.allowed_workdirs = config
        .policy
        .allowed_workdirs
        .iter()
        .map(|path| {
            if path.is_relative() {
                base.join(path)
            } else {
                path.clone()
            }
        })
        .collect();
    config.policy.writable_workdirs = config
        .policy
        .writable_workdirs
        .iter()
        .map(|path| {
            if path.is_relative() {
                base.join(path)
            } else {
                path.clone()
            }
        })
        .collect();
}

fn open_store(config: &AppConfig) -> Result<Arc<Store>> {
    let secret = load_master_key(config)?;
    let cipher = Arc::new(SecretCipher::from_passphrase(&secret)?);
    Ok(Arc::new(Store::open_with_cipher(
        &config.storage.sqlite_path,
        Some(cipher),
    )?))
}

fn bootstrap_provider_if_configured(store: &Store, config: &AppConfig) -> Result<()> {
    if let (Some(base_url), Some(api_key), Some(model)) = (
        config.llm.base_url.clone(),
        config.llm.api_key.clone(),
        config.llm.model.clone(),
    ) {
        let bootstrap_record = ProviderRecord {
            config: ProviderConfig {
                id: "bootstrap".into(),
                label: "Bootstrap".into(),
                kind: ProviderKind::OpenAiCompatible,
                base_url,
                api_key,
                model,
                enabled: true,
                extra_headers: vec![],
                created_at: Utc::now(),
            },
            is_default: store.get_default_provider()?.is_none(),
        };
        store.upsert_provider(&bootstrap_record)?;
    }
    Ok(())
}

fn load_master_key(config: &AppConfig) -> Result<String> {
    let env_name = config
        .security
        .master_key_env
        .clone()
        .unwrap_or_else(|| DEFAULT_MASTER_KEY_ENV.into());
    if let Ok(value) = std::env::var(&env_name) {
        if !value.trim().is_empty() {
            return Ok(value);
        }
    }
    if let Some(path) = &config.security.master_key_file {
        let value = fs::read_to_string(path)
            .with_context(|| format!("read master key file {}", path.display()))?;
        let trimmed = value.trim().to_string();
        if !trimmed.is_empty() {
            return Ok(trimmed);
        }
    }
    anyhow::bail!(
        "missing provider encryption key; set `{}` or create `{}`",
        env_name,
        config
            .security
            .master_key_file
            .as_ref()
            .map(|path| path.display().to_string())
            .unwrap_or_else(|| "<master-key-file>".into())
    )
}

fn resolve_or_default_config_path() -> Result<PathBuf> {
    for candidate in config_search_paths()? {
        if candidate.exists() {
            return Ok(candidate);
        }
    }
    Ok(default_config_path()?)
}

fn config_search_paths() -> Result<Vec<PathBuf>> {
    let mut paths = Vec::new();
    paths.push(std::env::current_dir()?.join("config.toml"));
    let exe = std::env::current_exe()?;
    if let Some(dir) = exe.parent() {
        paths.push(dir.join("config.toml"));
        if let Some(parent) = dir.parent() {
            paths.push(parent.join("config.toml"));
        }
    }
    paths.push(default_config_path()?);
    Ok(paths)
}

fn default_config_path() -> Result<PathBuf> {
    let base = if cfg!(windows) {
        std::env::var_os("APPDATA")
            .map(PathBuf::from)
            .or_else(|| std::env::var_os("LOCALAPPDATA").map(PathBuf::from))
            .context("APPDATA or LOCALAPPDATA is not set")?
    } else {
        std::env::var_os("XDG_CONFIG_HOME")
            .map(PathBuf::from)
            .or_else(|| std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".config")))
            .context("XDG_CONFIG_HOME or HOME is not set")?
    };
    Ok(base.join("hajimi-claw").join("config.toml"))
}

fn default_storage_path(config_path: &Path) -> PathBuf {
    config_path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join("data")
        .join("hajimi-claw.sqlite3")
}

fn default_master_key_file(config_path: &Path) -> PathBuf {
    config_path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join("master.key")
}

fn default_workdirs(config_path: &Path) -> Vec<PathBuf> {
    let mut paths = vec![std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))];
    let data_dir = config_path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join("data");
    paths.push(data_dir);
    paths.push(std::env::temp_dir());
    paths
}

async fn interactive_onboard(config_path: PathBuf) -> Result<()> {
    if let Some(parent) = config_path.parent() {
        fs::create_dir_all(parent)?;
    }

    let mut config = if config_path.exists() {
        load_config(config_path.clone())?
    } else {
        default_app_config(&config_path)
    };

    println!("hajimi onboard");
    println!("config path: {}", config_path.display());

    if config.telegram.bot_token.trim().is_empty()
        || config.telegram.bot_token.contains("replace-me")
    {
        config.telegram.bot_token = prompt("Telegram bot token")?;
    }
    if config.policy.admin_user_id == 0 {
        config.policy.admin_user_id = prompt("Telegram admin user id")?.parse()?;
    }
    if config.policy.admin_chat_id == 0 {
        config.policy.admin_chat_id = prompt("Telegram admin chat id")?.parse()?;
    }

    let master_key_file = config
        .security
        .master_key_file
        .clone()
        .unwrap_or_else(|| default_master_key_file(&config_path));
    config.security.master_key_file = Some(master_key_file.clone());
    config.security.master_key_env = Some(
        config
            .security
            .master_key_env
            .clone()
            .unwrap_or_else(|| DEFAULT_MASTER_KEY_ENV.into()),
    );

    if !master_key_file.exists() {
        if let Some(parent) = master_key_file.parent() {
            fs::create_dir_all(parent)?;
        }
        let generated = format!("{}{}", Uuid::new_v4().simple(), Uuid::new_v4().simple());
        fs::write(&master_key_file, &generated)
            .with_context(|| format!("write master key file {}", master_key_file.display()))?;
        println!("generated master key file: {}", master_key_file.display());
    }

    save_config(&config_path, &config)?;

    let store = open_store(&config)?;
    bootstrap_provider_if_configured(&store, &config)?;

    println!();
    println!("Provider onboarding");
    let label = prompt_default("Provider label", "OpenAI")?;
    let kind = parse_provider_kind(&prompt_default(
        "Provider kind (openai-compatible/custom-chat-completions)",
        "openai-compatible",
    )?)?;
    let base_url = prompt_default("Provider base URL", "https://api.openai.com/v1")?;
    let api_key = prompt("Provider API key")?;
    let model = prompt_default("Default model", "gpt-4.1-mini")?;

    let record = ProviderRecord {
        config: ProviderConfig {
            id: slugify(&label),
            label,
            kind,
            base_url,
            api_key,
            model,
            enabled: true,
            extra_headers: vec![],
            created_at: Utc::now(),
        },
        is_default: store.get_default_provider()?.is_none(),
    };
    store.upsert_provider(&record)?;

    let health = test_provider(&reqwest::Client::new(), &record.config).await?;
    println!();
    println!(
        "Saved provider `{}` ({})",
        record.config.label, record.config.id
    );
    println!("Health: {}", health.message);
    println!("Config saved to {}", config_path.display());
    println!("Master key file: {}", master_key_file.display());
    Ok(())
}

async fn cli_models(provider_id: Option<String>) -> Result<()> {
    let loaded = load_config_from_env_or_default()?;
    let store = open_store(&loaded.config)?;
    bootstrap_provider_if_configured(&store, &loaded.config)?;
    let provider = if let Some(provider_id) = provider_id {
        store
            .get_provider(&provider_id)?
            .with_context(|| format!("provider not found: {provider_id}"))?
    } else {
        store
            .get_default_provider()?
            .context("no default provider configured; run `hajimi onboard`")?
    };
    let models = list_models(&reqwest::Client::new(), &provider.config).await?;
    if models.is_empty() {
        println!("provider returned no models");
    } else {
        for model in models {
            println!("{model}");
        }
    }
    Ok(())
}

fn cli_restart() -> Result<()> {
    #[cfg(windows)]
    {
        let status = Command::new("powershell")
            .args([
                "-NoProfile",
                "-Command",
                "Restart-Service -Name hajimi-claw -ErrorAction Stop",
            ])
            .status()
            .context("invoke Restart-Service")?;
        if !status.success() {
            anyhow::bail!(
                "failed to restart Windows service `hajimi-claw`; ensure the service exists"
            );
        }
    }

    #[cfg(not(windows))]
    {
        let status = Command::new("systemctl")
            .args(["restart", "hajimi-claw"])
            .status()
            .context("invoke systemctl restart hajimi-claw")?;
        if !status.success() {
            anyhow::bail!(
                "failed to restart systemd service `hajimi-claw`; ensure the service exists"
            );
        }
    }

    println!("hajimi-claw restarted");
    Ok(())
}

fn default_app_config(config_path: &Path) -> AppConfig {
    let storage = default_storage_path(config_path);
    let workdirs = default_workdirs(config_path);
    AppConfig {
        telegram: TelegramSection {
            bot_token: String::new(),
            poll_timeout_secs: Some(30),
        },
        llm: LlmSection {
            base_url: None,
            api_key: None,
            model: None,
            static_fallback_response: Some("LLM backend not configured.".into()),
        },
        storage: StorageSection {
            sqlite_path: storage,
        },
        security: SecuritySection {
            master_key_env: Some(DEFAULT_MASTER_KEY_ENV.into()),
            master_key_file: Some(default_master_key_file(config_path)),
        },
        policy: PolicyConfig {
            admin_user_id: 0,
            admin_chat_id: 0,
            allowed_workdirs: workdirs.clone(),
            writable_workdirs: vec![workdirs[1].clone(), workdirs[2].clone()],
            ..PolicyConfig::default()
        },
        execution: ExecutionSection {
            mode: Some("auto".into()),
        },
    }
}

fn save_config(path: &Path, config: &AppConfig) -> Result<()> {
    let mut output = config.clone();
    make_config_relative(&mut output, path);
    let serialized = toml::to_string_pretty(&output).context("serialize config")?;
    fs::write(path, serialized).with_context(|| format!("write config {}", path.display()))?;
    Ok(())
}

fn make_config_relative(config: &mut AppConfig, config_path: &Path) {
    let base = config_path.parent().unwrap_or_else(|| Path::new("."));
    config.storage.sqlite_path = make_relative(&config.storage.sqlite_path, base);
    if let Some(path) = config.security.master_key_file.as_mut() {
        *path = make_relative(path, base);
    }
    config.policy.allowed_workdirs = config
        .policy
        .allowed_workdirs
        .iter()
        .map(|path| make_relative(path, base))
        .collect();
    config.policy.writable_workdirs = config
        .policy
        .writable_workdirs
        .iter()
        .map(|path| make_relative(path, base))
        .collect();
}

fn make_relative(path: &Path, base: &Path) -> PathBuf {
    path.strip_prefix(base).unwrap_or(path).to_path_buf()
}

fn prompt(label: &str) -> Result<String> {
    print!("{label}: ");
    io::stdout().flush()?;
    let mut buffer = String::new();
    io::stdin().read_line(&mut buffer)?;
    let trimmed = buffer.trim().to_string();
    if trimmed.is_empty() {
        anyhow::bail!("{label} cannot be empty");
    }
    Ok(trimmed)
}

fn prompt_default(label: &str, default: &str) -> Result<String> {
    print!("{label} [{default}]: ");
    io::stdout().flush()?;
    let mut buffer = String::new();
    io::stdin().read_line(&mut buffer)?;
    let trimmed = buffer.trim();
    if trimmed.is_empty() {
        Ok(default.to_string())
    } else {
        Ok(trimmed.to_string())
    }
}

fn parse_provider_kind(raw: &str) -> Result<ProviderKind> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "openai-compatible" | "openai" => Ok(ProviderKind::OpenAiCompatible),
        "custom-chat-completions" | "custom" => Ok(ProviderKind::CustomChatCompletions),
        _ => {
            anyhow::bail!("provider kind must be `openai-compatible` or `custom-chat-completions`")
        }
    }
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
    let slug = slug.trim_matches('-').to_string();
    if slug.is_empty() {
        "provider".into()
    } else {
        slug
    }
}

fn print_help() {
    println!("{}", help_text());
}

fn help_text() -> &'static str {
    "Usage:
  hajimi                 Run the daemon
  hajimi daemon          Run the daemon
  hajimi onboard         Interactive local onboarding
  hajimi models [id]     List models for the default or named provider
  hajimi restart         Restart the installed service
  hajimi help            Show this help"
}

#[cfg(test)]
mod tests {
    use super::{default_config_path, select_platform_mode, slugify};
    use hajimi_claw_exec::PlatformMode;

    #[test]
    fn selects_explicit_mode() {
        assert!(matches!(
            select_platform_mode(Some("windows-safe")),
            PlatformMode::WindowsSafe
        ));
    }

    #[test]
    fn slugifies_labels() {
        assert_eq!(slugify("Moonshot AI"), "moonshot-ai");
    }

    #[test]
    fn default_config_path_has_filename() {
        assert_eq!(
            default_config_path().unwrap().file_name().unwrap(),
            "config.toml"
        );
    }
}
