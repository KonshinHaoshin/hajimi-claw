use std::fs;
use std::fs::OpenOptions;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use chrono::Utc;
use hajimi_claw_agent::{
    AgentRuntime, MarkdownPromptSource, MultiAgentConfig, default_system_prompt,
};
use hajimi_claw_bot::{FeishuBot, FeishuConfig, TelegramBot, TelegramConfig};
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
    #[serde(default)]
    pub channel: ChannelSection,
    pub telegram: TelegramSection,
    #[serde(default)]
    pub feishu: FeishuSection,
    pub llm: LlmSection,
    pub storage: StorageSection,
    pub security: SecuritySection,
    pub policy: PolicyConfig,
    pub execution: ExecutionSection,
    #[serde(default)]
    pub multi_agent: MultiAgentSection,
    #[serde(default)]
    pub persona: PersonaSection,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChannelSection {
    pub kind: String,
}

impl Default for ChannelSection {
    fn default() -> Self {
        Self {
            kind: "telegram".into(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TelegramSection {
    pub bot_token: String,
    pub poll_timeout_secs: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct FeishuSection {
    pub app_id: String,
    pub app_secret: String,
    pub listen_addr: Option<String>,
    pub event_path: Option<String>,
    pub card_callback_path: Option<String>,
    pub mode: Option<String>,
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiAgentSection {
    pub enabled: bool,
    pub auto_delegate: bool,
    pub default_workers: usize,
    pub max_workers: usize,
    pub worker_timeout_secs: u64,
    pub max_context_chars_per_worker: usize,
}

impl Default for MultiAgentSection {
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

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PersonaSection {
    pub directory: Option<PathBuf>,
    #[serde(default)]
    pub prompt_files: Vec<PathBuf>,
}

#[derive(Debug, Clone)]
pub struct LoadedConfig {
    pub path: PathBuf,
    pub config: AppConfig,
}

#[derive(Debug, Clone)]
struct TelegramBotIdentity {
    id: i64,
    username: Option<String>,
    first_name: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct TelegramPairing {
    user_id: i64,
    chat_id: i64,
}

pub async fn entry_from_env() -> Result<()> {
    init_tracing();
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    match args.first().map(String::as_str) {
        None => run_from_env().await,
        Some("daemon" | "run" | "start") => run_from_env().await,
        Some("launch") => cli_launch(),
        Some("stop") => cli_stop(),
        Some("status") => cli_status(),
        Some("providers") => cli_providers().await,
        Some("provider") => cli_provider_command(&args[1..]).await,
        Some("model") => cli_model_command(&args[1..]).await,
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
    run_loaded(loaded).await
}

pub async fn run(config: AppConfig) -> Result<()> {
    run_with_context(config, None).await
}

async fn run_loaded(loaded: LoadedConfig) -> Result<()> {
    run_with_context(loaded.config, Some(loaded.path)).await
}

async fn run_with_context(config: AppConfig, config_path: Option<PathBuf>) -> Result<()> {
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
    let persona_dir = resolve_persona_directory(&config);
    ensure_persona_files(&persona_dir)?;
    let prompt_files = resolve_persona_paths(config_path.as_deref(), &config.persona)?;
    let prompt_source = Arc::new(MarkdownPromptSource::new(
        default_system_prompt(),
        prompt_files,
    ));

    let runtime = Arc::new(AgentRuntime::new(
        llm,
        tools,
        store.clone(),
        policy.clone(),
        prompt_source,
        persona_dir,
        MultiAgentConfig {
            enabled: config.multi_agent.enabled,
            auto_delegate: config.multi_agent.auto_delegate,
            default_workers: config.multi_agent.default_workers,
            max_workers: config.multi_agent.max_workers,
            worker_timeout_secs: config.multi_agent.worker_timeout_secs,
            max_context_chars_per_worker: config.multi_agent.max_context_chars_per_worker,
        },
    ));
    let gateway = Arc::new(InProcessGateway::new(
        runtime,
        policy.clone(),
        store.clone(),
    ));
    match config.channel.kind.as_str() {
        "none" | "skip" => {
            anyhow::bail!(
                "no primary channel configured yet\nRun `hajimi onboard` and choose telegram or feishu to start the daemon."
            )
        }
        "feishu" => {
            let bot = Arc::new(FeishuBot::new(
                FeishuConfig {
                    app_id: config.feishu.app_id,
                    app_secret: config.feishu.app_secret,
                    listen_addr: config
                        .feishu
                        .listen_addr
                        .unwrap_or_else(|| "0.0.0.0:8787".into()),
                    event_path: normalize_event_path(
                        config
                            .feishu
                            .event_path
                            .as_deref()
                            .unwrap_or("/feishu/events"),
                    ),
                    card_callback_path: normalize_event_path(
                        config
                            .feishu
                            .card_callback_path
                            .as_deref()
                            .unwrap_or("/feishu/card"),
                    ),
                    mode: config
                        .feishu
                        .mode
                        .clone()
                        .unwrap_or_else(|| "webhook".into()),
                    admin_user_id: config.policy.admin_user_id,
                    admin_chat_id: config.policy.admin_chat_id,
                },
                gateway,
            ));
            bot.run().await
        }
        _ => {
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
    }
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
    config.storage.sqlite_path = expand_home(&config.storage.sqlite_path);
    if config.storage.sqlite_path.is_relative() {
        config.storage.sqlite_path = base.join(&config.storage.sqlite_path);
    }
    if let Some(path) = config.security.master_key_file.as_mut() {
        *path = expand_home(path);
        if path.is_relative() {
            *path = base.join(&*path);
        }
    }
    if let Some(path) = config.persona.directory.as_mut() {
        *path = expand_home(path);
        if path.is_relative() {
            *path = base.join(&*path);
        }
    }
    config.persona.prompt_files = config
        .persona
        .prompt_files
        .iter()
        .map(|path| {
            let path = expand_home(path);
            if path.is_relative() {
                base.join(path)
            } else {
                path
            }
        })
        .collect();
    config.policy.allowed_workdirs = config
        .policy
        .allowed_workdirs
        .iter()
        .map(|path| {
            let path = expand_home(path);
            if path.is_relative() {
                base.join(path)
            } else {
                path
            }
        })
        .collect();
    config.policy.writable_workdirs = config
        .policy
        .writable_workdirs
        .iter()
        .map(|path| {
            let path = expand_home(path);
            if path.is_relative() {
                base.join(path)
            } else {
                path
            }
        })
        .collect();
}

fn open_store(config: &AppConfig) -> Result<Arc<Store>> {
    if let Some(parent) = config.storage.sqlite_path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("create storage directory {}", parent.display()))?;
    }
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
                fallback_models: vec![],
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

fn default_hajimi_home() -> Result<PathBuf> {
    Ok(home_dir()?.join(".hajimi"))
}

fn home_dir() -> Result<PathBuf> {
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("USERPROFILE").map(PathBuf::from))
        .context("HOME or USERPROFILE is not set")
}

fn expand_home(path: &Path) -> PathBuf {
    let rendered = path.to_string_lossy();
    if rendered == "~" {
        return home_dir().unwrap_or_else(|_| path.to_path_buf());
    }
    if let Some(rest) = rendered
        .strip_prefix("~/")
        .or_else(|| rendered.strip_prefix("~\\"))
    {
        return home_dir()
            .map(|home| home.join(rest))
            .unwrap_or_else(|_| path.to_path_buf());
    }
    path.to_path_buf()
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

fn default_persona_dir() -> PathBuf {
    default_hajimi_home().unwrap_or_else(|_| PathBuf::from(".hajimi"))
}

fn default_workdirs(config_path: &Path) -> Vec<PathBuf> {
    let mut paths = vec![std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))];
    let data_dir = config_path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join("data");
    paths.push(data_dir);
    paths.push(default_persona_dir());
    paths.push(std::env::temp_dir());
    paths
}

fn resolve_persona_paths(
    config_path: Option<&Path>,
    persona: &PersonaSection,
) -> Result<Vec<PathBuf>> {
    let mut paths = Vec::new();
    if !persona.prompt_files.is_empty() {
        for path in &persona.prompt_files {
            let resolved = if path.is_relative() {
                config_path
                    .and_then(Path::parent)
                    .unwrap_or_else(|| Path::new("."))
                    .join(path)
            } else {
                path.clone()
            };
            if !paths.iter().any(|item| item == &resolved) {
                paths.push(resolved);
            }
        }
        return Ok(paths);
    }

    let mut roots = vec![std::env::current_dir()?];
    if let Some(config_dir) = config_path.and_then(Path::parent) {
        if roots.iter().all(|root| root != config_dir) {
            roots.push(config_dir.to_path_buf());
        }
    }
    let persona_dir = resolve_persona_directory_from_section(persona);
    if roots.iter().all(|root| root != &persona_dir) {
        roots.push(persona_dir);
    }

    for root in roots {
        for name in ["soul.md", "agents.md", "AGENTS.md", "tools.md", "skills.md"] {
            let candidate = root.join(name);
            if !paths.iter().any(|path| path == &candidate) {
                paths.push(candidate);
            }
        }
    }
    Ok(paths)
}

fn resolve_persona_directory(config: &AppConfig) -> PathBuf {
    resolve_persona_directory_from_section(&config.persona)
}

fn resolve_persona_directory_from_section(persona: &PersonaSection) -> PathBuf {
    persona
        .directory
        .clone()
        .unwrap_or_else(default_persona_dir)
}

fn ensure_persona_files(persona_dir: &Path) -> Result<()> {
    fs::create_dir_all(persona_dir)
        .with_context(|| format!("create persona directory {}", persona_dir.display()))?;
    for name in ["soul.md", "agents.md", "tools.md", "skills.md"] {
        let path = persona_dir.join(name);
        if !path.exists() {
            fs::write(&path, "")
                .with_context(|| format!("create persona file {}", path.display()))?;
        }
    }
    Ok(())
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
    let persona_dir = resolve_persona_directory(&config);
    ensure_persona_files(&persona_dir)?;
    ensure_path_list_contains(&mut config.policy.allowed_workdirs, &persona_dir);
    ensure_path_list_contains(&mut config.policy.writable_workdirs, &persona_dir);
    config.persona.directory = Some(persona_dir.clone());

    print_brand_banner();
    println!("hajimi onboard");
    println!("config path: {}", config_path.display());
    println!("persona dir: {}", persona_dir.display());

    println!();
    config.channel.kind = prompt_default(
        "Primary channel (telegram/feishu/skip)",
        &config.channel.kind,
    )?;
    match config.channel.kind.trim().to_ascii_lowercase().as_str() {
        "telegram" => {
            config.channel.kind = "telegram".into();
            println!("Telegram bot channel");
            if config.telegram.bot_token.trim().is_empty()
                || config.telegram.bot_token.contains("replace-me")
            {
                config.telegram.bot_token = prompt("Telegram bot token")?;
            }
            let bot_identity = verify_telegram_bot_token(&config.telegram.bot_token).await?;
            println!(
                "verified Telegram bot: {} ({})",
                bot_identity
                    .username
                    .as_deref()
                    .map(|username| format!("@{username}"))
                    .unwrap_or_else(|| bot_identity.first_name.clone()),
                bot_identity.id
            );

            let needs_pairing =
                config.policy.admin_user_id == 0 || config.policy.admin_chat_id == 0;
            if prompt_yes_no("Pair Telegram admin automatically now", needs_pairing)? {
                let pairing = pair_telegram_admin(
                    &config.telegram.bot_token,
                    bot_identity.username.as_deref(),
                )
                .await?;
                config.policy.admin_user_id = pairing.user_id;
                config.policy.admin_chat_id = pairing.chat_id;
                println!(
                    "paired Telegram admin: user_id={} chat_id={}",
                    pairing.user_id, pairing.chat_id
                );
            } else {
                if config.policy.admin_user_id == 0 {
                    config.policy.admin_user_id = prompt("Telegram admin user id")?.parse()?;
                }
                if config.policy.admin_chat_id == 0 {
                    config.policy.admin_chat_id = prompt("Telegram admin chat id")?.parse()?;
                }
            }
        }
        "feishu" => {
            config.channel.kind = "feishu".into();
            println!("Feishu bot channel");
            if config.feishu.app_id.trim().is_empty() {
                config.feishu.app_id = prompt("Feishu app_id")?;
            }
            if config.feishu.app_secret.trim().is_empty() {
                config.feishu.app_secret = prompt("Feishu app_secret")?;
            }
            config.feishu.mode = Some(prompt_default(
                "Feishu mode (webhook/long-connection)",
                config.feishu.mode.as_deref().unwrap_or("webhook"),
            )?);
            config.feishu.listen_addr = Some(prompt_default(
                "Feishu listen address",
                config
                    .feishu
                    .listen_addr
                    .as_deref()
                    .unwrap_or("0.0.0.0:8787"),
            )?);
            config.feishu.event_path = Some(normalize_event_path(&prompt_default(
                "Feishu event path",
                config
                    .feishu
                    .event_path
                    .as_deref()
                    .unwrap_or("/feishu/events"),
            )?));
            config.feishu.card_callback_path = Some(normalize_event_path(&prompt_default(
                "Feishu card callback path",
                config
                    .feishu
                    .card_callback_path
                    .as_deref()
                    .unwrap_or("/feishu/card"),
            )?));
            let token_info =
                verify_feishu_app(&config.feishu.app_id, &config.feishu.app_secret).await?;
            println!(
                "verified Feishu app credentials: tenant_access_token ok (expires_in={}s)",
                token_info.expire
            );

            let admin_open_id =
                prompt_optional("Feishu admin open_id (optional, blank = allow any sender)")?;
            let admin_chat_id =
                prompt_optional("Feishu admin chat_id (optional, blank = allow any chat)")?;
            config.policy.admin_user_id = admin_open_id
                .as_deref()
                .map(stable_channel_hash)
                .unwrap_or(0);
            config.policy.admin_chat_id = admin_chat_id
                .as_deref()
                .map(stable_channel_hash)
                .unwrap_or(0);
            println!(
                "Feishu {} ready on {}{}",
                config.feishu.mode.as_deref().unwrap_or("webhook"),
                config
                    .feishu
                    .listen_addr
                    .as_deref()
                    .unwrap_or("0.0.0.0:8787"),
                config
                    .feishu
                    .event_path
                    .as_deref()
                    .unwrap_or("/feishu/events")
            );
            println!(
                "Feishu card callback path: {}",
                config
                    .feishu
                    .card_callback_path
                    .as_deref()
                    .unwrap_or("/feishu/card")
            );
        }
        "skip" | "none" => {
            config.channel.kind = "none".into();
            println!("channel setup skipped for now");
        }
        other => {
            anyhow::bail!("channel must be `telegram`, `feishu`, or `skip`, got `{other}`")
        }
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
    let model = prompt_provider_model(&ProviderConfig {
        id: slugify(&label),
        label: label.clone(),
        kind: kind.clone(),
        base_url: base_url.clone(),
        api_key: api_key.clone(),
        model: String::new(),
        fallback_models: vec![],
        enabled: true,
        extra_headers: vec![],
        created_at: Utc::now(),
    })
    .await?;
    let fallback_models = prompt_optional("Fallback model ids (comma-separated, optional)")?
        .map(|raw| parse_csv_items(&raw))
        .unwrap_or_default();

    let record = ProviderRecord {
        config: ProviderConfig {
            id: slugify(&label),
            label,
            kind,
            base_url,
            api_key,
            model,
            fallback_models,
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
    if !record.config.fallback_models.is_empty() {
        println!(
            "Fallback models: {}",
            record.config.fallback_models.join(", ")
        );
    }
    println!("Config saved to {}", config_path.display());
    println!("Master key file: {}", master_key_file.display());
    println!("Persona files ready in {}", persona_dir.display());
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

async fn cli_providers() -> Result<()> {
    let loaded = load_config_from_env_or_default()?;
    let store = open_store(&loaded.config)?;
    bootstrap_provider_if_configured(&store, &loaded.config)?;
    let providers = store.list_providers()?;
    if providers.is_empty() {
        println!("no providers configured");
        return Ok(());
    }
    for provider in providers {
        println!(
            "{}\t{}\tmodel={}\tdefault={}",
            provider.config.id, provider.config.label, provider.config.model, provider.is_default
        );
    }
    Ok(())
}

async fn cli_provider_command(args: &[String]) -> Result<()> {
    match args.first().map(String::as_str) {
        Some("current") => cli_provider_current().await,
        Some("use") => {
            let provider_id = args
                .get(1)
                .context("usage: hajimi provider use <provider-id>")?;
            cli_provider_use(provider_id).await
        }
        Some("set-model") => {
            let provider_id = args
                .get(1)
                .context("usage: hajimi provider set-model <provider-id> <model>")?;
            let model = args
                .get(2)
                .context("usage: hajimi provider set-model <provider-id> <model>")?;
            cli_provider_set_model(provider_id, model).await
        }
        Some("models") => {
            let provider_id = args.get(1).cloned();
            cli_models(provider_id).await
        }
        _ => {
            println!("usage:");
            println!("  hajimi providers");
            println!("  hajimi provider current");
            println!("  hajimi provider use <provider-id>");
            println!("  hajimi provider models [provider-id]");
            println!("  hajimi provider set-model <provider-id> <model>");
            Ok(())
        }
    }
}

async fn cli_model_command(args: &[String]) -> Result<()> {
    match args.first().map(String::as_str) {
        Some("current") => cli_model_current().await,
        Some("use") => {
            let model = args.get(1).context("usage: hajimi model use <model>")?;
            cli_model_use(model).await
        }
        _ => {
            println!("usage:");
            println!("  hajimi model current");
            println!("  hajimi model use <model>");
            Ok(())
        }
    }
}

async fn cli_provider_current() -> Result<()> {
    let loaded = load_config_from_env_or_default()?;
    let store = open_store(&loaded.config)?;
    bootstrap_provider_if_configured(&store, &loaded.config)?;
    match store.get_default_provider()? {
        Some(provider) => {
            println!(
                "provider={}\nlabel={}\nmodel={}",
                provider.config.id, provider.config.label, provider.config.model
            );
        }
        None => println!("no default provider configured"),
    }
    Ok(())
}

async fn cli_provider_use(provider_id: &str) -> Result<()> {
    let loaded = load_config_from_env_or_default()?;
    let store = open_store(&loaded.config)?;
    bootstrap_provider_if_configured(&store, &loaded.config)?;
    if store.get_provider(provider_id)?.is_none() {
        anyhow::bail!("provider not found: {provider_id}");
    }
    store.set_default_provider(provider_id)?;
    println!("default provider set to `{provider_id}`");
    Ok(())
}

async fn cli_provider_set_model(provider_id: &str, model: &str) -> Result<()> {
    let loaded = load_config_from_env_or_default()?;
    let store = open_store(&loaded.config)?;
    bootstrap_provider_if_configured(&store, &loaded.config)?;
    if store.get_provider(provider_id)?.is_none() {
        anyhow::bail!("provider not found: {provider_id}");
    }
    store.update_provider_model(provider_id, model)?;
    println!("provider `{provider_id}` model set to `{model}`");
    Ok(())
}

async fn cli_model_current() -> Result<()> {
    let loaded = load_config_from_env_or_default()?;
    let store = open_store(&loaded.config)?;
    bootstrap_provider_if_configured(&store, &loaded.config)?;
    match store.get_default_provider()? {
        Some(provider) => println!("{}", provider.config.model),
        None => println!("no default provider configured"),
    }
    Ok(())
}

async fn cli_model_use(model: &str) -> Result<()> {
    let loaded = load_config_from_env_or_default()?;
    let store = open_store(&loaded.config)?;
    bootstrap_provider_if_configured(&store, &loaded.config)?;
    let provider = store
        .get_default_provider()?
        .context("no default provider configured")?;
    store.update_provider_model(&provider.config.id, model)?;
    println!(
        "provider `{}` now uses model `{}`",
        provider.config.id, model
    );
    Ok(())
}

fn cli_launch() -> Result<()> {
    let loaded = load_config_from_env_or_default()?;
    let pid_path = pid_file_path(&loaded.path);
    if let Some(pid) = read_pid_file(&pid_path)? {
        if is_process_running(pid) {
            anyhow::bail!("hajimi is already running in background with pid {}", pid);
        }
        let _ = fs::remove_file(&pid_path);
    }

    let log_path = log_file_path(&loaded.path);
    if let Some(parent) = log_path.parent() {
        fs::create_dir_all(parent)?;
    }
    let stdout = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_path)
        .with_context(|| format!("open log file {}", log_path.display()))?;
    let stderr = stdout
        .try_clone()
        .with_context(|| format!("clone log handle for {}", log_path.display()))?;

    let mut command = Command::new(std::env::current_exe()?);
    command
        .arg("daemon")
        .env("HAJIMI_CLAW_CONFIG", &loaded.path)
        .stdin(Stdio::null())
        .stdout(Stdio::from(stdout))
        .stderr(Stdio::from(stderr));

    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const DETACHED_PROCESS: u32 = 0x0000_0008;
        const CREATE_NEW_PROCESS_GROUP: u32 = 0x0000_0200;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        command.creation_flags(DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW);
    }

    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        command.process_group(0);
    }

    let child = command.spawn().context("spawn background daemon")?;
    write_pid_file(&pid_path, child.id())?;
    println!(
        "hajimi launched in background.\npid={}\nlog={}",
        child.id(),
        log_path.display()
    );
    Ok(())
}

fn cli_stop() -> Result<()> {
    let config_path = resolve_or_default_config_path()?;
    let pid_path = pid_file_path(&config_path);
    let Some(pid) = read_pid_file(&pid_path)? else {
        println!("hajimi is not running in background");
        return Ok(());
    };

    if !is_process_running(pid) {
        let _ = fs::remove_file(&pid_path);
        println!("removed stale pid file");
        return Ok(());
    }

    stop_process(pid)?;
    let _ = fs::remove_file(&pid_path);
    println!("stopped hajimi background process pid={pid}");
    Ok(())
}

fn cli_status() -> Result<()> {
    let config_path = resolve_or_default_config_path()?;
    let pid_path = pid_file_path(&config_path);
    let log_path = log_file_path(&config_path);
    match read_pid_file(&pid_path)? {
        Some(pid) if is_process_running(pid) => {
            println!(
                "background_status=running\npid={}\nlog={}",
                pid,
                log_path.display()
            );
        }
        Some(pid) => {
            println!(
                "background_status=stale\npid={}\nlog={}\nRemove with `hajimi stop`.",
                pid,
                log_path.display()
            );
        }
        None => {
            println!("background_status=stopped\nlog={}", log_path.display());
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

fn pid_file_path(config_path: &Path) -> PathBuf {
    config_path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join("hajimi.pid")
}

fn log_file_path(config_path: &Path) -> PathBuf {
    config_path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join("hajimi.log")
}

fn read_pid_file(path: &Path) -> Result<Option<u32>> {
    if !path.exists() {
        return Ok(None);
    }
    let raw =
        fs::read_to_string(path).with_context(|| format!("read pid file {}", path.display()))?;
    let pid = raw
        .trim()
        .parse::<u32>()
        .with_context(|| format!("parse pid from {}", path.display()))?;
    Ok(Some(pid))
}

fn write_pid_file(path: &Path, pid: u32) -> Result<()> {
    fs::write(path, format!("{pid}\n"))
        .with_context(|| format!("write pid file {}", path.display()))
}

fn is_process_running(pid: u32) -> bool {
    #[cfg(windows)]
    {
        Command::new("tasklist")
            .args(["/FI", &format!("PID eq {pid}"), "/FO", "CSV", "/NH"])
            .output()
            .map(|output| {
                output.status.success()
                    && String::from_utf8_lossy(&output.stdout)
                        .to_ascii_lowercase()
                        .contains(&format!("\"{pid}\""))
            })
            .unwrap_or(false)
    }

    #[cfg(not(windows))]
    {
        Command::new("kill")
            .args(["-0", &pid.to_string()])
            .status()
            .map(|status| status.success())
            .unwrap_or(false)
    }
}

fn stop_process(pid: u32) -> Result<()> {
    #[cfg(windows)]
    {
        let status = Command::new("taskkill")
            .args(["/PID", &pid.to_string(), "/T", "/F"])
            .status()
            .context("invoke taskkill")?;
        if !status.success() {
            anyhow::bail!("failed to stop process {}", pid);
        }
    }

    #[cfg(not(windows))]
    {
        let status = Command::new("kill")
            .arg(pid.to_string())
            .status()
            .context("invoke kill")?;
        if !status.success() {
            anyhow::bail!("failed to stop process {}", pid);
        }
    }

    Ok(())
}

fn default_app_config(config_path: &Path) -> AppConfig {
    let storage = default_storage_path(config_path);
    let workdirs = default_workdirs(config_path);
    AppConfig {
        channel: ChannelSection::default(),
        telegram: TelegramSection {
            bot_token: String::new(),
            poll_timeout_secs: Some(30),
        },
        feishu: FeishuSection {
            app_id: String::new(),
            app_secret: String::new(),
            listen_addr: Some("0.0.0.0:8787".into()),
            event_path: Some("/feishu/events".into()),
            card_callback_path: Some("/feishu/card".into()),
            mode: Some("webhook".into()),
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
            writable_workdirs: vec![
                workdirs[1].clone(),
                workdirs[2].clone(),
                workdirs[3].clone(),
            ],
            ..PolicyConfig::default()
        },
        execution: ExecutionSection {
            mode: Some("auto".into()),
        },
        multi_agent: MultiAgentSection::default(),
        persona: PersonaSection {
            directory: Some(default_persona_dir()),
            prompt_files: vec![],
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
    if let Some(path) = config.persona.directory.as_mut() {
        *path = make_relative(path, base);
    }
    config.persona.prompt_files = config
        .persona
        .prompt_files
        .iter()
        .map(|path| make_relative(path, base))
        .collect();
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

fn ensure_path_list_contains(paths: &mut Vec<PathBuf>, target: &Path) {
    if paths.iter().all(|path| path != target) {
        paths.push(target.to_path_buf());
    }
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

fn prompt_optional(label: &str) -> Result<Option<String>> {
    print!("{label}: ");
    io::stdout().flush()?;
    let mut buffer = String::new();
    io::stdin().read_line(&mut buffer)?;
    let trimmed = buffer.trim().to_string();
    if trimmed.is_empty() {
        Ok(None)
    } else {
        Ok(Some(trimmed))
    }
}

fn parse_csv_items(raw: &str) -> Vec<String> {
    raw.split(',')
        .map(str::trim)
        .filter(|item| !item.is_empty())
        .map(ToString::to_string)
        .collect()
}

async fn prompt_provider_model(provider: &ProviderConfig) -> Result<String> {
    let client = reqwest::Client::new();
    match list_models(&client, provider).await {
        Ok(models) if !models.is_empty() => {
            println!("Discovered models:");
            for (index, model) in models.iter().enumerate() {
                println!("  {}. {}", index + 1, model);
            }
            let default = "1";
            let choice =
                prompt_default("Choose a model number or type a custom model id", default)?;
            if let Some(model) = resolve_model_choice(&choice, &models) {
                println!("selected model: {model}");
                Ok(model)
            } else {
                anyhow::bail!("invalid model selection")
            }
        }
        Ok(_) => prompt_default("Default model", "gpt-4.1-mini"),
        Err(err) => {
            println!("Model discovery failed: {err}");
            prompt_default("Default model", "gpt-4.1-mini")
        }
    }
}

fn resolve_model_choice(choice: &str, models: &[String]) -> Option<String> {
    let trimmed = choice.trim();
    if trimmed.is_empty() {
        return None;
    }
    if let Ok(index) = trimmed.parse::<usize>() {
        return models.get(index.checked_sub(1)?).cloned();
    }
    Some(trimmed.to_string())
}

fn prompt_yes_no(label: &str, default: bool) -> Result<bool> {
    let suffix = if default { "[Y/n]" } else { "[y/N]" };
    print!("{label} {suffix}: ");
    io::stdout().flush()?;
    let mut buffer = String::new();
    io::stdin().read_line(&mut buffer)?;
    let trimmed = buffer.trim().to_ascii_lowercase();
    if trimmed.is_empty() {
        return Ok(default);
    }
    match trimmed.as_str() {
        "y" | "yes" => Ok(true),
        "n" | "no" => Ok(false),
        _ => anyhow::bail!("please answer yes or no"),
    }
}

fn normalize_event_path(raw: &str) -> String {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return "/feishu/events".into();
    }
    if trimmed.starts_with('/') {
        trimmed.to_string()
    } else {
        format!("/{trimmed}")
    }
}

async fn verify_telegram_bot_token(token: &str) -> Result<TelegramBotIdentity> {
    let client = reqwest::Client::new();
    let response = client
        .post(telegram_api_url(token, "getMe"))
        .send()
        .await
        .context("telegram getMe request")?;
    if !response.status().is_success() {
        anyhow::bail!("telegram getMe returned {}", response.status());
    }
    let payload: TelegramEnvelope<TelegramUser> = response
        .json()
        .await
        .context("decode telegram getMe response")?;
    if !payload.ok {
        anyhow::bail!("telegram getMe returned ok=false");
    }
    Ok(TelegramBotIdentity {
        id: payload.result.id,
        username: payload.result.username,
        first_name: payload.result.first_name,
    })
}

async fn verify_feishu_app(app_id: &str, app_secret: &str) -> Result<FeishuTokenResponse> {
    let client = reqwest::Client::new();
    let response = client
        .post("https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal")
        .json(&serde_json::json!({
            "app_id": app_id,
            "app_secret": app_secret,
        }))
        .send()
        .await
        .context("Feishu tenant_access_token request")?;
    if !response.status().is_success() {
        anyhow::bail!("Feishu tenant_access_token returned {}", response.status());
    }
    let payload: FeishuTokenResponse = response
        .json()
        .await
        .context("decode Feishu tenant_access_token response")?;
    if payload.code != 0 {
        anyhow::bail!(
            "Feishu tenant_access_token failed: {}",
            payload.msg.as_deref().unwrap_or("unknown error")
        );
    }
    if payload
        .tenant_access_token
        .as_deref()
        .unwrap_or("")
        .is_empty()
    {
        anyhow::bail!("Feishu tenant_access_token response did not include a token");
    }
    Ok(payload)
}

async fn pair_telegram_admin(token: &str, username: Option<&str>) -> Result<TelegramPairing> {
    let client = reqwest::Client::new();
    let mut offset = latest_update_offset(&client, token).await?;
    let raw_code = Uuid::new_v4().simple().to_string();
    let code = format!("hajimi-{}", &raw_code[..8]);
    let target = username
        .map(|value| format!("@{value}"))
        .unwrap_or_else(|| "your bot".into());
    println!(
        "pairing code generated. Send `/start {code}` or `/pair {code}` to {target} within 90 seconds."
    );

    let deadline = Instant::now() + Duration::from_secs(90);
    while Instant::now() < deadline {
        let updates = telegram_get_updates(&client, token, offset, 10).await?;
        for update in updates {
            offset = offset.max(update.update_id + 1);
            if let Some(pairing) = match_pairing_update(&update, &code) {
                return Ok(pairing);
            }
        }
    }

    anyhow::bail!("telegram pairing timed out after 90 seconds")
}

async fn latest_update_offset(client: &reqwest::Client, token: &str) -> Result<i64> {
    let updates = telegram_get_updates(client, token, 0, 0).await?;
    Ok(updates
        .into_iter()
        .map(|update| update.update_id + 1)
        .max()
        .unwrap_or(0))
}

async fn telegram_get_updates(
    client: &reqwest::Client,
    token: &str,
    offset: i64,
    timeout_secs: u64,
) -> Result<Vec<TelegramUpdate>> {
    let response = client
        .post(telegram_api_url(token, "getUpdates"))
        .json(&serde_json::json!({
            "offset": offset,
            "timeout": timeout_secs,
        }))
        .send()
        .await
        .context("telegram getUpdates request")?;
    if !response.status().is_success() {
        anyhow::bail!("telegram getUpdates returned {}", response.status());
    }
    let payload: TelegramEnvelope<Vec<TelegramUpdate>> = response
        .json()
        .await
        .context("decode telegram getUpdates response")?;
    if !payload.ok {
        anyhow::bail!("telegram getUpdates returned ok=false");
    }
    Ok(payload.result)
}

fn match_pairing_update(update: &TelegramUpdate, code: &str) -> Option<TelegramPairing> {
    let message = update.message.as_ref()?;
    let text = message.text.as_deref()?.trim();
    if !pairing_text_matches(text, code) {
        return None;
    }
    let user_id = message.from.as_ref()?.id;
    Some(TelegramPairing {
        user_id,
        chat_id: message.chat.id,
    })
}

fn pairing_text_matches(text: &str, code: &str) -> bool {
    let normalized = text.trim();
    normalized == code
        || normalized == format!("/pair {code}")
        || normalized == format!("/start {code}")
}

fn telegram_api_url(token: &str, method: &str) -> String {
    format!("https://api.telegram.org/bot{token}/{method}")
}

fn stable_channel_hash(value: &str) -> i64 {
    use std::hash::{Hash, Hasher};

    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    value.hash(&mut hasher);
    (hasher.finish() & (i64::MAX as u64)) as i64
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

fn print_brand_banner() {
    println!(r"  /\_/\\      hajimi");
    println!(r" ( o.o )     cat x ferris");
    println!(r"  > ^ <   _  _");
    println!(r"         ( \/ )");
    println!(r"          \  /");
    println!("         /_/\\\\_\\\\");
    println!();
}

fn print_help() {
    print_brand_banner();
    println!("{}", help_text());
}

fn help_text() -> &'static str {
    "Usage:
  hajimi                 Run the daemon
  hajimi daemon          Run the daemon
  hajimi launch          Launch the daemon in background
  hajimi stop            Stop the background daemon
  hajimi status          Show background daemon status
  hajimi onboard         Interactive local onboarding (telegram/feishu/skip)
  hajimi providers       List configured providers
  hajimi provider ...    Manage providers
  hajimi model ...       Manage the active model
  hajimi models [id]     List models for the default or named provider
  hajimi launch          Start the configured channel and multi-agent runtime in background
  hajimi restart         Restart the installed service
  hajimi help            Show this help"
}

#[derive(Debug, Deserialize)]
struct TelegramEnvelope<T> {
    ok: bool,
    result: T,
}

#[derive(Debug, Deserialize)]
struct FeishuTokenResponse {
    code: i64,
    msg: Option<String>,
    tenant_access_token: Option<String>,
    expire: i64,
}

#[derive(Debug, Deserialize)]
struct TelegramUpdate {
    update_id: i64,
    message: Option<TelegramMessage>,
}

#[derive(Debug, Deserialize)]
struct TelegramMessage {
    chat: TelegramChat,
    from: Option<TelegramUser>,
    text: Option<String>,
}

#[derive(Debug, Deserialize)]
struct TelegramChat {
    id: i64,
}

#[derive(Debug, Deserialize)]
struct TelegramUser {
    id: i64,
    username: Option<String>,
    first_name: String,
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::Path;

    use super::{
        PersonaSection, default_config_path, ensure_persona_files, expand_home, log_file_path,
        open_store, pairing_text_matches, pid_file_path, resolve_model_choice,
        resolve_persona_paths, select_platform_mode, slugify,
    };
    use hajimi_claw_exec::PlatformMode;
    use tempfile::tempdir;

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

    #[test]
    fn pairing_text_accepts_start_and_pair_commands() {
        assert!(pairing_text_matches(
            "/start hajimi-abcd1234",
            "hajimi-abcd1234"
        ));
        assert!(pairing_text_matches(
            "/pair hajimi-abcd1234",
            "hajimi-abcd1234"
        ));
        assert!(!pairing_text_matches("/start other", "hajimi-abcd1234"));
    }

    #[test]
    fn persona_paths_include_local_and_home_files() {
        let dir = tempdir().unwrap();
        let config_path = dir.path().join("config.toml");
        let persona_dir = dir.path().join(".hajimi");
        fs::write(dir.path().join("soul.md"), "Speak tersely.").unwrap();
        ensure_persona_files(&persona_dir).unwrap();

        let paths = resolve_persona_paths(
            Some(&config_path),
            &PersonaSection {
                directory: Some(persona_dir.clone()),
                prompt_files: vec![],
            },
        )
        .unwrap();
        assert!(paths.contains(&dir.path().join("soul.md")));
        assert!(paths.contains(&persona_dir.join("skills.md")));
    }

    #[test]
    fn expands_tilde_paths() {
        let expanded = expand_home(Path::new("~/.hajimi"));
        assert!(expanded.ends_with(".hajimi"));
    }

    #[test]
    fn open_store_creates_storage_parent_dir() {
        let dir = tempdir().unwrap();
        let key_path = dir.path().join("master.key");
        fs::write(&key_path, "test-master-key").unwrap();

        let config = super::AppConfig {
            channel: super::ChannelSection::default(),
            telegram: super::TelegramSection {
                bot_token: "token".into(),
                poll_timeout_secs: Some(30),
            },
            feishu: super::FeishuSection::default(),
            llm: super::LlmSection {
                base_url: None,
                api_key: None,
                model: None,
                static_fallback_response: None,
            },
            storage: super::StorageSection {
                sqlite_path: dir
                    .path()
                    .join("nested")
                    .join("data")
                    .join("hajimi.sqlite3"),
            },
            security: super::SecuritySection {
                master_key_env: Some("HAJIMI_TEST_MASTER_KEY".into()),
                master_key_file: Some(key_path),
            },
            policy: hajimi_claw_policy::PolicyConfig::default(),
            execution: super::ExecutionSection { mode: None },
            multi_agent: super::MultiAgentSection::default(),
            persona: super::PersonaSection::default(),
        };

        open_store(&config).unwrap();
        assert!(config.storage.sqlite_path.exists());
    }

    #[test]
    fn resolves_model_choice_by_index_or_literal() {
        let models = vec!["gpt-4.1-mini".to_string(), "gpt-4.1".to_string()];
        assert_eq!(
            resolve_model_choice("1", &models).as_deref(),
            Some("gpt-4.1-mini")
        );
        assert_eq!(
            resolve_model_choice("custom-model", &models).as_deref(),
            Some("custom-model")
        );
        assert!(resolve_model_choice("0", &models).is_none());
    }

    #[test]
    fn background_paths_use_config_dir() {
        let dir = tempdir().unwrap();
        let config_path = dir.path().join("config.toml");
        assert_eq!(pid_file_path(&config_path), dir.path().join("hajimi.pid"));
        assert_eq!(log_file_path(&config_path), dir.path().join("hajimi.log"));
    }
}
