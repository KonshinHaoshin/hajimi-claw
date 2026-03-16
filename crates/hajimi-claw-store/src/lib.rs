use std::path::Path;
use std::sync::{Arc, Mutex};

use aes_gcm::aead::{Aead, KeyInit, OsRng, rand_core::RngCore};
use aes_gcm::{Aes256Gcm, Nonce};
use anyhow::{Context, Result};
use base64::Engine;
use base64::engine::general_purpose::STANDARD as BASE64;
use chrono::{DateTime, Utc};
use hajimi_claw_types::{
    ApprovalRequest, ConversationId, ConversationMessage, OnboardingSession, ProviderConfig,
    ProviderDraft, ProviderKind, ProviderRecord, SessionHandle, SessionSummary, TaskId, TaskKind,
    TaskStatus,
};
use rusqlite::{Connection, OptionalExtension, params};
use sha2::{Digest, Sha256};

pub struct Store {
    connection: Mutex<Connection>,
    cipher: Option<Arc<SecretCipher>>,
}

#[derive(Clone)]
pub struct SecretCipher {
    key: [u8; 32],
}

impl Store {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        Self::open_with_cipher(path, None)
    }

    pub fn open_with_cipher(
        path: impl AsRef<Path>,
        cipher: Option<Arc<SecretCipher>>,
    ) -> Result<Self> {
        let connection = Connection::open(path).context("open sqlite database")?;
        connection.pragma_update(None, "journal_mode", "WAL")?;
        Self::from_connection(connection, cipher)
    }

    pub fn open_in_memory() -> Result<Self> {
        Self::open_in_memory_with_cipher(None)
    }

    pub fn open_in_memory_with_cipher(cipher: Option<Arc<SecretCipher>>) -> Result<Self> {
        let connection = Connection::open_in_memory().context("open sqlite memory database")?;
        connection.pragma_update(None, "journal_mode", "WAL")?;
        Self::from_connection(connection, cipher)
    }

    fn from_connection(connection: Connection, cipher: Option<Arc<SecretCipher>>) -> Result<Self> {
        let store = Self {
            connection: Mutex::new(connection),
            cipher,
        };
        store.migrate()?;
        Ok(store)
    }

    fn migrate(&self) -> Result<()> {
        let connection = self.connection.lock().expect("store lock poisoned");
        connection.execute_batch(
            r#"
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS tasks (
                id TEXT PRIMARY KEY,
                kind TEXT NOT NULL,
                description TEXT NOT NULL,
                queued_at TEXT NOT NULL,
                started_at TEXT,
                finished_at TEXT,
                running INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS shell_sessions (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                cwd TEXT NOT NULL,
                created_at TEXT NOT NULL,
                last_used_at TEXT NOT NULL,
                active INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS approvals (
                id TEXT PRIMARY KEY,
                reason TEXT NOT NULL,
                risk_level TEXT NOT NULL,
                command_preview TEXT NOT NULL,
                cwd TEXT,
                expires_at TEXT NOT NULL,
                approved INTEGER
            );

            CREATE TABLE IF NOT EXISTS command_audit (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id TEXT,
                session_id TEXT,
                command_preview TEXT NOT NULL,
                exit_code INTEGER,
                duration_ms INTEGER NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS conversation_summaries (
                conversation_id TEXT PRIMARY KEY,
                summary_json TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS config_kv (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS providers (
                id TEXT PRIMARY KEY,
                label TEXT NOT NULL,
                kind TEXT NOT NULL,
                base_url TEXT NOT NULL,
                api_key TEXT NOT NULL,
                model TEXT NOT NULL,
                enabled INTEGER NOT NULL,
                extra_headers_json TEXT NOT NULL,
                is_default INTEGER NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS onboarding_sessions (
                chat_id INTEGER NOT NULL,
                user_id INTEGER NOT NULL,
                step TEXT NOT NULL,
                draft_json TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY(chat_id, user_id)
            );

            CREATE TABLE IF NOT EXISTS chat_provider_bindings (
                chat_id INTEGER PRIMARY KEY,
                provider_id TEXT NOT NULL
            );
            "#,
        )?;
        Ok(())
    }

    pub fn save_message(
        &self,
        conversation_id: ConversationId,
        message: &ConversationMessage,
    ) -> Result<()> {
        let connection = self.connection.lock().expect("store lock poisoned");
        connection.execute(
            "INSERT INTO messages (conversation_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            params![
                conversation_id.to_string(),
                format!("{:?}", message.role),
                message.content,
                message.created_at.to_rfc3339()
            ],
        )?;
        Ok(())
    }

    pub fn upsert_task(&self, task: &TaskStatus) -> Result<()> {
        let connection = self.connection.lock().expect("store lock poisoned");
        connection.execute(
            r#"
            INSERT INTO tasks (id, kind, description, queued_at, started_at, finished_at, running)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                kind=excluded.kind,
                description=excluded.description,
                queued_at=excluded.queued_at,
                started_at=excluded.started_at,
                finished_at=excluded.finished_at,
                running=excluded.running
            "#,
            params![
                task.id.to_string(),
                format!("{:?}", task.kind),
                task.description,
                task.queued_at.to_rfc3339(),
                task.started_at.map(|ts| ts.to_rfc3339()),
                task.finished_at.map(|ts| ts.to_rfc3339()),
                i64::from(task.running),
            ],
        )?;
        Ok(())
    }

    pub fn list_tasks(&self) -> Result<Vec<TaskStatus>> {
        let connection = self.connection.lock().expect("store lock poisoned");
        let mut stmt = connection.prepare(
            "SELECT id, kind, description, queued_at, started_at, finished_at, running FROM tasks ORDER BY queued_at DESC",
        )?;
        let rows = stmt.query_map([], |row| {
            Ok(TaskStatus {
                id: TaskId(uuid::Uuid::parse_str(row.get::<_, String>(0)?.as_str()).unwrap()),
                kind: match row.get::<_, String>(1)?.as_str() {
                    "PersistentShellTask" => TaskKind::PersistentShellTask,
                    _ => TaskKind::EphemeralAgentTask,
                },
                description: row.get(2)?,
                queued_at: parse_ts(row.get::<_, String>(3)?),
                started_at: row.get::<_, Option<String>>(4)?.map(parse_ts),
                finished_at: row.get::<_, Option<String>>(5)?.map(parse_ts),
                running: row.get::<_, i64>(6)? != 0,
            })
        })?;

        rows.collect::<rusqlite::Result<Vec<_>>>()
            .map_err(Into::into)
    }

    pub fn upsert_session(&self, session: &SessionHandle, active: bool) -> Result<()> {
        let connection = self.connection.lock().expect("store lock poisoned");
        connection.execute(
            r#"
            INSERT INTO shell_sessions (id, name, cwd, created_at, last_used_at, active)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                name=excluded.name,
                cwd=excluded.cwd,
                created_at=excluded.created_at,
                last_used_at=excluded.last_used_at,
                active=excluded.active
            "#,
            params![
                session.id.to_string(),
                session.name,
                session.cwd.display().to_string(),
                session.created_at.to_rfc3339(),
                session.last_used_at.to_rfc3339(),
                i64::from(active),
            ],
        )?;
        Ok(())
    }

    pub fn save_approval(&self, approval: &ApprovalRequest, approved: Option<bool>) -> Result<()> {
        let connection = self.connection.lock().expect("store lock poisoned");
        connection.execute(
            r#"
            INSERT INTO approvals (id, reason, risk_level, command_preview, cwd, expires_at, approved)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                reason=excluded.reason,
                risk_level=excluded.risk_level,
                command_preview=excluded.command_preview,
                cwd=excluded.cwd,
                expires_at=excluded.expires_at,
                approved=excluded.approved
            "#,
            params![
                approval.request_id.to_string(),
                approval.reason,
                format!("{:?}", approval.risk_level),
                approval.command_preview,
                approval.cwd.as_ref().map(|cwd| cwd.display().to_string()),
                approval.expires_at.to_rfc3339(),
                approved.map(i64::from),
            ],
        )?;
        Ok(())
    }

    pub fn get_approval_state(&self, request_id: &str) -> Result<Option<Option<bool>>> {
        let connection = self.connection.lock().expect("store lock poisoned");
        connection
            .query_row(
                "SELECT approved FROM approvals WHERE id = ?",
                params![request_id],
                |row| {
                    row.get::<_, Option<i64>>(0)
                        .map(|value| value.map(|v| v != 0))
                },
            )
            .optional()
            .map_err(Into::into)
    }

    pub fn append_command_audit(
        &self,
        task_id: Option<TaskId>,
        session_id: Option<String>,
        command_preview: &str,
        exit_code: Option<i32>,
        duration_ms: u128,
    ) -> Result<()> {
        let connection = self.connection.lock().expect("store lock poisoned");
        connection.execute(
            "INSERT INTO command_audit (task_id, session_id, command_preview, exit_code, duration_ms, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            params![
                task_id.map(|id| id.to_string()),
                session_id,
                command_preview,
                exit_code,
                duration_ms as i64,
                Utc::now().to_rfc3339(),
            ],
        )?;
        Ok(())
    }

    pub fn save_summary(&self, summary: &SessionSummary) -> Result<()> {
        let connection = self.connection.lock().expect("store lock poisoned");
        connection.execute(
            r#"
            INSERT INTO conversation_summaries (conversation_id, summary_json, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(conversation_id) DO UPDATE SET
                summary_json=excluded.summary_json,
                updated_at=excluded.updated_at
            "#,
            params![
                summary.session_id.to_string(),
                serde_json::to_string(summary)?,
                Utc::now().to_rfc3339(),
            ],
        )?;
        Ok(())
    }

    pub fn load_summary(&self, conversation_id: ConversationId) -> Result<Option<SessionSummary>> {
        let connection = self.connection.lock().expect("store lock poisoned");
        let payload = connection
            .query_row(
                "SELECT summary_json FROM conversation_summaries WHERE conversation_id = ?",
                params![conversation_id.to_string()],
                |row| row.get::<_, String>(0),
            )
            .optional()?;

        payload
            .map(|json| serde_json::from_str(&json).context("decode conversation summary"))
            .transpose()
    }

    pub fn set_config(&self, key: &str, value: &str) -> Result<()> {
        let connection = self.connection.lock().expect("store lock poisoned");
        connection.execute(
            "INSERT INTO config_kv (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            params![key, value],
        )?;
        Ok(())
    }

    pub fn get_config(&self, key: &str) -> Result<Option<String>> {
        let connection = self.connection.lock().expect("store lock poisoned");
        connection
            .query_row(
                "SELECT value FROM config_kv WHERE key = ?",
                params![key],
                |row| row.get(0),
            )
            .optional()
            .map_err(Into::into)
    }

    pub fn upsert_provider(&self, record: &ProviderRecord) -> Result<()> {
        let connection = self.connection.lock().expect("store lock poisoned");
        if record.is_default {
            connection.execute("UPDATE providers SET is_default = 0", [])?;
        }
        let encrypted_api_key = self.encrypt_secret(&record.config.api_key)?;
        connection.execute(
            r#"
            INSERT INTO providers (id, label, kind, base_url, api_key, model, enabled, extra_headers_json, is_default, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                label=excluded.label,
                kind=excluded.kind,
                base_url=excluded.base_url,
                api_key=excluded.api_key,
                model=excluded.model,
                enabled=excluded.enabled,
                extra_headers_json=excluded.extra_headers_json,
                is_default=excluded.is_default,
                created_at=excluded.created_at
            "#,
            params![
                record.config.id,
                record.config.label,
                record.config.kind.as_str(),
                record.config.base_url,
                encrypted_api_key,
                record.config.model,
                i64::from(record.config.enabled),
                serde_json::to_string(&record.config.extra_headers)?,
                i64::from(record.is_default),
                record.config.created_at.to_rfc3339(),
            ],
        )?;
        Ok(())
    }

    pub fn list_providers(&self) -> Result<Vec<ProviderRecord>> {
        let connection = self.connection.lock().expect("store lock poisoned");
        let mut stmt = connection.prepare(
            r#"
            SELECT id, label, kind, base_url, api_key, model, enabled, extra_headers_json, is_default, created_at
            FROM providers
            ORDER BY is_default DESC, created_at ASC
            "#,
        )?;
        let rows = stmt.query_map([], |row| self.row_to_provider(row))?;
        rows.collect::<rusqlite::Result<Vec<_>>>()
            .map_err(Into::into)
    }

    pub fn get_provider(&self, provider_id: &str) -> Result<Option<ProviderRecord>> {
        let connection = self.connection.lock().expect("store lock poisoned");
        connection
            .query_row(
                r#"
                SELECT id, label, kind, base_url, api_key, model, enabled, extra_headers_json, is_default, created_at
                FROM providers WHERE id = ?
                "#,
                params![provider_id],
                |row| self.row_to_provider(row),
            )
            .optional()
            .map_err(Into::into)
    }

    pub fn get_default_provider(&self) -> Result<Option<ProviderRecord>> {
        let connection = self.connection.lock().expect("store lock poisoned");
        connection
            .query_row(
                r#"
                SELECT id, label, kind, base_url, api_key, model, enabled, extra_headers_json, is_default, created_at
                FROM providers WHERE is_default = 1 LIMIT 1
                "#,
                [],
                |row| self.row_to_provider(row),
            )
            .optional()
            .map_err(Into::into)
    }

    pub fn get_first_provider(&self) -> Result<Option<ProviderRecord>> {
        let connection = self.connection.lock().expect("store lock poisoned");
        connection
            .query_row(
                r#"
                SELECT id, label, kind, base_url, api_key, model, enabled, extra_headers_json, is_default, created_at
                FROM providers
                ORDER BY created_at ASC
                LIMIT 1
                "#,
                [],
                |row| self.row_to_provider(row),
            )
            .optional()
            .map_err(Into::into)
    }

    pub fn set_default_provider(&self, provider_id: &str) -> Result<()> {
        let connection = self.connection.lock().expect("store lock poisoned");
        connection.execute("UPDATE providers SET is_default = 0", [])?;
        connection.execute(
            "UPDATE providers SET is_default = 1 WHERE id = ?",
            params![provider_id],
        )?;
        Ok(())
    }

    pub fn update_provider_model(&self, provider_id: &str, model: &str) -> Result<()> {
        let connection = self.connection.lock().expect("store lock poisoned");
        connection.execute(
            "UPDATE providers SET model = ? WHERE id = ?",
            params![model, provider_id],
        )?;
        Ok(())
    }

    pub fn bind_provider_to_chat(&self, chat_id: i64, provider_id: &str) -> Result<()> {
        let connection = self.connection.lock().expect("store lock poisoned");
        connection.execute(
            r#"
            INSERT INTO chat_provider_bindings (chat_id, provider_id)
            VALUES (?, ?)
            ON CONFLICT(chat_id) DO UPDATE SET provider_id=excluded.provider_id
            "#,
            params![chat_id, provider_id],
        )?;
        Ok(())
    }

    pub fn get_bound_provider_id(&self, chat_id: i64) -> Result<Option<String>> {
        let connection = self.connection.lock().expect("store lock poisoned");
        connection
            .query_row(
                "SELECT provider_id FROM chat_provider_bindings WHERE chat_id = ?",
                params![chat_id],
                |row| row.get(0),
            )
            .optional()
            .map_err(Into::into)
    }

    pub fn resolve_provider_for_chat(&self, chat_id: i64) -> Result<Option<ProviderRecord>> {
        if let Some(provider_id) = self.get_bound_provider_id(chat_id)? {
            if let Some(record) = self.get_provider(&provider_id)? {
                return Ok(Some(record));
            }
        }
        self.get_default_provider().and_then(|record| match record {
            Some(record) => Ok(Some(record)),
            None => self.get_first_provider(),
        })
    }

    pub fn save_onboarding_session(&self, session: &OnboardingSession) -> Result<()> {
        let connection = self.connection.lock().expect("store lock poisoned");
        connection.execute(
            r#"
            INSERT INTO onboarding_sessions (chat_id, user_id, step, draft_json, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(chat_id, user_id) DO UPDATE SET
                step=excluded.step,
                draft_json=excluded.draft_json,
                updated_at=excluded.updated_at
            "#,
            params![
                session.chat_id,
                session.user_id,
                format!("{:?}", session.step),
                serde_json::to_string(&session.draft)?,
                session.updated_at.to_rfc3339(),
            ],
        )?;
        Ok(())
    }

    pub fn load_onboarding_session(
        &self,
        chat_id: i64,
        user_id: i64,
    ) -> Result<Option<OnboardingSession>> {
        let connection = self.connection.lock().expect("store lock poisoned");
        connection
            .query_row(
                "SELECT step, draft_json, updated_at FROM onboarding_sessions WHERE chat_id = ? AND user_id = ?",
                params![chat_id, user_id],
                |row| {
                    let step = parse_onboarding_step(row.get::<_, String>(0)?);
                    let draft_json: String = row.get(1)?;
                    let draft: ProviderDraft = serde_json::from_str(&draft_json).map_err(to_sql_err)?;
                    let updated_at = parse_ts(row.get::<_, String>(2)?);
                    Ok(OnboardingSession {
                        user_id,
                        chat_id,
                        step,
                        draft,
                        updated_at,
                    })
                },
            )
            .optional()
            .map_err(Into::into)
    }

    pub fn clear_onboarding_session(&self, chat_id: i64, user_id: i64) -> Result<()> {
        let connection = self.connection.lock().expect("store lock poisoned");
        connection.execute(
            "DELETE FROM onboarding_sessions WHERE chat_id = ? AND user_id = ?",
            params![chat_id, user_id],
        )?;
        Ok(())
    }

    fn row_to_provider(&self, row: &rusqlite::Row<'_>) -> rusqlite::Result<ProviderRecord> {
        let kind = match row.get::<_, String>(2)?.as_str() {
            "custom-chat-completions" => ProviderKind::CustomChatCompletions,
            _ => ProviderKind::OpenAiCompatible,
        };
        let headers_json: String = row.get(7)?;
        let extra_headers = serde_json::from_str(&headers_json).map_err(to_sql_err)?;
        let api_key_raw: String = row.get(4)?;
        let api_key = self
            .decrypt_secret(&api_key_raw)
            .map_err(to_sql_anyhow_err)?;
        Ok(ProviderRecord {
            config: ProviderConfig {
                id: row.get(0)?,
                label: row.get(1)?,
                kind,
                base_url: row.get(3)?,
                api_key,
                model: row.get(5)?,
                enabled: row.get::<_, i64>(6)? != 0,
                extra_headers,
                created_at: parse_ts(row.get(9)?),
            },
            is_default: row.get::<_, i64>(8)? != 0,
        })
    }

    fn encrypt_secret(&self, value: &str) -> Result<String> {
        match &self.cipher {
            Some(cipher) => cipher.encrypt(value),
            None => Ok(value.to_string()),
        }
    }

    fn decrypt_secret(&self, value: &str) -> Result<String> {
        match &self.cipher {
            Some(cipher) => cipher.decrypt_or_passthrough(value),
            None => Ok(value.to_string()),
        }
    }
}

impl SecretCipher {
    pub fn from_passphrase(passphrase: &str) -> Result<Self> {
        if passphrase.trim().is_empty() {
            anyhow::bail!("master key must not be empty");
        }
        let digest = Sha256::digest(passphrase.as_bytes());
        let mut key = [0_u8; 32];
        key.copy_from_slice(&digest);
        Ok(Self { key })
    }

    pub fn encrypt(&self, plaintext: &str) -> Result<String> {
        let cipher = Aes256Gcm::new_from_slice(&self.key).context("initialize aes-256-gcm")?;
        let mut nonce_bytes = [0_u8; 12];
        OsRng.fill_bytes(&mut nonce_bytes);
        let nonce = Nonce::from_slice(&nonce_bytes);
        let ciphertext = cipher
            .encrypt(nonce, plaintext.as_bytes())
            .map_err(|_| anyhow::anyhow!("failed to encrypt secret"))?;
        let mut payload = nonce_bytes.to_vec();
        payload.extend(ciphertext);
        Ok(format!("enc:v1:{}", BASE64.encode(payload)))
    }

    pub fn decrypt_or_passthrough(&self, value: &str) -> Result<String> {
        if !value.starts_with("enc:v1:") {
            return Ok(value.to_string());
        }
        let encoded = value.trim_start_matches("enc:v1:");
        let payload = BASE64
            .decode(encoded)
            .context("decode encrypted provider secret")?;
        if payload.len() < 13 {
            anyhow::bail!("encrypted payload is too short");
        }
        let (nonce_bytes, ciphertext) = payload.split_at(12);
        let cipher = Aes256Gcm::new_from_slice(&self.key).context("initialize aes-256-gcm")?;
        let plaintext = cipher
            .decrypt(Nonce::from_slice(nonce_bytes), ciphertext)
            .map_err(|_| anyhow::anyhow!("failed to decrypt provider secret"))?;
        String::from_utf8(plaintext).context("provider secret is not valid utf-8")
    }
}

fn parse_ts(ts: String) -> DateTime<Utc> {
    DateTime::parse_from_rfc3339(&ts)
        .expect("timestamp stored in rfc3339")
        .with_timezone(&Utc)
}

fn parse_onboarding_step(step: String) -> hajimi_claw_types::OnboardingStep {
    match step.as_str() {
        "ProviderKind" => hajimi_claw_types::OnboardingStep::ProviderKind,
        "ProviderBaseUrl" => hajimi_claw_types::OnboardingStep::ProviderBaseUrl,
        "ProviderApiKey" => hajimi_claw_types::OnboardingStep::ProviderApiKey,
        "ProviderModel" => hajimi_claw_types::OnboardingStep::ProviderModel,
        "Completed" => hajimi_claw_types::OnboardingStep::Completed,
        _ => hajimi_claw_types::OnboardingStep::ProviderLabel,
    }
}

fn to_sql_err(err: serde_json::Error) -> rusqlite::Error {
    rusqlite::Error::FromSqlConversionFailure(0, rusqlite::types::Type::Text, Box::new(err))
}

fn to_sql_anyhow_err(err: anyhow::Error) -> rusqlite::Error {
    rusqlite::Error::FromSqlConversionFailure(
        0,
        rusqlite::types::Type::Text,
        Box::new(std::io::Error::other(err.to_string())),
    )
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use chrono::Utc;
    use hajimi_claw_types::{
        ConversationMessage, MessageRole, ProviderConfig, ProviderKind, ProviderRecord,
    };

    use super::{SecretCipher, Store};

    #[test]
    fn persists_message_and_task() {
        let store = Store::open_in_memory().unwrap();
        let conversation_id = hajimi_claw_types::ConversationId::new();
        store
            .save_message(
                conversation_id,
                &ConversationMessage {
                    role: MessageRole::User,
                    content: "hello".into(),
                    created_at: Utc::now(),
                },
            )
            .unwrap();

        let task = hajimi_claw_types::TaskStatus {
            id: hajimi_claw_types::TaskId::new(),
            kind: hajimi_claw_types::TaskKind::EphemeralAgentTask,
            description: "test".into(),
            queued_at: Utc::now(),
            started_at: None,
            finished_at: None,
            running: false,
        };
        store.upsert_task(&task).unwrap();
        let tasks = store.list_tasks().unwrap();
        assert_eq!(tasks.len(), 1);
    }

    #[test]
    fn encrypts_provider_api_key_when_cipher_enabled() {
        let cipher = Arc::new(SecretCipher::from_passphrase("secret").unwrap());
        let store = Store::open_in_memory_with_cipher(Some(cipher)).unwrap();
        store
            .upsert_provider(&ProviderRecord {
                config: ProviderConfig {
                    id: "demo".into(),
                    label: "Demo".into(),
                    kind: ProviderKind::OpenAiCompatible,
                    base_url: "https://example.com/v1".into(),
                    api_key: "top-secret".into(),
                    model: "gpt-demo".into(),
                    enabled: true,
                    extra_headers: vec![],
                    created_at: Utc::now(),
                },
                is_default: true,
            })
            .unwrap();
        let provider = store.get_default_provider().unwrap().unwrap();
        assert_eq!(provider.config.api_key, "top-secret");
    }
}
