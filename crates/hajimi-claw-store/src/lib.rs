use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use aes_gcm::aead::{Aead, KeyInit, OsRng, rand_core::RngCore};
use aes_gcm::{Aes256Gcm, Nonce};
use anyhow::{Context, Result};
use base64::Engine;
use base64::engine::general_purpose::STANDARD as BASE64;
use chrono::{DateTime, Utc};
use hajimi_claw_types::{
    ApprovalId, ApprovalRecord, ApprovalRequest, ConversationId, ConversationMessage,
    HeartbeatStatus, OnboardingSession, ProviderCapabilities, ProviderConfig, ProviderDraft,
    ProviderKind, ProviderRecord, SessionHandle, SessionSummary, TaskId, TaskKind, TaskRunState,
    TaskStatus, ToolInvocationRecord, ToolInvocationStatus,
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
                conversation_id TEXT NOT NULL DEFAULT '',
                kind TEXT NOT NULL,
                description TEXT NOT NULL,
                queued_at TEXT NOT NULL,
                started_at TEXT,
                finished_at TEXT,
                state TEXT NOT NULL DEFAULT 'Queued',
                running INTEGER NOT NULL,
                cwd TEXT,
                provider_id TEXT,
                current_session_id TEXT,
                result_preview TEXT,
                error TEXT,
                blocked_approval_id TEXT
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
                approved INTEGER,
                task_id TEXT,
                tool_name TEXT
            );

            CREATE TABLE IF NOT EXISTS tool_invocations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id TEXT NOT NULL,
                conversation_id TEXT NOT NULL,
                call_id TEXT,
                tool_name TEXT NOT NULL,
                arguments_json TEXT NOT NULL,
                status TEXT NOT NULL,
                output_content TEXT,
                output_structured_json TEXT,
                error TEXT,
                approval_id TEXT,
                sequence_no INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
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
                fallback_models_json TEXT NOT NULL DEFAULT '[]',
                capabilities_json TEXT NOT NULL DEFAULT '{"tool_calling":false,"streaming":false,"json_mode":false,"max_context_chars":null}',
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
        ensure_column(
            &connection,
            "providers",
            "fallback_models_json",
            "ALTER TABLE providers ADD COLUMN fallback_models_json TEXT NOT NULL DEFAULT '[]'",
        )?;
        ensure_column(
            &connection,
            "tasks",
            "conversation_id",
            "ALTER TABLE tasks ADD COLUMN conversation_id TEXT NOT NULL DEFAULT ''",
        )?;
        ensure_column(
            &connection,
            "tasks",
            "state",
            "ALTER TABLE tasks ADD COLUMN state TEXT NOT NULL DEFAULT 'Queued'",
        )?;
        ensure_column(
            &connection,
            "tasks",
            "cwd",
            "ALTER TABLE tasks ADD COLUMN cwd TEXT",
        )?;
        ensure_column(
            &connection,
            "tasks",
            "provider_id",
            "ALTER TABLE tasks ADD COLUMN provider_id TEXT",
        )?;
        ensure_column(
            &connection,
            "tasks",
            "current_session_id",
            "ALTER TABLE tasks ADD COLUMN current_session_id TEXT",
        )?;
        ensure_column(
            &connection,
            "tasks",
            "result_preview",
            "ALTER TABLE tasks ADD COLUMN result_preview TEXT",
        )?;
        ensure_column(
            &connection,
            "tasks",
            "error",
            "ALTER TABLE tasks ADD COLUMN error TEXT",
        )?;
        ensure_column(
            &connection,
            "tasks",
            "blocked_approval_id",
            "ALTER TABLE tasks ADD COLUMN blocked_approval_id TEXT",
        )?;
        ensure_column(
            &connection,
            "approvals",
            "task_id",
            "ALTER TABLE approvals ADD COLUMN task_id TEXT",
        )?;
        ensure_column(
            &connection,
            "approvals",
            "tool_name",
            "ALTER TABLE approvals ADD COLUMN tool_name TEXT",
        )?;
        ensure_column(
            &connection,
            "providers",
            "capabilities_json",
            "ALTER TABLE providers ADD COLUMN capabilities_json TEXT NOT NULL DEFAULT '{\"tool_calling\":false,\"streaming\":false,\"json_mode\":false,\"max_context_chars\":null}'",
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

    pub fn list_messages(
        &self,
        conversation_id: ConversationId,
        limit: usize,
    ) -> Result<Vec<ConversationMessage>> {
        let connection = self.connection.lock().expect("store lock poisoned");
        let mut stmt = connection.prepare(
            "SELECT role, content, created_at FROM messages WHERE conversation_id = ? ORDER BY id ASC LIMIT ?",
        )?;
        let rows = stmt.query_map(params![conversation_id.to_string(), limit as i64], |row| {
            Ok(ConversationMessage {
                role: parse_message_role(row.get::<_, String>(0)?),
                content: row.get(1)?,
                created_at: parse_ts(row.get(2)?),
            })
        })?;
        rows.collect::<rusqlite::Result<Vec<_>>>()
            .map_err(Into::into)
    }

    pub fn upsert_task(&self, task: &TaskStatus) -> Result<()> {
        let connection = self.connection.lock().expect("store lock poisoned");
        connection.execute(
            r#"
            INSERT INTO tasks (id, conversation_id, kind, description, queued_at, started_at, finished_at, state, running, cwd, provider_id, current_session_id, result_preview, error, blocked_approval_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                conversation_id=excluded.conversation_id,
                kind=excluded.kind,
                description=excluded.description,
                queued_at=excluded.queued_at,
                started_at=excluded.started_at,
                finished_at=excluded.finished_at,
                state=excluded.state,
                running=excluded.running,
                cwd=excluded.cwd,
                provider_id=excluded.provider_id,
                current_session_id=excluded.current_session_id,
                result_preview=excluded.result_preview,
                error=excluded.error,
                blocked_approval_id=excluded.blocked_approval_id
            "#,
            params![
                task.id.to_string(),
                task.conversation_id.to_string(),
                format!("{:?}", task.kind),
                task.description,
                task.queued_at.to_rfc3339(),
                task.started_at.map(|ts| ts.to_rfc3339()),
                task.finished_at.map(|ts| ts.to_rfc3339()),
                format!("{:?}", task.state),
                i64::from(task.running),
                task.cwd.as_ref().map(|cwd| cwd.display().to_string()),
                task.provider_id,
                task.current_session_id,
                task.result_preview,
                task.error,
                task.blocked_approval_id.map(|id| id.to_string()),
            ],
        )?;
        Ok(())
    }

    pub fn list_tasks(&self) -> Result<Vec<TaskStatus>> {
        let connection = self.connection.lock().expect("store lock poisoned");
        let mut stmt = connection.prepare(
            "SELECT id, conversation_id, kind, description, queued_at, started_at, finished_at, state, running, cwd, provider_id, current_session_id, result_preview, error, blocked_approval_id FROM tasks ORDER BY queued_at DESC",
        )?;
        let rows = stmt.query_map([], |row| {
            Ok(TaskStatus {
                id: TaskId(uuid::Uuid::parse_str(row.get::<_, String>(0)?.as_str()).unwrap()),
                conversation_id: parse_conversation_id(row.get::<_, String>(1)?),
                kind: match row.get::<_, String>(2)?.as_str() {
                    "PersistentShellTask" => TaskKind::PersistentShellTask,
                    _ => TaskKind::EphemeralAgentTask,
                },
                description: row.get(3)?,
                queued_at: parse_ts(row.get::<_, String>(4)?),
                started_at: row.get::<_, Option<String>>(5)?.map(parse_ts),
                finished_at: row.get::<_, Option<String>>(6)?.map(parse_ts),
                state: parse_task_state(row.get::<_, String>(7)?),
                running: row.get::<_, i64>(8)? != 0,
                cwd: row
                    .get::<_, Option<String>>(9)?
                    .map(|value| PathBuf::from(value)),
                provider_id: row.get(10)?,
                current_session_id: row.get(11)?,
                result_preview: row.get(12)?,
                error: row.get(13)?,
                blocked_approval_id: row
                    .get::<_, Option<String>>(14)?
                    .and_then(|value| uuid::Uuid::parse_str(&value).ok())
                    .map(ApprovalId),
            })
        })?;

        rows.collect::<rusqlite::Result<Vec<_>>>()
            .map_err(Into::into)
    }

    pub fn get_task(&self, task_id: TaskId) -> Result<Option<TaskStatus>> {
        let connection = self.connection.lock().expect("store lock poisoned");
        connection
            .query_row(
                "SELECT id, conversation_id, kind, description, queued_at, started_at, finished_at, state, running, cwd, provider_id, current_session_id, result_preview, error, blocked_approval_id FROM tasks WHERE id = ?",
                params![task_id.to_string()],
                |row| {
                    Ok(TaskStatus {
                        id: TaskId(uuid::Uuid::parse_str(row.get::<_, String>(0)?.as_str()).unwrap()),
                        conversation_id: parse_conversation_id(row.get::<_, String>(1)?),
                        kind: match row.get::<_, String>(2)?.as_str() {
                            "PersistentShellTask" => TaskKind::PersistentShellTask,
                            _ => TaskKind::EphemeralAgentTask,
                        },
                        description: row.get(3)?,
                        queued_at: parse_ts(row.get::<_, String>(4)?),
                        started_at: row.get::<_, Option<String>>(5)?.map(parse_ts),
                        finished_at: row.get::<_, Option<String>>(6)?.map(parse_ts),
                        state: parse_task_state(row.get::<_, String>(7)?),
                        running: row.get::<_, i64>(8)? != 0,
                        cwd: row
                            .get::<_, Option<String>>(9)?
                            .map(PathBuf::from),
                        provider_id: row.get(10)?,
                        current_session_id: row.get(11)?,
                        result_preview: row.get(12)?,
                        error: row.get(13)?,
                        blocked_approval_id: row
                            .get::<_, Option<String>>(14)?
                            .and_then(|value| uuid::Uuid::parse_str(&value).ok())
                            .map(ApprovalId),
                    })
                },
            )
            .optional()
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

    pub fn save_approval_record(&self, approval: &ApprovalRecord) -> Result<()> {
        let connection = self.connection.lock().expect("store lock poisoned");
        connection.execute(
            r#"
            INSERT INTO approvals (id, reason, risk_level, command_preview, cwd, expires_at, approved, task_id, tool_name)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                reason=excluded.reason,
                risk_level=excluded.risk_level,
                command_preview=excluded.command_preview,
                cwd=excluded.cwd,
                expires_at=excluded.expires_at,
                approved=excluded.approved,
                task_id=excluded.task_id,
                tool_name=excluded.tool_name
            "#,
            params![
                approval.request.request_id.to_string(),
                approval.request.reason,
                format!("{:?}", approval.request.risk_level),
                approval.request.command_preview,
                approval
                    .request
                    .cwd
                    .as_ref()
                    .map(|cwd| cwd.display().to_string()),
                approval.request.expires_at.to_rfc3339(),
                approval.approved.map(i64::from),
                approval.task_id.map(|id| id.to_string()),
                approval.tool_name,
            ],
        )?;
        Ok(())
    }

    pub fn save_approval(&self, approval: &ApprovalRequest, approved: Option<bool>) -> Result<()> {
        self.save_approval_record(&ApprovalRecord {
            request: approval.clone(),
            approved,
            task_id: None,
            tool_name: None,
        })
    }

    pub fn get_approval_record(&self, request_id: &str) -> Result<Option<ApprovalRecord>> {
        let connection = self.connection.lock().expect("store lock poisoned");
        connection
            .query_row(
                "SELECT reason, risk_level, command_preview, cwd, expires_at, approved, task_id, tool_name FROM approvals WHERE id = ?",
                params![request_id],
                |row| {
                    let request_uuid = uuid::Uuid::parse_str(request_id).map_err(to_sql_uuid_err)?;
                    Ok(ApprovalRecord {
                        request: ApprovalRequest {
                            request_id: ApprovalId(request_uuid),
                            reason: row.get(0)?,
                            risk_level: parse_risk_level(row.get::<_, String>(1)?),
                            command_preview: row.get(2)?,
                            cwd: row
                                .get::<_, Option<String>>(3)?
                                .map(PathBuf::from),
                            expires_at: parse_ts(row.get(4)?),
                        },
                        approved: row
                            .get::<_, Option<i64>>(5)?
                            .map(|value| value != 0),
                        task_id: row
                            .get::<_, Option<String>>(6)?
                            .and_then(|value| uuid::Uuid::parse_str(&value).ok())
                            .map(TaskId),
                        tool_name: row.get(7)?,
                    })
                },
            )
            .optional()
            .map_err(Into::into)
    }

    pub fn list_pending_approvals(&self) -> Result<Vec<ApprovalRecord>> {
        let connection = self.connection.lock().expect("store lock poisoned");
        let mut stmt = connection.prepare(
            "SELECT id, reason, risk_level, command_preview, cwd, expires_at, approved, task_id, tool_name FROM approvals WHERE approved IS NULL ORDER BY expires_at ASC",
        )?;
        let rows = stmt.query_map([], |row| {
            let request_id = row.get::<_, String>(0)?;
            let request_uuid = uuid::Uuid::parse_str(&request_id).map_err(to_sql_uuid_err)?;
            Ok(ApprovalRecord {
                request: ApprovalRequest {
                    request_id: ApprovalId(request_uuid),
                    reason: row.get(1)?,
                    risk_level: parse_risk_level(row.get::<_, String>(2)?),
                    command_preview: row.get(3)?,
                    cwd: row.get::<_, Option<String>>(4)?.map(PathBuf::from),
                    expires_at: parse_ts(row.get(5)?),
                },
                approved: row.get::<_, Option<i64>>(6)?.map(|value| value != 0),
                task_id: row
                    .get::<_, Option<String>>(7)?
                    .and_then(|value| uuid::Uuid::parse_str(&value).ok())
                    .map(TaskId),
                tool_name: row.get(8)?,
            })
        })?;
        rows.collect::<rusqlite::Result<Vec<_>>>()
            .map_err(Into::into)
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

    pub fn save_tool_invocation(&self, record: &ToolInvocationRecord) -> Result<()> {
        let connection = self.connection.lock().expect("store lock poisoned");
        connection.execute(
            r#"
            INSERT INTO tool_invocations (task_id, conversation_id, call_id, tool_name, arguments_json, status, output_content, output_structured_json, error, approval_id, sequence_no, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            "#,
            params![
                record.task_id.to_string(),
                record.conversation_id.to_string(),
                record.call_id,
                record.tool_name,
                serde_json::to_string(&record.arguments)?,
                format!("{:?}", record.status),
                record.output_content,
                record
                    .output_structured
                    .as_ref()
                    .map(serde_json::to_string)
                    .transpose()?,
                record.error,
                record.approval_id.map(|id| id.to_string()),
                record.sequence,
                record.created_at.to_rfc3339(),
                record.updated_at.to_rfc3339(),
            ],
        )?;
        Ok(())
    }

    pub fn list_tool_invocations(&self, task_id: TaskId) -> Result<Vec<ToolInvocationRecord>> {
        let connection = self.connection.lock().expect("store lock poisoned");
        let mut stmt = connection.prepare(
            "SELECT task_id, conversation_id, call_id, tool_name, arguments_json, status, output_content, output_structured_json, error, approval_id, sequence_no, created_at, updated_at FROM tool_invocations WHERE task_id = ? ORDER BY sequence_no ASC, id ASC",
        )?;
        let rows = stmt.query_map(params![task_id.to_string()], |row| {
            let task_id_raw = row.get::<_, String>(0)?;
            let conversation_id_raw = row.get::<_, String>(1)?;
            let arguments_json: String = row.get(4)?;
            let output_structured_json: Option<String> = row.get(7)?;
            Ok(ToolInvocationRecord {
                task_id: TaskId(uuid::Uuid::parse_str(&task_id_raw).map_err(to_sql_uuid_err)?),
                conversation_id: parse_conversation_id(conversation_id_raw),
                call_id: row.get(2)?,
                tool_name: row.get(3)?,
                arguments: serde_json::from_str(&arguments_json).map_err(to_sql_err)?,
                status: parse_tool_status(row.get::<_, String>(5)?),
                output_content: row.get(6)?,
                output_structured: output_structured_json
                    .map(|json| serde_json::from_str(&json).map_err(to_sql_err))
                    .transpose()?,
                error: row.get(8)?,
                approval_id: row
                    .get::<_, Option<String>>(9)?
                    .and_then(|value| uuid::Uuid::parse_str(&value).ok())
                    .map(ApprovalId),
                sequence: row.get(10)?,
                created_at: parse_ts(row.get(11)?),
                updated_at: parse_ts(row.get(12)?),
            })
        })?;
        rows.collect::<rusqlite::Result<Vec<_>>>()
            .map_err(Into::into)
    }

    pub fn list_active_sessions(&self) -> Result<Vec<SessionHandle>> {
        let connection = self.connection.lock().expect("store lock poisoned");
        let mut stmt = connection.prepare(
            "SELECT id, name, cwd, created_at, last_used_at FROM shell_sessions WHERE active = 1 ORDER BY last_used_at DESC",
        )?;
        let rows = stmt.query_map([], |row| {
            let session_id = row.get::<_, String>(0)?;
            Ok(SessionHandle {
                id: hajimi_claw_types::SessionId(
                    uuid::Uuid::parse_str(&session_id).map_err(to_sql_uuid_err)?,
                ),
                name: row.get(1)?,
                cwd: Path::new(&row.get::<_, String>(2)?).to_path_buf(),
                created_at: parse_ts(row.get(3)?),
                last_used_at: parse_ts(row.get(4)?),
            })
        })?;
        rows.collect::<rusqlite::Result<Vec<_>>>()
            .map_err(Into::into)
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

    pub fn set_heartbeat(&self, heartbeat: &HeartbeatStatus) -> Result<()> {
        self.set_config(
            "heartbeat.last_seen_at",
            &heartbeat.last_seen_at.to_rfc3339(),
        )?;
        self.set_config(
            "heartbeat.pid",
            &heartbeat.pid.map(|pid| pid.to_string()).unwrap_or_default(),
        )?;
        self.set_config(
            "heartbeat.channel",
            heartbeat.channel.as_deref().unwrap_or_default(),
        )?;
        Ok(())
    }

    pub fn get_heartbeat(&self) -> Result<Option<HeartbeatStatus>> {
        let Some(last_seen_at) = self.get_config("heartbeat.last_seen_at")? else {
            return Ok(None);
        };
        if last_seen_at.trim().is_empty() {
            return Ok(None);
        }
        let pid = self
            .get_config("heartbeat.pid")?
            .and_then(|value| value.trim().parse::<u32>().ok());
        let channel = self
            .get_config("heartbeat.channel")?
            .filter(|value| !value.trim().is_empty());
        Ok(Some(HeartbeatStatus {
            last_seen_at: parse_ts(last_seen_at),
            pid,
            channel,
        }))
    }

    pub fn upsert_provider(&self, record: &ProviderRecord) -> Result<()> {
        let connection = self.connection.lock().expect("store lock poisoned");
        if record.is_default {
            connection.execute("UPDATE providers SET is_default = 0", [])?;
        }
        let encrypted_api_key = self.encrypt_secret(&record.config.api_key)?;
        connection.execute(
            r#"
            INSERT INTO providers (id, label, kind, base_url, api_key, model, fallback_models_json, capabilities_json, enabled, extra_headers_json, is_default, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                label=excluded.label,
                kind=excluded.kind,
                base_url=excluded.base_url,
                api_key=excluded.api_key,
                model=excluded.model,
                fallback_models_json=excluded.fallback_models_json,
                capabilities_json=excluded.capabilities_json,
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
                serde_json::to_string(&record.config.fallback_models)?,
                serde_json::to_string(&record.config.capabilities)?,
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
            SELECT id, label, kind, base_url, api_key, model, fallback_models_json, capabilities_json, enabled, extra_headers_json, is_default, created_at
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
                SELECT id, label, kind, base_url, api_key, model, fallback_models_json, capabilities_json, enabled, extra_headers_json, is_default, created_at
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
                SELECT id, label, kind, base_url, api_key, model, fallback_models_json, capabilities_json, enabled, extra_headers_json, is_default, created_at
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
                SELECT id, label, kind, base_url, api_key, model, fallback_models_json, capabilities_json, enabled, extra_headers_json, is_default, created_at
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
        let fallback_models_json: String = row.get(6)?;
        let fallback_models = serde_json::from_str(&fallback_models_json).map_err(to_sql_err)?;
        let capabilities_json: String = row.get(7)?;
        let capabilities = serde_json::from_str(&capabilities_json).unwrap_or_else(|_| {
            ProviderCapabilities::default()
        });
        let headers_json: String = row.get(9)?;
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
                fallback_models,
                capabilities,
                enabled: row.get::<_, i64>(8)? != 0,
                extra_headers,
                created_at: parse_ts(row.get(11)?),
            },
            is_default: row.get::<_, i64>(10)? != 0,
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

fn parse_message_role(raw: String) -> hajimi_claw_types::MessageRole {
    match raw.as_str() {
        "System" => hajimi_claw_types::MessageRole::System,
        "Assistant" => hajimi_claw_types::MessageRole::Assistant,
        "Tool" => hajimi_claw_types::MessageRole::Tool,
        _ => hajimi_claw_types::MessageRole::User,
    }
}

fn parse_conversation_id(raw: String) -> ConversationId {
    uuid::Uuid::parse_str(&raw)
        .map(ConversationId)
        .unwrap_or_default()
}

fn parse_task_state(raw: String) -> TaskRunState {
    match raw.as_str() {
        "Running" => TaskRunState::Running,
        "BlockedApproval" => TaskRunState::BlockedApproval,
        "Completed" => TaskRunState::Completed,
        "Failed" => TaskRunState::Failed,
        "Cancelled" => TaskRunState::Cancelled,
        _ => TaskRunState::Queued,
    }
}

fn parse_tool_status(raw: String) -> ToolInvocationStatus {
    match raw.as_str() {
        "Running" => ToolInvocationStatus::Running,
        "BlockedApproval" => ToolInvocationStatus::BlockedApproval,
        "Completed" => ToolInvocationStatus::Completed,
        "Failed" => ToolInvocationStatus::Failed,
        "Cancelled" => ToolInvocationStatus::Cancelled,
        _ => ToolInvocationStatus::Pending,
    }
}

fn parse_risk_level(raw: String) -> hajimi_claw_types::RiskLevel {
    match raw.as_str() {
        "Dangerous" => hajimi_claw_types::RiskLevel::Dangerous,
        "Guarded" => hajimi_claw_types::RiskLevel::Guarded,
        _ => hajimi_claw_types::RiskLevel::Safe,
    }
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

fn ensure_column(connection: &Connection, table: &str, column: &str, sql: &str) -> Result<()> {
    let mut stmt = connection.prepare(&format!("PRAGMA table_info({table})"))?;
    let columns = stmt.query_map([], |row| row.get::<_, String>(1))?;
    let exists = columns
        .collect::<rusqlite::Result<Vec<_>>>()?
        .into_iter()
        .any(|name| name == column);
    if !exists {
        connection.execute(sql, [])?;
    }
    Ok(())
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

fn to_sql_uuid_err(err: uuid::Error) -> rusqlite::Error {
    rusqlite::Error::FromSqlConversionFailure(0, rusqlite::types::Type::Text, Box::new(err))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use chrono::Utc;
    use hajimi_claw_types::{
        ConversationMessage, HeartbeatStatus, MessageRole, ProviderCapabilities, ProviderConfig,
        ProviderKind, ProviderRecord,
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
            conversation_id: hajimi_claw_types::ConversationId::new(),
            kind: hajimi_claw_types::TaskKind::EphemeralAgentTask,
            description: "test".into(),
            queued_at: Utc::now(),
            started_at: None,
            finished_at: None,
            state: hajimi_claw_types::TaskRunState::Completed,
            running: false,
            cwd: None,
            provider_id: None,
            current_session_id: None,
            result_preview: None,
            error: None,
            blocked_approval_id: None,
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
                    fallback_models: vec![],
                    capabilities: ProviderCapabilities {
                        tool_calling: true,
                        streaming: false,
                        json_mode: false,
                        max_context_chars: Some(24_000),
                    },
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

    #[test]
    fn persists_heartbeat_status() {
        let store = Store::open_in_memory().unwrap();
        let heartbeat = HeartbeatStatus {
            last_seen_at: Utc::now(),
            pid: Some(1234),
            channel: Some("telegram".into()),
        };
        store.set_heartbeat(&heartbeat).unwrap();
        let loaded = store.get_heartbeat().unwrap().unwrap();
        assert_eq!(loaded.pid, Some(1234));
        assert_eq!(loaded.channel.as_deref(), Some("telegram"));
    }
}
