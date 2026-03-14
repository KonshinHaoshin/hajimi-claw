use std::path::Path;
use std::sync::Mutex;

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use hajimi_claw_types::{
    ApprovalRequest, ConversationId, ConversationMessage, SessionHandle, SessionSummary, TaskId,
    TaskKind, TaskStatus,
};
use rusqlite::{Connection, OptionalExtension, params};

pub struct Store {
    connection: Mutex<Connection>,
}

impl Store {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let connection = Connection::open(path).context("open sqlite database")?;
        connection.pragma_update(None, "journal_mode", "WAL")?;
        let store = Self {
            connection: Mutex::new(connection),
        };
        store.migrate()?;
        Ok(store)
    }

    pub fn open_in_memory() -> Result<Self> {
        let connection = Connection::open_in_memory().context("open sqlite memory database")?;
        connection.pragma_update(None, "journal_mode", "WAL")?;
        let store = Self {
            connection: Mutex::new(connection),
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
}

fn parse_ts(ts: String) -> DateTime<Utc> {
    DateTime::parse_from_rfc3339(&ts)
        .expect("timestamp stored in rfc3339")
        .with_timezone(&Utc)
}

#[cfg(test)]
mod tests {
    use chrono::Utc;
    use hajimi_claw_types::{ConversationMessage, MessageRole};

    use super::Store;

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
}
