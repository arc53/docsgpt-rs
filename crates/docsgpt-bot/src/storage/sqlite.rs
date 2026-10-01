//! SQLite storage: one file, WAL mode, queries on the blocking pool.
//!
//! The `conversations` and `chat_state` tables are compatible with the files
//! written by the Telegram bot v2, so it can move onto this crate in place.

use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use rusqlite::{Connection, OptionalExtension, params};
use serde_json::Value;

use super::{ChatState, Conversation, Scope, StatePatch, Storage, unix_now};
use crate::error::{Error, Result};

/// Storage in a SQLite file.
pub struct SqliteStorage {
    conn: Arc<Mutex<Connection>>,
}

const SCHEMA: &str = r#"
CREATE TABLE IF NOT EXISTS conversations (
    scope TEXT NOT NULL,
    agent TEXT NOT NULL,
    conversation_id TEXT NOT NULL,
    updated_at INTEGER NOT NULL,
    PRIMARY KEY (scope, agent)
);
CREATE TABLE IF NOT EXISTS chat_state (
    scope TEXT PRIMARY KEY,
    state TEXT NOT NULL,
    updated_at INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS records (
    bot TEXT NOT NULL,
    kind TEXT NOT NULL,
    id TEXT NOT NULL,
    value TEXT NOT NULL,
    updated_at INTEGER NOT NULL,
    PRIMARY KEY (bot, kind, id)
);
CREATE INDEX IF NOT EXISTS records_kind_updated ON records (kind, updated_at);
"#;

impl SqliteStorage {
    /// Open (creating the file, its directory and the tables if needed).
    /// `":memory:"` opens a private in-memory database.
    pub async fn open(path: &str) -> Result<Self> {
        let path = path.to_string();
        tokio::task::spawn_blocking(move || Self::open_blocking(&path))
            .await
            .map_err(Error::storage)?
    }

    /// [`SqliteStorage::open`] for non-async callers.
    pub fn open_blocking(path: &str) -> Result<Self> {
        if path != ":memory:"
            && let Some(dir) = Path::new(path).parent()
            && !dir.as_os_str().is_empty()
        {
            std::fs::create_dir_all(dir).map_err(|e| Error::storage(format!("creating {}: {e}", dir.display())))?;
        }
        let conn = Connection::open(path).map_err(|e| Error::storage(format!("opening {path}: {e}")))?;
        conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL; PRAGMA busy_timeout=5000;")
            .map_err(Error::storage)?;
        conn.execute_batch(SCHEMA).map_err(Error::storage)?;
        // Files from the Telegram bot v2 predate the turn counter. Their
        // conversations already hold answers that were never counted, so mark
        // them unknown (-1) rather than start at 0 and misplace feedback.
        let has_turns: bool = conn
            .prepare("SELECT 1 FROM pragma_table_info('conversations') WHERE name = 'turns'")
            .and_then(|mut s| s.exists([]))
            .map_err(Error::storage)?;
        if !has_turns {
            conn.execute_batch("ALTER TABLE conversations ADD COLUMN turns INTEGER NOT NULL DEFAULT -1;")
                .map_err(Error::storage)?;
        }
        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
        })
    }

    async fn run<T, F>(&self, f: F) -> Result<T>
    where
        T: Send + 'static,
        F: FnOnce(&mut Connection) -> rusqlite::Result<T> + Send + 'static,
    {
        let conn = self.conn.clone();
        tokio::task::spawn_blocking(move || {
            let mut guard = conn.lock().unwrap_or_else(|p| p.into_inner());
            f(&mut guard).map_err(Error::storage)
        })
        .await
        .map_err(Error::storage)?
    }
}

/// Upsert that keeps `turns` when the id is unchanged and resets it otherwise.
/// A negative `turns` means unknown.
const SET_CONVERSATION: &str = "INSERT INTO conversations(scope, agent, conversation_id, turns, updated_at)
     VALUES (?1, ?2, ?3, 0, ?4)
     ON CONFLICT(scope, agent) DO UPDATE SET
       turns = CASE WHEN conversation_id = excluded.conversation_id THEN turns ELSE 0 END,
       conversation_id = excluded.conversation_id,
       updated_at = excluded.updated_at";

#[async_trait]
impl Storage for SqliteStorage {
    async fn conversation(&self, scope: &Scope, agent: &str) -> Result<Option<Conversation>> {
        let (scope, agent) = (scope.key(), agent.to_string());
        self.run(move |c| {
            c.query_row(
                "SELECT conversation_id, turns FROM conversations WHERE scope = ?1 AND agent = ?2",
                params![scope, agent],
                |r| {
                    Ok(Conversation {
                        id: r.get(0)?,
                        turns: u32::try_from(r.get::<_, i64>(1)?).ok(),
                    })
                },
            )
            .optional()
        })
        .await
    }

    async fn set_conversation(&self, scope: &Scope, agent: &str, id: &str) -> Result<()> {
        let (scope, agent, id) = (scope.key(), agent.to_string(), id.to_string());
        self.run(move |c| {
            c.execute(SET_CONVERSATION, params![scope, agent, id, unix_now()])
                .map(drop)
        })
        .await
    }

    async fn clear_conversation(&self, scope: &Scope, agent: &str) -> Result<()> {
        let (scope, agent) = (scope.key(), agent.to_string());
        self.run(move |c| {
            c.execute(
                "DELETE FROM conversations WHERE scope = ?1 AND agent = ?2",
                params![scope, agent],
            )
            .map(drop)
        })
        .await
    }

    async fn record_turn(&self, scope: &Scope, agent: &str, id: &str) -> Result<Option<u32>> {
        let (scope, agent, id) = (scope.key(), agent.to_string(), id.to_string());
        self.run(move |c| {
            let tx = c.transaction()?;
            tx.execute(SET_CONVERSATION, params![scope, agent, id, unix_now()])?;
            let position: Option<u32> = tx.query_row(
                "UPDATE conversations SET turns = CASE WHEN turns < 0 THEN turns ELSE turns + 1 END
                 WHERE scope = ?1 AND agent = ?2
                 RETURNING CASE WHEN turns < 0 THEN NULL ELSE turns - 1 END",
                params![scope, agent],
                |r| r.get(0),
            )?;
            tx.commit()?;
            Ok(position)
        })
        .await
    }

    async fn chat_state(&self, scope: &Scope) -> Result<ChatState> {
        let scope = scope.key();
        let raw = self
            .run(move |c| {
                c.query_row("SELECT state FROM chat_state WHERE scope = ?1", params![scope], |r| {
                    r.get::<_, String>(0)
                })
                .optional()
            })
            .await?;
        Ok(raw.and_then(|s| serde_json::from_str(&s).ok()).unwrap_or_default())
    }

    async fn update_chat_state(&self, scope: &Scope, patch: StatePatch) -> Result<ChatState> {
        let scope = scope.key();
        self.run(move |c| {
            let tx = c.transaction()?;
            let raw: Option<String> = tx
                .query_row("SELECT state FROM chat_state WHERE scope = ?1", params![scope], |r| {
                    r.get(0)
                })
                .optional()?;
            let mut state: ChatState = raw.and_then(|s| serde_json::from_str(&s).ok()).unwrap_or_default();
            patch.apply(&mut state);
            let json = serde_json::to_string(&state).unwrap_or_else(|_| "{}".into());
            tx.execute(
                "INSERT INTO chat_state(scope, state, updated_at) VALUES (?1, ?2, ?3)
                 ON CONFLICT(scope) DO UPDATE SET state = excluded.state, updated_at = excluded.updated_at",
                params![scope, json, unix_now()],
            )?;
            tx.commit()?;
            Ok(state)
        })
        .await
    }

    async fn get_json(&self, bot: &str, kind: &str, id: &str) -> Result<Option<Value>> {
        let (bot, kind, id) = (bot.to_string(), kind.to_string(), id.to_string());
        let raw = self
            .run(move |c| {
                c.query_row(
                    "SELECT value FROM records WHERE bot = ?1 AND kind = ?2 AND id = ?3",
                    params![bot, kind, id],
                    |r| r.get::<_, String>(0),
                )
                .optional()
            })
            .await?;
        Ok(raw.and_then(|s| serde_json::from_str(&s).ok()))
    }

    async fn put_json(&self, bot: &str, kind: &str, id: &str, value: &Value) -> Result<()> {
        let (bot, kind, id, value) = (bot.to_string(), kind.to_string(), id.to_string(), value.to_string());
        self.run(move |c| {
            c.execute(
                "INSERT INTO records(bot, kind, id, value, updated_at) VALUES (?1, ?2, ?3, ?4, ?5)
                 ON CONFLICT(bot, kind, id) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at",
                params![bot, kind, id, value, unix_now()],
            )
            .map(drop)
        })
        .await
    }

    async fn delete_json(&self, bot: &str, kind: &str, id: &str) -> Result<()> {
        let (bot, kind, id) = (bot.to_string(), kind.to_string(), id.to_string());
        self.run(move |c| {
            c.execute(
                "DELETE FROM records WHERE bot = ?1 AND kind = ?2 AND id = ?3",
                params![bot, kind, id],
            )
            .map(drop)
        })
        .await
    }

    async fn prune_json(&self, kind: &str, older_than: Duration) -> Result<u64> {
        let kind = kind.to_string();
        let cutoff = unix_now() - older_than.as_secs() as i64;
        self.run(move |c| {
            c.execute(
                "DELETE FROM records WHERE kind = ?1 AND updated_at <= ?2",
                params![kind, cutoff],
            )
            .map(|n| n as u64)
        })
        .await
    }

    fn name(&self) -> &'static str {
        "sqlite"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn contract() {
        super::super::contract_tests(&SqliteStorage::open(":memory:").await.unwrap()).await;
    }

    #[tokio::test]
    async fn persists_across_reopen() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("nested/bot.db");
        let path = path.to_str().unwrap();
        let scope = Scope::new("b", "C", "");
        {
            let s = SqliteStorage::open(path).await.unwrap();
            assert_eq!(s.record_turn(&scope, "a", "conv").await.unwrap(), Some(0));
            s.update_chat_state(&scope, StatePatch::active_agent(Some("a")))
                .await
                .unwrap();
        }
        let s = SqliteStorage::open(path).await.unwrap();
        assert_eq!(
            s.conversation(&scope, "a").await.unwrap(),
            Some(Conversation {
                id: "conv".into(),
                turns: Some(1)
            })
        );
        assert_eq!(s.chat_state(&scope).await.unwrap().active_agent.as_deref(), Some("a"));
    }

    /// A database written by the Telegram bot v2 opens, keeps its data and
    /// gains the turn counter; its extra chat-state fields survive in `extra`.
    #[tokio::test]
    async fn upgrades_telegram_v2_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tg.db");
        {
            let c = Connection::open(&path).unwrap();
            c.execute_batch(
                "CREATE TABLE conversations (scope TEXT NOT NULL, agent TEXT NOT NULL, conversation_id TEXT NOT NULL,
                   updated_at INTEGER NOT NULL, PRIMARY KEY (scope, agent));
                 CREATE TABLE chat_state (scope TEXT PRIMARY KEY, state TEXT NOT NULL, updated_at INTEGER NOT NULL);
                 INSERT INTO conversations VALUES ('tg:42:0', 'default', 'conv-old', 1);
                 INSERT INTO chat_state VALUES ('tg:42:0', '{\"active_agent\":\"sales\",\"last_question\":\"hi\",\"user\":{\"id\":1}}', 1);",
            )
            .unwrap();
        }
        let s = SqliteStorage::open(path.to_str().unwrap()).await.unwrap();
        let scope = Scope::new("tg", "42", "0");
        assert_eq!(
            s.conversation(&scope, "default").await.unwrap(),
            Some(Conversation {
                id: "conv-old".into(),
                turns: None
            })
        );
        let st = s.chat_state(&scope).await.unwrap();
        assert_eq!(st.active_agent.as_deref(), Some("sales"));
        assert_eq!(st.extra["last_question"], "hi");
        // Answers already in that conversation weren't counted: its positions are unknown.
        assert_eq!(s.record_turn(&scope, "default", "conv-old").await.unwrap(), None);
        assert_eq!(s.record_turn(&scope, "default", "conv-old").await.unwrap(), None);
        // A new conversation is counted from the start.
        assert_eq!(s.record_turn(&scope, "default", "conv-new").await.unwrap(), Some(0));
        assert_eq!(s.record_turn(&scope, "default", "conv-new").await.unwrap(), Some(1));
        // Reopening doesn't mark counted rows unknown again.
        drop(s);
        let s = SqliteStorage::open(path.to_str().unwrap()).await.unwrap();
        assert_eq!(s.record_turn(&scope, "default", "conv-new").await.unwrap(), Some(2));
    }
}
