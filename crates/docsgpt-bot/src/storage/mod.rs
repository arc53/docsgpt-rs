//! What a bot remembers between messages.
//!
//! DocsGPT keeps the transcript, so a bot only stores which DocsGPT
//! conversation each chat is in (per agent), how many turns that conversation
//! has had (feedback needs each answer's position), small per-chat state such
//! as the active agent, and free-form JSON records for platform-specific state.

pub mod memory;
pub mod sqlite;

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::config::{Backend, StorageConfig};
use crate::error::Result;

/// Where a conversation lives: one bot, one chat ("space"), one thread, and
/// optionally a namespace that splits a chat further (per user, per business
/// connection, …). All parts are strings, so any platform's ids fit.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Scope {
    /// Bot name from the config.
    pub bot: String,
    /// Chat, channel or DM id (prefix with the workspace id where ids aren't global).
    pub space: String,
    /// Thread or topic id; `"0"` when the chat has none.
    pub thread: String,
    /// Extra partition, e.g. `user:U123`.
    pub namespace: Option<String>,
}

impl Scope {
    /// A scope without a namespace. An empty `thread` is stored as `"0"`.
    pub fn new(bot: impl Into<String>, space: impl Into<String>, thread: impl Into<String>) -> Self {
        let thread = thread.into();
        Self {
            bot: bot.into(),
            space: space.into(),
            thread: if thread.is_empty() { "0".into() } else { thread },
            namespace: None,
        }
    }

    /// The same scope inside `namespace`.
    pub fn with_namespace(mut self, namespace: impl Into<String>) -> Self {
        self.namespace = Some(namespace.into());
        self
    }

    /// Stable storage key: `bot[:namespace]:space:thread`.
    pub fn key(&self) -> String {
        match &self.namespace {
            Some(ns) => format!("{}:{}:{}:{}", self.bot, ns, self.space, self.thread),
            None => format!("{}:{}:{}", self.bot, self.space, self.thread),
        }
    }
}

/// The DocsGPT conversation a scope is in with one agent.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Conversation {
    /// DocsGPT conversation id.
    pub id: String,
    /// Turns recorded in it so far, i.e. the next answer's position. `None`
    /// when unknown: the conversation predates turn counting (a file from the
    /// Telegram bot v2), so its answers' positions can't be known.
    pub turns: Option<u32>,
}

/// Small per-scope state. `extra` holds platform-specific fields.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct ChatState {
    /// Agent picked with a command; `None` means the default.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub active_agent: Option<String>,
    /// Any other fields.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

/// Changes to a [`ChatState`]; fields left `None`/empty are untouched.
#[derive(Debug, Clone, Default)]
pub struct StatePatch {
    /// `Some(None)` clears the active agent.
    pub active_agent: Option<Option<String>>,
    /// Fields of `extra` to set.
    pub set: Map<String, Value>,
    /// Fields of `extra` to remove.
    pub remove: Vec<String>,
}

impl StatePatch {
    /// A patch that only sets (or with `None`, clears) the active agent.
    pub fn active_agent(agent: Option<&str>) -> Self {
        Self {
            active_agent: Some(agent.map(str::to_string)),
            ..Self::default()
        }
    }

    /// Also set `extra[key] = value`.
    pub fn set(mut self, key: impl Into<String>, value: impl Into<Value>) -> Self {
        self.set.insert(key.into(), value.into());
        self
    }

    /// Apply to `state`.
    pub fn apply(&self, state: &mut ChatState) {
        if let Some(a) = &self.active_agent {
            state.active_agent = a.clone();
        }
        for k in &self.remove {
            state.extra.remove(k);
        }
        for (k, v) in &self.set {
            state.extra.insert(k.clone(), v.clone());
        }
    }
}

/// Which answer a platform message holds, so a later 👍 on it can be sent to DocsGPT.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MessageRef {
    /// Agent that answered.
    pub agent: String,
    /// DocsGPT conversation.
    pub conversation_id: String,
    /// The answer's position in the conversation (feedback's `question_index`).
    pub position: u32,
}

/// The JSON record kind used by [`Storage::put_message_ref`].
pub const MESSAGE_REF_KIND: &str = "message_ref";

/// A storage backend. Implementations must be safe to share across tasks.
#[async_trait]
pub trait Storage: Send + Sync {
    /// The conversation `scope` is in with `agent`.
    async fn conversation(&self, scope: &Scope, agent: &str) -> Result<Option<Conversation>>;

    /// Point `scope` at conversation `id` with `agent`. Keeps the turn count if
    /// the id is unchanged and resets it to 0 otherwise.
    async fn set_conversation(&self, scope: &Scope, agent: &str, id: &str) -> Result<()>;

    /// Forget the conversation, so the next message starts a new one.
    async fn clear_conversation(&self, scope: &Scope, agent: &str) -> Result<()>;

    /// Count one turn in conversation `id` and return its position (0 for the
    /// first). Switching to a new id starts again at 0. Returns `None` while the
    /// conversation's count is unknown (see [`Conversation::turns`]).
    ///
    /// Call it once per turn, as soon as DocsGPT has reserved the answer (its
    /// `message_id` event) and the conversation id is known.
    async fn record_turn(&self, scope: &Scope, agent: &str, id: &str) -> Result<Option<u32>>;

    /// The scope's state (default when nothing is stored).
    async fn chat_state(&self, scope: &Scope) -> Result<ChatState>;

    /// Apply `patch` atomically and return the new state.
    async fn update_chat_state(&self, scope: &Scope, patch: StatePatch) -> Result<ChatState>;

    /// A JSON record by `(bot, kind, id)`.
    async fn get_json(&self, bot: &str, kind: &str, id: &str) -> Result<Option<Value>>;

    /// Store a JSON record, replacing any previous value.
    async fn put_json(&self, bot: &str, kind: &str, id: &str, value: &Value) -> Result<()>;

    /// Delete a JSON record.
    async fn delete_json(&self, bot: &str, kind: &str, id: &str) -> Result<()>;

    /// Delete records of `kind` not written for `older_than`; returns how many.
    async fn prune_json(&self, kind: &str, older_than: Duration) -> Result<u64>;

    /// Backend name for logs.
    fn name(&self) -> &'static str;

    /// Remember which answer a platform message (by the bot's own id for it) holds.
    async fn put_message_ref(&self, bot: &str, message: &str, r: &MessageRef) -> Result<()> {
        let v = serde_json::to_value(r).map_err(crate::Error::storage)?;
        self.put_json(bot, MESSAGE_REF_KIND, message, &v).await
    }

    /// Look up what [`Storage::put_message_ref`] stored.
    async fn message_ref(&self, bot: &str, message: &str) -> Result<Option<MessageRef>> {
        match self.get_json(bot, MESSAGE_REF_KIND, message).await? {
            Some(v) => Ok(serde_json::from_value(v).ok()),
            None => Ok(None),
        }
    }
}

/// Open the configured backend. `default_sqlite_path` is used when the config names no file.
pub async fn open(cfg: &StorageConfig, default_sqlite_path: &str) -> Result<Arc<dyn Storage>> {
    let storage: Arc<dyn Storage> = match cfg.backend {
        Backend::Memory => Arc::new(memory::MemoryStorage::default()),
        Backend::Sqlite => Arc::new(sqlite::SqliteStorage::open(cfg.sqlite_path(default_sqlite_path)).await?),
    };
    tracing::info!(backend = storage.name(), "storage ready");
    Ok(storage)
}

/// Behaviour every backend must have. Run it from your backend's tests.
#[cfg(any(test, feature = "testing"))]
pub async fn contract_tests(s: &dyn Storage) {
    let scope = Scope::new("bot", "C1", "");
    assert_eq!(scope.key(), "bot:C1:0");
    assert_eq!(s.conversation(&scope, "a").await.unwrap(), None);

    // Turns count per conversation and restart with a new one.
    s.set_conversation(&scope, "a", "conv-1").await.unwrap();
    assert_eq!(s.record_turn(&scope, "a", "conv-1").await.unwrap(), Some(0));
    assert_eq!(s.record_turn(&scope, "a", "conv-1").await.unwrap(), Some(1));
    s.set_conversation(&scope, "a", "conv-1").await.unwrap();
    assert_eq!(
        s.conversation(&scope, "a").await.unwrap(),
        Some(Conversation {
            id: "conv-1".into(),
            turns: Some(2)
        })
    );
    assert_eq!(s.record_turn(&scope, "a", "conv-2").await.unwrap(), Some(0));
    assert_eq!(
        s.conversation(&scope, "a").await.unwrap(),
        Some(Conversation {
            id: "conv-2".into(),
            turns: Some(1)
        })
    );
    s.set_conversation(&scope, "a", "conv-3").await.unwrap();
    assert_eq!(s.conversation(&scope, "a").await.unwrap().unwrap().turns, Some(0));
    // record_turn on a scope with no conversation yet.
    assert_eq!(s.record_turn(&scope, "b", "conv-b").await.unwrap(), Some(0));

    // Agents, threads and namespaces are separate.
    let thread = Scope::new("bot", "C1", "1712.5");
    let ns = Scope::new("bot", "C1", "").with_namespace("user:U1");
    assert_eq!(ns.key(), "bot:user:U1:C1:0");
    assert_eq!(s.conversation(&thread, "a").await.unwrap(), None);
    assert_eq!(s.conversation(&ns, "a").await.unwrap(), None);
    assert_eq!(s.conversation(&scope, "b").await.unwrap().unwrap().id, "conv-b");

    s.clear_conversation(&scope, "a").await.unwrap();
    assert_eq!(s.conversation(&scope, "a").await.unwrap(), None);
    assert!(s.conversation(&scope, "b").await.unwrap().is_some());

    // Chat state.
    assert_eq!(s.chat_state(&scope).await.unwrap(), ChatState::default());
    let st = s
        .update_chat_state(&scope, StatePatch::active_agent(Some("b")).set("lang", "de"))
        .await
        .unwrap();
    assert_eq!(st.active_agent.as_deref(), Some("b"));
    assert_eq!(s.chat_state(&scope).await.unwrap(), st);
    let st = s
        .update_chat_state(
            &scope,
            StatePatch {
                remove: vec!["lang".into()],
                ..StatePatch::active_agent(None)
            },
        )
        .await
        .unwrap();
    assert_eq!(st, ChatState::default());
    assert_eq!(s.chat_state(&thread).await.unwrap(), ChatState::default());

    // JSON records and message refs.
    assert_eq!(s.get_json("bot", "k", "1").await.unwrap(), None);
    s.put_json("bot", "k", "1", &serde_json::json!({"x": 1})).await.unwrap();
    s.put_json("bot", "k", "1", &serde_json::json!({"x": 2})).await.unwrap();
    assert_eq!(
        s.get_json("bot", "k", "1").await.unwrap(),
        Some(serde_json::json!({"x": 2}))
    );
    assert_eq!(s.get_json("other", "k", "1").await.unwrap(), None);
    assert_eq!(s.get_json("bot", "k2", "1").await.unwrap(), None);
    s.delete_json("bot", "k", "1").await.unwrap();
    assert_eq!(s.get_json("bot", "k", "1").await.unwrap(), None);

    let r = MessageRef {
        agent: "a".into(),
        conversation_id: "conv-1".into(),
        position: 3,
    };
    s.put_message_ref("bot", "C1/1712.6", &r).await.unwrap();
    assert_eq!(s.message_ref("bot", "C1/1712.6").await.unwrap(), Some(r));
    assert_eq!(
        s.prune_json(MESSAGE_REF_KIND, Duration::from_secs(3600)).await.unwrap(),
        0
    );
    assert_eq!(s.prune_json(MESSAGE_REF_KIND, Duration::ZERO).await.unwrap(), 1);
    assert_eq!(s.message_ref("bot", "C1/1712.6").await.unwrap(), None);
}

pub(crate) fn unix_now() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}
