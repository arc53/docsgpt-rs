//! In-process storage; everything is lost on restart.

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::Duration;

use async_trait::async_trait;
use serde_json::Value;

use super::{ChatState, Conversation, Scope, StatePatch, Storage, unix_now};
use crate::error::Result;

/// Storage in process memory. Good for trying a bot out and for tests.
#[derive(Default)]
pub struct MemoryStorage {
    inner: Mutex<Inner>,
}

#[derive(Default)]
struct Inner {
    conversations: HashMap<(String, String), Conversation>,
    states: HashMap<String, ChatState>,
    records: HashMap<(String, String, String), (Value, i64)>,
}

impl MemoryStorage {
    fn with<T>(&self, f: impl FnOnce(&mut Inner) -> T) -> T {
        let mut guard = self.inner.lock().unwrap_or_else(|p| p.into_inner());
        f(&mut guard)
    }
}

fn conv_key(scope: &Scope, agent: &str) -> (String, String) {
    (scope.key(), agent.to_string())
}

#[async_trait]
impl Storage for MemoryStorage {
    async fn conversation(&self, scope: &Scope, agent: &str) -> Result<Option<Conversation>> {
        Ok(self.with(|i| i.conversations.get(&conv_key(scope, agent)).cloned()))
    }

    async fn set_conversation(&self, scope: &Scope, agent: &str, id: &str) -> Result<()> {
        self.with(|i| {
            let entry = i.conversations.entry(conv_key(scope, agent)).or_insert(Conversation {
                id: id.into(),
                turns: Some(0),
            });
            if entry.id != id {
                *entry = Conversation {
                    id: id.into(),
                    turns: Some(0),
                };
            }
        });
        Ok(())
    }

    async fn clear_conversation(&self, scope: &Scope, agent: &str) -> Result<()> {
        self.with(|i| i.conversations.remove(&conv_key(scope, agent)));
        Ok(())
    }

    async fn record_turn(&self, scope: &Scope, agent: &str, id: &str) -> Result<Option<u32>> {
        Ok(self.with(|i| {
            let entry = i.conversations.entry(conv_key(scope, agent)).or_insert(Conversation {
                id: id.into(),
                turns: Some(0),
            });
            if entry.id != id {
                *entry = Conversation {
                    id: id.into(),
                    turns: Some(0),
                };
            }
            let position = entry.turns;
            entry.turns = position.map(|n| n + 1);
            position
        }))
    }

    async fn chat_state(&self, scope: &Scope) -> Result<ChatState> {
        Ok(self.with(|i| i.states.get(&scope.key()).cloned().unwrap_or_default()))
    }

    async fn update_chat_state(&self, scope: &Scope, patch: StatePatch) -> Result<ChatState> {
        Ok(self.with(|i| {
            let st = i.states.entry(scope.key()).or_default();
            patch.apply(st);
            st.clone()
        }))
    }

    async fn get_json(&self, bot: &str, kind: &str, id: &str) -> Result<Option<Value>> {
        Ok(self.with(|i| {
            i.records
                .get(&(bot.into(), kind.into(), id.into()))
                .map(|(v, _)| v.clone())
        }))
    }

    async fn put_json(&self, bot: &str, kind: &str, id: &str, value: &Value) -> Result<()> {
        self.with(|i| {
            i.records
                .insert((bot.into(), kind.into(), id.into()), (value.clone(), unix_now()))
        });
        Ok(())
    }

    async fn delete_json(&self, bot: &str, kind: &str, id: &str) -> Result<()> {
        self.with(|i| i.records.remove(&(bot.into(), kind.into(), id.into())));
        Ok(())
    }

    async fn prune_json(&self, kind: &str, older_than: Duration) -> Result<u64> {
        let cutoff = unix_now() - older_than.as_secs() as i64;
        Ok(self.with(|i| {
            let before = i.records.len();
            i.records.retain(|(_, k, _), (_, at)| k != kind || *at > cutoff);
            (before - i.records.len()) as u64
        }))
    }

    fn name(&self) -> &'static str {
        "memory"
    }
}

#[cfg(test)]
mod tests {
    #[tokio::test]
    async fn contract() {
        super::super::contract_tests(&super::MemoryStorage::default()).await;
    }
}
