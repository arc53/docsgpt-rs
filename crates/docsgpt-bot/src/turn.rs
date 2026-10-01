//! One question → one answer, on any platform.

use std::sync::Arc;
use std::time::Duration;

use docsgpt::{AskRequest, Event, Feedback, Source, ToolOutputs};
use futures_util::StreamExt;
use tokio::time::Instant;
use tokio_util::sync::CancellationToken;

use crate::agents::{Agents, Routed};
use crate::error::Result;
use crate::markdown::{close_open_fence, strip_images};
use crate::runtime::ScopeLocks;
use crate::storage::{MessageRef, Scope, StatePatch, Storage};
use crate::surface::{Final, Outcome, Progress, Surface, Turn};
use crate::util::truncate_chars;

/// What every turn of one bot shares.
pub struct BotCore {
    /// Bot name (storage partition, logs).
    pub name: String,
    /// DocsGPT client.
    pub client: docsgpt::Client,
    /// The bot's agents.
    pub agents: Agents,
    /// Conversation state.
    pub storage: Arc<dyn Storage>,
    /// One turn at a time per scope.
    pub locks: ScopeLocks,
    /// Tunables.
    pub options: TurnOptions,
}

impl BotCore {
    /// A bot with default [`TurnOptions`].
    pub fn new(name: impl Into<String>, client: docsgpt::Client, agents: Agents, storage: Arc<dyn Storage>) -> Self {
        Self {
            name: name.into(),
            client,
            agents,
            storage,
            locks: ScopeLocks::default(),
            options: TurnOptions::default(),
        }
    }
}

/// Tunables for [`run_turn`].
#[derive(Debug, Clone)]
pub struct TurnOptions {
    /// Download files that tools produced and pass them to [`Surface::send_file`].
    pub deliver_files: bool,
    /// At most this many files per turn.
    pub max_files: usize,
}

impl Default for TurnOptions {
    fn default() -> Self {
        Self {
            deliver_files: true,
            max_files: 10,
        }
    }
}

/// A message to answer.
#[derive(Debug, Clone)]
pub struct Ask {
    /// Where the conversation lives.
    pub scope: Scope,
    /// The message as typed; a leading `#agent` picks the agent.
    pub text: String,
    /// DocsGPT attachment ids already uploaded for this message.
    pub attachments: Vec<String>,
    /// Answer with this agent and skip `#agent` routing (e.g. from a slash command).
    pub agent: Option<String>,
    /// Stops the turn when cancelled; a fresh token when `None`.
    pub cancel: Option<CancellationToken>,
    /// Where the chat's active agent is read and switched; `scope` when `None`.
    /// Use a wider scope when every thread should share one agent choice (e.g.
    /// a whole DM where each message starts its own thread).
    pub state_scope: Option<Scope>,
}

impl Ask {
    /// A message in `scope`.
    pub fn new(scope: Scope, text: impl Into<String>) -> Self {
        Self {
            scope,
            text: text.into(),
            attachments: Vec::new(),
            agent: None,
            cancel: None,
            state_scope: None,
        }
    }

    /// Attach uploaded files.
    pub fn attachments(mut self, ids: Vec<String>) -> Self {
        self.attachments = ids;
        self
    }

    /// Use this token to stop the turn.
    pub fn cancel(mut self, token: CancellationToken) -> Self {
        self.cancel = Some(token);
        self
    }
}

/// What [`run_turn`] did.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum TurnReport {
    /// DocsGPT was asked; the surface got `finish`.
    Answered {
        /// How it ended.
        outcome: Outcome,
        /// DocsGPT conversation.
        conversation_id: Option<String>,
        /// The answer's position in it.
        position: Option<u32>,
        /// The platform message holding the answer, from [`Surface::finish`].
        message_id: Option<String>,
    },
    /// `#tag` named no agent; the surface got a notice listing the agents.
    UnknownAgent {
        /// The tag as typed.
        tag: String,
    },
    /// A bare `#agent` made that agent the chat's active one; the surface got a notice.
    SwitchedAgent {
        /// The agent now active.
        agent: String,
    },
    /// Nothing to ask (no text and no attachments). The surface was not called.
    Empty,
}

/// Answer one message: route it to an agent, stream DocsGPT's answer through
/// `surface`, record the conversation and the answer's position, and deliver
/// files the agent's tools produced.
///
/// Turns in the same scope run one after another. Returns an error only when
/// the surface fails to begin or finish; DocsGPT failures end the turn with
/// [`Outcome::Failed`] instead.
pub async fn run_turn<S: Surface>(core: &BotCore, surface: &S, ask: Ask) -> Result<TurnReport> {
    let state_scope = ask.state_scope.as_ref().unwrap_or(&ask.scope);
    let state = core.storage.chat_state(state_scope).await.unwrap_or_else(|e| {
        tracing::warn!(error = %e, "could not read chat state");
        Default::default()
    });
    let (agent, question, tagged) = match &ask.agent {
        Some(name) => match core.agents.get(name) {
            Some(a) => (a.clone(), ask.text.trim().to_string(), true),
            None => return unknown_agent(core, surface, name).await,
        },
        None => match core.agents.route(&ask.text, state.active_agent.as_deref()) {
            Routed::Agent {
                agent,
                question,
                tagged,
            } => (agent.clone(), question, tagged),
            Routed::UnknownTag { tag, .. } => return unknown_agent(core, surface, &tag).await,
        },
    };
    if question.is_empty() && ask.attachments.is_empty() {
        if !tagged || ask.agent.is_some() {
            return Ok(TurnReport::Empty);
        }
        core.storage
            .update_chat_state(state_scope, StatePatch::active_agent(Some(&agent.name)))
            .await?;
        let about = agent
            .description
            .as_deref()
            .map(|d| format!(" ({d})"))
            .unwrap_or_default();
        surface
            .notice(&format!("Now answering with #{}{about}.", agent.name))
            .await?;
        return Ok(TurnReport::SwitchedAgent { agent: agent.name });
    }

    let _lock = core.locks.lock(&ask.scope.key()).await;
    let previous = core
        .storage
        .conversation(&ask.scope, &agent.name)
        .await
        .unwrap_or_else(|e| {
            tracing::warn!(error = %e, "could not read conversation; starting a new one");
            None
        });
    let turn = Turn {
        bot: core.name.clone(),
        scope: ask.scope.clone(),
        agent: agent.clone(),
        question: question.clone(),
        cancel: ask.cancel.clone().unwrap_or_default(),
    };
    let mut draft = surface.begin(&turn).await?;

    let mut req = AskRequest::new(&agent.api_key, &question).attachments(ask.attachments.clone());
    if let Some(c) = &previous {
        req = req.conversation_id(&c.id);
    }
    tracing::info!(bot = %core.name, scope = %ask.scope.key(), agent = %agent.name, conversation = ?previous.as_ref().map(|c| &c.id), attachments = ask.attachments.len(), "asking DocsGPT");

    let mut st = TurnState::default();
    let interval = surface.update_interval();
    match core.client.stream(&req).await {
        Err(e) => {
            tracing::warn!(error = %e, "DocsGPT stream failed to open");
            st.fail(describe(&e, false));
        }
        Ok(mut stream) => {
            let mut last_update = Instant::now() - interval;
            loop {
                let due = st.dirty.then(|| last_update + interval);
                tokio::select! {
                    biased;
                    _ = turn.cancel.cancelled() => {
                        st.outcome = Some(Outcome::Stopped);
                        break;
                    }
                    ev = stream.next() => match ev {
                        None => break,
                        Some(Err(e)) => {
                            tracing::warn!(error = %e, "DocsGPT stream failed");
                            st.fail(describe(&e, true));
                            break;
                        }
                        Some(Ok(ev)) => {
                            if st.handle(ev, core, &turn).await {
                                break;
                            }
                        }
                    },
                    _ = sleep_until(due), if due.is_some() => {}
                }
                if st.dirty && (st.urgent || Instant::now() >= last_update + interval) {
                    let (visible, _) = strip_images(&st.answer);
                    let visible = close_open_fence(&visible);
                    let progress = Progress {
                        answer: &visible,
                        raw: &st.answer,
                        status: st.status.as_deref(),
                        thinking: st.thinking && st.answer.is_empty(),
                    };
                    if let Err(e) = surface.update(&turn, &mut draft, progress).await {
                        tracing::warn!(error = %e, "draft update failed");
                    }
                    last_update = Instant::now();
                    st.dirty = false;
                    st.urgent = false;
                }
            }
        }
    }

    let (answer, mut images) = strip_images(st.answer.trim());
    for u in &st.outputs.image_urls {
        if !images.contains(u) {
            images.push(u.clone());
        }
    }
    let result = Final {
        answer,
        raw: st.answer.clone(),
        images,
        sources: st.sources,
        outcome: st.outcome.unwrap_or(Outcome::Complete),
        conversation_id: st.conversation_id.clone(),
        position: st.position,
    };
    tracing::info!(bot = %core.name, outcome = ?result.outcome, chars = result.answer.chars().count(), sources = result.sources.len(), files = st.outputs.artifacts.len(), images = result.images.len(), "turn finished");
    let message_id = surface.finish(&turn, draft, &result).await?;

    if let (Some(msg), Some(conv), Some(position)) = (&message_id, &result.conversation_id, result.position)
        && !result.answer.is_empty()
    {
        let r = MessageRef {
            agent: agent.name.clone(),
            conversation_id: conv.clone(),
            position,
        };
        if let Err(e) = core.storage.put_message_ref(&core.name, msg, &r).await {
            tracing::warn!(error = %e, "could not store message ref");
        }
    }

    if core.options.deliver_files
        && let Some(conv) = &result.conversation_id
    {
        for art in st.outputs.artifacts.iter().take(core.options.max_files) {
            let sent = match core.client.download_artifact(&agent.api_key, conv, &art.id).await {
                Ok(file) => surface.send_file(&turn, file).await,
                Err(e) => Err(e.into()),
            };
            if let Err(e) = sent {
                tracing::warn!(error = %e, artifact = %art.id, "could not deliver file");
                let _ = surface.notice(&format!("I couldn't attach {}.", art.filename)).await;
            }
        }
    }

    Ok(TurnReport::Answered {
        outcome: result.outcome,
        conversation_id: result.conversation_id,
        position: result.position,
        message_id,
    })
}

async fn unknown_agent<S: Surface>(core: &BotCore, surface: &S, tag: &str) -> Result<TurnReport> {
    let names = core
        .agents
        .iter()
        .map(|a| format!("#{}", a.name))
        .collect::<Vec<_>>()
        .join(", ");
    surface
        .notice(&format!("Unknown agent #{tag}. Available: {names}"))
        .await?;
    Ok(TurnReport::UnknownAgent { tag: tag.to_string() })
}

async fn sleep_until(at: Option<Instant>) {
    match at {
        Some(at) => tokio::time::sleep_until(at).await,
        None => std::future::pending().await,
    }
}

/// A short reason to show the user for a DocsGPT failure.
fn describe(e: &docsgpt::Error, mid_stream: bool) -> String {
    match e {
        docsgpt::Error::Timeout { .. } => "Timed out waiting for the assistant.".into(),
        docsgpt::Error::Http { status: 401 | 403, .. } => "The agent's API key was rejected.".into(),
        docsgpt::Error::Http { status: 429, .. } => {
            "The assistant is busy or over its usage limit. Try again later.".into()
        }
        docsgpt::Error::Http { status, .. } if *status < 500 => {
            format!("The assistant rejected the request ({status}).")
        }
        _ if mid_stream => "The connection to the assistant dropped.".into(),
        _ => "The assistant could not be reached.".into(),
    }
}

#[derive(Default)]
struct TurnState {
    answer: String,
    sources: Vec<Source>,
    outputs: ToolOutputs,
    status: Option<String>,
    thinking: bool,
    conversation_id: Option<String>,
    position: Option<u32>,
    /// This turn has been counted in its conversation.
    recorded: bool,
    outcome: Option<Outcome>,
    /// Something changed since the last update.
    dirty: bool,
    /// Show it now rather than at the next interval (tool status changes).
    urgent: bool,
}

impl TurnState {
    fn fail(&mut self, detail: String) {
        self.outcome = Some(Outcome::Failed { detail });
    }

    /// Apply one event; true when the stream is over.
    async fn handle(&mut self, ev: Event, core: &BotCore, turn: &Turn) -> bool {
        match ev {
            Event::Answer(d) => {
                self.answer.push_str(&d);
                self.dirty = true;
            }
            Event::Thought(_) => {
                if !self.thinking && self.answer.is_empty() {
                    self.thinking = true;
                    self.dirty = true;
                    self.urgent = true;
                }
            }
            Event::ToolCall(tc) => {
                if tc.is_completed() || tc.status == "error" || tc.status == "denied" {
                    self.outputs.merge(tc.outputs());
                    self.status = None;
                } else {
                    self.status = Some(tc.label());
                }
                self.dirty = true;
                self.urgent = true;
            }
            Event::ToolCalls(list) => {
                for tc in list {
                    self.outputs.merge(tc.outputs());
                }
                if self.status.take().is_some() {
                    self.dirty = true;
                }
            }
            Event::Source(s) => self.sources = s,
            Event::MessageId {
                conversation_id: Some(c),
                ..
            }
            | Event::ConversationId(c) => self.conversation(c, core, turn).await,
            Event::StructuredAnswer(s) => {
                if self.answer.trim().is_empty() {
                    self.answer = s;
                    self.dirty = true;
                }
            }
            Event::Notice(n) => tracing::info!(notice = %n, "DocsGPT notice"),
            Event::Error(e) => {
                tracing::warn!(error = %e, "DocsGPT error event");
                self.fail(truncate_chars(&e, 300));
                return true;
            }
            Event::End => return true,
            other => tracing::debug!(event = ?other, "unhandled DocsGPT event"),
        }
        false
    }

    /// The conversation id is known: count this turn in it, once.
    async fn conversation(&mut self, id: String, core: &BotCore, turn: &Turn) {
        if id.is_empty() || self.conversation_id.as_ref() == Some(&id) {
            return;
        }
        if !self.recorded {
            self.recorded = true;
            match core.storage.record_turn(&turn.scope, &turn.agent.name, &id).await {
                Ok(p) => self.position = p,
                Err(e) => tracing::warn!(error = %e, "could not record turn"),
            }
        } else if let Err(e) = core.storage.set_conversation(&turn.scope, &turn.agent.name, &id).await {
            tracing::warn!(error = %e, "could not store conversation");
        }
        self.conversation_id = Some(id);
    }
}

/// Send a 👍/👎 for the answer in platform message `message_id` (as returned
/// by [`Surface::finish`]). Returns false when the message holds no known answer.
pub async fn submit_feedback(core: &BotCore, message_id: &str, feedback: Feedback) -> Result<bool> {
    let Some(r) = core.storage.message_ref(&core.name, message_id).await? else {
        return Ok(false);
    };
    let Some(agent) = core.agents.get(&r.agent) else {
        return Ok(false);
    };
    core.client
        .feedback(&agent.api_key, &r.conversation_id, r.position, feedback)
        .await?;
    Ok(true)
}

/// How long message refs are worth keeping by default (for `prune_json`).
pub const MESSAGE_REF_TTL: Duration = Duration::from_secs(30 * 24 * 3600);
