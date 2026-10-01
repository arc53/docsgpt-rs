//! `run_turn` end to end: the mock DocsGPT on one side, a recording surface on the other.

use std::sync::Arc;
use std::time::{Duration, Instant};

use docsgpt::mock::{MockDocsGpt, Step, StreamReply, answer_steps, ev, reply_text, sse};
use docsgpt::{Feedback, RetryPolicy, Source};
use docsgpt_bot::storage::memory::MemoryStorage;
use docsgpt_bot::testing::{FakeSurface, Recorded};
use docsgpt_bot::{
    AgentConfig, Agents, Ask, BotCore, CancelRegistry, Outcome, Scope, StatePatch, TurnReport, run_turn,
    submit_feedback,
};
use serde_json::json;
use tokio_util::sync::CancellationToken;

fn core(docs: &MockDocsGpt, agents: &[(&str, &str)]) -> Arc<BotCore> {
    let client = docsgpt::Client::builder(&docs.url)
        .retry(RetryPolicy::none())
        .stream_idle_timeout(Duration::from_secs(2))
        .build()
        .unwrap();
    let agents = Agents::new(agents.iter().map(|(n, k)| AgentConfig::new(*n, *k)).collect()).unwrap();
    Arc::new(BotCore::new("bot", client, agents, Arc::new(MemoryStorage::default())))
}

fn scope() -> Scope {
    Scope::new("bot", "C1", "T1")
}

fn answered(r: TurnReport) -> (Outcome, Option<String>, Option<u32>, Option<String>) {
    match r {
        TurnReport::Answered {
            outcome,
            conversation_id,
            position,
            message_id,
        } => (outcome, conversation_id, position, message_id),
        other => panic!("{other:?}"),
    }
}

#[tokio::test]
async fn streams_finishes_and_continues_the_conversation() {
    let docs = MockDocsGpt::start().await;
    docs.on_stream(|_| {
        sse(answer_steps(
            "Hello there, friend.",
            "conv-1",
            &[("Guide", "https://g")],
        ))
    });
    let core = core(&docs, &[("default", "key-d")]);
    let surface = FakeSurface::new();

    let r = answered(run_turn(&core, &surface, Ask::new(scope(), "hi")).await.unwrap());
    assert_eq!(
        r,
        (Outcome::Complete, Some("conv-1".into()), Some(0), Some("msg-1".into()))
    );

    let calls = surface.calls();
    assert_eq!(
        calls[0],
        Recorded::Begin {
            agent: "default".into(),
            question: "hi".into()
        }
    );
    assert!(!surface.updates().is_empty());
    let fin = surface.finished();
    assert_eq!(fin.answer, "Hello there, friend.");
    assert_eq!(
        fin.sources,
        vec![Source {
            title: "Guide".into(),
            url: Some("https://g".into())
        }]
    );
    assert!(fin.can_rate());
    let body = docs.rec.last("/stream").unwrap().body;
    assert_eq!(body, json!({"question": "hi", "api_key": "key-d"}));

    // The next turn continues the conversation and is position 1.
    let surface = FakeSurface::new().message_id(Some("msg-2"));
    let r = answered(run_turn(&core, &surface, Ask::new(scope(), "more")).await.unwrap());
    assert_eq!(r.2, Some(1));
    assert_eq!(docs.rec.last("/stream").unwrap().body["conversation_id"], "conv-1");

    // Feedback goes to the right answer.
    assert!(submit_feedback(&core, "msg-2", Feedback::Like).await.unwrap());
    assert_eq!(
        docs.rec.last("/api/feedback").unwrap().body,
        json!({"feedback": "like", "conversation_id": "conv-1", "question_index": 1, "api_key": "key-d"})
    );
    assert!(!submit_feedback(&core, "unknown", Feedback::Like).await.unwrap());
}

#[tokio::test]
async fn new_conversation_restarts_positions_per_agent_and_thread() {
    let docs = MockDocsGpt::start().await;
    let core = core(&docs, &[("support", "k1"), ("sales", "k2")]);
    docs.on_stream(|body| {
        let conv = format!("conv-{}", body["api_key"].as_str().unwrap());
        reply_text("ok", &conv)
    });
    let s = FakeSurface::new();
    assert_eq!(
        answered(run_turn(&core, &s, Ask::new(scope(), "a")).await.unwrap()).2,
        Some(0)
    );
    assert_eq!(
        answered(run_turn(&core, &s, Ask::new(scope(), "#sales b")).await.unwrap()).2,
        Some(0)
    );
    assert_eq!(
        answered(run_turn(&core, &s, Ask::new(scope(), "c")).await.unwrap()).2,
        Some(1)
    );
    let other_thread = Scope::new("bot", "C1", "T2");
    assert_eq!(
        answered(run_turn(&core, &s, Ask::new(other_thread, "d")).await.unwrap()).2,
        Some(0)
    );
    // Server started a new conversation (e.g. the old one was deleted): positions restart.
    docs.on_stream(|_| reply_text("ok", "conv-new"));
    assert_eq!(
        answered(run_turn(&core, &s, Ask::new(scope(), "e")).await.unwrap()).2,
        Some(0)
    );
}

#[tokio::test]
async fn agent_routing_switching_and_unknown_tags() {
    let docs = MockDocsGpt::start().await;
    let core = core(&docs, &[("support", "k-support"), ("sales", "k-sales")]);

    let s = FakeSurface::new();
    run_turn(&core, &s, Ask::new(scope(), "#sales price?")).await.unwrap();
    let body = docs.rec.last("/stream").unwrap().body;
    assert_eq!(
        (body["api_key"].as_str(), body["question"].as_str()),
        (Some("k-sales"), Some("price?"))
    );

    let s = FakeSurface::new();
    let r = run_turn(&core, &s, Ask::new(scope(), "#nope hello")).await.unwrap();
    assert_eq!(r, TurnReport::UnknownAgent { tag: "nope".into() });
    assert_eq!(s.notices(), vec!["Unknown agent #nope. Available: #support, #sales"]);
    assert_eq!(docs.rec.count("/stream"), 1);

    let s = FakeSurface::new();
    let r = run_turn(&core, &s, Ask::new(scope(), "#sales")).await.unwrap();
    assert_eq!(r, TurnReport::SwitchedAgent { agent: "sales".into() });
    assert_eq!(s.notices(), vec!["Now answering with #sales."]);
    run_turn(&core, &FakeSurface::new(), Ask::new(scope(), "and now?"))
        .await
        .unwrap();
    assert_eq!(docs.rec.last("/stream").unwrap().body["api_key"], "k-sales");

    assert_eq!(
        run_turn(&core, &FakeSurface::new(), Ask::new(scope(), "   "))
            .await
            .unwrap(),
        TurnReport::Empty
    );

    // A forced agent skips routing.
    core.storage
        .update_chat_state(&scope(), StatePatch::active_agent(None))
        .await
        .unwrap();
    let mut ask = Ask::new(scope(), "#hashtag question");
    ask.agent = Some("sales".into());
    run_turn(&core, &FakeSurface::new(), ask).await.unwrap();
    let body = docs.rec.last("/stream").unwrap().body;
    assert_eq!(
        (body["api_key"].as_str(), body["question"].as_str()),
        (Some("k-sales"), Some("#hashtag question"))
    );
}

#[tokio::test]
async fn throttles_updates_and_flushes_when_the_stream_goes_quiet() {
    let docs = MockDocsGpt::start().await;
    docs.on_stream(|_| {
        let mut steps = vec![ev::step(ev::message_id("m", "c"))];
        steps.extend((0..40).map(|i| ev::step(ev::answer(&format!("w{i} ")))));
        steps.push(Step::Hang(Duration::from_millis(600)));
        steps.push(ev::step(ev::answer("tail")));
        steps.push(ev::step(ev::end()));
        sse(steps)
    });
    let core = core(&docs, &[("default", "k")]);
    let surface = FakeSurface::new().interval(Duration::from_millis(200));
    run_turn(&core, &surface, Ask::new(scope(), "q")).await.unwrap();

    let updates = surface.updates();
    assert!(updates.len() <= 5, "{} updates: throttling failed", updates.len());
    // During the quiet 600 ms the full burst was shown, before "tail" arrived.
    assert!(
        updates.iter().any(|(a, _)| a.ends_with("w39") && !a.contains("tail")),
        "{updates:?}"
    );
    assert!(surface.finished().answer.ends_with("tail"));
}

#[tokio::test]
async fn stop_ends_the_turn_with_the_partial_answer() {
    let docs = MockDocsGpt::start().await;
    docs.on_stream(|_| {
        sse(vec![
            ev::step(ev::message_id("m", "conv-s")),
            ev::step(ev::answer("Partial ")),
            Step::Sleep(Duration::from_secs(5)),
            ev::step(ev::answer("never")),
            ev::step(ev::end()),
        ])
    });
    let core = core(&docs, &[("default", "k")]);
    let registry = CancelRegistry::default();
    let surface = Arc::new(FakeSurface::new().stop_with(registry.clone(), "draft-1"));

    let started = Instant::now();
    let (c, s) = (core.clone(), surface.clone());
    let task = tokio::spawn(async move { run_turn(&c, &*s, Ask::new(scope(), "q")).await });
    surface
        .wait_update(|a, _| a.contains("Partial"), Duration::from_secs(3))
        .await;
    assert!(registry.cancel("draft-1"));
    let r = answered(task.await.unwrap().unwrap());
    assert!(
        started.elapsed() < Duration::from_secs(3),
        "stop took {:?}",
        started.elapsed()
    );
    assert_eq!(r.0, Outcome::Stopped);
    let fin = surface.finished();
    assert_eq!(fin.answer, "Partial");
    assert_eq!(fin.display_text(), "Partial\n\n_Stopped._");
    assert!(registry.is_empty(), "the turn's Stop registration is dropped");
}

#[tokio::test]
async fn caller_token_stops_before_any_answer() {
    let docs = MockDocsGpt::start().await;
    docs.on_stream(|_| sse(vec![Step::Sleep(Duration::from_secs(5)), ev::step(ev::end())]));
    let core = core(&docs, &[("default", "k")]);
    let token = CancellationToken::new();
    let t = token.clone();
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(100)).await;
        t.cancel();
    });
    let surface = FakeSurface::new();
    let r = answered(
        run_turn(&core, &surface, Ask::new(scope(), "q").cancel(token))
            .await
            .unwrap(),
    );
    assert_eq!(r.0, Outcome::Stopped);
    assert_eq!(surface.finished().display_text(), "Stopped.");
}

#[tokio::test]
async fn failures_become_outcomes() {
    let docs = MockDocsGpt::start().await;
    let core = core(&docs, &[("default", "k")]);

    docs.on_stream(|_| StreamReply::Http(401, "{\"error\": \"Unauthorized\"}".into()));
    let s = FakeSurface::new();
    let r = answered(run_turn(&core, &s, Ask::new(scope(), "q")).await.unwrap());
    assert_eq!(
        r.0,
        Outcome::Failed {
            detail: "The agent's API key was rejected.".into()
        }
    );
    assert_eq!(
        s.finished().display_text(),
        "Sorry, I couldn't get an answer right now.\n\nThe agent's API key was rejected."
    );
    assert!(!s.finished().can_rate());

    // A partial answer, then an error: the user sees both (the Telegram bot dropped the error).
    docs.on_stream(|_| {
        sse(vec![
            ev::step(ev::message_id("m", "conv-e")),
            ev::step(ev::answer("Half an ")),
            ev::step(ev::error("LLM overloaded")),
        ])
    });
    let s = FakeSurface::new();
    let r = answered(run_turn(&core, &s, Ask::new(scope(), "q")).await.unwrap());
    assert_eq!(
        r.0,
        Outcome::Failed {
            detail: "LLM overloaded".into()
        }
    );
    assert_eq!(
        s.finished().display_text(),
        "Half an\n\n_The answer was cut short: LLM overloaded_"
    );
    // The server reserved the turn, so it counts.
    assert_eq!(r.2, Some(0));

    // Silence beyond the client's idle timeout.
    docs.on_stream(|_| sse(vec![ev::step(ev::answer("x")), Step::Hang(Duration::from_secs(5))]));
    let s = FakeSurface::new();
    let r = answered(run_turn(&core, &s, Ask::new(scope(), "q")).await.unwrap());
    assert_eq!(
        r.0,
        Outcome::Failed {
            detail: "Timed out waiting for the assistant.".into()
        }
    );
}

#[tokio::test]
async fn tool_status_files_and_images() {
    let docs = MockDocsGpt::start().await;
    docs.add_artifact("a1", "report.csv", "text/csv", &b"a,b\n1,2\n"[..]);
    let img = docs.image_url("cat.png");
    let img2 = img.clone();
    docs.on_stream(move |_| {
        sse(vec![
            ev::step(ev::message_id("m", "conv-t")),
            ev::step(ev::tool_call(
                json!({"tool_name": "code_executor", "call_id": "c1", "action_name": "run", "status": "pending"}),
            )),
            Step::Sleep(Duration::from_millis(100)),
            ev::step(ev::tool_call(
                json!({"tool_name": "code_executor", "call_id": "c1", "action_name": "run", "status": "completed",
                "artifacts": [{"id": "a1", "filename": "report.csv"}, {"id": "missing", "filename": "lost.txt"}]}),
            )),
            ev::step(ev::tool_call(
                json!({"tool_name": "imagegen", "call_id": "c2", "status": "completed",
                "result": format!("{{'image_urls': ['{img2}', 'https://x/second.png']}}")}),
            )),
            ev::step(ev::answer(&format!("Here you go:\n\n![cat]({img2})\n\nDone."))),
            ev::step(ev::end()),
        ])
    });
    let core = core(&docs, &[("default", "k")]);
    let surface = FakeSurface::new();
    run_turn(&core, &surface, Ask::new(scope(), "make things"))
        .await
        .unwrap();

    assert!(
        surface
            .updates()
            .iter()
            .any(|(_, s)| s.as_deref() == Some("Running code")),
        "{:?}",
        surface.updates()
    );
    let fin = surface.finished();
    assert_eq!(fin.answer, "Here you go:\n\nDone.");
    assert_eq!(fin.images, vec![img, "https://x/second.png".to_string()]);
    assert_eq!(surface.files(), vec!["report.csv"]);
    assert_eq!(surface.notices(), vec!["I couldn't attach lost.txt."]);
    let dl = docs.rec.calls("/api/artifacts/download");
    assert_eq!(dl[0].query["conversation_id"], "conv-t");
    // Files come after finish.
    let calls = surface.calls();
    let fin_at = calls.iter().position(|c| matches!(c, Recorded::Finish(_))).unwrap();
    let file_at = calls.iter().position(|c| matches!(c, Recorded::File { .. })).unwrap();
    assert!(file_at > fin_at);
}

#[tokio::test]
async fn failing_file_upload_becomes_a_notice() {
    let docs = MockDocsGpt::start().await;
    docs.add_artifact("a1", "r.txt", "text/plain", &b"x"[..]);
    docs.on_stream(|_| {
        sse(vec![
            ev::step(ev::message_id("m", "c")),
            ev::step(ev::tool_call(
                json!({"tool_name": "t", "status": "completed", "artifact_id": "a1"}),
            )),
            ev::step(ev::answer("ok")),
            ev::step(ev::end()),
        ])
    });
    let core = core(&docs, &[("default", "k")]);
    let surface = FakeSurface::new().fail_files();
    run_turn(&core, &surface, Ask::new(scope(), "q")).await.unwrap();
    assert_eq!(surface.notices(), vec!["I couldn't attach a1."]);
}

#[tokio::test]
async fn attachments_are_sent_and_thinking_shows_before_text() {
    let docs = MockDocsGpt::start().await;
    docs.on_stream(|_| {
        sse(vec![
            ev::step(ev::message_id("m", "c")),
            ev::step(ev::thought("hmm")),
            Step::Sleep(Duration::from_millis(100)),
            ev::step(ev::answer("A")),
            ev::step(ev::end()),
        ])
    });
    let core = core(&docs, &[("default", "k")]);
    let surface = FakeSurface::new();
    run_turn(
        &core,
        &surface,
        Ask::new(scope(), "about this").attachments(vec!["att-1".into()]),
    )
    .await
    .unwrap();
    assert_eq!(docs.rec.last("/stream").unwrap().body["attachments"], json!(["att-1"]));
    assert!(
        matches!(surface.calls()[1], Recorded::Update { thinking: true, .. }),
        "{:?}",
        surface.calls()
    );

    // An attachment with no text still asks.
    let s = FakeSurface::new();
    assert!(matches!(
        run_turn(&core, &s, Ask::new(scope(), "").attachments(vec!["att-2".into()]))
            .await
            .unwrap(),
        TurnReport::Answered { .. }
    ));
}

#[tokio::test]
async fn turns_in_one_scope_run_one_at_a_time() {
    let docs = MockDocsGpt::start().await;
    docs.on_stream(|_| {
        sse(vec![
            ev::step(ev::answer("slow")),
            Step::Sleep(Duration::from_millis(300)),
            ev::step(ev::end()),
        ])
    });
    let core = core(&docs, &[("default", "k")]);
    let (c1, c2) = (core.clone(), core.clone());
    let a = tokio::spawn(async move { run_turn(&c1, &FakeSurface::new(), Ask::new(scope(), "one")).await });
    tokio::time::sleep(Duration::from_millis(50)).await;
    let b = tokio::spawn(async move { run_turn(&c2, &FakeSurface::new(), Ask::new(scope(), "two")).await });
    a.await.unwrap().unwrap();
    b.await.unwrap().unwrap();
    let calls = docs.rec.calls("/stream");
    assert!(
        calls[1].at.duration_since(calls[0].at) >= Duration::from_millis(250),
        "second turn did not wait"
    );
}

#[tokio::test]
async fn state_scope_holds_the_agent_choice_across_threads() {
    let docs = MockDocsGpt::start().await;
    let core = core(&docs, &[("support", "k-support"), ("sales", "k-sales")]);
    let dm = Scope::new("bot", "T:D1", "0");
    let in_thread = |t: &str| {
        let mut a = Ask::new(Scope::new("bot", "T:D1", t), "");
        a.state_scope = Some(dm.clone());
        a
    };

    let mut switch = in_thread("1.1");
    switch.text = "#sales".into();
    assert_eq!(
        run_turn(&core, &FakeSurface::new(), switch).await.unwrap(),
        TurnReport::SwitchedAgent { agent: "sales".into() }
    );
    assert_eq!(
        core.storage.chat_state(&dm).await.unwrap().active_agent.as_deref(),
        Some("sales")
    );

    // A new thread in the same DM uses the DM's agent.
    let mut ask = in_thread("2.2");
    ask.text = "price?".into();
    run_turn(&core, &FakeSurface::new(), ask).await.unwrap();
    assert_eq!(docs.rec.last("/stream").unwrap().body["api_key"], "k-sales");
    // Without a state scope the thread's own (empty) state applies.
    run_turn(
        &core,
        &FakeSurface::new(),
        Ask::new(Scope::new("bot", "T:D1", "3.3"), "hi"),
    )
    .await
    .unwrap();
    assert_eq!(docs.rec.last("/stream").unwrap().body["api_key"], "k-support");
}

#[tokio::test]
async fn progress_raw_only_grows() {
    let docs = MockDocsGpt::start().await;
    docs.on_stream(|_| {
        sse(vec![
            ev::step(ev::message_id("m", "c")),
            ev::step(ev::answer("See ![x](https://i/x.png) and\n```py\nprint(1)")),
            Step::Sleep(Duration::from_millis(80)),
            ev::step(ev::answer("\n```\nDone")),
            ev::step(ev::end()),
        ])
    });
    let core = core(&docs, &[("default", "k")]);
    let surface = FakeSurface::new().interval(Duration::from_millis(20));
    run_turn(&core, &surface, Ask::new(scope(), "q")).await.unwrap();
    let raws: Vec<String> = surface
        .calls()
        .into_iter()
        .filter_map(|c| match c {
            Recorded::Update { raw, .. } => Some(raw),
            _ => None,
        })
        .collect();
    assert!(raws.len() >= 2, "{raws:?}");
    for w in raws.windows(2) {
        assert!(w[1].starts_with(&w[0]), "raw must only grow: {w:?}");
    }
    assert_eq!(raws[0], "See ![x](https://i/x.png) and\n```py\nprint(1)");
    // The display form closes the fence and drops the image.
    let first = &surface.updates()[0].0;
    assert!(first.ends_with("print(1)\n```") && !first.contains("!["), "{first}");
}

#[tokio::test]
async fn final_raw_and_note() {
    let docs = MockDocsGpt::start().await;
    docs.on_stream(|_| {
        sse(vec![
            ev::step(ev::message_id("m", "c")),
            ev::step(ev::answer(" Hi ![a](https://i/a.png) ")),
            ev::step(ev::error("boom")),
        ])
    });
    let core = core(&docs, &[("default", "k")]);
    let s = FakeSurface::new();
    run_turn(&core, &s, Ask::new(scope(), "q")).await.unwrap();
    let f = s.finished();
    assert_eq!(f.raw, " Hi ![a](https://i/a.png) ");
    assert_eq!(f.answer, "Hi");
    assert_eq!(f.note().as_deref(), Some("_The answer was cut short: boom_"));
    assert_eq!(f.display_text(), "Hi\n\n_The answer was cut short: boom_");
}

#[tokio::test]
async fn upgraded_conversations_get_no_feedback_until_a_new_one_starts() {
    let docs = MockDocsGpt::start().await;
    docs.on_stream(|body| reply_text("ok", body["conversation_id"].as_str().unwrap_or("conv-new")));
    // A Telegram v2 file: a conversation with uncounted answers.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tg.db");
    {
        let c = rusqlite::Connection::open(&path).unwrap();
        c.execute_batch(
            "CREATE TABLE conversations (scope TEXT NOT NULL, agent TEXT NOT NULL, conversation_id TEXT NOT NULL,
               updated_at INTEGER NOT NULL, PRIMARY KEY (scope, agent));
             INSERT INTO conversations VALUES ('bot:C1:T1', 'default', 'conv-old', 1);",
        )
        .unwrap();
    }
    let storage = docsgpt_bot::storage::sqlite::SqliteStorage::open(path.to_str().unwrap())
        .await
        .unwrap();
    let mut core = BotCore::new(
        "bot",
        docsgpt::Client::new(&docs.url).unwrap(),
        Agents::new(vec![AgentConfig::new("default", "k")]).unwrap(),
        Arc::new(storage),
    );
    core.options = Default::default();

    let s = FakeSurface::new();
    let r = answered(run_turn(&core, &s, Ask::new(scope(), "continue")).await.unwrap());
    assert_eq!(
        docs.rec.last("/stream").unwrap().body["conversation_id"],
        "conv-old",
        "the conversation continues"
    );
    assert_eq!((r.1.as_deref(), r.2), (Some("conv-old"), None));
    assert!(!s.finished().can_rate());
    assert_eq!(core.storage.message_ref("bot", "msg-1").await.unwrap(), None);

    // After /new the next conversation is counted from 0.
    core.storage.clear_conversation(&scope(), "default").await.unwrap();
    let r = answered(
        run_turn(&core, &FakeSurface::new(), Ask::new(scope(), "fresh"))
            .await
            .unwrap(),
    );
    assert_eq!((r.1.as_deref(), r.2), (Some("conv-new"), Some(0)));
}
