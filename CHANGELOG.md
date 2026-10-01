# Changelog

## 0.2.0 (unreleased)

### docsgpt-bot 0.2.0

- **Breaking:** `Storage::record_turn` returns `Option<u32>` and `Conversation::turns` is `Option<u32>`. `None` means the count is unknown.
- SQLite files from the Telegram bot v2 hold conversations whose earlier answers were never counted. When the turn counter is added to such a file, those conversations are marked unknown rather than starting at 0, so 👍/👎 in them can't land on the wrong answer. Each chat is counted again from its next new conversation.
- `run_turn` records no position and no message ref while the count is unknown, so such answers offer no feedback.

### docsgpt 0.2.0

No changes; released together with `docsgpt-bot`.

## 0.1.0 (2026-10-01)

### docsgpt-bot 0.1.0

First release. Generalized from the Telegram bot:

- `config`: `AgentConfig`, `StorageConfig`, `ServerConfig`, `${VAR}` expansion, `load_toml`, `agents_from_env`.
- `Agents`: validation and `#name` / active / default routing.
- `storage`: string-keyed `Scope`, conversations with a turn counter (feedback's `question_index`), chat state, JSON records, and message refs. SQLite and memory backends share one contract test suite. Reads Telegram bot v2 SQLite files.
- `runtime`: `ScopeLocks`, `CancelRegistry` (guards unregister on drop), `Shutdown` that drains turns in progress.
- `run_turn` + `Surface`: the platform-neutral answer loop. It handles routing, the per-scope lock, streaming with throttled updates (flushed when the stream goes quiet), tool status, Stop, failures as outcomes (a partial answer keeps its error note), turn positions, message refs for `submit_feedback`, and tool files.
- `testing` feature: `FakeSurface` and the storage contract tests.
- `markdown`: fence-aware block splitting and clamping (the Telegram bot could cut inside a code block), image extraction, monospace tables.

### docsgpt 0.1.0

First release. Ported from the Telegram bot's DocsGPT client, with these changes:

- Typed `Error`s (HTTP status, feature disabled, timeout, transport, decode, too large, task failed).
- Retries: requests that are safe to repeat retry on connection errors, timeouts and 429/502/503/504; `/stream`, `/api/answer` and uploads retry only when the request never reached DocsGPT.
- `/stream` fails with `Error::Timeout` after a configurable silence (default 150 s).
- New `Client::feedback` (`POST /api/feedback`).
- `tts` sends the agent key.
- Fix: a multi-byte character split across two network chunks no longer turns into `�`.
- The Python-repr reader handles `\x` and `\U` escapes and control characters.
- `mock` feature: an in-process fake DocsGPT with request recording and fault injection.
