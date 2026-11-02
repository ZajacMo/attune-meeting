# Attune protocol contract

Everything below is stable public surface. Anything not listed here may change.

- HTTP base: `http://<host>:<port>` (default `127.0.0.1:8787`)
- MCP: stdio (`attune-mcp`) or streamable HTTP (`attune-mcp --http`)
- Encoding: UTF-8 everywhere, JSON for structured payloads, Markdown for the
  two documents.

## Vocabulary

| Term | Meaning |
| --- | --- |
| **project** | The unit of identity. A registry, its speakers and its sessions. |
| **session** | One meeting. Holds the audio, the utterances, the resolutions and the two documents. |
| **engine speaker** | A recording-local label (`spk_0`) produced by the diarization engine. Meaningless outside its session. |
| **identity** | A persistent person in a project (`spk_1a2b3c4d`). Carries the voiceprint and the display name. |
| **binding** | The mapping from one engine speaker to one identity, scoped to a session. |

## Speech pipeline

```
audio ──▶ normalize (16 kHz mono 16-bit WAV)
      ──▶ engine  ──▶ utterances [{start, end, text, engine_speaker, embedding?}]
      ──▶ identity ──▶ bindings + pending resolutions
      ──▶ report   ──▶ minutes + transcript
```

An utterance's `embedding` is the voiceprint. When the engine cannot produce
one, the speaker is still tracked, but only by name — never by voice.

## HTTP API

### `GET /health`

```json
{"status": "ok", "version": "0.1.0"}
```

### `GET /v1/projects`

```json
{"projects": ["my-project"]}
```

### `POST /v1/projects/{project_id}/sessions`

Request body (all fields optional):

```json
{"session_id": "ses_abc", "title": "产品周会"}
```

Response:

```json
{"project_id": "my-project", "session_id": "ses_abc", "title": "产品周会"}
```

### `POST /v1/projects/{project_id}/sessions/{session_id}/audio`

**Raw request body** — the audio bytes. Not multipart.

- `Content-Type`: `audio/wav`, `audio/mpeg`, `audio/mp4`, `audio/ogg`,
  `audio/flac`, `audio/webm` (used to pick a file suffix; conversion is done by
  ffmpeg regardless).
- Optional query parameters: `title`, `filename`.

Response — the full session state:

```json
{
  "session_id": "ses_abc",
  "project_id": "my-project",
  "utterances": [{"start": 0.0, "end": 3.2, "text": "…", "engine_speaker": "spk_0"}],
  "bindings": {"spk_0": "spk_1a2b3c4d"},
  "resolutions": [
    {
      "engine_speaker": "spk_1",
      "kind": "ambiguous",
      "identity_id": "spk_9f8e7d6c",
      "score": 0.71,
      "candidates": [{"identity_id": "spk_9f8e7d6c", "display_name": "李四", "score": 0.71}],
      "suggested_name": "发言人 2",
      "resolved": false
    }
  ],
  "minutes": {"title": "…", "summary": "…", "participants": [], "action_items": []}
}
```

Errors: `400` empty body, `404` unknown session, `422` audio not convertible,
`503` no usable ASR engine installed.

### `GET /v1/projects/{project_id}/sessions/{session_id}/transcript`

`?format=json` (default) or `?format=markdown`.

JSON: `{"session_id", "project_id", "markdown", "utterances", "bindings"}`.
Markdown: `text/markdown` — the full transcript with names and timestamps.

### `GET /v1/projects/{project_id}/sessions/{session_id}/minutes`

`?format=json` (default) or `?format=markdown`.

JSON: `{"session_id", "project_id", "markdown", "minutes"}`.
Markdown: `text/markdown` — summary, per-person outline, action items, roster.

### `GET /v1/projects/{project_id}/sessions/{session_id}/speakers`

```json
{
  "session_id": "ses_abc",
  "bindings": {"spk_0": "spk_1a2b3c4d"},
  "pending": [ /* resolution objects with resolved == false */ ],
  "resolved": [ /* resolution objects with resolved == true */ ],
  "speakers": [ /* every identity in the project */ ]
}
```

### `POST /v1/projects/{project_id}/sessions/{session_id}/speakers/resolve`

```json
{
  "decisions": [
    {"engine_speaker": "spk_0", "action": "assign", "identity_id": "spk_1a2b3c4d", "display_name": "张三"},
    {"engine_speaker": "spk_1", "action": "merge",  "identity_id": "spk_9f8e7d6c"},
    {"engine_speaker": "spk_2", "action": "rename", "display_name": "王五"}
  ]
}
```

Actions:

| Action | Effect |
| --- | --- |
| `assign` | Bind this engine speaker to `identity_id`. Any provisional identity created for it is discarded. The target is marked confirmed. |
| `merge` | Fold the provisional identity into `identity_id`. Voiceprints, appearance counts and the old name (kept as an alias) all move across. |
| `rename` | Keep the speaker separate, but call them `display_name`. |

The response is the updated session state; both documents are re-rendered.

### `GET /v1/projects/{project_id}/speakers`

```json
{
  "project_id": "my-project",
  "high_threshold": 0.68,
  "low_threshold": 0.53,
  "speakers": [{"identity_id": "…", "display_name": "…", "aliases": [], "appearances": 7, "total_seconds": 512.4, "confirmed": true}]
}
```

## MCP tools

| Tool | Arguments | Returns |
| --- | --- | --- |
| `create_session` | `project_id`, `title=""`, `session_id=""` | `{project_id, session_id, title}` |
| `transcribe_audio` | `project_id`, `audio_path`, `session_id=""`, `title=""` | full session state |
| `get_transcript` | `project_id`, `session_id` | `{session_id, markdown, utterances}` |
| `get_minutes` | `project_id`, `session_id` | `{session_id, markdown, minutes}` |
| `list_speakers` | `project_id` | thresholds + every identity |
| `get_pending_speaker_decisions` | `project_id`, `session_id` | `{session_id, pending[]}` |
| `resolve_speaker` | `project_id`, `session_id`, `engine_speaker`, `action`, `identity_id=""`, `display_name=""` | `{session_id, engine_speaker, identity_id, pending[]}` |

## The merge-or-create flow

A host that renders speaker identity needs exactly this loop:

1. Call `transcribe_audio` (or upload to `/audio`).
2. Read `resolutions` where `resolved == false`.
3. For each one, show the user:
   - `suggested_name` — a guess from what was said, if one was found;
   - `candidates` — known identities with their similarity scores.
4. Send the answer back with `assign`, `merge` or `rename`.
5. Re-read the documents — they are re-rendered with the new names.

A host that does not want to prompt at all can ignore `resolutions` entirely:
the documents are still produced, with placeholder names.

## Guarantees and non-guarantees

**Guaranteed**

- An identity is never reassigned silently between the ambiguous and new
  bands. Anything ambiguous stays a question.
- Re-processing the same session with the same binding produces the same
  names.
- Documents are always generated together and always agree with each other.

**Not guaranteed**

- Correct diarization. That is the engine's job.
- Correct thresholds. Calibrate them on your own audio.
- That a spoken name is the speaker's own name. The suggestion is a hint.
