# Attune

**A meeting assistant that knows who is speaking — and remembers them next time.**

Attune turns a meeting recording into two documents:

1. **Minutes** — a per-person outline of what each participant said, plus the
   concrete action items that came out of the meeting.
2. **A full transcript** — every utterance, timestamped and attributed to a
   real name.

The hard part is not the transcription. It is the *identity*: diarization
engines can only say "speaker 0", "speaker 1" — labels that mean nothing
outside that one recording. Attune keeps a **project-scoped registry of
voiceprints** so the same person is recognised in next week's meeting, and it
never guesses silently when the match is not clear: it asks.

---

## Why this exists

Every diarization library ships the same limitation. `pyannote.audio` emits
`speaker_0` / `speaker_1`. FunASR emits a local `spk` index — its own
documentation says *"Indices are local to a recording, not known-person
identities."* You get turn structure, not people.

Attune adds the missing half:

| Stage | What Attune does |
| --- | --- |
| **Diarize** | Splits the audio into turns and groups them by voiceprint (FunASR + CAM++ by default, WhisperX optional). |
| **Match** | Compares each voiceprint against the project's identity registry. |
| **Decide** | Confident match → reuse the identity silently. Close match → **ask the user**: merge into an existing identity, or keep it as a new one. |
| **Remember** | Enrols the voiceprint under the project so the next meeting recognises the same person automatically. |
| **Report** | Produces the minutes and the transcript with real names. |

The registry lives on the **project**, not the session, which is exactly what
makes cross-session recognition work.

---

## Install

```bash
pip install attune-meeting         # core library only
pip install "attune-meeting[server]"       # + MCP server and HTTP API
pip install "attune-meeting[asr-funasr]"   # + Chinese ASR and speaker clustering
```

The ASR extras are deliberately optional: the core library, the identity
registry and the document generation stay light enough to embed anywhere.

### Chinese-first by default

FunASR gives you ASR, VAD, punctuation and speaker clustering in one pass:

```python
from attune.core.engines import FunASREngine

engine = FunASREngine(model="paraformer-zh", device="cpu")   # fsmn-vad + ct-punc + cam++
```

---

## Quick start

```bash
# Transcribe a recording, resolve speakers, print the minutes
attune transcribe meeting.wav --project my-project --title "产品周会"

# Inspect who the project knows about
attune speakers my-project

# Answer a merge-or-create prompt
attune resolve my-project ses_abc123 \
  --engine-speaker spk_0 --action rename --display-name "张三"

# Print the two documents
attune minutes   my-project ses_abc123
attune transcript my-project ses_abc123
```

From Python:

```python
from attune.core.engines import FunASREngine
from attune.core.identity import SpeakerRegistry
from attune.core.pipeline import MeetingPipeline, SpeakerDecision
from attune.core.store import Workspace

workspace = Workspace("data")
registry = workspace.load_registry("my-project")
pipeline = MeetingPipeline(FunASREngine(), registry)

result = pipeline.process("meeting.wav", session_id="ses_1", title="产品周会")

# Anything the user still has to decide
for item in result.resolutions:
    if not item.resolved:
        print(item.engine_speaker, item.kind, item.suggested_name,
              [(c.display_name, round(c.score, 2)) for c in item.candidates])

pipeline.resolve(result, [
    SpeakerDecision(engine_speaker="spk_0", action="rename", display_name="张三"),
])

workspace.save_session(result)
workspace.save_registry(registry)
print(result.minutes_markdown)
```

---

## Identity model

```json
{
  "version": 1,
  "project_id": "my-project",
  "high_threshold": 0.68,
  "low_threshold": 0.53,
  "profiles": [
    {
      "identity_id": "spk_1a2b3c4d",
      "display_name": "张三",
      "aliases": ["张经理"],
      "embeddings": [[0.12, -0.44, "..."]],
      "appearances": 7,
      "total_seconds": 512.4,
      "confirmed": true
    }
  ],
  "bindings": [
    {"session_id": "ses_1", "engine_speaker": "spk_0", "identity_id": "spk_1a2b3c4d"}
  ]
}
```

Matching compares a fresh voiceprint against every known identity and returns
one of three outcomes:

- **known** — similarity ≥ `high_threshold` → the identity is reused silently.
- **ambiguous** — similarity between the thresholds → the user is asked to
  merge or create. Attune does not guess here.
- **new** — similarity below `low_threshold` → a new identity is created and
  queued for naming.

> **Thresholds are a starting point, not a law.** The shipped defaults
> (0.68 / 0.53) were measured on real Mandarin meeting audio — see
> [`docs/voiceprint-calibration.md`](docs/voiceprint-calibration.md) for the
> data, the method and the reproducible scripts. Your microphones and rooms
> change the voiceprint, so re-measure on your own audio before trusting them.

---

## Interfaces

The same operations are exposed two ways, so any host can pick the one it
already speaks.

### MCP server

```bash
attune-mcp --workspace ./data          # stdio
attune-mcp --http --port 8788           # streamable HTTP
```

Tools: `create_session`, `transcribe_audio`, `get_transcript`, `get_minutes`,
`list_speakers`, `get_pending_speaker_decisions`, `resolve_speaker`.

### HTTP API

```bash
attune serve --host 127.0.0.1 --port 8787
```

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Liveness |
| `POST` | `/v1/projects/{project}/sessions` | Create a session |
| `POST` | `/v1/projects/{project}/sessions/{session}/audio` | Upload audio (raw body) and process it |
| `GET` | `/v1/projects/{project}/sessions/{session}/transcript` | Transcript (`?format=markdown` for Markdown) |
| `GET` | `/v1/projects/{project}/sessions/{session}/minutes` | Minutes (`?format=markdown` for Markdown) |
| `GET` | `/v1/projects/{project}/sessions/{session}/speakers` | Bindings plus pending decisions |
| `POST` | `/v1/projects/{project}/sessions/{session}/speakers/resolve` | Answer merge-or-create |
| `GET` | `/v1/projects/{project}/speakers` | Everything the project knows |

Full contract: [docs/protocol.md](docs/protocol.md).

---

## Joining an online meeting

Attune does not ship a meeting bot. It uses [Vexa](https://github.com/Vexa-ai/vexa)
(Apache-2.0) to get the audio, and does the rest itself.

```bash
export VEXA_API_KEY=vxa_...

attune vexa send --url https://meet.google.com/abc-defg-hij --language zh
attune vexa transcript --url https://meet.google.com/abc-defg-hij
attune vexa stop --url https://meet.google.com/abc-defg-hij
```

Google Meet, Microsoft Teams, Zoom and Jitsi URLs are all understood. Vexa's own
transcription is left **off** by default — Attune transcribes and diarizes
itself, so paying for Vexa's transcription unit would duplicate the work. Turn
it on with `--transcribe` if you want a cross-check.

---

## Framework guides

- [Cherry Studio](docs/integration-cherry-studio.md) — add as an MCP server
- [Hermes Agent](docs/integration-hermes.md) — Python package and HTTP API
- [OpenClaw](docs/integration-openclaw.md) — skill package shape
- [CyberColleague](docs/integration-cybercolleague.md) — desktop workbench

---

## Development

```bash
uv sync --extra dev --extra server
uv run pytest
uv run ruff check src tests
uv run ruff format --check src tests
```

## License

Apache-2.0. See [LICENSE](LICENSE).

Model weights used by the optional ASR extras carry their own licenses —
check the model card before shipping a product that bundles them.
