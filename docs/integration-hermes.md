# Attune in Hermes Agent

Hermes is a Python runtime, so the most direct route is the Python package.
The HTTP API is there when you want the agent to talk to a long-running service
instead.

## Option A — Python package (recommended)

```bash
pip install attune
```

```python
from attune.core.engines import FunASREngine
from attune.core.identity import SpeakerRegistry
from attune.core.pipeline import MeetingPipeline, SpeakerDecision
from attune.core.store import Workspace

workspace = Workspace("~/.hermes/attune")
registry = workspace.load_registry("team-standup")
pipeline = MeetingPipeline(FunASREngine(), registry)

result = pipeline.process("/recordings/standup.wav", session_id="ses_001")

# Surface anything the user must decide before writing the notes
for item in result.resolutions:
    if not item.resolved:
        print(f"{item.engine_speaker}: {item.kind}")
        for candidate in item.candidates[:3]:
            print(f"   {candidate.display_name} ({candidate.score:.2f})")
        print(f"   suggestion: {item.suggested_name}")
```

Then answer the prompts:

```python
pipeline.resolve(result, [
    SpeakerDecision(engine_speaker="spk_0", action="merge", identity_id="spk_9f8e7d6c"),
    SpeakerDecision(engine_speaker="spk_1", action="rename", display_name="王五"),
])

workspace.save_session(result)
workspace.save_registry(registry)
```

The two documents are then available as strings:

```python
result.minutes_markdown
result.transcript_markdown
```

## Option B — HTTP API

```bash
attune serve --host 127.0.0.1 --port 8787
```

```python
import httpx

base = "http://127.0.0.1:8787"
project = "team-standup"

with httpx.Client(base_url=base, timeout=600) as client:
    session = client.post(f"/v1/projects/{project}/sessions", json={"title": "Standup"}).json()

    with open("/recordings/standup.wav", "rb") as handle:
        state = client.post(
            f"/v1/projects/{project}/sessions/{session['session_id']}/audio",
            content=handle,
            headers={"content-type": "audio/wav"},
        ).json()

    for item in state["resolutions"]:
        if not item["resolved"]:
            print(item["engine_speaker"], item["suggested_name"], item["candidates"])

    minutes = client.get(
        f"/v1/projects/{project}/sessions/{session['session_id']}/minutes",
        params={"format": "markdown"},
    ).text
```

## Wiring it into a Hermes memory file

Attune writes its own workspace; Hermes can read the Markdown documents
directly and keep whatever summaries it likes:

```python
from pathlib import Path

memory_dir = Path("~/.hermes/attune/team-standup/sessions/ses_001")
notes = (memory_dir / "session.json").read_text(encoding="utf-8")
```

## Notes

- `pipeline.process()` blocks for the length of the transcription. For long
  meetings, run it in a worker and poll the session file.
- The registry is a plain JSON file — safe to back up, diff and commit if you
  want the team's identities to travel with the repository.
