# Attune in OpenClaw

OpenClaw is a **plugin protocol** for wearables: a skill is a small package
with a manifest, a handler, and a distribution format. The runtime expects
*structured data back* — the device and the agent own the presentation.

That maps onto Attune cleanly: Attune is the capability, OpenClaw's agent is
the worker, and the skill returns structured meeting data.

## The shape of a skill

A skill declares what it does, what it needs, and which intents should reach
it. For Attune that is roughly:

```json
{
  "name": "attune-meeting",
  "version": "0.1.0",
  "description": "Transcribe a meeting and report who said what, with minutes and action items.",
  "permissions": ["filesystem:read", "process:spawn"],
  "intents": [
    "record this meeting",
    "who said what",
    "summarise the meeting",
    "what are the action items"
  ],
  "handler": "attune_meeting.handler:handle"
}
```

## The handler

The handler receives the agent's structured payload and returns structured
data. Attune's HTTP API is the simplest way to implement it:

```python
# attune_meeting/handler.py
import httpx

BASE = "http://127.0.0.1:8787"


def handle(payload: dict) -> dict:
    project = payload.get("project", "default")
    audio_path = payload["audio_path"]

    with httpx.Client(base_url=BASE, timeout=900) as client:
        session = client.post(
            f"/v1/projects/{project}/sessions", json={"title": payload.get("title", "")}
        ).json()

        with open(audio_path, "rb") as audio:
            state = client.post(
                f"/v1/projects/{project}/sessions/{session['session_id']}/audio",
                content=audio,
                headers={"content-type": "audio/wav"},
            ).json()

        transcript = client.get(
            f"/v1/projects/{project}/sessions/{session['session_id']}/transcript",
            params={"format": "markdown"},
        ).text

    # Structured result: the agent decides how to present it.
    return {
        "session_id": state["session_id"],
        "transcript_markdown": transcript,
        "minutes": state["minutes"],
        "speakers": state["bindings"],
        "needs_confirmation": [
            {
                "engine_speaker": item["engine_speaker"],
                "suggested_name": item["suggested_name"],
                "candidates": item["candidates"],
            }
            for item in state["resolutions"]
            if not item["resolved"]
        ],
    }
```

## Realtime note

Attune is **offline** by design: it needs the whole recording before it can
separate speakers and produce documents. A realtime OpenClaw plugin should
therefore run the capture, and call Attune once the meeting ends.

For realtime speech-to-text, FunASR ships its own OpenClaw realtime plugin
(see the FunASR documentation) — use that for live captions and Attune for the
attributed minutes.

## Distribution

Skills install from a registry or straight from a Git repository. Attune's
Python package is the dependency; the skill is a thin wrapper:

```json
{
  "dependencies": ["attune>=0.1.0"],
  "entrypoints": {"attune": "attune_meeting.handler:handle"}
}
```

## What to hand back

| The agent wants | Give it |
| --- | --- |
| "who spoke" | `bindings` plus the per-person `outline` from the minutes |
| "what was decided" | `minutes.action_items` and `minutes.summary` |
| "the full text" | `GET /v1/projects/{p}/sessions/{s}/transcript?format=markdown` |
| "who is that" | `needs_confirmation` — ask the human, then call `resolve_speaker` |

Always surface `needs_confirmation` to the human. Attune will not pick a name
for you, and neither should the skill.
