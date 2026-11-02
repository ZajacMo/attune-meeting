# Attune in CyberColleague

CyberColleague is a desktop workbench that reads the shared screen, listens to
the user, and then researches, executes and verifies work. Attune is the
**multi-party meeting** counterpart: where CyberColleague understands *one*
person talking over *their* screen, Attune understands *several* people
talking in a *meeting*.

They are separate products that share a protocol, not a codebase. CyberColleague
consumes Attune the same way Cherry Studio, Hermes and OpenClaw do — over MCP
or the HTTP API. There is no submodule and no internal import.

## Why not a submodule

Three reasons, all of which show up in practice:

1. **Dependency weight.** Attune's ASR extras pull in torch. CyberColleague
   ships a deliberately small Python bridge inside an Electron installer.
2. **Release cadence.** The model stack moves much faster than the desktop
   app; coupling them means every model bump becomes an app release.
3. **One integration path.** If CyberColleague reached into Attune's internals
   while everyone else used the public protocol, the public protocol would
   quietly rot — and "reusable by other frameworks" would never actually be
   tested.

## Option A — MCP (recommended)

CyberColleague already speaks MCP: its `McpService` supports stdio, SSE and
streamable HTTP. Add Attune like any other server.

```json
{
  "mcpServers": {
    "attune": {
      "command": "attune-mcp",
      "args": ["--workspace", "C:\\Users\\you\\attune-data"]
    }
  }
}
```

Then the workbench's model can call `transcribe_audio`, read the minutes and
ask the user to confirm speakers — the same tools every other host gets.

## Option B — HTTP API

```bash
attune serve --host 127.0.0.1 --port 8787
```

```ts
const base = 'http://127.0.0.1:8787'

const session = await fetch(`${base}/v1/projects/${project}/sessions`, {
  method: 'POST',
  headers: { 'content-type': 'application/json' },
  body: JSON.stringify({ title: '产品周会' })
}).then((r) => r.json())

const state = await fetch(`${base}/v1/projects/${project}/sessions/${session.session_id}/audio`, {
  method: 'POST',
  headers: { 'content-type': 'audio/wav' },
  body: audioBuffer
}).then((r) => r.json())
```

## Fitting it into the workbench

CyberColleague's design rules already anticipate this shape:

- **The stage band is a view, not a state.** A meeting session is a *record*,
  shown under the research or execution stage — it must not change which stage
  the real task is in.
- **Perception must be observable.** When Attune is processing, show the
  pending speaker confirmations *before* the text composer, the same way
  spoken questions are surfaced.
- **Never synthesise progress.** Render Attune's actual states: transcribing,
  matching voices, waiting for a name, done. No indeterminate spinner.
- **Delivery needs evidence.** The minutes carry the source quote for every
  action item, so the workbench can show why a task item exists.

### A natural place for the action items

CyberColleague already has a task ledger (`memory/stores.py` → `TaskStore`)
for commitments extracted from conversations. Attune's `action_items` map onto
it directly:

```python
from cybercolleague.memory.stores import TaskStore

store = TaskStore.for_task(task_id)
for item in minutes.action_items:
    store.add(item.text, trigger=item.owner_name)
```

The owner is the person who made the commitment, and `source_quote` keeps the
provenance — both properties CyberColleague's evidence rules care about.

## Boundaries worth keeping

| Concern | Owner |
| --- | --- |
| Capturing the shared screen | CyberColleague |
| Capturing and separating meeting audio | Attune |
| Who said it | Attune's identity registry |
| What the user meant to build | CyberColleague's task brain |

The two products should never fight over the same state. CyberColleague owns
the task; Attune owns the meeting record.

## A concrete flow

1. The user shares a meeting link, or drops in a recording.
2. CyberColleague hands the audio to Attune and shows the real processing
   state.
3. Attune returns the transcript, the minutes and — when a voice is new or
   ambiguous — a short list of "who is this?" prompts with candidates.
4. CyberColleague asks the user, forwards the answers to
   `resolve_speaker`, and re-reads the two documents.
5. The action items are merged into CyberColleague's task ledger, and the
   meeting is attached to the task as evidence.
