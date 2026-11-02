"""On-disk workspace layout.

<root>/
  projects/
    <project_id>/
      speakers.json          # the persistent identity registry
      sessions/
        <session_id>/
          session.json       # utterances, resolutions, both documents
          audio/             # the uploaded recording, if any
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from ..models import MeetingMinutes, MeetingResult, SpeakerResolution, Utterance
from .identity import SpeakerRegistry

_SAFE = re.compile(r"[^A-Za-z0-9_.-]+")


def safe_name(value: str, fallback: str = "unnamed") -> str:
    cleaned = _SAFE.sub("-", value.strip()).strip("-")
    return cleaned or fallback


class Workspace:
    """Reads and writes projects, sessions and speaker registries."""

    def __init__(self, root: str | Path = "data") -> None:
        self.root = Path(root)

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------
    def project_dir(self, project_id: str) -> Path:
        return self.root / "projects" / safe_name(project_id)

    def registry_path(self, project_id: str) -> Path:
        return self.project_dir(project_id) / "speakers.json"

    def session_dir(self, project_id: str, session_id: str) -> Path:
        return self.project_dir(project_id) / "sessions" / safe_name(session_id)

    def session_path(self, project_id: str, session_id: str) -> Path:
        return self.session_dir(project_id, session_id) / "session.json"

    def audio_dir(self, project_id: str, session_id: str) -> Path:
        return self.session_dir(project_id, session_id) / "audio"

    # ------------------------------------------------------------------
    # Registry
    # ------------------------------------------------------------------
    def load_registry(self, project_id: str) -> SpeakerRegistry:
        path = self.registry_path(project_id)
        if not path.exists():
            return SpeakerRegistry(project_id=project_id)
        return SpeakerRegistry.load(path, project_id=project_id)

    def save_registry(self, registry: SpeakerRegistry) -> str:
        return registry.save(self.registry_path(registry.project_id))

    # ------------------------------------------------------------------
    # Sessions
    # ------------------------------------------------------------------
    def save_session(self, result: MeetingResult) -> str:
        path = self.session_path(result.project_id, result.session_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "session_id": result.session_id,
            "project_id": result.project_id,
            "title": result.title,
            "audio_path": result.audio_path,
            "bindings": result.bindings,
            "utterances": [
                {
                    "start": utterance.start,
                    "end": utterance.end,
                    "text": utterance.text,
                    "engine_speaker": utterance.engine_speaker,
                    "embedding": list(utterance.embedding) if utterance.embedding else None,
                }
                for utterance in result.utterances
            ],
            "resolutions": [resolution.to_dict() for resolution in result.resolutions],
            "minutes": result.minutes.to_dict(),
            "transcript_markdown": result.transcript_markdown,
            "minutes_markdown": result.minutes_markdown,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return str(path)

    def load_session(self, project_id: str, session_id: str) -> MeetingResult:
        path = self.session_path(project_id, session_id)
        if not path.exists():
            raise FileNotFoundError(f"Unknown session: {project_id}/{session_id}")
        payload = json.loads(path.read_text(encoding="utf-8"))

        utterances = [
            Utterance(
                start=float(row.get("start", 0.0)),
                end=float(row.get("end", 0.0)),
                text=str(row.get("text", "")),
                engine_speaker=str(row.get("engine_speaker", "spk_0")),
                embedding=tuple(row["embedding"]) if row.get("embedding") else None,
            )
            for row in payload.get("utterances", [])
        ]
        resolutions = [
            SpeakerResolution(
                engine_speaker=str(row.get("engine_speaker", "")),
                kind=str(row.get("kind", "new")),  # type: ignore[arg-type]
                identity_id=row.get("identity_id"),
                score=float(row.get("score", 0.0)),
                suggested_name=str(row.get("suggested_name", "")),
                resolved=bool(row.get("resolved", False)),
            )
            for row in payload.get("resolutions", [])
        ]
        minutes_payload = payload.get("minutes", {})
        minutes = MeetingMinutes(
            title=str(minutes_payload.get("title", "")),
            summary=str(minutes_payload.get("summary", "")),
            participants=[],
            action_items=[],
            generated_by=str(minutes_payload.get("generated_by", "heuristic")),
        )
        return MeetingResult(
            session_id=str(payload.get("session_id", session_id)),
            project_id=str(payload.get("project_id", project_id)),
            utterances=utterances,
            resolutions=resolutions,
            minutes=minutes,
            transcript_markdown=str(payload.get("transcript_markdown", "")),
            minutes_markdown=str(payload.get("minutes_markdown", "")),
            bindings={str(k): str(v) for k, v in (payload.get("bindings") or {}).items()},
            title=str(payload.get("title", "")),
            audio_path=str(payload.get("audio_path", "")),
        )

    def list_sessions(self, project_id: str) -> list[str]:
        directory = self.project_dir(project_id) / "sessions"
        if not directory.exists():
            return []
        return sorted(path.name for path in directory.iterdir() if path.is_dir())

    def list_projects(self) -> list[str]:
        directory = self.root / "projects"
        if not directory.exists():
            return []
        return sorted(path.name for path in directory.iterdir() if path.is_dir())
