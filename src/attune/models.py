"""Core data models shared by every stage of the Attune pipeline.

The vocabulary here is deliberately narrow:

``Utterance``
    What an engine produced for one recording. It carries a *recording-local*
    speaker key (``engine_speaker``) such as ``spk_0`` — never a real identity.

``SpeakerProfile``
    A persistent identity that belongs to a project and may reappear in many
    later meetings. This is where cross-session recognition lives.

``SpeakerBinding``
    The link between one recording-local speaker key and one persistent
    identity, stored per session so a re-processed meeting stays stable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

SpeakerMatchKind = Literal["known", "ambiguous", "new"]


@dataclass(frozen=True)
class Utterance:
    """A single speech turn produced by a transcription engine."""

    start: float
    end: float
    text: str
    engine_speaker: str
    embedding: tuple[float, ...] | None = None

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def to_dict(self) -> dict[str, object]:
        return {
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "text": self.text,
            "engine_speaker": self.engine_speaker,
        }


@dataclass
class SpeakerProfile:
    """A persistent, cross-session speaker identity."""

    identity_id: str
    display_name: str
    aliases: list[str] = field(default_factory=list)
    embeddings: list[list[float]] = field(default_factory=list)
    appearances: int = 0
    total_seconds: float = 0.0
    confirmed: bool = False

    def centroid(self) -> list[float]:
        """Mean voiceprint vector, used as a coarse identity anchor."""
        if not self.embeddings:
            return []
        width = min(len(vector) for vector in self.embeddings)
        accumulator = [0.0] * width
        for vector in self.embeddings:
            for index, value in enumerate(vector[:width]):
                accumulator[index] += value
        count = float(len(self.embeddings))
        return [value / count for value in accumulator]

    def to_dict(self) -> dict[str, object]:
        return {
            "identity_id": self.identity_id,
            "display_name": self.display_name,
            "aliases": list(self.aliases),
            "embeddings": [list(vector) for vector in self.embeddings],
            "appearances": self.appearances,
            "total_seconds": round(self.total_seconds, 3),
            "confirmed": self.confirmed,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> SpeakerProfile:
        return cls(
            identity_id=str(data.get("identity_id", "")),
            display_name=str(data.get("display_name", "")),
            aliases=[str(alias) for alias in data.get("aliases", []) or []],  # type: ignore[union-attr]
            embeddings=[list(vector) for vector in data.get("embeddings", []) or []],  # type: ignore[union-attr]
            appearances=int(data.get("appearances", 0) or 0),
            total_seconds=float(data.get("total_seconds", 0.0) or 0.0),
            confirmed=bool(data.get("confirmed", False)),
        )


@dataclass
class SpeakerCandidate:
    """A possible identity for a recording-local speaker."""

    identity_id: str
    display_name: str
    score: float

    def to_dict(self) -> dict[str, object]:
        return {
            "identity_id": self.identity_id,
            "display_name": self.display_name,
            "score": round(self.score, 4),
        }


@dataclass
class MatchResult:
    """Outcome of matching one voiceprint against the project registry."""

    kind: SpeakerMatchKind
    engine_speaker: str
    identity_id: str | None = None
    score: float = 0.0
    candidates: list[SpeakerCandidate] = field(default_factory=list)

    @property
    def needs_user_decision(self) -> bool:
        return self.kind == "ambiguous"

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "engine_speaker": self.engine_speaker,
            "identity_id": self.identity_id,
            "score": round(self.score, 4),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
        }


@dataclass
class SpeakerBinding:
    """Persistent mapping of one recording-local speaker key to an identity."""

    session_id: str
    engine_speaker: str
    identity_id: str

    def to_dict(self) -> dict[str, object]:
        return {
            "session_id": self.session_id,
            "engine_speaker": self.engine_speaker,
            "identity_id": self.identity_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> SpeakerBinding:
        return cls(
            session_id=str(data.get("session_id", "")),
            engine_speaker=str(data.get("engine_speaker", "")),
            identity_id=str(data.get("identity_id", "")),
        )


@dataclass
class SpeakerResolution:
    """A speaker that needs the user to confirm merge-or-new."""

    engine_speaker: str
    kind: SpeakerMatchKind
    identity_id: str | None = None
    score: float = 0.0
    candidates: list[SpeakerCandidate] = field(default_factory=list)
    suggested_name: str = ""
    resolved: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "engine_speaker": self.engine_speaker,
            "kind": self.kind,
            "identity_id": self.identity_id,
            "score": round(self.score, 4),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "suggested_name": self.suggested_name,
            "resolved": self.resolved,
        }


@dataclass
class ActionItem:
    """One concrete commitment extracted from the meeting."""

    text: str
    owner_identity_id: str | None = None
    owner_name: str = ""
    source_quote: str = ""
    confidence: float = 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "text": self.text,
            "owner_identity_id": self.owner_identity_id,
            "owner_name": self.owner_name,
            "source_quote": self.source_quote,
            "confidence": round(self.confidence, 3),
        }


@dataclass
class ParticipantSummary:
    """What one person contributed to the meeting."""

    identity_id: str
    display_name: str
    speaking_seconds: float = 0.0
    utterance_count: int = 0
    outline: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "identity_id": self.identity_id,
            "display_name": self.display_name,
            "speaking_seconds": round(self.speaking_seconds, 2),
            "utterance_count": self.utterance_count,
            "outline": list(self.outline),
        }


@dataclass
class MeetingMinutes:
    """The summary document produced for a meeting."""

    title: str = ""
    summary: str = ""
    participants: list[ParticipantSummary] = field(default_factory=list)
    action_items: list[ActionItem] = field(default_factory=list)
    generated_by: str = "heuristic"

    def to_dict(self) -> dict[str, object]:
        return {
            "title": self.title,
            "summary": self.summary,
            "participants": [participant.to_dict() for participant in self.participants],
            "action_items": [item.to_dict() for item in self.action_items],
            "generated_by": self.generated_by,
        }


@dataclass
class MeetingResult:
    """Everything the pipeline produced for one meeting."""

    session_id: str
    project_id: str
    utterances: list[Utterance] = field(default_factory=list)
    resolutions: list[SpeakerResolution] = field(default_factory=list)
    minutes: MeetingMinutes = field(default_factory=MeetingMinutes)
    transcript_markdown: str = ""
    minutes_markdown: str = ""
    #: engine_speaker -> persistent identity_id for this session
    bindings: dict[str, str] = field(default_factory=dict)
    title: str = ""
    audio_path: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "session_id": self.session_id,
            "project_id": self.project_id,
            "title": self.title,
            "utterances": [utterance.to_dict() for utterance in self.utterances],
            "resolutions": [resolution.to_dict() for resolution in self.resolutions],
            "minutes": self.minutes.to_dict(),
            "bindings": dict(self.bindings),
        }
