"""The end-to-end meeting pipeline.

    audio  ->  utterances (recording-local speaker keys)
            ->  identity resolution (project registry)
            ->  minutes + transcript

The pipeline never guesses silently. A voiceprint that lands in the ambiguous
band produces a :class:`SpeakerResolution` that the caller must resolve —
merge into an existing identity, or keep it as a new one.
"""

from __future__ import annotations

import re
import uuid
from collections.abc import Sequence
from dataclasses import dataclass

from ..models import MeetingResult, SpeakerResolution, Utterance
from .identity import SpeakerRegistry
from .report import (
    HeuristicMinutesGenerator,
    MinutesGenerator,
    render_minutes_markdown,
    render_transcript_markdown,
)

#: "张三说" / "李总认为" — a cheap way to guess who is being addressed.
_ADDRESSED_PATTERNS = (
    re.compile(r"([\u4e00-\u9fa5]{2,3})(?:说|讲|认为|觉得|提到|表示|建议)"),
    re.compile(r"([A-Z][a-z]{1,15})\s+(?:said|says|thinks|mentioned|proposed)"),
)

#: A capitalised two-word name, e.g. "Alice Chen".
_LATIN_NAME = re.compile(r"\b([A-Z][a-z]{1,15})\s+([A-Z][a-z]{1,15})\b")


def guess_names(texts: Sequence[str]) -> list[str]:
    """Guess at participant names from what was said.

    This is a hint, not a verdict — it feeds the suggestion shown next to the
    "merge or new?" prompt so the user has something to confirm.
    """
    found: list[str] = []
    for text in texts:
        for pattern in _ADDRESSED_PATTERNS:
            for match in pattern.finditer(text):
                name = match.group(1).strip()
                if name and name not in found:
                    found.append(name)
        for match in _LATIN_NAME.finditer(text):
            name = f"{match.group(1)} {match.group(2)}".strip()
            if name not in found:
                found.append(name)
    return found


@dataclass
class SpeakerDecision:
    """A user answer to one "merge or create?" prompt."""

    engine_speaker: str
    action: str  # "assign" | "merge" | "rename"
    identity_id: str = ""
    display_name: str = ""


class MeetingPipeline:
    """Turns an audio file into attributed utterances, minutes and a transcript."""

    def __init__(
        self,
        engine: object,
        registry: SpeakerRegistry,
        minutes_generator: MinutesGenerator | None = None,
    ) -> None:
        self.engine = engine
        self.registry = registry
        self.minutes_generator = minutes_generator or HeuristicMinutesGenerator()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def process(
        self,
        audio_path: str,
        *,
        session_id: str | None = None,
        title: str = "",
        project_id: str | None = None,
    ) -> MeetingResult:
        """Transcribe, resolve identities and produce both documents."""
        session_id = session_id or f"ses_{uuid.uuid4().hex[:10]}"
        utterances = self.engine.transcribe(audio_path)  # type: ignore[attr-defined]

        bindings, resolutions = self._resolve_identities(utterances, session_id)
        result = MeetingResult(
            session_id=session_id,
            project_id=project_id or self.registry.project_id,
            utterances=utterances,
            resolutions=resolutions,
            bindings=bindings,
            title=title,
            audio_path=audio_path,
        )
        self._regenerate(result)
        return result

    def resolve(self, result: MeetingResult, decisions: Sequence[SpeakerDecision]) -> MeetingResult:
        """Apply the user's merge / assign / rename answers and re-render."""
        for decision in decisions:
            engine_speaker = decision.engine_speaker
            provisional = result.bindings.get(engine_speaker)

            if decision.action == "merge" and provisional and decision.identity_id:
                self.registry.merge(provisional, decision.identity_id)
                result.bindings[engine_speaker] = decision.identity_id
                if decision.display_name:
                    self.registry.rename(decision.identity_id, decision.display_name)
            elif decision.action == "assign" and decision.identity_id:
                if provisional and provisional != decision.identity_id:
                    self.registry.remove(provisional)
                result.bindings[engine_speaker] = decision.identity_id
                if decision.display_name:
                    self.registry.rename(decision.identity_id, decision.display_name)
                profile = self.registry.profile(decision.identity_id)
                if profile:
                    profile.confirmed = True
            elif decision.action == "rename" and provisional:
                if decision.display_name:
                    self.registry.rename(provisional, decision.display_name)

            target_identity = result.bindings.get(engine_speaker)
            if target_identity:
                self.registry.bind(result.session_id, engine_speaker, target_identity)

            for resolution in result.resolutions:
                if resolution.engine_speaker == engine_speaker:
                    resolution.resolved = True
                    resolution.identity_id = target_identity

        self._regenerate(result)
        return result

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _resolve_identities(
        self, utterances: Sequence[Utterance], session_id: str
    ) -> tuple[dict[str, str], list[SpeakerResolution]]:
        bindings: dict[str, str] = {}
        resolutions: list[SpeakerResolution] = []

        by_speaker: dict[str, list[Utterance]] = {}
        for utterance in utterances:
            by_speaker.setdefault(utterance.engine_speaker, []).append(utterance)

        for index, engine_speaker in enumerate(sorted(by_speaker.keys())):
            rows = by_speaker[engine_speaker]

            existing = self.registry.identity_for_engine_speaker(session_id, engine_speaker)
            if existing and self.registry.profile(existing):
                bindings[engine_speaker] = existing
                self._refresh_profile(existing, rows)
                continue

            embedding = self._centroid(rows)
            if embedding is None:
                profile = self.registry.create_identity(
                    self._fallback_name(rows, index), None, confirmed=False
                )
                bindings[engine_speaker] = profile.identity_id
                resolutions.append(
                    SpeakerResolution(
                        engine_speaker=engine_speaker,
                        kind="new",
                        identity_id=profile.identity_id,
                        suggested_name=profile.display_name,
                    )
                )
                continue

            match = self.registry.match(embedding, engine_speaker)
            if match.kind == "known" and match.identity_id:
                bindings[engine_speaker] = match.identity_id
                self._refresh_profile(match.identity_id, rows, embedding)
                profile = self.registry.profile(match.identity_id)
                if profile is not None and not profile.confirmed:
                    # We have seen this voice before, but nobody has confirmed
                    # who it is yet. Keep asking until someone does.
                    resolutions.append(
                        SpeakerResolution(
                            engine_speaker=engine_speaker,
                            kind="known",
                            identity_id=match.identity_id,
                            score=match.score,
                            candidates=match.candidates,
                            suggested_name=profile.display_name,
                        )
                    )
                continue

            profile = self.registry.create_identity(
                self._suggest_name(rows, index), embedding, confirmed=False
            )
            bindings[engine_speaker] = profile.identity_id
            resolutions.append(
                SpeakerResolution(
                    engine_speaker=engine_speaker,
                    kind=match.kind,
                    identity_id=profile.identity_id,
                    score=match.score,
                    candidates=match.candidates,
                    suggested_name=profile.display_name,
                )
            )

        for engine_speaker, identity_id in bindings.items():
            self.registry.bind(session_id, engine_speaker, identity_id)
        return bindings, resolutions

    def _refresh_profile(
        self,
        identity_id: str,
        rows: Sequence[Utterance],
        embedding: Sequence[float] | None = None,
    ) -> None:
        profile = self.registry.profile(identity_id)
        if profile is None:
            return
        profile.appearances += 1
        profile.total_seconds += sum(row.duration for row in rows)
        if embedding:
            profile.embeddings.append([float(value) for value in embedding])

    @staticmethod
    def _centroid(rows: Sequence[Utterance]) -> list[float] | None:
        vectors = [row.embedding for row in rows if row.embedding]
        if not vectors:
            return None
        width = min(len(vector) for vector in vectors)  # type: ignore[arg-type]
        accumulator = [0.0] * width
        for vector in vectors:
            for index, value in enumerate(vector[:width]):  # type: ignore[index]
                accumulator[index] += float(value)
        count = float(len(vectors))
        return [value / count for value in accumulator]

    def _suggest_name(self, rows: Sequence[Utterance], index: int) -> str:
        guesses = guess_names([row.text for row in rows])
        if guesses:
            return guesses[0]
        return self._fallback_name(rows, index)

    @staticmethod
    def _fallback_name(rows: Sequence[Utterance], index: int) -> str:
        return f"发言人 {index + 1}"

    def _regenerate(self, result: MeetingResult) -> None:
        name_map = self.name_map(result)
        minutes = self.minutes_generator.generate(result.utterances, name_map, title=result.title)
        result.minutes = minutes
        result.transcript_markdown = render_transcript_markdown(
            result.utterances, name_map, title=result.title
        )
        result.minutes_markdown = render_minutes_markdown(minutes)

    def name_map(self, result: MeetingResult) -> dict[str, str]:
        """engine_speaker -> display name, resolved through the registry."""
        names: dict[str, str] = {}
        for engine_speaker, identity_id in result.bindings.items():
            profile = self.registry.profile(identity_id)
            names[engine_speaker] = profile.display_name if profile else identity_id
        return names
