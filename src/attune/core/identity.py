"""Cross-session speaker identity.

Diarization engines only ever produce *recording-local* labels (``spk_0``,
``spk_1``). They cannot tell you that ``spk_0`` in Tuesday's meeting is the
same person as ``spk_1`` in next week's. That is what this module adds:

* a **project-scoped registry** of persistent identities, each holding one or
  more voiceprints;
* **matching** of a fresh voiceprint against that registry;
* an explicit **ambiguous** outcome, so the user is asked to merge or create
  instead of the system guessing silently;
* **bindings** that pin a recording-local speaker key to an identity for the
  lifetime of a session, so re-processing a meeting is stable.

The registry lives on disk as a single JSON document per project.
"""

from __future__ import annotations

import json
import math
import uuid
from collections.abc import Sequence
from pathlib import Path

from ..models import MatchResult, SpeakerBinding, SpeakerCandidate, SpeakerProfile

REGISTRY_VERSION = 1

#: Similarity at or above which a voiceprint is accepted as a known identity
#: and the identity is reused without asking the user.
#:
#: Calibrated on real Mandarin meeting audio (AISHELL-4), 126 enrolment clips
#: from 14 speakers across 7 sessions. At this value a voiceprint of a
#: *different* person is accepted for roughly 1 in 120 comparisons against a
#: profile that holds three voiceprints — the only situation in which Attune
#: stays silent, and therefore the only situation in which it can be wrong
#: without the user noticing. See ``docs/voiceprint-calibration.md``.
DEFAULT_HIGH_THRESHOLD = 0.68

#: Similarity below which a voiceprint is treated as a brand new person.
#:
#: This is the equal-error point of the two similarity distributions: below it
#: "somebody I have never met" is the more likely reading, so Attune asks the
#: user who the new person is instead of offering a shortlist.
DEFAULT_LOW_THRESHOLD = 0.53


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Cosine similarity of two vectors, tolerant of differing lengths."""
    if not left or not right:
        return 0.0
    width = min(len(left), len(right))
    dot = 0.0
    left_norm = 0.0
    right_norm = 0.0
    for index in range(width):
        left_value = float(left[index])
        right_value = float(right[index])
        dot += left_value * right_value
        left_norm += left_value * left_value
        right_norm += right_value * right_value
    if left_norm <= 0.0 or right_norm <= 0.0:
        return 0.0
    return dot / (math.sqrt(left_norm) * math.sqrt(right_norm))


def new_identity_id() -> str:
    return f"spk_{uuid.uuid4().hex[:8]}"


class SpeakerRegistry:
    """Persistent, project-scoped directory of known speakers."""

    def __init__(
        self,
        project_id: str,
        profiles: list[SpeakerProfile] | None = None,
        bindings: list[SpeakerBinding] | None = None,
        *,
        high_threshold: float = DEFAULT_HIGH_THRESHOLD,
        low_threshold: float = DEFAULT_LOW_THRESHOLD,
    ) -> None:
        self.project_id = project_id
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self._profiles: dict[str, SpeakerProfile] = {
            profile.identity_id: profile for profile in (profiles or [])
        }
        self._bindings: list[SpeakerBinding] = list(bindings or [])

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    @property
    def profiles(self) -> list[SpeakerProfile]:
        return sorted(self._profiles.values(), key=lambda profile: profile.display_name)

    @property
    def bindings(self) -> list[SpeakerBinding]:
        return list(self._bindings)

    def profile(self, identity_id: str) -> SpeakerProfile | None:
        return self._profiles.get(identity_id)

    def identity_for_name(self, display_name: str) -> SpeakerProfile | None:
        needle = display_name.strip().lower()
        if not needle:
            return None
        for profile in self.profiles:
            if profile.display_name.strip().lower() == needle:
                return profile
            if any(alias.strip().lower() == needle for alias in profile.aliases):
                return profile
        return None

    # ------------------------------------------------------------------
    # Matching
    # ------------------------------------------------------------------
    def score_against(self, embedding: Sequence[float]) -> list[SpeakerCandidate]:
        """Score a voiceprint against every known identity, best first."""
        candidates: list[SpeakerCandidate] = []
        for profile in self._profiles.values():
            score = self._score_profile(profile, embedding)
            if score is None:
                continue
            candidates.append(
                SpeakerCandidate(
                    identity_id=profile.identity_id,
                    display_name=profile.display_name,
                    score=score,
                )
            )
        candidates.sort(key=lambda candidate: candidate.score, reverse=True)
        return candidates

    @staticmethod
    def _score_profile(profile: SpeakerProfile, embedding: Sequence[float]) -> float | None:
        """Score a voiceprint against one identity.

        The identity is represented by the **centroid** of its stored
        voiceprints, not by the best of them. Taking the best individual
        voiceprint gets more optimistic every time the person is enrolled
        again — an impostor's score rises with the number of voiceprints on
        file, so the threshold would have to move as the registry grows.
        Comparing against the average keeps the impostor score flat and still
        lifts the genuine score, which is what makes one fixed threshold
        meaningful over time. Measured on AISHELL-4, the centroid rule beats
        the best-of rule on both false accepts and recall at every operating
        point (see ``docs/voiceprint-calibration.md``).
        """
        if not profile.embeddings:
            return None
        anchor = profile.centroid()
        if not anchor:
            return None
        return cosine_similarity(embedding, anchor)

    def match(self, embedding: Sequence[float], engine_speaker: str = "") -> MatchResult:
        """Match one voiceprint against the registry.

        Three outcomes:

        ``known``
            Confident match — reuse that identity silently.
        ``ambiguous``
            Similar enough to be one of several identities, but not confident
            enough to choose. The caller must ask the user.
        ``new``
            Nobody close enough — a new person shows up.
        """
        candidates = self.score_against(embedding)
        if not candidates:
            return MatchResult(kind="new", engine_speaker=engine_speaker, score=0.0)

        best = candidates[0]
        if best.score >= self.high_threshold:
            return MatchResult(
                kind="known",
                engine_speaker=engine_speaker,
                identity_id=best.identity_id,
                score=best.score,
                candidates=candidates,
            )
        if best.score >= self.low_threshold:
            return MatchResult(
                kind="ambiguous",
                engine_speaker=engine_speaker,
                identity_id=None,
                score=best.score,
                candidates=candidates,
            )
        return MatchResult(
            kind="new",
            engine_speaker=engine_speaker,
            score=best.score,
            candidates=candidates,
        )

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------
    def enroll(
        self,
        identity_id: str,
        display_name: str,
        embedding: Sequence[float] | None = None,
        *,
        confirmed: bool = True,
        aliases: Sequence[str] = (),
    ) -> SpeakerProfile:
        """Create or refresh an identity, adding the voiceprint when given."""
        profile = self._profiles.get(identity_id)
        if profile is None:
            profile = SpeakerProfile(
                identity_id=identity_id,
                display_name=display_name.strip() or identity_id,
                confirmed=confirmed,
            )
            self._profiles[identity_id] = profile
        if display_name.strip():
            profile.display_name = display_name.strip()
        if confirmed:
            profile.confirmed = True
        for alias in aliases:
            self.add_alias(identity_id, alias)
        if embedding:
            profile.embeddings.append([float(value) for value in embedding])
        return profile

    def create_identity(
        self,
        display_name: str,
        embedding: Sequence[float] | None = None,
        *,
        confirmed: bool = False,
    ) -> SpeakerProfile:
        """Register a brand new identity with a generated id."""
        identity_id = new_identity_id()
        return self.enroll(identity_id, display_name, embedding, confirmed=confirmed)

    def rename(self, identity_id: str, display_name: str) -> SpeakerProfile:
        profile = self._require(identity_id)
        if display_name.strip():
            profile.display_name = display_name.strip()
        profile.confirmed = True
        return profile

    def add_alias(self, identity_id: str, alias: str) -> SpeakerProfile:
        profile = self._require(identity_id)
        cleaned = alias.strip()
        if cleaned and cleaned not in profile.aliases:
            profile.aliases.append(cleaned)
        return profile

    def merge(self, source_id: str, target_id: str) -> SpeakerProfile:
        """Fold ``source_id`` into ``target_id`` and drop the source.

        The source's voiceprints, appearance count and speaking time move to
        the target; the source's display name is kept as an alias so old
        transcripts still read sensibly.
        """
        source = self._require(source_id)
        target = self._require(target_id)
        if source_id == target_id:
            return target

        target.embeddings.extend(source.embeddings)
        target.appearances += source.appearances
        target.total_seconds += source.total_seconds
        if (
            source.display_name
            and source.display_name != target.display_name
            and source.display_name not in target.aliases
        ):
            target.aliases.append(source.display_name)
        for alias in source.aliases:
            if alias not in target.aliases:
                target.aliases.append(alias)
        target.confirmed = target.confirmed or source.confirmed

        del self._profiles[source_id]
        for binding in self._bindings:
            if binding.identity_id == source_id:
                binding.identity_id = target_id
        return target

    def remove(self, identity_id: str) -> bool:
        if identity_id not in self._profiles:
            return False
        del self._profiles[identity_id]
        self._bindings = [
            binding for binding in self._bindings if binding.identity_id != identity_id
        ]
        return True

    # ------------------------------------------------------------------
    # Session bindings
    # ------------------------------------------------------------------
    def bind(self, session_id: str, engine_speaker: str, identity_id: str) -> SpeakerBinding:
        for binding in self._bindings:
            if binding.session_id == session_id and binding.engine_speaker == engine_speaker:
                binding.identity_id = identity_id
                return binding
        binding = SpeakerBinding(
            session_id=session_id, engine_speaker=engine_speaker, identity_id=identity_id
        )
        self._bindings.append(binding)
        return binding

    def binding_for(self, session_id: str, engine_speaker: str) -> SpeakerBinding | None:
        for binding in self._bindings:
            if binding.session_id == session_id and binding.engine_speaker == engine_speaker:
                return binding
        return None

    def identity_for_engine_speaker(self, session_id: str, engine_speaker: str) -> str | None:
        binding = self.binding_for(session_id, engine_speaker)
        return binding.identity_id if binding else None

    def _require(self, identity_id: str) -> SpeakerProfile:
        profile = self._profiles.get(identity_id)
        if profile is None:
            raise KeyError(f"Unknown speaker identity: {identity_id}")
        return profile

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def to_dict(self) -> dict[str, object]:
        return {
            "version": REGISTRY_VERSION,
            "project_id": self.project_id,
            "high_threshold": self.high_threshold,
            "low_threshold": self.low_threshold,
            "profiles": [profile.to_dict() for profile in self.profiles],
            "bindings": [binding.to_dict() for binding in self._bindings],
        }

    def save(self, path: str | Path) -> str:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return str(target)

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        project_id: str | None = None,
        high_threshold: float | None = None,
        low_threshold: float | None = None,
    ) -> SpeakerRegistry:
        source = Path(path)
        if not source.exists():
            raise FileNotFoundError(f"Speaker registry not found: {source}")
        payload = json.loads(source.read_text(encoding="utf-8"))
        profiles = [SpeakerProfile.from_dict(row) for row in payload.get("profiles", [])]
        bindings = [SpeakerBinding.from_dict(row) for row in payload.get("bindings", [])]
        return cls(
            project_id=project_id or str(payload.get("project_id", "")),
            profiles=profiles,
            bindings=bindings,
            high_threshold=(
                high_threshold
                if high_threshold is not None
                else float(payload.get("high_threshold", DEFAULT_HIGH_THRESHOLD))
            ),
            low_threshold=(
                low_threshold
                if low_threshold is not None
                else float(payload.get("low_threshold", DEFAULT_LOW_THRESHOLD))
            ),
        )
