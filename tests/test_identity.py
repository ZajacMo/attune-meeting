"""Tests for the cross-session speaker identity registry."""

from __future__ import annotations

import math

from attune.core.identity import SpeakerRegistry, cosine_similarity, new_identity_id

VOICE_A = (1.0, 0.0, 0.0)
VOICE_B = (0.0, 1.0, 0.0)
VOICE_A_CLOSE = (0.99, 0.05, 0.0)
VOICE_MIDDLE = (0.70, 0.71, 0.0)


def test_cosine_similarity_basics() -> None:
    assert cosine_similarity(VOICE_A, VOICE_A) == 1.0
    assert cosine_similarity(VOICE_A, VOICE_B) == 0.0
    assert cosine_similarity(VOICE_A, VOICE_A_CLOSE) > 0.99
    assert cosine_similarity([], VOICE_A) == 0.0


def test_cosine_similarity_tolerates_length_mismatch() -> None:
    value = cosine_similarity((1.0, 0.0, 0.0), (1.0, 0.0))
    assert math.isclose(value, 1.0)


def test_empty_registry_reports_new_speaker() -> None:
    registry = SpeakerRegistry(project_id="p1")
    match = registry.match(VOICE_A, "spk_0")
    assert match.kind == "new"
    assert match.identity_id is None


def test_known_speaker_is_matched_silently() -> None:
    registry = SpeakerRegistry(project_id="p1")
    registry.enroll("spk_known", "张三", VOICE_A)

    match = registry.match(VOICE_A, "spk_0")
    assert match.kind == "known"
    assert match.identity_id == "spk_known"
    assert match.score > 0.99


def test_ambiguous_speaker_asks_the_user() -> None:
    registry = SpeakerRegistry(project_id="p1")
    registry.enroll("spk_known", "张三", VOICE_A)

    match = registry.match(VOICE_MIDDLE, "spk_0")
    assert match.kind == "ambiguous"
    assert match.identity_id is None
    assert match.needs_user_decision is True
    assert match.candidates[0].identity_id == "spk_known"
    assert match.score >= registry.low_threshold


def test_distant_speaker_is_new() -> None:
    registry = SpeakerRegistry(project_id="p1")
    registry.enroll("spk_known", "张三", VOICE_A)

    match = registry.match(VOICE_B, "spk_1")
    assert match.kind == "new"
    assert match.identity_id is None


def test_thresholds_are_configurable() -> None:
    registry = SpeakerRegistry(project_id="p1", high_threshold=0.99, low_threshold=0.5)
    registry.enroll("spk_known", "张三", VOICE_A)

    # 0.70 similarity: accepted by default, but "ambiguous" once the high
    # threshold is raised above it.
    match = registry.match(VOICE_MIDDLE, "spk_0")
    assert match.kind == "ambiguous"


def test_merge_moves_voiceprints_and_keeps_alias() -> None:
    registry = SpeakerRegistry(project_id="p1")
    registry.enroll("spk_a", "张三", VOICE_A, confirmed=True)
    registry.enroll("spk_b", "李四", VOICE_A, confirmed=True)
    registry.bind("session-1", "spk_0", "spk_b")

    registry.merge("spk_b", "spk_a")

    assert registry.profile("spk_b") is None
    target = registry.profile("spk_a")
    assert target is not None
    assert "李四" in target.aliases
    assert len(target.embeddings) == 2
    assert registry.identity_for_engine_speaker("session-1", "spk_0") == "spk_a"


def test_rename_marks_identity_confirmed() -> None:
    registry = SpeakerRegistry(project_id="p1")
    profile = registry.create_identity("发言人 1", VOICE_A, confirmed=False)
    assert profile.confirmed is False

    renamed = registry.rename(profile.identity_id, "王五")
    assert renamed.display_name == "王五"
    assert renamed.confirmed is True


def test_remove_drops_bindings() -> None:
    registry = SpeakerRegistry(project_id="p1")
    registry.enroll("spk_a", "张三", VOICE_A)
    registry.bind("session-1", "spk_0", "spk_a")

    assert registry.remove("spk_a") is True
    assert registry.identity_for_engine_speaker("session-1", "spk_0") is None
    assert registry.remove("spk_a") is False


def test_registry_round_trips_through_disk(tmp_path) -> None:
    registry = SpeakerRegistry(project_id="p1")
    registry.enroll("spk_a", "张三", VOICE_A, confirmed=True)
    registry.bind("session-1", "spk_0", "spk_a")
    path = registry.save(tmp_path / "speakers.json")

    loaded = SpeakerRegistry.load(path)
    assert loaded.project_id == "p1"
    assert loaded.profile("spk_a") is not None
    assert loaded.profile("spk_a").display_name == "张三"
    assert loaded.identity_for_engine_speaker("session-1", "spk_0") == "spk_a"


def test_identity_ids_are_unique() -> None:
    assert new_identity_id() != new_identity_id()


def test_identity_lookup_by_name_and_alias() -> None:
    registry = SpeakerRegistry(project_id="p1")
    registry.enroll("spk_a", "张三", VOICE_A)
    registry.add_alias("spk_a", "张经理")

    assert registry.identity_for_name("张三") is not None
    assert registry.identity_for_name("张经理") is not None
    assert registry.identity_for_name("不存在") is None
