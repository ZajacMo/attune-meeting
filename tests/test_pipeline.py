"""Tests for the end-to-end meeting pipeline."""

from __future__ import annotations

from attune.core.engines import StaticEngine
from attune.core.identity import SpeakerRegistry
from attune.core.pipeline import MeetingPipeline, SpeakerDecision, guess_names

VOICE_A = (1.0, 0.0, 0.0)
VOICE_B = (0.0, 1.0, 0.0)
VOICE_MIDDLE = (0.70, 0.71, 0.0)


def _utterances(voice_a=VOICE_A, voice_b=VOICE_B):
    return [
        __import__("attune").Utterance(0.0, 4.0, "今天讨论一下发布计划。", "spk_0", voice_a),
        __import__("attune").Utterance(4.0, 8.0, "我负责准备发布说明。", "spk_0", voice_a),
        __import__("attune").Utterance(8.0, 12.0, "测试环境我来跟进。", "spk_1", voice_b),
    ]


def test_first_meeting_creates_pending_resolutions() -> None:
    registry = SpeakerRegistry(project_id="p1")
    pipeline = MeetingPipeline(StaticEngine(_utterances()), registry)

    result = pipeline.process("meeting.wav", session_id="s1")

    assert len(result.resolutions) == 2
    assert all(item.kind == "new" for item in result.resolutions)
    assert all(not item.resolved for item in result.resolutions)
    assert set(result.bindings) == {"spk_0", "spk_1"}


def test_renaming_a_speaker_flows_into_both_documents() -> None:
    registry = SpeakerRegistry(project_id="p1")
    pipeline = MeetingPipeline(StaticEngine(_utterances()), registry)
    result = pipeline.process("meeting.wav", session_id="s1")

    pipeline.resolve(
        result,
        [
            SpeakerDecision(engine_speaker="spk_0", action="rename", display_name="张三"),
            SpeakerDecision(engine_speaker="spk_1", action="rename", display_name="李四"),
        ],
    )

    assert all(item.resolved for item in result.resolutions)
    assert "张三" in result.transcript_markdown
    assert "张三" in result.minutes_markdown
    assert "李四" in result.minutes_markdown


def test_same_voice_is_recognised_in_a_later_meeting() -> None:
    registry = SpeakerRegistry(project_id="p1")
    pipeline = MeetingPipeline(StaticEngine(_utterances()), registry)

    first = pipeline.process("meeting-1.wav", session_id="s1")
    pipeline.resolve(
        first,
        [SpeakerDecision(engine_speaker="spk_0", action="rename", display_name="张三")],
    )

    # A later meeting: the same voice arrives under a different local key.
    second_utterances = [
        __import__("attune").Utterance(0.0, 3.0, "这次同步一下进度。", "spk_0", VOICE_A),
        __import__("attune").Utterance(3.0, 6.0, "新的同事第一次参加。", "spk_1", VOICE_B),
    ]
    second = MeetingPipeline(StaticEngine(second_utterances), registry).process(
        "meeting-2.wav", session_id="s2"
    )

    assert "张三" in second.transcript_markdown
    # Only the genuinely new voice needs a decision.
    assert len(second.resolutions) == 1
    assert second.resolutions[0].engine_speaker == "spk_1"


def test_unnamed_speaker_keeps_asking_until_named() -> None:
    """An identity nobody has named yet must not be silently reused forever."""
    registry = SpeakerRegistry(project_id="p1")
    first = MeetingPipeline(StaticEngine(_utterances()), registry).process(
        "meeting-1.wav", session_id="s1"
    )
    assert len(first.resolutions) == 2

    # Only one of them gets named.
    MeetingPipeline(StaticEngine([]), registry).resolve(
        first, [SpeakerDecision(engine_speaker="spk_0", action="rename", display_name="张三")]
    )

    second = MeetingPipeline(StaticEngine(_utterances()), registry).process(
        "meeting-2.wav", session_id="s2"
    )

    # The unnamed voice is recognised, but still needs a name.
    pending = {item.engine_speaker for item in second.resolutions if not item.resolved}
    assert pending == {"spk_1"}


def test_ambiguous_voice_is_presented_to_the_user() -> None:
    registry = SpeakerRegistry(project_id="p1")
    registry.enroll("spk_seed", "张三", VOICE_A)

    ambiguous = [
        __import__("attune").Utterance(0.0, 3.0, "我们继续上次的方案。", "spk_0", VOICE_MIDDLE),
    ]
    pipeline = MeetingPipeline(StaticEngine(ambiguous), registry)
    result = pipeline.process("meeting.wav", session_id="s1")

    assert len(result.resolutions) == 1
    resolution = result.resolutions[0]
    assert resolution.kind == "ambiguous"
    assert resolution.candidates[0].identity_id == "spk_seed"
    assert resolution.candidates[0].display_name == "张三"


def test_merging_an_ambiguous_voice_keeps_the_chosen_name() -> None:
    registry = SpeakerRegistry(project_id="p1")
    registry.enroll("spk_seed", "张三", VOICE_A)

    ambiguous = [
        __import__("attune").Utterance(0.0, 3.0, "我们继续上次的方案。", "spk_0", VOICE_MIDDLE),
    ]
    pipeline = MeetingPipeline(StaticEngine(ambiguous), registry)
    result = pipeline.process("meeting.wav", session_id="s1")

    pipeline.resolve(
        result,
        [SpeakerDecision(engine_speaker="spk_0", action="merge", identity_id="spk_seed")],
    )

    assert result.bindings["spk_0"] == "spk_seed"
    assert "张三" in result.transcript_markdown
    assert result.resolutions[0].resolved is True


def test_assigning_to_an_existing_identity_drops_the_provisional() -> None:
    registry = SpeakerRegistry(project_id="p1")
    registry.enroll("spk_seed", "张三", VOICE_A)

    utterances = [
        __import__("attune").Utterance(0.0, 3.0, "我们继续上次的方案。", "spk_0", VOICE_MIDDLE),
    ]
    pipeline = MeetingPipeline(StaticEngine(utterances), registry)
    result = pipeline.process("meeting.wav", session_id="s1")
    provisional = result.bindings["spk_0"]

    pipeline.resolve(
        result,
        [SpeakerDecision(engine_speaker="spk_0", action="assign", identity_id="spk_seed")],
    )

    assert result.bindings["spk_0"] == "spk_seed"
    assert registry.profile(provisional) is None


def test_guess_names_from_transcript() -> None:
    names = guess_names(["张三说这个方案可行", "Alice Chen 认为进度偏慢"])
    assert "张三" in names
    assert "Alice Chen" in names
