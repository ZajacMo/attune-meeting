"""Minutes and transcript rendering.

Two documents come out of every meeting:

``transcript``
    The full verbatim transcript, tagged with a real name per speaker.

``minutes``
    A summary with a per-person outline of what they said and the concrete
    action items that came out of the meeting.

The generators are pluggable. The heuristic one is deterministic and works
offline, which keeps the pipeline testable and usable without a model. The
OpenAI-compatible one asks a model for a structured summary and degrades to
the heuristic result when the call fails.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from ..models import ActionItem, MeetingMinutes, ParticipantSummary, Utterance

#: Phrases that indicate a commitment rather than small talk.
ACTION_PATTERNS: tuple[str, ...] = (
    "负责",
    "跟进",
    "来完成",
    "完成",
    "提交",
    "准备",
    "输出",
    "安排",
    "确认一下",
    "我来",
    "你来",
    "我们下",
    "截止",
    "deadline",
    "follow up",
    "follow-up",
    "action item",
    "todo",
    "to-do",
    "i'll",
    "i will",
    "can you",
    "please ",
    "make sure",
    "assign",
    "owner",
    "take care of",
    "by monday",
    "by tuesday",
    "by wednesday",
    "by thursday",
    "by friday",
    "next week",
    "next month",
)

#: Words that make an utterance worth quoting as part of someone's outline.
_OUTLINE_MARKERS: tuple[str, ...] = (
    "所以",
    "因此",
    "结论",
    "决定",
    "方案",
    "问题",
    "原因",
    "风险",
    "计划",
    "目标",
    "需求",
    "设计",
    "实现",
    "测试",
    "the reason",
    "so the",
    "decision",
    "the plan",
    "the issue",
    "the risk",
    "we need",
    "the design",
)


@runtime_checkable
class MinutesGenerator(Protocol):
    """Produces a :class:`MeetingMinutes` from attributed utterances."""

    name: str

    def generate(
        self,
        utterances: Sequence[Utterance],
        speaker_names: dict[str, str],
        *,
        title: str = "",
    ) -> MeetingMinutes: ...


def _format_timestamp(seconds: float) -> str:
    total = max(0, int(seconds))
    return f"{total // 60:02d}:{total % 60:02d}"


def _looks_like_action(text: str) -> bool:
    lowered = text.lower()
    return any(pattern in lowered for pattern in ACTION_PATTERNS)


def _looks_like_outline(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in _OUTLINE_MARKERS)


def split_sentences(text: str) -> list[str]:
    """Split Chinese/English text into sentences on common terminators."""
    parts = re.split(r"(?<=[。！？!?；;\n])|(?<=[.!?])\s+", text)
    return [part.strip() for part in parts if part and part.strip()]


def extract_action_items(
    utterances: Sequence[Utterance],
    speaker_names: dict[str, str],
    *,
    known_names: Sequence[str] = (),
) -> list[ActionItem]:
    """Heuristically pull commitments out of the transcript.

    Ownership defaults to whoever made the statement. When the sentence names
    another participant next to a commitment verb, that person is used instead
    and the confidence is raised.
    """
    name_by_identity = dict(speaker_names)
    items: list[ActionItem] = []
    seen: set[str] = set()

    for utterance in utterances:
        speaker_identity = utterance.engine_speaker
        for sentence in split_sentences(utterance.text):
            if len(sentence) < 4 or not _looks_like_action(sentence):
                continue
            key = sentence.strip().lower()
            if key in seen:
                continue
            seen.add(key)

            owner_identity = speaker_identity
            owner_name = name_by_identity.get(speaker_identity, speaker_identity)
            confidence = 0.55

            for candidate_name in known_names:
                if candidate_name and candidate_name in sentence:
                    for identity, display in name_by_identity.items():
                        if display == candidate_name:
                            owner_identity = identity
                            owner_name = display
                            confidence = 0.8
                            break
                    break

            items.append(
                ActionItem(
                    text=sentence.strip(),
                    owner_identity_id=owner_identity,
                    owner_name=owner_name,
                    source_quote=utterance.text.strip(),
                    confidence=confidence,
                )
            )
    return items


def build_outline(utterances: Sequence[Utterance], limit: int = 3) -> list[str]:
    """Pick a few representative utterances for one person's outline."""
    if not utterances:
        return []
    ranked = sorted(
        utterances,
        key=lambda utterance: (
            0 if _looks_like_outline(utterance.text) else 1,
            -len(utterance.text),
            utterance.start,
        ),
    )
    chosen = ranked[:limit]
    chosen.sort(key=lambda utterance: utterance.start)
    return [utterance.text.strip() for utterance in chosen if utterance.text.strip()]


class HeuristicMinutesGenerator:
    """Deterministic, offline minutes built from statistics and keywords."""

    name = "heuristic"

    def generate(
        self,
        utterances: Sequence[Utterance],
        speaker_names: dict[str, str],
        *,
        title: str = "",
    ) -> MeetingMinutes:
        by_speaker: dict[str, list[Utterance]] = {}
        for utterance in utterances:
            by_speaker.setdefault(utterance.engine_speaker, []).append(utterance)

        participants: list[ParticipantSummary] = []
        for engine_speaker, rows in by_speaker.items():
            display_name = speaker_names.get(engine_speaker, engine_speaker)
            participants.append(
                ParticipantSummary(
                    identity_id=engine_speaker,
                    display_name=display_name,
                    speaking_seconds=sum(row.duration for row in rows),
                    utterance_count=len(rows),
                    outline=build_outline(rows),
                )
            )
        participants.sort(key=lambda item: item.speaking_seconds, reverse=True)

        known_names = [participant.display_name for participant in participants]
        action_items = extract_action_items(
            utterances, speaker_names, known_names=[name for name in known_names if name]
        )

        total_seconds = sum(participant.speaking_seconds for participant in participants)
        summary = (
            f"本次会议共 {len(participants)} 位发言人，"
            f"累计发言约 {total_seconds / 60:.1f} 分钟，"
            f"识别出 {len(action_items)} 项待办。"
        )

        return MeetingMinutes(
            title=title or "会议纪要",
            summary=summary,
            participants=participants,
            action_items=action_items,
            generated_by=self.name,
        )


class OpenAICompatibleMinutesGenerator:
    """Asks an OpenAI-compatible chat endpoint for a structured summary.

    Falls back to the heuristic generator whenever the request fails, so a
    flaky or missing model never costs the user their minutes.
    """

    name = "llm"

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        *,
        timeout: float = 60.0,
        fallback: MinutesGenerator | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.fallback = fallback or HeuristicMinutesGenerator()

    def _prompt(self, utterances: Sequence[Utterance], speaker_names: dict[str, str]) -> str:
        lines = []
        for utterance in utterances:
            name = speaker_names.get(utterance.engine_speaker, utterance.engine_speaker)
            lines.append(f"[{_format_timestamp(utterance.start)}] {name}: {utterance.text}")
        transcript = "\n".join(lines)
        return (
            "下面是一场会议的逐句转录，每行格式为 [时间] 发言人: 内容。\n"
            "请输出 JSON（不要包含任何解释文字），结构为：\n"
            '{"summary": "整体摘要", '
            '"participants": [{"identity_id": "发言人ID", "display_name": "姓名", '
            '"outline": ["该发言人主要观点，最多3条"]}], '
            '"action_items": [{"text": "待办事项", "owner_identity_id": "负责人ID或null", '
            '"owner_name": "负责人姓名", "source_quote": "原文依据"}]}\n'
            "要求：只使用转录中出现的事实；待办必须是明确的行动承诺，不要把讨论当承诺。\n\n"
            f"转录：\n{transcript}"
        )

    def generate(
        self,
        utterances: Sequence[Utterance],
        speaker_names: dict[str, str],
        *,
        title: str = "",
    ) -> MeetingMinutes:
        if not utterances:
            return self.fallback.generate(utterances, speaker_names, title=title)
        try:
            import httpx

            response = httpx.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "messages": [
                        {
                            "role": "user",
                            "content": self._prompt(utterances, speaker_names),
                        }
                    ],
                    "temperature": 0.2,
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
            payload = response.json()
            content = payload["choices"][0]["message"]["content"]
            data = json.loads(_extract_json(content))
        except Exception:
            return self.fallback.generate(utterances, speaker_names, title=title)

        participants = [
            ParticipantSummary(
                identity_id=str(row.get("identity_id", "")),
                display_name=str(row.get("display_name", "")),
                outline=[str(item) for item in row.get("outline", []) if str(item).strip()],
            )
            for row in data.get("participants", [])
            if isinstance(row, dict)
        ]
        if not participants:
            return self.fallback.generate(utterances, speaker_names, title=title)

        by_speaker: dict[str, list[Utterance]] = {}
        for utterance in utterances:
            by_speaker.setdefault(utterance.engine_speaker, []).append(utterance)
        for participant in participants:
            rows = by_speaker.get(participant.identity_id, [])
            participant.speaking_seconds = sum(row.duration for row in rows)
            participant.utterance_count = len(rows)

        action_items = [
            ActionItem(
                text=str(row.get("text", "")).strip(),
                owner_identity_id=(
                    str(row["owner_identity_id"]) if row.get("owner_identity_id") else None
                ),
                owner_name=str(row.get("owner_name", "")),
                source_quote=str(row.get("source_quote", "")),
                confidence=0.9,
            )
            for row in data.get("action_items", [])
            if isinstance(row, dict) and str(row.get("text", "")).strip()
        ]

        return MeetingMinutes(
            title=title or "会议纪要",
            summary=str(data.get("summary", "")).strip(),
            participants=participants,
            action_items=action_items,
            generated_by=self.name,
        )


def _extract_json(content: str) -> str:
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("Model response did not contain a JSON object")
    return text[start : end + 1]


def render_transcript_markdown(
    utterances: Sequence[Utterance], speaker_names: dict[str, str], *, title: str = ""
) -> str:
    """Render the full transcript with real names and timestamps."""
    header = title or "全程转录"
    lines = [f"# {header} — 全程转录", ""]
    if speaker_names:
        roster = "、".join(sorted({name for name in speaker_names.values() if name}))
        if roster:
            lines.append(f"**发言人：** {roster}")
            lines.append("")
    for utterance in utterances:
        name = speaker_names.get(utterance.engine_speaker, utterance.engine_speaker)
        stamp = f"{_format_timestamp(utterance.start)}–{_format_timestamp(utterance.end)}"
        lines.append(f"## {stamp} · {name}")
        lines.append("")
        lines.append(utterance.text.strip())
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render_minutes_markdown(minutes: MeetingMinutes) -> str:
    """Render the summary document."""
    lines = [f"# {minutes.title or '会议纪要'}", ""]
    if minutes.summary:
        lines += ["## 摘要", "", minutes.summary, ""]

    lines += ["## 参会人发言大纲", ""]
    if minutes.participants:
        for participant in minutes.participants:
            spoken = participant.speaking_seconds / 60.0
            lines.append(
                f"### {participant.display_name}（发言 {spoken:.1f} 分钟，{participant.utterance_count} 次）"
            )
            lines.append("")
            if participant.outline:
                for item in participant.outline:
                    lines.append(f"- {item}")
            else:
                lines.append("- （本次发言较短，没有可提炼的观点）")
            lines.append("")
    else:
        lines += ["（无发言记录）", ""]

    lines += ["## 待办事项", ""]
    if minutes.action_items:
        for item in minutes.action_items:
            owner = item.owner_name or "待定"
            lines.append(f"- [ ] {item.text} — 负责人：{owner}")
            if item.source_quote:
                lines.append(f"  > 依据：{item.source_quote}")
    else:
        lines.append("（未识别到明确的待办事项）")
    lines.append("")

    lines += [
        "## 发言人对照表",
        "",
        "| 发言人 | 发言时长（分钟） | 发言次数 |",
        "| --- | ---: | ---: |",
    ]
    for participant in minutes.participants:
        lines.append(
            f"| {participant.display_name} | {participant.speaking_seconds / 60.0:.1f} "
            f"| {participant.utterance_count} |"
        )
    lines.append("")
    return "\n".join(lines).rstrip() + "\n"
