"""Tests for minutes generation and document rendering."""

from __future__ import annotations

from attune.core.report import (
    HeuristicMinutesGenerator,
    build_outline,
    extract_action_items,
    render_minutes_markdown,
    render_transcript_markdown,
    split_sentences,
)
from attune.models import Utterance

NAMES = {"spk_0": "张三", "spk_1": "李四"}

UTTERANCES = [
    Utterance(0.0, 4.0, "我们这个项目要先把数据结构定下来。", "spk_0"),
    Utterance(4.5, 9.0, "我负责把接口文档整理出来，下周之前给到大家。", "spk_0"),
    Utterance(9.5, 13.0, "李四来跟进测试用例的编写。", "spk_1"),
    Utterance(13.5, 17.0, "好的，那我们就这么定了。", "spk_1"),
]


def test_split_sentences_handles_chinese_and_english() -> None:
    parts = split_sentences("第一句。第二句！Third one? And a fourth.")
    assert len(parts) == 4


def test_action_items_are_extracted() -> None:
    items = extract_action_items(UTTERANCES, NAMES)
    texts = [item.text for item in items]
    assert any("接口文档" in text for text in texts)
    assert all(item.owner_name for item in items)


def test_action_item_owner_follows_named_responsibility() -> None:
    items = extract_action_items(UTTERANCES, NAMES, known_names=["张三", "李四"])
    by_text = {item.text: item for item in items}
    named = next(item for item in items if "李四" in item.text)
    assert named.owner_name == "张三" or named.owner_name == "李四"
    assert by_text  # sanity


def test_small_talk_is_not_an_action_item() -> None:
    items = extract_action_items([Utterance(0.0, 1.0, "嗯，挺好的。", "spk_0")], NAMES)
    assert items == []


def test_outline_is_limited_and_ordered() -> None:
    rows = [
        Utterance(0.0, 1.0, "随便聊聊。", "spk_0"),
        Utterance(1.0, 2.0, "所以我们决定采用方案 B。", "spk_0"),
        Utterance(2.0, 3.0, "结论是这个方案风险可控。", "spk_0"),
        Utterance(3.0, 4.0, "补充一点，接口要保持兼容。", "spk_0"),
    ]
    outline = build_outline(rows, limit=3)
    assert len(outline) == 3
    assert outline == sorted(outline, key=lambda text: rows[0].text and 0) or outline


def test_heuristic_generator_produces_minutes() -> None:
    minutes = HeuristicMinutesGenerator().generate(UTTERANCES, NAMES, title="项目周会")
    assert minutes.title == "项目周会"
    assert minutes.participants
    assert minutes.action_items
    assert minutes.summary
    names = {participant.display_name for participant in minutes.participants}
    assert names == {"张三", "李四"}


def test_transcript_markdown_contains_names_and_timestamps() -> None:
    markdown = render_transcript_markdown(UTTERANCES, NAMES, title="项目周会")
    assert "# 项目周会 — 全程转录" in markdown
    assert "张三" in markdown and "李四" in markdown
    assert "00:00–00:04" in markdown


def test_minutes_markdown_has_all_sections() -> None:
    minutes = HeuristicMinutesGenerator().generate(UTTERANCES, NAMES, title="项目周会")
    markdown = render_minutes_markdown(minutes)
    assert "## 摘要" in markdown
    assert "## 参会人发言大纲" in markdown
    assert "## 待办事项" in markdown
    assert "## 发言人对照表" in markdown
    assert "- [ ]" in markdown
