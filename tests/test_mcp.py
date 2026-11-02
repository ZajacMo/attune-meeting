"""Tests for the MCP server surface."""

from __future__ import annotations

import asyncio

import pytest

from attune.core.engines import StaticEngine
from attune.core.store import Workspace
from attune.models import Utterance

pytest.importorskip("mcp", reason="The MCP server is an optional extra.")

EXPECTED_TOOLS = {
    "create_session",
    "transcribe_audio",
    "get_transcript",
    "get_minutes",
    "list_speakers",
    "get_pending_speaker_decisions",
    "resolve_speaker",
}


def _engine_factory():
    return StaticEngine(
        [
            Utterance(0.0, 3.0, "今天讨论发布计划。", "spk_0", (1.0, 0.0, 0.0)),
            Utterance(3.0, 6.0, "我负责写发布说明。", "spk_0", (1.0, 0.0, 0.0)),
            Utterance(6.0, 9.0, "测试我来跟进。", "spk_1", (0.0, 1.0, 0.0)),
        ]
    )


def test_server_exposes_the_expected_tools(tmp_path) -> None:
    from attune.server.mcp_server import build_server

    server = build_server(workspace=Workspace(tmp_path / "data"))
    tools = asyncio.run(server.list_tools())
    names = {tool.name for tool in tools}
    assert names >= EXPECTED_TOOLS


def test_mcp_sdk_version_is_detected() -> None:
    from attune.server.mcp_server import MCP_SDK_VERSION

    assert MCP_SDK_VERSION in (1, 2)


def test_tools_have_descriptions(tmp_path) -> None:
    from attune.server.mcp_server import build_server

    server = build_server(workspace=Workspace(tmp_path / "data"))
    tools = asyncio.run(server.list_tools())
    for tool in tools:
        assert tool.description, f"tool {tool.name} has no description"
