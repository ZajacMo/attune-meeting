"""Tests for the HTTP API and the workspace store."""

from __future__ import annotations

import pytest

from attune.core.engines import StaticEngine
from attune.core.store import Workspace
from attune.models import Utterance

VOICE_A = (1.0, 0.0, 0.0)
VOICE_B = (0.0, 1.0, 0.0)

fastapi = pytest.importorskip("fastapi", reason="FastAPI is an optional server extra.")


def _wav_bytes() -> bytes:
    import io
    import wave

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\x00\x00" * 1600)
    return buffer.getvalue()


def _engine_factory():
    return StaticEngine(
        [
            Utterance(0.0, 3.0, "今天讨论发布计划。", "spk_0", VOICE_A),
            Utterance(3.0, 6.0, "我负责写发布说明。", "spk_0", VOICE_A),
            Utterance(6.0, 9.0, "测试我来跟进。", "spk_1", VOICE_B),
        ]
    )


def _client(tmp_path):
    from fastapi.testclient import TestClient

    from attune.server.http import create_app

    workspace = Workspace(tmp_path / "data")
    app = create_app(workspace=workspace, engine_factory=_engine_factory, minutes_generator=None)
    return TestClient(app), workspace


def test_health_endpoint(tmp_path) -> None:
    client, _ = _client(tmp_path)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_full_meeting_flow(tmp_path) -> None:
    client, workspace = _client(tmp_path)

    created = client.post("/v1/projects/p1/sessions", json={"title": "周会"})
    assert created.status_code == 200
    session_id = created.json()["session_id"]

    upload = client.post(
        f"/v1/projects/p1/sessions/{session_id}/audio",
        content=_wav_bytes(),
        headers={"content-type": "audio/wav"},
    )
    assert upload.status_code == 200
    payload = upload.json()
    assert payload["session_id"] == session_id
    assert len(payload["resolutions"]) == 2

    transcript = client.get(f"/v1/projects/p1/sessions/{session_id}/transcript")
    assert transcript.status_code == 200
    assert "00:00" in transcript.json()["markdown"]

    markdown = client.get(
        f"/v1/projects/p1/sessions/{session_id}/transcript", params={"format": "markdown"}
    )
    assert markdown.status_code == 200
    assert markdown.headers["content-type"].startswith("text/markdown")

    speakers = client.get(f"/v1/projects/p1/sessions/{session_id}/speakers")
    assert speakers.status_code == 200
    assert len(speakers.json()["pending"]) == 2

    resolved = client.post(
        f"/v1/projects/p1/sessions/{session_id}/speakers/resolve",
        json={
            "decisions": [
                {"engine_speaker": "spk_0", "action": "rename", "display_name": "张三"},
                {"engine_speaker": "spk_1", "action": "rename", "display_name": "李四"},
            ]
        },
    )
    assert resolved.status_code == 200
    assert resolved.json()["resolutions"][0]["resolved"] is True

    minutes = client.get(
        f"/v1/projects/p1/sessions/{session_id}/minutes", params={"format": "markdown"}
    )
    assert minutes.status_code == 200
    assert "张三" in minutes.text

    registry = workspace.load_registry("p1")
    assert {profile.display_name for profile in registry.profiles} == {"张三", "李四"}


def test_unknown_session_returns_404(tmp_path) -> None:
    client, _ = _client(tmp_path)
    response = client.get("/v1/projects/p1/sessions/nope/transcript")
    assert response.status_code == 404


def test_empty_audio_body_returns_400(tmp_path) -> None:
    client, _ = _client(tmp_path)
    created = client.post("/v1/projects/p1/sessions", json={})
    session_id = created.json()["session_id"]
    response = client.post(
        f"/v1/projects/p1/sessions/{session_id}/audio",
        content=b"",
        headers={"content-type": "audio/wav"},
    )
    assert response.status_code == 400


def test_workspace_round_trip(tmp_path) -> None:
    from attune.core.identity import SpeakerRegistry
    from attune.models import MeetingResult

    workspace = Workspace(tmp_path / "data")
    registry = SpeakerRegistry(project_id="p1")
    registry.enroll("spk_a", "张三", VOICE_A)
    workspace.save_registry(registry)

    result = MeetingResult(
        session_id="s1",
        project_id="p1",
        utterances=[Utterance(0.0, 1.0, "你好", "spk_0", VOICE_A)],
        bindings={"spk_0": "spk_a"},
        title="周会",
    )
    workspace.save_session(result)

    assert workspace.list_sessions("p1") == ["s1"]
    loaded = workspace.load_session("p1", "s1")
    assert loaded.title == "周会"
    assert loaded.bindings == {"spk_0": "spk_a"}
    assert loaded.utterances[0].text == "你好"
    assert workspace.load_registry("p1").profile("spk_a") is not None
