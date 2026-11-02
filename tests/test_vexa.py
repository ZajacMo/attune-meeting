"""Tests for the Vexa integration."""

from __future__ import annotations

import json

import pytest

from attune.integrations.vexa import (
    VexaClient,
    VexaError,
    meeting_id_from_url,
    transcript_to_utterances,
)


@pytest.mark.parametrize(
    ("url", "platform", "meeting_id"),
    [
        ("https://meet.google.com/abc-defg-hij", "google_meet", "abc-defg-hij"),
        ("https://zoom.us/j/1234567890", "zoom", "1234567890"),
        ("https://zoom.us/wc/join/987654321", "zoom", "987654321"),
        ("https://meet.jit.si/TeamSync", "jitsi", "TeamSync"),
    ],
)
def test_meeting_id_from_url(url: str, platform: str, meeting_id: str) -> None:
    reference = meeting_id_from_url(url)
    assert reference.platform == platform
    assert reference.native_meeting_id == meeting_id


def test_meeting_id_from_teams_url() -> None:
    url = "https://teams.microsoft.com/l/meetup-join/19%3ameeting_ZjQxNzY4NzMtY2YzYS00ZmE3LWJhZDgtNzE5NzMzMzY5Mzk0%40thread.v2/0"
    reference = meeting_id_from_url(url)
    assert reference.platform == "teams"
    assert "meeting_" in reference.native_meeting_id


def test_unknown_url_raises() -> None:
    with pytest.raises(ValueError):
        meeting_id_from_url("https://example.com/somewhere")


def test_transcript_to_utterances_handles_segments() -> None:
    payload = {
        "segments": [
            {"start": 0.0, "end": 2.5, "text": "你好", "speaker": "A"},
            {"start": 2.5, "end": 5.0, "text": "开始吧", "speaker": "B"},
            {"start": 5.0, "end": 6.0, "text": "", "speaker": "B"},
        ]
    }
    utterances = transcript_to_utterances(payload)
    assert len(utterances) == 2
    assert utterances[0].engine_speaker == "spk_A"
    assert utterances[0].text == "你好"


def test_transcript_to_utterances_handles_bare_list() -> None:
    utterances = transcript_to_utterances([{"text": "hi", "speaker": 1}])
    assert len(utterances) == 1
    assert utterances[0].engine_speaker == "spk_1"


def test_client_requires_api_key() -> None:
    with pytest.raises(ValueError):
        VexaClient(api_key="")


def test_send_bot_posts_expected_payload() -> None:
    import httpx

    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.content)
        captured["key"] = request.headers.get("X-API-Key")
        return httpx.Response(200, json={"status": "started"})

    client = VexaClient(
        api_key="vxa_test",
        base_url="https://api.test",
        transport=httpx.MockTransport(handler),
    )
    payload = client.send_bot("google_meet", "abc-defg-hij", language="zh")

    assert payload == {"status": "started"}
    assert captured["method"] == "POST"
    assert captured["url"] == "https://api.test/bots"
    assert captured["key"] == "vxa_test"
    body = captured["body"]
    assert body["platform"] == "google_meet"
    assert body["native_meeting_id"] == "abc-defg-hij"
    assert body["language"] == "zh"
    # Attune does its own transcription, so Vexa's transcription stays off.
    assert body["transcribe_enabled"] is False


def test_client_rejects_unknown_platform() -> None:
    client = VexaClient(api_key="vxa_test", base_url="https://api.test")
    with pytest.raises(ValueError):
        client.send_bot("webex", "123")


def test_client_raises_on_api_error() -> None:
    import httpx

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="STT credentials missing")

    client = VexaClient(
        api_key="vxa_test",
        base_url="https://api.test",
        transport=httpx.MockTransport(handler),
    )
    with pytest.raises(VexaError):
        client.send_bot("google_meet", "abc")
