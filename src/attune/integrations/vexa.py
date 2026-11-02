"""Client for Vexa's meeting-bot API.

Vexa (Apache-2.0) runs bots inside Google Meet, Microsoft Teams, Zoom and
Jitsi and exposes the audio and the attributed transcript over a small HTTP
API. Attune uses it purely as an **audio source** — the diarization, the
identity registry and the minutes are all produced by Attune itself.

Quickstart::

    client = VexaClient(api_key="vxa_...")
    platform, meeting_id = meeting_id_from_url("https://meet.google.com/abc-defg-hij")
    client.send_bot(platform, meeting_id, language="zh")
    transcript = client.get_transcript(platform, meeting_id)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import unquote, urlparse

from ..models import Utterance

PLATFORMS: tuple[str, ...] = ("google_meet", "teams", "zoom", "jitsi")

DEFAULT_BASE_URL = "https://api.cloud.vexa.ai"

_GOOGLE_MEET = re.compile(r"meet\.google\.com/(?P<id>[a-z0-9-]+)", re.IGNORECASE)
_ZOOM = re.compile(r"zoom\.us/(?:j|wc/join)/(?P<id>[0-9]+)", re.IGNORECASE)
_TEAMS = re.compile(r"19%3ameeting_[^/]+|19:meeting_[^/\s]+", re.IGNORECASE)
_JITSI = re.compile(r"meet\.jit\.si/(?P<id>[^/?#]+)", re.IGNORECASE)


class VexaError(RuntimeError):
    """Raised when the Vexa API returns an error or an unexpected payload."""


@dataclass(frozen=True)
class MeetingRef:
    """A meeting identified by platform and native meeting id."""

    platform: str
    native_meeting_id: str

    def to_dict(self) -> dict[str, str]:
        return {"platform": self.platform, "native_meeting_id": self.native_meeting_id}


def meeting_id_from_url(url: str) -> MeetingRef:
    """Extract the platform and meeting id from a meeting URL.

    Supported: Google Meet, Microsoft Teams, Zoom, Jitsi.
    """
    cleaned = url.strip()
    parsed = urlparse(cleaned if "//" in cleaned else f"https://{cleaned}")
    host = (parsed.netloc or "").lower()

    if "meet.google.com" in host:
        match = _GOOGLE_MEET.search(cleaned)
        if match:
            return MeetingRef("google_meet", match.group("id"))
    if "zoom.us" in host:
        match = _ZOOM.search(cleaned)
        if match:
            return MeetingRef("zoom", match.group("id"))
    if "teams.microsoft.com" in host or "teams.live.com" in host:
        match = _TEAMS.search(unquote(cleaned))
        if match:
            return MeetingRef("teams", match.group(0))
    if "jit.si" in host:
        match = _JITSI.search(cleaned)
        if match:
            return MeetingRef("jitsi", match.group("id"))

    raise ValueError(f"Cannot derive a meeting id from: {url}")


class VexaClient:
    """Thin HTTP client over the Vexa meetings API."""

    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 30.0,
        transport: Any | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("A Vexa API key is required.")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._transport = transport

    # ------------------------------------------------------------------
    def _client(self) -> Any:
        import httpx

        if self._transport is not None:
            return httpx.Client(
                base_url=self.base_url,
                timeout=self.timeout,
                headers={"X-API-Key": self.api_key},
                transport=self._transport,
            )
        return httpx.Client(
            base_url=self.base_url,
            timeout=self.timeout,
            headers={"X-API-Key": self.api_key},
        )

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        with self._client() as client:
            response = client.request(method, path, **kwargs)
            if response.status_code >= 400:
                raise VexaError(
                    f"Vexa API {method} {path} failed with "
                    f"{response.status_code}: {response.text[:300]}"
                )
            if not response.content:
                return {}
            try:
                return response.json()
            except ValueError:
                return {"raw": response.text}

    # ------------------------------------------------------------------
    # Meetings API
    # ------------------------------------------------------------------
    def send_bot(
        self,
        platform: str,
        native_meeting_id: str,
        *,
        bot_name: str = "Attune",
        language: str | None = None,
        task: str = "transcribe",
        transcribe_enabled: bool = False,
        recording_enabled: bool = True,
    ) -> dict[str, Any]:
        """Send a bot into a meeting.

        ``transcribe_enabled`` defaults to ``False`` on purpose: Attune does
        its own transcription and diarization, so paying Vexa's transcription
        unit would duplicate the work. Turn it on only if you want Vexa's
        transcript as a cross-check.
        """
        if platform not in PLATFORMS:
            raise ValueError(f"Unsupported platform: {platform}. Use one of {PLATFORMS}.")
        payload: dict[str, Any] = {
            "platform": platform,
            "native_meeting_id": native_meeting_id,
            "bot_name": bot_name,
            "task": task,
            "transcribe_enabled": transcribe_enabled,
            "recording_enabled": recording_enabled,
        }
        if language:
            payload["language"] = language
        return self._request("POST", "/bots", json=payload)

    def get_transcript(self, platform: str, native_meeting_id: str) -> dict[str, Any]:
        """Fetch the live transcript for a meeting."""
        return self._request("GET", f"/transcripts/{platform}/{native_meeting_id}")

    def stop_bot(self, platform: str, native_meeting_id: str) -> dict[str, Any]:
        """Remove the bot from a meeting."""
        return self._request("DELETE", f"/bots/{platform}/{native_meeting_id}")

    def list_bots(self) -> list[dict[str, Any]]:
        payload = self._request("GET", "/bots/status")
        return _as_list(payload)

    def list_meetings(self) -> list[dict[str, Any]]:
        payload = self._request("GET", "/meetings")
        return _as_list(payload)

    def list_recordings(self) -> list[dict[str, Any]]:
        payload = self._request("GET", "/recordings")
        return _as_list(payload)


def _as_list(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        for key in ("data", "items", "results", "meetings", "recordings"):
            value = payload.get(key)
            if isinstance(value, list):
                return [row for row in value if isinstance(row, dict)]
    return []


def transcript_to_utterances(payload: Any) -> list[Utterance]:
    """Convert a Vexa transcript payload into Attune utterances.

    Vexa's transcript shape is documented as a live, speaker-attributed
    stream; this parser accepts the common segment shapes defensively and
    ignores anything it does not recognise rather than guessing.
    """
    rows: list[Any] = []
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        for key in ("segments", "transcript", "data", "parts", "utterances"):
            value = payload.get(key)
            if isinstance(value, list):
                rows = value
                break
        else:
            rows = [payload]

    utterances: list[Utterance] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        text = str(row.get("text", "") or row.get("content", "")).strip()
        if not text:
            continue
        speaker = row.get("speaker", row.get("speaker_id", row.get("spk", index)))
        start = _as_float(row.get("start", row.get("start_time", 0.0)))
        end = _as_float(row.get("end", row.get("end_time", start)))
        utterances.append(
            Utterance(
                start=start,
                end=end,
                text=text,
                engine_speaker=f"spk_{speaker}",
            )
        )
    return utterances


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
