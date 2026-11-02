"""Attune — a meeting assistant that knows who is speaking.

The public surface is intentionally small:

* :mod:`attune.core.engines` — turn audio into attributed utterances
* :mod:`attune.core.identity` — remember people across meetings
* :mod:`attune.core.report` — produce minutes and a full transcript
* :mod:`attune.core.pipeline` — the end-to-end flow
* :mod:`attune.server` — MCP and HTTP interfaces for other frameworks
* :mod:`attune.integrations.vexa` — pull audio from an online meeting
"""

from __future__ import annotations

__version__ = "0.1.0"

from .core.identity import SpeakerRegistry
from .core.pipeline import MeetingPipeline, SpeakerDecision
from .core.report import (
    HeuristicMinutesGenerator,
    OpenAICompatibleMinutesGenerator,
    render_minutes_markdown,
    render_transcript_markdown,
)
from .models import (
    ActionItem,
    MeetingMinutes,
    MeetingResult,
    SpeakerBinding,
    SpeakerCandidate,
    SpeakerProfile,
    SpeakerResolution,
    Utterance,
)

__all__ = [
    "__version__",
    "ActionItem",
    "HeuristicMinutesGenerator",
    "MeetingMinutes",
    "MeetingPipeline",
    "MeetingResult",
    "OpenAICompatibleMinutesGenerator",
    "SpeakerBinding",
    "SpeakerCandidate",
    "SpeakerDecision",
    "SpeakerProfile",
    "SpeakerRegistry",
    "SpeakerResolution",
    "Utterance",
    "render_minutes_markdown",
    "render_transcript_markdown",
]
