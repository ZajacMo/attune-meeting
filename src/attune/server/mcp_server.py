"""MCP server exposing Attune as tools.

Any MCP-speaking host — Cherry Studio, Claude Desktop, Hermes, or an agent
runtime — can add this server and drive a meeting with it.

Run it over stdio (the usual MCP transport)::

    attune-mcp --workspace ./data

or over streamable HTTP::

    attune-mcp --http --port 8788
"""

from __future__ import annotations

import argparse
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..core.pipeline import MeetingPipeline, SpeakerDecision
from ..core.store import Workspace

try:  # MCP SDK 2.x renamed FastMCP to MCPServer.
    from mcp.server.mcpserver import MCPServer as _ServerClass

    MCP_SDK_VERSION = 2
except ImportError:  # pragma: no cover - MCP SDK 1.x
    from mcp.server.fastmcp import FastMCP as _ServerClass

    MCP_SDK_VERSION = 1


def _default_engine_factory() -> Any:
    from ..core.engines import FunASREngine

    return FunASREngine()


def build_server(
    *,
    workspace: Workspace | None = None,
    engine_factory: Callable[[], Any] | None = None,
    minutes_generator: Any | None = None,
) -> Any:
    """Build the MCP server with all Attune tools registered."""
    space = workspace or Workspace()
    make_engine = engine_factory or _default_engine_factory

    server = _ServerClass("attune")

    def pipeline_for(project_id: str) -> tuple[MeetingPipeline, Any]:
        registry = space.load_registry(project_id)
        return MeetingPipeline(make_engine(), registry, minutes_generator), registry

    @server.tool()
    def create_session(project_id: str, title: str = "", session_id: str = "") -> dict[str, Any]:
        """Create a meeting session to attach audio to."""
        resolved = session_id or f"ses_{uuid.uuid4().hex[:10]}"
        space.session_dir(project_id, resolved).mkdir(parents=True, exist_ok=True)
        return {"project_id": project_id, "session_id": resolved, "title": title}

    @server.tool()
    def transcribe_audio(
        project_id: str,
        audio_path: str,
        session_id: str = "",
        title: str = "",
    ) -> dict[str, Any]:
        """Transcribe a local audio file, resolve speakers and produce both documents."""
        resolved = session_id or f"ses_{uuid.uuid4().hex[:10]}"
        pipeline, registry = pipeline_for(project_id)
        result = pipeline.process(
            audio_path, session_id=resolved, title=title, project_id=project_id
        )
        space.save_session(result)
        space.save_registry(registry)
        return result.to_dict()

    @server.tool()
    def get_transcript(project_id: str, session_id: str) -> dict[str, Any]:
        """Return the full speaker-attributed transcript of a session."""
        result = space.load_session(project_id, session_id)
        return {
            "session_id": session_id,
            "markdown": result.transcript_markdown,
            "utterances": [utterance.to_dict() for utterance in result.utterances],
        }

    @server.tool()
    def get_minutes(project_id: str, session_id: str) -> dict[str, Any]:
        """Return the meeting minutes: per-person outline plus action items."""
        result = space.load_session(project_id, session_id)
        return {
            "session_id": session_id,
            "markdown": result.minutes_markdown,
            "minutes": result.minutes.to_dict(),
        }

    @server.tool()
    def list_speakers(project_id: str) -> dict[str, Any]:
        """List every identity known for a project, with match thresholds."""
        registry = space.load_registry(project_id)
        return {
            "project_id": project_id,
            "high_threshold": registry.high_threshold,
            "low_threshold": registry.low_threshold,
            "speakers": [profile.to_dict() for profile in registry.profiles],
        }

    @server.tool()
    def get_pending_speaker_decisions(project_id: str, session_id: str) -> dict[str, Any]:
        """List the speakers that still need a merge-or-create answer."""
        result = space.load_session(project_id, session_id)
        return {
            "session_id": session_id,
            "pending": [
                resolution.to_dict() for resolution in result.resolutions if not resolution.resolved
            ],
        }

    @server.tool()
    def resolve_speaker(
        project_id: str,
        session_id: str,
        engine_speaker: str,
        action: str,
        identity_id: str = "",
        display_name: str = "",
    ) -> dict[str, Any]:
        """Answer a merge-or-create prompt.

        ``action`` is one of:

        * ``assign`` — this recording-local speaker is ``identity_id``
        * ``merge``  — fold this speaker into ``identity_id``
        * ``rename`` — keep them separate but call them ``display_name``
        """
        pipeline, registry = pipeline_for(project_id)
        result = space.load_session(project_id, session_id)
        pipeline.resolve(
            result,
            [
                SpeakerDecision(
                    engine_speaker=engine_speaker,
                    action=action,
                    identity_id=identity_id,
                    display_name=display_name,
                )
            ],
        )
        space.save_session(result)
        space.save_registry(registry)
        return {
            "session_id": session_id,
            "engine_speaker": engine_speaker,
            "identity_id": result.bindings.get(engine_speaker, ""),
            "pending": [
                resolution.to_dict() for resolution in result.resolutions if not resolution.resolved
            ],
        }

    return server


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - process entry point
    parser = argparse.ArgumentParser(prog="attune-mcp", description="Run the Attune MCP server.")
    parser.add_argument(
        "--workspace",
        default="data",
        help="Directory holding projects, sessions and speaker registries.",
    )
    parser.add_argument(
        "--http",
        action="store_true",
        help="Serve over streamable HTTP instead of stdio.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8788)
    args = parser.parse_args(argv)

    server = build_server(workspace=Workspace(Path(args.workspace)))
    if args.http:
        if MCP_SDK_VERSION >= 2:
            server.run(transport="streamable-http", host=args.host, port=args.port)
        else:  # pragma: no cover - MCP SDK 1.x takes the host/port elsewhere
            server.settings.host = args.host
            server.settings.port = args.port
            server.run(transport="streamable-http")
    else:
        server.run()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
