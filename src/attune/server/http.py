"""HTTP API for Attune.

A thin JSON/Markdown interface that any framework can call. The MCP server in
:mod:`attune.server.mcp_server` exposes exactly the same operations as MCP
tools, so a consumer can pick whichever it already speaks.

Audio upload uses a **raw request body** (``audio/wav``, ``audio/mpeg``, ...)
rather than multipart, which keeps the endpoint trivially callable from any
HTTP client.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..core.audio import normalize_to_wav
from ..core.pipeline import MeetingPipeline, SpeakerDecision
from ..core.store import Workspace

try:  # pragma: no cover - import guard for the optional server extra
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import JSONResponse, PlainTextResponse
except ImportError:  # pragma: no cover
    FastAPI = HTTPException = Request = JSONResponse = PlainTextResponse = Any  # type: ignore[misc,assignment]

_AUDIO_SUFFIXES = {
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
    "audio/wave": ".wav",
    "audio/mpeg": ".mp3",
    "audio/mp3": ".mp3",
    "audio/mp4": ".m4a",
    "audio/aac": ".aac",
    "audio/ogg": ".ogg",
    "audio/flac": ".flac",
    "audio/webm": ".webm",
    "audio/x-m4a": ".m4a",
}


def _default_engine_factory() -> Any:
    from ..core.engines import FunASREngine

    return FunASREngine()


def create_app(
    *,
    workspace: Workspace | None = None,
    engine_factory: Callable[[], Any] | None = None,
    minutes_generator: Any | None = None,
) -> Any:
    """Build the FastAPI application."""
    app = FastAPI(
        title="Attune",
        version="0.1.0",
        description=(
            "Meeting assistant: speaker-aware transcription, cross-session "
            "voiceprint identity, minutes and full transcripts."
        ),
    )

    space = workspace or Workspace()
    make_engine = engine_factory or _default_engine_factory

    def build_pipeline(project_id: str) -> tuple[MeetingPipeline, Any]:
        registry = space.load_registry(project_id)
        engine = make_engine()
        return MeetingPipeline(engine, registry, minutes_generator), registry

    def persist(project_id: str, result: Any, registry: Any) -> None:
        space.save_session(result)
        space.save_registry(registry)

    # ------------------------------------------------------------------
    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "version": app.version}

    @app.get("/v1/projects")
    def list_projects() -> dict[str, Any]:
        return {"projects": space.list_projects()}

    @app.get("/v1/projects/{project_id}/speakers")
    def list_speakers(project_id: str) -> dict[str, Any]:
        registry = space.load_registry(project_id)
        return {
            "project_id": project_id,
            "high_threshold": registry.high_threshold,
            "low_threshold": registry.low_threshold,
            "speakers": [profile.to_dict() for profile in registry.profiles],
        }

    @app.post("/v1/projects/{project_id}/sessions")
    def create_session(project_id: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        body = payload or {}
        session_id = str(body.get("session_id") or f"ses_{uuid.uuid4().hex[:10]}")
        title = str(body.get("title", ""))
        space.session_dir(project_id, session_id).mkdir(parents=True, exist_ok=True)
        return {"project_id": project_id, "session_id": session_id, "title": title}

    @app.get("/v1/projects/{project_id}/sessions")
    def list_sessions(project_id: str) -> dict[str, Any]:
        return {"project_id": project_id, "sessions": space.list_sessions(project_id)}

    @app.post("/v1/projects/{project_id}/sessions/{session_id}/audio")
    async def upload_audio(project_id: str, session_id: str, request: Request) -> JSONResponse:
        content_type = (request.headers.get("content-type") or "").split(";")[0].strip()
        body = await request.body()
        if not body:
            raise HTTPException(status_code=400, detail="Empty audio body.")

        suffix = _AUDIO_SUFFIXES.get(content_type, ".wav")
        filename = request.query_params.get("filename")
        if filename:
            suffix = Path(filename).suffix or suffix

        audio_dir = space.audio_dir(project_id, session_id)
        audio_dir.mkdir(parents=True, exist_ok=True)
        raw_path = audio_dir / f"source{suffix}"
        raw_path.write_bytes(body)

        try:
            wav_path = normalize_to_wav(raw_path, audio_dir / "normalized.wav")
        except Exception as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

        try:
            pipeline, registry = build_pipeline(project_id)
            result = pipeline.process(
                wav_path,
                session_id=session_id,
                title=request.query_params.get("title", ""),
                project_id=project_id,
            )
        except RuntimeError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error

        persist(project_id, result, registry)
        return JSONResponse(result.to_dict())

    @app.get("/v1/projects/{project_id}/sessions/{session_id}/transcript")
    def get_transcript(project_id: str, session_id: str, format: str = "json") -> Any:
        try:
            result = space.load_session(project_id, session_id)
        except FileNotFoundError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        if format == "markdown":
            return PlainTextResponse(result.transcript_markdown, media_type="text/markdown")
        return {
            "session_id": result.session_id,
            "project_id": project_id,
            "markdown": result.transcript_markdown,
            "utterances": [utterance.to_dict() for utterance in result.utterances],
            "bindings": result.bindings,
        }

    @app.get("/v1/projects/{project_id}/sessions/{session_id}/minutes")
    def get_minutes(project_id: str, session_id: str, format: str = "json") -> Any:
        try:
            result = space.load_session(project_id, session_id)
        except FileNotFoundError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        if format == "markdown":
            return PlainTextResponse(result.minutes_markdown, media_type="text/markdown")
        return {
            "session_id": result.session_id,
            "project_id": project_id,
            "markdown": result.minutes_markdown,
            "minutes": result.minutes.to_dict(),
        }

    @app.get("/v1/projects/{project_id}/sessions/{session_id}/speakers")
    def session_speakers(project_id: str, session_id: str) -> dict[str, Any]:
        try:
            result = space.load_session(project_id, session_id)
        except FileNotFoundError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        registry = space.load_registry(project_id)
        return {
            "session_id": session_id,
            "bindings": result.bindings,
            "pending": [
                resolution.to_dict() for resolution in result.resolutions if not resolution.resolved
            ],
            "resolved": [
                resolution.to_dict() for resolution in result.resolutions if resolution.resolved
            ],
            "speakers": [profile.to_dict() for profile in registry.profiles],
        }

    @app.post("/v1/projects/{project_id}/sessions/{session_id}/speakers/resolve")
    def resolve_speakers(
        project_id: str, session_id: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        try:
            result = space.load_session(project_id, session_id)
        except FileNotFoundError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

        decisions = [
            SpeakerDecision(
                engine_speaker=str(row.get("engine_speaker", "")),
                action=str(row.get("action", "assign")),
                identity_id=str(row.get("identity_id", "")),
                display_name=str(row.get("display_name", "")),
            )
            for row in payload.get("decisions", [])
        ]
        if not decisions:
            raise HTTPException(status_code=400, detail="No decisions supplied.")

        registry = space.load_registry(project_id)
        pipeline = MeetingPipeline(make_engine(), registry, minutes_generator)
        pipeline.resolve(result, decisions)
        persist(project_id, result, registry)
        return result.to_dict()

    return app


def serve(
    *,
    workspace: Workspace | None = None,
    host: str = "127.0.0.1",
    port: int = 8787,
    engine_factory: Callable[[], Any] | None = None,
) -> None:  # pragma: no cover - process entry point
    import uvicorn

    uvicorn.run(
        create_app(workspace=workspace, engine_factory=engine_factory),
        host=host,
        port=port,
    )
