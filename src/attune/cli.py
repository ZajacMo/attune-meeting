"""Command line interface for Attune."""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path
from typing import Any

from .core.audio import normalize_to_wav
from .core.engines import FunASREngine, JsonTranscriptEngine, StaticEngine
from .core.pipeline import MeetingPipeline, SpeakerDecision
from .core.report import (
    HeuristicMinutesGenerator,
    OpenAICompatibleMinutesGenerator,
)
from .core.store import Workspace


def _build_engine(args: argparse.Namespace) -> Any:
    if getattr(args, "transcript_json", None):
        return JsonTranscriptEngine(args.transcript_json)
    if getattr(args, "engine", "funasr") == "none":
        return StaticEngine([], name="none")
    return FunASREngine(
        model=getattr(args, "model", "paraformer-zh"),
        device=getattr(args, "device", "cpu"),
    )


def _build_generator(args: argparse.Namespace) -> Any:
    base_url = getattr(args, "llm_base_url", "") or ""
    api_key = getattr(args, "llm_api_key", "") or ""
    model = getattr(args, "llm_model", "") or ""
    if base_url and api_key and model:
        return OpenAICompatibleMinutesGenerator(base_url, api_key, model)
    return HeuristicMinutesGenerator()


def _workspace(args: argparse.Namespace) -> Workspace:
    return Workspace(Path(getattr(args, "workspace", "data")))


def _emit(payload: Any, as_json: bool = True) -> None:
    if as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(payload)


def cmd_transcribe(args: argparse.Namespace) -> int:
    workspace = _workspace(args)
    registry = workspace.load_registry(args.project)
    pipeline = MeetingPipeline(_build_engine(args), registry, _build_generator(args))

    source = Path(args.audio)
    if getattr(args, "transcript_json", ""):
        # A prepared transcript stands in for the audio file.
        audio_path = args.transcript_json
    else:
        audio_dir = workspace.audio_dir(args.project, args.session or "pending")
        audio_dir.mkdir(parents=True, exist_ok=True)
        audio_path = normalize_to_wav(source, audio_dir / "normalized.wav")

    result = pipeline.process(
        audio_path,
        session_id=args.session or None,
        title=args.title,
        project_id=args.project,
    )
    workspace.save_session(result)
    workspace.save_registry(registry)

    print(result.minutes_markdown)
    pending = [item for item in result.resolutions if not item.resolved]
    if pending:
        print("\n--- 待确认的说话人 ---", file=sys.stderr)
        for item in pending:
            candidates = ", ".join(
                f"{candidate.display_name}({candidate.score:.2f})"
                for candidate in item.candidates[:3]
            )
            print(
                f"{item.engine_speaker}: {item.kind} 建议名={item.suggested_name or '-'} "
                f"候选=[{candidates}]",
                file=sys.stderr,
            )
    return 0


def cmd_transcript(args: argparse.Namespace) -> int:
    workspace = _workspace(args)
    result = workspace.load_session(args.project, args.session)
    print(result.transcript_markdown)
    return 0


def cmd_minutes(args: argparse.Namespace) -> int:
    workspace = _workspace(args)
    result = workspace.load_session(args.project, args.session)
    print(result.minutes_markdown)
    return 0


def cmd_speakers(args: argparse.Namespace) -> int:
    workspace = _workspace(args)
    registry = workspace.load_registry(args.project)

    if args.rename:
        source, _, target = args.rename.partition("=")
        if not target:
            print("--rename expects OLD_ID=NEW_NAME", file=sys.stderr)
            return 2
        registry.rename(source, target)
        workspace.save_registry(registry)
        _emit([profile.to_dict() for profile in registry.profiles])
        return 0

    if args.merge:
        source, _, target = args.merge.partition("=")
        if not target:
            print("--merge expects SOURCE_ID=TARGET_ID", file=sys.stderr)
            return 2
        registry.merge(source, target)
        workspace.save_registry(registry)
        _emit([profile.to_dict() for profile in registry.profiles])
        return 0

    _emit(
        {
            "project_id": args.project,
            "high_threshold": registry.high_threshold,
            "low_threshold": registry.low_threshold,
            "speakers": [profile.to_dict() for profile in registry.profiles],
        }
    )
    return 0


def cmd_resolve(args: argparse.Namespace) -> int:
    workspace = _workspace(args)
    registry = workspace.load_registry(args.project)
    pipeline = MeetingPipeline(_build_engine(args), registry, _build_generator(args))
    result = workspace.load_session(args.project, args.session)
    pipeline.resolve(
        result,
        [
            SpeakerDecision(
                engine_speaker=args.engine_speaker,
                action=args.action,
                identity_id=args.identity_id,
                display_name=args.display_name,
            )
        ],
    )
    workspace.save_session(result)
    workspace.save_registry(registry)
    print(result.minutes_markdown)
    return 0


def cmd_vexa_send(args: argparse.Namespace) -> int:
    from .integrations.vexa import VexaClient, meeting_id_from_url

    reference = meeting_id_from_url(args.url)
    client = VexaClient(api_key=args.api_key, base_url=args.base_url)
    payload = client.send_bot(
        reference.platform,
        reference.native_meeting_id,
        bot_name=args.bot_name,
        language=args.language,
        transcribe_enabled=args.transcribe,
        recording_enabled=True,
    )
    _emit({"meeting": reference.to_dict(), "response": payload})
    return 0


def cmd_vexa_transcript(args: argparse.Namespace) -> int:
    from .integrations.vexa import VexaClient, meeting_id_from_url

    reference = meeting_id_from_url(args.url)
    client = VexaClient(api_key=args.api_key, base_url=args.base_url)
    _emit(client.get_transcript(reference.platform, reference.native_meeting_id))
    return 0


def cmd_vexa_stop(args: argparse.Namespace) -> int:
    from .integrations.vexa import VexaClient, meeting_id_from_url

    reference = meeting_id_from_url(args.url)
    client = VexaClient(api_key=args.api_key, base_url=args.base_url)
    _emit(client.stop_bot(reference.platform, reference.native_meeting_id))
    return 0


def cmd_serve(args: argparse.Namespace) -> int:  # pragma: no cover - process entry point
    from .server.http import serve

    serve(workspace=_workspace(args), host=args.host, port=args.port)
    return 0


def cmd_mcp(args: argparse.Namespace) -> int:  # pragma: no cover - process entry point
    from .server.mcp_server import main as mcp_main

    argv = ["--workspace", args.workspace]
    if args.http:
        argv.append("--http")
    return mcp_main(argv)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="attune",
        description="Meeting assistant with cross-session speaker identity.",
    )
    parser.add_argument("--workspace", default="data", help="Data directory.")
    sub = parser.add_subparsers(dest="command", required=True)

    transcribe = sub.add_parser("transcribe", help="Transcribe an audio file.")
    transcribe.add_argument("audio")
    transcribe.add_argument("--project", required=True)
    transcribe.add_argument("--session", default="")
    transcribe.add_argument("--title", default="")
    transcribe.add_argument("--model", default="paraformer-zh")
    transcribe.add_argument("--device", default="cpu")
    transcribe.add_argument(
        "--transcript-json", default="", help="Use a JSON transcript instead of ASR."
    )
    transcribe.add_argument("--llm-base-url", default="")
    transcribe.add_argument("--llm-api-key", default="")
    transcribe.add_argument("--llm-model", default="")
    transcribe.set_defaults(func=cmd_transcribe)

    transcript = sub.add_parser("transcript", help="Print a session transcript.")
    transcript.add_argument("project")
    transcript.add_argument("session")
    transcript.set_defaults(func=cmd_transcript)

    minutes = sub.add_parser("minutes", help="Print session minutes.")
    minutes.add_argument("project")
    minutes.add_argument("session")
    minutes.set_defaults(func=cmd_minutes)

    speakers = sub.add_parser("speakers", help="Inspect or edit known speakers.")
    speakers.add_argument("project")
    speakers.add_argument("--rename", default="", help="OLD_ID=NEW_NAME")
    speakers.add_argument("--merge", default="", help="SOURCE_ID=TARGET_ID")
    speakers.set_defaults(func=cmd_speakers)

    resolve = sub.add_parser("resolve", help="Answer a merge-or-create prompt.")
    resolve.add_argument("project")
    resolve.add_argument("session")
    resolve.add_argument("--engine-speaker", required=True)
    resolve.add_argument("--action", required=True, choices=["assign", "merge", "rename"])
    resolve.add_argument("--identity-id", default="")
    resolve.add_argument("--display-name", default="")
    resolve.add_argument("--transcript-json", default="")
    resolve.add_argument("--llm-base-url", default="")
    resolve.add_argument("--llm-api-key", default="")
    resolve.add_argument("--llm-model", default="")
    resolve.set_defaults(func=cmd_resolve)

    vexa = sub.add_parser("vexa", help="Drive an online meeting through Vexa.")
    vexa.add_argument("--api-key", default="", help="Vexa API key (or $VEXA_API_KEY).")
    vexa.add_argument("--base-url", default="https://api.cloud.vexa.ai")
    vexa_sub = vexa.add_subparsers(dest="vexa_command", required=True)

    vexa_send = vexa_sub.add_parser("send", help="Send a bot into a meeting.")
    vexa_send.add_argument("--url", required=True)
    vexa_send.add_argument("--bot-name", default="Attune")
    vexa_send.add_argument("--language", default=None)
    vexa_send.add_argument("--transcribe", action="store_true", help="Also use Vexa's transcript.")
    vexa_send.set_defaults(func=cmd_vexa_send)

    vexa_transcript = vexa_sub.add_parser("transcript", help="Fetch the live transcript.")
    vexa_transcript.add_argument("--url", required=True)
    vexa_transcript.set_defaults(func=cmd_vexa_transcript)

    vexa_stop = vexa_sub.add_parser("stop", help="Remove the bot from a meeting.")
    vexa_stop.add_argument("--url", required=True)
    vexa_stop.set_defaults(func=cmd_vexa_stop)

    serve = sub.add_parser("serve", help="Run the HTTP API.")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8787)
    serve.set_defaults(func=cmd_serve)

    mcp = sub.add_parser("mcp", help="Run the MCP server.")
    mcp.add_argument("--http", action="store_true")
    mcp.set_defaults(func=cmd_mcp)

    return parser


def main(argv: list[str] | None = None) -> int:
    # Always emit UTF-8 so the output can be piped into another tool or read
    # by an agent, regardless of the Windows console code page.
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            with contextlib.suppress(AttributeError, ValueError):
                reconfigure(encoding="utf-8")

    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "api_key", "") == "":
        import os

        args.api_key = os.environ.get("VEXA_API_KEY", "")
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
