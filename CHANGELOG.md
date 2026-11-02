# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] — 2026-11-03

### Added

- Core library: audio normalization to 16 kHz mono WAV, pluggable transcription
  engines (FunASR, WhisperX, static/JSON fixtures) and a shared utterance model.
- Cross-session speaker identity registry: voiceprint enrolment, cosine
  matching, known / ambiguous / new outcomes, merge and rename operations, and
  per-project JSON persistence.
- Minutes generation with a per-person outline and extracted action items,
  plus a deterministic offline generator and an OpenAI-compatible one.
- Full transcript rendering with real names and timestamps.
- HTTP API (`attune.server.http`) and MCP server (`attune-mcp`) exposing the
  same operations, so any MCP-speaking host can drive it.
- Vexa integration: meeting URL parsing, bot control, transcript retrieval and
  recording listing.
- Command line interface: `attune transcribe | transcript | minutes | speakers
  | resolve | vexa | serve | mcp`.
- GitHub Actions CI covering lint, format, tests on Linux and Windows, plus
  package build and PyPI publishing on tags.
