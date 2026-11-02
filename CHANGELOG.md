# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Reproducible calibration.** `experiments/` holds the scripts behind every
  number in `docs/voiceprint-calibration.md`: sample selection, cached voiceprint
  extraction, threshold calibration, diarization on stitched meetings, and a
  measurement of how many rows the diarization engine actually clusters.

### Changed

- **Voiceprint thresholds recalibrated.** The shipped defaults are now
  `high_threshold = 0.68` and `low_threshold = 0.53`, measured on 126 real
  Mandarin meeting clips from 14 speakers across 7 sessions. The previous
  values (0.82 / 0.62) were guesses: on that data 0.82 accepts only 14.8% of
  genuine same-speaker pairs. See `docs/voiceprint-calibration.md`.
- **An identity is scored against the centroid of its stored voiceprints**
  instead of the best of them. Taking the best of N voiceprints lets an
  impostor's score drift upwards as the registry grows, so one fixed threshold
  would stop meaning anything over time.

### Fixed

- `FunASREngine` now reads the speaker centroids from `spk_embedding_center`
  (requested with `return_spk_center=True`). The field it used to read,
  `sentence["spk_embedding"]`, does not exist in the engine output, so every
  utterance was being produced without a voiceprint.

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
