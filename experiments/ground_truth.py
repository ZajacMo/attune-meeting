"""Join meta.csv with the REAL-PS4 file list to recover mixture composition.

AISHELL-4 mixtures are built by overlapping *enrolment* utterances. The metadata
table records, for every mixture and every microphone channel, which enrolment
utterance was mixed in (``enrolment_speakers_utterance``), who said it
(``speaker``) and what it said (``ground_truth_transcript``).

The table uses a different date/microphone prefix than the audio files, so the
join has to happen on ``(session, speaker, start, end)``.

This module produces, per mixture, the list of constituent utterances with their
speaker and transcript — the ground truth needed to tell which diarization cluster
belongs to whom.
"""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from pathlib import Path

ENROLMENT = re.compile(
    r"AISHELL-4_(?P<date>\d+)_[A-Z]_(?P<session>R\d+S\d+C\d+)_(?P<speaker>\d+-[MFU])_"
    r"(?P<start>[\d.]+)_(?P<end>[\d.]+)\.wav$"
)
MIXTURE = re.compile(
    r"AISHELL-4_(?P<date>\d+)_[A-Z]_(?P<session>R\d+S\d+C\d+)_mixture_"
    r"(?P<start>[\d.]+)_(?P<end>[\d.]+)\.wav$"
)
META_MIXTURE = re.compile(
    r"AISHELL-4_(?P<date>\d+)_[A-Z]_(?P<session>R\d+S\d+C\d+)_mixture_"
    r"(?P<start>[\d.]+)_(?P<end>[\d.]+)$"
)
META_ENROLMENT = re.compile(
    r"AISHELL-4_(?P<date>\d+)_[A-Z]_(?P<session>R\d+S\d+C\d+)_(?P<speaker>\d+-[MFU])_"
    r"(?P<start>[\d.]+)_(?P<end>[\d.]+)$"
)


def read_meta(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def enrolment_key(name: str) -> tuple[str, str, str, str] | None:
    match = ENROLMENT.search(name) or META_ENROLMENT.search(name)
    if not match:
        return None
    info = match.groupdict()
    return (info["session"], info["speaker"], info["start"], info["end"])


def mixture_key(name: str) -> tuple[str, str, str] | None:
    match = META_MIXTURE.search(name)
    if not match:
        return None
    info = match.groupdict()
    return (info["session"], info["start"], info["end"])


def build_index(
    repo_files: list[str],
) -> tuple[
    dict[tuple[str, str, str, str], str],
    dict[tuple[str, str, str], list[dict[str, str]]],
]:
    """Return ``(enrolment_key -> repo path, mixture_key -> metadata rows)``."""
    enrolment_index: dict[tuple[str, str, str, str], str] = {}
    for path in repo_files:
        key = enrolment_key(path)
        if key:
            enrolment_index[key] = path

    meta_rows = read_meta(Path(__file__).parent / "data" / "ps4" / "AISHELL-4_meta.csv")
    mixtures: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in meta_rows:
        key = mixture_key(row.get("mixture_utterance", ""))
        if key:
            mixtures[key].append(row)
    return enrolment_index, mixtures


def mixture_composition(
    mixture_key_value: tuple[str, str, str],
    meta_rows: list[dict[str, str]],
    enrolment_index: dict[tuple[str, str, str, str], str],
) -> list[dict[str, str]]:
    """Deduplicate the per-channel metadata rows into constituent utterances."""
    seen: set[tuple[str, str, str]] = set()
    constituents: list[dict[str, str]] = []
    for row in meta_rows:
        enrolment = row.get("enrolment_speakers_utterance", "")
        key = enrolment_key(enrolment)
        if key is None:
            continue
        signature = (key[1], key[2], key[3])
        if signature in seen:
            continue
        seen.add(signature)
        constituents.append(
            {
                "speaker": row.get("speaker", ""),
                "transcript": row.get("ground_truth_transcript", ""),
                "session": key[0],
                "start": key[2],
                "end": key[3],
                "enrolment_key": "|".join(key),
                "repo_path": enrolment_index.get(key, ""),
                "speaker_ratio": row.get("speaker_ratio", ""),
                "overlap_duration": row.get("mixture_overlap_duration", ""),
            }
        )
    return constituents
