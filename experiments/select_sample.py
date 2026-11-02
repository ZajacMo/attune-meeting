"""Select the experiment sample from REAL-PS4 / AISHELL-4.

File names carry most of what we need:

    AISHELL-4_<date>_<mic>_<session>_<speaker>-<gender>_<start>_<end>.wav

``<session>`` is the meeting recording and ``<speaker>`` the person, so the
same person appearing in two different sessions is a genuine cross-session
voiceprint pair.

One wrinkle: the metadata table names mixtures with a *different* date/mic
prefix than the audio files actually use, so mixtures are joined on
``(session, start, end)`` rather than on the full file name.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

from ground_truth import MIXTURE, mixture_composition
from huggingface_hub import hf_hub_download, list_repo_files

REPO = "tian0105/REAL-PS4"

#: Repository root — every path stored in the manifest is relative to it, so the
#: scripts work no matter which directory they are started from.
ROOT = Path(__file__).resolve().parents[1]

ENROLMENT = re.compile(
    r"AISHELL-4_(?P<date>\d+)_[A-Z]_(?P<session>R\d+S\d+C\d+)_(?P<speaker>\d+-[MFU])_"
    r"(?P<start>[\d.]+)_(?P<end>[\d.]+)\.wav$"
)
META_MIXTURE = re.compile(
    r"AISHELL-4_(?P<date>\d+)_[A-Z]_(?P<session>R\d+S\d+C\d+)_mixture_"
    r"(?P<start>[\d.]+)_(?P<end>[\d.]+)$"
)


def parse_enrolment(path: str) -> dict[str, str] | None:
    match = ENROLMENT.search(path)
    return match.groupdict() if match else None


def parse_mixture(path: str) -> dict[str, str] | None:
    match = MIXTURE.search(path)
    return match.groupdict() if match else None


def fetch(repo_file: str, out: Path) -> Path:
    target = out / Path(repo_file).name
    if not target.exists():
        downloaded = hf_hub_download(
            REPO, repo_file, repo_type="dataset", local_dir=str(out / "raw")
        )
        target.write_bytes(Path(downloaded).read_bytes())
    return target


def relative(path: Path) -> str:
    """Store paths relative to the repository root."""
    try:
        return str(path.resolve().relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(ROOT / "experiments" / "data" / "ps4"))
    parser.add_argument("--speakers", type=int, default=8)
    parser.add_argument("--sessions-per-speaker", type=int, default=3)
    parser.add_argument("--clips-per-speaker-session", type=int, default=3)
    parser.add_argument("--mixtures", type=int, default=8)
    args = parser.parse_args(argv)

    out = Path(args.out)
    (out / "enrolment").mkdir(parents=True, exist_ok=True)
    (out / "mixtures").mkdir(parents=True, exist_ok=True)

    files = list_repo_files(REPO, repo_type="dataset")

    # ---- metadata -------------------------------------------------
    meta_path = fetch("AISHELL-4/TRAIN/AISHELL-4_meta.csv", out)
    meta_rows = list(csv.DictReader(meta_path.open(encoding="utf-8")))
    meta_by_key: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in meta_rows:
        match = META_MIXTURE.search(row.get("mixture_utterance", ""))
        if match:
            info = match.groupdict()
            meta_by_key[(info["session"], info["start"], info["end"])].append(row)
    print(f"metadata rows: {len(meta_rows)}, mixture keys: {len(meta_by_key)}")

    # enrolment audio, keyed by (session, speaker, start, end)
    enrolment_index: dict[tuple[str, str, str, str], str] = {}
    for path in files:
        info = parse_enrolment(path)
        if info:
            enrolment_index[(info["session"], info["speaker"], info["start"], info["end"])] = path
    print(f"enrolment files indexed: {len(enrolment_index)}")

    # ---- mixtures first: they decide which speakers we need --------
    mixture_candidates: list[tuple[str, dict[str, str], list[dict[str, str]]]] = []
    for path in files:
        info = parse_mixture(path)
        if not info:
            continue
        key = (info["session"], info["start"], info["end"])
        entries = meta_by_key.get(key)
        if entries:
            mixture_candidates.append((path, info, entries))
    print(f"mixtures with metadata: {len(mixture_candidates)}")

    (out / "constituents").mkdir(parents=True, exist_ok=True)
    chosen_mixtures: list[dict[str, object]] = []
    seen_sessions: set[str] = set()

    # One mixture per meeting session. Some windows in the metadata only
    # document part of the speakers that are audible, so a window is only
    # usable when every documented speaker is present.
    by_session: dict[str, list[tuple[str, dict[str, str], list[dict[str, str]]]]] = defaultdict(
        list
    )
    for candidate in mixture_candidates:
        by_session[candidate[1]["session"]].append(candidate)

    for session in sorted(by_session):
        if len(chosen_mixtures) >= args.mixtures:
            break
        picked: tuple[str, dict[str, str], list[dict[str, str]], list[dict[str, str]]] | None = None
        for repo_path, info, entries in by_session[session]:
            key = (info["session"], info["start"], info["end"])
            constituents = mixture_composition(key, entries, enrolment_index)
            documented = sorted({entry["speaker"] for entry in constituents})
            expected = entries[0].get("total_number_of_speaker", "")
            try:
                expected_count = int(expected)
            except (TypeError, ValueError):
                expected_count = len(documented)
            if not documented or len(documented) < expected_count:
                continue
            picked = (repo_path, info, entries, constituents)
            break
        if picked is None:
            print(f"  {session}: no window with complete speaker metadata, skipped")
            continue

        repo_path, info, entries, constituents = picked
        seen_sessions.add(info["session"])
        local = fetch(repo_path, out / "mixtures")
        speakers = sorted({entry["speaker"] for entry in constituents})
        # Keep two clean enrolment clips per speaker: they are the
        # same-session reference the mixture has to be recognised against.
        reference_clips: list[dict[str, str]] = []
        per_speaker: dict[str, int] = defaultdict(int)
        for constituent in constituents:
            speaker = constituent["speaker"]
            if per_speaker[speaker] >= 2 or not constituent["repo_path"]:
                continue
            per_speaker[speaker] += 1
            local_clip = fetch(constituent["repo_path"], out / "constituents")
            reference_clips.append(
                {
                    "speaker": speaker,
                    "session": constituent["session"],
                    "transcript": constituent["transcript"],
                    "path": relative(local_clip),
                    "source": constituent["repo_path"],
                    "start": constituent["start"],
                    "end": constituent["end"],
                }
            )
        chosen_mixtures.append(
            {
                "kind": "mixture",
                "session": info["session"],
                "path": relative(local),
                "source": repo_path,
                "window": {"start": info["start"], "end": info["end"]},
                "target_speakers": speakers,
                "expected_speakers": entries[0].get("total_number_of_speaker"),
                "constituents": constituents,
                "reference_clips": reference_clips,
                "ground_truth": [
                    {
                        "speaker": speaker,
                        "transcript": transcript,
                        "speaker_ratio": ratio,
                    }
                    for speaker, transcript, ratio in sorted(
                        {
                            (
                                entry["speaker"],
                                entry["transcript"],
                                entry["speaker_ratio"],
                            )
                            for entry in constituents
                        }
                    )
                ],
            }
        )
        if len(chosen_mixtures) >= args.mixtures:
            break
    print(f"mixtures chosen: {len(chosen_mixtures)} from sessions {sorted(seen_sessions)}")

    # ---- enrolment clips ------------------------------------------
    by_speaker: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for path in files:
        info = parse_enrolment(path)
        if info:
            by_speaker[info["speaker"]][info["session"]].append(path)

    needed: set[str] = set()
    for row in chosen_mixtures:
        needed.update(row["target_speakers"])  # type: ignore[union-attr]

    manifest: list[dict[str, object]] = []
    for speaker in sorted(needed):
        sessions = by_speaker.get(speaker, {})
        for session in sorted(sessions)[: args.sessions_per_speaker]:
            for clip in sorted(sessions[session])[: args.clips_per_speaker_session]:
                local = fetch(clip, out / "enrolment")
                manifest.append(
                    {
                        "kind": "enrolment",
                        "speaker": speaker,
                        "session": session,
                        "path": relative(local),
                        "source": clip,
                    }
                )

    # Plus the speakers with the widest session coverage, for the stability study.
    ranked = sorted(by_speaker.items(), key=lambda item: -len(item[1]))
    extra = 0
    for speaker, sessions in ranked:
        if speaker in needed or extra >= args.speakers:
            continue
        for session in sorted(sessions)[: args.sessions_per_speaker]:
            for clip in sorted(sessions[session])[: args.clips_per_speaker_session]:
                local = fetch(clip, out / "enrolment")
                manifest.append(
                    {
                        "kind": "enrolment",
                        "speaker": speaker,
                        "session": session,
                        "path": relative(local),
                        "source": clip,
                    }
                )
        extra += 1

    manifest.extend(chosen_mixtures)
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    enrolment_rows = [row for row in manifest if row["kind"] == "enrolment"]
    print(f"enrolment clips: {len(enrolment_rows)}")
    print(f"speakers: {sorted({row['speaker'] for row in enrolment_rows})}")
    print(f"sessions: {sorted({row['session'] for row in enrolment_rows})}")
    print(f"mixtures: {len(chosen_mixtures)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
