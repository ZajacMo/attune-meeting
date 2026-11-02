"""How much overlapping speech can the diarizer actually take?

AISHELL-4 mixtures are built by summing *overlapping* enrolment utterances, so
they are the hard case for speaker diarization. This samples windows across the
overlap spectrum and measures how many speaker clusters the engine finds
against how many speakers the metadata documents.

The result decides what Attune can promise: if the engine cannot separate
people at all, there is nothing to recognise across meetings.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from extract_voiceprints import Extractor  # noqa: E402
from ground_truth import META_MIXTURE, MIXTURE, read_meta  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REPO = "tian0105/REAL-PS4"

OVERLAP_BUCKETS = [0.2, 0.4, 0.6, 0.8, 1.0]
SPEAKER_COUNTS = [2, 3, 4]


def load_windows(meta_path: Path) -> dict[tuple[str, str, str], dict[str, object]]:
    """Mixture windows with their speakers, duration and overlap fraction."""
    rows = read_meta(meta_path)
    windows: dict[tuple[str, str, str], dict[str, object]] = {}
    for row in rows:
        name = row.get("mixture_utterance", "")
        match = META_MIXTURE.search(name)
        if not match:
            continue
        info = match.groupdict()
        key = (info["session"], info["start"], info["end"])
        entry = windows.setdefault(
            key,
            {
                "speakers": set(),
                "duration": float(row.get("mixture_duration", 0) or 0),
                "overlap": float(row.get("mixture_overlap_duration", 0) or 0),
            },
        )
        entry["speakers"].add(row.get("speaker", ""))  # type: ignore[union-attr]
    return windows


def bucket_of(fraction: float) -> float:
    for bucket in OVERLAP_BUCKETS:
        if fraction <= bucket + 0.05:
            return bucket
    return OVERLAP_BUCKETS[-1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(ROOT / "experiments" / "data" / "ps4"))
    parser.add_argument("--per-bucket", type=int, default=3)
    parser.add_argument("--out", default=str(ROOT / "experiments"))
    args = parser.parse_args(argv)

    data_dir = Path(args.data)
    (data_dir / "overlap").mkdir(parents=True, exist_ok=True)

    windows = load_windows(data_dir / "AISHELL-4_meta.csv")
    print(f"mixture windows in metadata: {len(windows)}")

    # Repo paths for the mixture audio, keyed the same way.
    from huggingface_hub import list_repo_files

    repo_mixtures: dict[tuple[str, str, str], str] = {}
    for path in list_repo_files(REPO, repo_type="dataset"):
        match = MIXTURE.search(path)
        if match:
            info = match.groupdict()
            repo_mixtures[(info["session"], info["start"], info["end"])] = path
    print(f"mixture audio files: {len(repo_mixtures)}")

    # Pick samples per (speaker count, overlap bucket), preferring long windows.
    picked: dict[tuple[int, float], list[tuple[tuple[str, str, str], dict[str, object]]]] = (
        defaultdict(list)
    )
    for key, entry in windows.items():
        speakers = len(entry["speakers"])  # type: ignore[arg-type]
        if speakers not in SPEAKER_COUNTS:
            continue
        duration = float(entry["duration"])  # type: ignore[arg-type]
        overlap = float(entry["overlap"])  # type: ignore[arg-type]
        if duration <= 0:
            continue
        bucket = bucket_of(overlap / duration)
        picked[(speakers, bucket)].append((key, entry))

    samples: list[tuple[tuple[int, float], tuple[str, str, str], dict[str, object]]] = []
    for group in sorted(picked):
        group_samples = sorted(
            picked[group],
            key=lambda item: -float(item[1]["duration"]),  # type: ignore[index]
        )
        for key, entry in group_samples[: args.per_bucket]:
            samples.append((group, key, entry))
    print(f"windows to test: {len(samples)}")

    extractor = Extractor()
    results: list[dict[str, object]] = []
    for (_speaker_count, bucket), key, entry in samples:
        repo_path = repo_mixtures.get(key)
        if not repo_path:
            continue
        target = data_dir / "overlap" / Path(repo_path).name
        if not target.exists():
            downloaded = hf_hub_download(
                REPO, repo_path, repo_type="dataset", local_dir=str(data_dir / "raw")
            )
            target.write_bytes(Path(downloaded).read_bytes())
        outcome = extractor.run(str(target))
        detected = len(outcome["centroids"])
        documented = len(entry["speakers"])  # type: ignore[arg-type]
        duration = float(entry["duration"])  # type: ignore[arg-type]
        overlap = float(entry["overlap"])  # type: ignore[arg-type]
        record = {
            "session": key[0],
            "window": f"{key[1]}-{key[2]}",
            "documented_speakers": documented,
            "detected_clusters": detected,
            "duration_s": round(duration, 1),
            "overlap_fraction": round(overlap / duration, 2) if duration else 0.0,
            "overlap_bucket": bucket,
        }
        results.append(record)
        print(
            f"  {record['session']} {record['duration_s']:5.1f}s "
            f"overlap={record['overlap_fraction']:.2f} "
            f"speakers={documented} clusters={detected}",
            flush=True,
        )

    # Aggregate.
    aggregate: dict[str, dict[str, object]] = {}
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in results:
        grouped[
            f"{record['documented_speakers']} speakers / overlap ≤ {record['overlap_bucket']}"
        ].append(record)
    for group, rows in grouped.items():
        aggregate[group] = {
            "n": len(rows),
            "documented_speakers": rows[0]["documented_speakers"],
            "mean_detected_clusters": round(
                sum(int(row["detected_clusters"]) for row in rows) / len(rows), 2
            ),
            "fully_separated": sum(
                1
                for row in rows
                if int(row["detected_clusters"]) == int(row["documented_speakers"])
            ),
        }

    report = {
        "source": "REAL-PS4 / AISHELL-4 mixtures (real recorded Mandarin speech, "
        "synthesised overlap)",
        "engine": "FunASR paraformer-zh + fsmn-vad + ct-punc + cam++ (CPU)",
        "samples": len(results),
        "fully_separated_overall": round(
            sum(
                1
                for row in results
                if int(row["detected_clusters"]) == int(row["documented_speakers"])
            )
            / max(1, len(results)),
            4,
        ),
        "aggregate": aggregate,
        "details": results,
    }
    out_dir = Path(args.out)
    (out_dir / "diarization_overlap.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = [
        "# Diarization vs overlapping speech",
        "",
        f"Engine: {report['engine']}",
        f"Windows tested: {report['samples']} · fully separated: "
        f"{report['fully_separated_overall']}",
        "",
        "| speakers | overlap ≤ | windows | mean clusters | fully separated |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for group, stats in aggregate.items():
        speakers = stats["documented_speakers"]
        bucket = group.split("overlap ≤ ")[1]
        lines.append(
            f"| {speakers} | {bucket} | {stats['n']} | "
            f"{stats['mean_detected_clusters']} | {stats['fully_separated']}/{stats['n']} |"
        )
    lines.append("")
    (out_dir / "diarization_overlap.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n" + "\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
