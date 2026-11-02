"""How many rows the cluster backend actually sees.

FunASR cuts every VAD segment into 1.5-second windows with a 0.75-second hop
(``funasr/models/campplus/utils.py: sv_chunk``) and clusters those windows.
The cluster backend refuses to split anything below 20 rows
(``funasr/models/campplus/cluster_backend.py``).

That rule is easy to state and easy to misapply: the rows are *chunks*, not
ASR sentences and not VAD segments. This measures the real number for every
clip in the sample and checks the rule against the clusters the engine
returned.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from extract_voiceprints import resolve  # noqa: E402


def relative(path: str) -> str:
    """Path relative to the repository root, so the report stays portable."""
    try:
        return str(Path(path).resolve().relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return Path(path).name


def measure(paths: list[str], device: str = "cpu") -> list[dict[str, object]]:
    """Run the pipeline once per file and record the cluster input size."""
    import funasr.auto.auto_model as auto_model

    original_sv_chunk = auto_model.sv_chunk
    counter = {"chunks": 0, "calls": 0}

    def counting_sv_chunk(vad_segments: list, fs: int = 16000) -> list:
        chunks = original_sv_chunk(vad_segments, fs=fs)
        counter["chunks"] += len(chunks)
        counter["calls"] += 1
        return chunks

    # ``auto_model`` binds the name at import time, so the pipeline has to be
    # patched where it is used, not where it was defined.
    auto_model.sv_chunk = counting_sv_chunk

    from funasr import AutoModel

    model = AutoModel(
        model="paraformer-zh",
        vad_model="fsmn-vad",
        punc_model="ct-punc",
        spk_model="cam++",
        device=device,
        hub="hf",
        disable_pbar=True,
        disable_update=True,
    )

    rows: list[dict[str, object]] = []
    for index, path in enumerate(paths, start=1):
        counter["chunks"] = 0
        counter["calls"] = 0
        result = model.generate(input=path, return_spk_center=True)
        payload = result[0] if isinstance(result, list) and result else {}
        centers = payload.get("spk_embedding_center")
        sentences = payload.get("sentence_info") or []
        cluster_count = 0 if centers is None else len(centers)
        chunks = int(counter["chunks"])
        rows.append(
            {
                "path": relative(path),
                "chunks": chunks,
                "clusters": cluster_count,
                "sentences": len(sentences),
                "forced_single_cluster": chunks < 20,
            }
        )
        print(
            f"  {index}/{len(paths)} {Path(path).name[-40:]}: "
            f"chunks={chunks} clusters={cluster_count} sentences={len(sentences)}",
            flush=True,
        )
    auto_model.sv_chunk = original_sv_chunk
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(ROOT / "experiments" / "data" / "ps4"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default=str(ROOT / "experiments"))
    args = parser.parse_args(argv)

    data = Path(args.data)
    out_dir = Path(args.out)
    manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    mixtures = [
        resolve(str(row["path"]))
        for row in manifest
        if row.get("kind") == "mixture" and Path(resolve(str(row["path"]))).exists()
    ]
    # Only the meetings the diarization experiment actually used.
    meeting_report = out_dir / "meeting_diarization.json"
    meeting_names: list[str] = []
    if meeting_report.exists():
        previous = json.loads(meeting_report.read_text(encoding="utf-8"))
        meeting_names = [str(row["meeting"]) for row in previous["results"]]
    available = {path.name: path for path in (data / "meetings").glob("meeting_*.wav")}
    meetings = [str(available[name]) for name in meeting_names if name in available]
    if not meetings:
        meetings = [str(path) for path in sorted(available.values())]

    report: dict[str, object] = {
        "source": "REAL-PS4 / AISHELL-4 — the cluster input measured, not inferred",
        "note": "chunks are the rows the cluster backend clusters: 1.5-second "
        "windows over the VAD segments with a 0.75-second hop. The backend "
        "returns a single cluster below 20 rows.",
        "mixtures": measure(mixtures, device=args.device),
        "meetings": measure(meetings, device=args.device),
    }

    out_path = out_dir / "chunk_budget.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Cluster input size",
        "",
        "| clip | chunks | clusters | sentences | forced single cluster |",
        "| --- | ---: | ---: | ---: | :---: |",
    ]
    for group in ("mixtures", "meetings"):
        for row in report[group]:  # type: ignore[union-attr]
            lines.append(
                f"| {Path(str(row['path'])).name[-38:]} | {row['chunks']} | {row['clusters']} | "
                f"{row['sentences']} | {'yes' if row['forced_single_cluster'] else 'no'} |"
            )
    (out_dir / "chunk_budget.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
