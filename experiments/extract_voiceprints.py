"""Extract voiceprints and transcripts once, then cache them.

Running the FunASR pipeline over the whole sample takes a few minutes, and the
analysis that follows wants to iterate quickly on thresholds and scoring rules.
So the ASR work happens here, once, and lands in ``voiceprints.json``; every
analysis afterwards reads that cache and runs in seconds.

Cache layout::

    {"<absolute path>": {"centroids": {"0": [...], ...}, "turns": [...]}}

``centroids`` maps a diarization cluster id to its 192-dimensional CAM++ voice
print, ``turns`` is the timed transcript with the cluster each turn belongs to.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

CACHE_NAME = "voiceprints.json"

#: Repository root — manifest paths are stored relative to it.
ROOT = Path(__file__).resolve().parents[1]


def resolve(path: str) -> str:
    """Absolute path for a manifest entry, wherever the script is run from."""
    candidate = Path(path)
    if candidate.is_absolute():
        return str(candidate)
    return str(ROOT / candidate)


def collect_targets(manifest: list[dict[str, object]]) -> list[str]:
    """Every audio file in the manifest, including nested reference clips."""
    paths: list[str] = []
    for row in manifest:
        if row.get("kind") == "enrolment":
            paths.append(resolve(str(row["path"])))
        elif row.get("kind") == "mixture":
            paths.append(resolve(str(row["path"])))
            for clip in row.get("reference_clips", []) or []:  # type: ignore[union-attr]
                paths.append(resolve(str(clip["path"])))
    seen: set[str] = set()
    unique: list[str] = []
    for path in paths:
        if path not in seen:
            seen.add(path)
            unique.append(path)
    return unique


class Extractor:
    """Runs the FunASR pipeline and returns one voiceprint per speaker."""

    def __init__(self, device: str = "cpu", hub: str = "hf") -> None:
        from funasr import AutoModel

        self.model = AutoModel(
            model="paraformer-zh",
            vad_model="fsmn-vad",
            punc_model="ct-punc",
            spk_model="cam++",
            device=device,
            hub=hub,
            disable_pbar=True,
            disable_update=True,
        )

    def run(self, audio_path: str) -> dict[str, object]:
        result = self.model.generate(input=audio_path, return_spk_center=True)
        payload = result[0] if isinstance(result, list) and result else {}
        centers = payload.get("spk_embedding_center")
        sentences = payload.get("sentence_info") or []

        centroids: dict[str, list[float]] = {}
        if centers is not None:
            for index, vector in enumerate(centers):
                centroids[str(index)] = [float(value) for value in vector]

        turns: list[dict[str, object]] = []
        for index, sentence in enumerate(sentences):
            if not isinstance(sentence, dict):
                continue
            text = str(sentence.get("text", "")).strip()
            if not text:
                continue
            speaker = int(sentence.get("spk", index))
            turns.append(
                {
                    "start": round(float(sentence.get("start", 0.0)) / 1000.0, 3),
                    "end": round(float(sentence.get("end", 0.0)) / 1000.0, 3),
                    "text": text,
                    "cluster": speaker,
                }
            )
        return {"centroids": centroids, "turns": turns}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(ROOT / "experiments" / "data" / "ps4"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--force", action="store_true", help="Re-extract cached files.")
    args = parser.parse_args(argv)

    data = Path(args.data)
    manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    cache_path = data / CACHE_NAME

    cache: dict[str, dict[str, object]] = {}
    if cache_path.exists() and not args.force:
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
        print(f"loaded cache with {len(cache)} entries")

    targets = collect_targets(manifest)
    pending = [path for path in targets if path not in cache]
    print(f"{len(targets)} files, {len(pending)} to extract")

    if pending:
        extractor = Extractor(device=args.device)
        started = time.time()
        for index, path in enumerate(pending, start=1):
            cache[path] = extractor.run(path)
            if index % 10 == 0 or index == len(pending):
                rate = (time.time() - started) / index
                print(
                    f"  {index}/{len(pending)} ({rate:.2f}s per file)",
                    flush=True,
                )
        cache_path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    else:
        cache_path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")

    print(f"cache written to {cache_path} ({len(cache)} entries)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
