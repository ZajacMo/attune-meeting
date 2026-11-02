"""Pre-flight check: download the FunASR weights and run the pipeline once.

Useful before a long experiment, so a download problem or a missing model does
not show up half an hour into a run.

    python experiments/warm_models.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from attune.core.audio import write_wav  # noqa: E402

SCRATCH = Path(__file__).resolve().parent / "dl"


def main() -> int:
    SCRATCH.mkdir(parents=True, exist_ok=True)
    wav = write_wav(SCRATCH / "silence.wav", [0.0] * 16000, 16000)

    from funasr import AutoModel

    print("building model (this downloads weights)...", flush=True)
    start = time.time()
    model = AutoModel(
        model="paraformer-zh",
        vad_model="fsmn-vad",
        punc_model="ct-punc",
        spk_model="cam++",
        device="cpu",
        hub="hf",
        disable_pbar=True,
        disable_update=True,
    )
    print(f"model ready in {time.time() - start:.1f}s", flush=True)

    result = model.generate(input=wav, return_spk_center=True)
    payload = result[0]
    print("result keys:", sorted(payload.keys()))
    print("spk_embedding_center present:", "spk_embedding_center" in payload)
    print("text:", payload.get("text"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
