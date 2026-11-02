"""Print the calibration report in a readable form."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "experiments/report.json")
    report = json.loads(path.read_text(encoding="utf-8"))

    for key in (
        "source",
        "clips",
        "speakers",
        "sessions",
        "same_speaker_cross_session",
        "same_speaker_same_session",
        "different_speakers",
        "best_separating_threshold",
        "equal_error_rate",
        "shipped_defaults",
        "defaults_on_this_data",
    ):
        print(f"{key} = {json.dumps(report[key], ensure_ascii=False)}")

    print("\n--- threshold sweep ---")
    for row in report["threshold_sweep"]:
        print(
            f"{row['threshold']:.2f}  recall={row['recall_same_speaker']:.4f}  "
            f"far={row['false_accept_other_speaker']:.4f}"
        )

    cross = report["cross_meeting_recognition"]
    print("\n--- cross meeting ---")
    for key in (
        "mixtures",
        "cluster_agreement",
        "same_speaker_score",
        "runner_up_score",
        "wrong_cluster_score",
        "cross_meeting_threshold",
    ):
        print(f"{key} = {json.dumps(cross[key], ensure_ascii=False)}")

    print("\n| mixture | speaker | det | exp | ref | enrolled | score | runner | same |")
    for case in cross["cases"]:
        print(
            f"| {case['mixture'][:36]} | {case['speaker']} | {case['detected_speakers']} | "
            f"{case['expected_speakers']} | {case['reference_cluster']} | "
            f"{case['enrolled_best_cluster']} | {case['enrolled_score']:.3f} | "
            f"{case['enrolled_runner_up']:.3f} | {'YES' if case['same_cluster'] else 'NO'} |"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
