"""Does the Attune identity model actually work on real Chinese speech?

Everything runs offline on top of the cached voiceprints produced by
``extract_voiceprints.py``, so thresholds and scoring rules can be iterated in
seconds instead of minutes.

Three experiments on AISHELL-4 (real recorded Mandarin meetings, 8-channel
array, the same people recurring across sessions):

1. **Voiceprint stability** — cosine similarity between voiceprints of the same
   person taken from different sessions, versus different people, and which
   threshold actually separates them.
2. **Scoring rule** — Attune keeps *several* voiceprints per identity. Scoring
   against the best one of them gets more optimistic the more times someone is
   enrolled; scoring against the centroid does not. This measures both rules as
   the number of enrolments grows, because it decides what the threshold in
   ``core/identity.py`` has to mean.
3. **Cross-meeting recognition** — enrol a person in one meeting, then find them
   inside a *different* real meeting that contains overlapping speech.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from attune.core.identity import cosine_similarity  # noqa: E402

#: Repository root — manifest paths are stored relative to it.
ROOT = Path(__file__).resolve().parents[1]

#: Everything that is not a letter or a digit — Chinese punctuation, ASCII
#: punctuation and whitespace all carry no identity signal.
PUNCTUATION = re.compile(r"[\W_]+", re.UNICODE)


# ----------------------------------------------------------------------
# Statistics helpers
# ----------------------------------------------------------------------
def describe(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "min": round(ordered[0], 4),
        "p05": round(ordered[max(0, int(len(ordered) * 0.05) - 1)], 4),
        "p25": round(ordered[max(0, int(len(ordered) * 0.25) - 1)], 4),
        "median": round(statistics.median(ordered), 4),
        "p75": round(ordered[min(len(ordered) - 1, int(len(ordered) * 0.75))], 4),
        "p95": round(ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))], 4),
        "max": round(ordered[-1], 4),
        "mean": round(statistics.fmean(ordered), 4),
    }


def best_threshold(positives: list[float], negatives: list[float]) -> dict[str, float]:
    """The threshold that best separates the two distributions."""
    candidates = sorted({round(value, 4) for value in positives + negatives})
    best: dict[str, float] = {"threshold": 0.0, "accuracy": 0.0, "errors": 0, "total": 0}
    for threshold in candidates:
        errors = sum(1 for value in positives if value < threshold) + sum(
            1 for value in negatives if value >= threshold
        )
        accuracy = 1 - errors / max(1, len(positives) + len(negatives))
        if accuracy > best["accuracy"]:
            best = {
                "threshold": threshold,
                "accuracy": round(accuracy, 4),
                "errors": errors,
                "total": len(positives) + len(negatives),
            }
    return best


def equal_error_rate(positives: list[float], negatives: list[float]) -> dict[str, float]:
    """Threshold where false-accept rate equals false-reject rate."""
    candidates = sorted({round(value, 4) for value in positives + negatives})
    best: dict[str, float] = {"threshold": 0.0, "eer": 1.0}
    for threshold in candidates:
        far = sum(1 for value in negatives if value >= threshold) / max(1, len(negatives))
        frr = sum(1 for value in positives if value < threshold) / max(1, len(positives))
        eer = (far + frr) / 2
        if eer < best["eer"]:
            best = {"threshold": threshold, "eer": round(eer, 4)}
    return best


def threshold_sweep(
    positives: list[float], negatives: list[float], grid: list[float]
) -> list[dict[str, float]]:
    """Recall and false-accept rate at each candidate threshold."""
    rows: list[dict[str, float]] = []
    for threshold in grid:
        recall = sum(1 for value in positives if value >= threshold) / max(1, len(positives))
        false_accept = sum(1 for value in negatives if value >= threshold) / max(1, len(negatives))
        rows.append(
            {
                "threshold": threshold,
                "recall_same_speaker": round(recall, 4),
                "false_accept_other_speaker": round(false_accept, 4),
            }
        )
    return rows


def centroid(vectors: list[list[float]]) -> list[float]:
    """Mean voiceprint vector."""
    if not vectors:
        return []
    width = min(len(vector) for vector in vectors)
    accumulator = [0.0] * width
    for vector in vectors:
        for index, value in enumerate(vector[:width]):
            accumulator[index] += value
    count = float(len(vectors))
    return [value / count for value in accumulator]


def score_centroid(profile: list[list[float]], query: list[float]) -> float:
    """Attune's proposed rule: compare against the average voiceprint."""
    anchor = centroid(profile)
    return cosine_similarity(query, anchor) if anchor else 0.0


def score_max(profile: list[list[float]], query: list[float]) -> float:
    """The naive rule: take the best of the stored voiceprints."""
    if not profile:
        return 0.0
    return max(cosine_similarity(query, vector) for vector in profile)


def normalize(text: str) -> str:
    return PUNCTUATION.sub("", text)


def text_similarity(left: str, right: str) -> float:
    return SequenceMatcher(None, normalize(left), normalize(right)).ratio()


# ----------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------
class Dataset:
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
        self.cache = json.loads((data_dir / "voiceprints.json").read_text(encoding="utf-8"))
        self.enrolment = [row for row in self.manifest if row["kind"] == "enrolment"]
        self.mixtures = [row for row in self.manifest if row["kind"] == "mixture"]

    def resolve(self, path: str) -> str:
        """Absolute path for a manifest entry, wherever the script is run from."""
        candidate = Path(path)
        if candidate.is_absolute():
            return str(candidate)
        return str(ROOT / candidate)

    def centroids(self, path: str) -> dict[str, list[float]]:
        entry = self.cache.get(self.resolve(path), {})
        return {key: list(value) for key, value in entry.get("centroids", {}).items()}

    def turns(self, path: str) -> list[dict[str, object]]:
        entry = self.cache.get(self.resolve(path), {})
        return list(entry.get("turns", []))


# ----------------------------------------------------------------------
# Experiment 1 — voiceprint stability
# ----------------------------------------------------------------------
def experiment_one(data: Dataset) -> dict[str, object]:
    print("[1/3] voiceprint stability", flush=True)
    vectors: list[dict[str, object]] = []
    for row in data.enrolment:
        centroids = data.centroids(str(row["path"]))
        if not centroids:
            continue
        vector = next(iter(centroids.values()))
        vectors.append(
            {
                "speaker": row["speaker"],
                "session": row["session"],
                "vector": vector,
            }
        )

    positives: list[float] = []
    negatives: list[float] = []
    same_session: list[float] = []
    impostor_pairs: list[tuple[float, str, str, str, str]] = []
    positive_pairs: list[tuple[float, str, str, str, str]] = []
    for i, left in enumerate(vectors):
        for right in vectors[i + 1 :]:
            score = cosine_similarity(left["vector"], right["vector"])
            if left["speaker"] == right["speaker"]:
                entry = (
                    score,
                    str(left["speaker"]),
                    str(left["session"]),
                    str(right["session"]),
                    "",
                )
                positive_pairs.append(entry)
                if left["session"] == right["session"]:
                    same_session.append(score)
                else:
                    positives.append(score)
            else:
                negatives.append(score)
                impostor_pairs.append(
                    (
                        score,
                        str(left["speaker"]),
                        str(left["session"]),
                        str(right["speaker"]),
                        str(right["session"]),
                    )
                )

    grid = [
        0.45,
        0.50,
        0.53,
        0.55,
        0.58,
        0.60,
        0.62,
        0.65,
        0.67,
        0.68,
        0.70,
        0.72,
        0.75,
        0.78,
        0.80,
        0.82,
        0.85,
    ]
    return {
        "clips": len(vectors),
        "speakers": sorted({row["speaker"] for row in vectors}),
        "sessions": sorted({row["session"] for row in vectors}),
        "same_speaker_cross_session": describe(positives),
        "same_speaker_same_session": describe(same_session),
        "different_speakers": describe(negatives),
        "best_separating_threshold": best_threshold(positives, negatives),
        "equal_error_rate": equal_error_rate(positives, negatives),
        "threshold_sweep": threshold_sweep(positives, negatives, grid),
        "hardest_impostors": [
            {
                "score": round(score, 4),
                "speaker_a": speaker_a,
                "session_a": session_a,
                "speaker_b": speaker_b,
                "session_b": session_b,
            }
            for score, speaker_a, session_a, speaker_b, session_b in sorted(
                impostor_pairs, reverse=True
            )[:15]
        ],
        "hardest_genuine": [
            {
                "score": round(score, 4),
                "speaker": speaker_a,
                "session_a": session_a,
                "session_b": session_b,
            }
            for score, speaker_a, session_a, session_b, _ in sorted(positive_pairs)[:15]
        ],
    }


# ----------------------------------------------------------------------
# Experiment 2 — scoring rule vs number of enrolments
# ----------------------------------------------------------------------
def experiment_two(data: Dataset) -> dict[str, object]:
    print("[2/3] scoring rule and enrolment count", flush=True)

    by_speaker_session: dict[tuple[str, str], list[list[float]]] = defaultdict(list)
    for row in data.enrolment:
        centroids = data.centroids(str(row["path"]))
        if centroids:
            vector = next(iter(centroids.values()))
            by_speaker_session[(str(row["speaker"]), str(row["session"]))].append(vector)

    sessions = sorted({session for _, session in by_speaker_session})
    rules = {
        "centroid": score_centroid,
        "max": score_max,
    }
    # scores[rule][k] = (same-speaker scores, impostor scores)
    scores: dict[str, dict[int, tuple[list[float], list[float]]]] = {
        rule: defaultdict(lambda: ([], [])) for rule in rules
    }
    counts: dict[int, int] = defaultdict(int)

    for speaker in sorted({key[0] for key in by_speaker_session}):
        for enrol_session in sessions:
            profile_pool = by_speaker_session.get((speaker, enrol_session), [])
            if len(profile_pool) < 2:
                continue
            for test_session in sessions:
                if test_session == enrol_session:
                    continue
                for k in range(1, len(profile_pool) + 1):
                    profile = profile_pool[:k]
                    counts[k] += 1
                    for other_speaker, other_session in by_speaker_session:
                        if other_session != test_session:
                            continue
                        for query in by_speaker_session[(other_speaker, other_session)]:
                            if other_speaker == speaker:
                                for rule, scorer in rules.items():
                                    scores[rule][k][0].append(scorer(profile, query))
                            else:
                                for rule, scorer in rules.items():
                                    scores[rule][k][1].append(scorer(profile, query))

    grid = [0.50, 0.53, 0.55, 0.60, 0.62, 0.65, 0.68, 0.70, 0.75, 0.82]
    summary: dict[str, object] = {"grid": grid, "profiles_per_k": dict(counts), "rules": {}}
    for rule in rules:
        per_k: list[dict[str, object]] = []
        for k in sorted(scores[rule]):
            positives, negatives = scores[rule][k]
            if not positives:
                continue
            per_k.append(
                {
                    "k": k,
                    "n_positive": len(positives),
                    "n_negative": len(negatives),
                    "same_speaker_score": describe(positives),
                    "impostor_score": describe(negatives),
                    "sweep": threshold_sweep(positives, negatives, grid),
                }
            )
        summary["rules"][rule] = per_k  # type: ignore[index]
    return summary


# ----------------------------------------------------------------------
# Experiment 3 — cross-meeting recognition
# ----------------------------------------------------------------------
def label_turns(
    turns: list[dict[str, object]], transcripts: dict[str, str]
) -> list[dict[str, object]]:
    """Give every ASR turn the speaker whose words it contains."""
    labelled: list[dict[str, object]] = []
    for turn in turns:
        text = str(turn.get("text", ""))
        if not text:
            continue
        scores = {
            speaker: text_similarity(text, transcript)
            for speaker, transcript in transcripts.items()
        }
        if not scores:
            continue
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        best_speaker, best_score = ranked[0]
        runner_up = ranked[1][1] if len(ranked) > 1 else 0.0
        if best_score < 0.30 or best_score - runner_up < 0.05:
            continue
        labelled.append(
            {
                "cluster": int(turn.get("cluster", 0)),
                "speaker": best_speaker,
                "score": round(best_score, 4),
            }
        )
    return labelled


def cluster_speakers(labelled: list[dict[str, object]]) -> dict[int, str]:
    """Assign each documented speaker to exactly one diarization cluster.

    Turns are first labelled with the speaker whose words they contain. A
    speaker then belongs to the cluster that carries most of their turns — the
    assignment is one-to-one, so two clusters can never claim the same person
    and a cluster without a documented speaker stays unassigned.
    """
    by_speaker: dict[str, Counter[int]] = defaultdict(Counter)
    for row in labelled:
        by_speaker[str(row["speaker"])][int(row["cluster"])] += 1

    claimed: set[int] = set()
    result: dict[int, str] = {}
    for speaker in sorted(by_speaker):
        for cluster, _votes in by_speaker[speaker].most_common():
            if cluster in claimed:
                continue
            claimed.add(cluster)
            result[cluster] = speaker
            break
    return result


def experiment_three(data: Dataset) -> dict[str, object]:
    print("[3/3] cross-meeting recognition", flush=True)

    # Cross-session enrolment pool: every speaker's voiceprints, by session.
    pool: dict[str, dict[str, list[list[float]]]] = defaultdict(lambda: defaultdict(list))
    for row in data.enrolment:
        centroids = data.centroids(str(row["path"]))
        if centroids:
            pool[str(row["speaker"])][str(row["session"])].append(next(iter(centroids.values())))

    cases: list[dict[str, object]] = []
    for row in data.mixtures:
        path = str(row["path"])
        transcripts = {
            str(entry["speaker"]): str(entry["transcript"])
            for entry in row.get("constituents", [])  # type: ignore[union-attr]
        }
        if not transcripts:
            continue
        turns = data.turns(path)
        centroids = data.centroids(path)
        if len(centroids) < 2:
            print(f"    {Path(path).name[:34]}: only {len(centroids)} cluster(s)", flush=True)
            continue

        labelled = label_turns(turns, transcripts)
        speaker_of_cluster = cluster_speakers(labelled)
        if not speaker_of_cluster:
            continue

        # How good was the diarization itself?
        purity = sum(
            1
            for row_ in labelled
            if str(row_["speaker"]) == speaker_of_cluster.get(int(row_["cluster"]), "")
        ) / max(1, len(labelled))

        # Same-session reference: the clean audio that was mixed in.
        session_reference: dict[str, list[list[float]]] = defaultdict(list)
        for clip in row.get("reference_clips", []) or []:  # type: ignore[union-attr]
            clip_centroids = data.centroids(str(clip["path"]))
            if clip_centroids:
                session_reference[str(clip["speaker"])].append(next(iter(clip_centroids.values())))

        for cluster, speaker in sorted(speaker_of_cluster.items()):
            # Enrolled elsewhere: a different session than this meeting.
            elsewhere = [
                vector
                for session, vectors in pool.get(speaker, {}).items()
                if session != row["session"]
                for vector in vectors
            ]
            if not elsewhere:
                continue
            enrolled = elsewhere[0]
            scores = {
                int(key): cosine_similarity(enrolled, vector) for key, vector in centroids.items()
            }
            best_cluster = max(scores, key=lambda key: scores[key])

            record: dict[str, object] = {
                "mixture": Path(path).name,
                "session": row["session"],
                "speaker": speaker,
                "clusters_detected": len(centroids),
                "clusters_ground_truth": len(speaker_of_cluster),
                "diarization_purity": round(purity, 4),
                "true_cluster": cluster,
                "enrolled_best_cluster": best_cluster,
                "same_cluster": best_cluster == cluster,
                "score_true_cluster": round(scores.get(cluster, 0.0), 4),
                "score_best_cluster": round(scores[best_cluster], 4),
                "margin": round(scores[best_cluster] - scores.get(cluster, 0.0), 4),
            }

            reference = session_reference.get(speaker, [])
            if reference:
                reference_scores = {
                    int(key): cosine_similarity(reference[0], vector)
                    for key, vector in centroids.items()
                }
                reference_best = max(reference_scores, key=lambda key: reference_scores[key])
                record["same_session_best_cluster"] = reference_best
                record["same_session_same_cluster"] = reference_best == cluster
                record["same_session_score_true_cluster"] = round(
                    reference_scores.get(cluster, 0.0), 4
                )
            cases.append(record)
            print(
                f"    {Path(path).name[:30]:30s} {speaker} true=spk_{cluster} "
                f"best=spk_{best_cluster} score={scores[best_cluster]:.3f} "
                f"{'MATCH' if best_cluster == cluster else 'MISS'}",
                flush=True,
            )

    positives = [float(case["score_true_cluster"]) for case in cases]
    best_scores = [float(case["score_best_cluster"]) for case in cases]
    agreement = sum(1 for case in cases if case["same_cluster"]) / len(cases) if cases else 0.0
    same_session = [case for case in cases if "same_session_same_cluster" in case]
    return {
        "cases": len(cases),
        "cluster_agreement": round(agreement, 4),
        "score_true_cluster": describe(positives),
        "score_best_cluster": describe(best_scores),
        "same_session_agreement": round(
            sum(1 for case in same_session if case["same_session_same_cluster"])
            / max(1, len(same_session)),
            4,
        ),
        "same_session_cases": len(same_session),
        "diarization_purity": describe([float(case["diarization_purity"]) for case in cases]),
        "details": cases,
    }


# ----------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------
def render_markdown(report: dict[str, object]) -> str:
    def stats_row(name: str, stats: dict[str, float]) -> str:
        if not stats:
            return f"| {name} | (no data) | | | | | | |"
        return (
            f"| {name} | {stats['count']} | {stats['min']:.3f} | {stats['p05']:.3f} | "
            f"{stats['median']:.3f} | {stats['p95']:.3f} | {stats['max']:.3f} | "
            f"{stats['mean']:.3f} |"
        )

    one = report["experiment_one"]
    two = report["experiment_two"]
    three = report["experiment_three"]

    lines = [
        "# Voiceprint calibration on real Chinese meetings",
        "",
        f"Source: {report['source']}",
        f"Enrolment clips: {one['clips']} · speakers: {len(one['speakers'])} · "
        f"sessions: {len(one['sessions'])}",
        "",
        "## 1. Voiceprint stability",
        "",
        "| population | n | min | p05 | median | p95 | max | mean |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        stats_row("same speaker, different session", one["same_speaker_cross_session"]),
        stats_row("same speaker, same session", one["same_speaker_same_session"]),
        stats_row("different speakers", one["different_speakers"]),
        "",
        f"- best separating threshold: **{one['best_separating_threshold']['threshold']}** "
        f"(accuracy {one['best_separating_threshold']['accuracy']})",
        f"- equal error rate at threshold **{one['equal_error_rate']['threshold']}** "
        f"(EER {one['equal_error_rate']['eer']})",
        "",
        "### Threshold sweep",
        "",
        "| threshold | recall (same speaker) | false accept (other speaker) |",
        "| ---: | ---: | ---: |",
    ]
    for entry in one["threshold_sweep"]:
        lines.append(
            f"| {entry['threshold']:.2f} | {entry['recall_same_speaker']:.3f} | "
            f"{entry['false_accept_other_speaker']:.3f} |"
        )

    lines += [
        "",
        "### Hardest impostor pairs (different people scoring highest)",
        "",
        "| score | speaker A | session A | speaker B | session B |",
        "| ---: | --- | --- | --- | --- |",
    ]
    for entry in one["hardest_impostors"]:
        lines.append(
            f"| {entry['score']:.3f} | {entry['speaker_a']} | {entry['session_a']} | "
            f"{entry['speaker_b']} | {entry['session_b']} |"
        )

    lines += [
        "",
        "### Hardest genuine pairs (same person scoring lowest)",
        "",
        "| score | speaker | session A | session B |",
        "| ---: | --- | --- | --- |",
    ]
    for entry in one["hardest_genuine"]:
        lines.append(
            f"| {entry['score']:.3f} | {entry['speaker']} | {entry['session_a']} | "
            f"{entry['session_b']} |"
        )

    lines += [
        "",
        "## 2. Scoring rule vs number of enrolments",
        "",
        "Attune stores several voiceprints per identity. `centroid` compares "
        "against their average, `max` against the best of them. Only `centroid` "
        "keeps an impostor's score from drifting up as identities accumulate.",
        "",
    ]
    for rule, per_k in two["rules"].items():
        lines += [
            f"### rule: {rule}",
            "",
            "| k enrolments | same speaker median | impostor median | impostor p95 | impostor max |",
            "| ---: | ---: | ---: | ---: | ---: |",
        ]
        for entry in per_k:
            impostor = entry["impostor_score"]
            lines.append(
                f"| {entry['k']} | {entry['same_speaker_score']['median']:.3f} | "
                f"{impostor['median']:.3f} | {impostor['p95']:.3f} | {impostor['max']:.3f} |"
            )
        lines.append("")

    lines += [
        "## 3. Cross-meeting recognition",
        "",
        f"- meetings processed: {three['cases']}",
        f"- cluster agreement (enrolled in another meeting → right person): "
        f"**{three['cluster_agreement']}**",
        f"- same-session reference agreement (ceiling): **{three['same_session_agreement']}** "
        f"over {three['same_session_cases']} cases",
        "",
        "| population | n | min | p05 | median | p95 | max | mean |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        stats_row("enrolled speaker vs its true cluster", three["score_true_cluster"]),
        stats_row("enrolled speaker vs best cluster", three["score_best_cluster"]),
        stats_row("diarization purity vs text ground truth", three["diarization_purity"]),
        "",
        "| meeting | speaker | clusters | true | enrolled best | score | same-session ref | same? |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | :---: |",
    ]
    for case in three["details"]:
        same_session = (
            "✓"
            if case.get("same_session_same_cluster")
            else "✗"
            if "same_session_same_cluster" in case
            else "—"
        )
        lines.append(
            f"| {case['mixture'][:30]} | {case['speaker']} | {case['clusters_detected']} | "
            f"spk_{case['true_cluster']} | spk_{case['enrolled_best_cluster']} | "
            f"{case['score_true_cluster']:.3f} | {same_session} | "
            f"{'✓' if case['same_cluster'] else '✗'} |"
        )
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(ROOT / "experiments" / "data" / "ps4"))
    parser.add_argument("--out", default=str(ROOT / "experiments"))
    args = parser.parse_args(argv)

    data = Dataset(Path(args.data))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, object] = {
        "source": "REAL-PS4 / AISHELL-4 (real recorded Mandarin meetings)",
        "experiment_one": experiment_one(data),
        "experiment_two": experiment_two(data),
        "experiment_three": experiment_three(data),
    }
    (out_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "report.md").write_text(render_markdown(report), encoding="utf-8")
    print("\n" + render_markdown(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
