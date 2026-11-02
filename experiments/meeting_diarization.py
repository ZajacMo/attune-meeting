"""Diarization and cross-meeting recognition on realistic-length audio.

FunASR's cluster backend refuses to split anything when it sees fewer than
20 segments (``funasr/models/campplus/cluster_backend.py``), so a 15-second
clip always comes back as a single speaker. That makes short test clips
useless for judging diarization quality — a real meeting is minutes long.

This builds meetings that are minutes long by stitching together several
AISHELL-4 mixture windows from the *same* recorded session. Every window is
real meeting audio with real overlapping speakers, and the metadata says who
spoke in it, so the stitched meeting still has known ground truth.

It then measures two things:

* how many speakers the engine separates, against how many are present;
* whether a voiceprint enrolled in a *different* meeting finds the right
  person inside the stitched meeting.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import wave
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from extract_voiceprints import Extractor  # noqa: E402
from ground_truth import META_MIXTURE, MIXTURE, read_meta  # noqa: E402
from huggingface_hub import hf_hub_download, list_repo_files  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REPO = "tian0105/REAL-PS4"

PUNCTUATION = None  # set in main


def normalize(text: str) -> str:
    import re

    return re.sub(r"[\W_]+", "", text, flags=re.UNICODE)


def text_similarity(left: str, right: str) -> float:
    return SequenceMatcher(None, normalize(left), normalize(right)).ratio()


def load_overlap_records(data_dir: Path) -> dict[tuple[str, float, float], dict[str, object]]:
    """The time layout of every mixture window.

    ``overlap_records.json`` states, for each mixture window, which person was
    speaking during which part of it. Those times are in the mixture's own
    clock, which is what makes a turn attributable to a person by time rather
    than by how the ASR mangled overlapping speech.

    The enrolment metadata cannot be used for this: it gives the times the
    utterances were *recorded*, which are unrelated to where they land inside
    the mixture.
    """
    path = data_dir / "raw" / "AISHELL-4" / "TRAIN" / "json" / "AISHELL-4" / "overlap_records.json"
    if not path.exists():
        downloaded = hf_hub_download(
            REPO,
            "AISHELL-4/TRAIN/json/AISHELL-4/overlap_records.json",
            repo_type="dataset",
            local_dir=str(data_dir / "raw"),
        )
        path = Path(downloaded)
    records = json.loads(path.read_text(encoding="utf-8"))
    index: dict[tuple[str, float, float], dict[str, object]] = {}
    for key, entries in records.items():
        session = re.sub(r"^\d{8}_[A-Z]_", "", key)
        for entry in entries:
            info = entry["overlap_info"]
            window = (
                session,
                round(float(info["overlap_start_time"]), 2),
                round(float(info["overlap_end_time"]), 2),
            )
            index[window] = entry
    return index


def cross_session_measurement(
    meeting_name: str,
    session: str,
    clusters_of_speaker: dict[str, list[int]],
    centroids: dict[int, list[float]],
    enrolment: dict[tuple[str, str], list[list[float]]],
    same_session: dict[str, list[list[float]]],
) -> list[dict[str, object]]:
    """Match one meeting's clusters against voiceprints enrolled elsewhere.

    The only difference between the two ground truths is which clusters belong
    to which person; the scoring is identical.
    """
    records: list[dict[str, object]] = []
    for speaker, owned in clusters_of_speaker.items():
        elsewhere = [
            vector
            for (other, other_session), vectors in enrolment.items()
            if other == speaker and other_session != session
            for vector in vectors
        ]
        if not elsewhere:
            continue
        enrolled = elsewhere[0]
        scores = {
            int(key): cosine_similarity(enrolled, vector) for key, vector in centroids.items()
        }
        best_cluster = max(scores, key=lambda key: scores[key])
        ranked = sorted(scores.values(), reverse=True)
        record: dict[str, object] = {
            "meeting": meeting_name,
            "session": session,
            "speaker": speaker,
            "owned_clusters": owned,
            "best_cluster": best_cluster,
            "correct_person": best_cluster in owned,
            "exact_cluster": best_cluster == owned[0],
            "score": round(scores[best_cluster], 4),
            "runner_up": round(ranked[1], 4) if len(ranked) > 1 else 0.0,
            "margin": round(scores[best_cluster] - (ranked[1] if len(ranked) > 1 else 0.0), 4),
        }
        # The realistic cross-session condition: the identity holds every
        # voiceprint enrolled from other sessions, and is scored against
        # their centroid — the same rule the registry uses.
        if len(elsewhere) > 1:
            anchor = [sum(values) / len(values) for values in zip(*elsewhere, strict=True)]
            profile_scores = {
                int(key): cosine_similarity(anchor, vector) for key, vector in centroids.items()
            }
            profile_best = max(profile_scores, key=lambda key: profile_scores[key])
            record["profile_best_cluster"] = profile_best
            record["profile_correct_person"] = profile_best in owned
            record["profile_score"] = round(profile_scores[profile_best], 4)

        # Control: the same voice, recorded in this very meeting, before
        # the overlap was mixed in.
        clean_vectors = same_session.get(speaker, [])
        if clean_vectors:
            anchor = [sum(values) / len(values) for values in zip(*clean_vectors, strict=True)]
            clean_scores = {
                int(key): cosine_similarity(anchor, vector) for key, vector in centroids.items()
            }
            clean_best = max(clean_scores, key=lambda key: clean_scores[key])
            record["same_session_best"] = clean_best
            record["same_session_correct"] = clean_best in owned
            record["same_session_score"] = round(clean_scores[clean_best], 4)
        records.append(record)
    return records


def meeting_to_meeting_measurement(
    meeting_clusters: dict[str, tuple[dict[int, list[float]], dict[str, list[int]]]],
    meeting_index: dict[str, dict[str, list[list[float]]]],
) -> list[dict[str, object]]:
    """A voice enrolled in one meeting, looked for in another one."""
    rows: list[dict[str, object]] = []
    for meeting_name, (centroids, clusters_of_speaker) in meeting_clusters.items():
        for speaker, owned in clusters_of_speaker.items():
            source = None
            for other_name, other_speakers in meeting_index.items():
                if other_name == meeting_name or speaker not in other_speakers:
                    continue
                source = other_name
                break
            if source is None:
                continue
            enrolled_clusters = meeting_index[source][speaker]
            anchor = [sum(values) / len(values) for values in zip(*enrolled_clusters, strict=True)]
            scores = {
                int(key): cosine_similarity(anchor, vector) for key, vector in centroids.items()
            }
            best = max(scores, key=lambda key: scores[key])
            rows.append(
                {
                    "meeting": meeting_name,
                    "enrolled_from": source,
                    "speaker": speaker,
                    "owned_clusters": owned,
                    "best_cluster": best,
                    "correct_person": best in owned,
                    "score": round(scores[best], 4),
                }
            )
    return rows


def read_frames(path: Path) -> tuple[tuple[int, int, int], bytes]:
    with wave.open(str(path), "rb") as handle:
        params = handle.getparams()
        return (params.nchannels, params.framerate, params.sampwidth), handle.readframes(
            handle.getnframes()
        )


def write_frames(path: Path, params: tuple[int, int, int], frames: bytes) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(params[0])
        handle.setframerate(params[1])
        handle.setsampwidth(params[2])
        handle.writeframes(frames)


def silence(params: tuple[int, int, int], seconds: float) -> bytes:
    return b"\x00" * int(params[1] * seconds) * params[0] * params[2]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    import math

    width = min(len(left), len(right))
    dot = sum(left[i] * right[i] for i in range(width))
    left_norm = math.sqrt(sum(left[i] * left[i] for i in range(width)))
    right_norm = math.sqrt(sum(right[i] * right[i] for i in range(width)))
    if left_norm <= 0 or right_norm <= 0:
        return 0.0
    return dot / (left_norm * right_norm)


def enrolment_pool(data_dir: Path) -> dict[tuple[str, str], list[list[float]]]:
    """Voiceprints of every speaker, grouped by the session they come from."""
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    cache = json.loads((data_dir / "voiceprints.json").read_text(encoding="utf-8"))
    root = Path(__file__).resolve().parents[1]
    pool: dict[tuple[str, str], list[list[float]]] = defaultdict(list)
    for row in manifest:
        if row.get("kind") != "enrolment":
            continue
        key = str(root / str(row["path"]))
        entry = cache.get(key, {})
        centroids = entry.get("centroids", {})
        if not centroids:
            continue
        pool[(str(row["speaker"]), str(row["session"]))].append(
            list(next(iter(centroids.values())))
        )
    return pool


_INDEX: tuple[dict, dict] | None = None


def clean_clips_for(
    data_dir: Path, session: str, window_key: tuple[str, str, str]
) -> dict[str, str]:
    """The clean enrolment audio that was mixed into one window, per speaker.

    This is the control condition: it is the same voice, recorded in the same
    meeting, before the overlap was mixed in.
    """
    global _INDEX
    from ground_truth import build_index, mixture_composition
    from huggingface_hub import list_repo_files

    if _INDEX is None:
        _INDEX = build_index(list_repo_files(REPO, repo_type="dataset"))
    index, mixtures = _INDEX
    rows = mixtures.get(window_key, [])
    out: dict[str, str] = {}
    for constituent in mixture_composition(window_key, rows, index):
        if not constituent["repo_path"]:
            continue
        out.setdefault(constituent["speaker"], constituent["repo_path"])
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(ROOT / "experiments" / "data" / "ps4"))
    parser.add_argument("--meetings", type=int, default=6)
    parser.add_argument("--windows-per-meeting", type=int, default=4)
    parser.add_argument("--out", default=str(ROOT / "experiments"))
    args = parser.parse_args(argv)

    #: Thresholds every sweep in this script is measured on.
    grid = [0.45, 0.50, 0.53, 0.55, 0.58, 0.60, 0.62, 0.65, 0.68, 0.70, 0.72, 0.75, 0.80]

    data_dir = Path(args.data)
    build_dir = data_dir / "meetings"
    build_dir.mkdir(parents=True, exist_ok=True)

    # ---- metadata: windows grouped by session, with their speakers -------
    rows = read_meta(data_dir / "AISHELL-4_meta.csv")
    windows: dict[tuple[str, str, str], dict[str, object]] = {}
    for row in rows:
        match = META_MIXTURE.search(row.get("mixture_utterance", ""))
        if not match:
            continue
        info = match.groupdict()
        key = (info["session"], info["start"], info["end"])
        entry = windows.setdefault(
            key,
            {
                "speakers": set(),
                "transcripts": {},
                "duration": float(row.get("mixture_duration", 0) or 0),
                "overlap": float(row.get("mixture_overlap_duration", 0) or 0),
            },
        )
        speaker = row.get("speaker", "")
        entry["speakers"].add(speaker)  # type: ignore[union-attr]
        entry["transcripts"][speaker] = row.get("ground_truth_transcript", "")  # type: ignore[index]

    repo_mixtures: dict[tuple[str, str, str], str] = {}
    for path in list_repo_files(REPO, repo_type="dataset"):
        match = MIXTURE.search(path)
        if match:
            info = match.groupdict()
            repo_mixtures[(info["session"], info["start"], info["end"])] = path

    # Group windows by session and by speaker set, so a stitched meeting keeps
    # a consistent cast.
    by_cast: dict[
        tuple[str, frozenset[str]], list[tuple[tuple[str, str, str], dict[str, object]]]
    ] = defaultdict(list)
    for key, entry in windows.items():
        speakers = frozenset(entry["speakers"])  # type: ignore[arg-type]
        if len(speakers) < 2:
            continue
        by_cast[(key[0], speakers)].append((key, entry))

    casts = sorted(by_cast.items(), key=lambda item: -len(item[1]))
    print(f"casts available: {len(casts)}")

    # One meeting per speaker count, so the report covers 2, 3 and 4 people.
    by_count: dict[int, list[tuple[tuple[str, frozenset[str]], list]]] = defaultdict(list)
    for cast, group in casts:
        by_count[len(cast[1])].append((cast, group))

    chosen_casts: list[tuple[tuple[str, frozenset[str]], list]] = []
    per_count = max(1, args.meetings // max(1, len(by_count)))
    for count in sorted(by_count):
        for cast, group in sorted(by_count[count], key=lambda item: -len(item[1]))[:per_count]:
            chosen_casts.append((cast, group))

    # ---- build the meetings ---------------------------------------------
    meetings: list[dict[str, object]] = []
    for (session, _speakers), group in chosen_casts:
        if len(meetings) >= args.meetings:
            break
        group = sorted(group, key=lambda item: -float(item[1]["duration"]))
        chosen = group[: args.windows_per_meeting]
        if len(chosen) < 2:
            continue

        frames = b""
        params: tuple[int, int, int] | None = None
        used: list[dict[str, object]] = []
        offset = 0.0
        for key, entry in chosen:
            repo_path = repo_mixtures.get(key)
            if not repo_path:
                continue
            source = build_dir / Path(repo_path).name
            if not source.exists():
                downloaded = hf_hub_download(
                    REPO, repo_path, repo_type="dataset", local_dir=str(data_dir / "raw")
                )
                source.write_bytes(Path(downloaded).read_bytes())
            chunk_params, chunk = read_frames(source)
            if params is None:
                params = chunk_params
            elif chunk_params[:2] != params[:2]:
                continue
            if frames:
                frames += silence(params, 0.4)
                offset += 0.4
            window_offset = offset
            frames += chunk
            offset += len(chunk) / (params[1] * params[0] * params[2])
            used.append(
                {
                    "key": key,
                    "offset_s": round(window_offset, 3),
                    "speakers": sorted(entry["speakers"]),  # type: ignore[arg-type]
                    "transcripts": entry["transcripts"],
                    "duration": entry["duration"],
                    "overlap_fraction": round(float(entry["overlap"]) / float(entry["duration"]), 2)
                    if entry["duration"]
                    else 0.0,
                }
            )
        if params is None or len(used) < 2:
            continue

        meeting_path = build_dir / f"meeting_{session}_{len(meetings):02d}.wav"
        write_frames(meeting_path, params, frames)
        duration = len(frames) / (params[1] * params[0] * params[2])
        meetings.append(
            {
                "path": str(meeting_path),
                "session": session,
                "speakers": sorted(set().union(*[set(u["speakers"]) for u in used])),
                "duration_s": round(duration, 1),
                "windows": used,
            }
        )
        print(
            f"  built {meeting_path.name}: {duration:.0f}s, "
            f"{len(used)} windows, speakers={meetings[-1]['speakers']}",
            flush=True,
        )

    # ---- the time layout of every window -------------------------------
    overlap_records = load_overlap_records(data_dir)

    # ---- run the engine --------------------------------------------------
    cache_path = data_dir / "meeting_voiceprints.json"
    cache: dict[str, dict[str, object]] = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
    extractor: Extractor | None = None
    results: list[dict[str, object]] = []
    cross_results: list[dict[str, object]] = []
    cross_results_time: list[dict[str, object]] = []
    # meeting name -> speaker -> the voiceprints the registry would store
    meeting_index: dict[str, dict[str, list[list[float]]]] = {}
    # meeting name -> (cluster centroids, clusters per speaker)
    meeting_clusters: dict[str, tuple[dict[int, list[float]], dict[str, list[int]]]] = {}
    # meeting name -> (owner per cluster under the timeline, cluster centroids)
    timeline_data: dict[str, dict[str, object]] = {}
    for meeting in meetings:
        key = str(meeting["path"])
        if key not in cache:
            if extractor is None:
                extractor = Extractor()
            cache[key] = extractor.run(key)
            cache_path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
        outcome = cache[key]
        turns = outcome["turns"]
        # JSON object keys are strings; cluster ids are numbers.
        centroids = {int(name): vector for name, vector in outcome["centroids"].items()}

        # Which speaker does each turn belong to? A stitched meeting holds
        # several windows, each with its own transcripts, so a turn is matched
        # against every transcript its speaker could have come from.
        transcripts: dict[str, list[str]] = defaultdict(list)
        for window in meeting["windows"]:  # type: ignore[union-attr]
            for speaker, transcript in window["transcripts"].items():  # type: ignore[union-attr]
                if transcript and transcript not in transcripts[speaker]:
                    transcripts[speaker].append(transcript)

        labelled: list[dict[str, object]] = []
        for turn in turns:
            text = str(turn.get("text", ""))
            scores = {
                speaker: max(text_similarity(text, transcript) for transcript in options)
                for speaker, options in transcripts.items()
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

        # Merge audit. A cluster that carries attributed turns from two
        # different documented speakers is a merge: the engine put two people
        # into one speaker. Clusters with no attributed turn cannot be judged.
        cluster_speakers: dict[int, set[str]] = defaultdict(set)
        for row in labelled:
            cluster_speakers[int(row["cluster"])].add(str(row["speaker"]))
        merged_clusters = sum(1 for speakers in cluster_speakers.values() if len(speakers) > 1)

        # ---- time-based ground truth -------------------------------------
        # The overlap records say who was speaking during which part of each
        # window, in the window's own clock. Placing every ASR turn on the
        # timeline of the stitched meeting therefore attributes it to a person
        # by *time*, with no reliance on how the ASR mangled overlapping
        # speech.
        timeline: list[tuple[float, float, str]] = []
        for window in meeting["windows"]:  # type: ignore[union-attr]
            key = tuple(window["key"])  # type: ignore[index]
            window_start = float(key[1])
            window_offset = float(window["offset_s"])  # type: ignore[arg-type]
            record = overlap_records.get(
                (str(key[0]), round(window_start, 2), round(float(key[2]), 2))
            )
            if record is None:
                continue
            for segment in record["overlap_segments"]:  # type: ignore[index]
                start = window_offset + (float(segment["start_time"]) - window_start)
                end = window_offset + (float(segment["end_time"]) - window_start)
                timeline.append((start, end, str(segment["speaker"])))

        timeline_clusters: dict[int, Counter[str]] = defaultdict(Counter)
        timeline_attributed = 0
        for turn in turns:
            turn_start = float(turn.get("start", 0.0))
            turn_end = float(turn.get("end", 0.0))
            duration = max(turn_end - turn_start, 1e-6)
            overlaps = sorted(
                (
                    (min(turn_end, end) - max(turn_start, start), speaker)
                    for start, end, speaker in timeline
                    if min(turn_end, end) - max(turn_start, start) > 0
                ),
                reverse=True,
            )
            if not overlaps:
                continue
            best, best_speaker = overlaps[0]
            runner_up = overlaps[1][0] if len(overlaps) > 1 else 0.0
            # A turn that two people spoke at once belongs to both; only a turn
            # with one clear owner can be used to judge a cluster.
            if best < 0.3 * duration or runner_up > 0.5 * best:
                continue
            timeline_attributed += 1
            timeline_clusters[int(turn.get("cluster", 0))][best_speaker] += 1

        measured = {cluster: counts for cluster, counts in timeline_clusters.items() if counts}
        timeline_merged = sum(1 for counts in measured.values() if len(counts) > 1)
        timeline_purity = (
            sum(max(counts.values()) / sum(counts.values()) for counts in measured.values())
            / len(measured)
            if measured
            else 0.0
        )
        merged_detail = [
            {
                "cluster": cluster,
                "speakers": dict(sorted(counts.items())),
            }
            for cluster, counts in sorted(measured.items())
            if len(counts) > 1
        ]
        # The owner of a cluster under the time-based ground truth: whoever
        # holds most of its attributable turns.
        time_owner = {
            cluster: max(counts.items(), key=lambda item: (item[1], item[0]))[0]
            for cluster, counts in measured.items()
        }
        timeline_data[Path(str(meeting["path"])).name] = {
            "owner": time_owner,
            "centroids": centroids,
        }
        # The same mapping, one cluster per person: a person may own several
        # clusters when the engine over-segments.
        clusters_of_speaker_time: dict[str, list[int]] = defaultdict(list)
        for cluster, speaker in time_owner.items():
            clusters_of_speaker_time[speaker].append(cluster)

        # One cluster per speaker. A speaker may own several clusters when the
        # engine over-segments, so the mapping keeps every cluster a speaker
        # holds.
        by_speaker: dict[str, Counter[int]] = defaultdict(Counter)
        for row in labelled:
            by_speaker[str(row["speaker"])][int(row["cluster"])] += 1
        claimed: set[int] = set()
        speaker_of_cluster: dict[int, str] = {}
        clusters_of_speaker: dict[str, list[int]] = defaultdict(list)
        for speaker in sorted(by_speaker):
            for cluster, _ in by_speaker[speaker].most_common():
                if cluster in claimed:
                    continue
                claimed.add(cluster)
                speaker_of_cluster[cluster] = speaker
                clusters_of_speaker[speaker].append(cluster)

        # ---- cross-session recognition inside this meeting ---------------
        enrolment = enrolment_pool(data_dir)
        clean_cache = data_dir / "clean_clips.json"
        clean: dict[str, dict[str, object]] = (
            json.loads(clean_cache.read_text(encoding="utf-8")) if clean_cache.exists() else {}
        )
        # The clean audio that was mixed into this meeting, per speaker.
        same_session: dict[str, list[list[float]]] = defaultdict(list)
        for window in meeting["windows"]:  # type: ignore[union-attr]
            window_key = tuple(window["key"])  # type: ignore[index]
            speakers = clean_clips_for(data_dir, str(meeting["session"]), window_key)  # type: ignore[arg-type]
            for speaker, repo_path in speakers.items():
                if speaker not in window["speakers"]:  # type: ignore[operator]
                    continue
                clip_path = build_dir / Path(repo_path).name
                if str(clip_path) not in clean:
                    if not clip_path.exists():
                        downloaded = hf_hub_download(
                            REPO, repo_path, repo_type="dataset", local_dir=str(data_dir / "raw")
                        )
                        clip_path.write_bytes(Path(downloaded).read_bytes())
                    if extractor is None:
                        extractor = Extractor()
                    clean[str(clip_path)] = extractor.run(str(clip_path))
                    clean_cache.write_text(json.dumps(clean, ensure_ascii=False), encoding="utf-8")
                clip_centroids = clean[str(clip_path)]["centroids"]
                if clip_centroids:
                    same_session[speaker].append(list(next(iter(clip_centroids.values()))))

        # The condition the product actually runs in: the identity is enrolled
        # from one *meeting* and the voice has to be found in another meeting.
        meeting_name = Path(str(meeting["path"])).name
        meeting_index[meeting_name] = {
            speaker: [centroids[int(cluster)] for cluster in owned]
            for speaker, owned in clusters_of_speaker.items()
            if all(int(cluster) in centroids for cluster in owned)
        }
        meeting_clusters[meeting_name] = (centroids, dict(clusters_of_speaker))

        cross = cross_session_measurement(
            meeting_name,
            str(meeting["session"]),
            clusters_of_speaker,
            centroids,
            enrolment,
            same_session,
        )
        cross_time = cross_session_measurement(
            meeting_name,
            str(meeting["session"]),
            dict(clusters_of_speaker_time),
            centroids,
            enrolment,
            same_session,
        )

        documented = len(meeting["speakers"])  # type: ignore[arg-type]
        detected = len(centroids)
        correct = sum(
            1
            for cluster, speaker in speaker_of_cluster.items()
            if speaker in meeting["speakers"]  # type: ignore[operator]
        )
        results.append(
            {
                "meeting": Path(str(meeting["path"])).name,
                "session": meeting["session"],
                "duration_s": meeting["duration_s"],
                "segments": len(turns),
                "documented_speakers": documented,
                "detected_clusters": detected,
                "clusters_with_speaker": len(speaker_of_cluster),
                "correctly_labelled_clusters": correct,
                "clusters_with_attributed_speaker": len(cluster_speakers),
                "text_merged_clusters": merged_clusters,
                "timeline_turns_attributed": timeline_attributed,
                "timeline_clusters_measured": len(measured),
                "timeline_merged_clusters": timeline_merged,
                "timeline_purity": round(timeline_purity, 4),
                "timeline_merged_detail": merged_detail,
                "turns_labelled": len(labelled),
                "turns_total": len(turns),
            }
        )
        print(
            f"  {results[-1]['meeting']}: {results[-1]['duration_s']}s "
            f"segments={len(turns)} speakers={documented} clusters={detected} "
            f"labelled_correctly={correct}/{len(speaker_of_cluster)}",
            flush=True,
        )
        for row in cross:
            cross_results.append(row)
            print(
                f"      cross-session {row['speaker']}: enrolled in another meeting "
                f"-> spk_{row['best_cluster']} score={row['score']:.3f} "
                f"{'RIGHT PERSON' if row['correct_person'] else 'WRONG PERSON'}",
                flush=True,
            )
        for row in cross_time:
            cross_results_time.append(row)

    # ---- the same measurement against the time-based ground truth -------
    # The transcript-based attribution cannot be trusted on overlapping
    # speech: the ASR mangles two people talking at once into one sentence.
    # This repeats the registry measurement using the ownership the timeline
    # gives, so the numbers rest on the same kind of evidence as the rest of
    # this document.
    time_rows: list[dict[str, object]] = []
    for meeting_name, entry in timeline_data.items():
        owner = entry["owner"]  # type: ignore[assignment]
        meeting_centroids = entry["centroids"]  # type: ignore[assignment]
        registry: list[tuple[str, list[float]]] = []
        by_speaker: dict[str, list[list[float]]] = defaultdict(list)
        for other_name, other in timeline_data.items():
            if other_name == meeting_name:
                continue
            for cluster, speaker in other["owner"].items():  # type: ignore[union-attr]
                vector = other["centroids"].get(cluster)  # type: ignore[union-attr]
                if vector is not None:
                    by_speaker[str(speaker)].append(vector)
        for speaker, vectors in sorted(by_speaker.items()):
            anchor = [sum(values) / len(values) for values in zip(*vectors, strict=True)]
            registry.append((speaker, anchor))
        if len(registry) < 2:
            continue
        for cluster, speaker in owner.items():
            vector = meeting_centroids.get(cluster)
            if vector is None:
                continue
            ranked = sorted(
                ((cosine_similarity(anchor, vector), name) for name, anchor in registry),
                reverse=True,
            )
            best_score, best_name = ranked[0]
            runner_up = ranked[1][0] if len(ranked) > 1 else 0.0
            time_rows.append(
                {
                    "meeting": meeting_name,
                    "cluster": cluster,
                    "speaker": speaker,
                    "best": best_name,
                    "correct_person": best_name == speaker,
                    "score": round(best_score, 4),
                    "runner_up": round(runner_up, 4),
                    "margin": round(best_score - runner_up, 4),
                }
            )

    time_sweep: list[dict[str, object]] = []
    for threshold in grid:
        accepted = [row for row in time_rows if float(row["score"]) >= threshold]
        right = sum(1 for row in accepted if bool(row["correct_person"]))
        time_sweep.append(
            {
                "threshold": threshold,
                "accepted": len(accepted),
                "correct": right,
                "precision": round(right / len(accepted), 4) if accepted else None,
                "recall": round(right / len(time_rows), 4) if time_rows else None,
            }
        )

    # ---- meeting to meeting: the condition the product actually runs in ---
    meeting_to_meeting = meeting_to_meeting_measurement(meeting_clusters, meeting_index)

    # The same, with the clusters owned by whoever was speaking at the time.
    meeting_clusters_time: dict[str, tuple[dict[int, list[float]], dict[str, list[int]]]] = {}
    meeting_index_time: dict[str, dict[str, list[list[float]]]] = {}
    for meeting_name, entry in timeline_data.items():
        owner = entry["owner"]  # type: ignore[assignment]
        meeting_centroids = entry["centroids"]  # type: ignore[assignment]
        owned_by_speaker: dict[str, list[int]] = defaultdict(list)
        for cluster, speaker in owner.items():
            owned_by_speaker[str(speaker)].append(cluster)
        meeting_clusters_time[meeting_name] = (meeting_centroids, dict(owned_by_speaker))
        meeting_index_time[meeting_name] = {
            speaker: [meeting_centroids[cluster] for cluster in owned]
            for speaker, owned in owned_by_speaker.items()
            if all(cluster in meeting_centroids for cluster in owned)
        }
    meeting_to_meeting_time = meeting_to_meeting_measurement(
        meeting_clusters_time, meeting_index_time
    )

    # ---- identification inside a registry of known people -----------------
    # This is the operation the product performs: a voice comes in, and the
    # registry holds several identities. The question is not only "is the best
    # match above a threshold" but "is it clearly better than the runner-up".
    registry_rows: list[dict[str, object]] = []
    for meeting_name, (centroids, clusters_of_speaker) in meeting_clusters.items():
        # The registry holds one profile per person, exactly as the product
        # does: the centroid of every voiceprint learnt for that person.
        registry: list[tuple[str, list[float]]] = []
        by_speaker: dict[str, list[list[float]]] = defaultdict(list)
        for other_name, other_speakers in meeting_index.items():
            if other_name == meeting_name:
                continue
            for speaker, vectors in other_speakers.items():
                by_speaker[speaker].extend(vectors)
        for speaker, vectors in sorted(by_speaker.items()):
            anchor = [sum(values) / len(values) for values in zip(*vectors, strict=True)]
            registry.append((speaker, anchor))
        if len(registry) < 2:
            continue
        for speaker, owned in clusters_of_speaker.items():
            for cluster in owned:
                if cluster not in centroids:
                    continue
                ranked = sorted(
                    (
                        (cosine_similarity(anchor, centroids[cluster]), name)
                        for name, anchor in registry
                    ),
                    reverse=True,
                )
                best_score, best_name = ranked[0]
                runner_up = ranked[1][0] if len(ranked) > 1 else 0.0
                registry_rows.append(
                    {
                        "meeting": meeting_name,
                        "cluster": cluster,
                        "speaker": speaker,
                        "best": best_name,
                        "best_speaker": best_name,
                        "correct_person": best_name == speaker,
                        "score": round(best_score, 4),
                        "runner_up": round(runner_up, 4),
                        "margin": round(best_score - runner_up, 4),
                    }
                )

    registry_identification = {
        "cases": len(registry_rows),
        "correct_person": sum(1 for row in registry_rows if bool(row["correct_person"])),
        "rows": registry_rows,
    }

    report = {
        "source": "REAL-PS4 / AISHELL-4 stitched meetings (real Mandarin meeting "
        "audio with real overlapping speech)",
        "engine": "FunASR paraformer-zh + fsmn-vad + ct-punc + cam++ (CPU)",
        "note": "FunASR's cluster backend returns a single cluster whenever it "
        "sees fewer than 20 segments, so short clips cannot be used to judge "
        "diarization quality.",
        "meetings": len(results),
        "fully_separated": sum(
            1 for row in results if int(row["detected_clusters"]) == int(row["documented_speakers"])
        ),
        "results": results,
        "cross_session": {
            "cases": len(cross_results),
            "correct_person": sum(1 for row in cross_results if bool(row["correct_person"])),
            "exact_cluster": sum(1 for row in cross_results if bool(row["exact_cluster"])),
            "same_session_cases": sum(1 for row in cross_results if "same_session_correct" in row),
            "same_session_correct_person": sum(
                1 for row in cross_results if bool(row.get("same_session_correct"))
            ),
            "profile_cases": sum(1 for row in cross_results if "profile_correct_person" in row),
            "profile_correct_person": sum(
                1 for row in cross_results if bool(row.get("profile_correct_person"))
            ),
            "details": cross_results,
        },
        "meeting_to_meeting": {
            "cases": len(meeting_to_meeting),
            "correct_person": sum(1 for row in meeting_to_meeting if bool(row["correct_person"])),
            "details": meeting_to_meeting,
        },
        "meeting_to_meeting_time": {
            "note": "The same, with cluster ownership taken from the timeline.",
            "cases": len(meeting_to_meeting_time),
            "correct_person": sum(
                1 for row in meeting_to_meeting_time if bool(row["correct_person"])
            ),
            "details": meeting_to_meeting_time,
        },
        "cross_session_time": {
            "note": "The same, with cluster ownership taken from the timeline.",
            "cases": len(cross_results_time),
            "correct_person": sum(1 for row in cross_results_time if bool(row["correct_person"])),
            "same_session_cases": sum(
                1 for row in cross_results_time if "same_session_correct" in row
            ),
            "same_session_correct_person": sum(
                1 for row in cross_results_time if bool(row.get("same_session_correct"))
            ),
            "profile_cases": sum(
                1 for row in cross_results_time if "profile_correct_person" in row
            ),
            "profile_correct_person": sum(
                1 for row in cross_results_time if bool(row.get("profile_correct_person"))
            ),
        },
        "registry_identification": registry_identification,
        "registry_identification_time": {
            "note": "The same measurement, with cluster ownership taken from the "
            "timeline instead of the ASR transcript.",
            "cases": len(time_rows),
            "correct_person": sum(1 for row in time_rows if bool(row["correct_person"])),
            "sweep": time_sweep,
            "rows": time_rows,
        },
        "timeline": {
            "note": "Turns attributed by time against the documented utterances, "
            "not by the ASR transcript.",
            "turns_attributed": sum(int(row["timeline_turns_attributed"]) for row in results),
            "clusters_measured": sum(int(row["timeline_clusters_measured"]) for row in results),
            "merged_clusters": sum(int(row["timeline_merged_clusters"]) for row in results),
            "meetings_with_a_merge": sum(
                1 for row in results if int(row["timeline_merged_clusters"]) > 0
            ),
            "mean_purity": round(
                sum(float(row["timeline_purity"]) for row in results) / len(results), 4
            )
            if results
            else 0.0,
        },
    }
    out_dir = Path(args.out)

    registry = report["registry_identification"]
    reg_sweep: list[dict[str, object]] = []
    for threshold in grid:
        for margin in (0.0, 0.02, 0.05, 0.08, 0.12):
            accepted = [
                row
                for row in registry["rows"]
                if float(row["score"]) >= threshold and float(row["margin"]) >= margin
            ]
            right = sum(1 for row in accepted if bool(row["correct_person"]))
            reg_sweep.append(
                {
                    "threshold": threshold,
                    "margin": margin,
                    "accepted": len(accepted),
                    "correct": right,
                    "precision": round(right / len(accepted), 4) if accepted else None,
                    "recall": round(right / registry["cases"], 4) if registry["cases"] else None,
                }
            )
    report["registry_threshold_margin"] = reg_sweep

    lines = [
        "# Diarization on realistic-length meetings",
        "",
        f"Engine: {report['engine']}",
        "",
        "| meeting | duration | segments | speakers | clusters | clusters labelled | correct |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in results:
        lines.append(
            f"| {row['meeting']} | {row['duration_s']}s | {row['segments']} | "
            f"{row['documented_speakers']} | {row['detected_clusters']} | "
            f"{row['clusters_with_speaker']} | {row['correctly_labelled_clusters']} |"
        )
    cross = report["cross_session"]
    # The decision that matters: if the engine says "known", how often is it
    # right? The thresholds were calibrated on clean clips; this measures them
    # where they are actually used — against a voiceprint taken from a meeting.
    sweep: list[dict[str, object]] = []
    for threshold in grid:
        accepted = [row for row in cross["details"] if float(row["score"]) >= threshold]
        right = sum(1 for row in accepted if bool(row["correct_person"]))
        sweep.append(
            {
                "threshold": threshold,
                "accepted": len(accepted),
                "correct": right,
                "precision": round(right / len(accepted), 4) if accepted else None,
                "recall": round(right / cross["cases"], 4) if cross["cases"] else None,
            }
        )
    report["threshold_in_meeting"] = sweep

    m2m = report["meeting_to_meeting"]
    m2m_sweep: list[dict[str, object]] = []
    for threshold in grid:
        accepted = [row for row in m2m["details"] if float(row["score"]) >= threshold]
        right = sum(1 for row in accepted if bool(row["correct_person"]))
        m2m_sweep.append(
            {
                "threshold": threshold,
                "accepted": len(accepted),
                "correct": right,
                "precision": round(right / len(accepted), 4) if accepted else None,
                "recall": round(right / m2m["cases"], 4) if m2m["cases"] else None,
            }
        )
    report["threshold_meeting_to_meeting"] = m2m_sweep

    lines += [
        "",
        "## Cross-session recognition inside the meeting",
        "",
        "A voiceprint enrolled in a *different* meeting is matched against every "
        "cluster of this meeting.",
        "",
        f"- cases: {cross['cases']}",
        f"- the right person was found: **{cross['correct_person']}/{cross['cases']}**",
        f"- the exact cluster was found: {cross['exact_cluster']}/{cross['cases']}",
        f"- control (enrolled from the same meeting's clean audio): "
        f"**{cross['same_session_correct_person']}/{cross['same_session_cases']}**",
        f"- cross-session with a full profile (centroid of every enrolment from "
        f"other sessions): **{cross['profile_correct_person']}/{cross['profile_cases']}**",
        "",
        "| meeting | speaker | clusters owned | best | score | right person | profile | same-meeting control |",
        "| --- | --- | ---: | ---: | ---: | :---: | :---: | :---: |",
    ]
    for row in cross["details"]:
        control = (
            "✓"
            if row.get("same_session_correct")
            else "✗"
            if "same_session_correct" in row
            else "—"
        )
        profile = (
            "✓"
            if row.get("profile_correct_person")
            else "✗"
            if "profile_correct_person" in row
            else "—"
        )
        lines.append(
            f"| {row['meeting']} | {row['speaker']} | {len(row['owned_clusters'])} | "
            f"spk_{row['best_cluster']} | {row['score']:.3f} | "
            f"{'✓' if row['correct_person'] else '✗'} | {profile} | {control} |"
        )
    lines += [
        "",
        "## Meeting to meeting — the condition the product runs in",
        "",
        "The identity is enrolled from one meeting and the voice has to be found in another.",
        "",
        f"- cases: {m2m['cases']}",
        f"- the right person was found: **{m2m['correct_person']}/{m2m['cases']}**",
        "",
        "| meeting | enrolled from | speaker | best | score | right person |",
        "| --- | --- | --- | ---: | ---: | :---: |",
    ]
    for row in m2m["details"]:
        lines.append(
            f"| {row['meeting']} | {row['enrolled_from']} | {row['speaker']} | "
            f"spk_{row['best_cluster']} | {row['score']:.3f} | "
            f"{'✓' if row['correct_person'] else '✗'} |"
        )
    lines += [
        "",
        "## Identification inside a registry of known people",
        "",
        "Every voice in a meeting, matched against every identity learnt from the other meetings.",
        "",
        f"- cases: {registry['cases']}",
        f"- the right identity was chosen: **{registry['correct_person']}/{registry['cases']}**",
        "",
        "A decision needs the score to be above the threshold **and** clearly "
        "better than the runner-up. The margin is what catches two people whose "
        "voices are alike.",
        "",
        "| threshold | margin | accepted | right | precision | recall |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for entry in reg_sweep:
        precision = entry["precision"]
        lines.append(
            f"| {entry['threshold']:.2f} | {entry['margin']:.2f} | {entry['accepted']} | "
            f"{entry['correct']} | {'—' if precision is None else f'{precision:.3f}'} | "
            f"{entry['recall']:.3f} |"
        )
    lines.append("")
    lines += [
        "",
        "## Where the threshold actually lands",
        "",
        'If the engine says "known" and is wrong, nobody notices. These are the '
        "numbers that decide the shipped threshold.",
        "",
        "| threshold | clean clip: accepted | right | precision | | "
        "meeting: accepted | right | precision |",
        "| ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |",
    ]
    for entry, entry_m2m in zip(report["threshold_in_meeting"], m2m_sweep, strict=True):
        precision = entry["precision"]
        precision_m2m = entry_m2m["precision"]
        lines.append(
            f"| {entry['threshold']:.2f} | {entry['accepted']} | {entry['correct']} | "
            f"{'—' if precision is None else f'{precision:.3f}'} | | "
            f"{entry_m2m['accepted']} | {entry_m2m['correct']} | "
            f"{'—' if precision_m2m is None else f'{precision_m2m:.3f}'} |"
        )
    lines.append("")
    (out_dir / "meeting_diarization.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "meeting_diarization.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n" + "\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
