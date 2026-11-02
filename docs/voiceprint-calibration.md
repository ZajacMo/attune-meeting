# Voiceprint calibration

Attune's promise is that it remembers *who a voice is* across meetings. That
promise rests on two numbers: the similarity thresholds in
`src/attune/core/identity.py`, and the way an identity is scored against the
voiceprints stored for it. Both are measured here, on real Mandarin meeting
audio, and both are reproducible with the scripts in `experiments/`.

Nothing in this document is a guess. Where a number could not be measured, the
document says so — and where the measurement came out worse than expected,
that is written down too.

## The data

| | |
| --- | --- |
| Source | `tian0105/REAL-PS4` on Hugging Face — the AISHELL-4 portion |
| What it is | Real recorded Mandarin meetings, 8-channel array, the same people recurring across sessions |
| Enrolment clips | 126 single-speaker utterances, 14 speakers, 7 sessions |
| Mixtures | 8 windows of real meeting audio with genuinely overlapping speakers |
| Stitched meetings | 18 meetings of 27–110 seconds, built from mixture windows of one recorded session |
| Engine | FunASR `paraformer-zh` + `fsmn-vad` + `ct-punc` + `cam++`, CPU |
| Voiceprint | CAM++ 192-dimensional centroid (`spk_embedding_center`) |
| Ground truth | `overlap_records.json` — who was speaking during which part of each window |

The corpus is picked by `experiments/select_sample.py`; the exact files used are
recorded in `experiments/data/ps4/manifest.json`, which is committed.

**Read this before reading the numbers:** every AISHELL-4 mixture window
overlaps — the corpus has no window without overlap, and the overlap fraction
runs from 0.2 to 1.0. So the recognition numbers below are the *worst case*.
A real meeting is mostly one person talking at a time.

## Reproducing

```bash
uv venv
uv sync --extra dev --extra server
uv pip install "funasr>=1.4.0" torch torchaudio huggingface_hub

# 1. fetch the sample and record it in the manifest
python experiments/select_sample.py --sessions-per-speaker 3 --mixtures 8

# 2. run the ASR once and cache the voiceprints
python experiments/extract_voiceprints.py

# 3. the analyses below — all offline, seconds instead of minutes
python experiments/validate_identity.py
python experiments/diarization_overlap.py --per-bucket 3
python experiments/meeting_diarization.py --meetings 20 --windows-per-meeting 4

# 4. how many rows the cluster backend actually sees (slow: re-runs the ASR)
python experiments/chunk_budget.py
```

Step 2 caches into `experiments/data/ps4/voiceprints.json`, so steps 3 can be
re-run as often as needed while thresholds and rules are being changed.

## 1. How stable is a voiceprint?

Cosine similarity between CAM++ centroids, over every pair of the 126
enrolment clips.

| population | n | min | p05 | median | p95 | max | mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| same speaker, different session | 378 | 0.185 | 0.407 | 0.737 | 0.847 | 0.885 | 0.692 |
| same speaker, same session | 126 | 0.337 | 0.494 | 0.779 | 0.876 | 0.912 | 0.753 |
| different speakers | 7371 | -0.040 | 0.147 | 0.326 | 0.520 | 0.848 | 0.329 |

- best separating threshold: **0.6179** (accuracy 0.9794)
- equal error rate at **0.5313** (EER 0.1026)

The two distributions overlap: the same person across sessions can score 0.185,
and two different people can score 0.848. No threshold separates them
perfectly, and pretending otherwise is the mistake this document exists to
prevent.

### Where the difficulty actually sits

The highest-scoring impostor pairs are not spread evenly — they are one pair of
people, repeatedly:

| score | speaker A | speaker B |
| ---: | --- | --- |
| 0.848 | 002-F | 003-F |
| 0.835 | 002-F | 003-F |
| 0.828 | 002-F | 003-F |
| 0.825 | 002-F | 003-F |
| 0.814 | 002-F | 003-F |

Eleven of the fifteen hardest impostor pairs are 002-F against 003-F, scoring
between 0.772 and 0.848 against each other — above the threshold that
separates most other pairs. The next hardest pairs are 005-F against 006-F
(0.809) and 001-M against 006-M (0.805). A registry containing both 002-F and
003-F will need the user to disambiguate; no threshold can fix it, because the
two voices genuinely are that close.

The hardest genuine pairs are the mirror image: 003-F and 001-M vary the most
between sessions, scoring as low as 0.185 for the same person.

**This is why the shipped threshold is lower than the one a vendor would
quote.** A threshold of 0.82 looks precise because it rejects almost every
impostor — and that is true — but on this data it also accepts only 14.8% of
genuine pairs. It would be a threshold that says "I don't know" almost every
time.

## 2. How should an identity be scored?

An identity stores one voiceprint per meeting it appears in. The question is
which of them a new voiceprint is compared against.

Attune compares against the **centroid** — the average of the stored
voiceprints — rather than the best of them. Measured over profiles built from
1, 2 and 3 enrolments, scored against genuine and impostor clips from a
*different* session:

| rule | k enrolments | genuine median | impostor median | impostor p95 | impostor max |
| --- | ---: | ---: | ---: | ---: | ---: |
| centroid | 1 | 0.731 | 0.312 | 0.511 | 0.814 |
| centroid | 2 | 0.786 | 0.339 | 0.535 | 0.863 |
| centroid | 3 | 0.804 | 0.351 | 0.552 | 0.869 |
| best-of | 1 | 0.731 | 0.312 | 0.511 | 0.814 |
| best-of | 2 | 0.769 | 0.344 | 0.539 | 0.835 |
| best-of | 3 | 0.786 | 0.358 | 0.555 | 0.835 |

At equal false-accept rates the centroid rule also recognises more genuine
pairs:

| threshold | centroid: recall / false accept | best-of: recall / false accept |
| ---: | --- | --- |
| 0.62 | 0.841 / 0.0187 | 0.857 / 0.0165 |
| 0.65 | 0.833 / 0.0117 | 0.825 / 0.0123 |
| 0.68 | **0.794 / 0.0084** | 0.790 / 0.0095 |
| 0.70 | 0.758 / 0.0073 | 0.750 / 0.0092 |
| 0.75 | 0.671 / 0.0064 | 0.627 / 0.0070 |
| 0.82 | 0.429 / 0.0014 | 0.290 / 0.0014 |

(k = 3 enrolments.)

The reason is that taking the best of N voiceprints makes the score of an
*impostor* drift upwards as N grows — the more samples, the more likely one
of them happens to look like the stranger. The centroid does not have that
property: it converges on the person, and a stranger gets no benefit from it.
That is what makes one fixed threshold meaningful as a registry grows.

This is why `SpeakerRegistry._score_profile` compares against the centroid.

## 3. Can the engine tell speakers apart at all?

This is the part that decides whether the product can work, so it was measured
carefully.

### The rule that decides it — and what it counts

FunASR's cluster backend returns a single cluster whenever it is given fewer
than 20 rows:

```python
# funasr/models/campplus/cluster_backend.py
if X.shape[0] < 20:
    return np.zeros(X.shape[0], dtype="int")
```

The rows are **not** ASR sentences and **not** VAD segments. Before clustering,
every VAD segment is cut into 1.5-second windows with a 0.75-second hop
(`funasr/models/campplus/utils.py: sv_chunk`), and it is those voice-print
windows that get clustered. The limit is therefore a limit on *how much speech
there is*, not on how many people are talking.

Measured on every clip in the sample (`experiments/chunk_budget.py`), the rule
holds exactly:

| clip | voice-print windows | clusters returned |
| --- | ---: | ---: |
| the eight mixture windows | 12 – 25 | one speaker in six of them, 2 and 3 in the other two |
| the eighteen stitched meetings | 35 – 142 | 2 – 6 |

Every clip below 20 windows came back as one speaker. Two of the mixture
windows were above the guard and still came back as one speaker: with enough
speech to work with, the engine found no second voice to separate. Every
meeting was above the guard, and every meeting was separated.

Measured on 40 windows spread across the overlap spectrum
(`experiments/diarization_overlap.py`), only 30% were separated into the
documented number of speakers — so on short, heavily overlapping audio the
engine is not usable, and short clips cannot tell you why.

### Realistic-length meetings

To measure the engine rather than the guard, `experiments/meeting_diarization.py`
stitches several mixture windows from the *same* recorded session into a
meeting. Every piece of it is real meeting audio with real overlapping
speakers, and the metadata still says who said what.

Over 18 meetings:

| | |
| --- | ---: |
| meetings | 18 |
| clusters the engine produced | 60 |
| meetings with exactly the right number of clusters | 10 / 18 |

Ten meetings came out with exactly the right number of speakers. The rest
mostly over-segment: eight meetings split one person across several clusters
(2 speakers → 5 clusters, 3 speakers → 6 clusters, 4 speakers → 3 clusters).

### Where the engine does get it wrong: simultaneous speech

Over-segmentation is the safe direction — the identity layer can merge the
fragments once the user confirms. Merging two people into one cluster is not,
so it was measured directly.

Every turn was attributed to a person by **time**, using the overlap records
that state who was speaking during which part of each window
(`overlap_records.json`). The alternative — matching the ASR transcript to the
documented transcript — cannot be trusted here: when two people talk at once,
the ASR mangles both into one sentence.

| | |
| --- | ---: |
| turns attributable to one clear owner | 385 |
| clusters that could be judged | 58 |
| of those, clusters holding more than one person | **25** |
| meetings with at least one such cluster | 14 / 18 |
| mean cluster purity | 0.874 |

**So the engine does merge people: 25 of the 58 judgeable clusters (43%) hold
more than one.** The pairs that get merged most often are 002-F with 003-F
(four clusters) and 003-M with 004-M (four clusters).

Those are two different failures. 002-F and 003-F are the pair that scores
0.77–0.85 against each other, and the engine merges clusters whose voiceprints
are at least 0.78 similar — these two are above that line. 003-M and 004-M are
nothing alike; the only explanation left is that they were talking at the same
time, and a 1.5-second window holding two voices produces a voiceprint of
both.

Every measurement in this document is the worst case: **every** window in this
corpus overlaps. Real meetings are mostly one person at a time, and the
identity layer's rule — ask the user rather than guess — is what covers the
rest.

## 4. Can a voice enrolled in one meeting be found in another?

Four conditions, all on the same 18 stitched meetings. Every row was measured
twice: once with the cluster ownership taken from the transcript, once with the
ownership taken from the timeline. The second is the trustworthy one; the first
is kept because it is what the earlier numbers in this document were built on,
and the two agreeing is the evidence that neither is an artefact.

| condition | enrolled from | transcript | timeline |
| --- | --- | ---: | ---: |
| same meeting, clean audio | the clean enrolment clips that were mixed into this very meeting | 35 / 38 (92%) | **41 / 44 (93%)** |
| another session, one clip | a single enrolment clip of a different session | 19 / 38 (50%) | 21 / 45 (47%) |
| another session, full profile | the centroid of every enrolment clip from other sessions | 20 / 38 (53%) | 24 / 45 (53%) |
| another meeting | the voiceprint the registry learnt from another meeting | 25 / 36 (69%) | **34 / 45 (76%)** |

Read from the top:

- **Inside one meeting the identity layer works.** The voiceprint the registry
  holds finds the right person 93% of the time.
- **A registry built from clean clips is much weaker** — about half. Giving
  the identity more voiceprints barely helps (21 → 24).
- **A registry built from meetings is much stronger** (34/45, 76%). The cluster
  voiceprint averages over tens of seconds of speech, so it is a far more
  stable estimate of a person than any single short clip. This is the
  condition the product actually runs in, and it is the better of the two.

### The operation the product performs

Every voice in every meeting, matched against a registry holding one profile
per person, learnt from the other meetings:

| | transcript | timeline |
| --- | ---: | ---: |
| cases | 55 | 58 |
| the right identity chosen | 30 (55%) | **36 (62%)** |

And the number that decides the threshold — of the voices the engine would
silently accept as "known", how many are the right person? Measured on the
registry population, with the timeline ground truth:

| threshold | accepted as known | right | precision | recall |
| ---: | ---: | ---: | ---: | ---: |
| 0.55 | 57 | 36 | 0.632 | 0.621 |
| 0.60 | 57 | 36 | 0.632 | 0.621 |
| 0.65 | 56 | 36 | **0.643** | 0.621 |
| 0.68 | 54 | 34 | 0.630 | 0.586 |
| 0.70 | 53 | 33 | 0.623 | 0.569 |
| 0.75 | 49 | 30 | 0.612 | 0.517 |
| 0.80 | 36 | 21 | 0.583 | 0.362 |

The transcript-based measurement of the same thing has the same shape: 27
accepted / 19 right (0.704) at 0.68, falling to 16 / 10 (0.625) at 0.80.

**Raising the threshold does not fix the errors.** They are not low-scoring
wrong matches that a stricter bar would filter out; they are high-scoring
confusions between two people whose voices really are alike. On the
meeting-to-meeting population, raising the threshold from 0.68 to 0.80 removes
more correct matches than wrong ones and the precision goes *down*. On the
registry population it buys a little precision and costs a third of the
recall. Neither is a fix.

### What does help: a margin over the runner-up

A decision needs the score to be above the threshold **and** clearly better
than the next candidate. Measured on the registry above (transcript ground
truth):

| threshold | margin over runner-up | accepted | right | precision | recall |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.68 | 0.00 | 50 | 26 | 0.520 | 0.473 |
| 0.68 | 0.02 | 40 | 22 | 0.550 | 0.400 |
| 0.68 | 0.05 | 34 | 19 | **0.559** | 0.345 |
| 0.60 | 0.05 | 38 | 23 | **0.605** | 0.418 |

A margin requirement moves precision in the right direction where the
threshold alone cannot, because it targets exactly the case the threshold
cannot see: two identities whose scores are both high and close together.

**This is not implemented.** It changes the matching contract — a new rule, a
new field in `speakers.json`, and a decision for the caller — so it belongs to
whoever owns the product, not to a threshold calibration. It is the first
change this document would recommend.

## 5. The shipped thresholds

```python
DEFAULT_HIGH_THRESHOLD = 0.68
DEFAULT_LOW_THRESHOLD = 0.53
```

**High (0.68)** sits at the end of the plateau of the in-meeting precision
curve (0.630–0.643 across 0.55–0.68), and keeps the false-accept rate of the
clean-clip measurement at 0.84% against a profile holding three voiceprints —
roughly 1 in 120. The silent path is the only one that can be wrong without the
user noticing, so it gets the tightest budget.

**Low (0.53)** is the equal-error point of the two distributions. Below it,
"somebody I have never met" is the more likely reading, so Attune asks who the
new person is instead of offering a shortlist of people it half-remembers.

Between the two, Attune asks the user which of the known people this is. That
band is wide on purpose. Every answer adds a voiceprint to the identity, the
centroid sharpens, and the silent path starts working more often on its own —
at three enrolments the genuine-match median has already risen from 0.731 to
0.804.

## What these numbers do not tell you

- **Your microphones, your rooms.** This corpus is one recording array. A
  different microphone changes the voiceprint. Re-run the scripts on your own
  audio before trusting the defaults.
- **The thresholds were calibrated on clean single-speaker clips.** A voiceprint
  taken from a meeting is noisier, and the precision inside meetings is
  correspondingly lower (0.63 against a much higher figure on clean clips).
  The shipped values are the optimistic end of the range.
- **The recognition numbers are the worst case.** Every AISHELL-4 window
  overlaps. Real meetings are mostly one person at a time, and the same
  measurement on mostly-clean speech will do better. How much better could not
  be measured here.
- **The engine merges people who talk at the same time.** 43% of the judgeable
  clusters held more than one person. Nothing in the identity layer can undo
  that; the user has to be asked.
- **A registry of very similar voices is a known hard case.** 002-F and 003-F
  score 0.772–0.848 against each other. No threshold resolves that; the user
  has to, which is what the margin rule in the previous section is for.
- **Short audio cannot be diarized by this engine at all.** Below 20
  voice-print windows — roughly 15 seconds of speech — the engine reports one
  speaker.
- **The diarization engine was not compared against alternatives.** pyannote's
  diarization models are gated on Hugging Face and could not be downloaded in
  this environment, so the comparison against them has not been made.
