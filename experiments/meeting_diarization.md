# Diarization on realistic-length meetings

Engine: FunASR paraformer-zh + fsmn-vad + ct-punc + cam++ (CPU)

| meeting | duration | segments | speakers | clusters | clusters labelled | correct |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| meeting_R002S04C01_00.wav | 110.1s | 64 | 2 | 5 | 4 | 4 |
| meeting_R002S01C01_01.wav | 100.9s | 59 | 2 | 3 | 3 | 3 |
| meeting_R002S08C01_02.wav | 106.6s | 49 | 2 | 2 | 2 | 2 |
| meeting_R002S06C01_03.wav | 101.1s | 56 | 2 | 6 | 5 | 5 |
| meeting_R002S02C01_04.wav | 86.4s | 46 | 2 | 2 | 2 | 2 |
| meeting_R002S03C01_05.wav | 69.8s | 48 | 2 | 2 | 2 | 2 |
| meeting_R002S01C01_06.wav | 94.9s | 55 | 3 | 3 | 1 | 1 |
| meeting_R001S03C01_07.wav | 92.0s | 46 | 3 | 3 | 3 | 3 |
| meeting_R002S08C01_08.wav | 76.8s | 37 | 3 | 3 | 2 | 2 |
| meeting_R002S01C01_09.wav | 94.0s | 47 | 3 | 5 | 5 | 5 |
| meeting_R001S03C01_10.wav | 78.8s | 36 | 3 | 3 | 3 | 3 |
| meeting_R002S03C01_11.wav | 66.6s | 48 | 3 | 3 | 3 | 3 |
| meeting_R002S01C01_12.wav | 94.2s | 56 | 4 | 3 | 3 | 3 |
| meeting_R002S08C01_13.wav | 95.2s | 60 | 4 | 4 | 4 | 4 |
| meeting_R002S08C01_14.wav | 63.9s | 34 | 4 | 3 | 3 | 3 |
| meeting_R001S03C01_15.wav | 70.6s | 40 | 4 | 3 | 3 | 3 |
| meeting_R002S08C01_16.wav | 26.6s | 13 | 4 | 4 | 4 | 4 |
| meeting_R002S08C01_17.wav | 52.7s | 25 | 4 | 3 | 3 | 3 |

## Cross-session recognition inside the meeting

A voiceprint enrolled in a *different* meeting is matched against every cluster of this meeting.

- cases: 38
- the right person was found: **19/38**
- the exact cluster was found: 14/38
- control (enrolled from the same meeting's clean audio): **35/38**
- cross-session with a full profile (centroid of every enrolment from other sessions): **20/38**

| meeting | speaker | clusters owned | best | score | right person | profile | same-meeting control |
| --- | --- | ---: | ---: | ---: | :---: | :---: | :---: |
| meeting_R002S04C01_00.wav | 002-F | 3 | spk_2 | 0.630 | ✗ | ✗ | ✓ |
| meeting_R002S04C01_00.wav | 003-F | 1 | spk_3 | 0.578 | ✗ | ✗ | ✓ |
| meeting_R002S01C01_01.wav | 002-F | 3 | spk_1 | 0.551 | ✓ | ✓ | ✓ |
| meeting_R002S08C01_02.wav | 003-F | 1 | spk_1 | 0.452 | ✗ | ✗ | ✓ |
| meeting_R002S08C01_02.wav | 004-F | 1 | spk_1 | 0.443 | ✓ | ✓ | ✓ |
| meeting_R002S06C01_03.wav | 003-M | 3 | spk_3 | 0.745 | ✗ | ✗ | ✓ |
| meeting_R002S06C01_03.wav | 004-M | 2 | spk_4 | 0.444 | ✗ | ✗ | ✓ |
| meeting_R002S02C01_04.wav | 002-F | 1 | spk_1 | 0.497 | ✓ | ✓ | ✓ |
| meeting_R002S02C01_04.wav | 003-F | 1 | spk_1 | 0.634 | ✗ | ✗ | ✓ |
| meeting_R002S03C01_05.wav | 002-F | 2 | spk_1 | 0.607 | ✓ | ✓ | ✓ |
| meeting_R002S01C01_06.wav | 003-F | 1 | spk_0 | 0.396 | ✓ | ✓ | ✗ |
| meeting_R001S03C01_07.wav | 003-F | 1 | spk_1 | 0.500 | ✓ | ✓ | ✓ |
| meeting_R001S03C01_07.wav | 004-M | 1 | spk_0 | 0.467 | ✓ | ✓ | ✓ |
| meeting_R001S03C01_07.wav | 006-M | 1 | spk_2 | 0.806 | ✓ | ✓ | ✗ |
| meeting_R002S08C01_08.wav | 004-M | 2 | spk_2 | 0.382 | ✓ | ✓ | ✓ |
| meeting_R002S01C01_09.wav | 002-F | 2 | spk_4 | 0.683 | ✓ | ✓ | ✓ |
| meeting_R002S01C01_09.wav | 003-F | 2 | spk_3 | 0.452 | ✗ | ✗ | ✓ |
| meeting_R002S01C01_09.wav | 004-M | 1 | spk_2 | 0.363 | ✗ | ✓ | ✓ |
| meeting_R001S03C01_10.wav | 001-F | 2 | spk_0 | 0.576 | ✓ | ✓ | ✓ |
| meeting_R001S03C01_10.wav | 003-F | 1 | spk_2 | 0.558 | ✗ | ✗ | ✓ |
| meeting_R002S03C01_11.wav | 002-F | 2 | spk_1 | 0.612 | ✓ | ✓ | ✓ |
| meeting_R002S03C01_11.wav | 003-M | 1 | spk_0 | 0.699 | ✗ | ✗ | ✓ |
| meeting_R002S01C01_12.wav | 001-F | 2 | spk_1 | 0.512 | ✓ | ✓ | ✓ |
| meeting_R002S01C01_12.wav | 003-F | 1 | spk_0 | 0.396 | ✗ | ✗ | ✗ |
| meeting_R002S08C01_13.wav | 001-F | 1 | spk_2 | 0.486 | ✗ | ✗ | ✓ |
| meeting_R002S08C01_13.wav | 002-F | 1 | spk_0 | 0.807 | ✗ | ✗ | ✓ |
| meeting_R002S08C01_13.wav | 003-F | 1 | spk_2 | 0.511 | ✗ | ✗ | ✓ |
| meeting_R002S08C01_13.wav | 004-M | 1 | spk_3 | 0.325 | ✓ | ✓ | ✓ |
| meeting_R002S08C01_14.wav | 001-F | 1 | spk_2 | 0.478 | ✗ | ✗ | ✓ |
| meeting_R002S08C01_14.wav | 004-M | 1 | spk_2 | 0.382 | ✓ | ✓ | ✓ |
| meeting_R002S08C01_14.wav | 005-M | 1 | spk_2 | 0.499 | ✗ | ✓ | ✓ |
| meeting_R001S03C01_15.wav | 001-F | 1 | spk_0 | 0.549 | ✗ | ✗ | ✓ |
| meeting_R001S03C01_15.wav | 003-F | 2 | spk_0 | 0.576 | ✓ | ✗ | ✓ |
| meeting_R002S08C01_16.wav | 001-F | 2 | spk_2 | 0.355 | ✓ | ✓ | ✓ |
| meeting_R002S08C01_16.wav | 002-M | 1 | spk_1 | 0.426 | ✗ | ✗ | ✓ |
| meeting_R002S08C01_16.wav | 005-M | 1 | spk_1 | 0.427 | ✓ | ✓ | ✓ |
| meeting_R002S08C01_17.wav | 001-F | 1 | spk_1 | 0.474 | ✗ | ✗ | ✓ |
| meeting_R002S08C01_17.wav | 002-M | 2 | spk_1 | 0.652 | ✓ | ✓ | ✓ |

## Meeting to meeting — the condition the product runs in

The identity is enrolled from one meeting and the voice has to be found in another.

- cases: 36
- the right person was found: **25/36**

| meeting | enrolled from | speaker | best | score | right person |
| --- | --- | --- | ---: | ---: | :---: |
| meeting_R002S04C01_00.wav | meeting_R002S01C01_01.wav | 002-F | spk_3 | 0.801 | ✓ |
| meeting_R002S04C01_00.wav | meeting_R002S08C01_02.wav | 003-F | spk_1 | 0.857 | ✗ |
| meeting_R002S01C01_01.wav | meeting_R002S04C01_00.wav | 002-F | spk_1 | 0.841 | ✓ |
| meeting_R002S08C01_02.wav | meeting_R002S04C01_00.wav | 003-F | spk_1 | 0.576 | ✗ |
| meeting_R002S06C01_03.wav | meeting_R002S03C01_11.wav | 003-M | spk_5 | 0.808 | ✓ |
| meeting_R002S06C01_03.wav | meeting_R001S03C01_07.wav | 004-M | spk_4 | 0.816 | ✗ |
| meeting_R002S02C01_04.wav | meeting_R002S04C01_00.wav | 002-F | spk_0 | 0.828 | ✗ |
| meeting_R002S02C01_04.wav | meeting_R002S04C01_00.wav | 003-F | spk_0 | 0.404 | ✓ |
| meeting_R002S03C01_05.wav | meeting_R002S04C01_00.wav | 002-F | spk_1 | 0.701 | ✓ |
| meeting_R002S01C01_06.wav | meeting_R002S04C01_00.wav | 003-F | spk_1 | 0.788 | ✗ |
| meeting_R001S03C01_07.wav | meeting_R002S04C01_00.wav | 003-F | spk_1 | 0.377 | ✓ |
| meeting_R001S03C01_07.wav | meeting_R002S06C01_03.wav | 004-M | spk_0 | 0.772 | ✓ |
| meeting_R002S08C01_08.wav | meeting_R002S06C01_03.wav | 004-M | spk_2 | 0.747 | ✓ |
| meeting_R002S01C01_09.wav | meeting_R002S04C01_00.wav | 002-F | spk_3 | 0.840 | ✓ |
| meeting_R002S01C01_09.wav | meeting_R002S04C01_00.wav | 003-F | spk_2 | 0.802 | ✓ |
| meeting_R002S01C01_09.wav | meeting_R002S06C01_03.wav | 004-M | spk_0 | 0.722 | ✓ |
| meeting_R001S03C01_10.wav | meeting_R002S01C01_12.wav | 001-F | spk_2 | 0.781 | ✓ |
| meeting_R001S03C01_10.wav | meeting_R002S04C01_00.wav | 003-F | spk_1 | 0.453 | ✓ |
| meeting_R002S03C01_11.wav | meeting_R002S04C01_00.wav | 002-F | spk_1 | 0.684 | ✓ |
| meeting_R002S03C01_11.wav | meeting_R002S06C01_03.wav | 003-M | spk_0 | 0.857 | ✗ |
| meeting_R002S01C01_12.wav | meeting_R001S03C01_10.wav | 001-F | spk_0 | 0.680 | ✓ |
| meeting_R002S01C01_12.wav | meeting_R002S04C01_00.wav | 003-F | spk_1 | 0.788 | ✗ |
| meeting_R002S08C01_13.wav | meeting_R001S03C01_10.wav | 001-F | spk_2 | 0.871 | ✗ |
| meeting_R002S08C01_13.wav | meeting_R002S04C01_00.wav | 002-F | spk_2 | 0.830 | ✓ |
| meeting_R002S08C01_13.wav | meeting_R002S04C01_00.wav | 003-F | spk_1 | 0.831 | ✓ |
| meeting_R002S08C01_13.wav | meeting_R002S06C01_03.wav | 004-M | spk_3 | 0.716 | ✓ |
| meeting_R002S08C01_14.wav | meeting_R001S03C01_10.wav | 001-F | spk_0 | 0.832 | ✓ |
| meeting_R002S08C01_14.wav | meeting_R002S06C01_03.wav | 004-M | spk_2 | 0.747 | ✓ |
| meeting_R002S08C01_14.wav | meeting_R002S08C01_16.wav | 005-M | spk_1 | 0.882 | ✓ |
| meeting_R001S03C01_15.wav | meeting_R001S03C01_10.wav | 001-F | spk_0 | 0.886 | ✗ |
| meeting_R001S03C01_15.wav | meeting_R002S04C01_00.wav | 003-F | spk_0 | 0.457 | ✓ |
| meeting_R002S08C01_16.wav | meeting_R001S03C01_10.wav | 001-F | spk_2 | 0.597 | ✓ |
| meeting_R002S08C01_16.wav | meeting_R002S08C01_17.wav | 002-M | spk_1 | 0.506 | ✗ |
| meeting_R002S08C01_16.wav | meeting_R002S08C01_14.wav | 005-M | spk_1 | 0.882 | ✓ |
| meeting_R002S08C01_17.wav | meeting_R001S03C01_10.wav | 001-F | spk_1 | 0.589 | ✗ |
| meeting_R002S08C01_17.wav | meeting_R002S08C01_16.wav | 002-M | spk_1 | 0.426 | ✓ |

## Identification inside a registry of known people

Every voice in a meeting, matched against every identity learnt from the other meetings.

- cases: 55
- the right identity was chosen: **30/55**

A decision needs the score to be above the threshold **and** clearly better than the runner-up. The margin is what catches two people whose voices are alike.

| threshold | margin | accepted | right | precision | recall |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.45 | 0.00 | 55 | 30 | 0.545 | 0.545 |
| 0.45 | 0.02 | 45 | 26 | 0.578 | 0.473 |
| 0.45 | 0.05 | 38 | 23 | 0.605 | 0.418 |
| 0.45 | 0.08 | 27 | 14 | 0.518 | 0.255 |
| 0.45 | 0.12 | 9 | 5 | 0.556 | 0.091 |
| 0.50 | 0.00 | 55 | 30 | 0.545 | 0.545 |
| 0.50 | 0.02 | 45 | 26 | 0.578 | 0.473 |
| 0.50 | 0.05 | 38 | 23 | 0.605 | 0.418 |
| 0.50 | 0.08 | 27 | 14 | 0.518 | 0.255 |
| 0.50 | 0.12 | 9 | 5 | 0.556 | 0.091 |
| 0.53 | 0.00 | 55 | 30 | 0.545 | 0.545 |
| 0.53 | 0.02 | 45 | 26 | 0.578 | 0.473 |
| 0.53 | 0.05 | 38 | 23 | 0.605 | 0.418 |
| 0.53 | 0.08 | 27 | 14 | 0.518 | 0.255 |
| 0.53 | 0.12 | 9 | 5 | 0.556 | 0.091 |
| 0.55 | 0.00 | 54 | 30 | 0.556 | 0.545 |
| 0.55 | 0.02 | 44 | 26 | 0.591 | 0.473 |
| 0.55 | 0.05 | 38 | 23 | 0.605 | 0.418 |
| 0.55 | 0.08 | 27 | 14 | 0.518 | 0.255 |
| 0.55 | 0.12 | 9 | 5 | 0.556 | 0.091 |
| 0.58 | 0.00 | 54 | 30 | 0.556 | 0.545 |
| 0.58 | 0.02 | 44 | 26 | 0.591 | 0.473 |
| 0.58 | 0.05 | 38 | 23 | 0.605 | 0.418 |
| 0.58 | 0.08 | 27 | 14 | 0.518 | 0.255 |
| 0.58 | 0.12 | 9 | 5 | 0.556 | 0.091 |
| 0.60 | 0.00 | 54 | 30 | 0.556 | 0.545 |
| 0.60 | 0.02 | 44 | 26 | 0.591 | 0.473 |
| 0.60 | 0.05 | 38 | 23 | 0.605 | 0.418 |
| 0.60 | 0.08 | 27 | 14 | 0.518 | 0.255 |
| 0.60 | 0.12 | 9 | 5 | 0.556 | 0.091 |
| 0.62 | 0.00 | 54 | 30 | 0.556 | 0.545 |
| 0.62 | 0.02 | 44 | 26 | 0.591 | 0.473 |
| 0.62 | 0.05 | 38 | 23 | 0.605 | 0.418 |
| 0.62 | 0.08 | 27 | 14 | 0.518 | 0.255 |
| 0.62 | 0.12 | 9 | 5 | 0.556 | 0.091 |
| 0.65 | 0.00 | 52 | 28 | 0.538 | 0.509 |
| 0.65 | 0.02 | 42 | 24 | 0.571 | 0.436 |
| 0.65 | 0.05 | 36 | 21 | 0.583 | 0.382 |
| 0.65 | 0.08 | 26 | 13 | 0.500 | 0.236 |
| 0.65 | 0.12 | 9 | 5 | 0.556 | 0.091 |
| 0.68 | 0.00 | 50 | 26 | 0.520 | 0.473 |
| 0.68 | 0.02 | 40 | 22 | 0.550 | 0.400 |
| 0.68 | 0.05 | 34 | 19 | 0.559 | 0.345 |
| 0.68 | 0.08 | 25 | 12 | 0.480 | 0.218 |
| 0.68 | 0.12 | 9 | 5 | 0.556 | 0.091 |
| 0.70 | 0.00 | 48 | 24 | 0.500 | 0.436 |
| 0.70 | 0.02 | 38 | 20 | 0.526 | 0.364 |
| 0.70 | 0.05 | 33 | 18 | 0.545 | 0.327 |
| 0.70 | 0.08 | 24 | 11 | 0.458 | 0.200 |
| 0.70 | 0.12 | 8 | 4 | 0.500 | 0.073 |
| 0.72 | 0.00 | 46 | 23 | 0.500 | 0.418 |
| 0.72 | 0.02 | 36 | 19 | 0.528 | 0.345 |
| 0.72 | 0.05 | 32 | 17 | 0.531 | 0.309 |
| 0.72 | 0.08 | 23 | 10 | 0.435 | 0.182 |
| 0.72 | 0.12 | 8 | 4 | 0.500 | 0.073 |
| 0.75 | 0.00 | 41 | 20 | 0.488 | 0.364 |
| 0.75 | 0.02 | 32 | 16 | 0.500 | 0.291 |
| 0.75 | 0.05 | 29 | 15 | 0.517 | 0.273 |
| 0.75 | 0.08 | 22 | 10 | 0.455 | 0.182 |
| 0.75 | 0.12 | 7 | 4 | 0.571 | 0.073 |
| 0.80 | 0.00 | 30 | 17 | 0.567 | 0.309 |
| 0.80 | 0.02 | 26 | 15 | 0.577 | 0.273 |
| 0.80 | 0.05 | 24 | 14 | 0.583 | 0.255 |
| 0.80 | 0.08 | 20 | 10 | 0.500 | 0.182 |
| 0.80 | 0.12 | 7 | 4 | 0.571 | 0.073 |


## Where the threshold actually lands

If the engine says "known" and is wrong, nobody notices. These are the numbers that decide the shipped threshold.

| threshold | clean clip: accepted | right | precision | | meeting: accepted | right | precision |
| ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| 0.45 | 27 | 12 | 0.444 | | 33 | 22 | 0.667 |
| 0.50 | 19 | 10 | 0.526 | | 31 | 20 | 0.645 |
| 0.53 | 16 | 8 | 0.500 | | 30 | 20 | 0.667 |
| 0.55 | 15 | 8 | 0.533 | | 30 | 20 | 0.667 |
| 0.58 | 10 | 5 | 0.500 | | 29 | 20 | 0.690 |
| 0.60 | 10 | 5 | 0.500 | | 27 | 19 | 0.704 |
| 0.62 | 8 | 3 | 0.375 | | 27 | 19 | 0.704 |
| 0.65 | 6 | 3 | 0.500 | | 27 | 19 | 0.704 |
| 0.68 | 5 | 2 | 0.400 | | 27 | 19 | 0.704 |
| 0.70 | 3 | 1 | 0.333 | | 25 | 17 | 0.680 |
| 0.72 | 3 | 1 | 0.333 | | 23 | 15 | 0.652 |
| 0.75 | 2 | 1 | 0.500 | | 20 | 12 | 0.600 |
| 0.80 | 2 | 1 | 0.500 | | 16 | 10 | 0.625 |
