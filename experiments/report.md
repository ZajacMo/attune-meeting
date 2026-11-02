# Voiceprint calibration on real Chinese meetings

Source: REAL-PS4 / AISHELL-4 (real recorded Mandarin meetings)
Enrolment clips: 126 · speakers: 14 · sessions: 7

## 1. Voiceprint stability

| population | n | min | p05 | median | p95 | max | mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| same speaker, different session | 378 | 0.185 | 0.407 | 0.737 | 0.847 | 0.885 | 0.692 |
| same speaker, same session | 126 | 0.337 | 0.494 | 0.779 | 0.876 | 0.912 | 0.753 |
| different speakers | 7371 | -0.040 | 0.147 | 0.326 | 0.520 | 0.848 | 0.329 |

- best separating threshold: **0.6179** (accuracy 0.9794)
- equal error rate at threshold **0.5313** (EER 0.1026)

### Threshold sweep

| threshold | recall (same speaker) | false accept (other speaker) |
| ---: | ---: | ---: |
| 0.45 | 0.907 | 0.139 |
| 0.50 | 0.855 | 0.066 |
| 0.53 | 0.836 | 0.042 |
| 0.55 | 0.815 | 0.033 |
| 0.58 | 0.796 | 0.019 |
| 0.60 | 0.788 | 0.013 |
| 0.62 | 0.765 | 0.010 |
| 0.65 | 0.709 | 0.007 |
| 0.67 | 0.669 | 0.007 |
| 0.68 | 0.638 | 0.006 |
| 0.70 | 0.593 | 0.005 |
| 0.72 | 0.548 | 0.005 |
| 0.75 | 0.458 | 0.004 |
| 0.78 | 0.328 | 0.002 |
| 0.80 | 0.259 | 0.001 |
| 0.82 | 0.148 | 0.001 |
| 0.85 | 0.037 | 0.000 |

### Hardest impostor pairs (different people scoring highest)

| score | speaker A | session A | speaker B | session B |
| ---: | --- | --- | --- | --- |
| 0.848 | 002-F | R001S02C01 | 003-F | R001S02C01 |
| 0.835 | 002-F | R001S03C01 | 003-F | R001S02C01 |
| 0.828 | 002-F | R001S03C01 | 003-F | R001S02C01 |
| 0.825 | 002-F | R001S01C01 | 003-F | R001S02C01 |
| 0.814 | 002-F | R001S02C01 | 003-F | R001S01C01 |
| 0.814 | 002-F | R001S03C01 | 003-F | R001S02C01 |
| 0.809 | 005-F | R001S03C01 | 006-F | R001S02C01 |
| 0.808 | 002-F | R001S01C01 | 003-F | R001S02C01 |
| 0.805 | 001-M | R001S01C01 | 006-M | R001S02C01 |
| 0.799 | 002-F | R001S02C01 | 003-F | R001S01C01 |
| 0.797 | 002-F | R001S02C01 | 003-F | R001S03C01 |
| 0.783 | 001-M | R001S01C01 | 006-M | R001S02C01 |
| 0.775 | 005-F | R001S03C01 | 006-F | R001S02C01 |
| 0.774 | 002-F | R001S03C01 | 003-F | R001S02C01 |
| 0.772 | 002-F | R001S03C01 | 003-F | R001S02C01 |

### Hardest genuine pairs (same person scoring lowest)

| score | speaker | session A | session B |
| ---: | --- | --- | --- |
| 0.185 | 003-F | R001S01C01 | R001S02C01 |
| 0.241 | 001-M | R001S01C01 | R001S03C01 |
| 0.244 | 003-F | R001S02C01 | R001S03C01 |
| 0.283 | 003-F | R001S01C01 | R001S02C01 |
| 0.290 | 001-M | R001S01C01 | R001S03C01 |
| 0.294 | 003-F | R001S01C01 | R001S02C01 |
| 0.337 | 003-F | R001S02C01 | R001S02C01 |
| 0.343 | 003-F | R001S02C01 | R001S03C01 |
| 0.343 | 001-M | R001S01C01 | R001S02C01 |
| 0.344 | 005-F | R001S02C01 | R001S03C01 |
| 0.357 | 003-F | R001S02C01 | R001S03C01 |
| 0.373 | 001-M | R001S01C01 | R001S03C01 |
| 0.379 | 005-F | R001S02C01 | R001S03C01 |
| 0.381 | 003-F | R001S02C01 | R001S03C01 |
| 0.391 | 003-F | R001S02C01 | R001S03C01 |

## 2. Scoring rule vs number of enrolments

Attune stores several voiceprints per identity. `centroid` compares against their average, `max` against the best of them. Only `centroid` keeps an impostor's score from drifting up as identities accumulate.

### rule: centroid

| k enrolments | same speaker median | impostor median | impostor p95 | impostor max |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.731 | 0.312 | 0.511 | 0.814 |
| 2 | 0.786 | 0.339 | 0.535 | 0.863 |
| 3 | 0.804 | 0.351 | 0.552 | 0.869 |

### rule: max

| k enrolments | same speaker median | impostor median | impostor p95 | impostor max |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.731 | 0.312 | 0.511 | 0.814 |
| 2 | 0.769 | 0.344 | 0.539 | 0.835 |
| 3 | 0.786 | 0.358 | 0.555 | 0.835 |

## 3. Cross-meeting recognition

- meetings processed: 5
- cluster agreement (enrolled in another meeting → right person): **1.0**
- same-session reference agreement (ceiling): **1.0** over 5 cases

| population | n | min | p05 | median | p95 | max | mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| enrolled speaker vs its true cluster | 5 | 0.604 | 0.604 | 0.711 | 0.794 | 0.794 | 0.696 |
| enrolled speaker vs best cluster | 5 | 0.604 | 0.604 | 0.711 | 0.794 | 0.794 | 0.696 |
| diarization purity vs text ground truth | 5 | 0.833 | 0.833 | 0.833 | 1.000 | 1.000 | 0.900 |

| meeting | speaker | clusters | true | enrolled best | score | same-session ref | same? |
| --- | --- | ---: | ---: | ---: | ---: | ---: | :---: |
| AISHELL-4_20200616_M_R001S07C0 | 001-M | 3 | spk_0 | spk_0 | 0.604 | ✓ | ✓ |
| AISHELL-4_20200616_M_R001S07C0 | 002-F | 3 | spk_1 | spk_1 | 0.746 | ✓ | ✓ |
| AISHELL-4_20200616_M_R001S07C0 | 004-F | 3 | spk_2 | spk_2 | 0.626 | ✓ | ✓ |
| AISHELL-4_20200616_M_R001S08C0 | 002-F | 2 | spk_0 | spk_0 | 0.794 | ✓ | ✓ |
| AISHELL-4_20200616_M_R001S08C0 | 004-F | 2 | spk_1 | spk_1 | 0.711 | ✓ | ✓ |
