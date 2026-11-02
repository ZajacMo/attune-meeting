# Diarization vs overlapping speech

Engine: FunASR paraformer-zh + fsmn-vad + ct-punc + cam++ (CPU)
Windows tested: 40 · fully separated: 0.3

| speakers | overlap ≤ | windows | mean clusters | fully separated |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 0.2 | 3 | 2.0 | 3/3 |
| 2 | 0.4 | 3 | 3.67 | 2/3 |
| 2 | 0.6 | 3 | 3.0 | 1/3 |
| 2 | 0.8 | 3 | 2.0 | 1/3 |
| 2 | 1.0 | 3 | 2.33 | 1/3 |
| 3 | 0.2 | 3 | 4.33 | 2/3 |
| 3 | 0.4 | 3 | 2.0 | 1/3 |
| 3 | 0.6 | 3 | 1.67 | 0/3 |
| 3 | 0.8 | 3 | 2.0 | 1/3 |
| 3 | 1.0 | 3 | 1.0 | 0/3 |
| 4 | 0.4 | 3 | 6.0 | 0/3 |
| 4 | 0.6 | 3 | 2.67 | 0/3 |
| 4 | 0.8 | 3 | 2.33 | 0/3 |
| 4 | 1.0 | 1 | 2.0 | 0/1 |
