# Active Learning Report

Generated: 2026-03-26T22:43:41.109817

## Strategy: entropy

|   iteration |   n_labeled |   accuracy |     f1 | strategy   |
|------------:|------------:|-----------:|-------:|:-----------|
|           0 |          50 |     0.5    | 0.4993 | entropy    |
|           1 |          70 |     0.4636 | 0.4635 | entropy    |
|           2 |          90 |     0.4864 | 0.4851 | entropy    |
|           3 |         110 |     0.5136 | 0.5124 | entropy    |
|           4 |         130 |     0.5045 | 0.5033 | entropy    |
|           5 |         150 |     0.5227 | 0.5215 | entropy    |

**Final:** accuracy=0.5227, f1=0.5215

## Strategy: random

|   iteration |   n_labeled |   accuracy |     f1 | strategy   |
|------------:|------------:|-----------:|-------:|:-----------|
|           0 |          50 |     0.5    | 0.4993 | random     |
|           1 |          70 |     0.5409 | 0.5388 | random     |
|           2 |          90 |     0.5591 | 0.5523 | random     |
|           3 |         110 |     0.5136 | 0.51   | random     |
|           4 |         130 |     0.5227 | 0.5175 | random     |
|           5 |         150 |     0.5409 | 0.5358 | random     |

**Final:** accuracy=0.5409, f1=0.5358

## Comparison: Entropy vs Random

- Entropy final: accuracy=0.5227, f1=0.5215
- Random final:  accuracy=0.5409, f1=0.5358
- Δ accuracy: -0.0182
- Δ F1:       -0.0143
