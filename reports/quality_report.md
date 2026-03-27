# Data Quality Report

Generated: 2026-03-26T22:26:01.789895

## Issues Detected

- **Rows:** 1100
- **Missing values:** {}
- **Duplicates:** {'count': 0, 'percent': 0.0}
- **Outliers:** 1 features with outliers
- **Imbalance:** ratio=0.9366

## Cleaning Strategy

```json
{
  "missing": "drop",
  "duplicates": "drop",
  "outliers": "clip_iqr"
}
```

## Before / After Comparison

| metric               |    before |     after |   delta |   delta_pct |
|:---------------------|----------:|----------:|--------:|------------:|
| rows                 | 1100      | 1100      |       0 |           0 |
| columns              |    4      |    4      |       0 |           0 |
| missing_values_total |    0      |    0      |       0 |           0 |
| duplicates           |    0      |    0      |       0 |           0 |
| outliers_total       |    9      |    0      |      -9 |        -100 |
| imbalance_ratio      |    0.9366 |    0.9366 |       0 |           0 |
