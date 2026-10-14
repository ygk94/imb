# Résultats des expériences

| Date | Idée | RMSE CV |
|---|---|---|
| 2026-10-14 | HistGradientBoostingRegressor + OrdinalEncoder, 5-fold GroupKFold | 7.5495 ± 0.1454 |
| 2026-10-14 | + patient history features (prev/next, leave-one-out mean, median, counts, visit rank, age since first) | 4.6417 ± 0.0935 |
| 2026-10-14 | src/pipeline_ref.py: rater correction (log, iterative) + ON→OFF-equivalent + per-patient robust fits + likelihood feature + LightGBM + per-patient quadratic smoothing | **2.9875 ± 0.063** |
