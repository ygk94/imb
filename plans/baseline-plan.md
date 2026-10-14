# Baseline Plan — OFF Score Prediction

## Overview

Build two scripts from scratch to establish a reproducible starting point for the Kaggle OFF score prediction competition.

- **src/explore.py** — data summary (no modelling, no plotting files)
- **src/baseline.py** — first ML pipeline with grouped cross-validation and skore report
- **src/RESULTS.md** — log the baseline RMSE

Validation split is always by `patient_id` (GroupKFold) because the leaderboard separates patients.
Python runtime: `.venv\Scripts\python.exe` only.
CSV files must never be read directly in this conversation — only through scripts.

---

## Sub-Task 1 — src/explore.py

**Status:** `[ ] pending`

### Intent
Print a structured data summary to the console so the data scientist understands the shape, missingness, feature types, patient distribution, and target. No files are written (no charts, no CSVs).

### Expected Outcomes
- Script runs without errors.
- Console output shows: shape, dtypes, NaN counts per column, number of unique patients, distribution of visits per patient (min/median/max), target (`y_train`) summary statistics.
- The relationship between the `off` column in X_train and the target is shown (correlation + a few quantile rows).

### Todo List
1. Load `X_train.csv` and `y_train.csv` using `pd.read_csv`.
2. Print shape of both DataFrames.
3. Print dtypes and NaN count per column for `X_train`.
4. Compute and print number of unique patients and visits-per-patient stats (min, median, max, mean) using `groupby("patient_id").size()`.
5. Print `y_train["target"].describe()` for target distribution.
6. Merge `X_train["off"]` with `y_train["target"]` on index, print `corr()` and a `pd.cut` quantile breakdown (mean target per decile of `off`).

### Relevant Context
- Data columns: `patient_id, cohort, sexM, gene, age_at_diagnosis, age, ledd, time_since_intake_on, time_since_intake_off, rater_id, on, off`. Many NaN.
- `patient_id` must never be used as a feature.
- CSV files: `X_train.csv`, `y_train.csv` at workspace root.

---

## Sub-Task 2 — src/baseline.py

**Status:** `[ ] pending`

### Intent
Train a first scikit-learn pipeline using `HistGradientBoostingRegressor` (handles NaN natively) with minimal categorical preprocessing, evaluate it with 5-fold `GroupKFold` on `patient_id`, save a skore `CrossValidationReport` in local mode, and print mean RMSE ± std to the console.

### Expected Outcomes
- Script runs without errors.
- Console prints mean RMSE ± std from 5-fold GroupKFold.
- A skore local project file `kaggle-off.skore` is created/updated in the workspace root.
- The `CrossValidationReport` is stored under the key `"baseline"` in the project.

### Todo List
1. Load `X_train.csv` and `y_train.csv`.
2. Identify categorical columns (`cohort`, `sexM`, `gene`, `rater_id`) and numeric columns (the rest, excluding `patient_id`).
3. Build a `ColumnTransformer`:
   - Categorical: `OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, encoded_missing_value=-2)`
   - Numeric: `passthrough` (HistGradientBoosting handles NaN natively)
4. Build a `Pipeline`: preprocessor → `HistGradientBoostingRegressor(random_state=0)`.
5. Precompute splits **before** dropping `patient_id`:
   `splits = list(GroupKFold(n_splits=5).split(X, y, groups=X["patient_id"]))`
6. Drop `patient_id` from `X` after computing splits.
7. Instantiate `CrossValidationReport(pipeline, X=X, y=y, splitter=splits)`.
8. Extract RMSE from `report.metrics.summarize()` — parse the RMSE column, compute mean and std across the 5 folds.
9. Print `f"Baseline RMSE: {mean:.4f} ± {std:.4f}"`.
10. Open skore project: `Project("kaggle-off", mode="local")` and `project.put("baseline", report)`.

### Relevant Context
- `HistGradientBoostingRegressor` is preferred: no imputation needed, handles NaN and mixed types.
- `CrossValidationReport` accepts precomputed splits as a list of `(train_idx, test_idx)` tuples via the `splitter` parameter — no `groups` kwarg needed.
- Splits must be computed while `patient_id` is still in `X`; drop it afterward so it is never seen by the estimator.
- skore `Project`: `Project(name, mode="local")` stores to a local `.skore` directory.
- `OrdinalEncoder` with `handle_unknown` and `encoded_missing_value` covers unknown categories and NaN in categoricals.

---

## Sub-Task 3 — Log results in src/RESULTS.md

**Status:** `[ ] pending`

### Intent
Record the baseline result so the experiment is traceable over time.

### Expected Outcomes
- `src/RESULTS.md` has one new row with today's date, model description, and RMSE CV value.

### Todo List
1. Run `src/baseline.py` and read the printed RMSE from output.
2. Append a row to the `| Date | Idée | RMSE CV |` table in `src/RESULTS.md`:
   - Date: today's date (YYYY-MM-DD)
   - Idée: `HistGradientBoostingRegressor + OrdinalEncoder, 5-fold GroupKFold`
   - RMSE CV: value from script output

### Relevant Context
- `src/RESULTS.md` already has the table header. Append one `| … | … | … |` row below it.

---

## Implementation Notes

- Sub-tasks must be executed in order (explore → baseline → log).
- After Sub-Task 2 confirm the skore project file exists before moving to Sub-Task 3.
- Split order matters: compute `GroupKFold` splits with `patient_id` present, then drop `patient_id` from `X` before passing to `CrossValidationReport`.
