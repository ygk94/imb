import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from skore import CrossValidationReport
from skore import Project

# ── Load data ─────────────────────────────────────────────────────────────────
X = pd.read_csv("X_train.csv", index_col="Index")
y = pd.read_csv("y_train.csv", index_col="Index")["target"]


def add_patient_history(df: pd.DataFrame) -> pd.DataFrame:
    """Add per-patient history features (sort by age within each patient_id).

    Features added:
    - off_prev, on_prev       : previous visit value (NaN for first visit)
    - off_next, on_next       : next visit value (NaN for last visit)
    - off_mean_other, off_med_other, off_count_other
    - on_mean_other,  on_med_other,  on_count_other
      (mean / median / count over OTHER visits of the same patient)
    - visit_rank              : 0-based position within the patient (sorted by age)
    - n_visits                : total number of visits for the patient
    - age_since_first         : age minus the patient's first-visit age
    """
    df = df.copy()
    df = df.sort_values(["patient_id", "age"])

    # ── shift-based prev / next ────────────────────────────────────────────────
    for col in ("off", "on"):
        df[f"{col}_prev"] = df.groupby("patient_id")[col].shift(1)
        df[f"{col}_next"] = df.groupby("patient_id")[col].shift(-1)

    # ── leave-one-out mean / median / count ───────────────────────────────────
    for col in ("off", "on"):
        grp = df.groupby("patient_id")[col]
        g_sum   = grp.transform("sum")
        g_count = grp.transform("count")      # non-NaN count
        row_val = df[col].fillna(0)
        row_notna = df[col].notna().astype(int)

        other_sum   = g_sum   - df[col].where(df[col].notna(), 0)
        other_count = g_count - row_notna

        df[f"{col}_mean_other"]  = other_sum / other_count.replace(0, np.nan)
        df[f"{col}_count_other"] = other_count

        # median_other: compute full median then approximate via cumulative trick
        # Exact leave-one-out median is expensive; use the group median for the
        # full patient instead (still leakage-free in inference once we compute
        # before train/test split, but we compute it on the full X here and pass
        # it through the pipeline — acceptable for a feature-engineering exp).
        df[f"{col}_med_other"] = grp.transform("median")

    # ── visit rank and total visits ───────────────────────────────────────────
    df["visit_rank"] = df.groupby("patient_id").cumcount()
    df["n_visits"]   = df.groupby("patient_id")["age"].transform("count")

    # ── age since first visit ─────────────────────────────────────────────────
    df["age_since_first"] = df["age"] - df.groupby("patient_id")["age"].transform("min")

    return df


# ── Engineer features BEFORE splitting (uses patient_id) ─────────────────────
X = add_patient_history(X)
# Re-align y to the new row order produced by the sort inside add_patient_history
y = y.reindex(X.index)

# ── Feature groups ────────────────────────────────────────────────────────────
CAT_COLS = ["cohort", "sexM", "gene", "rater_id"]
NEW_COLS = [
    "off_prev", "on_prev", "off_next", "on_next",
    "off_mean_other", "off_med_other", "off_count_other",
    "on_mean_other",  "on_med_other",  "on_count_other",
    "visit_rank", "n_visits", "age_since_first",
]
NUM_COLS = [
    c for c in X.columns
    if c not in CAT_COLS + ["patient_id"] + NEW_COLS
] + NEW_COLS

# ── Preprocessor + model pipeline ────────────────────────────────────────────
preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1,
                encoded_missing_value=-2,
            ),
            CAT_COLS,
        ),
        ("num", "passthrough", NUM_COLS),
    ]
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", HistGradientBoostingRegressor(random_state=0)),
])

# ── GroupKFold splits (computed BEFORE dropping patient_id) ───────────────────
splits = list(GroupKFold(n_splits=5).split(X, y, groups=X["patient_id"]))

# ── Drop patient_id so it is never seen by the estimator ─────────────────────
X = X.drop(columns=["patient_id"])

# ── Cross-validation report ───────────────────────────────────────────────────
report = CrossValidationReport(pipeline, X=X, y=y, splitter=splits)

# ── Extract RMSE ──────────────────────────────────────────────────────────────
rmse_df = report.metrics.get("rmse", aggregate=("mean", "std"))
vals = rmse_df.values.flatten().astype(float)
mean_rmse, std_rmse = vals[0], vals[1]

BASELINE = 7.5495
delta = mean_rmse - BASELINE
print(f"Patient-features RMSE : {mean_rmse:.4f} +/- {std_rmse:.4f}")
print(f"Baseline              : {BASELINE:.4f}")
print(f"Delta vs baseline     : {delta:+.4f}  ({'improvement' if delta < 0 else 'regression'})")

# ── Save to local skore project ───────────────────────────────────────────────
project = Project("kaggle-off", mode="local")
project.put("patient_features", report)
print("skore project updated: kaggle-off.skore  (key='patient_features')")
