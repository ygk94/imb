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

# ── Feature groups ────────────────────────────────────────────────────────────
CAT_COLS = ["cohort", "sexM", "gene", "rater_id"]
NUM_COLS = [c for c in X.columns if c not in CAT_COLS + ["patient_id"]]

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

print(f"Baseline RMSE: {mean_rmse:.4f} ± {std_rmse:.4f}")

# ── Save to local skore project ───────────────────────────────────────────────
project = Project("kaggle-off", mode="local")
project.put("baseline", report)
print("skore project updated: kaggle-off.skore")
