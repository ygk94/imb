import pandas as pd

X = pd.read_csv("X_train.csv")
y = pd.read_csv("y_train.csv")
X_test = pd.read_csv("X_test.csv")

# ── Shapes ───────────────────────────────────────────────────────────────────
print("=== Shapes ===")
print(f"X_train : {X.shape}")
print(f"y_train : {y.shape}")

# ── Dtypes & NaN ─────────────────────────────────────────────────────────────
print("\n=== X_train — dtypes & NaN counts ===")
info = pd.DataFrame({
    "dtype": X.dtypes,
    "nan_count": X.isna().sum(),
    "nan_pct": (X.isna().mean() * 100).round(1),
})
print(info.to_string())

# ── Patient distribution ──────────────────────────────────────────────────────
print("\n=== Patient distribution ===")
visits = X.groupby("patient_id").size()
train_patients = set(X["patient_id"].unique())
test_patients  = set(X_test["patient_id"].unique())
overlap        = train_patients & test_patients
print(f"Unique train patients : {X['patient_id'].nunique()}")
print(f"Unique test patients  : {X_test['patient_id'].nunique()}")
print(f"Train/test overlap    : {len(overlap)} patient(s) — {sorted(overlap) if overlap else 'none'}")
print(f"Visits per patient — min: {visits.min()}, median: {visits.median()}, "
      f"mean: {visits.mean():.1f}, max: {visits.max()}")

# ── Target summary ────────────────────────────────────────────────────────────
print("\n=== y_train['target'] — summary statistics ===")
print(y["target"].describe().to_string())

# ── 'off' column vs target ────────────────────────────────────────────────────
print("\n=== 'off' column vs target ===")
merged = pd.concat([X["off"], y["target"]], axis=1)
corr = merged["off"].corr(merged["target"])
print(f"Pearson correlation (off ~ target) : {corr:.4f}")

print("\nMean target per decile of 'off' (NaN rows excluded):")
merged_clean = merged.dropna(subset=["off"])
merged_clean = merged_clean.copy()
merged_clean["off_decile"] = pd.cut(
    merged_clean["off"],
    bins=10,
    include_lowest=True,
)
decile_summary = (
    merged_clean.groupby("off_decile", observed=True)["target"]
    .agg(count="count", mean_target="mean")
    .reset_index()
)
print(decile_summary.to_string(index=False))
