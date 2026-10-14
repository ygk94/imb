"""Reference pipeline: OFF score prediction (target-free denoising + LightGBM + per-patient smoothing).

Run from the project root:  .venv\\Scripts\\python.exe src\\pipeline_ref.py [--submit]
CV (5-fold GroupKFold on patient_id): RMSE 2.99 after smoothing (3.04 raw).
"""
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from sklearn.model_selection import GroupKFold

PARAMS = dict(n_estimators=3000, learning_rate=0.015, num_leaves=63, min_child_samples=50,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=1.0, verbose=-1)


def rater_effects(df, col, iters=15):
    """Multiplicative bias of each rater on `col`, estimated without the target.

    log(col) = patient trend (ridge-shrunk line in age) + rater effect, fitted by alternating means.
    """
    s = df[df[col] > 0].copy()
    s["lv"] = np.log(s[col])
    s["ac"] = s.age - s.groupby("patient_id").age.transform("mean")
    s["r"] = 0.0
    for _ in range(iters):
        z = s.lv - s.r
        g = s.assign(z=z).groupby("patient_id")
        mz, ma = g.z.transform("mean"), g.ac.transform("mean")
        cov = ((s.ac - ma) * (z - mz)).groupby(s.patient_id).transform("sum")
        var = ((s.ac - ma) ** 2).groupby(s.patient_id).transform("sum")
        slope = (cov / (var + 2.0)).where(g.z.transform("count") >= 3, 0)
        s["r"] = (s.lv - (mz + slope * (s.ac - ma))).groupby(s.rater_id).transform("mean")
    r = s.groupby("rater_id").r.first()
    return r - r.mean()


def per_patient_fit(df, col, deg=1):
    """Per-patient least-squares polynomial of `col` vs age, evaluated at every visit (+ slope)."""
    fit = np.full(len(df), np.nan)
    slope = np.full(len(df), np.nan)
    age, val = df.age.to_numpy(), df[col].to_numpy()
    for idx in df.groupby("patient_id").indices.values():
        a, v = age[idx], val[idx]
        ok = ~np.isnan(v)
        if ok.sum() == 0:
            continue
        a0 = a[ok].mean()
        d = min(deg, len(np.unique(a[ok])) - 1)
        if d >= 1:
            c = np.polyfit(a[ok] - a0, v[ok], d)
            fit[idx] = np.polyval(c, a - a0)
            slope[idx] = c[-2]
        else:
            fit[idx] = v[ok].mean()
    return fit, slope


def robust_log_fit(df, col, deg=1, c=0.25, ridge=(0.0, 2.0, 20.0)):
    """Per-patient Huber fit of log(col) vs age (measurement noise is multiplicative and skewed)."""
    out = np.full(len(df), np.nan)
    age, val = df.age.to_numpy(), df[col].to_numpy()
    for idx in df.groupby("patient_id").indices.values():
        a, v = age[idx], val[idx]
        ok = ~np.isnan(v) & (v > 0)
        if ok.sum() == 0:
            continue
        a0 = a[ok].mean()
        d = min(deg, ok.sum() - 1)
        A = np.vander(a[ok] - a0, d + 1, increasing=True)
        z = np.log(v[ok])
        w = np.ones(len(z))
        for _ in range(10):
            beta = np.linalg.solve(A.T @ (A * w[:, None]) + np.diag(ridge[: d + 1]) + 1e-9 * np.eye(d + 1),
                                   A.T @ (w * z))
            r = np.abs(z - A @ beta)
            w = np.where(r <= c, 1.0, c / r)
        out[idx] = np.exp(np.vander(a - a0, d + 1, increasing=True) @ beta)
    return out


def build_features(df):
    """All features are computed from X only (no target), so they are valid on X_test."""
    d = df.copy()
    # 1. Remove each rater's multiplicative bias (estimated on train+test X together).
    for col in ["off", "on"]:
        d[col + "_c"] = d[col] / np.exp(d.rater_id.map(rater_effects(d, col)).fillna(0))
    # 2. Convert ON into an OFF-equivalent: OFF and ON of the same visit share the day's fluctuation,
    #    so log(on/off) only depends on the intake time and on the patient's drug response.
    both = (d.off_c > 0) & (d.on_c > 0)
    tbin = pd.cut(d.time_since_intake_on.fillna(-0.5), [-1, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 2, 2.5, 3, 4, 10])
    log_ratio = np.log(d.on_c / d.off_c).where(both)
    d["lh"] = tbin.map(log_ratio.groupby(tbin, observed=False).median()).astype(float)
    resp = (log_ratio - d.lh).groupby(d.patient_id).agg(["sum", "count"])
    d["lRp"] = d.patient_id.map(resp["sum"] / (resp["count"] + 2.0)).fillna(0)
    d["Rp_n"] = d.patient_id.map(resp["count"]).fillna(0)
    d["off_eq"] = np.exp(np.log(d.on_c.clip(lower=0.5)) - d.lh - d.lRp)
    d["off_best"] = d.off_c.fillna(d.off_eq)
    d["log_best"] = np.log1p(d.off_best)
    # 3. Visit context.
    g = d.groupby("patient_id")
    d["dur"] = d.age - d.age_at_diagnosis
    d["age_c"] = d.age - g.age.transform("mean")
    d["k"] = g.cumcount()
    d["n"] = g.age.transform("size")
    d["med"] = d.on.notna().groupby(d.patient_id).cummax().astype(int)
    d["first_on_rel"] = d.age.where(d.on.notna()).groupby(d.patient_id).transform("min") - g.age.transform("min")
    d["frac_on"] = g.on.transform(lambda s: s.notna().mean())
    # 4. Per-patient summaries of the denoised measurements.
    g = d.groupby("patient_id")
    for col in ["off_c", "on_c", "off_best", "log_best"]:
        d[col + "_max"] = g[col].transform("max")
        d[col + "_q75"] = g[col].transform(lambda s: s.quantile(0.75))
        d[col + "_med"] = g[col].transform("median")
        d[col + "_mean"] = g[col].transform("mean")
        d[col + "_cnt"] = g[col].transform("count")
        d[col + "_fit"], d[col + "_slope"] = per_patient_fit(d, col)
    d["off_best_fit2"], _ = per_patient_fit(d, "off_best", deg=2)
    for col in ["off_best", "off_c"]:
        for deg in [1, 2]:
            d[f"{col}_rob{deg}"] = robust_log_fit(d, col, deg)
    for sh in [1, 2]:
        d[f"best_prev{sh}"] = g.off_best.shift(sh)
        d[f"best_next{sh}"] = g.off_best.shift(-sh)
    d["off_cummax"] = g.off_c.cummax()
    d["off_futmax"] = d.iloc[::-1].groupby("patient_id").off_c.cummax().iloc[::-1]
    d["off_first"] = g.off_c.transform("first")
    d["premed_off_mean"] = d.off_c.where(d.med == 0).groupby(d.patient_id).transform("mean")
    d = d.drop(columns=["patient_id"])
    for col in ["cohort", "gene", "rater_id"]:
        d[col] = d[col].astype("category")
    return d


LOG_GRID = np.linspace(-3, 1.5, 901)


def noise_log_densities(F, y):
    """Empirical log-density of log(measurement / true OFF), learned on training rows only.

    Three measurement types: 0 = OFF at the first visit, 1 = OFF later, 2 = ON converted to OFF-equivalent.
    """
    mtype, meas = measurement_type(F)
    t = y.to_numpy()
    dens = {}
    for ty in [0, 1, 2]:
        sel = (mtype == ty) & (t > 15) & (meas > 0)
        e = np.log(meas[sel] / t[sel])
        e = e[(e > LOG_GRID[0]) & (e < LOG_GRID[-1])]
        dens[ty] = np.log(np.clip(gaussian_kde(e, bw_method=0.08)(LOG_GRID), 1e-6, None))
    return dens


def measurement_type(F):
    mtype = np.where(F.off_c.notna(), np.where(F.k == 0, 0, 1), np.where(F.off_eq.notna(), 2, -1))
    meas = F.off_c.fillna(F.off_eq).to_numpy(dtype=float)
    return mtype, meas


def likelihood_features(F, patient_id, dens):
    """Per-patient posterior mean of a log-linear OFF trajectory under the empirical (skewed) noise."""
    mtype, meas = measurement_type(F)
    ac = (F.age - F.age.groupby(patient_id.to_numpy()).transform("mean")).to_numpy()
    est = np.full(len(F), np.nan)
    slope = np.full(len(F), np.nan)
    for idx in pd.Series(np.arange(len(F))).groupby(patient_id.to_numpy()).indices.values():
        m = (mtype[idx] >= 0) & (meas[idx] > 0)
        if m.sum() == 0:
            continue
        z, a, ty = np.log(meas[idx][m]), ac[idx][m], mtype[idx][m]
        b0, b1 = np.meshgrid(np.linspace(np.median(z) - 1.2, np.median(z) + 1.5, 121),
                             np.linspace(-0.25, 0.35, 61), indexing="ij")
        ll = -0.5 * ((b1 - 0.07) / 0.08) ** 2  # weak prior: about +7 %/year
        for j in range(len(z)):
            ll = ll + np.interp(z[j] - b0 - b1 * a[j], LOG_GRID, dens[ty[j]])
        w = np.exp(ll - ll.max())
        w /= w.sum()
        est[idx] = [(w * np.exp(b0 + b1 * aa)).sum() for aa in ac[idx]]
        slope[idx] = b1.ravel()[np.argmax(ll)]
    return F.assign(mle=est, mle_slope=slope)


def smooth_per_patient(df, pred, deg=2):
    """The true OFF is a smooth curve per patient: refit the predictions with a quadratic in age."""
    out = np.empty(len(df))
    age = df.age.to_numpy()
    for idx in df.groupby("patient_id").indices.values():
        a = age[idx] - age[idx].mean()
        d = min(deg, len(np.unique(a)) - 1)
        out[idx] = np.polyval(np.polyfit(a, pred[idx], d), a) if d >= 1 else pred[idx].mean()
    return out


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def main(submit=False):
    X = pd.read_csv("X_train.csv", index_col="Index")
    y = pd.read_csv("y_train.csv", index_col="Index").target.loc[X.index]
    X_test = pd.read_csv("X_test.csv", index_col="Index")
    # Rows must stay sorted by patient then age (features use shift/cummax).
    full = pd.concat([X, X_test]).sort_values(["patient_id", "age"], kind="stable")
    feats = build_features(full)
    F, F_test = feats.loc[X.index], feats.loc[X_test.index]

    oof = np.zeros(len(X))
    for tr, va in GroupKFold(n_splits=5).split(F, y, groups=X.patient_id):
        # the noise density uses the target: learn it on the training patients of the fold only
        Fk = likelihood_features(F, X.patient_id, noise_log_densities(F.iloc[tr], y.iloc[tr]))
        model = lgb.LGBMRegressor(random_state=1, **PARAMS).fit(Fk.iloc[tr], y.iloc[tr])
        oof[va] = model.predict(Fk.iloc[va])
    print(f"CV RMSE raw      : {rmse(oof, y):.4f}")
    print(f"CV RMSE smoothed : {rmse(smooth_per_patient(X, oof), y):.4f}")

    if submit:
        dens = noise_log_densities(F, y)
        Fa = likelihood_features(F, X.patient_id, dens)
        Ft = likelihood_features(F_test, X_test.patient_id, dens)
        model = lgb.LGBMRegressor(random_state=1, **PARAMS).fit(Fa, y)
        pred = smooth_per_patient(X_test, model.predict(Ft))
        sub = pd.DataFrame({"Index": X_test.index, "target": np.clip(pred, 0, None)})
        path = "submissions/lgbm_denoise.csv"
        sub.to_csv(path, index=False)
        print(path, "written:", sub.shape)


if __name__ == "__main__":
    main(submit="--submit" in sys.argv)
