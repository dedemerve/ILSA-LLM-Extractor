#!/usr/bin/env python3
"""
Unified expanding-window (rolling-origin) backtest for country-level ILSA means.

One harness, one test set per fold, every model scored on exactly that set.

Forecast origin (Definition A): to forecast cycle t of a program-domain, the
forecaster stands at the public release of the previous cycle of that same
program-domain. Only information released by then is used:
  - own score history (cycles < t; all released by the origin)
  - own-program covariates, latest source cycle released by the origin
  - TALIS covariates for PISA only (TALIS-PISA are designed as linked
    lower-secondary surveys); no other cross-program features
Release dates are in RELEASE (year, month); they are external knowledge
(ASSUMPTION) and must be checked against official publications.

Test set per fold: (country, t) with a target and >= 1 prior own score.
Countries with no prior score are written to no_history.csv and never enter
the main metrics. A fold is "complete" only if every model produced a
prediction for every test row; incomplete folds are reported but excluded
from the summaries.

Models: Persistence, Drift, InternationalMean, AR1, LinearTrend,
Ridge_M0A (score-only), Ridge_M0 (+covariates), Ridge_M1 (+covariates,
sqrt(w) KB scaling), RF/XGB score-only and +covariates. TimesFM 2.5
zero-shot is joined from saved forecasts and compared on its own subset.

Covariate rules: KB v3 feature map (scripts/kb_weights.py); source cycles
listed in excluded_source_cycles (TALIS:2018 degenerate scales) are masked.

Outputs (refuses to overwrite): outputs/unified_backtest_v1/ by default.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

# numpy 2.x on macOS Accelerate emits spurious matmul RuntimeWarnings even for
# well-conditioned inputs; Ridge/OLS were checked against closed-form solutions
# (max diff < 1e-12). Predictions are asserted finite in main().
warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from kb_weights import KBWeights  # noqa: E402

ESTIMATES = ROOT / "outputs" / "stage4" / "country_estimates_full_v2.csv"
COVARIATES = ROOT / "outputs" / "stage5" / "covariate_estimates.csv"
TALIS = ROOT / "outputs" / "stage5" / "talis_covariate_estimates.csv"
PISA2000_XWALK = ROOT / "analysis" / "03_data_feasibility" / "audit" / "pisa_2000_reading_country_crosswalk.csv"
TIMESFM_FILES = {
    "PISA": ROOT / "outputs" / "pisa_timesfm25_ils_adapter_v1" / "all_models_historical_predictions.csv",
    "OTHER": ROOT / "outputs" / "ilsa_timesfm_all_programs_v1" / "predictions.csv",
}
DEFAULT_OUT = ROOT / "outputs" / "unified_backtest_v1"

SEED = 20261027
ALPHA_GRID = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
FALLBACK_ALPHA = 10.0
MIN_TRAIN = 10
PRIMARY_EXCLUDED_TEST_CYCLES = {2025}

MAIN = [("PISA", "mathematics"), ("PISA", "reading"), ("PISA", "science"),
        ("TIMSS_G8", "mathematics"), ("TIMSS_G8", "science"),
        ("TIMSS_G4", "mathematics"), ("TIMSS_G4", "science"), ("PIRLS", "reading")]
APPENDIX = [("ICCS", "civic_knowledge"), ("ICILS", "computer_literacy"),
            ("PIAAC", "literacy"), ("PIAAC", "numeracy")]

# Own-program covariates per target program (KB v3 variable names).
PROGRAM_COVARIATES = {
    "PISA": ["ESCS", "HOMEPOS", "BELONG", "ICTAVHOM", "ICTAVSCH",
             "SECLSS", "TCDISCS", "TEFFPROS", "TJSPROS"],
    "TIMSS_G8": ["BSDGEDUP"],
    "TIMSS_G4": ["ASDHEDUP"],
    "PIRLS": ["ASDHEDUP", "ASDHELA"],
    "ICCS": ["ICCS_ses_composite", "ICCS_parental_education"],
    "ICILS": ["ICILS_ses_composite", "ICILS_parental_education", "ICILS_home_literacy_activities"],
    "PIAAC": ["PIAAC_parental_education"],
}
# Covariates stored as "domains" in country_estimates.
ESTIMATE_COVARIATE_DOMAINS = {
    "ICCS": ["ses_composite", "parental_education"],
    "ICILS": ["ses_composite", "parental_education", "home_literacy_activities"],
    "PIAAC": ["parental_education"],
}
TALIS_NAME = {"SECLSS": "SECLSS", "T3SECLS": "SECLSS", "T4SECLS": "SECLSS",
              "TCDISCS": "TCDISCS", "T3DISC": "TCDISCS",
              "TEFFPROS": "TEFFPROS", "T3EFFPD": "TEFFPROS",
              "TJSPROS": "TJSPROS", "T3JSPRO": "TJSPROS"}

# Public release (year, month) of international results. ASSUMPTION: verify.
RELEASE = {
    ("PISA", 2000): (2001, 12), ("PISA", 2003): (2004, 12), ("PISA", 2006): (2007, 12),
    ("PISA", 2009): (2010, 12), ("PISA", 2012): (2013, 12), ("PISA", 2015): (2016, 12),
    ("PISA", 2018): (2019, 12), ("PISA", 2022): (2023, 12), ("PISA", 2025): (2026, 9),
    ("TIMSS_G8", 1995): (1996, 11), ("TIMSS_G8", 1999): (2000, 12), ("TIMSS_G8", 2003): (2004, 12),
    ("TIMSS_G8", 2007): (2008, 12), ("TIMSS_G8", 2011): (2012, 12), ("TIMSS_G8", 2015): (2016, 11),
    ("TIMSS_G8", 2019): (2020, 12), ("TIMSS_G8", 2023): (2024, 12),
    ("TIMSS_G4", 1995): (1997, 6), ("TIMSS_G4", 2003): (2004, 12), ("TIMSS_G4", 2007): (2008, 12),
    ("TIMSS_G4", 2011): (2012, 12), ("TIMSS_G4", 2015): (2016, 11), ("TIMSS_G4", 2019): (2020, 12),
    ("TIMSS_G4", 2023): (2024, 12),
    ("PIRLS", 2001): (2003, 4), ("PIRLS", 2006): (2007, 11), ("PIRLS", 2011): (2012, 12),
    ("PIRLS", 2016): (2017, 12), ("PIRLS", 2021): (2023, 5),
    ("ICCS", 2009): (2010, 11), ("ICCS", 2016): (2017, 11), ("ICCS", 2022): (2023, 11),
    ("ICILS", 2013): (2014, 11), ("ICILS", 2018): (2019, 11), ("ICILS", 2023): (2024, 11),
    ("PIAAC", 2012): (2013, 10), ("PIAAC", 2022): (2024, 12),
    ("TALIS", 2008): (2009, 6), ("TALIS", 2013): (2014, 6), ("TALIS", 2018): (2019, 6),
    ("TALIS", 2024): (2025, 10),
}

SCORE_FEATURES = ["lag1", "lag2f", "delta", "has_lag2", "gap"]


def month_index(program: str, cycle: int) -> int:
    year, month = RELEASE[(program, int(cycle))]
    return year * 12 + month


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_estimates() -> tuple[pd.DataFrame, dict]:
    est = pd.read_csv(ESTIMATES, dtype={"country_iso3": str})
    est["program"] = est["program"].str.upper().replace({"TIMSS": "TIMSS_G8"})
    est["domain"] = est["domain"].str.lower()
    log = {}

    xw = pd.read_csv(PISA2000_XWALK, dtype=str)
    xmap = dict(zip(xw["source_country_id"], xw["canonical_alpha3"]))
    is_p2000 = (est["program"] == "PISA") & (est["cycle"] == 2000)
    est.loc[is_p2000, "country_iso3"] = est.loc[is_p2000, "country_iso3"].map(xmap).fillna(
        est.loc[is_p2000, "country_iso3"])
    log["pisa2000_crosswalk_rows"] = int(is_p2000.sum())

    bad = ~est["country_iso3"].str.fullmatch(r"[A-Z]{3}")
    log["dropped_non_iso_codes"] = sorted(
        f"{p}:{c}:{i}" for p, c, i in est.loc[bad, ["program", "cycle", "country_iso3"]].drop_duplicates().itertuples(index=False))
    est = est[~bad].copy()

    piaac17 = (est["program"] == "PIAAC") & (est["cycle"] == 2017)
    est.loc[piaac17, "cycle"] = 2022
    log["piaac_2017_relabelled_2022_rows"] = int(piaac17.sum())
    return est, log


def load_covariates(est: pd.DataFrame, kb: KBWeights) -> tuple[pd.DataFrame, dict]:
    """Long table: target_program, variable, source_program, source_cycle, country_iso3, value."""
    frames = []
    cov = pd.read_csv(COVARIATES, dtype={"country_iso3": str})
    cov["program"] = cov["program"].str.upper().replace({"TIMSS": "TIMSS_G8"})
    n_dup = int(cov.duplicated(["program", "cycle", "variable", "country_iso3"]).sum())
    cov = (cov.sort_values(["program", "cycle", "variable", "country_iso3", "n_total"],
                           ascending=[True, True, True, True, False])
              .drop_duplicates(["program", "cycle", "variable", "country_iso3"]))
    for prog, variables in PROGRAM_COVARIATES.items():
        sub = cov[(cov["program"] == prog) & cov["variable"].isin(variables)]
        frames.append(pd.DataFrame({
            "target_program": prog, "variable": sub["variable"], "source_program": prog,
            "source_cycle": sub["cycle"].astype(int), "country_iso3": sub["country_iso3"],
            "value": sub["mean"].astype(float)}))

    talis = pd.read_csv(TALIS, dtype={"country_iso3": str})
    talis = talis[talis["variable"].isin(TALIS_NAME)].copy()
    talis["variable"] = talis["variable"].map(TALIS_NAME)
    talis = talis.sort_values(["cycle", "variable", "country_iso3"]).drop_duplicates(
        ["cycle", "variable", "country_iso3"])
    frames.append(pd.DataFrame({
        "target_program": "PISA", "variable": talis["variable"], "source_program": "TALIS",
        "source_cycle": talis["cycle"].astype(int), "country_iso3": talis["country_iso3"],
        "value": talis["mean"].astype(float)}))

    for prog, domains in ESTIMATE_COVARIATE_DOMAINS.items():
        sub = est[(est["program"] == prog) & est["domain"].isin(domains)]
        frames.append(pd.DataFrame({
            "target_program": prog, "variable": prog + "_" + sub["domain"], "source_program": prog,
            "source_cycle": sub["cycle"].astype(int), "country_iso3": sub["country_iso3"],
            "value": sub["mean"].astype(float)}))

    out = pd.concat(frames, ignore_index=True)
    masked = []
    keep = np.ones(len(out), dtype=bool)
    for var in sorted(out["variable"].unique()):
        excl = set(kb.excluded_source_cycles(var))  # raises on unmapped variables
        if kb.is_excluded(var):
            keep &= out["variable"].to_numpy() != var
            masked.append(f"{var}:all (excluded in KB map)")
            continue
        if excl:
            tag = out["source_program"] + ":" + out["source_cycle"].astype(str)
            hit = (out["variable"] == var) & tag.isin(excl)
            keep &= ~hit.to_numpy()
            masked.append(f"{var}:{sorted(excl)} rows={int(hit.sum())}")
    out = out[keep].reset_index(drop=True)
    return out, {"covariate_duplicate_keys_resolved_by_max_n_total": n_dup,
                 "kb_masked_sources": masked}


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------

class Series:
    """Score history and covariate availability for one program-domain."""

    def __init__(self, program: str, domain: str, est: pd.DataFrame, cov: pd.DataFrame):
        self.program, self.domain = program, domain
        s = est[(est["program"] == program) & (est["domain"] == domain)]
        self.y = {(r.country_iso3, int(r.cycle)): float(r.mean) for r in s.itertuples()}
        self.se = {(r.country_iso3, int(r.cycle)): float(r.se) for r in s.itertuples()}
        self.cycles = sorted(s["cycle"].astype(int).unique())
        self.cov = cov[cov["target_program"] == program].copy()
        self.cov["release"] = [month_index(p, c) for p, c in
                               zip(self.cov["source_program"], self.cov["source_cycle"])]
        self.cov_vars = [v for v in PROGRAM_COVARIATES.get(program, [])
                         if v in set(self.cov["variable"])]

    def prev_cycle(self, t: int) -> int | None:
        prior = [c for c in self.cycles if c < t]
        return prior[-1] if prior else None

    def origin(self, t: int) -> int | None:
        prev = self.prev_cycle(t)
        return None if prev is None else month_index(self.program, prev)

    def history(self, country: str, t: int) -> list[tuple[int, float]]:
        return [(c, self.y[(country, c)]) for c in self.cycles if c < t and (country, c) in self.y]

    def covariate_frame(self, t: int) -> tuple[pd.DataFrame, dict]:
        """Latest released source per (country, variable) at the origin of t."""
        origin = self.origin(t)
        c = self.cov[(self.cov["source_cycle"] < t)]
        info = {}
        if origin is None or c.empty:
            return pd.DataFrame(columns=self.cov_vars), info
        late = c[c["release"] > origin]
        c = c[c["release"] <= origin]
        for v in self.cov_vars:
            used = sorted(f"{p}:{y}" for p, y in c.loc[c["variable"] == v, ["source_program", "source_cycle"]]
                          .drop_duplicates().itertuples(index=False))
            blocked = sorted(f"{p}:{y}" for p, y in late.loc[late["variable"] == v, ["source_program", "source_cycle"]]
                             .drop_duplicates().itertuples(index=False))
            info[v] = {"available_sources": used, "masked_after_origin": blocked}
        latest = (c.sort_values("source_cycle").groupby(["country_iso3", "variable"]).tail(1)
                    .pivot(index="country_iso3", columns="variable", values="value"))
        return latest.reindex(columns=self.cov_vars), info

    def rows(self, t: int) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Feature rows for target cycle t: (with history, without history)."""
        covf, _ = self.covariate_frame(t)
        recs, nohist = [], []
        for (country, cyc), y in sorted(self.y.items()):
            if cyc != t:
                continue
            hist = self.history(country, t)
            if not hist:
                nohist.append({"country_iso3": country, "cycle": t, "y": y})
                continue
            lag1_c, lag1 = hist[-1]
            lag2 = hist[-2][1] if len(hist) >= 2 else np.nan
            rec = {"country_iso3": country, "cycle": t, "y": y, "se": self.se.get((country, t), np.nan),
                   "lag1": lag1, "lag2f": lag2 if np.isfinite(lag2) else lag1,
                   "delta": (lag1 - lag2) if np.isfinite(lag2) else 0.0,
                   "has_lag2": float(np.isfinite(lag2)), "gap": float(t - lag1_c),
                   "hist_years": [c for c, _ in hist], "hist_vals": [v for _, v in hist]}
            for v in self.cov_vars:
                rec[v] = covf.at[country, v] if (country in covf.index and v in covf.columns) else np.nan
            recs.append(rec)
        return pd.DataFrame(recs), pd.DataFrame(nohist)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def _prep(train: pd.DataFrame, test: pd.DataFrame, cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    xtr = train[cols].to_numpy(dtype=float)
    xte = test[cols].to_numpy(dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mu = np.nanmean(xtr, axis=0)
    # A column can be all-NaN inside an inner-CV split; it becomes a constant.
    mu = np.where(np.isnan(mu), 0.0, mu)
    xtr = np.where(np.isnan(xtr), mu, xtr)
    xte = np.where(np.isnan(xte), mu, xte)
    return xtr, xte


def fit_ridge(train: pd.DataFrame, test: pd.DataFrame, cols: list[str],
              scale: np.ndarray) -> tuple[np.ndarray, float, str]:
    """Ridge with alpha chosen on the last training target cycle (time-ordered)."""
    def fit_predict(tr, te, alpha):
        xtr, xte = _prep(tr, te, cols)
        sc = StandardScaler().fit(xtr)
        m = Ridge(alpha=alpha).fit(sc.transform(xtr) * scale, tr["y"].to_numpy())
        return m.predict(sc.transform(xte) * scale)

    cycles = sorted(train["cycle"].unique())
    inner_tr = train[train["cycle"] < cycles[-1]]
    inner_va = train[train["cycle"] == cycles[-1]]
    if len(cycles) >= 2 and len(inner_tr) >= MIN_TRAIN and len(inner_va) >= 3:
        maes = [np.mean(np.abs(fit_predict(inner_tr, inner_va, a) - inner_va["y"].to_numpy())) for a in ALPHA_GRID]
        alpha = ALPHA_GRID[int(np.argmin(maes))]
        mode = "inner_time_cv" + ("_boundary_low" if alpha == ALPHA_GRID[0] else
                                  "_boundary_high" if alpha == ALPHA_GRID[-1] else "")
    else:
        alpha, mode = FALLBACK_ALPHA, "fallback_fixed"
    return fit_predict(train, test, alpha), alpha, mode


def fit_tree(kind: str, train: pd.DataFrame, test: pd.DataFrame, cols: list[str]) -> np.ndarray:
    xtr, xte = _prep(train, test, cols)
    if kind == "RF":
        m = RandomForestRegressor(n_estimators=300, max_depth=4, min_samples_leaf=5,
                                  max_features=1.0, random_state=SEED, n_jobs=1)
    else:
        m = XGBRegressor(n_estimators=250, max_depth=2, learning_rate=0.03, min_child_weight=5,
                         subsample=0.8, colsample_bytree=0.9, reg_lambda=10.0,
                         objective="reg:squarederror", random_state=SEED, n_jobs=1, verbosity=0)
    m.fit(xtr, train["y"].to_numpy())
    return np.asarray(m.predict(xte), dtype=float)


def linear_trend(years: list[int], vals: list[float], t: int) -> float:
    if len(years) < 2:
        return float(vals[-1])
    slope, intercept = np.polyfit(np.asarray(years, float), np.asarray(vals, float), 1)
    return float(intercept + slope * t)


# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------

def run_series(ser: Series, kb: KBWeights) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    preds, folds, nohist_rows, alphas = [], [], [], []
    for t in ser.cycles:
        prev = ser.prev_cycle(t)
        if prev is None:
            continue
        test, nohist = ser.rows(t)
        prev_vals = {c: v for (c, cy), v in ser.y.items() if cy == prev}
        intl_mean = float(np.mean(list(prev_vals.values())))
        for r in nohist.itertuples():
            nohist_rows.append({"program": ser.program, "domain": ser.domain, "test_cycle": t,
                                "country_iso3": r.country_iso3, "y_true": r.y,
                                "InternationalMean": intl_mean})
        if test.empty:
            continue

        train = pd.concat([ser.rows(s)[0] for s in ser.cycles if s < t], ignore_index=True) \
            if any(s < t for s in ser.cycles) else pd.DataFrame()
        _, cov_info = ser.covariate_frame(t)

        # Baselines (no training needed)
        pp = ser.prev_cycle(prev)
        drift = 0.0
        if pp is not None:
            both = [ser.y[(c, prev)] - ser.y[(c, pp)] for (c, cy) in ser.y if cy == prev and (c, pp) in ser.y]
            drift = float(np.mean(both)) if both else 0.0
        out = {
            "Persistence": test["lag1"].to_numpy(),
            "Drift": test["lag1"].to_numpy() + drift,
            "InternationalMean": np.full(len(test), intl_mean),
            "LinearTrend": np.array([linear_trend(y, v, t) for y, v in zip(test["hist_years"], test["hist_vals"])]),
        }

        learned = ["AR1", "Ridge_M0A", "Ridge_M0", "Ridge_M1",
                   "RF_score", "XGB_score", "RF_cov", "XGB_cov"]
        cov_used, cov_dropped = [], []
        if len(train) >= MIN_TRAIN:
            for v in ser.cov_vars:
                col = train[v].to_numpy(dtype=float) if v in train else np.array([np.nan])
                ok = np.isfinite(col).any() and np.nanstd(col) > 1e-9
                (cov_used if ok else cov_dropped).append(v)
            ytr = train["y"].to_numpy()
            out["AR1"] = LinearRegression().fit(train[["lag1"]].to_numpy(), ytr).predict(test[["lag1"]].to_numpy())
            ones = np.ones(len(SCORE_FEATURES))
            out["Ridge_M0A"], a, m = fit_ridge(train, test, SCORE_FEATURES, ones)
            alphas.append({"model": "Ridge_M0A", "alpha": a, "mode": m})
            cols = SCORE_FEATURES + cov_used
            out["Ridge_M0"], a, m = fit_ridge(train, test, cols, np.ones(len(cols)))
            alphas.append({"model": "Ridge_M0", "alpha": a, "mode": m})
            w = kb.vector(cols, neutral=set(SCORE_FEATURES))
            out["Ridge_M1"], a, m = fit_ridge(train, test, cols, np.sqrt(w))
            alphas.append({"model": "Ridge_M1", "alpha": a, "mode": m})
            out["RF_score"] = fit_tree("RF", train, test, SCORE_FEATURES)
            out["XGB_score"] = fit_tree("XGB", train, test, SCORE_FEATURES)
            out["RF_cov"] = fit_tree("RF", train, test, cols)
            out["XGB_cov"] = fit_tree("XGB", train, test, cols)
            for al in alphas[-3:]:
                al.update({"program": ser.program, "domain": ser.domain, "test_cycle": t})
        complete = all(k in out for k in learned)

        folds.append({
            "program": ser.program, "domain": ser.domain, "test_cycle": t, "prev_cycle": prev,
            "origin_year_month": f"{RELEASE[(ser.program, prev)][0]}-{RELEASE[(ser.program, prev)][1]:02d}",
            "n_train": len(train), "n_test": len(test), "n_no_history": len(nohist),
            "complete": complete, "drift": round(drift, 4),
            "covariates_used": "|".join(cov_used), "covariates_dropped_no_train_signal": "|".join(cov_dropped),
            "covariate_sources": json.dumps({v: cov_info.get(v, {}) for v in ser.cov_vars}, sort_keys=True),
        })
        for model, yhat in out.items():
            for r, p in zip(test.itertuples(), yhat):
                preds.append({"program": ser.program, "domain": ser.domain, "test_cycle": t,
                              "country_iso3": r.country_iso3, "model": model, "y_true": r.y,
                              "se_true": r.se, "y_pred": float(p), "fold_complete": complete})
    return preds, folds, nohist_rows, alphas


def load_timesfm() -> pd.DataFrame:
    a = pd.read_csv(TIMESFM_FILES["PISA"])
    a = a[a["model"] == "TimesFM2_5_zero_shot"].assign(program="PISA").rename(columns={"target_cycle": "test_cycle"})
    b = pd.read_csv(TIMESFM_FILES["OTHER"])
    b = b[b["model"] == "TimesFM2_5_zero_shot"]
    cols = ["program", "domain", "test_cycle", "country_iso3", "true_score", "prediction"]
    return pd.concat([a[cols], b[cols]], ignore_index=True)


def summarise(p: pd.DataFrame) -> pd.DataFrame:
    p = p.assign(ae=(p["y_pred"] - p["y_true"]).abs(), se2=(p["y_pred"] - p["y_true"]) ** 2)
    fold = p.groupby(["program", "domain", "model", "test_cycle"])["ae"].mean().rename("fold_mae").reset_index()
    s = (p.groupby(["program", "domain", "model"])
          .agg(n=("ae", "size"), n_folds=("test_cycle", "nunique"),
               MAE=("ae", "mean"), RMSE=("se2", lambda v: float(np.sqrt(v.mean()))))
          .reset_index())
    s = s.merge(fold.groupby(["program", "domain", "model"])["fold_mae"].mean().rename("MAE_fold_mean").reset_index(),
                on=["program", "domain", "model"])
    base = s[s["model"] == "Persistence"].set_index(["program", "domain"])["MAE"]
    s["rel_MAE_vs_persistence"] = [r.MAE / base.get((r.program, r.domain), np.nan) for r in s.itertuples()]
    s["rank"] = s.groupby(["program", "domain"])["MAE"].rank(method="min").astype(int)
    return s.sort_values(["program", "domain", "MAE"]).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    out_dir = args.out if args.out.is_absolute() else ROOT / args.out
    if out_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {out_dir}")

    kb = KBWeights()
    est, est_log = load_estimates()
    cov, cov_log = load_covariates(est, kb)

    preds, folds, nohist, alphas = [], [], [], []
    for prog, dom in MAIN + APPENDIX:
        p, f, n, a = run_series(Series(prog, dom, est, cov), kb)
        preds += p; folds += f; nohist += n; alphas += a
    pred = pd.DataFrame(preds)
    fold = pd.DataFrame(folds)
    if not np.isfinite(pred["y_pred"]).all():
        raise AssertionError("non-finite predictions")

    # Same-n check: within a complete fold every model has every test row.
    comp = pred[pred["fold_complete"]]
    counts = comp.groupby(["program", "domain", "test_cycle", "model"]).size().unstack("model")
    if not (counts.nunique(axis=1) == 1).all():
        raise AssertionError("models differ in n within a complete fold")

    group = {(p, d): ("main" if (p, d) in MAIN else "appendix") for p, d in MAIN + APPENDIX}
    pred["group"] = [group[(p, d)] for p, d in zip(pred["program"], pred["domain"])]
    main_c = comp[[group[(p, d)] == "main" for p, d in zip(comp["program"], comp["domain"])]]
    primary = summarise(main_c[~main_c["test_cycle"].isin(PRIMARY_EXCLUDED_TEST_CYCLES)])
    sensitivity = summarise(main_c)
    appendix_all = pred[pred["group"] == "appendix"]
    appendix = summarise(appendix_all[appendix_all["fold_complete"]])
    appendix_baseline = summarise(appendix_all[~appendix_all["fold_complete"]])

    # TimesFM subset: rows where saved TimesFM forecasts exist; all models re-scored there.
    tf = load_timesfm()
    key = ["program", "domain", "test_cycle", "country_iso3"]
    base = main_c[main_c["model"] == "Persistence"][key + ["y_true"]]
    tfm = base.merge(tf, on=key, how="inner")
    mismatch = tfm[(tfm["true_score"] - tfm["y_true"]).abs() > 0.01]
    tfm = tfm[(tfm["true_score"] - tfm["y_true"]).abs() <= 0.01]
    tf_pred = tfm.assign(model="TimesFM2_5_zero_shot", y_pred=tfm["prediction"])[key + ["model", "y_true", "y_pred"]]
    subset = main_c.merge(tfm[key], on=key)[key + ["model", "y_true", "y_pred"]]
    tf_all = pd.concat([subset, tf_pred], ignore_index=True)
    timesfm_primary = summarise(tf_all[~tf_all["test_cycle"].isin(PRIMARY_EXCLUDED_TEST_CYCLES)])

    out_dir.mkdir(parents=True)
    fmt = "%.6f"
    pred.drop(columns=["group"]).to_csv(out_dir / "predictions.csv", index=False, float_format=fmt)
    fold.to_csv(out_dir / "folds.csv", index=False, float_format=fmt)
    pd.DataFrame(nohist).to_csv(out_dir / "no_history.csv", index=False, float_format=fmt)
    pd.DataFrame(alphas)[["program", "domain", "test_cycle", "model", "alpha", "mode"]].to_csv(
        out_dir / "ridge_alpha_log.csv", index=False)
    primary.to_csv(out_dir / "summary_primary_excl2025.csv", index=False, float_format=fmt)
    sensitivity.to_csv(out_dir / "summary_sensitivity_incl2025.csv", index=False, float_format=fmt)
    appendix.to_csv(out_dir / "summary_appendix_complete_folds.csv", index=False, float_format=fmt)
    appendix_baseline.to_csv(out_dir / "summary_appendix_baselines_incomplete_folds.csv", index=False, float_format=fmt)
    timesfm_primary.to_csv(out_dir / "summary_timesfm_subset_excl2025.csv", index=False, float_format=fmt)

    git = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    inputs = [ESTIMATES, COVARIATES, TALIS, PISA2000_XWALK, *TIMESFM_FILES.values(),
              ROOT / "outputs/stage5/predictor_weights_v3.csv", ROOT / "outputs/stage5/kb_feature_map_v3.csv"]
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_head": git("rev-parse", "HEAD"),
        "python": platform.python_version(), "sklearn": sklearn.__version__,
        "xgboost": xgboost.__version__, "numpy": np.__version__, "pandas": pd.__version__,
        "inputs": {str(p.relative_to(ROOT)): sha256(p) for p in inputs},
        "design": {
            "origin": "Definition A: release of the previous cycle of the same program-domain",
            "primary_excludes_test_cycles": sorted(PRIMARY_EXCLUDED_TEST_CYCLES),
            "min_train_rows": MIN_TRAIN, "alpha_grid": ALPHA_GRID, "fallback_alpha": FALLBACK_ALPHA,
            "score_features": SCORE_FEATURES, "program_covariates": PROGRAM_COVARIATES,
            "release_dates_status": "ASSUMPTION (external knowledge)",
        },
        "data_log": {**est_log, **cov_log, "timesfm_true_score_mismatches_dropped": int(len(mismatch))},
        "n_predictions": int(len(pred)), "n_folds": int(len(fold)),
        "n_complete_folds": int(fold["complete"].sum()),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    pd.set_option("display.width", 200)
    print("=== Primary (main program-domains, complete folds, excl. 2025) ===")
    print(primary[["program", "domain", "model", "n", "n_folds", "MAE", "RMSE", "rel_MAE_vs_persistence", "rank"]]
          .to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nSaved to {out_dir}/")


if __name__ == "__main__":
    main()
