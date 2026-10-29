#!/usr/bin/env python3
"""
ILSA Tabular Benchmarks: RF, XGBoost, Ridge, Persistence, LinearTrend
across all ILSA programs and their primary outcome domains.

Programs covered:
  PISA      : mathematics, reading, science       (LOCO, 2000-2025)
  TIMSS     : mathematics, science                (LOCO, 1995-2023)
  TIMSS_G4  : mathematics, science                (LOCO)
  PIRLS     : reading                             (LOCO, 2001-2021)
  ICCS      : civic_knowledge                     (LOCO, 2009-2022)
  ICILS     : computer_literacy                   (LOCO, 2013-2023)
  PIAAC     : skipped (only 2 cycles)
  TALIS     : skipped (predictor-only, no score outcome)

Features (score-only, program-agnostic):
  lag1_score, lag2_score, score_change,
  target_cycle, gap_since_last_cycle, n_prior_observations

Output: outputs/ilsa_tabular_benchmark_v1/
"""
from __future__ import annotations

import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
STAGE5 = ROOT / "outputs" / "stage5"
OUT_DIR = ROOT / "outputs" / "ilsa_tabular_benchmark_v1"
SEED = 20261015

FEATURES = (
    "lag1_score",
    "lag2_score",
    "score_change",
    "target_cycle",
    "gap_since_last_cycle",
    "n_prior_observations",
)

# Primary outcome column per program
PROGRAM_OUTCOME: dict[str, str] = {
    "PISA":     "score",
    "TIMSS_G8": "score",
    "TIMSS_G4": "score",
    "PIRLS":    "score",
    "ICCS":     "ICCS_civic_knowledge",
    "ICILS":    "ICILS_computer_literacy",
}

PROGRAM_DOMAINS: dict[str, list[str]] = {
    "PISA":     ["mathematics", "reading", "science"],
    "TIMSS_G8": ["mathematics", "science"],
    "TIMSS_G4": ["mathematics", "science"],
    "PIRLS":    ["reading"],
    "ICCS":     ["civic_knowledge"],
    "ICILS":    ["computer_literacy"],
}


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def load_pisa() -> dict[str, pd.DataFrame]:
    """Return {domain: DataFrame(country_iso3, cycle, score)} for PISA."""
    panel = pd.read_csv(STAGE5 / "pisa_forecasting_panel_v2.csv")
    panel = panel[panel["program"] == "PISA"].copy()
    panel["cycle"] = panel["cycle"].astype(int)
    out = {}
    for domain in PROGRAM_DOMAINS["PISA"]:
        col = f"PISA_{domain}"
        if col not in panel.columns:
            continue
        sub = panel[panel["domain"] == domain][["country_iso3", "cycle", col]].dropna()
        sub = sub.rename(columns={col: "score"})
        out[domain] = sub
    return out


def load_enriched(program: str) -> dict[str, pd.DataFrame]:
    """Return {domain: DataFrame(country_iso3, cycle, score)} for TIMSS_G8/TIMSS_G4/PIRLS."""
    panel = pd.read_csv(STAGE5 / "enriched_panel.csv")
    # Map TIMSS_G8 label back to raw "TIMSS" in the CSV
    raw_program = "TIMSS" if program == "TIMSS_G8" else program
    panel = panel[panel["program"] == raw_program].copy()
    panel["cycle"] = panel["cycle"].astype(int)
    out = {}
    for domain in PROGRAM_DOMAINS[program]:
        # Score column is named {RAW_PROGRAM}_{domain} e.g. TIMSS_mathematics
        col = f"{raw_program}_{domain}"
        if col not in panel.columns:
            continue
        sub = panel[panel["domain"] == domain][["country_iso3", "cycle", col]].dropna(subset=[col])
        sub = sub.rename(columns={col: "score"})
        out[domain] = sub
    return out


def load_iccs() -> dict[str, pd.DataFrame]:
    panel = pd.read_csv(STAGE5 / "iccs_enriched_panel.csv")
    panel["cycle"] = panel["cycle"].astype(int)
    sub = panel[["country_iso3", "cycle", "ICCS_civic_knowledge"]].dropna()
    return {"civic_knowledge": sub.rename(columns={"ICCS_civic_knowledge": "score"})}


def load_icils() -> dict[str, pd.DataFrame]:
    panel = pd.read_csv(STAGE5 / "icils_enriched_panel.csv")
    panel["cycle"] = panel["cycle"].astype(int)
    sub = panel[["country_iso3", "cycle", "ICILS_computer_literacy"]].dropna()
    return {"computer_literacy": sub.rename(columns={"ICILS_computer_literacy": "score"})}


def load_all_programs() -> dict[str, dict[str, pd.DataFrame]]:
    data: dict[str, dict[str, pd.DataFrame]] = {}
    data["PISA"]     = load_pisa()
    data["TIMSS_G8"] = load_enriched("TIMSS_G8")
    data["TIMSS_G4"] = load_enriched("TIMSS_G4")
    data["PIRLS"]    = load_enriched("PIRLS")
    data["ICCS"]     = load_iccs()
    data["ICILS"]    = load_icils()
    return data


# ---------------------------------------------------------------------------
# Feature engineering (same as run_pisa_tabular_benchmarks.py)
# ---------------------------------------------------------------------------

def feature_row(country: str, years: list[int], scores: list[float], target: int) -> dict:
    prior_years  = [y for y in years  if y < target]
    prior_scores = [s for y, s in zip(years, scores) if y < target]
    if not prior_scores:
        raise ValueError(f"No context for {country} @ {target}")
    lag1 = float(prior_scores[-1])
    lag2 = float(prior_scores[-2]) if len(prior_scores) >= 2 else np.nan
    return {
        "lag1_score":           lag1,
        "lag2_score":           lag2,
        "score_change":         lag1 - lag2 if np.isfinite(lag2) else np.nan,
        "target_cycle":         int(target),
        "gap_since_last_cycle": int(target - prior_years[-1]),
        "n_prior_observations": len(prior_scores),
    }


def training_examples(series: pd.DataFrame, cutoff: int,
                      min_prior: int = 2) -> tuple[pd.DataFrame, np.ndarray]:
    features, targets = [], []
    for country, group in series.groupby("country_iso3", sort=True):
        group  = group.sort_values("cycle")
        years  = group["cycle"].astype(int).tolist()
        scores = group["score"].astype(float).tolist()
        for i in range(1, len(years)):
            if years[i] >= cutoff:
                break
            if i < min_prior:
                continue
            row = feature_row(country, years[:i], scores[:i], years[i])
            features.append({k: row[k] for k in FEATURES})
            targets.append(scores[i])
    if not features:
        raise ValueError(f"No training examples before {cutoff}")
    return pd.DataFrame(features, columns=list(FEATURES)), np.asarray(targets, dtype=float)


def linear_trend(years: list[int], scores: list[float], target: int) -> float:
    if len(years) < 2:
        return float("nan")
    slope, intercept = np.polyfit(
        np.asarray(years, dtype=float), np.asarray(scores, dtype=float), 1
    )
    return float(intercept + slope * target)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def make_models() -> dict[str, object]:
    imp = lambda: SimpleImputer(strategy="median")
    return {
        "Ridge": make_pipeline(imp(), StandardScaler(), Ridge(alpha=10.0)),
        "RandomForest": make_pipeline(
            imp(),
            RandomForestRegressor(
                n_estimators=300, max_depth=4, min_samples_leaf=5,
                max_features=1.0, random_state=SEED, n_jobs=-1,
            ),
        ),
        "XGBoost": make_pipeline(
            imp(),
            XGBRegressor(
                n_estimators=250, max_depth=2, learning_rate=0.03,
                min_child_weight=5, subsample=0.8, colsample_bytree=0.9,
                reg_lambda=10.0, objective="reg:squarederror",
                eval_metric="mae", random_state=SEED, n_jobs=-1, verbosity=0,
            ),
        ),
    }


# ---------------------------------------------------------------------------
# LOCO evaluation for one program-domain series
# ---------------------------------------------------------------------------

def run_loco(program: str, domain: str, series: pd.DataFrame,
             min_prior: int = 2, min_test: int = 3) -> list[dict]:
    series = series.copy().sort_values(["country_iso3", "cycle"])
    all_cycles = sorted(series["cycle"].unique())
    results: list[dict] = []

    for test_cycle in all_cycles:
        train_series = series[series["cycle"] < test_cycle]
        test_series  = series[series["cycle"] == test_cycle]

        if train_series["cycle"].nunique() < min_prior:
            continue

        try:
            x_train, y_train = training_examples(train_series, test_cycle, min_prior)
        except ValueError:
            continue

        test_records: list[dict] = []
        for country, group in test_series.groupby("country_iso3"):
            hist = series[
                (series["country_iso3"] == country) & (series["cycle"] < test_cycle)
            ].sort_values("cycle")
            if len(hist) < min_prior:
                continue
            true_val = group["score"].iloc[0]
            if not np.isfinite(true_val):
                continue
            years  = hist["cycle"].astype(int).tolist()
            scores = hist["score"].astype(float).tolist()
            try:
                feat = feature_row(country, years, scores, test_cycle)
            except ValueError:
                continue
            test_records.append({
                "country_iso3": country,
                "true_score":   float(true_val),
                "persistence":  scores[-1],
                "linear_trend": linear_trend(years, scores, test_cycle),
                "features":     feat,
            })

        if len(test_records) < min_test:
            continue

        x_test = pd.DataFrame(
            [{k: r["features"][k] for k in FEATURES} for r in test_records],
            columns=list(FEATURES),
        )

        model_preds: dict[str, np.ndarray] = {
            "Persistence": np.asarray([r["persistence"]  for r in test_records]),
            "LinearTrend": np.asarray([r["linear_trend"] for r in test_records]),
        }
        for name, model in make_models().items():
            model.fit(x_train[list(FEATURES)], y_train)
            model_preds[name] = np.asarray(model.predict(x_test), dtype=float)

        for name, preds in model_preds.items():
            for rec, pred in zip(test_records, preds):
                err = float(pred - rec["true_score"])
                results.append({
                    "program":      program,
                    "domain":       domain,
                    "test_cycle":   test_cycle,
                    "country_iso3": rec["country_iso3"],
                    "model":        name,
                    "true_score":   rec["true_score"],
                    "prediction":   float(pred),
                    "signed_error": err,
                    "abs_error":    abs(err),
                    "sq_error":     err ** 2,
                    "n_train":      len(y_train),
                    "n_test":       len(test_records),
                })

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if OUT_DIR.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing output: {OUT_DIR}"
        )

    print("Loading program data...")
    all_data = load_all_programs()

    all_predictions: list[dict] = []

    for program, domain_data in all_data.items():
        for domain, series in domain_data.items():
            n_countries = series["country_iso3"].nunique()
            n_cycles    = series["cycle"].nunique()
            print(f"  {program}/{domain}: {n_countries} countries, {n_cycles} cycles")

            if n_cycles < 2:
                print(f"    → skipped (< 2 cycles)")
                continue

            # Programs with few cycles use min_prior=1 to produce training examples
            min_prior = 1 if n_cycles <= 3 else 2
            rows = run_loco(program, domain, series, min_prior=min_prior)
            if not rows:
                print(f"    → no LOCO folds produced")
                continue

            all_predictions.extend(rows)
            folds = len(set((r["test_cycle"] for r in rows)))
            print(f"    → {len(rows)} predictions across {folds} folds")

    if not all_predictions:
        print("No predictions produced.")
        return

    pred_df = pd.DataFrame(all_predictions)

    metrics = (
        pred_df.groupby(["program", "domain", "test_cycle", "model"], as_index=False)
        .agg(
            n_countries=("country_iso3", "nunique"),
            MAE=("abs_error",    "mean"),
            RMSE=("sq_error",    lambda v: float(np.sqrt(v.mean()))),
            mean_bias=("signed_error", "mean"),
        )
        .sort_values(["program", "domain", "test_cycle", "model"])
        .reset_index(drop=True)
    )

    summary = (
        pred_df.groupby(["program", "domain", "model"], as_index=False)
        .agg(
            n_folds=("test_cycle",  "nunique"),
            n_preds=("country_iso3","count"),
            MAE=("abs_error",       "mean"),
            RMSE=("sq_error",       lambda v: float(np.sqrt(v.mean()))),
        )
        .sort_values(["program", "domain", "MAE"])
        .reset_index(drop=True)
    )

    OUT_DIR.mkdir(parents=True, exist_ok=False)
    pred_df.to_csv(OUT_DIR / "predictions.csv", index=False)
    metrics.to_csv(OUT_DIR / "metrics_by_fold.csv", index=False, float_format="%.4f")
    summary.to_csv(OUT_DIR / "summary.csv", index=False, float_format="%.4f")

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python":       platform.python_version(),
        "sklearn":      sklearn.__version__,
        "xgboost":      xgboost.__version__,
        "models":       ["Persistence", "LinearTrend", "Ridge", "RandomForest", "XGBoost"],
        "features":     list(FEATURES),
        "programs":     {p: list(d.keys()) for p, d in all_data.items()},
        "min_prior_observations": 2,
        "min_test_countries":     3,
        "outputs": {
            "predictions":    "predictions.csv",
            "metrics_by_fold":"metrics_by_fold.csv",
            "summary":        "summary.csv",
        },
    }
    (OUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    print(f"\n=== Summary (mean across folds) ===")
    print(summary.to_string(index=False))
    print(f"\nSaved to: {OUT_DIR.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
