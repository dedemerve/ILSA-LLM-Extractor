#!/usr/bin/env python3
"""Run leakage-controlled PISA historical tabular forecasting benchmarks.

Evaluates persistence, country-specific linear trend, Ridge, Random Forest,
and XGBoost over the frozen common-country rolling-origin folds. PISA 2025 is
not scored or used here; the manifest marks it as outcome-exposed.
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
PANEL_PATH = ROOT / "outputs" / "stage5" / "pisa_forecasting_panel_v2.csv"
PANEL_MANIFEST_PATH = (
    ROOT / "outputs" / "stage5" / "pisa_forecasting_panel_v2_manifest.json"
)
FOLDS_PATH = (
    ROOT
    / "analysis"
    / "03_data_feasibility"
    / "audit"
    / "pisa_historical_folds_v1.json"
)
OUT_DIR = ROOT / "outputs" / "pisa_tabular_benchmark_v1"
DOMAINS = ("mathematics", "reading", "science")
FEATURES = (
    "lag1_score",
    "lag2_score",
    "score_change",
    "target_cycle",
    "gap_since_last_cycle",
    "n_prior_observations",
)
SEED = 20261015


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def feature_row(country: str, years: list[int], scores: list[float], target: int) -> dict:
    prior_years = [year for year in years if year < target]
    prior_scores = [score for year, score in zip(years, scores) if year < target]
    if not prior_scores:
        raise ValueError(f"No context for {country} at target {target}")
    lag1 = float(prior_scores[-1])
    lag2 = float(prior_scores[-2]) if len(prior_scores) >= 2 else np.nan
    return {
        "country_iso3": country,
        "lag1_score": lag1,
        "lag2_score": lag2,
        "score_change": lag1 - lag2 if np.isfinite(lag2) else np.nan,
        "target_cycle": int(target),
        "gap_since_last_cycle": int(target - prior_years[-1]),
        "n_prior_observations": len(prior_scores),
        "context_cycles": prior_years,
        "context_scores": prior_scores,
    }


def training_examples(series: pd.DataFrame, cutoff: int) -> tuple[pd.DataFrame, np.ndarray]:
    features: list[dict] = []
    targets: list[float] = []
    for country, group in series.groupby("country_iso3", sort=True):
        group = group.sort_values("cycle")
        years = group["cycle"].astype(int).tolist()
        scores = group["score"].astype(float).tolist()
        for i in range(1, len(years)):
            target_year = years[i]
            if target_year >= cutoff:
                break
            row = feature_row(country, years[:i], scores[:i], target_year)
            features.append({name: row[name] for name in FEATURES})
            targets.append(scores[i])
    if not features:
        raise ValueError(f"No training examples before cycle {cutoff}")
    return pd.DataFrame(features, columns=FEATURES), np.asarray(targets, dtype=float)


def make_models() -> dict[str, object]:
    imputer = lambda: SimpleImputer(strategy="median")
    return {
        "Ridge": make_pipeline(
            imputer(), StandardScaler(), Ridge(alpha=10.0)
        ),
        "RandomForest": make_pipeline(
            imputer(),
            RandomForestRegressor(
                n_estimators=300,
                max_depth=4,
                min_samples_leaf=5,
                max_features=1.0,
                random_state=SEED,
                n_jobs=-1,
            ),
        ),
        "XGBoost": make_pipeline(
            imputer(),
            XGBRegressor(
                n_estimators=250,
                max_depth=2,
                learning_rate=0.03,
                min_child_weight=5,
                subsample=0.8,
                colsample_bytree=0.9,
                reg_lambda=10.0,
                objective="reg:squarederror",
                eval_metric="mae",
                random_state=SEED,
                n_jobs=-1,
                verbosity=0,
            ),
        ),
    }


def linear_trend(years: list[int], scores: list[float], target: int) -> float:
    if len(years) < 2:
        return float("nan")
    slope, intercept = np.polyfit(np.asarray(years, dtype=float), np.asarray(scores, dtype=float), 1)
    return float(intercept + slope * target)


def main() -> None:
    for path in (PANEL_PATH, PANEL_MANIFEST_PATH, FOLDS_PATH):
        if not path.is_file():
            raise FileNotFoundError(path)
    if OUT_DIR.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing experiment output: {OUT_DIR}"
        )

    panel_manifest = json.loads(PANEL_MANIFEST_PATH.read_text(encoding="utf-8"))
    if panel_manifest["output_sha256"] != sha256(PANEL_PATH):
        raise ValueError("Panel hash does not match its build manifest")
    folds = json.loads(FOLDS_PATH.read_text(encoding="utf-8"))
    if folds["panel_sha256"] != sha256(PANEL_PATH):
        raise ValueError("Fold manifest was built from a different panel version")
    if folds["pisa_2025"]["excluded_from_historical_fold_manifest"] is not True:
        raise ValueError("PISA 2025 must remain outside the historical development folds")

    panel = pd.read_csv(PANEL_PATH, dtype={"country_iso3": str})
    panel = panel.loc[panel["program"].eq("PISA")].copy()
    panel["cycle"] = pd.to_numeric(panel["cycle"], errors="raise").astype(int)

    predictions: list[dict] = []
    fold_training_counts: dict[str, dict[str, int]] = {}
    for fold in folds["folds"]:
        target_cycle = int(fold["target_cycle"])
        common_countries = set(fold["common_three_domain_countries"])
        fold_training_counts[str(target_cycle)] = {}

        for domain in DOMAINS:
            score_col = f"PISA_{domain}"
            series = panel.loc[
                (panel["domain"] == domain) & panel[score_col].notna(),
                ["country_iso3", "cycle", score_col],
            ].rename(columns={score_col: "score"})
            x_train, y_train = training_examples(series, target_cycle)
            fold_training_counts[str(target_cycle)][domain] = len(y_train)

            test_records: list[dict] = []
            for country in sorted(common_countries):
                observed = series.loc[series["country_iso3"] == country].sort_values("cycle")
                context = observed.loc[observed["cycle"] < target_cycle]
                target = observed.loc[observed["cycle"] == target_cycle, "score"]
                if len(context) < 3 or len(target) != 1:
                    raise ValueError(
                        f"Fold mismatch: {country}/{domain}/{target_cycle}, "
                        f"context={len(context)}, target_rows={len(target)}"
                    )
                years = context["cycle"].astype(int).tolist()
                scores = context["score"].astype(float).tolist()
                features = feature_row(country, years, scores, target_cycle)
                test_records.append(
                    {
                        "country_iso3": country,
                        "true_score": float(target.iloc[0]),
                        "n_context": len(scores),
                        "last_cycle": years[-1],
                        "persistence": scores[-1],
                        "linear_trend": linear_trend(years, scores, target_cycle),
                        "features": features,
                    }
                )

            x_test = pd.DataFrame(
                [{feature: item["features"][feature] for feature in FEATURES} for item in test_records],
                columns=FEATURES,
            )
            model_predictions: dict[str, np.ndarray] = {
                "Persistence": np.asarray([item["persistence"] for item in test_records]),
                "LinearTrend": np.asarray([item["linear_trend"] for item in test_records]),
            }
            for name, model in make_models().items():
                model.fit(x_train.loc[:, FEATURES], y_train)
                model_predictions[name] = np.asarray(model.predict(x_test), dtype=float)

            for name, values in model_predictions.items():
                if len(values) != len(test_records) or not np.isfinite(values).all():
                    raise ValueError(
                        f"Invalid predictions for {name}/{domain}/{target_cycle}"
                    )
                for record, prediction in zip(test_records, values):
                    error = float(prediction - record["true_score"])
                    predictions.append(
                        {
                            "target_cycle": target_cycle,
                            "domain": domain,
                            "country_iso3": record["country_iso3"],
                            "model": name,
                            "n_context": record["n_context"],
                            "last_cycle": record["last_cycle"],
                            "true_score": record["true_score"],
                            "prediction": float(prediction),
                            "signed_error": error,
                            "absolute_error": abs(error),
                            "squared_error": error**2,
                            "train_examples": len(y_train),
                        }
                    )

    prediction_df = pd.DataFrame(predictions)
    metrics = (
        prediction_df.groupby(["target_cycle", "domain", "model"], as_index=False)
        .agg(
            n=("country_iso3", "nunique"),
            MAE=("absolute_error", "mean"),
            RMSE=("squared_error", lambda values: float(np.sqrt(values.mean()))),
            mean_signed_error=("signed_error", "mean"),
            n_overpredicted=("signed_error", lambda values: int((values > 0).sum())),
            n_underpredicted=("signed_error", lambda values: int((values < 0).sum())),
            n_tied=("signed_error", lambda values: int((values == 0).sum())),
        )
        .sort_values(["target_cycle", "domain", "model"])
        .reset_index(drop=True)
    )

    OUT_DIR.mkdir(parents=True, exist_ok=False)
    predictions_path = OUT_DIR / "historical_predictions.csv"
    metrics_path = OUT_DIR / "historical_metrics.csv"
    manifest_path = OUT_DIR / "run_manifest.json"
    prediction_df.to_csv(predictions_path, index=False)
    metrics.to_csv(metrics_path, index=False, float_format="%.6f")
    run_manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "sklearn": sklearn.__version__,
        "xgboost": xgboost.__version__,
        "panel": str(PANEL_PATH.relative_to(ROOT)),
        "panel_sha256": sha256(PANEL_PATH),
        "fold_manifest": str(FOLDS_PATH.relative_to(ROOT)),
        "fold_manifest_sha256": sha256(FOLDS_PATH),
        "outer_folds": [fold["target_cycle"] for fold in folds["folds"]],
        "domains": list(DOMAINS),
        "test_country_rule": "common country set across all three domains per target fold",
        "minimum_prior_cycles": 3,
        "feature_names": list(FEATURES),
        "feature_definitions": {
            "lag1_score": "most recent country-domain score strictly before the row target",
            "lag2_score": "second most recent country-domain score; missing if unavailable",
            "score_change": "lag1_score minus lag2_score",
            "target_cycle": "assessment cycle year of the supervised training target",
            "gap_since_last_cycle": "target cycle year minus latest prior observed cycle year",
            "n_prior_observations": "number of score observations strictly before the supervised target",
        },
        "model_configuration": {
            "Ridge": {"alpha": 10.0, "preprocessing": "training-fold median imputation and standardization"},
            "RandomForest": {
                "n_estimators": 300,
                "max_depth": 4,
                "min_samples_leaf": 5,
                "max_features": 1.0,
                "seed": SEED,
                "preprocessing": "training-fold median imputation",
            },
            "XGBoost": {
                "n_estimators": 250,
                "max_depth": 2,
                "learning_rate": 0.03,
                "min_child_weight": 5,
                "subsample": 0.8,
                "colsample_bytree": 0.9,
                "reg_lambda": 10.0,
                "seed": SEED,
                "preprocessing": "training-fold median imputation",
            },
        },
        "cutoff_rule": "For target cycle t, all supervised training target cycles are < t.",
        "hyperparameter_policy": "Fixed before this run; no PISA 2025 results used for selection.",
        "pisa_2025_role": "outcome-exposed retrospective benchmark; excluded from this run",
        "training_examples_per_fold_domain": fold_training_counts,
        "outputs": {
            "predictions": predictions_path.name,
            "metrics": metrics_path.name,
        },
    }
    manifest_path.write_text(json.dumps(run_manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {predictions_path.relative_to(ROOT)} ({len(prediction_df)} predictions)")
    print(f"Wrote {metrics_path.relative_to(ROOT)} ({len(metrics)} metric rows)")
    print(f"Wrote {manifest_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
