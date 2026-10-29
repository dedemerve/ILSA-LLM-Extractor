#!/usr/bin/env python3
"""Score frozen tabular baselines on the outcome-exposed PISA 2025 panel.

The configuration is copied from the completed historical benchmark. No
PISA 2025 outcomes are used for feature selection or hyperparameter tuning.
This is explicitly a retrospective, outcome-exposed benchmark.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from run_pisa_tabular_benchmarks import (
    DOMAINS,
    FEATURES,
    PANEL_MANIFEST_PATH,
    PANEL_PATH,
    feature_row,
    linear_trend,
    make_models,
    sha256,
    training_examples,
)


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs/pisa_2025_retrospective_tabular_v1"
TARGET_CYCLE = 2025
MIN_CONTEXT = 3


def main() -> None:
    if OUT_DIR.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {OUT_DIR}")
    if not PANEL_PATH.is_file() or not PANEL_MANIFEST_PATH.is_file():
        raise FileNotFoundError("Canonical panel or its manifest is missing")
    panel_hash = sha256(PANEL_PATH)
    panel_manifest = json.loads(PANEL_MANIFEST_PATH.read_text(encoding="utf-8"))
    if panel_manifest["output_sha256"] != panel_hash:
        raise ValueError("Canonical PISA panel hash does not match its build manifest")

    panel = pd.read_csv(PANEL_PATH, dtype={"country_iso3": str})
    panel = panel.loc[panel["program"].eq("PISA")].copy()
    panel["cycle"] = pd.to_numeric(panel["cycle"], errors="raise").astype(int)

    eligible_by_domain: dict[str, set[str]] = {}
    domain_series: dict[str, pd.DataFrame] = {}
    for domain in DOMAINS:
        col = f"PISA_{domain}"
        series = panel.loc[
            panel["domain"].eq(domain) & panel[col].notna(),
            ["country_iso3", "cycle", col],
        ].rename(columns={col: "score"})
        domain_series[domain] = series
        eligible = set()
        for country, group in series.groupby("country_iso3"):
            context = group.loc[group["cycle"] < TARGET_CYCLE]
            target = group.loc[group["cycle"].eq(TARGET_CYCLE)]
            if len(context) >= MIN_CONTEXT and len(target) == 1:
                eligible.add(country)
        eligible_by_domain[domain] = eligible
    common = set.intersection(*(eligible_by_domain[d] for d in DOMAINS))
    if not common:
        raise ValueError("No common three-domain eligible countries for 2025")

    predictions: list[dict] = []
    train_sizes: dict[str, int] = {}
    for domain in DOMAINS:
        series = domain_series[domain]
        x_train, y_train = training_examples(series, TARGET_CYCLE)
        train_sizes[domain] = len(y_train)
        test_records = []
        for country in sorted(common):
            observed = series.loc[series["country_iso3"].eq(country)].sort_values("cycle")
            context = observed.loc[observed["cycle"] < TARGET_CYCLE]
            target = observed.loc[observed["cycle"].eq(TARGET_CYCLE), "score"]
            if len(context) < MIN_CONTEXT or len(target) != 1:
                raise ValueError(f"Eligibility inconsistency: {country}/{domain}")
            years = context["cycle"].astype(int).tolist()
            values = context["score"].astype(float).tolist()
            row = feature_row(country, years, values, TARGET_CYCLE)
            test_records.append(
                {
                    "target_cycle": TARGET_CYCLE,
                    "domain": domain,
                    "country_iso3": country,
                    "n_context": len(values),
                    "last_cycle": years[-1],
                    "context_cycles": ";".join(map(str, years)),
                    "true_score": float(target.iloc[0]),
                    "features": row,
                    "persistence": values[-1],
                    "linear_trend": linear_trend(years, values, TARGET_CYCLE),
                }
            )
        x_test = pd.DataFrame(
            [{feature: row["features"][feature] for feature in FEATURES} for row in test_records],
            columns=FEATURES,
        )
        values_by_model = {
            "Persistence": np.asarray([row["persistence"] for row in test_records]),
            "LinearTrend": np.asarray([row["linear_trend"] for row in test_records]),
        }
        for name, model in make_models().items():
            model.fit(x_train.loc[:, FEATURES], y_train)
            values_by_model[name] = np.asarray(model.predict(x_test), dtype=float)
        for name, forecasts in values_by_model.items():
            if len(forecasts) != len(test_records) or not np.isfinite(forecasts).all():
                raise ValueError(f"Invalid predictions from {name}/{domain}")
            for row, prediction in zip(test_records, forecasts):
                error = float(prediction - row["true_score"])
                predictions.append(
                    {
                        key: value
                        for key, value in row.items()
                        if key not in {"features", "persistence", "linear_trend"}
                    }
                    | {
                        "model": name,
                        "prediction": float(prediction),
                        "signed_error": error,
                        "absolute_error": abs(error),
                        "squared_error": error**2,
                    }
                )

    prediction_frame = pd.DataFrame(predictions)
    if prediction_frame.duplicated(["target_cycle", "domain", "country_iso3", "model"]).any():
        raise ValueError("Duplicate 2025 country/domain/model forecast keys")
    metrics = []
    for (domain, model_name), group in prediction_frame.groupby(["domain", "model"], sort=True):
        error = group["signed_error"].to_numpy(float)
        metrics.append(
            {
                "target_cycle": TARGET_CYCLE,
                "domain": domain,
                "model": model_name,
                "n": len(group),
                "MAE": float(np.abs(error).mean()),
                "RMSE": float(np.sqrt(np.square(error).mean())),
                "mean_signed_error": float(error.mean()),
                "n_overpredicted": int((error > 0).sum()),
                "n_underpredicted": int((error < 0).sum()),
                "n_tied": int((error == 0).sum()),
            }
        )
    metric_frame = pd.DataFrame(metrics)

    OUT_DIR.mkdir(parents=True)
    prediction_frame.to_csv(OUT_DIR / "tabular_predictions.csv", index=False)
    metric_frame.to_csv(OUT_DIR / "tabular_metrics.csv", index=False)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "sklearn": importlib.metadata.version("scikit-learn"),
        "xgboost": importlib.metadata.version("xgboost"),
        "panel": str(PANEL_PATH.relative_to(ROOT)),
        "panel_sha256": panel_hash,
        "target_cycle": TARGET_CYCLE,
        "role": "outcome-exposed retrospective benchmark; not a blind test and not used for model selection",
        "common_three_domain_n": len(common),
        "common_countries": sorted(common),
        "domain_eligible_counts": {domain: len(values) for domain, values in eligible_by_domain.items()},
        "minimum_context": MIN_CONTEXT,
        "training_example_counts": train_sizes,
        "feature_names": list(FEATURES),
        "cutoff_rule": "For target 2025, supervised training targets are strictly before 2025; preprocessing is fit to historical training examples only.",
        "hyperparameter_policy": "Use the unchanged fixed configurations from the historical tabular benchmark; no 2025 outcome used to select settings.",
        "outputs": ["tabular_predictions.csv", "tabular_metrics.csv"],
    }
    (OUT_DIR / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(metric_frame.to_string(index=False))
    print(f"Countries: {len(common)}; outputs: {OUT_DIR}")


if __name__ == "__main__":
    main()
