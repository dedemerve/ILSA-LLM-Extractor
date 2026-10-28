#!/usr/bin/env python3
"""Sensitivity summaries and paired country bootstrap for PISA forecasts.

The PISA 2025 results are outcome-exposed retrospective evidence. This script
does not fit or tune models; it summarizes their saved, matched predictions.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS = ROOT / "outputs/pisa_2025_retrospective_timesfm_v1/all_models_predictions.csv"
METRICS = ROOT / "outputs/pisa_2025_retrospective_timesfm_v1/metrics.csv"
FLAGS = ROOT / "analysis/03_data_feasibility/audit/pisa2025_se_validation.csv"
OUTPUT = ROOT / "outputs/pisa2025_forecast_sensitivity_v7"
DOMAINS = ("mathematics", "reading", "science")
SEED = 20261015
BOOTSTRAP_REPLICATES = 20000
METHODS = (
    "Persistence",
    "LinearTrend",
    "Ridge",
    "RandomForest",
    "XGBoost",
    "TimesFM3_zero_shot",
    "TimesFM2_5_zero_shot",
    "TimesFM2_5_ILSA_LoRA",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def metrics_for(group: pd.DataFrame, subset: str) -> dict:
    error = group["prediction"].to_numpy(dtype=float) - group["true_score"].to_numpy(dtype=float)
    return {
        "subset": subset,
        "domain": str(group["domain"].iloc[0]),
        "model": str(group["model"].iloc[0]),
        "n": len(group),
        "MAE": float(np.abs(error).mean()),
        "RMSE": float(np.sqrt(np.square(error).mean())),
        "mean_signed_error": float(error.mean()),
        "n_overpredicted": int((error > 0).sum()),
        "n_underpredicted": int((error < 0).sum()),
        "n_tied": int((error == 0).sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    for path in (PREDICTIONS, METRICS, FLAGS):
        if not path.is_file():
            raise FileNotFoundError(path)

    predictions = pd.read_csv(PREDICTIONS)
    predictions = predictions.loc[predictions["target_cycle"].eq(2025)].copy()
    required = {"country_iso3", "domain", "model", "prediction", "true_score", "n_context"}
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"Prediction file lacks columns: {missing}")
    if predictions.duplicated(["target_cycle", "domain", "country_iso3", "model"]).any():
        raise ValueError("2025 prediction keys are not unique")
    if set(predictions["model"].unique()) != set(METHODS):
        raise ValueError("Unexpected or missing model names in saved forecasts")
    if not np.isfinite(predictions[["prediction", "true_score"]].to_numpy()).all():
        raise ValueError("Non-finite predictions or evaluation targets")

    flags = pd.read_csv(FLAGS, dtype={"iso3": str})
    flag_map = flags.set_index("iso3")["quality_flag"].astype(str).str.lower().eq("true").to_dict()
    predictions["quality_flag"] = (
        predictions["country_iso3"].map(flag_map).astype("boolean").fillna(False).astype(bool)
    )
    flagged = sorted(predictions.loc[predictions["quality_flag"], "country_iso3"].unique())
    if flagged != ["ALB", "CAN", "NLD", "NOR", "NZL", "USA"]:
        raise ValueError(f"Unexpected PISA 2025 quality-flag set: {flagged}")
    if predictions.groupby("domain")["country_iso3"].nunique().to_dict() != {d: 66 for d in DOMAINS}:
        raise ValueError("Expected the common 66-country sample for all three domains")
    if predictions.groupby(["domain", "country_iso3"])["n_context"].nunique().gt(1).any():
        raise ValueError("Context counts vary across methods for a country/domain")

    # Confirm the full-sample metrics reproduce the saved headline table.
    saved = pd.read_csv(METRICS)
    recomputed = pd.DataFrame(
        [metrics_for(group, "all_66") for _, group in predictions.groupby(["domain", "model"], sort=True)]
    )
    merged = recomputed.merge(saved, on=["domain", "model"], suffixes=("_new", "_saved"), validate="one_to_one")
    for metric in ("MAE", "RMSE", "mean_signed_error"):
        delta = (merged[f"{metric}_new"] - merged[f"{metric}_saved"]).abs().max()
        if delta > 1e-10:
            raise ValueError(f"Full-sample {metric} does not reproduce saved evaluation table: {delta}")

    subsets = {
        "all_66": np.ones(len(predictions), dtype=bool),
        "quality_clean_60": ~predictions["quality_flag"].to_numpy(),
        "exclude_albania_65": predictions["country_iso3"].ne("ALB").to_numpy(),
        "clean_exclude_albania_60": (
            ~predictions["quality_flag"] & predictions["country_iso3"].ne("ALB")
        ).to_numpy(),
    }
    clean_counts = (
        predictions.loc[subsets["quality_clean_60"], ["domain", "country_iso3"]]
        .drop_duplicates()
        .groupby("domain")["country_iso3"]
        .nunique()
        .to_dict()
    )
    if clean_counts != {domain: 60 for domain in DOMAINS}:
        raise ValueError("Expected 60 clean unique countries per domain")
    clean_no_albania_counts = (
        predictions.loc[subsets["clean_exclude_albania_60"], ["domain", "country_iso3"]]
        .drop_duplicates()
        .groupby("domain")["country_iso3"]
        .nunique()
        .to_dict()
    )
    if clean_no_albania_counts != {domain: 60 for domain in DOMAINS}:
        raise ValueError("Albania is quality-flagged; excluding it from clean set should leave n=60")
    sensitivity_rows = []
    for subset_name, mask in subsets.items():
        subset = predictions.loc[mask]
        for _, group in subset.groupby(["domain", "model"], sort=True):
            sensitivity_rows.append(metrics_for(group, subset_name))

    predictions["context_group"] = predictions["n_context"].map(
        lambda n: str(int(n)) if int(n) < 7 else "7_plus"
    )
    for (domain, context_group, model), group in predictions.groupby(
        ["domain", "context_group", "model"], sort=True
    ):
        sensitivity_rows.append(metrics_for(group, f"n_context_{context_group}"))

    # Resample countries jointly across the matched 2025 model forecasts.
    # The paired loss difference is model absolute error minus persistence AE.
    rng = np.random.default_rng(SEED)
    bootstrap_rows = []
    for domain in DOMAINS:
        domain_data = predictions.loc[predictions["domain"].eq(domain)]
        wide_abs = domain_data.assign(
            absolute_error=(domain_data["prediction"] - domain_data["true_score"]).abs()
        ).pivot(index="country_iso3", columns="model", values="absolute_error")
        if set(wide_abs.columns) != set(METHODS) or wide_abs.isna().any().any() or len(wide_abs) != 66:
            raise ValueError(f"Incomplete paired model errors for {domain}")
        persistence = wide_abs["Persistence"].to_numpy(dtype=float)
        comparisons = [(model, "Persistence") for model in METHODS if model != "Persistence"]
        comparisons.append(("TimesFM2_5_ILSA_LoRA", "TimesFM2_5_zero_shot"))
        for model, comparator in comparisons:
            paired_delta = wide_abs[model].to_numpy(dtype=float) - wide_abs[comparator].to_numpy(dtype=float)
            draws = np.empty(BOOTSTRAP_REPLICATES, dtype=float)
            for start in range(0, BOOTSTRAP_REPLICATES, 500):
                stop = min(start + 500, BOOTSTRAP_REPLICATES)
                sampled = rng.integers(0, len(paired_delta), size=(stop - start, len(paired_delta)))
                draws[start:stop] = paired_delta[sampled].mean(axis=1)
            bootstrap_rows.append(
                {
                    "domain": domain,
                    "model": model,
                    "comparator": comparator,
                    "n_countries": len(paired_delta),
                    "delta_MAE_model_minus_comparator": float(paired_delta.mean()),
                    "bootstrap_ci_2_5": float(np.quantile(draws, 0.025)),
                    "bootstrap_ci_97_5": float(np.quantile(draws, 0.975)),
                    "bootstrap_probability_delta_below_zero": float((draws < 0).mean()),
                    "replicates": BOOTSTRAP_REPLICATES,
                    "seed": SEED,
                }
            )

    output.mkdir(parents=True)
    pd.DataFrame(sensitivity_rows).to_csv(output / "subset_and_context_metrics.csv", index=False)
    pd.DataFrame(bootstrap_rows).to_csv(output / "paired_country_bootstrap.csv", index=False)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "role": "outcome-exposed retrospective sensitivity; no model fitting or tuning",
        "prediction_input": str(PREDICTIONS.relative_to(ROOT)),
        "prediction_sha256": sha256(PREDICTIONS),
        "headline_metrics_input": str(METRICS.relative_to(ROOT)),
        "headline_metrics_sha256": sha256(METRICS),
        "quality_flag_input": str(FLAGS.relative_to(ROOT)),
        "quality_flag_sha256": sha256(FLAGS),
        "script_sha256": sha256(Path(__file__).resolve()),
        "output_directory": str(output),
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
        "quality_flag_countries": flagged,
        "country_count_by_domain": predictions.groupby("domain")["country_iso3"].nunique().to_dict(),
        "recomputed_metric_max_abs_differences": {
            metric: float((merged[f"{metric}_new"] - merged[f"{metric}_saved"]).abs().max())
            for metric in ("MAE", "RMSE", "mean_signed_error")
        },
        "bootstrap": {
            "unit": "country; paired model losses within each domain",
            "comparators": ["Persistence", "TimesFM2_5_zero_shot for TimesFM2_5_ILSA_LoRA"],
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": SEED,
            "interval": "percentile 95% interval; exploratory, not a confirmatory significance test",
        },
        "outputs": ["subset_and_context_metrics.csv", "paired_country_bootstrap.csv"],
    }
    (output / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
