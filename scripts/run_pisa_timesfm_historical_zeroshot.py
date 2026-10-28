#!/usr/bin/env python3
"""Evaluate TimesFM 3.0 zero-shot on the frozen historical PISA folds.

This is a country-level, one-assessment-step-ahead benchmark. PISA 2025 is
deliberately excluded because its outcomes have already been exposed in this
project. The run uses the corrected, hash-locked panel and CPU inference for
reproducibility on the current macOS environment.
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
import torch
from timesfm3 import ModelConfig, TimesFM3Evaluator


ROOT = Path(__file__).resolve().parents[1]
PANEL_PATH = ROOT / "outputs/stage5/pisa_forecasting_panel_v2.csv"
PANEL_MANIFEST_PATH = ROOT / "outputs/stage5/pisa_forecasting_panel_v2_manifest.json"
FOLDS_PATH = ROOT / "analysis/03_data_feasibility/audit/pisa_historical_folds_v1.json"
TABULAR_PATH = ROOT / "outputs/pisa_tabular_benchmark_v1/historical_predictions.csv"
OUT_DIR = ROOT / "outputs/pisa_timesfm_historical_zeroshot_v2"
DOMAINS = ("mathematics", "reading", "science")
MODEL_ID = "google/timesfm-3.0-pytorch"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def metric_rows(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (target_cycle, domain, model), group in predictions.groupby(
        ["target_cycle", "domain", "model"], sort=True
    ):
        error = group["prediction"].to_numpy(dtype=float) - group["true_score"].to_numpy(dtype=float)
        rows.append(
            {
                "target_cycle": int(target_cycle),
                "domain": domain,
                "model": model,
                "n": int(len(group)),
                "MAE": float(np.mean(np.abs(error))),
                "RMSE": float(np.sqrt(np.mean(np.square(error)))),
                "mean_signed_error": float(np.mean(error)),
                "n_overpredicted": int(np.sum(error > 0)),
                "n_underpredicted": int(np.sum(error < 0)),
                "n_tied": int(np.sum(error == 0)),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    if OUT_DIR.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {OUT_DIR}")
    for path in (PANEL_PATH, PANEL_MANIFEST_PATH, FOLDS_PATH, TABULAR_PATH):
        if not path.is_file():
            raise FileNotFoundError(path)

    panel_hash = sha256(PANEL_PATH)
    panel_manifest = json.loads(PANEL_MANIFEST_PATH.read_text(encoding="utf-8"))
    if panel_manifest["output_sha256"] != panel_hash:
        raise ValueError("Panel hash does not match the panel build manifest")
    folds = json.loads(FOLDS_PATH.read_text(encoding="utf-8"))
    if folds["panel_sha256"] != panel_hash:
        raise ValueError("Fold manifest references a different panel")
    if folds["pisa_2025"]["excluded_from_historical_fold_manifest"] is not True:
        raise ValueError("PISA 2025 must remain excluded from historical development")

    tabular_hash = sha256(TABULAR_PATH)
    panel = pd.read_csv(PANEL_PATH, dtype={"country_iso3": str})
    panel = panel.loc[panel["program"].eq("PISA")].copy()
    panel["cycle"] = pd.to_numeric(panel["cycle"], errors="raise").astype(int)

    # The local MPS backend is present but unsupported by this machine's OS.
    # CPU was smoke-checked against the cached checkpoint before this run.
    evaluator = TimesFM3Evaluator(
        ModelConfig(
            checkpoint_path=MODEL_ID,
            per_core_batch_size=16,
            device="cpu",
            local_files_only=True,
        )
    )

    predictions: list[dict] = []
    for fold in folds["folds"]:
        target_cycle = int(fold["target_cycle"])
        countries = sorted(fold["common_three_domain_countries"])
        for domain in DOMAINS:
            score_col = f"PISA_{domain}"
            series = panel.loc[
                panel["domain"].eq(domain) & panel[score_col].notna(),
                ["country_iso3", "cycle", score_col],
            ]
            contexts: list[np.ndarray] = []
            records: list[dict] = []
            for country in countries:
                country_rows = series.loc[series["country_iso3"].eq(country)].sort_values("cycle")
                context = country_rows.loc[country_rows["cycle"] < target_cycle]
                target = country_rows.loc[country_rows["cycle"].eq(target_cycle), score_col]
                if len(context) < 3 or len(target) != 1:
                    raise ValueError(
                        f"Fold/panel mismatch: {target_cycle}/{domain}/{country}; "
                        f"context={len(context)}, targets={len(target)}"
                    )
                years = context["cycle"].astype(int).tolist()
                values = context[score_col].astype(float).to_numpy()
                contexts.append(values.astype(np.float32))
                records.append(
                    {
                        "target_cycle": target_cycle,
                        "domain": domain,
                        "country_iso3": country,
                        "n_context": len(values),
                        "last_cycle": years[-1],
                        "true_score": float(target.iloc[0]),
                        "context_cycles": ";".join(map(str, years)),
                    }
                )

            outputs = list(
                evaluator.predict_batch(
                    contexts,
                    horizon=1,
                    return_quantiles=False,
                    use_symmetric_averaging=True,
                )
            )
            if len(outputs) != len(records):
                raise RuntimeError(
                    f"Prediction count mismatch for {target_cycle}/{domain}: "
                    f"expected {len(records)}, got {len(outputs)}"
                )
            for record, output in zip(records, outputs):
                forecast = np.asarray(output.forecast, dtype=float).reshape(-1)
                if forecast.size != 1 or not np.isfinite(forecast[0]):
                    raise ValueError(
                        f"Invalid one-step forecast: {target_cycle}/{domain}/"
                        f"{record['country_iso3']} -> {forecast}"
                    )
                prediction = float(forecast[0])
                error = prediction - record["true_score"]
                predictions.append(
                    {
                        **record,
                        "model": "TimesFM3_zero_shot",
                        "prediction": prediction,
                        "signed_error": error,
                        "absolute_error": abs(error),
                        "squared_error": error**2,
                    }
                )

    timesfm_predictions = pd.DataFrame(predictions)
    tabular_predictions = pd.read_csv(TABULAR_PATH)
    expected_key = ["target_cycle", "domain", "country_iso3"]
    if timesfm_predictions.duplicated(expected_key).any():
        raise ValueError("Duplicate TimesFM country-fold predictions")
    combined = pd.concat([tabular_predictions, timesfm_predictions], ignore_index=True, sort=False)
    if combined.duplicated(expected_key + ["model"]).any():
        raise ValueError("Duplicate model prediction keys after combining baselines")
    if not (combined.groupby(expected_key)["true_score"].nunique() == 1).all():
        raise ValueError("TimesFM and persistence targets differ on shared fold keys")

    metrics = metric_rows(combined)
    equal_fold_summary = (
        metrics.groupby(["domain", "model"], as_index=False)
        .agg(
            equal_fold_MAE=("MAE", "mean"),
            equal_fold_RMSE=("RMSE", "mean"),
            equal_fold_mean_signed_error=("mean_signed_error", "mean"),
            folds=("target_cycle", "nunique"),
        )
    )
    pooled_summary = (
        combined.assign(
            signed_error=lambda frame: frame["prediction"] - frame["true_score"],
            absolute_error=lambda frame: (frame["prediction"] - frame["true_score"]).abs(),
            squared_error=lambda frame: (frame["prediction"] - frame["true_score"]) ** 2,
        )
        .groupby(["domain", "model"], as_index=False)
        .agg(
            n=("country_iso3", "size"),
            pooled_MAE=("absolute_error", "mean"),
            pooled_RMSE=("squared_error", lambda values: float(np.sqrt(values.mean()))),
            pooled_mean_signed_error=("signed_error", "mean"),
            folds=("target_cycle", "nunique"),
        )
    )

    OUT_DIR.mkdir(parents=True)
    timesfm_predictions.to_csv(OUT_DIR / "timesfm_predictions.csv", index=False)
    combined.to_csv(OUT_DIR / "timesfm_vs_tabular_predictions.csv", index=False)
    metrics.to_csv(OUT_DIR / "historical_metrics.csv", index=False)
    equal_fold_summary.to_csv(OUT_DIR / "equal_fold_summary.csv", index=False)
    pooled_summary.to_csv(OUT_DIR / "pooled_summary.csv", index=False)

    cache_ref = Path.home() / ".cache/huggingface/hub/models--google--timesfm-3.0-pytorch/refs/main"
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "timesfm_distribution_version": importlib.metadata.version("timesfm"),
        "device": "cpu",
        "panel": str(PANEL_PATH.relative_to(ROOT)),
        "panel_sha256": panel_hash,
        "fold_manifest": str(FOLDS_PATH.relative_to(ROOT)),
        "fold_manifest_sha256": sha256(FOLDS_PATH),
        "tabular_predictions": str(TABULAR_PATH.relative_to(ROOT)),
        "tabular_predictions_sha256": tabular_hash,
        "checkpoint": MODEL_ID,
        "checkpoint_cached_revision": cache_ref.read_text().strip() if cache_ref.exists() else None,
        "folds": [int(fold["target_cycle"]) for fold in folds["folds"]],
        "domains": list(DOMAINS),
        "country_rule": "frozen common-country set across three domains within each fold",
        "minimum_prior_observations": 3,
        "forecast_horizon": "one assessment cycle; input values are ordered by cycle, calendar gaps are not explicitly modeled",
        "inference": {
            "return_quantiles": False,
            "use_symmetric_averaging": True,
            "local_files_only": True,
        },
        "pisa_2025_role": "outcome-exposed retrospective benchmark; excluded from this run and all model selection",
        "outputs": [
            "timesfm_predictions.csv",
            "timesfm_vs_tabular_predictions.csv",
            "historical_metrics.csv",
            "equal_fold_summary.csv",
            "pooled_summary.csv",
        ],
    }
    (OUT_DIR / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Predictions: {len(timesfm_predictions):,}")
    print(f"Metrics: {OUT_DIR / 'historical_metrics.csv'}")
    print(f"Manifest: {OUT_DIR / 'run_manifest.json'}")
    print("\nEqual-fold mean MAE:")
    print(equal_fold_summary[["domain", "model", "equal_fold_MAE"]].to_string(index=False))
    print("\nPooled MAE over country-fold predictions:")
    print(pooled_summary[["domain", "model", "pooled_MAE"]].to_string(index=False))


if __name__ == "__main__":
    main()
