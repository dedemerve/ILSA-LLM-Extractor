#!/usr/bin/env python3
"""Add a literature-motivated linear-vs-quadratic trend baseline.

All model choice and fitting for a country/domain/fold use only observations
strictly before the target cycle. Existing benchmark outputs are read-only;
this script writes a new versioned output directory.
"""

from __future__ import annotations

import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy.stats import f as f_distribution


ROOT = Path(__file__).resolve().parents[1]
PANEL_PATH = ROOT / "outputs/stage5/pisa_forecasting_panel_v2.csv"
PANEL_MANIFEST_PATH = ROOT / "outputs/stage5/pisa_forecasting_panel_v2_manifest.json"
FOLDS_PATH = ROOT / "analysis/03_data_feasibility/audit/pisa_historical_folds_v1.json"
BASELINE_DIR = ROOT / "outputs/pisa_tabular_benchmark_v1"
BASELINE_PREDICTIONS = BASELINE_DIR / "historical_predictions.csv"
BASELINE_METRICS = BASELINE_DIR / "historical_metrics.csv"
BASELINE_MANIFEST = BASELINE_DIR / "run_manifest.json"
OUT_DIR = ROOT / "outputs/pisa_tabular_benchmark_v2"
DOMAINS = ("mathematics", "reading", "science")
ALPHA = 0.05


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fit_polynomial(years: np.ndarray, scores: np.ndarray, target: int, degree: int) -> tuple[float, float]:
    """Return forecast and residual sum of squares using centered cycle years."""
    center = float(years.mean())
    x = years - center
    x_target = float(target - center)
    design = np.column_stack([x**power for power in range(degree + 1)])
    coefficients, _, rank, _ = np.linalg.lstsq(design, scores, rcond=None)
    if rank != degree + 1:
        raise ValueError(f"Rank-deficient degree-{degree} trend for years {years.tolist()}")
    residuals = scores - design @ coefficients
    target_row = np.asarray([x_target**power for power in range(degree + 1)])
    return float(target_row @ coefficients), float(residuals @ residuals)


def choose_trend(years: list[int], values: list[float], target: int) -> dict:
    """Test the nested quadratic term on pre-target observations only."""
    year_array = np.asarray(years, dtype=float)
    score_array = np.asarray(values, dtype=float)
    linear_forecast, linear_rss = fit_polynomial(year_array, score_array, target, 1)
    n = len(year_array)
    if n < 4:
        return {
            "prediction": linear_forecast,
            "selected_model": "linear_insufficient_cycles_for_quadratic_test",
            "linear_prediction": linear_forecast,
            "quadratic_prediction": np.nan,
            "linear_rss": linear_rss,
            "quadratic_rss": np.nan,
            "quadratic_f_statistic": np.nan,
            "quadratic_p_value": np.nan,
        }

    quadratic_forecast, quadratic_rss = fit_polynomial(year_array, score_array, target, 2)
    improvement = max(linear_rss - quadratic_rss, 0.0)
    tolerance = np.finfo(float).eps * max(float(score_array @ score_array), 1.0) * 100
    if quadratic_rss <= tolerance:
        f_statistic = float("inf") if improvement > tolerance else 0.0
        p_value = 0.0 if improvement > tolerance else 1.0
    else:
        f_statistic = float(improvement / (quadratic_rss / (n - 3)))
        p_value = float(f_distribution.sf(f_statistic, 1, n - 3))

    use_quadratic = p_value < ALPHA
    return {
        "prediction": quadratic_forecast if use_quadratic else linear_forecast,
        "selected_model": "quadratic" if use_quadratic else "linear",
        "linear_prediction": linear_forecast,
        "quadratic_prediction": quadratic_forecast,
        "linear_rss": linear_rss,
        "quadratic_rss": quadratic_rss,
        "quadratic_f_statistic": f_statistic,
        "quadratic_p_value": p_value,
    }


def main() -> None:
    required = (
        PANEL_PATH,
        PANEL_MANIFEST_PATH,
        FOLDS_PATH,
        BASELINE_PREDICTIONS,
        BASELINE_METRICS,
        BASELINE_MANIFEST,
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    if OUT_DIR.exists():
        raise FileExistsError(f"Refusing to overwrite existing experiment output: {OUT_DIR}")

    panel_manifest = json.loads(PANEL_MANIFEST_PATH.read_text(encoding="utf-8"))
    if panel_manifest["output_sha256"] != sha256(PANEL_PATH):
        raise ValueError("Panel hash does not match its build manifest")
    folds = json.loads(FOLDS_PATH.read_text(encoding="utf-8"))
    if folds["panel_sha256"] != sha256(PANEL_PATH):
        raise ValueError("Fold manifest was built from a different panel")
    if folds["pisa_2025"]["excluded_from_historical_fold_manifest"] is not True:
        raise ValueError("PISA 2025 must remain outside historical development folds")
    baseline_manifest = json.loads(BASELINE_MANIFEST.read_text(encoding="utf-8"))
    if baseline_manifest["panel_sha256"] != sha256(PANEL_PATH):
        raise ValueError("Existing benchmark uses a different panel")
    if baseline_manifest["fold_manifest_sha256"] != sha256(FOLDS_PATH):
        raise ValueError("Existing benchmark uses a different fold manifest")

    panel = pd.read_csv(PANEL_PATH, dtype={"country_iso3": str})
    panel = panel.loc[panel["program"].eq("PISA")].copy()
    panel["cycle"] = pd.to_numeric(panel["cycle"], errors="raise").astype(int)
    if panel.duplicated(["domain", "country_iso3", "cycle"]).any():
        raise ValueError("PISA panel has duplicate country/domain/cycle rows")
    base_predictions = pd.read_csv(BASELINE_PREDICTIONS)
    if base_predictions.duplicated(["target_cycle", "domain", "country_iso3", "model"]).any():
        raise ValueError("Existing benchmark prediction keys are not unique")

    new_predictions: list[dict] = []
    diagnostics: list[dict] = []
    for fold in folds["folds"]:
        target_cycle = int(fold["target_cycle"])
        common_countries = sorted(fold["common_three_domain_countries"])
        for domain in DOMAINS:
            score_col = f"PISA_{domain}"
            series = panel.loc[
                (panel["domain"] == domain) & panel[score_col].notna(),
                ["country_iso3", "cycle", score_col],
            ].rename(columns={score_col: "score"})
            for country in common_countries:
                country_series = series.loc[series["country_iso3"].eq(country)]
                context = country_series.loc[country_series["cycle"] < target_cycle].sort_values("cycle")
                target_rows = country_series.loc[country_series["cycle"].eq(target_cycle), "score"]
                if len(context) < 3 or len(target_rows) != 1:
                    raise ValueError(f"Fold mismatch: {country}/{domain}/{target_cycle}")
                years = context["cycle"].astype(int).tolist()
                scores = context["score"].astype(float).tolist()
                chosen = choose_trend(years, scores, target_cycle)
                truth = float(target_rows.iloc[0])
                prediction = float(chosen["prediction"])
                error = prediction - truth
                row = {
                    "target_cycle": target_cycle,
                    "domain": domain,
                    "country_iso3": country,
                    "model": "LinearQuadraticTrend",
                    "n_context": len(years),
                    "last_cycle": years[-1],
                    "true_score": truth,
                    "prediction": prediction,
                    "signed_error": error,
                    "absolute_error": abs(error),
                    "squared_error": error**2,
                    "train_examples": np.nan,
                }
                new_predictions.append(row)
                diagnostics.append(
                    {
                        **{k: row[k] for k in ("target_cycle", "domain", "country_iso3", "true_score")},
                        "context_cycles": ";".join(map(str, years)),
                        "n_context": len(years),
                        **{k: v for k, v in chosen.items() if k != "prediction"},
                        "selected_prediction": prediction,
                    }
                )

    extra = pd.DataFrame(new_predictions)
    diagnostic_df = pd.DataFrame(diagnostics)
    keys = ["target_cycle", "domain", "country_iso3"]
    old_persistence = base_predictions.loc[base_predictions["model"].eq("Persistence")]
    if len(old_persistence) != len(extra):
        raise ValueError(f"Expected {len(old_persistence)} forecast targets; produced {len(extra)}")
    check = extra.merge(
        old_persistence[keys + ["true_score", "n_context", "last_cycle"]],
        on=keys,
        how="outer",
        suffixes=("_new", "_old"),
        indicator=True,
        validate="one_to_one",
    )
    if not check["_merge"].eq("both").all():
        raise ValueError("New trend forecast targets do not match the locked historical benchmark")
    for col in ("true_score", "n_context", "last_cycle"):
        if not np.allclose(check[f"{col}_new"], check[f"{col}_old"], rtol=0, atol=1e-10):
            raise ValueError(f"New trend baseline differs from existing target metadata: {col}")

    combined = pd.concat([base_predictions, extra], ignore_index=True)
    if combined.duplicated(["target_cycle", "domain", "country_iso3", "model"]).any():
        raise ValueError("Combined benchmark prediction keys are not unique")
    metrics = (
        combined.groupby(["target_cycle", "domain", "model"], as_index=False)
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
    pred_path = OUT_DIR / "historical_predictions.csv"
    metric_path = OUT_DIR / "historical_metrics.csv"
    diagnostic_path = OUT_DIR / "trend_selection_diagnostics.csv"
    manifest_path = OUT_DIR / "run_manifest.json"
    combined.to_csv(pred_path, index=False)
    metrics.to_csv(metric_path, index=False, float_format="%.6f")
    diagnostic_df.to_csv(diagnostic_path, index=False, float_format="%.10g")
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "script": str(Path(__file__).resolve().relative_to(ROOT)),
        "script_sha256": sha256(Path(__file__).resolve()),
        "panel": str(PANEL_PATH.relative_to(ROOT)),
        "panel_sha256": sha256(PANEL_PATH),
        "fold_manifest": str(FOLDS_PATH.relative_to(ROOT)),
        "fold_manifest_sha256": sha256(FOLDS_PATH),
        "base_benchmark_predictions": str(BASELINE_PREDICTIONS.relative_to(ROOT)),
        "base_benchmark_predictions_sha256": sha256(BASELINE_PREDICTIONS),
        "base_benchmark_manifest_sha256": sha256(BASELINE_MANIFEST),
        "outer_folds": [int(fold["target_cycle"]) for fold in folds["folds"]],
        "domains": list(DOMAINS),
        "forecast_target_rule": "Same common-country/domain/target keys and observed values as the v1 historical benchmark.",
        "trend_method": {
            "candidates": ["linear", "quadratic"],
            "selection": "Nested F test of the quadratic term; select quadratic iff p < 0.05; otherwise linear.",
            "quadratic_minimum_context_cycles": 4,
            "three_cycle_rule": "Linear trend only; a quadratic through three observations is not used.",
            "fit_and_selection_cutoff": "Only same-country/domain scores with cycle < target_cycle.",
            "time_variable": "Calendar year, centered on the training-window mean for numerical stability.",
        },
        "literature_motivation": {
            "citation": "Becker & Coyle (2026), DOI 10.3389/feduc.2026.1734988",
            "publisher_url": "https://www.frontiersin.org/journals/education/articles/10.3389/feduc.2026.1734988/full",
            "scope_note": "Country-level PISA 2022 trend forecast precedent; corpus eligibility remains unresolved and does not alter the frozen 1,266-record frequencies.",
        },
        "pisa_2025_role": "Outcome-exposed retrospective benchmark only; not used in fit, selection or this run.",
        "checks": {
            "same_target_keys_as_v1": bool(check["_merge"].eq("both").all()),
            "same_observed_scores_as_v1": True,
            "all_trend_predictions_finite": bool(np.isfinite(extra["prediction"]).all()),
            "unique_prediction_keys": not bool(combined.duplicated(["target_cycle", "domain", "country_iso3", "model"]).any()),
            "prediction_rows": int(len(combined)),
            "trend_diagnostic_rows": int(len(diagnostic_df)),
            "selected_quadratic_count": int(diagnostic_df["selected_model"].eq("quadratic").sum()),
            "linear_due_to_insufficient_history_count": int(diagnostic_df["selected_model"].eq("linear_insufficient_cycles_for_quadratic_test").sum()),
        },
        "outputs": {
            pred_path.name: sha256(pred_path),
            metric_path.name: sha256(metric_path),
            diagnostic_path.name: sha256(diagnostic_path),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest["checks"], indent=2))
    print(f"Wrote {OUT_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
