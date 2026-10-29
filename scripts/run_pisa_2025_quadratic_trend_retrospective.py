#!/usr/bin/env python3
"""Run the leakage-controlled linear/quadratic trend rule on PISA 2025.

This is an outcome-exposed retrospective benchmark. The trend fit and model
selection use PISA cycles strictly before 2025; the PISA 2025 target is attached
only for scoring and key validation.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from run_pisa_quadratic_trend_benchmark import (
    DOMAINS,
    PANEL_MANIFEST_PATH,
    PANEL_PATH,
    choose_trend,
    sha256,
)


ROOT = Path(__file__).resolve().parents[1]
TABULAR_P25 = ROOT / "outputs/pisa_2025_retrospective_tabular_v1/tabular_predictions.csv"
TABULAR_P25_MANIFEST = ROOT / "outputs/pisa_2025_retrospective_tabular_v1/run_manifest.json"
HISTORICAL_V2 = ROOT / "outputs/pisa_tabular_benchmark_v2/historical_predictions.csv"
OUT_DIR = ROOT / "outputs/pisa_2025_quadratic_trend_retrospective_v1"
TARGET_CYCLE = 2025


def main() -> None:
    for path in (PANEL_PATH, PANEL_MANIFEST_PATH, TABULAR_P25, TABULAR_P25_MANIFEST, HISTORICAL_V2):
        if not path.is_file():
            raise FileNotFoundError(path)
    if OUT_DIR.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {OUT_DIR}")
    panel_manifest = json.loads(PANEL_MANIFEST_PATH.read_text(encoding="utf-8"))
    if panel_manifest["output_sha256"] != sha256(PANEL_PATH):
        raise ValueError("Panel hash does not match its build manifest")
    p25_manifest = json.loads(TABULAR_P25_MANIFEST.read_text(encoding="utf-8"))
    if p25_manifest["panel_sha256"] != sha256(PANEL_PATH) or p25_manifest["target_cycle"] != TARGET_CYCLE:
        raise ValueError("PISA 2025 comparator manifest is not based on this panel/target")

    panel = pd.read_csv(PANEL_PATH, dtype={"country_iso3": str})
    panel = panel.loc[panel.program.eq("PISA")].copy()
    panel["cycle"] = pd.to_numeric(panel.cycle, errors="raise").astype(int)
    p25 = pd.read_csv(TABULAR_P25, dtype={"country_iso3": str})
    p25 = p25.loc[p25.target_cycle.eq(TARGET_CYCLE)].copy()
    expected = p25.loc[p25.model.eq("Persistence"), ["target_cycle", "domain", "country_iso3", "n_context", "last_cycle", "context_cycles", "true_score"]]
    if len(expected) != 66 * len(DOMAINS) or expected.duplicated(["target_cycle", "domain", "country_iso3"]).any():
        raise ValueError("Existing retrospective benchmark does not define 66 common unique country targets")
    if p25.groupby("domain").country_iso3.nunique().to_dict() != {d: 66 for d in DOMAINS}:
        raise ValueError("Existing PISA 2025 country set differs by domain")

    forecasts: list[dict] = []
    diagnostics: list[dict] = []
    for domain in DOMAINS:
        score_col = f"PISA_{domain}"
        series = panel.loc[(panel.domain.eq(domain)) & panel[score_col].notna(),
                           ["country_iso3", "cycle", score_col]].rename(columns={score_col: "score"})
        for base in expected.loc[expected.domain.eq(domain)].itertuples(index=False):
            history = series.loc[series.country_iso3.eq(base.country_iso3)].sort_values("cycle")
            context = history.loc[history.cycle.lt(TARGET_CYCLE)]
            observed_target = history.loc[history.cycle.eq(TARGET_CYCLE), "score"]
            if len(context) != int(base.n_context) or len(observed_target) != 1:
                raise ValueError(f"Context/target mismatch: {base.country_iso3}/{domain}")
            years = context.cycle.astype(int).tolist()
            values = context.score.astype(float).tolist()
            if ";".join(map(str, years)) != str(base.context_cycles):
                raise ValueError(f"Context-cycle mismatch: {base.country_iso3}/{domain}")
            if not np.isclose(float(observed_target.iloc[0]), float(base.true_score), rtol=0, atol=1e-10):
                raise ValueError(f"PISA 2025 score mismatch: {base.country_iso3}/{domain}")
            result = choose_trend(years, values, TARGET_CYCLE)
            prediction = float(result["prediction"])
            error = prediction - float(base.true_score)
            forecasts.append({
                "target_cycle": TARGET_CYCLE,
                "domain": domain,
                "country_iso3": base.country_iso3,
                "n_context": len(years),
                "last_cycle": years[-1],
                "context_cycles": ";".join(map(str, years)),
                "true_score": float(base.true_score),
                "model": "LinearQuadraticTrend",
                "prediction": prediction,
                "signed_error": error,
                "absolute_error": abs(error),
                "squared_error": error**2,
            })
            diagnostics.append({
                "target_cycle": TARGET_CYCLE,
                "domain": domain,
                "country_iso3": base.country_iso3,
                "n_context": len(years),
                "context_cycles": ";".join(map(str, years)),
                **{k: v for k, v in result.items() if k != "prediction"},
                "selected_prediction": prediction,
            })

    forecast_frame = pd.DataFrame(forecasts)
    diagnostic_frame = pd.DataFrame(diagnostics)
    if len(forecast_frame) != 198 or forecast_frame.duplicated(["target_cycle", "domain", "country_iso3", "model"]).any():
        raise ValueError("Expected 198 finite unique forecasts")
    if not np.isfinite(forecast_frame.prediction.to_numpy(float)).all():
        raise ValueError("Non-finite trend forecasts")
    metrics = forecast_frame.groupby(["target_cycle", "domain", "model"], as_index=False).agg(
        n=("country_iso3", "nunique"), MAE=("absolute_error", "mean"),
        RMSE=("squared_error", lambda x: float(np.sqrt(x.mean()))),
        mean_signed_error=("signed_error", "mean"),
    )

    OUT_DIR.mkdir(parents=True, exist_ok=False)
    pred_path = OUT_DIR / "tabular_predictions.csv"
    metrics_path = OUT_DIR / "tabular_metrics.csv"
    diag_path = OUT_DIR / "trend_selection_diagnostics.csv"
    manifest_path = OUT_DIR / "run_manifest.json"
    forecast_frame.to_csv(pred_path, index=False, float_format="%.10g")
    metrics.to_csv(metrics_path, index=False, float_format="%.8f")
    diagnostic_frame.to_csv(diag_path, index=False, float_format="%.10g")
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve().relative_to(ROOT)),
        "script_sha256": sha256(Path(__file__).resolve()),
        "trend_method_script": "scripts/run_pisa_quadratic_trend_benchmark.py",
        "trend_method_script_sha256": sha256(ROOT / "scripts/run_pisa_quadratic_trend_benchmark.py"),
        "panel": str(PANEL_PATH.relative_to(ROOT)),
        "panel_sha256": sha256(PANEL_PATH),
        "existing_2025_benchmark_predictions": str(TABULAR_P25.relative_to(ROOT)),
        "existing_2025_benchmark_predictions_sha256": sha256(TABULAR_P25),
        "historical_v2_predictions": str(HISTORICAL_V2.relative_to(ROOT)),
        "historical_v2_predictions_sha256": sha256(HISTORICAL_V2),
        "target_cycle": TARGET_CYCLE,
        "evaluation_role": "Outcome-exposed retrospective comparison; not a blind or prospective test.",
        "common_country_count": 66,
        "domains": list(DOMAINS),
        "fit_cutoff": "Only PISA score observations with cycle < 2025 are used for trend fitting and selection.",
        "selection_rule": "Nested F test; quadratic selected iff p < 0.05; otherwise linear. Same rule as historical v2.",
        "checks": {
            "same_2025_common_country_keys_as_existing_benchmark": True,
            "context_cycles_reproduced_from_panel": True,
            "2025_targets_match_existing_benchmark_and_panel": True,
            "finite_predictions": bool(np.isfinite(forecast_frame.prediction).all()),
            "unique_prediction_keys": not bool(forecast_frame.duplicated(["target_cycle", "domain", "country_iso3", "model"]).any()),
            "prediction_rows": int(len(forecast_frame)),
            "selected_quadratic_count": int(diagnostic_frame.selected_model.eq("quadratic").sum()),
            "linear_due_to_insufficient_history_count": int(diagnostic_frame.selected_model.eq("linear_insufficient_cycles_for_quadratic_test").sum()),
        },
        "outputs": {pred_path.name: sha256(pred_path), metrics_path.name: sha256(metrics_path), diag_path.name: sha256(diag_path)},
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(metrics.to_string(index=False))
    print(json.dumps(manifest["checks"], indent=2))


if __name__ == "__main__":
    main()
