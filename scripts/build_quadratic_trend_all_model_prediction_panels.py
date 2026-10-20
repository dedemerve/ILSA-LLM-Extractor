#!/usr/bin/env python3
"""Build versioned 9-method historical and retrospective prediction panels."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
HIST_BASE = ROOT / "outputs/pisa_timesfm25_ils_adapter_v1/all_models_historical_predictions.csv"
HIST_LQ = ROOT / "outputs/pisa_tabular_benchmark_v2/historical_predictions.csv"
HIST_DIAG = ROOT / "outputs/pisa_tabular_benchmark_v2/trend_selection_diagnostics.csv"
P25_BASE = ROOT / "outputs/pisa_2025_retrospective_timesfm_v1/all_models_predictions.csv"
P25_TABULAR = ROOT / "outputs/pisa_2025_retrospective_tabular_v1/tabular_predictions.csv"
P25_LQ = ROOT / "outputs/pisa_2025_quadratic_trend_retrospective_v1/tabular_predictions.csv"
P25_DIAG = ROOT / "outputs/pisa_2025_quadratic_trend_retrospective_v1/trend_selection_diagnostics.csv"
OUT_DIR = ROOT / "outputs/pisa_all_models_quadratic_trend_v1"
KEYS = ["target_cycle", "domain", "country_iso3", "model"]
TARGET_KEYS = ["target_cycle", "domain", "country_iso3"]
LQ_MODEL = "LinearQuadraticTrend"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite_unique(data: pd.DataFrame, label: str) -> None:
    if data.duplicated(KEYS).any():
        raise ValueError(f"Duplicate prediction keys in {label}")
    if not np.isfinite(data[["prediction", "true_score"]].to_numpy(float)).all():
        raise ValueError(f"Non-finite forecasts/targets in {label}")


def attach_context(lq: pd.DataFrame, diagnostics_path: Path) -> pd.DataFrame:
    diagnostics = pd.read_csv(diagnostics_path, dtype={"country_iso3": str})
    if diagnostics.duplicated(TARGET_KEYS).any():
        raise ValueError(f"Duplicate diagnostic keys: {diagnostics_path}")
    diagnostic_context = diagnostics[TARGET_KEYS + ["context_cycles"]]
    if "context_cycles" in lq.columns:
        check = lq[TARGET_KEYS + ["context_cycles"]].merge(
            diagnostic_context, on=TARGET_KEYS, how="left", suffixes=("_pred", "_diag"), validate="one_to_one"
        )
        if check.context_cycles_diag.isna().any() or not check.context_cycles_pred.astype(str).equals(check.context_cycles_diag.astype(str)):
            raise ValueError(f"Context-cycle diagnostics disagree: {diagnostics_path}")
        out = lq.copy()
    else:
        out = lq.merge(diagnostic_context, on=TARGET_KEYS, how="left", validate="one_to_one")
    if out.context_cycles.isna().any():
        raise ValueError(f"Missing context cycles: {diagnostics_path}")
    return out


def add_model(base_path: Path, extra: pd.DataFrame, label: str) -> tuple[pd.DataFrame, dict]:
    base = pd.read_csv(base_path, dtype={"country_iso3": str})
    finite_unique(base, f"base {label}")
    if extra.model.ne(LQ_MODEL).any():
        raise ValueError(f"Unexpected model in extension for {label}")
    finite_unique(extra, f"new method {label}")
    persistence = base.loc[base.model.eq("Persistence"), TARGET_KEYS + ["true_score"]]
    check = extra[TARGET_KEYS + ["true_score"]].merge(
        persistence, on=TARGET_KEYS, how="outer", suffixes=("_new", "_base"), indicator=True, validate="one_to_one"
    )
    if not check._merge.eq("both").all() or not np.allclose(check.true_score_new, check.true_score_base, rtol=0, atol=1e-10):
        raise ValueError(f"New trend method target keys/scores differ from {label} benchmark")
    shared = sorted(set(base.model) & set(extra.model))
    if shared:
        raise ValueError(f"New method already exists in {label} base panel")
    for model in sorted(set(base.model) & {"Persistence", "LinearTrend", "Ridge", "RandomForest", "XGBoost"}):
        other = base.loc[base.model.eq(model), TARGET_KEYS + ["true_score", "prediction"]].copy()
        if label == "2025":
            reference = pd.read_csv(P25_TABULAR, dtype={"country_iso3": str})
        else:
            reference = pd.read_csv(HIST_LQ, dtype={"country_iso3": str})
        reference = reference.loc[reference.model.eq(model), TARGET_KEYS + ["true_score", "prediction"]].copy()
        other = other.sort_values(TARGET_KEYS).reset_index(drop=True)
        reference = reference.sort_values(TARGET_KEYS).reset_index(drop=True)
        if not other[TARGET_KEYS].equals(reference[TARGET_KEYS]) or not np.allclose(
            other[["true_score", "prediction"]], reference[["true_score", "prediction"]], rtol=0, atol=1e-10
        ):
            raise ValueError(f"Existing model differs between source panels: {label}/{model}")

    for column in base.columns:
        if column not in extra.columns:
            extra[column] = np.nan
    for column in extra.columns:
        if column not in base.columns:
            base[column] = np.nan
    combined = pd.concat([base, extra[base.columns]], ignore_index=True)
    finite_unique(combined, f"combined {label}")
    expected = 9 * (624 if label == "historical" else 198)
    if len(combined) != expected or combined.groupby("model").size().ne(624 if label == "historical" else 198).any():
        raise ValueError(f"Unexpected prediction row coverage for {label}: {len(combined)}")
    return combined, {"base_rows": len(base), "new_method_rows": len(extra), "combined_rows": len(combined),
                     "models": sorted(combined.model.unique().tolist()), "keys_unique": True,
                     "new_method_target_keys_match": True, "existing_tabular_methods_match": True}


def main() -> None:
    inputs = [HIST_BASE, HIST_LQ, HIST_DIAG, P25_BASE, P25_TABULAR, P25_LQ, P25_DIAG]
    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(path)
    if OUT_DIR.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {OUT_DIR}")

    hist_lq = pd.read_csv(HIST_LQ, dtype={"country_iso3": str})
    hist_lq = attach_context(hist_lq.loc[hist_lq.model.eq(LQ_MODEL)].copy(), HIST_DIAG)
    p25_lq = pd.read_csv(P25_LQ, dtype={"country_iso3": str})
    p25_lq = attach_context(p25_lq, P25_DIAG)
    hist, hist_checks = add_model(HIST_BASE, hist_lq, "historical")
    p25, p25_checks = add_model(P25_BASE, p25_lq, "2025")
    p25["evaluation_role"] = "outcome_exposed_retrospective_benchmark"
    hist["evaluation_role"] = "historical_temporal_validation"

    OUT_DIR.mkdir(parents=True, exist_ok=False)
    hist_path = OUT_DIR / "historical_all_models_predictions.csv"
    p25_path = OUT_DIR / "pisa2025_all_models_predictions.csv"
    hist.to_csv(hist_path, index=False, float_format="%.10g")
    p25.to_csv(p25_path, index=False, float_format="%.10g")
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve().relative_to(ROOT)),
        "script_sha256": sha256(Path(__file__).resolve()),
        "input_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in inputs},
        "models": sorted(hist.model.unique().tolist()),
        "historical": {**hist_checks, "folds": [2012, 2015, 2018, 2022], "domains": sorted(hist.domain.unique().tolist())},
        "pisa2025": {**p25_checks, "target_cycle": 2025, "role": "Outcome-exposed retrospective benchmark; not blind.",
                     "domains": sorted(p25.domain.unique().tolist())},
        "outputs": {hist_path.name: sha256(hist_path), p25_path.name: sha256(p25_path)},
        "overwrite_policy": "Output directory must not exist before execution.",
    }
    (OUT_DIR / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"historical": hist_checks, "pisa2025": p25_checks}, indent=2))


if __name__ == "__main__":
    main()
