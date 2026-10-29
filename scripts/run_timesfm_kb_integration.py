#!/usr/bin/env python3
"""Temporal pilot of literature-weighted covariate correction for TimesFM 2.5.

Uses saved, genuinely out-of-sample TimesFM 2.5 forecasts. The correction
models are trained only on earlier forecast cycles and predict TimesFM errors
from lagged ILSA-derived PISA covariates. This is an XReg-like integration
pilot, not the native TimesFM XReg implementation.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS = ROOT / "outputs/pisa_timesfm25_ils_adapter_v1/all_models_historical_predictions.csv"
PANEL = ROOT / "outputs/stage5/pisa_forecasting_panel_v2.csv"
WEIGHTS = ROOT / "outputs/stage5/predictor_weights_v2.csv"
OUT = ROOT / "outputs/timesfm_kb_integration_v3"
TARGET_CYCLE = 2022
FEATURES = ["lag_ESCS", "lag_HOMEPOS"]
MODEL_NAME = "TimesFM2_5_zero_shot"
RIDGE_ALPHA = 10.0
BOOTSTRAP_REPS = 5000
SEED = 20261025


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUT}")

    forecasts = pd.read_csv(PREDICTIONS)
    forecasts = forecasts.loc[
        forecasts["model"].eq(MODEL_NAME),
        ["target_cycle", "domain", "country_iso3", "true_score", "prediction"],
    ].copy()
    forecast_key = ["target_cycle", "domain", "country_iso3"]
    if forecasts.duplicated(forecast_key).any():
        raise ValueError("TimesFM zero-shot source has duplicate forecast keys")
    panel = pd.read_csv(PANEL)
    panel["pisa_target_score"] = np.nan
    for domain in ("mathematics", "reading", "science"):
        domain_rows = panel["domain"].eq(domain)
        panel.loc[domain_rows, "pisa_target_score"] = panel.loc[domain_rows, f"PISA_{domain}"]
    panel = panel.loc[
        panel["program"].eq("PISA"),
        ["cycle", "domain", "country_iso3", "pisa_target_score", *FEATURES],
    ].copy()
    if panel.duplicated(["cycle", "domain", "country_iso3"]).any():
        raise ValueError("PISA panel has duplicate cycle/domain/country keys")
    data = forecasts.merge(
        panel,
        left_on=["target_cycle", "domain", "country_iso3"],
        right_on=["cycle", "domain", "country_iso3"],
        how="left",
        validate="one_to_one",
    ).drop(columns="cycle")
    if data["pisa_target_score"].isna().any():
        raise ValueError("A source forecast did not match an observed PISA target score")
    score_error = (data["true_score"] - data["pisa_target_score"]).abs()
    if (score_error > 1e-6).any():
        raise ValueError(f"Saved forecast targets disagree with source panel (max diff={score_error.max()})")
    if data[FEATURES].isna().any(axis=None):
        # Missing covariates are retained for reporting, never silently imputed.
        pass

    weight_table = pd.read_csv(WEIGHTS)
    weight_map = dict(zip(weight_table["feature_name"], weight_table["w_norm"]))
    feature_weight_map = {"lag_ESCS": float(weight_map["ESCS"]), "lag_HOMEPOS": float(weight_map["HOMEPOS"])}
    if any(not np.isfinite(w) or w <= 0 for w in feature_weight_map.values()):
        raise ValueError(f"Invalid corpus weights: {feature_weight_map}")

    # A prediction at t can use only earlier forecast errors and covariates
    # observed by t. The common 2022 test set is complete on both predictors.
    target = data.loc[data["target_cycle"].eq(TARGET_CYCLE)].copy()
    target = target.dropna(subset=FEATURES + ["true_score", "prediction"])
    train = data.loc[data["target_cycle"].lt(TARGET_CYCLE)].dropna(
        subset=FEATURES + ["true_score", "prediction"]
    )
    if target.empty or train.empty:
        raise ValueError("No complete temporal train/test observations")
    if not (train["target_cycle"] < TARGET_CYCLE).all():
        raise ValueError("Temporal leakage: training includes target or future cycle")

    outputs: list[pd.DataFrame] = []
    fit_diagnostics: list[dict] = []
    for domain in sorted(target["domain"].unique()):
        tr = train.loc[train["domain"].eq(domain)].copy()
        te = target.loc[target["domain"].eq(domain)].copy()
        if tr.empty or te.empty:
            continue
        tr["error_to_correct"] = tr["true_score"] - tr["prediction"]
        # The zero-feature control is the training-period mean error correction.
        correction_bias = float(tr["error_to_correct"].mean())
        predictions = {
            "TimesFM2_5_zero_shot": te["prediction"].to_numpy(),
            "M0_bias_correction": te["prediction"].to_numpy() + correction_bias,
        }
        scaler = StandardScaler().fit(tr[FEATURES])
        x_train = scaler.transform(tr[FEATURES])
        x_test = scaler.transform(te[FEATURES])
        y_train = tr["error_to_correct"].to_numpy()
        unweighted = Ridge(alpha=RIDGE_ALPHA, fit_intercept=True).fit(x_train, y_train)
        weights = np.sqrt(np.array([feature_weight_map[c] for c in FEATURES]))
        weighted = Ridge(alpha=RIDGE_ALPHA, fit_intercept=True).fit(x_train * weights, y_train)
        correction_unweighted = unweighted.predict(x_test)
        correction_weighted = weighted.predict(x_test * weights)
        predictions["M1_unweighted_covariate_correction"] = te["prediction"].to_numpy() + correction_unweighted
        predictions["M2_literature_weighted_correction"] = te["prediction"].to_numpy() + correction_weighted
        for model, values in predictions.items():
            part = te[["target_cycle", "domain", "country_iso3", "true_score"]].copy()
            part["model"] = model
            part["prediction"] = values
            part["absolute_error"] = (part["true_score"] - part["prediction"]).abs()
            part["squared_error"] = (part["true_score"] - part["prediction"]) ** 2
            outputs.append(part)
        fit_diagnostics.append({
            "target_cycle": TARGET_CYCLE,
            "domain": domain,
            "n_train": int(len(tr)),
            "train_cycles": sorted(map(int, tr["target_cycle"].unique())),
            "n_test_common": int(len(te)),
            "train_countries": int(tr["country_iso3"].nunique()),
            "test_countries": int(te["country_iso3"].nunique()),
            "bias_correction": correction_bias,
            "ridge_alpha_fixed": RIDGE_ALPHA,
            "features": FEATURES,
            "corpus_weights": feature_weight_map,
        })

    pred = pd.concat(outputs, ignore_index=True)
    # All methods are scored on identical country-domain target rows.
    row_counts = pred.groupby("model").size()
    if row_counts.nunique() != 1:
        raise ValueError(f"Unequal evaluation rows by arm: {row_counts.to_dict()}")
    key_sets = {
        model: set(map(tuple, group[["target_cycle", "domain", "country_iso3"]].to_numpy()))
        for model, group in pred.groupby("model")
    }
    if any(keys != next(iter(key_sets.values())) for keys in key_sets.values()):
        raise ValueError("Model arms were evaluated on different country-domain keys")
    pred["signed_error"] = pred["prediction"] - pred["true_score"]
    metrics = pred.groupby("model").agg(
        n=("absolute_error", "size"),
        countries=("country_iso3", "nunique"),
        MAE=("absolute_error", "mean"),
        RMSE=("squared_error", lambda x: float(np.sqrt(x.mean()))),
        mean_error=("signed_error", "mean"),
    ).reset_index()

    wide_abs = pred.pivot(index=["country_iso3", "domain"], columns="model", values="absolute_error")
    rng = np.random.default_rng(SEED)
    countries = sorted(wide_abs.index.get_level_values("country_iso3").unique())
    pairs = [("M0_bias_correction", "TimesFM2_5_zero_shot"),
             ("M1_unweighted_covariate_correction", "TimesFM2_5_zero_shot"),
             ("M2_literature_weighted_correction", "TimesFM2_5_zero_shot"),
             ("M2_literature_weighted_correction", "M1_unweighted_covariate_correction")]
    intervals = []
    for model, reference in pairs:
        point = float((wide_abs[reference] - wide_abs[model]).mean())
        by_country = (wide_abs[reference] - wide_abs[model]).groupby(level="country_iso3").mean()
        draws = np.empty(BOOTSTRAP_REPS)
        vals = by_country.to_numpy()
        for i in range(BOOTSTRAP_REPS):
            draws[i] = rng.choice(vals, size=len(vals), replace=True).mean()
        intervals.append({"model": model, "reference": reference,
                          "mean_MAE_improvement_reference_minus_model": point,
                          "country_cluster_bootstrap_95_low": float(np.quantile(draws, 0.025)),
                          "country_cluster_bootstrap_95_high": float(np.quantile(draws, 0.975)),
                          "n_country_clusters": int(len(countries)),
                          "bootstrap_replicates": BOOTSTRAP_REPS})

    OUT.mkdir(parents=True)
    pred.to_csv(OUT / "predictions.csv", index=False)
    metrics.to_csv(OUT / "metrics.csv", index=False)
    pd.DataFrame(intervals).to_csv(OUT / "paired_country_bootstrap.csv", index=False)
    pd.DataFrame(fit_diagnostics).to_json(OUT / "fit_diagnostics.json", orient="records", indent=2)
    missingness = data.groupby("target_cycle")[FEATURES].agg(lambda s: int(s.notna().sum())).reset_index()
    missingness.to_csv(OUT / "covariate_availability_by_cycle.csv", index=False)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "Temporal pilot of literature-weighted PISA covariate correction to TimesFM 2.5 zero-shot",
        "primary_target_cycle": TARGET_CYCLE,
        "baseline_predictions_source": str(PREDICTIONS.relative_to(ROOT)),
        "baseline_model": MODEL_NAME,
        "knowledge_features": FEATURES,
        "corpus_feature_weights": feature_weight_map,
        "training_rule": "Within domain, only earlier target cycles with complete lagged covariates; correct prior TimesFM forecast errors.",
        "test_rule": "2022 target cycle, common complete-case country/domain rows for all methods.",
        "target_alignment_check": "Every stored forecast target score was matched against the PISA source panel with tolerance 1e-6.",
        "control_models": ["TimesFM2_5_zero_shot", "M0_bias_correction", "M1_unweighted_covariate_correction", "M2_literature_weighted_correction"],
        "xreg_relationship": "XReg-like residual correction around saved TimesFM forecasts; not native forecast_with_covariates() because the official PyTorch checkpoint is not locally cached and network access is unavailable.",
        "feature_semantics": "lag_ESCS and lag_HOMEPOS in the target-cycle row represent the previous PISA cycle, per scripts/run_loco_forecasting.py.",
        "uncertainty": "Paired bootstrap resampling country clusters; domains remain nested within country.",
        "limitations": ["Only one held-out target cycle (2022).", "Historical feature completeness restricts calibration training to earlier complete rows; inspect fit_diagnostics.json.", "Not evidence of a causal effect or performance across all ILSA programs/domains."],
        "inputs": {str(p.relative_to(ROOT)): sha256(p) for p in [PREDICTIONS, PANEL, WEIGHTS]},
        "python": __import__("platform").python_version(),
    }
    (OUT / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(metrics.to_string(index=False))
    print("\nPaired country bootstrap:")
    print(pd.DataFrame(intervals).to_string(index=False))
    print(f"\nOutputs: {OUT}")


if __name__ == "__main__":
    main()
