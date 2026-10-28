#!/usr/bin/env python3
"""Build matched, paired evidence tables for RQ1 and RQ2 from saved forecasts.

The script does not refit models or alter source outputs. RQ1 uses common
country/domain/target rows within each program-domain group. PISA's TimesFM
arms are compared on their registered four historical folds; other programs
use the common-fold all-program tabular benchmark. RQ2 uses paired LOCO M0/M1
and persistence predictions, with country-cluster bootstrap intervals.
"""
from __future__ import annotations

import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "rq_evidence_v6"
TABULAR = ROOT / "outputs/ilsa_tabular_benchmark_v1/predictions.csv"
PISA_TIMESFM = ROOT / "outputs/pisa_timesfm25_ils_adapter_v1/all_models_historical_predictions.csv"
LOCO = ROOT / "outputs/stage5/loco_predictions.csv"
METHOD_FREQ = ROOT / "outputs/forecast_synthesis_v2/method_frequency.csv"
LITERATURE_AUDIT = ROOT / "analysis/03_data_feasibility/audit/ilsa_ml_method_frequency_v1.csv"
BOOTSTRAPS = 5000
SEED = 20261025

PISA_MODEL_MAP = {
    "TimesFM2_5_zero_shot": "TimesFM_v25_zeroshot",
    "TimesFM2_5_ILSA_LoRA": "TimesFM_v25_LoRA",
    "TimesFM3_zero_shot": "TimesFM_v30_zeroshot",
}
PISA_MODELS = ["Persistence", "LinearTrend", "Ridge", "RandomForest", "XGBoost", *PISA_MODEL_MAP.values()]
OTHER_MODELS = ["Persistence", "LinearTrend", "Ridge", "RandomForest", "XGBoost"]
ACHIEVEMENT_DOMAINS = {
    "ICCS": {"civic_knowledge"},
    "ICILS": {"computer_literacy"},
    "PIRLS": {"reading"},
    "PISA": {"mathematics", "reading", "science"},
    "TIMSS": {"mathematics", "science"},
    "TIMSS_G4": {"mathematics", "science"},
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_rq1_predictions() -> pd.DataFrame:
    tab = pd.read_csv(TABULAR)
    tab = tab.rename(columns={"test_cycle": "target_cycle"})
    tab = tab[["program", "domain", "target_cycle", "country_iso3", "model", "true_score", "prediction"]]

    pisa = pd.read_csv(PISA_TIMESFM)
    pisa = pisa.rename(columns={"true_score": "true_score"})
    pisa["program"] = "PISA"
    pisa["model"] = pisa["model"].replace(PISA_MODEL_MAP)
    pisa = pisa[pisa["model"].isin(PISA_MODELS)]
    pisa = pisa[["program", "domain", "target_cycle", "country_iso3", "model", "true_score", "prediction"]]

    # Use the registered matched-fold PISA panel for all PISA methods so TimesFM
    # is evaluated on exactly the same target cycles and countries as RF/XGBoost.
    non_pisa = tab[tab["program"] != "PISA"]
    return pd.concat([non_pisa, pisa], ignore_index=True)


def matched_wide(group: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    g = group[group["model"].isin(models)].copy()
    dup = g.duplicated(["target_cycle", "country_iso3", "model"])
    if dup.any():
        raise ValueError(f"Duplicate prediction keys: {g.loc[dup].head().to_dict('records')}")
    pred = g.pivot(index=["target_cycle", "country_iso3"], columns="model", values="prediction")
    truth = g.groupby(["target_cycle", "country_iso3"], sort=False)["true_score"].nunique(dropna=True)
    if (truth > 1).any():
        raise ValueError("Conflicting true scores within a target/country key")
    y = g.groupby(["target_cycle", "country_iso3"], sort=False)["true_score"].first().rename("true_score")
    wide = pred.join(y)
    missing = set(models) - set(wide.columns)
    if missing:
        raise ValueError(f"Expected models missing from group: {sorted(missing)}")
    return wide.dropna(subset=[*models, "true_score"]).reset_index()


def fold_metric_rows(wide: pd.DataFrame, program: str, domain: str, models: list[str]) -> list[dict]:
    rows = []
    for cycle, fold in wide.groupby("target_cycle", sort=True):
        for model in models:
            err = fold[model].to_numpy(float) - fold["true_score"].to_numpy(float)
            rows.append({
                "program": program, "domain": domain, "target_cycle": int(cycle),
                "model": model, "n_countries": int(fold["country_iso3"].nunique()),
                "MAE": float(np.mean(np.abs(err))), "RMSE": float(np.sqrt(np.mean(err**2))),
            })
    return rows


def cluster_bootstrap_delta(wide: pd.DataFrame, model: str, reference: str,
                            rng: np.random.Generator, n_boot: int = BOOTSTRAPS) -> tuple[float, float, float]:
    """Equal-fold MAE(reference)-MAE(model), resampling whole countries."""
    countries = np.array(sorted(wide["country_iso3"].unique()))
    by_fold = []
    for _, fold in wide.groupby("target_cycle", sort=True):
        idx = fold["country_iso3"].map({c: i for i, c in enumerate(countries)}).to_numpy(int)
        diff = np.abs(fold[reference].to_numpy(float) - fold["true_score"].to_numpy(float)) - np.abs(
            fold[model].to_numpy(float) - fold["true_score"].to_numpy(float)
        )
        by_fold.append((idx, diff))
    point = float(np.mean([float(np.mean(d)) for _, d in by_fold]))
    draws = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        # Redraw only if a sparse fold receives zero sampled country clusters.
        for _ in range(1000):
            sampled = rng.integers(0, len(countries), size=len(countries))
            counts = np.bincount(sampled, minlength=len(countries))
            if all(counts[idx].sum() > 0 for idx, _ in by_fold):
                break
        else:
            raise RuntimeError("Could not draw country clusters covering every target fold")
        fold_means = [float(np.sum(counts[idx] * diff) / counts[idx].sum())
                      for idx, diff in by_fold]
        draws[b] = np.mean(fold_means)
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return point, float(lo), float(hi)


def build_rq1() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    raw = load_rq1_predictions()
    model_rows, fold_rows, contrast_rows = [], [], []
    rng = np.random.default_rng(SEED)
    common_summary = []
    for (program, domain), group in raw.groupby(["program", "domain"], sort=True):
        models = PISA_MODELS if program == "PISA" else OTHER_MODELS
        wide = matched_wide(group, models)
        if wide.empty:
            continue
        folds = fold_metric_rows(wide, program, domain, models)
        fold_rows.extend(folds)
        summary = (pd.DataFrame(folds).groupby("model", as_index=False)
                   .agg(equal_fold_MAE=("MAE", "mean"),
                        equal_fold_RMSE=("RMSE", lambda x: float(np.sqrt(np.mean(np.square(x))))),
                        n_folds=("target_cycle", "nunique")))
        for row in summary.to_dict("records"):
            model = row["model"]
            model_rows.append({
                "program": program, "domain": domain, **row,
                "n_predictions": int(len(wide)),
                "n_countries": int(wide["country_iso3"].nunique()),
                "target_cycles": ";".join(map(str, sorted(wide["target_cycle"].unique()))),
            })
        common_summary.append({"program": program, "domain": domain,
                               "n_folds": int(wide["target_cycle"].nunique()),
                               "n_predictions": len(wide), "n_countries": wide["country_iso3"].nunique(),
                               "target_cycles": ";".join(map(str, sorted(wide["target_cycle"].unique())))})
        references = ["Persistence", "LinearTrend"]
        for model in models:
            for reference in references:
                if model == reference:
                    continue
                # Primary contrasts: RF/XGBoost vs both classical time-series
                # baselines; TimesFM arms vs persistence. Other rows are retained
                # as descriptive paired comparisons.
                if model not in {"RandomForest", "XGBoost", *PISA_MODEL_MAP.values()}:
                    continue
                delta, lo, hi = cluster_bootstrap_delta(wide, model, reference, rng)
                contrast_rows.append({
                    "program": program, "domain": domain, "model": model,
                    "reference_model": reference,
                    "delta_reference_minus_model_MAE": delta,
                    "cluster_bootstrap_95pct_low": lo,
                    "cluster_bootstrap_95pct_high": hi,
                    "direction": "model lower MAE" if delta > 0 else ("tie" if delta == 0 else "reference lower MAE"),
                    "n_country_clusters": int(wide["country_iso3"].nunique()),
                    "n_folds": int(wide["target_cycle"].nunique()),
                    "bootstrap_replicates": BOOTSTRAPS,
                })
    models_df = pd.DataFrame(model_rows)
    models_df["rank_in_program_domain"] = models_df.groupby(["program", "domain"])["equal_fold_MAE"].rank(method="min").astype(int)
    return (models_df.sort_values(["program", "domain", "rank_in_program_domain"]),
            pd.DataFrame(fold_rows).sort_values(["program", "domain", "target_cycle", "model"]),
            pd.DataFrame(contrast_rows).sort_values(["program", "domain", "model", "reference_model"]),
            {"common_test_panels": common_summary})


def build_rq2() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    df = pd.read_csv(LOCO)
    df = df.rename(columns={"cycle": "target_cycle", "y_true": "true_score",
                            "y_M0": "M0", "y_M1": "M1", "y_M2": "Persistence"})
    df = df[df.apply(lambda r: r["domain"] in ACHIEVEMENT_DOMAINS.get(r["program"], set()), axis=1)].copy()
    rows, contrast_rows = [], []
    rng = np.random.default_rng(SEED + 1)
    for (program, domain), group in df.groupby(["program", "domain"], sort=True):
        wide = group[["target_cycle", "country_iso3", "true_score", "M0", "M1", "Persistence"]]
        if wide.duplicated(["target_cycle", "country_iso3"]).any():
            raise ValueError(f"Duplicate LOCO keys in {program}/{domain}")
        contrast_sets = [
            ("M1", "M0", "KB_weighted_M1_vs_unweighted_M0", wide.dropna(subset=["true_score", "M0", "M1"])),
            ("M1", "Persistence", "KB_weighted_M1_vs_persistence", wide.dropna(subset=["true_score", "M1", "Persistence"])),
        ]
        for model, reference, contrast, paired in contrast_sets:
            if paired.empty:
                continue
            for cycle, fold in paired.groupby("target_cycle", sort=True):
                err_model = fold[model].to_numpy(float) - fold["true_score"].to_numpy(float)
                err_ref = fold[reference].to_numpy(float) - fold["true_score"].to_numpy(float)
                e_model = np.abs(err_model)
                e_ref = np.abs(err_ref)
                rows.append({"program": program, "domain": domain, "target_cycle": int(cycle),
                             "contrast": contrast, "reference_model": reference,
                             "n_countries": int(fold["country_iso3"].nunique()),
                             "MAE_reference": float(e_ref.mean()), "MAE_M1": float(e_model.mean()),
                             "delta_reference_minus_M1_MAE": float(e_ref.mean() - e_model.mean()),
                             "RMSE_reference": float(np.sqrt(np.mean(err_ref**2))),
                             "RMSE_M1": float(np.sqrt(np.mean(err_model**2))),
                             "delta_reference_minus_M1_RMSE": float(np.sqrt(np.mean(err_ref**2)) - np.sqrt(np.mean(err_model**2)))})
            delta, lo, hi = cluster_bootstrap_delta(paired, model, reference, rng)
            contrast_rows.append({"program": program, "domain": domain, "contrast": contrast,
                                  "delta_reference_minus_M1_MAE": delta,
                                  "cluster_bootstrap_95pct_low": lo,
                                  "cluster_bootstrap_95pct_high": hi,
                                  "direction": "M1 lower MAE" if delta > 0 else ("tie" if delta == 0 else "reference lower MAE"),
                                  "n_country_clusters": int(paired["country_iso3"].nunique()),
                                  "n_folds": int(paired["target_cycle"].nunique()),
                                  "bootstrap_replicates": BOOTSTRAPS})
    folds = pd.DataFrame(rows)
    pairs = pd.DataFrame(contrast_rows)
    group_summary = []
    for (program, domain), group in folds.groupby(["program", "domain"], sort=True):
        row = {"program": program, "domain": domain}
        for contrast, suffix in [("KB_weighted_M1_vs_unweighted_M0", "M0"),
                                 ("KB_weighted_M1_vs_persistence", "persistence")]:
            g = group[group["contrast"] == contrast]
            if g.empty:
                continue
            rmse_ref = float(np.sqrt(np.mean(np.square(g["RMSE_reference"]))))
            rmse_m1 = float(np.sqrt(np.mean(np.square(g["RMSE_M1"]))))
            row.update({f"n_folds_{suffix}": int(g["target_cycle"].nunique()),
                        f"n_predictions_{suffix}": int(g["n_countries"].sum()),
                        f"equal_fold_MAE_reference_{suffix}": float(g["MAE_reference"].mean()),
                        f"equal_fold_MAE_M1_{suffix}": float(g["MAE_M1"].mean()),
                        f"delta_MAE_reference_minus_M1_{suffix}": float(g["delta_reference_minus_M1_MAE"].mean()),
                        f"equal_fold_RMSE_reference_{suffix}": rmse_ref,
                        f"equal_fold_RMSE_M1_{suffix}": rmse_m1,
                        f"delta_RMSE_reference_minus_M1_{suffix}": rmse_ref - rmse_m1})
            if suffix == "M0":
                row["M1_improves_over_M0"] = bool(g["delta_reference_minus_M1_MAE"].mean() > 0)
                ref_mae = float(g["MAE_reference"].mean())
                row["pct_MAE_improvement_M1_vs_M0"] = (100 * float(g["delta_reference_minus_M1_MAE"].mean()) / ref_mae) if ref_mae else None
        group_summary.append(row)
    return pd.DataFrame(group_summary), folds, pairs


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {OUT}")
    OUT.mkdir(parents=True, exist_ok=False)
    rq1, rq1_folds, rq1_contrasts, rq1_meta = build_rq1()
    rq2, rq2_folds, rq2_contrasts = build_rq2()
    rq1.to_csv(OUT / "rq1_matched_model_metrics.csv", index=False, float_format="%.6f")
    rq1_folds.to_csv(OUT / "rq1_fold_metrics.csv", index=False, float_format="%.6f")
    rq1_contrasts.to_csv(OUT / "rq1_paired_country_bootstrap.csv", index=False, float_format="%.6f")
    rq2.to_csv(OUT / "rq2_model_summary.csv", index=False, float_format="%.6f")
    rq2_folds.to_csv(OUT / "rq2_fold_metrics.csv", index=False, float_format="%.6f")
    rq2_contrasts.to_csv(OUT / "rq2_paired_country_bootstrap.csv", index=False, float_format="%.6f")

    freq = pd.read_csv(METHOD_FREQ)
    audit = pd.read_csv(LITERATURE_AUDIT)
    narrow = audit[(audit["field"] == "ml_primary") &
                   (audit["scope"] == "country_level_prediction_task_D")].copy()
    narrow = narrow.sort_values(["study_count", "method"], ascending=[False, True])
    summary = {
        "rq1_literature_frequency": {
            "coded_predictive_primary_method_total": int(freq["study_count"].sum()),
            "top_methods": freq.head(5)[["method", "study_count", "share_pct"]].to_dict("records"),
            "narrow_country_level_task_D_total": int(narrow["scope_coded_n"].max()) if len(narrow) else 0,
            "narrow_country_level_task_D_top_methods": narrow.head(10)[["method", "study_count", "share_of_coded_scope"]].to_dict("records"),
            "scope_source": str(LITERATURE_AUDIT.relative_to(ROOT)),
            "scope_note": "The 147-study broad predictive-method ranking and the 17-record country-level task-D ranking are distinct coded scopes; neither proves an exhaustive literature search.",
        },
        "rq1_matched_evaluation": rq1_meta,
        "rq2_weighted_model_result": {
            "program_domain_pairs": int(len(rq2)),
            "M1_improves_over_M0_pairs": int(rq2["M1_improves_over_M0"].fillna(False).sum()) if len(rq2) else 0,
            "mean_equal_fold_MAE_improvement_where_positive": float(rq2.loc[rq2["delta_MAE_reference_minus_M1_M0"] > 0, "delta_MAE_reference_minus_M1_M0"].mean()) if (rq2["delta_MAE_reference_minus_M1_M0"] > 0).any() else None,
            "integration_scope": "Literature-weighted Ridge (M1) with ILSA-derived lagged covariates, compared with unweighted Ridge (M0) and Persistence.",
            "interpretation": "Predictive association under the existing LOCO protocol; the intervention is limited to weighted Ridge and does not establish improved performance for all time-series architectures.",
        },
        "limitations": [
            "PISA TimesFM comparison is limited to the registered historical targets 2012, 2015, 2018, and 2022.",
            "PIAAC has only two cycles and is excluded from the multi-origin benchmark; TALIS has no achievement-score outcome.",
            "Country-cluster bootstrap intervals condition on observed folds and model predictions; they do not propagate score/link uncertainty or model refitting uncertainty.",
            "The 2025 PISA evaluation is outcome-exposed and is not included in the historical primary comparison.",
            "The literature corpus is not proven exhaustive; frequency counts describe the coded corpus, not all published research.",
        ],
    }
    (OUT / "rq_evidence_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    inputs = [TABULAR, PISA_TIMESFM, LOCO, METHOD_FREQ, LITERATURE_AUDIT]
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(), "seed": SEED, "bootstrap_replicates": BOOTSTRAPS,
        "inputs": {str(p.relative_to(ROOT)): sha256(p) for p in inputs},
        "method": {
            "rq1": "Equal-fold MAE/RMSE on common complete country-target rows within each program-domain. Country-cluster bootstrap of paired MAE contrasts; PISA uses the matched registered TimesFM historical panel.",
            "rq2": "Existing LOCO M0/M1/Persistence predictions, complete-key comparison, equal weighting of target folds, paired country-cluster bootstrap.",
            "bootstrap_interval": "Percentile 95% interval from 5,000 resamples of country IDs with all observed target-fold rows retained per sampled country.",
        },
        "outputs": sorted(p.name for p in OUT.iterdir()),
        "source_scope": "No models refit; this script summarizes saved predictions. It does not alter existing source outputs.",
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"RQ1 matched program-domain groups: {rq1[['program','domain']].drop_duplicates().shape[0]}")
    print(f"RQ2 program-domain groups: {len(rq2)}; M1 improves over M0: {int(rq2['M1_improves_over_M0'].sum())}/{len(rq2)}")
    print(f"Saved: {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
