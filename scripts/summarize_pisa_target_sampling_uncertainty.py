#!/usr/bin/env python3
"""Propagate published PISA target-score sampling SEs into saved forecast metrics.

This is a conditional target-measurement sensitivity analysis. It does not refit
models and it does not represent forecast, linking, or model-training uncertainty.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ESTIMATES = ROOT / "outputs/stage4/country_estimates.csv"
CROSSWALK = ROOT / "analysis/03_data_feasibility/audit/pisa_2000_reading_country_crosswalk.csv"
HIST = ROOT / "outputs/pisa_timesfm25_ils_adapter_v1/all_models_historical_predictions.csv"
PISA25 = ROOT / "outputs/pisa_2025_retrospective_timesfm_v1/all_models_predictions.csv"
OECD25 = ROOT / "outputs/pisa2025_oecd_mean_audit_v3/country_domain_comparison.csv"
OUT = ROOT / "outputs/pisa_target_sampling_uncertainty_v2"
DOMAINS = ("mathematics", "reading", "science")
N_DRAWS = 20_000
SEED = 20261015
Q = (0.025, 0.5, 0.975)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def check(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def quantile_row(values: np.ndarray, **keys: object) -> dict[str, object]:
    lo, med, hi = np.quantile(values, Q)
    return {**keys, "q025": float(lo), "median": float(med), "q975": float(hi)}


def metric_draws(frame: pd.DataFrame, actual_draws: np.ndarray, n_draws: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return per-model metric draws and paired MAE-difference draws."""
    targets = sorted(frame.target_cycle.unique().tolist())
    models = sorted(frame.model.unique().tolist())
    domains = sorted(frame.domain.unique().tolist())
    records: list[dict[str, object]] = []
    paired_records: list[dict[str, object]] = []
    winner_records: list[dict[str, object]] = []
    for domain in domains:
        domain_frame = frame.loc[frame.domain.eq(domain)]
        # Require identical country/fold keys across models by construction.
        keys = sorted(set(zip(domain_frame.target_cycle, domain_frame.country_iso3)))
        idx = pd.MultiIndex.from_tuples(keys, names=["target_cycle", "country_iso3"])
        truth_idx = pd.MultiIndex.from_frame(domain_frame.drop_duplicates(["target_cycle", "country_iso3"])[["target_cycle", "country_iso3"]])
        check(len(set(truth_idx)) == len(truth_idx), f"Duplicate targets in {domain}")
        truth_series = domain_frame.drop_duplicates(["target_cycle", "country_iso3"]).set_index(["target_cycle", "country_iso3"]).sort_index()
        check(truth_series.index.equals(idx), f"Unstable target key ordering in {domain}")
        ydraw = actual_draws[domain]
        pred_by_model: dict[str, np.ndarray] = {}
        for model in models:
            sub = domain_frame.loc[domain_frame.model.eq(model)].set_index(["target_cycle", "country_iso3"]).reindex(idx)
            check(sub.prediction.notna().all(), f"Missing predictions for {domain}/{model}")
            pred_by_model[model] = sub.prediction.to_numpy(float)
        if len(targets) > 1:
            fold_index = np.array([targets.index(t) for t, _ in keys], dtype=int)
            fold_mae: dict[str, np.ndarray] = {}
            for model in models:
                ae = np.abs(ydraw - pred_by_model[model][None, :])
                perfold = np.stack([ae[:, fold_index == i].mean(axis=1) for i in range(len(targets))], axis=1)
                fold_mae[model] = perfold.mean(axis=1)
                records.append(quantile_row(fold_mae[model], evaluation="historical_equal_fold", domain=domain, model=model,
                                            point_estimate=float(np.mean([np.abs(truth_series.true_score.reindex(
                                                [k for k in keys if k[0] == t]).to_numpy() -
                                                pred_by_model[model][fold_index == i]).mean() for i, t in enumerate(targets)])),
                                            n_targets=len(keys), n_folds=len(targets)))
            score_matrix = np.column_stack([fold_mae[model] for model in models])
            winners = np.argmin(score_matrix, axis=1)
            for j, model in enumerate(models):
                winner_records.append({"evaluation": "historical_equal_fold", "domain": domain, "model": model,
                                       "winner_probability": float(np.mean(winners == j)), "n_draws": n_draws,
                                       "n_folds": len(targets)})
            comparisons = [(m, "Persistence") for m in models if m != "Persistence"]
            if "TimesFM2_5_ILSA_LoRA" in models and "TimesFM2_5_zero_shot" in models:
                comparisons.append(("TimesFM2_5_ILSA_LoRA", "TimesFM2_5_zero_shot"))
            for model, comparator in comparisons:
                diff = fold_mae[model] - fold_mae[comparator]
                paired_records.append(quantile_row(diff, evaluation="historical_equal_fold", domain=domain,
                    model=model, comparator=comparator, probability_better=float(np.mean(diff < 0)),
                    point_delta=float(np.mean([np.mean(np.abs(truth_series.true_score.to_numpy()[fold_index == i] - pred_by_model[model][fold_index == i])) -
                                               np.mean(np.abs(truth_series.true_score.to_numpy()[fold_index == i] - pred_by_model[comparator][fold_index == i]))
                                               for i in range(len(targets))])), n_folds=len(targets)))
            # Per-fold metrics are also reported to preserve temporal detail.
            for i, target in enumerate(targets):
                mask = fold_index == i
                for model in models:
                    vals = np.abs(ydraw[:, mask] - pred_by_model[model][None, mask]).mean(axis=1)
                    records.append(quantile_row(vals, evaluation="historical_fold", domain=domain, model=model,
                                                target_cycle=int(target), n_targets=int(mask.sum())))
        else:
            preds = pred_by_model
            abs_draws = {m: np.abs(ydraw - preds[m][None, :]).mean(axis=1) for m in models}
            point_y = truth_series.true_score.to_numpy(float)
            for model in models:
                records.append(quantile_row(abs_draws[model], evaluation="pisa2025_all66", domain=domain, model=model,
                                            point_estimate=float(np.abs(point_y - preds[model]).mean()), n_targets=len(keys)))
            score_matrix = np.column_stack([abs_draws[model] for model in models])
            winners = np.argmin(score_matrix, axis=1)
            for j, model in enumerate(models):
                winner_records.append({"evaluation": "pisa2025_all66", "domain": domain, "model": model,
                                       "winner_probability": float(np.mean(winners == j)), "n_draws": n_draws,
                                       "n_folds": 1})
            for model in models:
                if model == "Persistence":
                    continue
                diff = abs_draws[model] - abs_draws["Persistence"]
                paired_records.append(quantile_row(diff, evaluation="pisa2025_all66", domain=domain,
                    model=model, comparator="Persistence", probability_better=float(np.mean(diff < 0)),
                    point_delta=float(np.abs(point_y - preds[model]).mean() - np.abs(point_y - preds["Persistence"]).mean()),
                    n_targets=len(keys)))
    return pd.DataFrame(records), pd.DataFrame(paired_records), pd.DataFrame(winner_records)


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {OUT}")
    estimates = pd.read_csv(ESTIMATES, dtype={"country_iso3": str})
    estimates = estimates.loc[estimates.program.eq("PISA")].copy()
    estimates["country_iso3"] = estimates.country_iso3.str.strip().str.upper()
    # PISA 2000 reading country identifiers in the estimate extract are ISO numeric codes.
    cw = pd.read_csv(CROSSWALK, dtype={"source_country_id": int, "canonical_alpha3": str})
    cw = cw.loc[cw.verification_status.eq("VERIFIED")]
    mapping = dict(zip(cw.source_country_id.astype(str).str.zfill(3), cw.canonical_alpha3))
    mask2000 = estimates.cycle.eq(2000)
    estimates.loc[mask2000, "country_iso3"] = estimates.loc[mask2000, "country_iso3"].map(mapping)
    check(estimates.loc[mask2000, "country_iso3"].notna().all(), "Unmapped PISA 2000 reading country codes")
    check(not estimates.duplicated(["cycle", "domain", "country_iso3"]).any(), "Duplicate PISA country estimates")
    check(estimates.loc[estimates.cycle.ne(2000), "country_iso3"].str.fullmatch(r"[A-Z]{3}").all(), "Unexpected non-ISO alpha3 country key")

    hist = pd.read_csv(HIST, dtype={"country_iso3": str})
    hist = hist.loc[hist.target_cycle.isin([2012, 2015, 2018, 2022])].copy()
    hist_se = estimates.loc[estimates.cycle.isin(hist.target_cycle.unique()) & estimates.domain.isin(DOMAINS),
                            ["cycle", "domain", "country_iso3", "mean", "se"]].rename(
                                columns={"cycle": "target_cycle", "mean": "estimate_mean", "se": "target_se"})
    hist = hist.merge(hist_se, on=["target_cycle", "domain", "country_iso3"], how="left", validate="many_to_one")
    check(hist.target_se.notna().all() and (hist.target_se > 0).all(), "Historical target SE join incomplete or invalid")
    check(np.allclose(hist.true_score, hist.estimate_mean, atol=0.00011, rtol=0), "Historical target scores disagree with country estimates")
    check(not hist.duplicated(["target_cycle", "domain", "country_iso3", "model"]).any(), "Duplicate historical prediction key")

    p25 = pd.read_csv(PISA25, dtype={"country_iso3": str})
    p25 = p25.loc[p25.target_cycle.eq(2025)].copy()
    oecd = pd.read_csv(OECD25, dtype={"country_iso3": str})
    oecd = oecd[["country_iso3", "domain", "oecd_mean_unrounded", "oecd_sampling_se"]]
    p25 = p25.merge(oecd, on=["country_iso3", "domain"], how="left", validate="many_to_one")
    check(p25.oecd_sampling_se.notna().all() and (p25.oecd_sampling_se > 0).all(), "Official PISA 2025 SE join incomplete")
    check(p25.groupby("domain").country_iso3.nunique().to_dict() == {d: 66 for d in DOMAINS}, "PISA 2025 expected 66-country common sample")
    # Saved panel means and the higher-precision OECD publication values differ by
    # at most 0.01155 points in the audited 198 country-domain rows.
    check(np.allclose(p25.true_score, p25.oecd_mean_unrounded, atol=0.012, rtol=0), "PISA 2025 targets disagree materially with official OECD means")
    check(not p25.duplicated(["target_cycle", "domain", "country_iso3", "model"]).any(), "Duplicate PISA 2025 prediction key")

    rng = np.random.default_rng(SEED)
    # metric_draws expects mappings keyed by domain and stable row ordering per evaluation;
    # use one evaluation at a time so draws align exactly with that domain's sorted keys.
    rows, paired, winners = [], [], []
    p25_for_uncertainty = p25.copy()
    p25_for_uncertainty["true_score"] = p25_for_uncertainty["oecd_mean_unrounded"]
    for label, frame, means_key in [
        ("historical", hist, "true_score"),
        ("pisa2025", p25_for_uncertainty.rename(columns={"oecd_sampling_se": "target_se"}), "true_score"),
    ]:
        local_map: dict[str, np.ndarray] = {}
        for domain in DOMAINS:
            sub = frame.loc[frame.domain.eq(domain)].drop_duplicates(["target_cycle", "country_iso3"]).sort_values(["target_cycle", "country_iso3"])
            mean_vals = sub[means_key].to_numpy(float)
            se_vals = sub["target_se"].to_numpy(float)
            local_map[domain] = rng.normal(mean_vals[None, :], se_vals[None, :], size=(N_DRAWS, len(sub)))
        r, p, w = metric_draws(frame, local_map, N_DRAWS)
        rows.append(r); paired.append(p); winners.append(w)
    metrics = pd.concat(rows, ignore_index=True)
    comparisons = pd.concat(paired, ignore_index=True)
    winner_probs = pd.concat(winners, ignore_index=True)
    # For transparency: target SE distribution and the historical/PISA25 join coverage.
    se_summary = pd.concat([
        hist[["target_cycle", "domain", "country_iso3", "target_se"]].drop_duplicates().assign(source="project_country_estimates"),
        p25[["target_cycle", "domain", "country_iso3", "oecd_sampling_se"]].drop_duplicates().rename(columns={"oecd_sampling_se":"target_se"}).assign(source="OECD_2025_published"),
    ], ignore_index=True).groupby(["source", "target_cycle", "domain"], as_index=False).agg(
        n_countries=("country_iso3", "nunique"), mean_se=("target_se", "mean"), median_se=("target_se", "median"), max_se=("target_se", "max"))

    OUT.mkdir(parents=True)
    metrics.to_csv(OUT / "metric_uncertainty_quantiles.csv", index=False)
    comparisons.to_csv(OUT / "paired_comparison_uncertainty.csv", index=False)
    winner_probs.to_csv(OUT / "model_winner_probability.csv", index=False)
    se_summary.to_csv(OUT / "target_se_join_summary.csv", index=False)
    manifest = {
        "analysis": "PISA target score sampling-SE sensitivity using saved model predictions",
        "interpretation": "conditional sampling-error-only sensitivity; models and predictions fixed",
        "draw_model": "independent country-domain normal draws centered at published country mean with published sampling SE; each drawn target shared across all models",
        "draws": N_DRAWS, "seed": SEED,
        "excluded_uncertainty": ["cross-cycle scale linking error and covariance", "model fit and hyperparameter uncertainty", "forecast/input uncertainty", "country score covariance across domains and cycles", "selection uncertainty"],
        "pisa2025_status": "outcome-exposed retrospective benchmark, not a blind forecast evaluation",
        "inputs": {str(p.relative_to(ROOT)): sha256(p) for p in [ESTIMATES, CROSSWALK, HIST, PISA25, OECD25]},
        "checks": {"historical_target_join_complete": True, "pisa2025_official_se_join_complete": True,
                   "saved_target_means_match_estimate_sources": True, "unique_prediction_keys": True,
                   "common_pisa2025_sample_66_each_domain": True},
        "outputs": ["metric_uncertainty_quantiles.csv", "paired_comparison_uncertainty.csv", "model_winner_probability.csv", "target_se_join_summary.csv"],
    }
    (OUT / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {len(metrics)} metric summaries and {len(comparisons)} paired comparisons to {OUT.relative_to(ROOT)}")
    print(comparisons.to_string(index=False, max_rows=30))


if __name__ == "__main__":
    main()
