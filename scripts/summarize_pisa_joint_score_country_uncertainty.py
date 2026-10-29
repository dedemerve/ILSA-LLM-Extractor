#!/usr/bin/env python3
"""Jointly propagate target-score sampling SE and country-cluster uncertainty.

Predictions stay fixed. The analysis resamples country clusters and draws one
plausible target mean per country/fold, shared by every model. Scale-link error,
model-fit uncertainty, and between-cycle uncertainty remain out of scope.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
HIST = ROOT / "outputs/pisa_timesfm25_ils_adapter_v1/all_models_historical_predictions.csv"
EST = ROOT / "outputs/stage4/country_estimates.csv"
CROSSWALK = ROOT / "analysis/03_data_feasibility/audit/pisa_2000_reading_country_crosswalk.csv"
P25 = ROOT / "outputs/pisa_2025_retrospective_timesfm_v1/all_models_predictions.csv"
OECD25 = ROOT / "outputs/pisa2025_oecd_mean_audit_v3/country_domain_comparison.csv"
FLAGS25 = ROOT / "analysis/03_data_feasibility/audit/pisa2025_se_validation.csv"
OUT = ROOT / "outputs/pisa_joint_score_country_uncertainty_v2"
DOMAINS = ("mathematics", "reading", "science")
N_DRAWS = 20_000
SEED = 20261017
REFERENCE = "Persistence"
LO = ("TimesFM2_5_ILSA_LoRA", "TimesFM2_5_zero_shot")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def qrow(values: np.ndarray, **attrs: object) -> dict[str, object]:
    lo, med, hi = np.quantile(values, [0.025, 0.5, 0.975])
    return {**attrs, "q025": float(lo), "median": float(med), "q975": float(hi)}


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def summarize(frame: pd.DataFrame, seed: int, evaluation_name: str | None = None) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    rng = np.random.default_rng(seed)
    metrics, comparisons, winners = [], [], []
    historical = frame.target_cycle.nunique() > 1
    models = sorted(frame.model.unique().tolist())
    require(REFERENCE in models, "Persistence reference model missing")
    if LO[0] in models and LO[1] in models:
        pairs = [(m, REFERENCE) for m in models if m != REFERENCE] + [LO]
    else:
        pairs = [(m, REFERENCE) for m in models if m != REFERENCE]

    for domain in DOMAINS:
        d = frame.loc[frame.domain.eq(domain)].copy()
        folds = sorted(d.target_cycle.unique().tolist())
        countries = sorted(d.country_iso3.unique().tolist())
        ci = {c: i for i, c in enumerate(countries)}
        n_c, n_f, n_m = len(countries), len(folds), len(models)
        pred = np.full((n_f, n_c, n_m), np.nan)
        obs = np.full((n_f, n_c), np.nan)
        se = np.full((n_f, n_c), np.nan)
        for fi, fold in enumerate(folds):
            f = d.loc[d.target_cycle.eq(fold)]
            ys = f.drop_duplicates("country_iso3").set_index("country_iso3")
            for country, row in ys.iterrows():
                obs[fi, ci[country]] = float(row.true_score)
                se[fi, ci[country]] = float(row.target_se)
            for mi, model in enumerate(models):
                pm = f.loc[f.model.eq(model)].set_index("country_iso3").prediction
                pred[fi, [ci[c] for c in pm.index], mi] = pm.to_numpy(float)
        eligible = np.isfinite(obs)
        require(np.isfinite(se[eligible]).all() and (se[eligible] > 0).all(), f"Invalid target SE: {domain}")
        require(np.isfinite(pred[eligible]).all(), f"Incomplete matched predictions: {domain}")

        # Shared country-cluster bootstrap across temporal folds.
        draws = rng.integers(0, n_c, size=(N_DRAWS, n_c))
        multiplicity = np.zeros((N_DRAWS, n_c), dtype=np.int16)
        rr = np.repeat(np.arange(N_DRAWS), n_c)
        np.add.at(multiplicity, (rr, draws.ravel()), 1)
        # Draw uncertain country means independently by fold/country; one draw is
        # shared across models and duplicated cluster copies within a replicate.
        y_draw = rng.normal(obs[None, :, :], se[None, :, :], size=(N_DRAWS, n_f, n_c))
        fold_mae = np.zeros((N_DRAWS, n_f, n_m), dtype=np.float64)
        for fi in range(n_f):
            idx = np.flatnonzero(eligible[fi])
            w = multiplicity[:, idx]
            den = w.sum(axis=1)
            require((den > 0).all(), f"Bootstrap omitted all eligible {domain}/{folds[fi]} countries")
            for mi in range(n_m):
                ae = np.abs(y_draw[:, fi, idx] - pred[fi, idx, mi][None, :])
                fold_mae[:, fi, mi] = (ae * w).sum(axis=1) / den
        aggregate = fold_mae.mean(axis=1)
        point_fold = np.array([
            np.abs(obs[fi, eligible[fi]][:, None] - pred[fi, eligible[fi], :]).mean(axis=0)
            for fi in range(n_f)
        ])
        point_aggregate = point_fold.mean(axis=0)
        evaluation = evaluation_name or ("historical_equal_fold_joint" if historical else "pisa2025_all66_joint")
        for mi, model in enumerate(models):
            metrics.append(qrow(aggregate[:, mi], evaluation=evaluation, domain=domain, model=model,
                                point_estimate=float(point_aggregate[mi]), n_countries=n_c,
                                n_targets=int(eligible.sum()), n_folds=n_f))
        best = np.argmin(aggregate, axis=1)
        for mi, model in enumerate(models):
            winners.append({"evaluation": evaluation, "domain": domain, "model": model,
                            "winner_probability": float(np.mean(best == mi)), "n_draws": N_DRAWS,
                            "n_country_clusters": n_c, "n_folds": n_f})
        pair_deltas = {}
        for model, comparator in pairs:
            im, ic = models.index(model), models.index(comparator)
            delta = aggregate[:, im] - aggregate[:, ic]
            pair_deltas[(model, comparator)] = delta
            point_delta = float(point_aggregate[im] - point_aggregate[ic])
            comparisons.append(qrow(delta, evaluation=evaluation, domain=domain, model=model,
                comparator=comparator, point_delta=point_delta, probability_better=float(np.mean(delta < 0)),
                n_country_clusters=n_c, n_folds=n_f))
    return metrics, comparisons, winners


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {OUT}")
    est = pd.read_csv(EST, dtype={"country_iso3": str})
    est = est.loc[est.program.eq("PISA")].copy()
    est["country_iso3"] = est.country_iso3.str.strip().str.upper()
    cw = pd.read_csv(CROSSWALK, dtype={"source_country_id": int, "canonical_alpha3": str})
    mapping = dict(zip(cw.loc[cw.verification_status.eq("VERIFIED"), "source_country_id"].astype(str).str.zfill(3),
                       cw.loc[cw.verification_status.eq("VERIFIED"), "canonical_alpha3"]))
    m2000 = est.cycle.eq(2000)
    est.loc[m2000, "country_iso3"] = est.loc[m2000, "country_iso3"].map(mapping)
    require(est.loc[m2000, "country_iso3"].notna().all(), "PISA 2000 crosswalk incomplete")

    hist = pd.read_csv(HIST, dtype={"country_iso3": str})
    hist = hist.loc[hist.target_cycle.isin([2012, 2015, 2018, 2022])].copy()
    hse = est.loc[est.cycle.isin(hist.target_cycle.unique()), ["cycle", "domain", "country_iso3", "se"]].rename(
        columns={"cycle": "target_cycle", "se": "target_se"})
    hist = hist.merge(hse, on=["target_cycle", "domain", "country_iso3"], how="left", validate="many_to_one")
    require(hist.target_se.notna().all() and (hist.target_se > 0).all(), "Historical score SE join incomplete")
    require(not hist.duplicated(["target_cycle", "domain", "country_iso3", "model"]).any(), "Duplicate historical prediction key")

    p25 = pd.read_csv(P25, dtype={"country_iso3": str})
    oecd = pd.read_csv(OECD25, dtype={"country_iso3": str})
    oecd = oecd[["country_iso3", "domain", "oecd_mean_unrounded", "oecd_sampling_se"]]
    p25 = p25.merge(oecd, on=["country_iso3", "domain"], how="left", validate="many_to_one")
    p25["target_se"] = p25.oecd_sampling_se
    p25["true_score"] = p25.oecd_mean_unrounded
    require(p25.target_se.notna().all() and (p25.target_se > 0).all(), "OECD 2025 score SE join incomplete")
    require(p25.groupby("domain").country_iso3.nunique().to_dict() == {d: 66 for d in DOMAINS}, "Expected common 66-country 2025 sample")
    require(not p25.duplicated(["target_cycle", "domain", "country_iso3", "model"]).any(), "Duplicate PISA 2025 prediction key")

    flags = pd.read_csv(FLAGS25, dtype={"iso3": str})
    flagged_codes = set(flags.loc[flags.quality_flag.astype(bool), "iso3"].str.strip().str.upper())
    p25["quality_flag"] = p25.country_iso3.isin(flagged_codes)
    p25_clean = p25.loc[~p25.quality_flag].copy()
    require(p25_clean.groupby("domain").country_iso3.nunique().to_dict() == {d: 60 for d in DOMAINS}, "Expected 60 quality-clean 2025 countries per domain")

    result_sets = [
        summarize(hist, SEED, "historical_equal_fold_joint"),
        summarize(p25, SEED + 1, "pisa2025_all66_joint"),
        summarize(p25_clean, SEED + 2, "pisa2025_quality_clean60_joint"),
    ]
    metrics = pd.DataFrame(sum((r[0] for r in result_sets), []))
    comparisons = pd.DataFrame(sum((r[1] for r in result_sets), []))
    winners = pd.DataFrame(sum((r[2] for r in result_sets), []))
    OUT.mkdir(parents=True)
    metrics.to_csv(OUT / "joint_metric_uncertainty.csv", index=False)
    comparisons.to_csv(OUT / "joint_paired_comparisons.csv", index=False)
    winners.to_csv(OUT / "joint_model_winner_probability.csv", index=False)
    manifest = {
        "analysis": "joint country-cluster resampling and target-score sampling-SE propagation, saved forecasts fixed",
        "draws": N_DRAWS, "seeds": {"historical": SEED, "pisa2025": SEED + 1},
        "historical_targets": [2012, 2015, 2018, 2022], "historical_folds_per_domain": 4,
        "pisa2025_role": "outcome-exposed retrospective benchmark, not blind prediction test",
        "pisa2025_subsets": {"all66": 66, "quality_clean60": 60, "excluded_quality_flag_codes": sorted(flagged_codes)},
        "country_cluster_design": "resample country IDs with replacement within domain; repeated folds for selected country remain paired",
        "target_score_design": "independent Normal(mean, published sampling SE) draw per country and target cycle, shared by all models; repeated cluster copies share that draw",
        "uncertainty_not_included": ["PISA scale-link errors and cross-cycle link covariance", "model training and hyperparameter uncertainty", "future-cycle/process uncertainty", "data/feature selection uncertainty", "country/domain/cycle score covariance"],
        "inputs": {str(p.relative_to(ROOT)): sha256(p) for p in [HIST, EST, CROSSWALK, P25, OECD25, FLAGS25]},
        "checks": {"historical_se_join_complete": True, "oecd_2025_se_join_complete": True,
                   "unique_prediction_keys": True, "2025_common_country_count_per_domain": 66,
                   "2025_quality_clean_country_count_per_domain": 60,
                   "all_predictions_fixed": True},
        "outputs": ["joint_metric_uncertainty.csv", "joint_paired_comparisons.csv", "joint_model_winner_probability.csv"],
    }
    (OUT / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(comparisons.to_string(index=False, max_rows=45))
    print(f"Wrote joint uncertainty summaries to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
