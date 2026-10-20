#!/usr/bin/env python3
"""Refit-aware model-input uncertainty for historical PISA tabular forecasts.

Perturbs the historical PISA context scores used as model inputs by drawing
from N(mean, se^2) for each historical country/domain/cycle observation, then
reruns tabular models (Persistence, LinearTrend, Ridge, RandomForest, XGBoost)
on perturbed histories. Held-out evaluation targets stay fixed at published values.

This extends Phase 3.25, which only perturbed the target scores.  Here the
*inputs* to the models (the PISA history used for feature construction) are
perturbed while the evaluation criterion (true_score) stays unchanged.

Design notes
- SE data: outputs/stage4/country_estimates.csv, column 'se', for PISA
  cycles 2000-2022.  PISA 2000 reading identifiers use the verified project
  crosswalk.  Incomplete SE coverage fails preflight rather than silently
  substituting published means.
- Draws: N_DRAWS independent replications.  Each draw samples all input
  observations independently (i.e. within a draw, the same country/domain
  observation has the same perturbed value regardless of which fold uses it).
  Draws are seeded deterministically.
- Scope: historical tarihsel folds only (2012, 2015, 2018, 2022 targets).
  PISA 2025 is not included; it is an outcome-exposed retrospective.
- Models: Persistence, LinearTrend, Ridge, RandomForest, XGBoost.  The frozen
  common three-domain country set is used in every fold/domain.
  TimesFM refitting per draw would require GPU; excluded with note.
- Output: distributions of equal-fold-averaged MAE per domain per model,
  and paired MAE-delta distributions (each model minus Persistence).

The run is checkpointed in small batches so an interrupted run can resume from
completed batches. Other experiment output directories are never modified.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge as RidgeModel
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


# ── paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
PANEL_PATH = ROOT / "outputs/stage5/pisa_forecasting_panel_v2.csv"
PANEL_MANIFEST_PATH = ROOT / "outputs/stage5/pisa_forecasting_panel_v2_manifest.json"
FOLDS_PATH = ROOT / "analysis/03_data_feasibility/audit/pisa_historical_folds_v1.json"
COUNTRY_EST_PATH = ROOT / "outputs/stage4/country_estimates.csv"
CROSSWALK_PATH = ROOT / "analysis/03_data_feasibility/audit/pisa_2000_reading_country_crosswalk.csv"
OUT_DIR = ROOT / "outputs/pisa_model_input_uncertainty_v3"

DOMAINS = ("mathematics", "reading", "science")
FEATURES = (
    "lag1_score",
    "lag2_score",
    "score_change",
    "target_cycle",
    "gap_since_last_cycle",
    "n_prior_observations",
)
SEED = 20261015
N_DRAWS = 2000
BATCH_DRAWS = 25


# ── helpers ──────────────────────────────────────────────────────────────────
def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def feature_row(
    country: str,
    years: list[int],
    scores: list[float],
    target: int,
) -> dict:
    prior_years = [y for y in years if y < target]
    prior_scores = [s for y, s in zip(years, scores) if y < target]
    if not prior_scores:
        raise ValueError(f"No context for {country} at target {target}")
    lag1 = float(prior_scores[-1])
    lag2 = float(prior_scores[-2]) if len(prior_scores) >= 2 else np.nan
    return {
        "country_iso3": country,
        "lag1_score": lag1,
        "lag2_score": lag2,
        "score_change": lag1 - lag2 if np.isfinite(lag2) else np.nan,
        "target_cycle": int(target),
        "gap_since_last_cycle": int(target - prior_years[-1]),
        "n_prior_observations": len(prior_scores),
        "context_cycles": prior_years,
        "context_scores": prior_scores,
    }


def training_examples(
    series: pd.DataFrame,
    cutoff: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    features, targets = [], []
    for country, group in series.groupby("country_iso3", sort=True):
        group = group.sort_values("cycle")
        years = group["cycle"].astype(int).tolist()
        scores = group["score"].astype(float).tolist()
        for i in range(1, len(years)):
            target_year = years[i]
            if target_year >= cutoff:
                break
            row = feature_row(country, years[:i], scores[:i], target_year)
            features.append({name: row[name] for name in FEATURES})
            targets.append(scores[i])
    if not features:
        raise ValueError(f"No training examples before cycle {cutoff}")
    return pd.DataFrame(features, columns=list(FEATURES)), np.asarray(targets, dtype=float)


def make_models() -> dict[str, object]:
    imp = lambda: SimpleImputer(strategy="median")
    return {
        "Ridge": make_pipeline(imp(), StandardScaler(), RidgeModel(alpha=10.0)),
        "RandomForest": make_pipeline(
            imp(),
            RandomForestRegressor(
                n_estimators=300, max_depth=4, min_samples_leaf=5,
                max_features=1.0, random_state=SEED, n_jobs=1,
            ),
        ),
        "XGBoost": make_pipeline(
            imp(),
            XGBRegressor(
                n_estimators=300, max_depth=3, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                random_state=SEED, n_jobs=1, verbosity=0,
            ),
        ),
    }


def linear_forecast(years: list[int], scores: list[float], target: int) -> float:
    x = np.asarray(years, dtype=float)
    y = np.asarray(scores, dtype=float)
    cx = x.mean()
    xc = x - cx
    coef = np.polyfit(xc, y, 1)
    return float(np.polyval(coef, target - cx))


# ── SE lookup table ──────────────────────────────────────────────────────────
def build_se_table(
    country_est: pd.DataFrame,
    crosswalk: pd.DataFrame,
) -> dict[tuple[str, int, str], float]:
    """Return {(iso3, cycle, domain): se} for PISA rows with valid SE."""
    mapping = dict(zip(
        crosswalk["source_country_id"].astype(str).str.zfill(3),
        crosswalk["canonical_alpha3"].astype(str),
    ))
    pisa = country_est[
        (country_est["program"] == "PISA")
        & country_est["se"].notna()
        & (country_est["se"] > 0)
    ].copy()
    pisa["cycle"] = pisa["cycle"].astype(int)
    numeric_2000_reading = pisa["cycle"].eq(2000) & pisa["domain"].eq("reading")
    pisa.loc[numeric_2000_reading, "country_iso3"] = (
        pisa.loc[numeric_2000_reading, "country_iso3"]
        .astype(str)
        .str.zfill(3)
        .map(mapping)
    )
    if pisa["country_iso3"].isna().any():
        raise ValueError("PISA 2000 reading SE crosswalk left unmapped country identifiers")
    return {
        (row["country_iso3"], row["cycle"], row["domain"]): float(row["se"])
        for _, row in pisa.iterrows()
    }


# ── perturbed score lookup ────────────────────────────────────────────────────
def draw_perturbed_scores(
    panel: pd.DataFrame,
    se_table: dict,
    domain: str,
    rng: np.random.Generator,
) -> dict[tuple[str, int], float]:
    """For one draw: return {(iso3, cycle): perturbed_score} for the domain.

    Missing SEs fall back to the published mean; preflight separately requires
    complete SE coverage for every historical panel observation.
    """
    result: dict[tuple[str, int], float] = {}
    sub = panel[(panel["domain"] == domain) & (panel["cycle"] < 2025)]
    for _, row in sub.iterrows():
        iso3 = str(row["country_iso3"])
        cycle = int(row["cycle"])
        mean = float(row["score"])
        se = se_table.get((iso3, cycle, domain), 0.0)
        if se > 0:
            result[(iso3, cycle)] = float(rng.normal(mean, se))
        else:
            result[(iso3, cycle)] = mean
    return result


# ── one-fold evaluation on perturbed scores ───────────────────────────────────
def evaluate_fold_perturbed(
    domain: str,
    fold: dict,
    panel_domain: pd.DataFrame,
    perturbed: dict[tuple[str, int], float],
    trained_models: dict[str, object],
) -> list[dict]:
    """Return per-country prediction rows using perturbed input scores."""
    target_cycle = int(fold["target_cycle"])
    country_set = set(fold["countries"])
    rows = []

    # Build a perturbed series view: replace published scores with draws
    # (only context observations, i.e. cycle < target_cycle)
    def get_scores(country: str) -> tuple[list[int], list[float]]:
        sub = panel_domain[
            (panel_domain["country_iso3"] == country)
            & (panel_domain["cycle"] < target_cycle)
        ].sort_values("cycle")
        years = sub["cycle"].astype(int).tolist()
        scores = [
            perturbed.get((country, y), float(
                sub.loc[sub["cycle"] == y, "score"].values[0]
            ))
            for y in years
        ]
        return years, scores

    # True target score is always the published value (not perturbed)
    target_map = {
        str(r["country_iso3"]): float(r["score"])
        for _, r in panel_domain[panel_domain["cycle"] == target_cycle].iterrows()
    }

    for country in sorted(country_set):
        if country not in target_map:
            continue
        true_score = target_map[country]
        years, scores = get_scores(country)
        if not scores:
            continue
        n_context = len(scores)

        # Persistence
        pers_pred = scores[-1]
        rows.append({
            "target_cycle": target_cycle, "domain": domain,
            "country_iso3": country, "model": "Persistence",
            "true_score": true_score, "prediction": pers_pred,
            "absolute_error": abs(pers_pred - true_score),
        })

        # LinearTrend (needs ≥2 points)
        if n_context >= 2:
            try:
                lt_pred = linear_forecast(years, scores, target_cycle)
                rows.append({
                    "target_cycle": target_cycle, "domain": domain,
                    "country_iso3": country, "model": "LinearTrend",
                    "true_score": true_score, "prediction": lt_pred,
                    "absolute_error": abs(lt_pred - true_score),
                })
            except Exception:
                pass

        # ML models: need feature row
        try:
            feat = feature_row(country, years, scores, target_cycle)
        except Exception:
            continue
        feat_df = pd.DataFrame([{name: feat[name] for name in FEATURES}])

        for model_name, model in trained_models.items():
            try:
                pred = float(model.predict(feat_df)[0])
                rows.append({
                    "target_cycle": target_cycle, "domain": domain,
                    "country_iso3": country, "model": model_name,
                    "true_score": true_score, "prediction": pred,
                    "absolute_error": abs(pred - true_score),
                })
            except Exception:
                pass

    return rows


# ── train models once per fold on perturbed training data ─────────────────────
def train_models_perturbed(
    domain: str,
    target_cycle: int,
    panel_domain: pd.DataFrame,
    perturbed: dict[tuple[str, int], float],
) -> dict[str, object]:
    """Train ML models using perturbed context scores as features."""
    # Build training set: all (country, target_year) pairs where target_year < target_cycle
    features, targets = [], []
    for country, group in panel_domain.groupby("country_iso3", sort=True):
        group = group.sort_values("cycle")
        years = group["cycle"].astype(int).tolist()
        # Replace published scores with perturbed values for context cycles
        scores = [
            perturbed.get((country, y), float(
                group.loc[group["cycle"] == y, "score"].values[0]
            ))
            for y in years
        ]
        for i in range(1, len(years)):
            t_year = years[i]
            if t_year >= target_cycle:
                break
            context_years = years[:i]
            context_scores = scores[:i]
            # Target score for training: use perturbed value too
            t_score = perturbed.get((country, t_year), scores[i])
            try:
                row = feature_row(country, context_years, context_scores, t_year)
                features.append({name: row[name] for name in FEATURES})
                targets.append(t_score)
            except Exception:
                pass

    if not features:
        raise ValueError(f"No training examples for {domain} cutoff {target_cycle}")

    X = pd.DataFrame(features, columns=list(FEATURES))
    y = np.asarray(targets, dtype=float)
    models = make_models()
    trained = {}
    for name, mdl in models.items():
        mdl.fit(X, y)
        trained[name] = mdl
    return trained


# ── summarise draw results ────────────────────────────────────────────────────
def summarise_draws(
    all_draw_fold_maes: list[dict],
    folds: list[dict],
    domains: tuple[str, ...],
    models_all: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (metric_quantiles, paired_comparisons, winner_probs) DataFrames."""
    df = pd.DataFrame(all_draw_fold_maes)

    # Input has one aggregated MAE per draw × fold × domain × model.
    equal_fold_mae = (
        df.groupby(["draw", "domain", "model"])["fold_mae"]
        .mean()
        .reset_index(name="equal_fold_mae")
    )

    # Quantiles of equal-fold MAE
    metric_rows = []
    for (domain, model), grp in equal_fold_mae.groupby(["domain", "model"]):
        vals = grp["equal_fold_mae"].values
        metric_rows.append({
            "domain": domain, "model": model,
            "point_mae": float(np.mean(vals)),
            "q025": float(np.percentile(vals, 2.5)),
            "q500": float(np.percentile(vals, 50.0)),
            "q975": float(np.percentile(vals, 97.5)),
            "sd": float(np.std(vals)),
            "n_draws": len(vals),
        })
    metric_df = pd.DataFrame(metric_rows)

    # Paired comparisons: each model minus Persistence
    comparator = "Persistence"
    pair_rows = []
    for domain in domains:
        pers = equal_fold_mae[
            (equal_fold_mae["domain"] == domain)
            & (equal_fold_mae["model"] == comparator)
        ].set_index("draw")["equal_fold_mae"]
        for model in models_all:
            if model == comparator:
                continue
            mod = equal_fold_mae[
                (equal_fold_mae["domain"] == domain)
                & (equal_fold_mae["model"] == model)
            ].set_index("draw")["equal_fold_mae"]
            aligned = pd.concat([mod, pers], axis=1, keys=["model", "pers"]).dropna()
            if aligned.empty:
                continue
            delta = aligned["model"] - aligned["pers"]
            pair_rows.append({
                "domain": domain, "model": model, "comparator": comparator,
                "point_delta": float(delta.mean()),
                "q025": float(delta.quantile(0.025)),
                "q500": float(delta.quantile(0.500)),
                "q975": float(delta.quantile(0.975)),
                "prob_model_better": float((delta < 0).mean()),
                "n_draws": len(delta),
            })
    pair_df = pd.DataFrame(pair_rows)

    # Winner probability: which model has lowest equal-fold MAE per draw per domain
    winner_rows = []
    for domain in domains:
        sub = equal_fold_mae[equal_fold_mae["domain"] == domain]
        pivot = sub.pivot(index="draw", columns="model", values="equal_fold_mae")
        pivot = pivot.dropna(axis=1, how="all")
        winner_counts = pivot.idxmin(axis=1).value_counts(normalize=True)
        for model, prob in winner_counts.items():
            winner_rows.append({"domain": domain, "model": model, "winner_prob": float(prob)})
    winner_df = pd.DataFrame(winner_rows)

    return metric_df, pair_df, winner_df


# ── main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    if (OUT_DIR / "run_manifest.json").exists():
        raise SystemExit(f"Completed run already exists: {OUT_DIR}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = OUT_DIR / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    # ── load inputs ──────────────────────────────────────────────────────────
    panel_raw = pd.read_csv(PANEL_PATH)
    folds_def = json.loads(FOLDS_PATH.read_text())
    country_est = pd.read_csv(COUNTRY_EST_PATH)
    crosswalk = pd.read_csv(CROSSWALK_PATH, dtype=str)

    run_config = {
        "version": "v3_refit_aware",
        "n_draws": N_DRAWS,
        "batch_draws": BATCH_DRAWS,
        "seed": SEED,
        "script_sha256": sha256(Path(__file__)),
        "inputs": {
            "panel_sha256": sha256(PANEL_PATH),
            "folds_sha256": sha256(FOLDS_PATH),
            "country_estimates_sha256": sha256(COUNTRY_EST_PATH),
            "crosswalk_sha256": sha256(CROSSWALK_PATH),
        },
    }
    config_path = OUT_DIR / "run_config.json"
    if config_path.exists():
        if json.loads(config_path.read_text()) != run_config:
            raise ValueError("Existing checkpoint run configuration does not match this script/inputs")
    elif any(checkpoint_dir.glob("draw_metrics_*.csv")):
        raise ValueError("Checkpoint files exist but run_config.json is missing")
    else:
        config_path.write_text(json.dumps(run_config, indent=2))

    # Melt panel to long form: country_iso3, cycle, domain, score
    id_cols = ["country_iso3", "cycle"]
    pisa_score_cols = {"PISA_mathematics": "mathematics", "PISA_reading": "reading", "PISA_science": "science"}
    domain_frames = []
    for col, domain in pisa_score_cols.items():
        if col not in panel_raw.columns:
            raise ValueError(f"Expected column {col} in panel")
        sub = panel_raw[id_cols + [col]].dropna(subset=[col]).copy()
        sub = sub.rename(columns={col: "score"})
        sub["domain"] = domain
        domain_frames.append(sub)
    panel = pd.concat(domain_frames, ignore_index=True)
    panel["cycle"] = panel["cycle"].astype(int)

    se_table = build_se_table(country_est, crosswalk)
    se_coverage = {}
    for domain in DOMAINS:
        sub = panel[(panel["domain"] == domain) & (panel["cycle"] < 2025)]
        total = len(sub)
        covered = sum(1 for _, r in sub.iterrows() if se_table.get((str(r["country_iso3"]), int(r["cycle"]), domain), 0.0) > 0)
        se_coverage[domain] = {"total_obs": total, "obs_with_se": covered}
        if covered != total:
            missing = [
                (str(r["country_iso3"]), int(r["cycle"]))
                for _, r in sub.iterrows()
                if se_table.get((str(r["country_iso3"]), int(r["cycle"]), domain), 0.0) <= 0
            ]
            raise ValueError(f"Incomplete historical SE coverage for {domain}: {missing[:10]}")

    print("SE coverage for input perturbation:")
    for d, v in se_coverage.items():
        print(f"  {d}: {v['obs_with_se']}/{v['total_obs']} observations have SE > 0")

    # ── identify folds ───────────────────────────────────────────────────────
    if isinstance(folds_def, dict) and "folds" in folds_def:
        folds_raw = folds_def["folds"]
    elif isinstance(folds_def, list):
        folds_raw = folds_def
    else:
        raise ValueError("Unexpected folds file structure")

    # Normalise to a flat list of {target_cycle, domain, countries}
    # folds_v1.json stores domain-specific eligible_countries plus a
    # common_three_domain_countries list per fold.
    folds: list[dict] = []
    for f in folds_raw:
        t = int(f["target_cycle"])
        if t > 2022:
            continue
        for domain in DOMAINS:
            if "common_three_domain_countries" in f:
                countries = f["common_three_domain_countries"]
            else:
                raise ValueError(f"Fold {t} lacks frozen common three-domain country set")
            folds.append({"target_cycle": t, "domain": domain, "countries": countries})

    historical_cycles = sorted({f["target_cycle"] for f in folds})
    print(f"Historical target cycles: {historical_cycles}")
    models_all_names = ["Persistence", "LinearTrend", "Ridge", "RandomForest", "XGBoost"]

    # ── run draws ────────────────────────────────────────────────────────────
    rng_master = np.random.default_rng(SEED)
    draw_seeds = rng_master.integers(0, 2**31, size=N_DRAWS)

    completed_draws: set[int] = set()
    for checkpoint in sorted(checkpoint_dir.glob("draw_metrics_*.csv")):
        saved = pd.read_csv(checkpoint)
        stem = checkpoint.stem.removeprefix("draw_metrics_")
        start_text, end_text = stem.split("_")
        expected_batch = set(range(int(start_text), int(end_text)))
        if set(saved["draw"].astype(int).unique()) != expected_batch:
            raise ValueError(f"Incomplete or malformed checkpoint: {checkpoint}")
        expected_batch_rows = len(expected_batch) * len(DOMAINS) * len(historical_cycles) * len(models_all_names)
        if len(saved) != expected_batch_rows:
            raise ValueError(f"Checkpoint row count mismatch: {checkpoint}")
        completed_draws.update(saved["draw"].astype(int).unique().tolist())
    if any(draw < 0 or draw >= N_DRAWS for draw in completed_draws):
        raise ValueError("Checkpoint contains draw indices outside configured run")

    panel_by_domain = {d: panel[panel["domain"] == d].copy() for d in DOMAINS}
    for batch_start in range(0, N_DRAWS, BATCH_DRAWS):
        batch_end = min(batch_start + BATCH_DRAWS, N_DRAWS)
        batch_path = checkpoint_dir / f"draw_metrics_{batch_start:04d}_{batch_end:04d}.csv"
        batch_draws = [d for d in range(batch_start, batch_end) if d not in completed_draws]
        if not batch_draws:
            continue

        print(f"  draws {batch_start}–{batch_end - 1} / {N_DRAWS}", flush=True)
        batch_rows: list[dict] = []
        batch_t0 = time.monotonic()
        for draw_idx in batch_draws:
            rng = np.random.default_rng(int(draw_seeds[draw_idx]))

            # One sampled value per country/domain/cycle is reused both as a
            # context input and, where applicable, as a training target.
            perturbed_by_domain: dict[str, dict] = {}
            for domain in DOMAINS:
                perturbed_by_domain[domain] = draw_perturbed_scores(
                    panel_by_domain[domain], se_table, domain, rng
                )

            trained_cache: dict[tuple[str, int], dict] = {}
            for fold in folds:
                t_cycle = int(fold["target_cycle"])
                domain = fold["domain"]
                panel_domain = panel_by_domain[domain]
                perturbed = perturbed_by_domain[domain]
                fold_countries = fold["countries"]

                cache_key = (domain, t_cycle)
                if cache_key not in trained_cache:
                    trained_cache[cache_key] = train_models_perturbed(
                        domain, t_cycle, panel_domain, perturbed
                    )

                rows = evaluate_fold_perturbed(
                    domain,
                    {"target_cycle": t_cycle, "countries": fold_countries},
                    panel_domain,
                    perturbed,
                    trained_cache[cache_key],
                )
                fold_results = pd.DataFrame(rows)
                if fold_results.empty:
                    raise ValueError(f"No predictions for {domain} fold {t_cycle}")
                if set(fold_results["model"].unique()) != set(models_all_names):
                    raise ValueError(f"Incomplete model coverage for {domain} fold {t_cycle}")
                for model_name, group in fold_results.groupby("model", sort=True):
                    if group["country_iso3"].nunique() != len(fold_countries):
                        raise ValueError(
                            f"Country coverage mismatch for {domain}/{t_cycle}/{model_name}"
                        )
                    batch_rows.append({
                        "draw": draw_idx,
                        "target_cycle": t_cycle,
                        "domain": domain,
                        "model": model_name,
                        "fold_mae": float(group["absolute_error"].mean()),
                        "n_countries": int(group["country_iso3"].nunique()),
                    })

        batch_frame = pd.DataFrame(batch_rows)
        if set(batch_frame["draw"].unique()) != set(batch_draws):
            raise ValueError(f"Batch {batch_start}:{batch_end} is missing draw results")
        temporary_path = batch_path.with_suffix(".csv.tmp")
        batch_frame.to_csv(temporary_path, index=False)
        os.replace(temporary_path, batch_path)
        completed_draws.update(batch_draws)
        print(
            f"    checkpointed {len(batch_frame)} fold metrics in "
            f"{time.monotonic() - batch_t0:.1f}s",
            flush=True,
        )

    if len(completed_draws) != N_DRAWS:
        raise ValueError(f"Incomplete run: {len(completed_draws)}/{N_DRAWS} draws checkpointed")

    all_draw_fold_maes = pd.concat(
        [pd.read_csv(p) for p in sorted(checkpoint_dir.glob("draw_metrics_*.csv"))],
        ignore_index=True,
    )
    expected_rows = N_DRAWS * len(folds) * len(models_all_names)
    if len(all_draw_fold_maes) != expected_rows:
        raise ValueError(f"Expected {expected_rows} fold metrics, found {len(all_draw_fold_maes)}")
    print(f"Total fold/model/draw metrics: {len(all_draw_fold_maes)}")

    # ── summarise ────────────────────────────────────────────────────────────
    metric_df, pair_df, winner_df = summarise_draws(
        all_draw_fold_maes, folds, DOMAINS, models_all_names
    )

    # ── write outputs ─────────────────────────────────────────────────────────
    metric_path = OUT_DIR / "model_input_uncertainty_metrics.csv"
    pair_path = OUT_DIR / "model_input_uncertainty_paired.csv"
    winner_path = OUT_DIR / "model_input_uncertainty_winner_prob.csv"
    metric_df.to_csv(metric_path, index=False)
    pair_df.to_csv(pair_path, index=False)
    winner_df.to_csv(winner_path, index=False)

    # ── manifest ─────────────────────────────────────────────────────────────
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).name),
        "script_sha256": sha256(Path(__file__)),
        "n_draws": N_DRAWS,
        "seed": SEED,
        "draw_seed_range": [int(draw_seeds[0]), int(draw_seeds[-1])],
        "historical_target_cycles": historical_cycles,
        "domains": list(DOMAINS),
        "models": models_all_names,
        "se_coverage": se_coverage,
        "sample_definition": "Frozen common three-domain country set in each historical fold.",
        "sampling_assumption": (
            "Within each draw, each country/domain/cycle score is sampled once from "
            "N(published mean, published marginal SE^2). That value is shared when it "
            "serves as a historical context input or training label. Cross-country and "
            "cross-cycle sampling/link covariance are not modeled."
        ),
        "refit_scope": (
            "Ridge, Random Forest, and XGBoost are refitted on each draw's perturbed "
            "historical features and training labels for every domain and target cycle. "
            "Held-out target means remain fixed for scoring. Persistence and linear trend "
            "are recomputed from the perturbed context."
        ),
        "fit_parallelism": "Single-threaded. Compared with n_jobs=-1 on the same seeded draw; maximum fold/model MAE difference was 3.2e-14.",
        "note_timesfm_excluded": (
            "TimesFM model-input uncertainty excluded: refitting TimesFM per draw "
            "requires GPU and is infeasible in this analysis. TimesFM point predictions "
            "from pisa_all_models_quadratic_trend_v1 are not perturbed here."
        ),
        "note_target_held_fixed": (
            "Target scores (true_score) are always the published OECD means. "
            "Only context input scores are perturbed. This is orthogonal to the "
            "target-score SE analysis in pisa_target_sampling_uncertainty_v3."
        ),
        "note_retrospective": (
            "PISA 2025 is excluded. All folds are historical temporal validation. "
            "Results do not constitute prospective or blind evaluation."
        ),
        "inputs": {
            "panel": str(PANEL_PATH),
            "panel_sha256": sha256(PANEL_PATH),
            "folds": str(FOLDS_PATH),
            "folds_sha256": sha256(FOLDS_PATH),
            "country_estimates": str(COUNTRY_EST_PATH),
            "country_estimates_sha256": sha256(COUNTRY_EST_PATH),
            "pisa_2000_reading_crosswalk": str(CROSSWALK_PATH),
            "pisa_2000_reading_crosswalk_sha256": sha256(CROSSWALK_PATH),
        },
        "outputs": {
            "metrics": str(metric_path),
            "paired": str(pair_path),
            "winner_prob": str(winner_path),
            "checkpoints": str(checkpoint_dir),
        },
        "output_sha256": {
            "metrics": sha256(metric_path),
            "paired": sha256(pair_path),
            "winner_prob": sha256(winner_path),
        },
        "python_version": platform.python_version(),
        "sklearn_version": sklearn.__version__,
        "xgboost_version": xgboost.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
    }
    (OUT_DIR / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str)
    )
    print(f"\nOutputs written to {OUT_DIR}")
    print("\nMetrics summary (equal-fold MAE quantiles):")
    print(metric_df.sort_values(["domain", "point_mae"]).to_string(index=False))
    print("\nPaired comparisons (model minus Persistence, equal-fold MAE delta):")
    print(pair_df.sort_values(["domain", "point_delta"]).to_string(index=False))
    print("\nWinner probabilities:")
    print(winner_df.sort_values(["domain", "winner_prob"], ascending=[True, False]).to_string(index=False))


if __name__ == "__main__":
    main()
