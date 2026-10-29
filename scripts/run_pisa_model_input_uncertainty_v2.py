#!/usr/bin/env python3
"""Model-input uncertainty analysis for PISA historical tabular benchmarks (v2).

Approach: models are trained ONCE per (domain, target_cycle) fold using the
published PISA means.  For each of N_DRAWS, the context input scores are
perturbed by drawing N(published_mean, SE²) independently per observation,
prediction-time features are recomputed from the perturbed context, and
predictions are generated from the fixed trained models.  Target scores are
held at published values throughout.

This isolates how measurement uncertainty in the historical PISA context scores
propagates through prediction features into forecast errors.  It is 2-3 orders
of magnitude faster than retraining per draw because models are fitted only
12 times (3 domains × 4 target cycles) rather than 12 × N_DRAWS times.

Distinction from Phase 3.25: Phase 3.25 perturbed the *targets* while holding
predictions fixed.  This analysis perturbs the *inputs* while holding models
fixed.  A simultaneous joint propagation is not attempted.

Existing outputs are never overwritten (script fails if OUT_DIR exists).
"""

from __future__ import annotations

import hashlib
import json
import platform
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
FOLDS_PATH = ROOT / "analysis/03_data_feasibility/audit/pisa_historical_folds_v1.json"
COUNTRY_EST_PATH = ROOT / "outputs/stage4/country_estimates.csv"
OUT_DIR = ROOT / "outputs/pisa_model_input_uncertainty_v1"

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


# ── helpers ──────────────────────────────────────────────────────────────────
def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def feature_row(
    years: list[int],
    scores: list[float],
    target: int,
) -> dict | None:
    prior_years = [y for y in years if y < target]
    prior_scores = [s for y, s in zip(years, scores) if y < target]
    if not prior_scores:
        return None
    lag1 = float(prior_scores[-1])
    lag2 = float(prior_scores[-2]) if len(prior_scores) >= 2 else np.nan
    return {
        "lag1_score": lag1,
        "lag2_score": lag2,
        "score_change": lag1 - lag2 if np.isfinite(lag2) else np.nan,
        "target_cycle": int(target),
        "gap_since_last_cycle": int(target - prior_years[-1]),
        "n_prior_observations": len(prior_scores),
    }


def training_data(
    panel_domain: pd.DataFrame,
    cutoff: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    features, targets = [], []
    for country, group in panel_domain.groupby("country_iso3", sort=True):
        group = group.sort_values("cycle")
        years = group["cycle"].astype(int).tolist()
        scores = group["score"].astype(float).tolist()
        for i in range(1, len(years)):
            t_year = years[i]
            if t_year >= cutoff:
                break
            row = feature_row(years[:i], scores[:i], t_year)
            if row is None:
                continue
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
                max_features=1.0, random_state=SEED, n_jobs=-1,
            ),
        ),
        "XGBoost": make_pipeline(
            imp(),
            XGBRegressor(
                n_estimators=300, max_depth=3, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                random_state=SEED, n_jobs=-1, verbosity=0,
            ),
        ),
    }


def linear_forecast(years: list[int], scores: list[float], target: int) -> float | None:
    if len(years) < 2:
        return None
    x = np.asarray(years, dtype=float)
    y = np.asarray(scores, dtype=float)
    cx = x.mean()
    coef = np.polyfit(x - cx, y, 1)
    return float(np.polyval(coef, target - cx))


# ── SE lookup ────────────────────────────────────────────────────────────────
def build_se_table(country_est: pd.DataFrame) -> dict[tuple[str, int, str], float]:
    pisa = country_est[
        (country_est["program"] == "PISA")
        & country_est["se"].notna()
        & (country_est["se"] > 0)
    ].copy()
    pisa["cycle"] = pisa["cycle"].astype(int)
    return {
        (row["country_iso3"], row["cycle"], row["domain"]): float(row["se"])
        for _, row in pisa.iterrows()
    }


# ── main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    if OUT_DIR.exists():
        raise SystemExit(f"Output directory already exists: {OUT_DIR}\nDelete it to rerun.")
    OUT_DIR.mkdir(parents=True)

    # ── load data ─────────────────────────────────────────────────────────────
    panel_raw = pd.read_csv(PANEL_PATH)
    folds_def = json.loads(FOLDS_PATH.read_text())
    country_est = pd.read_csv(COUNTRY_EST_PATH)

    # Melt panel to long form
    pisa_cols = {
        "PISA_mathematics": "mathematics",
        "PISA_reading": "reading",
        "PISA_science": "science",
    }
    id_cols = ["country_iso3", "cycle"]
    frames = []
    for col, domain in pisa_cols.items():
        sub = panel_raw[id_cols + [col]].dropna(subset=[col]).copy()
        sub = sub.rename(columns={col: "score"})
        sub["domain"] = domain
        frames.append(sub)
    panel = pd.concat(frames, ignore_index=True)
    panel["cycle"] = panel["cycle"].astype(int)

    se_table = build_se_table(country_est)

    # SE coverage summary
    se_coverage: dict[str, dict] = {}
    for domain in DOMAINS:
        sub = panel[(panel["domain"] == domain) & (panel["cycle"] < 2025)]
        total = len(sub)
        covered = sum(
            1 for _, r in sub.iterrows()
            if se_table.get((str(r["country_iso3"]), int(r["cycle"]), domain), 0.0) > 0
        )
        se_coverage[domain] = {"total_obs": total, "obs_with_se": covered}
        print(f"  SE coverage {domain}: {covered}/{total}")

    # ── parse folds ──────────────────────────────────────────────────────────
    folds_raw = folds_def["folds"] if isinstance(folds_def, dict) else folds_def
    folds: list[dict] = []
    for f in folds_raw:
        t = int(f["target_cycle"])
        if t > 2022:
            continue
        for domain in DOMAINS:
            if "domain_specific" in f and domain in f["domain_specific"]:
                countries = f["domain_specific"][domain].get("eligible_countries", [])
            else:
                countries = f.get("common_three_domain_countries", f.get("countries", []))
            folds.append({"target_cycle": t, "domain": domain, "countries": list(countries)})

    historical_cycles = sorted({f["target_cycle"] for f in folds})
    print(f"Historical folds: {historical_cycles}, {len(folds)} fold-domain combinations")

    # ── Step 1: train models ONCE on published data per (domain, target_cycle) ──
    print("Training models on published data (once per fold)...")
    trained: dict[tuple[str, int], dict[str, object]] = {}
    for domain in DOMAINS:
        panel_d = panel[panel["domain"] == domain].copy()
        for t_cycle in historical_cycles:
            X, y = training_data(panel_d, t_cycle)
            mdl_dict = make_models()
            for name, mdl in mdl_dict.items():
                mdl.fit(X, y)
            trained[(domain, t_cycle)] = mdl_dict
            print(f"  trained {domain} → {t_cycle} ({len(y)} examples)")

    # ── Step 2: build published context lookups ────────────────────────────────
    # For each (country, domain): sorted list of (cycle, score) before each target
    # shape: {(country, domain): [(cycle, score), ...]}
    published_series: dict[tuple[str, str], list[tuple[int, float]]] = {}
    for (country, domain_grp), grp in panel.groupby(["country_iso3", "domain"]):
        grp_sorted = grp.sort_values("cycle")
        published_series[(str(country), str(domain_grp))] = [
            (int(r["cycle"]), float(r["score"])) for _, r in grp_sorted.iterrows()
        ]

    # Published target scores per (country, domain, target_cycle)
    target_scores: dict[tuple[str, str, int], float] = {}
    for _, row in panel.iterrows():
        target_scores[(str(row["country_iso3"]), str(row["domain"]), int(row["cycle"]))] = float(row["score"])

    # ── Step 3: N_DRAWS ──────────────────────────────────────────────────────
    print(f"Running {N_DRAWS} draws...")
    rng_master = np.random.default_rng(SEED)
    draw_seeds = rng_master.integers(0, 2**31, size=N_DRAWS)

    all_rows: list[dict] = []

    for draw_idx in range(N_DRAWS):
        if draw_idx % 200 == 0:
            print(f"  draw {draw_idx}/{N_DRAWS}")
        rng = np.random.default_rng(int(draw_seeds[draw_idx]))

        # Sample perturbed scores for ALL (country, domain, cycle) at once
        perturbed: dict[tuple[str, str, int], float] = {}
        for domain in DOMAINS:
            for (country, dom_key), series in published_series.items():
                if dom_key != domain:
                    continue
                for (cycle, mean) in series:
                    se = se_table.get((country, cycle, domain), 0.0)
                    perturbed[(country, domain, cycle)] = (
                        float(rng.normal(mean, se)) if se > 0 else mean
                    )

        for fold in folds:
            t_cycle = int(fold["target_cycle"])
            domain = fold["domain"]
            mdl_dict = trained[(domain, t_cycle)]

            for country in fold["countries"]:
                true_score = target_scores.get((country, domain, t_cycle))
                if true_score is None:
                    continue

                # Build perturbed context (cycles strictly before target)
                series = published_series.get((country, domain), [])
                years = [c for c, _ in series if c < t_cycle]
                scores = [
                    perturbed.get((country, domain, c), s)
                    for c, s in series if c < t_cycle
                ]
                if not scores:
                    continue

                # Persistence
                all_rows.append({
                    "draw": draw_idx, "target_cycle": t_cycle, "domain": domain,
                    "country_iso3": country, "model": "Persistence",
                    "true_score": true_score,
                    "absolute_error": abs(scores[-1] - true_score),
                })

                # LinearTrend
                lt = linear_forecast(years, scores, t_cycle)
                if lt is not None:
                    all_rows.append({
                        "draw": draw_idx, "target_cycle": t_cycle, "domain": domain,
                        "country_iso3": country, "model": "LinearTrend",
                        "true_score": true_score,
                        "absolute_error": abs(lt - true_score),
                    })

                # ML models
                row = feature_row(years, scores, t_cycle)
                if row is None:
                    continue
                feat_df = pd.DataFrame([{name: row[name] for name in FEATURES}])
                for name, mdl in mdl_dict.items():
                    pred = float(mdl.predict(feat_df)[0])
                    all_rows.append({
                        "draw": draw_idx, "target_cycle": t_cycle, "domain": domain,
                        "country_iso3": country, "model": name,
                        "true_score": true_score,
                        "absolute_error": abs(pred - true_score),
                    })

    print(f"Total rows: {len(all_rows)}")

    # ── Step 4: summarise ────────────────────────────────────────────────────
    df = pd.DataFrame(all_rows)

    # Equal-fold-weighted MAE per draw × domain × model
    fold_mae = (
        df.groupby(["draw", "domain", "model", "target_cycle"])["absolute_error"]
        .mean()
        .reset_index(name="fold_mae")
    )
    equal_mae = (
        fold_mae.groupby(["draw", "domain", "model"])["fold_mae"]
        .mean()
        .reset_index(name="equal_fold_mae")
    )

    models_all = sorted(df["model"].unique())
    comparator = "Persistence"

    # Metric quantiles
    metric_rows = []
    for (domain, model), grp in equal_mae.groupby(["domain", "model"]):
        v = grp["equal_fold_mae"].values
        metric_rows.append({
            "domain": domain, "model": model,
            "point_mae": float(np.mean(v)),
            "q025": float(np.percentile(v, 2.5)),
            "q500": float(np.percentile(v, 50.0)),
            "q975": float(np.percentile(v, 97.5)),
            "sd": float(np.std(v)),
            "n_draws": int(len(v)),
        })
    metric_df = pd.DataFrame(metric_rows)

    # Paired comparisons
    pair_rows = []
    for domain in DOMAINS:
        pers = (
            equal_mae[(equal_mae["domain"] == domain) & (equal_mae["model"] == comparator)]
            .set_index("draw")["equal_fold_mae"]
        )
        for model in models_all:
            if model == comparator:
                continue
            mod = (
                equal_mae[(equal_mae["domain"] == domain) & (equal_mae["model"] == model)]
                .set_index("draw")["equal_fold_mae"]
            )
            aligned = pd.concat([mod, pers], axis=1, keys=["m", "p"]).dropna()
            if aligned.empty:
                continue
            delta = aligned["m"] - aligned["p"]
            pair_rows.append({
                "domain": domain, "model": model, "comparator": comparator,
                "point_delta": float(delta.mean()),
                "q025": float(delta.quantile(0.025)),
                "q500": float(delta.quantile(0.500)),
                "q975": float(delta.quantile(0.975)),
                "prob_model_better": float((delta < 0).mean()),
                "n_draws": int(len(delta)),
            })
    pair_df = pd.DataFrame(pair_rows)

    # Winner probabilities
    winner_rows = []
    for domain in DOMAINS:
        sub = equal_mae[equal_mae["domain"] == domain]
        pivot = sub.pivot(index="draw", columns="model", values="equal_fold_mae").dropna(axis=1, how="all")
        wc = pivot.idxmin(axis=1).value_counts(normalize=True)
        for model, prob in wc.items():
            winner_rows.append({"domain": domain, "model": model, "winner_prob": float(prob)})
    winner_df = pd.DataFrame(winner_rows)

    # ── write outputs ─────────────────────────────────────────────────────────
    metric_path = OUT_DIR / "model_input_uncertainty_metrics.csv"
    pair_path = OUT_DIR / "model_input_uncertainty_paired.csv"
    winner_path = OUT_DIR / "model_input_uncertainty_winner_prob.csv"
    metric_df.to_csv(metric_path, index=False)
    pair_df.to_csv(pair_path, index=False)
    winner_df.to_csv(winner_path, index=False)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": Path(__file__).name,
        "script_sha256": sha256(Path(__file__)),
        "version": "v2",
        "approach": (
            "Models trained once on published data per (domain, target_cycle). "
            "Per draw: context input scores perturbed N(mean, SE^2); "
            "prediction features recomputed; predictions generated from fixed models. "
            "Targets always fixed at published means."
        ),
        "n_draws": N_DRAWS,
        "seed": SEED,
        "historical_target_cycles": historical_cycles,
        "domains": list(DOMAINS),
        "models": models_all,
        "se_coverage": se_coverage,
        "note_timesfm_excluded": (
            "TimesFM excluded: its predictions do not use sklearn-style predict() "
            "and GPU refitting per draw is infeasible."
        ),
        "inputs": {
            "panel": str(PANEL_PATH),
            "panel_sha256": sha256(PANEL_PATH),
            "folds": str(FOLDS_PATH),
            "folds_sha256": sha256(FOLDS_PATH),
            "country_estimates": str(COUNTRY_EST_PATH),
            "country_estimates_sha256": sha256(COUNTRY_EST_PATH),
        },
        "python_version": platform.python_version(),
        "sklearn_version": sklearn.__version__,
        "xgboost_version": xgboost.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
    }
    (OUT_DIR / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))

    print(f"\nOutputs → {OUT_DIR}")
    print("\n=== Metric quantiles (equal-fold MAE) ===")
    print(metric_df.sort_values(["domain", "point_mae"]).to_string(index=False))
    print("\n=== Paired comparisons (model − Persistence) ===")
    print(pair_df.sort_values(["domain", "point_delta"]).to_string(index=False))
    print("\n=== Winner probabilities ===")
    print(winner_df.sort_values(["domain", "winner_prob"], ascending=[True, False]).to_string(index=False))


if __name__ == "__main__":
    main()
