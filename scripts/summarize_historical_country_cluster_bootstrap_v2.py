#!/usr/bin/env python3
"""Extend the paired historical country-cluster bootstrap to new trend baseline.

The original eight-model TimesFM comparison remains immutable. This version
adds the literature-motivated LinearQuadraticTrend from tabular benchmark v2,
checks exact common-key predictions for existing arms, then bootstraps paired
country clusters over the same four historical target cycles.
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
BASE_INPUT = ROOT / "outputs/pisa_timesfm25_ils_adapter_v1/all_models_historical_predictions.csv"
TABULAR_V2 = ROOT / "outputs/pisa_tabular_benchmark_v2/historical_predictions.csv"
TABULAR_DIAGNOSTICS = ROOT / "outputs/pisa_tabular_benchmark_v2/trend_selection_diagnostics.csv"
OUT_DIR = ROOT / "outputs/historical_country_cluster_bootstrap_v2"
SUMMARY_PATH = OUT_DIR / "equal_fold_paired_country_bootstrap.csv"
DETAIL_PATH = OUT_DIR / "bootstrap_prediction_panel.csv"
MANIFEST_PATH = OUT_DIR / "run_manifest.json"
N_BOOTSTRAP = 20_000
SEED = 20261017
REFERENCE = "Persistence"
ADAPTATION_MODEL = "TimesFM2_5_ILSA_LoRA"
ADAPTATION_REFERENCE = "TimesFM2_5_zero_shot"
ADDED_MODEL = "LinearQuadraticTrend"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def assert_unique_and_finite(data: pd.DataFrame, label: str) -> None:
    keys = ["target_cycle", "domain", "country_iso3", "model"]
    if data.duplicated(keys).any():
        raise ValueError(f"{label} prediction keys are not unique")
    if not np.isfinite(data[["prediction", "true_score"]].to_numpy(dtype=float)).all():
        raise ValueError(f"{label} contains non-finite prediction/target values")


def main() -> None:
    for path in (BASE_INPUT, TABULAR_V2, TABULAR_DIAGNOSTICS):
        if not path.is_file():
            raise FileNotFoundError(path)
    if OUT_DIR.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {OUT_DIR}")

    base = pd.read_csv(BASE_INPUT)
    tabular = pd.read_csv(TABULAR_V2)
    assert_unique_and_finite(base, "TimesFM benchmark")
    assert_unique_and_finite(tabular, "Tabular v2 benchmark")
    if not tabular["model"].eq(ADDED_MODEL).any():
        raise ValueError(f"Tabular v2 is missing {ADDED_MODEL}")
    extra = tabular.loc[tabular["model"].eq(ADDED_MODEL)].copy()
    keys = ["target_cycle", "domain", "country_iso3"]
    trend_diagnostics = pd.read_csv(TABULAR_DIAGNOSTICS, dtype={"country_iso3": str})
    if trend_diagnostics.duplicated(keys).any():
        raise ValueError("Trend-selection diagnostics have duplicate target keys")
    extra = extra.merge(
        trend_diagnostics[keys + ["context_cycles"]],
        on=keys,
        how="left",
        validate="one_to_one",
    )
    if extra["context_cycles"].isna().any():
        raise ValueError("Missing context-cycle metadata for new trend baseline")

    base_persistence = base.loc[base["model"].eq(REFERENCE), keys + ["true_score"]]
    if len(base_persistence) != len(extra):
        raise ValueError("New trend baseline row count differs from the established target panel")
    key_check = extra[keys + ["true_score"]].merge(
        base_persistence,
        on=keys,
        how="outer",
        suffixes=("_new", "_base"),
        indicator=True,
        validate="one_to_one",
    )
    if not key_check["_merge"].eq("both").all() or not np.allclose(
        key_check["true_score_new"], key_check["true_score_base"], rtol=0, atol=1e-10
    ):
        raise ValueError("New trend baseline target keys or observed scores differ from TimesFM panel")

    # Existing learned and simple tabular arms must match the TimesFM all-model
    # input exactly before it is extended with the new method.
    shared_models = sorted(set(base["model"]) & (set(tabular["model"]) - {ADDED_MODEL}))
    for model in shared_models:
        left = base.loc[base["model"].eq(model), keys + ["true_score", "prediction"]].sort_values(keys).reset_index(drop=True)
        right = tabular.loc[tabular["model"].eq(model), keys + ["true_score", "prediction"]].sort_values(keys).reset_index(drop=True)
        if not left[keys].equals(right[keys]):
            raise ValueError(f"Common keys differ for existing method {model}")
        if not np.allclose(left[["true_score", "prediction"]], right[["true_score", "prediction"]], rtol=0, atol=1e-10):
            raise ValueError(f"Existing method scores/predictions differ for {model}")

    combined = pd.concat([base, extra[base.columns]], ignore_index=True)
    assert_unique_and_finite(combined, "Combined benchmark")
    models = sorted(combined["model"].unique().tolist())
    domains = sorted(combined["domain"].unique().tolist())
    targets = sorted(combined["target_cycle"].astype(int).unique().tolist())

    # Store absolute errors by domain/fold/model/country and enforce exact
    # matched-country sets before paired resampling.
    errors: dict[tuple[str, int, str], dict[str, float]] = {}
    for (domain, target, model), group in combined.groupby(["domain", "target_cycle", "model"], sort=True):
        errors[(str(domain), int(target), str(model))] = {
            str(row.country_iso3): abs(float(row.prediction) - float(row.true_score))
            for row in group.itertuples(index=False)
        }
    for domain in domains:
        for target in targets:
            sets = [set(errors.get((domain, target, model), {})) for model in models]
            if not sets[0] or any(country_set != sets[0] for country_set in sets[1:]):
                raise ValueError(f"Model country sets are incomplete or unpaired for {domain}/{target}")

    comparisons = [(model, REFERENCE) for model in models if model != REFERENCE]
    if (ADAPTATION_MODEL, ADAPTATION_REFERENCE) not in comparisons:
        comparisons.append((ADAPTATION_MODEL, ADAPTATION_REFERENCE))

    rng = np.random.default_rng(SEED)
    results: list[dict] = []
    for domain in domains:
        countries = sorted(
            {
                country
                for target in targets
                for country in errors[(domain, target, REFERENCE)]
            }
        )
        country_index = {country: i for i, country in enumerate(countries)}
        # A shared country-cluster draw matrix across pairwise comparisons
        # improves comparability of the uncertainty summaries.
        draws = rng.integers(0, len(countries), size=(N_BOOTSTRAP, len(countries)))
        multiplicities = np.zeros((N_BOOTSTRAP, len(countries)), dtype=np.int16)
        row_index = np.repeat(np.arange(N_BOOTSTRAP), len(countries))
        np.add.at(multiplicities, (row_index, draws.reshape(-1)), 1)

        for model, comparator in comparisons:
            observed_fold_deltas: list[float] = []
            bootstrap_fold_deltas: list[np.ndarray] = []
            n_domain_countries = len(countries)
            for target in targets:
                model_error = errors[(domain, target, model)]
                comparator_error = errors[(domain, target, comparator)]
                fold_countries = sorted(model_error)
                if fold_countries != sorted(comparator_error):
                    raise ValueError(f"Unpaired comparison: {domain}/{target}/{model}/{comparator}")
                idx = np.asarray([country_index[c] for c in fold_countries], dtype=np.int64)
                paired_diff = np.asarray(
                    [model_error[c] - comparator_error[c] for c in fold_countries], dtype=float
                )
                observed_fold_deltas.append(float(paired_diff.mean()))
                fold_weights = multiplicities[:, idx]
                denominator = fold_weights.sum(axis=1)
                if np.any(denominator == 0):
                    raise RuntimeError("A bootstrap resample omitted all eligible countries in a fold")
                bootstrap_fold_deltas.append((fold_weights @ paired_diff) / denominator)

            observed_delta = float(np.mean(observed_fold_deltas))
            bootstrap_delta = np.mean(np.vstack(bootstrap_fold_deltas), axis=0)
            low, high = np.quantile(bootstrap_delta, [0.025, 0.975])
            results.append(
                {
                    "domain": domain,
                    "model": model,
                    "comparator": comparator,
                    "n_country_clusters": n_domain_countries,
                    "n_outer_folds": len(targets),
                    "outer_targets": ";".join(map(str, targets)),
                    "delta_equal_fold_MAE_model_minus_comparator": round(observed_delta, 8),
                    "bootstrap_95pct_low": round(float(low), 8),
                    "bootstrap_95pct_high": round(float(high), 8),
                    "bootstrap_probability_delta_lt_0": round(float(np.mean(bootstrap_delta < 0)), 8),
                    "bootstrap_replicates": N_BOOTSTRAP,
                    "seed": SEED,
                    "interpretation": "Exploratory paired country-cluster interval conditional on four observed folds and published point estimates.",
                }
            )

    OUT_DIR.mkdir(parents=True, exist_ok=False)
    summary = pd.DataFrame(results).sort_values(["domain", "model", "comparator"]).reset_index(drop=True)
    summary.to_csv(SUMMARY_PATH, index=False, float_format="%.8f")
    combined.to_csv(DETAIL_PATH, index=False, float_format="%.10g")
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "script": str(Path(__file__).resolve().relative_to(ROOT)),
        "script_sha256": sha256(Path(__file__).resolve()),
        "base_prediction_input": str(BASE_INPUT.relative_to(ROOT)),
        "base_prediction_input_sha256": sha256(BASE_INPUT),
        "new_trend_input": str(TABULAR_V2.relative_to(ROOT)),
        "new_trend_input_sha256": sha256(TABULAR_V2),
        "new_trend_diagnostics_input": str(TABULAR_DIAGNOSTICS.relative_to(ROOT)),
        "new_trend_diagnostics_sha256": sha256(TABULAR_DIAGNOSTICS),
        "new_method": ADDED_MODEL,
        "existing_common_methods_exactly_match": shared_models,
        "combined_models": models,
        "domains": domains,
        "outer_targets": targets,
        "prediction_rows": int(len(combined)),
        "unique_prediction_keys": int(len(combined)),
        "all_model_country_sets_match_within_fold_domain": True,
        "resampling": {
            "unit": "country cluster within domain; repeated fold errors for sampled country stay paired",
            "replicates": N_BOOTSTRAP,
            "seed": SEED,
            "interval": "percentile 95%",
            "estimand": "equal mean across the four outer-fold MAEs; paired model-minus-comparator",
            "draws": "A shared country-cluster draw matrix is used for all pairwise comparisons within each domain.",
            "comparisons": "Every method vs Persistence; TimesFM2_5_ILSA_LoRA additionally vs matched TimesFM2_5_zero_shot.",
        },
        "limitations": [
            "Conditional on four observed historical target cycles; does not quantify between-cycle uncertainty.",
            "Conditions on published point estimates; does not propagate plausible-value, complex-survey, or scale-link uncertainty.",
            "Participating countries are a selected set, not a probability sample of all education systems.",
            "Intervals are exploratory and not confirmatory significance tests or evidence of prospective performance.",
        ],
        "checks": {
            "new_method_matches_locked_target_keys_and_scores": True,
            "existing_tabular_methods_predictions_match_timesfm_aggregate": True,
            "finite_predictions": bool(np.isfinite(combined[["prediction", "true_score"]].to_numpy(dtype=float)).all()),
            "unique_prediction_keys": not bool(combined.duplicated(["target_cycle", "domain", "country_iso3", "model"]).any()),
            "comparison_rows": int(len(summary)),
        },
        "outputs": {
            SUMMARY_PATH.name: sha256(SUMMARY_PATH),
            DETAIL_PATH.name: sha256(DETAIL_PATH),
        },
        "overwrite_policy": "Output directory must not exist before execution.",
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(summary.loc[summary["model"].eq(ADDED_MODEL), ["domain", "model", "delta_equal_fold_MAE_model_minus_comparator", "bootstrap_95pct_low", "bootstrap_95pct_high"]].to_string(index=False))
    print(f"Wrote {len(summary)} comparisons to {OUT_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
