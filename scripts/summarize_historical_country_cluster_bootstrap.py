#!/usr/bin/env python3
"""Paired country-cluster bootstrap for fixed historical PISA forecasts.

Resamples country clusters within each domain and preserves each selected
country's paired model errors across all available outer target folds. The
estimand is the equal-fold mean MAE used by the project reports. This is an
exploratory conditional interval, not a forecast of unseen cycles/countries.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "outputs/pisa_timesfm25_ils_adapter_v1/all_models_historical_predictions.csv"
OUT_DIR = ROOT / "outputs/historical_country_cluster_bootstrap_v1"
SUMMARY = OUT_DIR / "equal_fold_paired_country_bootstrap.csv"
MANIFEST = OUT_DIR / "run_manifest.json"
N_BOOTSTRAP = 20_000
SEED = 20261016
REFERENCE = "Persistence"
ADAPTATION_MODEL = "TimesFM2_5_ILSA_LoRA"
ADAPTATION_REFERENCE = "TimesFM2_5_zero_shot"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("Cannot write empty summary")
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    if OUT_DIR.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {OUT_DIR}")
    if not INPUT.is_file():
        raise FileNotFoundError(INPUT)
    rows = load_rows(INPUT)
    required = {"target_cycle", "domain", "country_iso3", "model", "prediction", "true_score"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Missing required prediction columns: {required - set(rows[0] if rows else [])}")

    # Key uniqueness and finite predictions are prerequisites for paired loss resampling.
    key_counts: dict[tuple[str, str, str, str], int] = defaultdict(int)
    normalized: list[dict[str, object]] = []
    for row in rows:
        key = (row["target_cycle"], row["domain"], row["country_iso3"], row["model"])
        key_counts[key] += 1
        prediction = float(row["prediction"])
        truth = float(row["true_score"])
        if not np.isfinite(prediction) or not np.isfinite(truth):
            raise ValueError(f"Non-finite prediction/target for {key}")
        normalized.append(
            {
                "target_cycle": int(row["target_cycle"]),
                "domain": row["domain"],
                "country_iso3": row["country_iso3"],
                "model": row["model"],
                "absolute_error": abs(prediction - truth),
            }
        )
    duplicates = [key for key, count in key_counts.items() if count != 1]
    if duplicates:
        raise ValueError(f"Duplicate prediction keys found; examples: {duplicates[:5]}")

    domains = sorted({str(row["domain"]) for row in normalized})
    models = sorted({str(row["model"]) for row in normalized})
    targets = sorted({int(row["target_cycle"]) for row in normalized})
    if REFERENCE not in models or ADAPTATION_MODEL not in models or ADAPTATION_REFERENCE not in models:
        raise ValueError("Missing a required reference model")

    # Validate paired key coverage: every model must forecast the same country
    # in each fold/domain before any paired difference is computed.
    by_group_model: dict[tuple[str, int, str], dict[str, dict[str, float]]] = defaultdict(dict)
    for row in normalized:
        group = (str(row["domain"]), int(row["target_cycle"]), str(row["model"]))
        country = str(row["country_iso3"])
        by_group_model[group][country] = float(row["absolute_error"])
    for domain in domains:
        for target in targets:
            sets = [set(by_group_model[(domain, target, model)]) for model in models]
            if not sets[0] or any(countries != sets[0] for countries in sets[1:]):
                raise ValueError(f"Model country sets do not match at {domain}/{target}")

    rng = np.random.default_rng(SEED)
    output: list[dict[str, object]] = []
    for domain in domains:
        domain_rows = [r for r in normalized if r["domain"] == domain]
        countries = sorted({str(r["country_iso3"]) for r in domain_rows})
        country_index = {country: index for index, country in enumerate(countries)}
        fold_data = {}
        for target in targets:
            fold_data[target] = {}
            for model in models:
                mapping = by_group_model[(domain, target, model)]
                fold_data[target][model] = (
                    np.array([country_index[c] for c in mapping], dtype=np.int64),
                    np.array(list(mapping.values()), dtype=np.float64),
                )

        comparisons = [(model, REFERENCE) for model in models if model != REFERENCE]
        if (ADAPTATION_MODEL, ADAPTATION_REFERENCE) not in comparisons:
            comparisons.append((ADAPTATION_MODEL, ADAPTATION_REFERENCE))

        for model, comparator in comparisons:
            observed_model_fold = []
            observed_comparator_fold = []
            for target in targets:
                idx_m, err_m = fold_data[target][model]
                idx_c, err_c = fold_data[target][comparator]
                if not np.array_equal(idx_m, idx_c):
                    raise ValueError(f"Unpaired countries at {domain}/{target}: {model} vs {comparator}")
                observed_model_fold.append(float(np.mean(err_m)))
                observed_comparator_fold.append(float(np.mean(err_c)))
            observed_model = float(np.mean(observed_model_fold))
            observed_comparator = float(np.mean(observed_comparator_fold))
            observed_delta = observed_model - observed_comparator

            bootstrap_deltas = np.empty(N_BOOTSTRAP, dtype=np.float64)
            n_country_clusters = len(countries)
            for b in range(N_BOOTSTRAP):
                draws = rng.integers(0, n_country_clusters, size=n_country_clusters)
                multiplicity = np.bincount(draws, minlength=n_country_clusters)
                fold_deltas = []
                for target in targets:
                    idx_m, err_m = fold_data[target][model]
                    idx_c, err_c = fold_data[target][comparator]
                    weights = multiplicity[idx_m]
                    denominator = weights.sum()
                    if denominator == 0:
                        raise RuntimeError("Bootstrap sample omitted all eligible countries in a fold")
                    paired_loss_diff = err_m - err_c
                    fold_deltas.append(float(np.dot(weights, paired_loss_diff) / denominator))
                bootstrap_deltas[b] = float(np.mean(fold_deltas))

            low, high = np.quantile(bootstrap_deltas, [0.025, 0.975])
            output.append(
                {
                    "domain": domain,
                    "model": model,
                    "comparator": comparator,
                    "n_country_clusters": n_country_clusters,
                    "n_outer_folds": len(targets),
                    "outer_targets": ";".join(map(str, targets)),
                    "model_equal_fold_MAE": round(observed_model, 8),
                    "comparator_equal_fold_MAE": round(observed_comparator, 8),
                    "delta_MAE_model_minus_comparator": round(observed_delta, 8),
                    "bootstrap_95pct_low": round(float(low), 8),
                    "bootstrap_95pct_high": round(float(high), 8),
                    "bootstrap_probability_delta_lt_0": round(float(np.mean(bootstrap_deltas < 0)), 8),
                    "bootstrap_replicates": N_BOOTSTRAP,
                    "seed": SEED,
                    "interpretation": "Exploratory country-cluster interval conditional on these four observed target folds and point score estimates.",
                }
            )

    OUT_DIR.mkdir(parents=True, exist_ok=False)
    write_csv(SUMMARY, output)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": Path(__file__).relative_to(ROOT).as_posix(),
        "script_sha256": sha256(Path(__file__)),
        "input": INPUT.relative_to(ROOT).as_posix(),
        "input_sha256": sha256(INPUT),
        "input_rows": len(rows),
        "unique_prediction_keys": len(key_counts),
        "prediction_key_duplicates": 0,
        "models": models,
        "domains": domains,
        "outer_targets": targets,
        "country_sets_identical_across_models_within_domain_fold": True,
        "resampling": {
            "unit": "country cluster within domain; repeated fold errors for a sampled country stay paired",
            "replicates": N_BOOTSTRAP,
            "seed": SEED,
            "interval": "percentile 95%",
            "estimand": "equal mean across the four outer-fold MAEs; paired model minus comparator",
            "comparators": "every model vs Persistence; TimesFM2_5_ILSA_LoRA additionally vs matched TimesFM2_5_zero_shot",
        },
        "limitations": [
            "The bootstrap is conditional on the four observed historical target cycles and does not quantify between-cycle uncertainty.",
            "It conditions on the published point-score estimates and does not propagate plausible-value, survey-design, or PISA scale-link uncertainty.",
            "Countries are a selected participating set, not a random sample of all possible education systems.",
            "Intervals are exploratory and are not confirmatory significance tests or evidence of prospective performance.",
        ],
        "output": {"path": SUMMARY.relative_to(ROOT).as_posix(), "sha256": sha256(SUMMARY), "rows": len(output)},
        "overwrite_policy": "Output directory must not exist before execution.",
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for row in output:
        print(
            row["domain"], row["model"], "vs", row["comparator"],
            "delta", row["delta_MAE_model_minus_comparator"],
            "CI", f'[{row["bootstrap_95pct_low"]}, {row["bootstrap_95pct_high"]}]',
            "p(delta<0)", row["bootstrap_probability_delta_lt_0"],
        )
    print(f"Wrote {len(output)} comparisons to {OUT_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
