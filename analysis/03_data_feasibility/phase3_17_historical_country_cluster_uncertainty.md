# Phase 3.17 — Historical PISA Forecast Comparison: Country-Cluster Uncertainty

**Date:** 2026-10-15  
**Status:** Exploratory paired country-cluster bootstrap completed; no model was refit.  
**Input predictions:** `outputs/pisa_timesfm25_ils_adapter_v1/all_models_historical_predictions.csv`  
**Script:** `scripts/summarize_historical_country_cluster_bootstrap.py`  
**Output:** `outputs/historical_country_cluster_bootstrap_v1/equal_fold_paired_country_bootstrap.csv`  
**Manifest:** `outputs/historical_country_cluster_bootstrap_v1/run_manifest.json`

## Question and estimand

The four historical outer targets (2012, 2015, 2018, 2022) reuse some countries. Treating every country-fold row as independent would understate dependence. This analysis resamples **country clusters within each domain**, keeps a sampled country's paired model errors together over its available folds, and calculates model-minus-comparator differences in the project's equal-fold mean MAE. It uses 20,000 replicates and a fixed seed (`20261016`).

The intervals are conditional on these four observed targets and the saved point-score estimates. They do not quantify uncertainty across future assessment cycles, propagate PISA sampling/plausible-value/scale-link uncertainty, or generalize to a random population of countries. They are descriptive/exploratory and are not confirmatory significance tests. No multiplicity adjustment is applied.

## Main paired comparisons

Entries are `MAE(model) − MAE(Persistence)`; positive values favor Persistence. Intervals are percentile 95% country-cluster bootstrap intervals.

| Domain | Model | Δ equal-fold MAE | Exploratory 95% interval |
|---|---|---:|---:|
| Mathematics | XGBoost | +0.24 | [−0.34, +0.84] |
| Mathematics | Random Forest | +0.48 | [−0.14, +1.16] |
| Mathematics | TimesFM 3.0 zero-shot | +1.30 | [+0.60, +2.04] |
| Mathematics | TimesFM 2.5 ILSA-LoRA | +0.45 | [−0.11, +1.03] |
| Reading | XGBoost | +0.22 | [−0.73, +1.25] |
| Reading | Random Forest | +0.60 | [−0.27, +1.54] |
| Reading | TimesFM 3.0 zero-shot | +1.34 | [+0.51, +2.24] |
| Reading | TimesFM 2.5 ILSA-LoRA | +0.12 | [−0.43, +0.67] |
| Science | XGBoost | +0.12 | [−0.40, +0.66] |
| Science | Random Forest | +0.73 | [+0.19, +1.29] |
| Science | TimesFM 3.0 zero-shot | +1.26 | [+0.72, +1.84] |
| Science | TimesFM 2.5 ILSA-LoRA | +0.34 | [−0.16, +0.84] |

The bootstrap is consistent with the point estimate that Persistence has the lowest equal-fold MAE in each domain. The intervals for TimesFM 3.0 versus Persistence are above zero across all domains in this conditional analysis. Most learned tabular model comparisons overlap zero; this does not establish equivalence or superiority. Random Forest's science difference is above zero under this specified bootstrap. These are exploratory patterns across the observed folds, not evidence that the same ranking will hold in a later cycle.

## Does ILSA adaptation improve its matched TimesFM 2.5 comparator?

Entries below are `MAE(TimesFM 2.5 ILSA-LoRA) − MAE(TimesFM 2.5 zero-shot)`; negative values favor adaptation.

| Domain | Δ equal-fold MAE | Exploratory 95% interval |
|---|---:|---:|
| Mathematics | −0.10 | [−0.19, −0.01] |
| Reading | −0.11 | [−0.20, −0.04] |
| Science | −0.07 | [−0.15, +0.01] |

This indicates a small directionally favorable adaptation difference in these saved historical forecasts. Two domain-specific intervals exclude zero under the country-cluster resampling scheme; the science interval overlaps zero. The magnitude remains under 0.12 MAE points in each domain, the validation sets for two outer folds were small, and no adjustment was made for multiple comparisons. The result should not be upgraded to a practically important or prospective transfer benefit.

## Verification

- The input contains 4,992 predictions: 4 targets × 3 domains × 8 methods × fold-specific country counts.
- Model/country key uniqueness and finite forecasts were checked before resampling.
- All eight methods have identical country sets within each target/domain; all paired differences therefore use matched rows.
- The script recomputes absolute errors from raw `prediction` and `true_score` fields.
- The manifest records input and code hashes, model list, fold list, seed, replicate count, output hash, and inferential limitations.
- Output directory creation is fail-fast: an existing version is not overwritten.

## Interpretation alongside PISA 2025

The historical equal-fold comparison has Persistence as the point-estimate leader in all three domains. The outcome-exposed PISA 2025 retrospective benchmark instead has XGBoost leading mathematics, Ridge leading reading, and Persistence leading science. This cycle-to-cycle variation is a reason to retain temporal validation and eventual prospective confirmation. Because PISA 2025 outcomes and prior project results were visible before the current model comparison, 2025 remains a retrospective benchmark, not a blinded final test.
