# Phase 3.13 — PISA 2025 Sensitivity and Paired Country Bootstrap

**Date:** 2026-10-15  
**Status:** Executed on saved forecasts; no model refitting or 2025-based tuning  
**Role:** Outcome-exposed retrospective sensitivity analysis  
**Run:** `outputs/pisa2025_forecast_sensitivity_v7/run_manifest.json`

## Design and verification

The analysis uses the 1,584 saved predictions from the common set of 66 countries, three domains, and eight methods. It re-computes forecast errors from `prediction - true_score`; the full-sample MAE, RMSE, and signed error reproduce the Phase 3.10 metrics to a maximum floating-point difference below `8e-15`.

Four descriptive samples are reported: all 66 countries; the 60 without an OECD sampling-quality caution flag; all 65 after excluding Albania; and the clean sample after excluding Albania. **Albania is one of the six flagged countries**, so excluding it after removing the six flagged countries leaves 60, not 59. The earlier protocol's “59 for clean” parenthetical is an arithmetic error and is corrected here.

For uncertainty, 20,000 paired bootstrap samples resample countries within each domain and preserve each sampled country's paired model errors. The principal comparator is Persistence; the adaptation-specific comparator is TimesFM 2.5 zero-shot. Intervals are percentile 95% intervals. They describe resampling variation over these observed countries conditional on the published point means; they do not propagate PISA's complex sampling/score-estimation uncertainty. The participating countries are not a random sample of all possible countries, and PISA 2025 is outcome-exposed, so these are **exploratory intervals, not confirmatory significance tests**. No Diebold–Mariano test is used for this single target cycle.

## Error rankings remain stable in the main sensitivity sets

| Sample | n per domain | Mathematics lowest MAE | Reading lowest MAE | Science lowest MAE |
|---|---:|---|---|---|
| All eligible countries | 66 | XGBoost, 9.32 | Ridge, 11.36 | Persistence, 9.38 |
| Exclude six quality-flagged countries | 60 | XGBoost, 8.86 | Ridge, 10.90 | Persistence, 8.95 |
| Exclude Albania only | 65 | XGBoost, 8.61 | Ridge, 10.91 | Persistence, 8.62 |

The leading method in each domain does not change in these three samples. The clean-after-Albania sample is also `n=60`, because Albania is already removed with the other five flagged countries. Full model-by-domain metrics for all subsets are in `subset_and_context_metrics.csv`.

## Paired country-bootstrap comparisons

Selected comparisons against Persistence are shown below as `MAE(model) - MAE(Persistence)`; negative values favor the named model. They are not multiplicity-adjusted and should be read descriptively.

| Domain | Model | Δ MAE | Exploratory 95% interval |
|---|---|---:|---:|
| Mathematics | XGBoost | −2.27 | [−3.63, −0.86] |
| Mathematics | Ridge | −2.13 | [−3.29, −1.02] |
| Mathematics | TimesFM 3.0 zero-shot | +1.28 | [−0.28, +2.78] |
| Reading | Ridge | −3.86 | [−4.96, −2.80] |
| Reading | XGBoost | −2.28 | [−3.75, −0.78] |
| Reading | TimesFM 3.0 zero-shot | +1.53 | [−0.18, +3.23] |
| Science | Ridge | +0.13 | [−1.33, +1.59] |
| Science | XGBoost | +0.35 | [−0.87, +1.63] |
| Science | TimesFM 3.0 zero-shot | +0.70 | [−0.60, +1.95] |

The observed 2025 rankings are compatible with the point-estimate table: learned tabular models have lower errors than persistence in mathematics and reading, while persistence has the lowest point MAE in science. This is a country-resampling description of one exposed cycle, not proof of a general performance ordering.

## Does ILSA LoRA improve its matched TimesFM 2.5 baseline?

TimesFM 2.5 LoRA minus its matched zero-shot model has lower MAE in all three domains, but the effect is small:

| Domain | Δ MAE LoRA − zero-shot | Exploratory paired 95% interval | Probability bootstrap Δ < 0 |
|---|---:|---:|---:|
| Mathematics | −0.159 | [−0.228, −0.096] | 1.000 |
| Reading | −0.132 | [−0.233, −0.038] | 0.997 |
| Science | −0.166 | [−0.296, −0.063] | 1.000 |

These intervals exclude zero under the specified country bootstrap, but the effect is less than 0.2 score points in each domain. The country sample is selected, the benchmark is outcome-exposed, multiple comparisons were not adjusted, and uncertainty in the national score estimates was not propagated. The defensible conclusion is that the saved 2025 predictions show a **small, directionally consistent within-family reduction in MAE**, not a practically important or confirmed general transfer advantage. LoRA still does not beat the best method in any domain.

## Context-length strata

Context counts and every model's metrics are reported for 3, 4, 5, 6, and 7+ pre-2025 PISA cycles in `subset_and_context_metrics.csv`. Several strata have only 4–7 countries, so their apparent best methods are unstable and are not used to claim subgroup superiority. The largest group is 7+ cycles (mathematics/science `n=36`, reading `n=41`).

## Reproducibility and next action

`scripts/summarize_pisa_forecast_sensitivity.py` recreates the metrics and bootstrap output from the hashed forecast file, quality-flag file, and fixed seed. The complete results and versions are recorded in the run manifest. Run it again to a new directory with `--output outputs/pisa2025_forecast_sensitivity_reproduction`; the script refuses to overwrite an existing directory. Model metrics have not been altered; Phase 3.10's erroneous science linear-trend RMSE/bias and its description of the 2025 LoRA science comparison have been corrected against the saved row-level forecasts.

The remaining model-evaluation limitation is external validity: 2025 cannot become a blind test after prior project exposure. A future PISA cycle would need an outcome-independent, frozen protocol to serve as prospective confirmation.
