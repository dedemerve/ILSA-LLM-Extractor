# Phase 3.25 — Linear/quadratic trend added to the full-model retrospective evaluation

**Date:** 2026-10-15

**Status:** The literature-motivated trend-selection baseline was added to the PISA 2025 outcome-exposed retrospective comparison and to both existing target-score uncertainty analyses. Existing model predictions and prior uncertainty runs were not modified.

**Prediction construction:** [`scripts/run_pisa_2025_quadratic_trend_retrospective.py`](../../scripts/run_pisa_2025_quadratic_trend_retrospective.py), [`scripts/build_quadratic_trend_all_model_prediction_panels.py`](../../scripts/build_quadratic_trend_all_model_prediction_panels.py)

**Uncertainty runs:** [`scripts/summarize_quadratic_trend_target_se_uncertainty_v1.py`](../../scripts/summarize_quadratic_trend_target_se_uncertainty_v1.py), [`scripts/summarize_quadratic_trend_joint_uncertainty_v1.py`](../../scripts/summarize_quadratic_trend_joint_uncertainty_v1.py)

**Outputs:** [`outputs/pisa_2025_quadratic_trend_retrospective_v1/`](../../outputs/pisa_2025_quadratic_trend_retrospective_v1/), [`outputs/pisa_all_models_quadratic_trend_v1/`](../../outputs/pisa_all_models_quadratic_trend_v1/), [`outputs/pisa_target_sampling_uncertainty_v3/`](../../outputs/pisa_target_sampling_uncertainty_v3/), [`outputs/pisa_joint_score_country_uncertainty_v3/`](../../outputs/pisa_joint_score_country_uncertainty_v3/)

## PISA 2025 retrospective results

The linear-versus-quadratic selection rule uses PISA scores strictly before 2025. Selection diagnostics and fitted contexts were checked against the saved PISA panel. The 2025 target is attached only for scoring and comparison. This is an **outcome-exposed retrospective evaluation**, not a blind forecast test.

The common 66-country/three-domain sample matches the existing eight-method benchmark. The new method produced 198 unique finite forecasts. A quadratic was selected for 30 forecasts; 14 used linear trend because there were only three usable prior observations; the remaining 154 were tested and retained linear trend because the quadratic term did not reach the prespecified p < .05 threshold.

| Domain | LinearQuadraticTrend MAE | Existing best point-MAE method | Existing best MAE |
|---|---:|---|---:|
| Mathematics | 12.728 | XGBoost | 9.318 |
| Reading | 18.277 | Ridge | 11.357 |
| Science | 10.194 | Persistence | 9.378 |

Thus the added trend-selection baseline does not change the leading point-MAE method in any PISA 2025 domain. In the historical four-fold comparison it also did not beat fixed linear trend or persistence (Phase 3.24).

## Country-score uncertainty extension

The two existing uncertainty workflows were rerun in separate v3 output directories with all nine methods. Predictions remain fixed. The target-only sensitivity draws one published-SE normal target per country/domain/cycle and shares it across methods. The joint sensitivity combines country-cluster resampling with those target-score draws. Neither analysis propagates link error or refits models.

The paired results below report `MAE(LinearQuadraticTrend) - MAE(Persistence)`. Positive values favor Persistence. Intervals are conditional Monte Carlo percentiles, not confirmatory tests.

| Evaluation | Domain | Point Δ MAE | Target-SE-only 95% interval | Joint country + score-SE 95% interval |
|---|---|---:|---:|---:|
| Historical four-fold equal-weighted | Mathematics | +2.236 | [+1.805, +2.380] | [+1.073, +3.216] |
| Historical four-fold equal-weighted | Reading | +3.942 | [+3.564, +4.197] | [+2.426, +5.430] |
| Historical four-fold equal-weighted | Science | +3.228 | [+2.836, +3.356] | [+2.063, +4.162] |
| PISA 2025 all 66, retrospective | Mathematics | +1.144 | [+0.568, +1.725] | [−1.021, +3.194] |
| PISA 2025 all 66, retrospective | Reading | +3.061 | [+2.833, +3.409] | [+0.459, +5.649] |
| PISA 2025 all 66, retrospective | Science | +0.816 | [+0.162, +1.370] | [−0.991, +2.540] |

The quality-clean 60-country joint sensitivity yields LQ-minus-Persistence intervals of [−0.480, +3.438] for mathematics, [+1.382, +6.247] for reading, and [−0.722, +2.769] for science. Therefore, for 2025 the uncertainty-aware comparison remains inconclusive in mathematics and science, while the observed country-cluster/score-SE procedure favors Persistence over the trend baseline in reading. This remains retrospective and conditional on point forecasts and the chosen resampling model.

Under the all-66 joint simulation, the model with the lowest MAE most often was XGBoost in mathematics (54.3%), Ridge in reading (99.6%), and Persistence in science (40.2%). The quadratic trend model's corresponding winner frequencies were 0.1%, 0.0%, and 8.5%. The non-dominant science distribution shows why a point-MAE leader alone should not be described as clearly superior.

## Reproducibility and audit checks

- Historical unified panel: 5,616 rows = 9 methods × 624 targets. PISA 2025 unified panel: 1,782 rows = 9 methods × 198 targets.
- Existing five tabular method predictions match exactly between their original tabular and TimesFM all-method files in both evaluation sets. The newly added model matches the existing Persistence target keys and published score rows.
- Historical country/context keys, PISA 2025 66-country keys, and prediction uniqueness were checked. PISA 2025 context cycles and scores were regenerated from the panel and matched to the existing benchmark.
- Target-SE-only analysis wrote 162 metric summaries and 51 paired comparisons. Joint country-cluster + target-SE analysis covered historical folds, PISA 2025 all 66, and the quality-clean 60; output manifests record input, code, and output hashes.
- All new scripts and output directories are versioned separately and fail rather than overwrite an existing result.

## Limits and interpretation

PISA 2025 cannot be restored as a blind final test because its outcomes were already present in the project. The PISA score SE analyses perturb only target estimates while holding the nine forecasts fixed. The joint country resampling is conditional on the selected participating countries and does not quantify between-cycle uncertainty. The prior OECD link-covariance diagnostic also remains unresolved for science; the results here omit scale-link error. Historical model-input uncertainty, TimesFM refitting uncertainty, and prospective confirmation are still open requirements. Do not call these fully uncertainty-propagated forecasts.
