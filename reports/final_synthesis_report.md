# Final synthesis: forecasting national PISA mean scores

**Report date:** 2026-10-20  
**Evidence status:** Integrates Phases 3.3 and 3.25–3.30 and the cited prediction/uncertainty outputs. The existing forecasts and analysis inputs were not changed to prepare this report.

## 1. Research question and prediction unit

This project evaluates how well selected statistical, machine-learning, and time-series methods forecast **country/economy-level PISA mean scores** in mathematics, reading, and science. The unit is a country/domain/cycle mean, not an individual student score. Historical evaluation uses rolling-origin targets in 2012, 2015, 2018, and 2022; the PISA 2025 comparison is retrospective and outcome-exposed because 2025 results were already present in the project.

The eligible samples are common-sample country/economy subsets meeting the project’s cycle-history and data-quality rules. They are not a census of all countries. The historical three-domain common samples contain 40, 51, 57, and 60 country/economy observations in the four target folds, respectively; the PISA 2025 common sample contains 66 country/economies in each domain. These counts are recorded in the historical and retrospective prediction panels and Phase 3.24/3.25 reports.

## 2. Data and evaluation design

Historical predictions use information available before each target cycle. The four rolling-origin folds are summarized by giving each fold equal weight; the historical MAE below is the arithmetic mean of the four fold-specific MAEs, not a pooled-country MAE. The project compares nine methods: Persistence, LinearTrend, LinearQuadraticTrend, Ridge, RandomForest, XGBoost, TimesFM 3.0 zero-shot, TimesFM 2.5 zero-shot, and TimesFM 2.5 adapted to ILSA data with LoRA. The historical results use common country/domain/target keys in the unified panel.

For 2025, all nine methods are evaluated on the same 66 countries per domain. The 2025 outcome is used only to score already produced forecasts in the retrospective comparison. This is not a blind test or an independent prospective validation.

## 3. Main findings: historical four-fold comparison

**Table 1. Equal-fold-weighted MAE in PISA score points.** Each cell is the mean of that method’s four fold-specific MAEs. Values are rounded to three decimals. Source: [`historical_all_models_predictions.csv`](../outputs/pisa_all_models_quadratic_trend_v1/historical_all_models_predictions.csv), grouped by domain, model, and target cycle; target-cycle sample sizes are 40/51/57/60. The original tabular and TimesFM fold metrics are in [`historical_metrics.csv` for tabular v2](../outputs/pisa_tabular_benchmark_v2/historical_metrics.csv) and [`historical_metrics.csv` for TimesFM zero-shot v2](../outputs/pisa_timesfm_historical_zeroshot_v2/historical_metrics.csv).

| Domain | Persistence | LinearTrend | LinearQuadraticTrend | Ridge | RandomForest | XGBoost | TimesFM 3.0 zero-shot | TimesFM 2.5 zero-shot | TimesFM 2.5 ILSA LoRA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Mathematics | 9.328 | 11.297 | 11.564 | 9.961 | 9.811 | 9.563 | 10.629 | 9.873 | 9.773 |
| Reading | 10.156 | 13.333 | 14.099 | 10.834 | 10.759 | 10.379 | 11.495 | 10.385 | 10.273 |
| Science | 8.442 | 10.799 | 11.671 | 9.124 | 9.168 | 8.565 | 9.704 | 8.850 | 8.783 |

Persistence has the lowest equal-fold MAE in all three domains in this historical summary. XGBoost is the next-lowest point MAE in mathematics and science; TimesFM 2.5 LoRA is next-lowest in reading. These are observed rankings over four temporal folds and their eligible samples. They do not establish general method superiority; folds reuse countries, and 2022 is a distinct pandemic-era target.

## 4. PISA 2025 outcome-exposed retrospective comparison

**Table 2. PISA 2025 MAE in score points (66 countries per domain).** Source: [`pisa2025_all_models_predictions.csv`](../outputs/pisa_all_models_quadratic_trend_v1/pisa2025_all_models_predictions.csv), grouped by domain and model. The source panel contains 1,782 rows (nine methods × 66 countries × three domains). The tabular v1 and quadratic trend reports independently provide corresponding subsets: [`tabular_metrics.csv`](../outputs/pisa_2025_retrospective_tabular_v1/tabular_metrics.csv) and [`tabular_metrics.csv`](../outputs/pisa_2025_quadratic_trend_retrospective_v1/tabular_metrics.csv).

| Domain | Persistence | LinearTrend | LinearQuadraticTrend | Ridge | RandomForest | XGBoost | TimesFM 3.0 zero-shot | TimesFM 2.5 zero-shot | TimesFM 2.5 ILSA LoRA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Mathematics | 11.584 | 15.188 | 12.728 | 9.457 | 11.557 | **9.318** | 12.864 | 14.989 | 14.831 |
| Reading | 15.216 | 19.584 | 18.277 | **11.357** | 13.032 | 12.932 | 16.750 | 17.849 | 17.717 |
| Science | **9.378** | 10.582 | 10.194 | 9.506 | 9.752 | 9.731 | 10.076 | 10.767 | 10.600 |

The observed point-MAE leaders are XGBoost in mathematics, Ridge in reading, and Persistence in science. TimesFM 3.0 zero-shot is not the point-MAE leader in any of the three domains. The 2025 rankings describe this selected sample and this outcome-exposed retrospective exercise; they are not independent forward-looking evidence or a universal ranking.

## 5. Uncertainty analyses

### Target-score sampling and country-cluster uncertainty

Phase 3.25 re-evaluated fixed predictions while varying the held-out country score estimates using their published marginal SEs, and separately combining those target-score draws with country-cluster resampling. The paired target was `MAE(LinearQuadraticTrend) − MAE(Persistence)`; positive values favor Persistence. These are conditional Monte Carlo intervals, not confirmatory hypothesis tests.

| Evaluation | Domain | Point Δ MAE | Joint 95% interval: country clusters + target-score SE |
|---|---|---:|---:|
| Historical four-fold equal-weighted | Mathematics | +2.236 | [+1.073, +3.216] |
| Historical four-fold equal-weighted | Reading | +3.942 | [+2.426, +5.430] |
| Historical four-fold equal-weighted | Science | +3.228 | [+2.063, +4.162] |
| PISA 2025, all 66 | Mathematics | +1.144 | [−1.021, +3.194] |
| PISA 2025, all 66 | Reading | +3.061 | [+0.459, +5.649] |
| PISA 2025, all 66 | Science | +0.816 | [−0.991, +2.540] |

Source: [`joint_paired_comparisons.csv`](../outputs/pisa_joint_score_country_uncertainty_v3/joint_paired_comparisons.csv), filtered to `historical_equal_fold_joint` and `pisa2025_all66_joint`; the design and interpretation are documented in [Phase 3.25](../analysis/03_data_feasibility/phase3_25_quadratic_trend_full_model_uncertainty_extension.md). Historical intervals are conditional on four folds and the chosen country-cluster procedure. For the 2025 trend-versus-persistence comparison, the interval includes zero in mathematics and science; reading’s interval is positive under this procedure.

### Historical model-input perturbation and refitting

The corrected Phase 3.26 run uses 2,000 draws of historical country/domain/cycle scores from marginal normal distributions based on published score SEs. It perturbs training/context inputs and labels, refits Ridge, RandomForest, and XGBoost within each draw, and keeps held-out target means fixed. Persistence and LinearTrend are recomputed from the perturbed context. TimesFM is excluded from this refit-aware run. The original checkpoint series mixed two software environments for RF/XGBoost draws 0–199 versus 200–1,999; the results below use the uniform, version-matched rerun. The original directory remains preserved for audit.

| Domain | Model | Mean equal-fold MAE over 2,000 draws | 2.5th–97.5th Monte Carlo quantiles |
|---|---|---:|---:|
| Mathematics | Persistence | 9.639 | 9.287–9.986 |
| Mathematics | Ridge | 10.164 | 9.651–10.714 |
| Mathematics | RandomForest | 10.140 | 9.668–10.628 |
| Mathematics | XGBoost | 10.782 | 10.054–11.548 |
| Mathematics | LinearTrend | 11.616 | 11.085–12.147 |
| Reading | Persistence | 10.435 | 10.077–10.783 |
| Reading | Ridge | 11.010 | 10.549–11.492 |
| Reading | RandomForest | 11.061 | 10.596–11.520 |
| Reading | XGBoost | 11.240 | 10.523–11.984 |
| Reading | LinearTrend | 13.658 | 13.176–14.170 |
| Science | Persistence | 8.730 | 8.384–9.078 |
| Science | Ridge | 9.366 | 8.874–9.929 |
| Science | RandomForest | 9.363 | 8.903–9.841 |
| Science | XGBoost | 9.776 | 9.134–10.453 |
| Science | LinearTrend | 11.127 | 10.615–11.673 |

Source: [`model_input_uncertainty_metrics.csv`](../outputs/reproducibility_check_v3/pisa_model_input_uncertainty_v4/model_input_uncertainty_metrics.csv); paired deltas versus Persistence are in [`model_input_uncertainty_paired.csv`](../outputs/reproducibility_check_v3/pisa_model_input_uncertainty_v4/model_input_uncertainty_paired.csv). Across these draws, Persistence has the lowest equal-fold MAE. The intervals reflect the stated marginal-SE sampling design only; they are not full predictive intervals and exclude cross-cycle and cross-country dependence, held-out target uncertainty, link covariance, future-cycle variation, and model specification uncertainty. Phase 3.25 and 3.26 are separate conditional analyses and must not be combined as independent components into one interval.

Persistence had the lowest equal-fold MAE in **97.45%** of mathematics draws, **98.55%** of reading draws, and **99.75%** of science draws. These are empirical winner frequencies under this 2,000-draw marginal-SE input perturbation design, not general probabilities that Persistence will be best on future PISA cycles. Source: [`model_input_uncertainty_winner_prob.csv`](../outputs/reproducibility_check_v3/pisa_model_input_uncertainty_v4/model_input_uncertainty_winner_prob.csv). The paired comparison file also shows that every tested alternative’s 95% Monte Carlo interval for MAE difference versus Persistence was above zero in each domain; the interpretation remains conditional on this input sampling design.

The mismatch is traced to the original checkpoint series mixing software environments: tree-model draws 0–199 match the older-library rerun, while draws 200–1,999 match the source-manifest-version rerun. Persistence, LinearTrend, and Ridge match both environments. The results above use the uniform, version-matched rerun in [`reproducibility_check_v3`](../outputs/reproducibility_check_v3/pisa_model_input_uncertainty_v4/); original Phase 3.26 files remain unchanged and are retained for audit. See [Phase 3.32](../analysis/03_data_feasibility/phase3_32_reproducibility_followup.md).

### Link-error covariance

Phase 3.22 reconstructed 21 pairwise cycle-to-2025 link-error SEs from OECD score and score-difference SEs across 1,280 country/economy rows; all 21 summaries matched the NCES table to three decimals. Phase 3.23 tested whether the OECD 2018/2022/2025 trend-link SEs could be represented by two correlated additive offsets. Mathematics and reading passed that specific necessary bound; science did not. Phase 3.29 documents the source-table discrepancies and concludes that no full historical covariance matrix for this project’s input vector has yet been identified and validated. Therefore, none of the uncertainty intervals above includes PISA scale-link covariance.

## 6. Literature-scope and reproducibility audits

The 1,266 records are a coded literature corpus, not verified database search-hit counts. Phase 3.28 and the subsequent workspace-wide recovery audit found 725 saved extraction JSON files (423 Scopus, 302 Web of Science), but no original executed-query history, raw database exports, hit totals, export completeness, deduplication, or screening records. Five borderline candidates were independently outcome-coded: two describe country-level PISA mean-score forecasting (with an estimand qualification for the sex-stratified projections), one projects proficiency-threshold shares rather than mean scores, and two do not forecast ILSA country mean scores. Phase 3.31 reconstructed Becker–Coyle's country/domain 2022 predictions from the publisher supplement's R scripts and compared them with the 60-country project PISA 2022 fold. Its reconstructed forecasts had higher MAE than Persistence, Ridge, Random Forest, and XGBoost in all three domains; the supplement's observed target means and project target scores differ by at most 0.0371 points across 180 matched rows. See [Phase 3.31](../analysis/03_data_feasibility/phase3_31_becker_coyle_and_search_provenance.md) and the versioned [comparison outputs](../outputs/becker_coyle_comparison_v4/). This is a reconstruction, not a publisher-supplied prediction table, and the source method's R-squared F-test differs from the project's standard nested-regression test.

Phase 3.30’s end-to-end audit found one exact rerun: the pairwise link-error audit was rerun in `outputs/reproducibility_check_v1/` and both output CSVs matched their recorded hashes byte-for-byte. Other requested reruns were classified **MISMATCH** where manifests did not match the requested run/artifacts or lacked essential provenance, and **INFEASIBLE** where safe output redirection was unavailable or the run was explicitly excluded because it required TimesFM. The v1 model-input uncertainty directory named in the audit request is absent; the completed Phase 3.26 result is instead in v4. No prediction outputs were overwritten. Details and per-run classifications are in [Phase 3.30](../analysis/03_data_feasibility/phase3_30_reproducibility_audit.md).

## 7. Limitations and interpretation

1. **No blind 2025 test:** PISA 2025 is outcome-exposed and retrospective. It cannot support an independent prospective claim. Independent confirmation requires preregistering and freezing the protocol, data cutoffs, code, model settings, and forecasts before a future PISA outcome is released.
2. **Country-level estimand and limited sample:** Results concern selected country/economy mean scores, not individual students or every education system. The historical folds share countries and provide only four target cycles.
3. **Literature scope is incomplete:** The saved inventory does not reconstruct the original systematic-search execution or establish corpus recall. Avoid describing RF/XGBoost as the most-used methods in a verified country-level future-cycle ILSA forecasting corpus.
4. **Incomplete uncertainty propagation:** Target-score sampling, country clusters, and historical model-input/refitting uncertainty were analyzed in separate conditional designs. Scale-link covariance, a joint combination of all uncertainty components, TimesFM refitting uncertainty, model-specification uncertainty, and future-cycle variability are not included.
5. **Reproducibility gaps:** Independent reruns confirmed byte-identical historical tabular benchmark v2 and all TimesFM 3.0, TimesFM 2.5 LoRA, and PISA 2025 TimesFM CSVs. Historical tabular v1 and PISA 2025 tabular predictions are numerically equivalent within floating-point tolerance. The Phase 3.26 tree-model discrepancy was traced to mixed package environments across checkpoint batches; corrected results now use a single version-matched 2,000-draw run, and the original output remains preserved for audit. See [Phase 3.32](../analysis/03_data_feasibility/phase3_32_reproducibility_followup.md).
6. **No causal interpretation:** Feature contributions or predictive rankings do not show that a factor causes national achievement changes. Country-level correlations and model importance are descriptive.

Accordingly, report model rankings as observed in these defined samples and protocols, with the uncertainty scope stated beside each result. Do not claim that one method is generally superior or that the intervals are fully uncertainty-adjusted.

## 8. Next steps

1. Obtain original Scopus/WoS search histories, raw exports, date/filter/hit-count records, and screening decisions from the original researcher/database accounts; the local workspace search found none. Obtain an independent second reviewer for borderline study coding.
2. Obtain OECD clarification and the full machine-usable historical covariance inputs. Require positive semidefiniteness and reproduction of official pairwise and trend link-error values before propagating link uncertainty.
3. Keep Phase 3.25 target-score/country-cluster sensitivity separate from Phase 3.26 model-input/refitting sensitivity unless a justified joint sampling design is implemented.
4. Execute the prepared [PISA 2029 forecast-lock protocol](../analysis/03_data_feasibility/prospective_pisa_forecast_lock_protocol_v1.md): finalize freeze date, data cutoff, eligibility list, and access controls; hash and seal the code, inputs, environment, and forecasts before target results are accessible. OECD schedules the 2029 main survey and expects core results in 2030; exact publication date remains TBD.

## Provenance index

Primary result sources are the local prediction panels and uncertainty CSVs linked above. Method interpretation and audit status come from [`Phase 3.25`](../analysis/03_data_feasibility/phase3_25_quadratic_trend_full_model_uncertainty_extension.md), [`Phase 3.26`](../analysis/03_data_feasibility/phase3_26_model_input_uncertainty.md), [`Phase 3.28`](../analysis/03_data_feasibility/phase3_28_literature_scope_audit.md), [`Phase 3.29`](../analysis/03_data_feasibility/phase3_29_link_error_covariance_resolution.md), [`Phase 3.30`](../analysis/03_data_feasibility/phase3_30_reproducibility_audit.md), [`Phase 3.31`](../analysis/03_data_feasibility/phase3_31_becker_coyle_and_search_provenance.md), and [`Phase 3.32`](../analysis/03_data_feasibility/phase3_32_reproducibility_followup.md). The prospective runbook is in [`prospective_pisa_forecast_lock_protocol_v1.md`](../analysis/03_data_feasibility/prospective_pisa_forecast_lock_protocol_v1.md). Phase 3.3 is retained as an earlier provisional account and explicitly marks superseded conclusions; this synthesis uses its relevant historical context only where later evidence has not superseded it.
