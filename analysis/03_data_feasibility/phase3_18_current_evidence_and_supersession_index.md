# Phase 3.18 — Current Evidence and Supersession Index

**Date:** 2026-10-20
**Purpose:** Single current-state reference for the ILSA country-level forecasting study. Earlier dated reports are retained as historical records; where they conflict with this index and the reports cited here, use the newer evidence.

## Current research target

The executed forecasting target is country-level mean PISA achievement in mathematics, reading, and science, one assessment cycle ahead. It is not individual-student achievement. PISA 2025 is an outcome-exposed retrospective benchmark in this project and is not a blind final test.

## Evidence status

| Requirement | Current evidence | Status / limit |
|---|---|---|
| PISA panel for three domains with canonical country IDs | `outputs/stage5/pisa_forecasting_panel_v2.csv` and its manifest; 2000 reading crosswalk applied | Complete for current model panel. Mathematics/science begin in 2003; panel is not eight complete cycles for every domain. |
| Frozen historical temporal folds | `analysis/03_data_feasibility/audit/pisa_historical_folds_v1.json` | Complete: common three-domain targets 2012, 2015, 2018, 2022. These are development/temporal-validation folds. |
| Literature-motivated tabular models and baselines | `phase3_7_historical_tabular_benchmark.md` | Executed: Persistence, linear trend, Ridge, Random Forest, XGBoost. Persistence leads equal-fold MAE in every domain. |
| Quadratic trend literature baseline | `phase3_24_literature_motivated_quadratic_trend_benchmark.md` and `outputs/pisa_tabular_benchmark_v2/` | Added per-country linear-vs-quadratic F-test selection using pre-target cycles only. It does not improve on fixed linear trend or persistence in the four historical folds. |
| TimesFM 3.0 zero-shot | `phase3_8_historical_timesfm_comparison.md` | Executed on matching historical folds. It does not beat Persistence in any domain's equal-fold summary. |
| ILSA-adapted TimesFM | `phase3_9_timesfm25_adaptation_results.md` | Executed with TimesFM 2.5 LoRA and matched 2.5 zero-shot comparator. Historical reduction is <0.2 MAE points/domain; practical value is unproven. |
| PISA 2025 comparison | `phase3_10_pisa2025_retrospective_evaluation.md` and `phase3_25_quadratic_trend_full_model_uncertainty_extension.md` | Extended to 66 common countries × 3 domains × 9 methods. Retrospective/exposure-qualified, not blind. |
| Validation of the 2025 point means | `phase3_12_oecd_pisa2025_mean_validation.md` | 198/198 country-domain means matched to OECD tables; this validates target values in the evaluation sample, not test blindness or all source-pipeline steps. |
| PISA 2025 sensitivity | `phase3_13_pisa2025_sensitivity_uncertainty.md` and `outputs/pisa_target_sampling_uncertainty_v3/`, `outputs/pisa_joint_score_country_uncertainty_v3/` | Target-score SE-only and country-cluster + score-SE sensitivity rerun across all nine methods; both condition on fixed forecasts and omit link uncertainty. |
| Cross-cycle corpus candidates | `phase3_14_cross_cycle_review_and_scope_correction.md`, `phase3_24_literature_motivated_quadratic_trend_benchmark.md`, `phase3_27_literature_candidate_adjudication.md`, and `audit/literature_candidate_adjudication_v1.csv` | Five records source-reviewed. Becker & Coyle (2026) is a verified country-specific PISA 2022 forecast; Kaplan et al. (2025) is a one-cycle-ahead PISA 2025 method precedent with a different displayed target and no observed-2025 evaluation; a related arXiv preprint projects minimum-proficiency percentages rather than mean scores. Two DOI-unmatched records are out of scope. Independent review and original corpus eligibility remain open. |
| Saved search JSON inventory | `phase3_15_saved_search_inventory_audit.md` and `audit/literature_search_inventory_v1/` | 725 local JSONs; 665 distinct DOI across sources; 663 match corpus. Two unmatched records were inspected and excluded from the ILSA forecasting scope. Search execution date, raw export completeness, exact database query execution, and original screening decisions remain unverified. |
| Targeted literature candidates | `phase3_16_targeted_forecasting_literature_search.md` and `phase3_27_literature_candidate_adjudication.md` | Exploratory, not exhaustive. Publisher sources verify Becker–Coyle's country-specific PISA 2022 trend forecast and Kaplan et al.'s PISA 2025 one-cycle-ahead forecast-method example; a related preprint forecasts threshold shares, not mean scores. Original corpus eligibility is unknown; independent review remains open. |
| Historical paired uncertainty | `phase3_17_historical_country_cluster_uncertainty.md`, `outputs/historical_country_cluster_bootstrap_v2/`, and `outputs/pisa_joint_score_country_uncertainty_v3/` | Original eight-model bootstrap retained; v2 and joint score-SE v3 include LinearQuadraticTrend. Conditional on four folds/point estimates, not between-cycle uncertainty. |
| Historical model-input and refit uncertainty | `phase3_26_model_input_uncertainty.md`; corrected script `scripts/run_pisa_model_input_uncertainty_v4.py`; `outputs/pisa_model_input_uncertainty_v4/` | Complete and validated: 2,000 draws, 120,000 fold-level metrics, no duplicate keys. Persistence wins in 97.5%, 98.5%, and 99.75% of draws in mathematics, reading, and science, respectively. v3 remains invalid and excluded. |

## Current supported findings

1. Persistence is the strongest equal-fold point-estimate baseline across the four historical targets in all three domains.
2. TimesFM 3.0 zero-shot has higher equal-fold MAE than Persistence across all three domains in the historical benchmark.
3. TimesFM 2.5 LoRA adaptation modestly reduces historical MAE against its matched 2.5 zero-shot version. The estimated reduction is small; it does not displace Persistence as the historical point-estimate leader.
4. PISA 2025 retrospective rankings differ by domain: XGBoost leads mathematics, Ridge leads reading, and Persistence leads science. The different 2025 ranking illustrates temporal variation; it is not independent prospective confirmation.
5. RF and XGBoost are justified as general predictive ILSA comparators, not as the most common methods among verified country-level future-cycle forecasts.
6. The fixed 1,266-record corpus cannot currently support an exhaustive literature-wide zero/coverage claim. Search-origin and screening records are incomplete; a verified country-specific PISA 2022 trend forecast and a closely related PISA 2025 one-cycle-ahead growth-forecast study were found outside the corpus.
7. The external Becker–Coyle linear/quadratic trend-selection rule was added as a historical comparator using only pre-target scores. It had higher equal-fold MAE than the fixed linear trend and persistence in every domain; RF/XGBoost remain useful general predictive comparators, not literature leaders among confirmed future-cycle country-mean forecasts.
8. In the extended paired country-cluster bootstrap, LinearQuadraticTrend had a positive MAE difference versus Persistence in all domains, with exploratory 95% intervals above zero. This is conditional evidence for these four historical folds and fixed score estimates only.
9. LinearQuadraticTrend was also scored on the outcome-exposed PISA 2025 66-country sample. It did not displace the prior point-MAE leader in any domain; the joint country-cluster/target-score-SE comparison favors Persistence over it in reading but is inconclusive in mathematics and science.

## Older claims that must not be reused without qualification

The following files preserve earlier analysis but contain claims that are now stale, superseded, or too broad for manuscript use:

- `analysis/02_prediction_taxonomy/phase2_1_report.md`, `phase2_2_report.md`, and `phase2_taxonomy_report.md`: unqualified “zero confirmed in the literature/corpus” claims. Use the fixed-corpus and coverage qualification in Phase 3.14–3.16.
- `analysis/03_data_feasibility/phase3_2_integrated_synthesis.md`, `phase3_2_predictive_evidence.md`, and `phase3_2_forecasting_diagnostics.md`: old single-fold findings and statements that PISA 2025 evaluation had not occurred. Use Phases 3.7–3.17 for current results.
- `analysis/03_data_feasibility/phase3_1_1_validation_addendum.md`, `phase3_3_pre_evaluation_check.md`, `phase3_3_final_evaluation_protocol.md`, and `phase3_3_final_readiness_gate.md`: historical language describing a prospective/blind PISA 2025 test. PISA 2025 outcomes and prior predictions were visible in the project before the executed comparison. Use “outcome-exposed retrospective benchmark.”
- Other Phase 3.1 design proposals: retain as protocol history, not current execution evidence.

These old artifacts are not deleted or silently edited. Phase 3.10–3.18 and the hash-locked run outputs are the current evidence trail.

## Remaining gates before manuscript-ready completion

1. Recover raw search exports/query history or record a documented failed retrieval; complete eligibility decisions for both external studies against a defensible protocol. Issue a separately versioned corpus only if the criteria justify membership changes.
2. Complete independent second review of all four candidate-register records and expand recall-oriented screening beyond the heuristic flags before making any exhaustive coverage claim.
3. Target-score sampling-SE and corrected historical context/input/model-refit sensitivity analyses are complete as separate conditional analyses. OECD cross-cycle scale-link covariance remains open. Keep these distinct from country-cluster resampling variation.
4. Perform a final reproduction/audit from saved code and manifests for all model arms, including environment/checkpoint hashes, training cutoffs, equal-key coverage, and output metrics.
5. Consolidate results, methods, limitations, and historical findings into a single manuscript-facing report and ensure it contains no superseded claims.
6. For genuine independent confirmation, lock an outcome-independent protocol and hash/freeze predictions before a future PISA cycle's target scores become available. PISA 2025 cannot be made blind retroactively.

## Completion decision

The project has delivered a reproducible, leakage-controlled **retrospective/historical benchmark** across the requested model families and all three PISA domains, with PISA 2025 scores independently checked against OECD tables. The broader objective is not yet fully complete: literature-search completeness and a genuinely independent future-cycle evaluation remain unresolved. This index does not certify manuscript readiness or an exhaustive systematic review.
