# Thesis numbers (generated; do not edit by hand)

Every number below is computed from the listed output file by `scripts/build_thesis_numbers_v1.py`.
Rows with an expected value were checked against the Phase 6.0 claim map.

## RQ1

| ID | Description | Value | Check | Source |
|---|---|---|---|---|
| `rq1.geo_rel_mae.Persistence` | Geometric-mean relative MAE vs persistence, Persistence | 1.000 | ok | `outputs/rq1_tables_v2/rq1_aggregate_relative_mae.csv` |
| `rq1.geo_rel_mae.AR1` | Geometric-mean relative MAE vs persistence, AR1 | 1.027 | ok | `outputs/rq1_tables_v2/rq1_aggregate_relative_mae.csv` |
| `rq1.geo_rel_mae.Drift` | Geometric-mean relative MAE vs persistence, Drift | 1.030 | ok | `outputs/rq1_tables_v2/rq1_aggregate_relative_mae.csv` |
| `rq1.geo_rel_mae.Ridge_M0A` | Geometric-mean relative MAE vs persistence, Ridge_M0A | 1.035 | ok | `outputs/rq1_tables_v2/rq1_aggregate_relative_mae.csv` |
| `rq1.geo_rel_mae.Ridge_M1` | Geometric-mean relative MAE vs persistence, Ridge_M1 | 1.093 | ok | `outputs/rq1_tables_v2/rq1_aggregate_relative_mae.csv` |
| `rq1.geo_rel_mae.Ridge_M0` | Geometric-mean relative MAE vs persistence, Ridge_M0 | 1.095 | ok | `outputs/rq1_tables_v2/rq1_aggregate_relative_mae.csv` |
| `rq1.geo_rel_mae.XGB_score` | Geometric-mean relative MAE vs persistence, XGB_score | 1.182 | ok | `outputs/rq1_tables_v2/rq1_aggregate_relative_mae.csv` |
| `rq1.geo_rel_mae.XGB_cov` | Geometric-mean relative MAE vs persistence, XGB_cov | 1.195 | ok | `outputs/rq1_tables_v2/rq1_aggregate_relative_mae.csv` |
| `rq1.geo_rel_mae.RF_score` | Geometric-mean relative MAE vs persistence, RF_score | 1.218 | ok | `outputs/rq1_tables_v2/rq1_aggregate_relative_mae.csv` |
| `rq1.geo_rel_mae.RF_cov` | Geometric-mean relative MAE vs persistence, RF_cov | 1.220 | ok | `outputs/rq1_tables_v2/rq1_aggregate_relative_mae.csv` |
| `rq1.geo_rel_mae.LinearTrend` | Geometric-mean relative MAE vs persistence, LinearTrend | 1.299 | ok | `outputs/rq1_tables_v2/rq1_aggregate_relative_mae.csv` |
| `rq1.geo_rel_mae.InternationalMean` | Geometric-mean relative MAE vs persistence, InternationalMean | 3.281 | ok | `outputs/rq1_tables_v2/rq1_aggregate_relative_mae.csv` |
| `rq1.n_point_rank1.Persistence` | Program-domains where Persistence has the lowest point MAE | 3 | ok | `outputs/rq1_tables_v2/rq1_aggregate_relative_mae.csv` |
| `rq1.n_point_rank1.AR1` | Program-domains where AR1 has the lowest point MAE | 2 | ok | `outputs/rq1_tables_v2/rq1_aggregate_relative_mae.csv` |
| `rq1.n_point_rank1.Drift` | Program-domains where Drift has the lowest point MAE | 1 | ok | `outputs/rq1_tables_v2/rq1_aggregate_relative_mae.csv` |
| `rq1.n_point_rank1.Ridge` | Program-domains where a Ridge model has the lowest point MAE | 2 | ok | `outputs/rq1_tables_v2/rq1_aggregate_relative_mae.csv` |
| `rq1.point_winner_p_rank1.min` | Bootstrap P(rank 1) of the point-MAE winner (min) | 0.21 | ok | `outputs/rq1_tables_v2/rq1_per_program_domain.csv` |
| `rq1.point_winner_p_rank1.max` | Bootstrap P(rank 1) of the point-MAE winner (max) | 0.79 | ok | `outputs/rq1_tables_v2/rq1_per_program_domain.csv` |
| `rq1.persistence_rank_interval_includes_1` | Program-domains whose persistence 95% rank interval includes rank 1 | 8 | ok | `outputs/rq1_tables_v2/rq1_per_program_domain.csv` |
| `rq1.persistence_mae.min` | Persistence MAE (score points), common test set (min) | 9.02 | ok | `outputs/rq1_tables_v2/rq1_per_program_domain.csv` |
| `rq1.persistence_mae.max` | Persistence MAE (score points), common test set (max) | 18.34 | ok | `outputs/rq1_tables_v2/rq1_per_program_domain.csv` |
| `rq1.n_pairings_per_scheme` | Model-vs-persistence pairings per bootstrap scheme | 96 | ok | `outputs/inference_v1/contrasts_all.csv` |
| `rq1.n_sig_better_any_scheme` | Pairings significantly better than persistence (Holm, any of three schemes) | 0 | ok | `outputs/inference_v1/contrasts_all.csv` |
| `rq1.RF_score.n_pisa_sig_worse` | PISA domains where RF_score is significantly worse (mathematics, reading, science) | 3 | ok | `outputs/inference_v1/contrasts_all.csv` |
| `rq1.RF_cov.n_pisa_sig_worse` | PISA domains where RF_cov is significantly worse (mathematics, reading, science) | 3 | ok | `outputs/inference_v1/contrasts_all.csv` |
| `rq1.XGB_score.n_pisa_sig_worse` | PISA domains where XGB_score is significantly worse (mathematics, science) | 2 | ok | `outputs/inference_v1/contrasts_all.csv` |
| `rq1.XGB_cov.n_pisa_sig_worse` | PISA domains where XGB_cov is significantly worse (mathematics, science) | 2 | ok | `outputs/inference_v1/contrasts_all.csv` |
| `rq1.trees.n_point_better` | Program-domains where RF/XGBoost have lower point MAE than persistence | 0 | ok | `outputs/inference_v1/contrasts_all.csv` |
| `rq1.trees.pisa_sig_worse_gap.min` | RF/XGBoost excess MAE over persistence where significant (PISA) (min) | 1.29 | ok | `outputs/inference_v1/contrasts_all.csv` |
| `rq1.trees.pisa_sig_worse_gap.max` | RF/XGBoost excess MAE over persistence where significant (PISA) (max) | 1.92 | ok | `outputs/inference_v1/contrasts_all.csv` |
| `rq1.trees.timss_pirls_gap.min` | RF/XGBoost excess MAE over persistence, TIMSS and PIRLS (min) | 2.50 |  | `outputs/inference_v1/contrasts_all.csv` |
| `rq1.trees.timss_pirls_gap.max` | RF/XGBoost excess MAE over persistence, TIMSS and PIRLS (max) | 5.64 |  | `outputs/inference_v1/contrasts_all.csv` |
| `rq1.trees.timss_pirls_ci_halfwidth.min` | Half-width of 95% CI for those contrasts (min) | 2.84 |  | `outputs/inference_v1/contrasts_all.csv` |
| `rq1.trees.timss_pirls_ci_halfwidth.max` | Half-width of 95% CI for those contrasts (max) | 4.92 |  | `outputs/inference_v1/contrasts_all.csv` |
| `rq1.trees.timss_pirls_n_sig` | Significant RF/XGBoost contrasts in TIMSS and PIRLS | 0 | ok | `outputs/inference_v1/contrasts_all.csv` |
| `rq1.ridge_m0_m1.pisa_sig_worse_gap.min` | Ridge M0/M1 excess MAE over persistence where significant (all PISA) (min) | 2.35 | ok | `outputs/inference_v1/contrasts_all.csv` |
| `rq1.ridge_m0_m1.pisa_sig_worse_gap.max` | Ridge M0/M1 excess MAE over persistence where significant (all PISA) (max) | 2.70 | ok | `outputs/inference_v1/contrasts_all.csv` |
| `rq1.timesfm.timss_g4_math_delta` | TimesFM minus persistence MAE, TIMSS G4 mathematics | 2.28 | ok | `outputs/inference_v1/contrasts_all.csv` |
| `rq1.timesfm.timss_g4_math_p_holm` | Holm p, TimesFM vs persistence, TIMSS G4 mathematics | 0.008 | ok | `outputs/inference_v1/contrasts_all.csv` |
| `rq1.timesfm.n_equivalent` | Program-domains where TimesFM is equivalent to persistence (PISA reading) | 1 | ok | `outputs/inference_v1/contrasts_all.csv` |

## Methods

| ID | Description | Value | Check | Source |
|---|---|---|---|---|
| `rq1.bootstrap_B_tables` | Bootstrap replicates for RQ1 table CIs and rank intervals | 5000 | ok | `outputs/rq1_tables_v2/manifest.json` |
| `scale.median_se.min` | Median standard error of a country mean (min) | 2.51 | ok | `outputs/inference_v1/scale_context.csv` |
| `scale.median_se.max` | Median standard error of a country mean (max) | 3.29 | ok | `outputs/inference_v1/scale_context.csv` |
| `methods.no_history_country_cycles` | Country-cycles without prior same-program score | 580 | ok | `outputs/unified_backtest_v1/no_history.csv` |
| `methods.bootstrap_B_tests` | Bootstrap replicates for paired tests (Holm, TOST) | 5000 | ok | `outputs/inference_v1/manifest.json` |

## Results

| ID | Description | Value | Check | Source |
|---|---|---|---|---|
| `scale.sampling_floor_mae.min` | Sampling-error floor for persistence MAE (min) | 2.95 | ok | `outputs/inference_v1/scale_context.csv` |
| `scale.sampling_floor_mae.max` | Sampling-error floor for persistence MAE (max) | 3.91 | ok | `outputs/inference_v1/scale_context.csv` |
| `scale.persistence_to_floor_ratio.min` | Persistence MAE as a multiple of the sampling floor (min) | 3.06 | ok | `outputs/inference_v1/scale_context.csv` |
| `scale.persistence_to_floor_ratio.max` | Persistence MAE as a multiple of the sampling floor (max) | 5.40 | ok | `outputs/inference_v1/scale_context.csv` |

## RQ2

| ID | Description | Value | Check | Source |
|---|---|---|---|---|
| `rq2.n_kb_improves` | Program-domains meeting the pre-registered KB criterion | 0 | ok | `outputs/inference_v1/rq2_preregistered_decision.csv` |
| `rq2.m1_equiv_m0.country` | Program-domains with M1 equivalent to M0 (TOST ±1, Holm), country | 8 | ok | `outputs/inference_v1/contrasts_all.csv` |
| `rq2.m1_equiv_m0.cycle` | Program-domains with M1 equivalent to M0 (TOST ±1, Holm), cycle | 8 | ok | `outputs/inference_v1/contrasts_all.csv` |
| `rq2.m1_equiv_m0.twoway` | Program-domains with M1 equivalent to M0 (TOST ±1, Holm), twoway | 8 | ok | `outputs/inference_v1/contrasts_all.csv` |
| `rq2.m1_minus_m0.sig_gain_share_of_margin.min` | M1 gain over M0 as share of the 1-point margin where the CI excludes 0 (PISA mathematics, PISA science) (min) | 0.05 | ok | `outputs/inference_v1/contrasts_all.csv` |
| `rq2.m1_minus_m0.sig_gain_share_of_margin.max` | M1 gain over M0 as share of the 1-point margin where the CI excludes 0 (PISA mathematics, PISA science) (max) | 0.08 | ok | `outputs/inference_v1/contrasts_all.csv` |
| `rq2.m1_minus_m0a.pisa_gap.min` | M1 excess MAE over M0A, PISA (min) | 1.59 | ok | `outputs/inference_v1/contrasts_all.csv` |
| `rq2.m1_minus_m0a.pisa_gap.max` | M1 excess MAE over M0A, PISA (max) | 1.70 | ok | `outputs/inference_v1/contrasts_all.csv` |
| `rq2.m1_vs_m0a.pisa_country_p_holm_max` | Max Holm p, M1 vs M0A, PISA, country scheme | 0.002 | ok | `outputs/inference_v1/contrasts_all.csv` |
| `rq2.m1_vs_m0a.pisa_cycle_twoway_p_holm_min` | Min Holm p, M1 vs M0A, PISA, cycle and two-way schemes (K5: not significant) | 0.499 | ok | `outputs/inference_v1/contrasts_all.csv` |
| `rq2.m1_vs_m0a.pirls_ci90_hi` | Upper 90% CI bound, M0A minus M1, PIRLS (excludes 0) | -0.06 | ok | `outputs/inference_v1/contrasts_all.csv` |
| `rq2.m1_vs_m0a.pirls_p_holm` | Holm p, M1 vs M0A, PIRLS (K5: not significant) | 0.315 | ok | `outputs/inference_v1/contrasts_all.csv` |
| `rq2.perm_share_all.min` | Share of 200 KB-weight permutations as good as or better than M1, all 8 (min) | 0.23 | ok | `outputs/rq2_v3/contrasts_primary_excl2025.csv` |
| `rq2.perm_share_all.max` | Share of 200 KB-weight permutations as good as or better than M1, all 8 (max) | 0.99 | ok | `outputs/rq2_v3/contrasts_primary_excl2025.csv` |
| `rq2.perm_share_pisa.min` | Same share, PISA only (min) | 0.42 | ok | `outputs/rq2_v3/contrasts_primary_excl2025.csv` |
| `rq2.perm_share_pisa.max` | Same share, PISA only (max) | 0.99 | ok | `outputs/rq2_v3/contrasts_primary_excl2025.csv` |
| `rq2.n_beats_perm_p05` | Program-domains where M1 beats the permutation 5th percentile | 0 | ok | `outputs/rq2_v3/contrasts_primary_excl2025.csv` |
| `rq2.m0_minus_m1_max_gain` | Largest M1 gain over M0 (score points) | 0.08 | ok | `outputs/rq2_v3/contrasts_primary_excl2025.csv` |
| `rq2.n_placebo_ge_m1` | Program-domains where the best shrinkage placebo matches or beats M1 | 8 | ok | `outputs/rq2_v3/contrasts_primary_excl2025.csv` |
| `rq2.topk.n_testable` | Program-domains where KB top-k vs random top-k is testable (K3) | 4 | ok | `outputs/rq2_v3/contrasts_primary_excl2025.csv` |
| `rq2.topk.n_better_than_random` | KB top-k configurations below the random top-k 5th percentile | 0 | ok | `outputs/rq2_v3/contrasts_primary_excl2025.csv` |

## KB

| ID | Description | Value | Check | Source |
|---|---|---|---|---|
| `kb.intra.n_records` | Gold records re-coded (intra-rater) | 173 | ok | `outputs/kb_integrity_v1/kappa_summary.json` |
| `kb.intra.n_studies` | Studies in intra-rater re-coding | 30 | ok | `outputs/kb_integrity_v1/kappa_summary.json` |
| `kb.intra.kappa` | Intra-rater kappa vs pre-adjudication labels | 0.65 | ok | `outputs/kb_integrity_v1/kappa_summary.json` |
| `kb.intra.kappa_ci_lo` | Intra-rater kappa, 95% study-bootstrap CI lower | 0.45 | ok | `outputs/kb_integrity_v1/kappa_summary.json` |
| `kb.intra.kappa_ci_hi` | Intra-rater kappa, 95% study-bootstrap CI upper | 0.78 | ok | `outputs/kb_integrity_v1/kappa_summary.json` |
| `kb.validation.n_records` | Study-construct pairs in blind expert validation | 51 | ok | `outputs/kb_integrity_v1/evidence_validation_agreement_v1.json` |
| `kb.validation.n_studies` | Studies in blind expert validation | 35 | ok | `outputs/kb_integrity_v1/evidence_validation_agreement_v1.json` |
| `kb.validation.kappa.canonical_construct` | LLM-vs-expert kappa, canonical_construct | 0.70 | ok | `outputs/kb_integrity_v1/evidence_validation_agreement_v1.json` |
| `kb.validation.kappa.effect_type` | LLM-vs-expert kappa, effect_type | 0.21 | ok | `outputs/kb_integrity_v1/evidence_validation_agreement_v1.json` |
| `kb.validation.kappa.analysis_level` | LLM-vs-expert kappa, analysis_level | 0.38 | ok | `outputs/kb_integrity_v1/evidence_validation_agreement_v1.json` |
| `kb.validation.kappa.direction` | LLM-vs-expert kappa, direction | 0.01 | ok | `outputs/kb_integrity_v1/evidence_validation_agreement_v1.json` |
| `kb.validation.kappa_construct_ci_lo` | Construct kappa, 95% CI lower | 0.57 | ok | `outputs/kb_integrity_v1/evidence_validation_agreement_v1.json` |
| `kb.validation.kappa_construct_ci_hi` | Construct kappa, 95% CI upper | 0.83 | ok | `outputs/kb_integrity_v1/evidence_validation_agreement_v1.json` |
| `kb.validation.n_llm_signed` | Pairs with an LLM-assigned sign | 10 | ok | `outputs/kb_integrity_v1/evidence_matrix_validation_sample_v1.csv` |
| `kb.validation.n_sign_confirmed` | LLM-assigned signs confirmed by the expert | 1 | ok | `outputs/kb_integrity_v1/evidence_matrix_validation_sample_v1.csv` |
| `kb.cls_mgmt.n_sampled` | Sampled pairs labelled classroom-management self-efficacy | 6 | ok | `outputs/kb_integrity_v1/evidence_validation_per_construct_v1.csv` |
| `kb.cls_mgmt.n_correct` | Of those, labels confirmed correct (K1: 6/6 mislabelled) | 0 | ok | `outputs/kb_integrity_v1/evidence_validation_per_construct_v1.csv` |
| `kb.cls_mgmt.weight` | KB weight, classroom-management self-efficacy | 0.854 | ok | `outputs/kb_integrity_v1/predictor_weights_v3_sample_corrected.csv` |
| `kb.cls_mgmt.weight_corrected` | Sample-corrected weight | 0.677 | ok | `outputs/kb_integrity_v1/predictor_weights_v3_sample_corrected.csv` |
| `kb.evidence.n_rows` | Evidence rows in the knowledge base | 530 | ok | `outputs/stage5/evidence_matrix.csv` |
| `kb.evidence.n_studies` | Studies in the knowledge base | 80 | ok | `outputs/stage5/evidence_matrix.csv` |
| `kb.evidence.n_student_level` | Student-level evidence rows | 486 | ok | `outputs/stage5/evidence_matrix.csv` |
| `kb.evidence.n_country_level` | Country-level evidence rows | 4 | ok | `outputs/stage5/evidence_matrix.csv` |
| `kb.gold.n_rows` | Gold corpus rows | 246 | ok | `outputs/finetuning/data/gold_standard_full_corpus_v1.4.1.csv` |
| `kb.gold.n_studies` | Gold corpus studies | 42 | ok | `outputs/finetuning/data/gold_standard_full_corpus_v1.4.1.csv` |
