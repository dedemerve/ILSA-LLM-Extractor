# Phase 0 — CLEAN Excel Sheets 5–7 Schema Reference

**Parent document:** [phase0_methodological_specification.md](./phase0_methodological_specification.md)  
**Workbook:** `outputs/ILSA_Meta_Analysis_Dataset_CLEAN.xlsx`  
**Policy:** Append-only operational layers; Sheets 0–4 semantics unchanged.

---

## Sheet 5 — `5_Verification_Claims`

### Column headers (horizontal)

`claim_id` | `file_name` | `doi` | `finding_row_hash` | `claim_text` | `normalized_isa` | `normalized_cycle` | `normalized_grade` | `normalized_domain` | `countries_iso3` | `analysis_level` | `literature_outcome_label` | `literature_predictor_labels` | `official_outcome_vars` | `official_predictor_vars` | `weight_var` | `weight_profile_key` | `pv_rule` | `pv_draw_used_by_paper` | `catalog_file_id` | `catalog_match_confidence` | `verification_eligible` | `verification_priority` | `bridge_provenance` | `created_at`

### Data validation / dropdown bindings

| Column | `_FilterLists` key | Notes |
|--------|-------------------|--------|
| `normalized_isa` | `normalized_isa` | Locked enum |
| `normalized_domain` | `target_domain` | Reuses Sheet 2 vocabulary |
| `analysis_level` | `analysis_level` | New Phase 0 enum |
| `pv_rule` | `pv_rule` | New Phase 0 enum |
| `catalog_match_confidence` | `catalog_match_confidence` | High / Medium / Low |
| `verification_eligible` | `verification_eligible` | TRUE / FALSE |
| `verification_priority` | `verification_priority` | P1–P4 |
| `paper_reported_direction` | — | Not on Sheet 5 (Sheet 6) |

Free-text columns (no dropdown): `claim_id`, `file_name`, `doi`, `finding_row_hash`, `claim_text`, `countries_iso3`, `literature_*`, `official_*`, `weight_var`, `weight_profile_key`, `catalog_file_id`, `bridge_provenance`, `created_at`.

### Pilot sample row

| Field | Value |
|-------|-------|
| `claim_id` | `CLM_PISA_2012_000001` |
| `file_name` | `pilot_pisa2012_tur_escs_math.pdf` |
| `doi` | `10.0000/pilot.pisa2012.escs` |
| `finding_row_hash` | `sha256:pilot_finding_row_001` |
| `claim_text` | ESCS is positively associated with mathematics achievement among 15-year-olds in Turkey (PISA 2012). |
| `normalized_isa` | `PISA` |
| `normalized_cycle` | `2012` |
| `normalized_grade` | `15` |
| `normalized_domain` | `Mathematics` |
| `countries_iso3` | `TUR` |
| `analysis_level` | `student` |
| `literature_outcome_label` | Mathematics achievement (plausible values) |
| `literature_predictor_labels` | ESCS; socio-economic status |
| `official_outcome_vars` | `PISA_PV_MATH_C6` |
| `official_predictor_vars` | `ESCS` |
| `weight_var` | `W_FSTUWT` |
| `weight_profile_key` | `PISA_STUDENT_WGT_C6` |
| `pv_rule` | `rubin_rules` |
| `pv_draw_used_by_paper` | *(blank)* |
| `catalog_file_id` | `PISA_pending_phase1_scan` |
| `catalog_match_confidence` | `Medium` |
| `verification_eligible` | `TRUE` |
| `verification_priority` | `P1_pilot` |
| `bridge_provenance` | `ALIAS_ESCS;PISA_PV_MATH_C6;RULE_PV_01` |
| `created_at` | `2026-05-31T12:00:00Z` |

---

## Sheet 6 — `6_Verification_Results`

### Column headers (horizontal)

`result_id` | `claim_id` | `check_type` | `check_subtype` | `paper_reported_value` | `paper_reported_direction` | `paper_significance` | `replicated_value` | `replicated_se` | `replicated_p_value` | `replicated_direction` | `match_delta` | `match_status` | `alignment_narrative` | `n_analytic` | `pv_rule_applied` | `weight_var_applied` | `country_filter_applied` | `evidence_path` | `evidence_artifact_list` | `engine_version` | `run_timestamp` | `run_duration_sec` | `failure_code`

### Data validation / dropdown bindings

| Column | `_FilterLists` key |
|--------|-------------------|
| `check_type` | `check_type` |
| `paper_reported_direction` | `effect_direction` |
| `paper_significance` | `paper_significance` |
| `replicated_direction` | `effect_direction` |
| `match_status` | `match_status` |
| `pv_rule_applied` | `pv_rule` |

### Pilot sample row (linked via `claim_id`)

| Field | Value |
|-------|-------|
| `result_id` | `RES_CLM_PISA_2012_000001_correlation_direction_001` |
| `claim_id` | `CLM_PISA_2012_000001` |
| `check_type` | `correlation_direction` |
| `check_subtype` | `weighted_pearson_rubin_pooled` |
| `paper_reported_value` | Positive association (direction only reported) |
| `paper_reported_direction` | `Positive` |
| `paper_significance` | `Significant` |
| `replicated_value` | *(blank — pending Phase 1 engine run)* |
| `replicated_se` | *(blank)* |
| `replicated_p_value` | *(blank)* |
| `replicated_direction` | *(blank)* |
| `match_delta` | *(blank)* |
| `match_status` | `Not_Runnable` |
| `alignment_narrative` | Phase 0 schema row; deterministic replication pending microdata catalog build. |
| `n_analytic` | *(blank)* |
| `pv_rule_applied` | `rubin_rules` |
| `weight_var_applied` | `W_FSTUWT` |
| `country_filter_applied` | `TUR` |
| `evidence_path` | `outputs/verification/runs/CLM_PISA_2012_000001/` |
| `evidence_artifact_list` | `manifest.json;claim_spec.json` |
| `engine_version` | `0.0.0-phase0` |
| `run_timestamp` | `2026-05-31T12:00:00Z` |
| `run_duration_sec` | `0` |
| `failure_code` | `CATALOG_NOT_BUILT` |

---

## Sheet 7 — `7_Prediction_Inventory`

### Column headers (horizontal)

`inventory_id` | `target_domain` | `canonical_variable` | `registry_outcome_key` | `isa_coverage` | `cycle_coverage` | `predictor_category_mix` | `top_predictors_canonical` | `n_supporting_studies` | `n_verified_claims` | `dominant_algorithms_in_literature` | `dominant_canonical_method` | `effect_direction_consensus` | `microdata_ready` | `feature_store_status` | `multi_output_pipeline_priority` | `recommended_model_class` | `measurement_invariance_note` | `last_inventory_update`

### Data validation / dropdown bindings

| Column | `_FilterLists` key |
|--------|-------------------|
| `target_domain` | `target_domain` |
| `canonical_variable` | `canonical_variable` |
| `effect_direction_consensus` | `effect_direction` |
| `microdata_ready` | `verification_eligible` |
| `feature_store_status` | `feature_store_status` |
| `multi_output_pipeline_priority` | `pipeline_priority` |

### Pilot sample row (linked via `canonical_variable`)

| Field | Value |
|-------|-------|
| `inventory_id` | `INV_Math_Achievement_Mathematics` |
| `target_domain` | `Mathematics` |
| `canonical_variable` | `Math_Achievement` |
| `registry_outcome_key` | `PISA_MATH_ACHIEVEMENT_PV` |
| `isa_coverage` | `PISA;TIMSS` |
| `cycle_coverage` | `2012;2015;2018;2022` |
| `predictor_category_mix` | `Student: SES;Student: Demographic` |
| `top_predictors_canonical` | `SES;Gender_Demographics;Prior_Achievement` |
| `n_supporting_studies` | `245` |
| `n_verified_claims` | `0` |
| `dominant_algorithms_in_literature` | `Traditional_Stats;Ensemble_Learning;Deep_Learning` |
| `dominant_canonical_method` | `Traditional_Stats` |
| `effect_direction_consensus` | `Positive` |
| `microdata_ready` | `FALSE` |
| `feature_store_status` | `Planned` |
| `multi_output_pipeline_priority` | `Critical` |
| `recommended_model_class` | `weighted_mixed_effects_rubin_pv` |
| `measurement_invariance_note` | PISA PV scales are cycle-specific; do not pool 2012 and 2022 without OECD linking documentation. |
| `last_inventory_update` | `2026-05-31T12:00:00Z` |

---

## Dynamic linking diagram

```
1_Articles_Master.file_name
        │
        ▼
5_Verification_Claims.claim_id  ◄── finding_row_hash ── 2_Main_Findings
        │
        ├──► 6_Verification_Results.claim_id
        │
Canonical_View.canonical_variable
        │
        ▼
7_Prediction_Inventory.canonical_variable
        │
        └──► Layer 2 registry_outcome_key (PISA_MATH_ACHIEVEMENT_PV)
```

**Initializer:** `python scripts/init_phase0_verification_sheets.py`
