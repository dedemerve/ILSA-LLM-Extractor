# ILSA Literature Dataset — Phase 1 Audit Report

**Date:** 2026-10-14  
**Analyst:** automated audit (Phase 1, Research Protocol v1)  
**Authoritative source:** `outputs/ILSA_Meta_Analysis_Dataset_CLEAN.xlsx`  
**Supplementary source:** `outputs/ILSA_Analytical_Meta_Analysis_Master.xlsx`, `outputs/ILSA_Structured_Meta_Analysis.xlsx`

---

## 1. Repository Inventory

| Item | Location | Status |
|------|----------|--------|
| Primary authoritative file | `outputs/ILSA_Meta_Analysis_Dataset_CLEAN.xlsx` | ✅ Present |
| Analytical master (ML fields) | `outputs/ILSA_Analytical_Meta_Analysis_Master.xlsx` | ✅ Present |
| Structured meta-analysis | `outputs/ILSA_Structured_Meta_Analysis.xlsx` | ✅ Present |
| HF-published parquet (hf_final) | `outputs/hf_final/data/` | ⚠️ Aggregated summaries only — not article-level |
| Fine-tuning gold standard | `outputs/finetuning/gold_standard.csv` | ✅ Present (do not overwrite) |
| Fine-tuning v2 | `outputs/finetuning/v2/` | ✅ Present (do not overwrite) |
| Article-level JSONs | `outputs/ilsa_survey_articles/` | Present (not audited this phase) |

**Finding:** The Hugging Face–published parquet files (`hf_final/`) contain only aggregated summaries (174 knowledge-synthesis rows, 36 meta-analysis rows, etc.) — they do **not** contain the 1,266 article records. For literature analysis, `ILSA_Meta_Analysis_Dataset_CLEAN.xlsx` is the authoritative source.

---

## 2. Record Counts

| Table | Rows | Unique DOIs | Unique file_names |
|-------|------|-------------|-------------------|
| 1_Articles_Master | 1,266 | 995 | 1,266 |
| 2_Main_Findings | 2,128 | 995 | — |
| 3_Confounders | 8,336 | 995 | — |

**Interpretation:**
- The published count of 1,266 unique study records is confirmed at the file-name level.
- DOI uniqueness is lower (995) because **271 articles (21.4%) have no DOI**. These are likely conference papers, grey literature, or technical reports.
- No duplicate file_names exist.
- No orphaned findings or confounder records: all finding and confounder DOIs exist in the articles table.

---

## 3. Key Field Missingness

| Field | Missing/Unknown | % |
|-------|-----------------|---|
| doi | 271 | 21.4% |
| title | 7 | 0.6% |
| year | 60 | 4.7% |
| research_design_type | 0 | 0.0% |
| outcome_summary | 0 | 0.0% |
| student_weights_used | 25 | 2.0% |
| plausible_values_handling | 0 | 0.0% |
| missing_data_handling | 0 | 0.0% |

**Finding:** Core bibliographic and methodological fields are well populated. The 271 missing DOIs are an expected characteristic of grey literature, not a data quality failure.

---

## 4. Publication Year Coverage

- Range: 1994–2026 (with 60 records missing year)
- Accelerating trend: 103 studies in 2021, 201 in 2025, 112 in 2026 (partial year)
- The corpus is heavily weighted toward 2020–2026 (801 of ~1,206 dated records = 66%)

---

## 5. Research Design Distribution

| Design Type | Count | % of total |
|-------------|-------|------------|
| exploratory | 1,050 | 82.9% |
| predictive | 166 | 13.1% |
| causal_observational | 46 | 3.6% |
| causal_experimental | 4 | 0.3% |

**Critical finding:** Only **166 studies (13.1%) are tagged as predictive**. This is the primary pool for RQ2–RQ4. The remaining 1,100 studies use descriptive, exploratory, or causal designs — they use ML for variable selection or explanation, not prediction.

This count aligns with the study-filter breakdown in the Analytical Master:
- Empirical Study – Machine Learning: **166 studies**
- Empirical Study – Traditional Statistics: 555 studies
- Technical/Assessment Framework: 545 studies (technical reports)
- Descriptive National Report: further non-ML

---

## 6. ML Study Subset — Methods

*Based on 166 studies tagged as Empirical Study – Machine Learning.*

### ML Family
| Family | Count |
|--------|-------|
| Tree-Based / Ensemble | 85 |
| Other ML / Not Classified | 38 |
| Generalized Linear Models (GLM) | 25 |
| Deep Learning | 18 |

### ML Primary Algorithm (top 15)
| Algorithm | Count | % of 166 ML studies |
|-----------|-------|----------------------|
| Random Forest | 44 | 26.5% |
| XGBoost | 18 | 10.8% |
| Neural Network | 16–17 | 10.2% |
| SVM | 13–14 | 8.4% |
| LASSO | 10 | 6.0% |
| Logistic Regression | 6 | 3.6% |
| Decision Tree | 5 | 3.0% |
| Gradient Boosting | 5 | 3.0% |
| LightGBM | 4 | 2.4% |
| Elastic Net | 4 | 2.4% |
| CatBoost | 3 | 1.8% |
| Not reported | 1,099 (full corpus) | — |

**Note:** 1,099/1,266 articles (86.8%) have no ML primary technique — these are non-ML studies. Percentages above are computed over the 166 ML-subset studies only.

---

## 7. ILSA Program Coverage (in Findings table)

| Program | Findings records | Note |
|---------|-----------------|------|
| PISA | 901 | Dominant program |
| TIMSS | 263 | |
| TALIS | 163 | Teacher surveys — not student achievement |
| PIRLS | 154 | |
| PIAAC | 134 | Adult skills — different population |
| ICCS | 69 | Civic education |
| ICILS | 50 | Digital literacy |

**Finding:** All seven intended programs are present. Coverage is heavily PISA-weighted. TALIS and PIAAC measure different constructs (teachers and adult skills, respectively) — they will require separate treatment in the prediction taxonomy.

---

## 8. Target Domain Distribution (ML Findings, n=309)

| Domain | Count |
|--------|-------|
| Other / Unspecified | 110 |
| Composite / Multi-Domain | 82 |
| Mathematics | 44 |
| Reading | 20 |
| Science | 19 |
| Non-Cognitive / Process Output | 19 |
| Problem Solving | 13 |
| Digital/Computer Literacy | 1 |
| Civic Education | 1 |

**Finding:** ~36% of ML findings target "Other/Unspecified" or non-cognitive outcomes. Genuine student cognitive achievement prediction (Math, Reading, Science) accounts for ~83 findings (27% of ML findings). This is a smaller base than the raw 309 figure suggests.

---

## 9. Survey Methodology Fields

| Field | Value | Count |
|-------|-------|-------|
| plausible_values_handling | not_applicable | 971 |
| | rubin_rules (correct) | 149 |
| | average_pv (potentially incorrect) | 59 |
| | single_pv (potentially incorrect) | 45 |
| | not_reported | 3 |
| student_weights_used | Yes (1.0) | 1,060 |
| | No (0.0) | 181 |
| | Unknown | 25 |

**Finding:** Of the ~295 studies that used plausible values (non-not_applicable):
- 149 (50.5%) correctly applied Rubin's rules
- 104 (35.3%) used potentially incorrect methods (single PV or simple average)
- 3 did not report their approach

This is a documented methodological quality issue in the literature — noted for RQ3.

---

## 10. Performance Metrics Availability

| Subset | With Metrics | Total | % |
|--------|-------------|-------|---|
| All findings | 1,763 | 2,128 | 82.8% |
| ML findings only | 309 | 309 | 100.0% |

**Finding:** All 309 ML findings have some performance metric text. However, these are free-text strings (e.g., "XGBoost — AUC: 0.79"), not structured numeric fields. Extracting comparable numeric values requires normalization in Phase 4.

---

## 11. Temporal Forecasting Gap (Preliminary)

The `research_design_type` field does not distinguish between:
- Within-cycle random split (predicting unseen students from the same assessment)
- Cross-cycle forecasting (predicting a future assessment outcome)

No dedicated "forecast_horizon" or "temporal_validation" field exists in the current schema. This must be coded manually or semi-automatically in Phase 3.

**Preliminary search of dataset_used strings for multi-cycle patterns:**
The ML findings `Dataset_Used` field contains entries like "PISA 2015–2018", "PISA 2006–2022", suggesting some studies use multiple cycles. However, multi-cycle data use ≠ temporal forecasting. Phase 3 must verify this distinction.

---

## 12. Data Quality Issues Identified

| Issue | Severity | Recommendation |
|-------|----------|----------------|
| 271 articles without DOI | Medium | Use file_name as primary key; DOI as supplement |
| 104 studies with likely-incorrect PV handling | Medium | Flag in methodology quality field |
| `dataset_used` is free text — not normalized | High | Requires ILSA-program normalization in Phase 2 |
| No temporal_validation flag in schema | High | Must be added in Phase 3 taxonomy |
| performance_metrics is free text | Medium | Requires regex extraction in Phase 4 |
| ~110 ML findings with "Other/Unspecified" domain | Medium | Requires domain clarification in Phase 3 |
| HF-published data does not include article-level records | Informational | Noted — local Excel file is authoritative |

---

## 13. Assessment of Research Protocol Conditions

| Condition | Status |
|-----------|--------|
| 1,266 unique study records confirmed | ✅ Confirmed (by file_name) |
| All 7 ILSA programs present | ✅ Confirmed |
| ML study subset identified | ✅ 166 studies |
| Gold standard / fine-tuning data untouched | ✅ Not modified |
| Original data files not modified | ✅ Read-only |
| HF dataset ≠ article-level corpus | ✅ Documented |

---

## 14. Action Plan (Priority Order)

### Immediate (Phase 2)
1. **Normalize `dataset_used`** into structured fields: ILSA program, cycle year(s), country scope — required for all subsequent analyses
2. **Build prediction taxonomy** — distinguish cross-sectional within-cycle splits from cross-cycle temporal forecasting; this is the most critical classification missing from the current schema
3. **Normalize `ml_all_techniques`** — map free-text algorithm names to canonical taxonomy

### Requiring manual verification
4. **Sample 30–50 ML studies** to verify `study_filter_type` classification and temporal validation coding
5. **Verify plausible-values handling** for a sample of studies tagged `average_pv` or `single_pv` against source articles

### Requiring additional data
6. **Assemble ILSA outcome time series** — country-level mean scores by cycle for PISA (2000–2022) and TIMSS (1995–2023); these are NOT in the literature corpus
7. **Check TimesFM repository** for current version, checkpoint, and API (Phase 8)

### Postpone until feasibility established
8. **TimesFM experiment design** — cannot finalize until temporal observation count per series is confirmed
9. **Cross-program generalizability analysis** — dependent on prediction taxonomy completion

---

## 15. Decision Gate: Proceed to Phase 2?

**Recommendation: Yes, proceed to Phase 2.**

The dataset is structurally sound and confirms the reported 1,266 record count. The ML study subset (n=166) is large enough for a meaningful frequency analysis. The primary gap — absence of a temporal forecasting flag — is addressable through taxonomy construction in Phase 2. No show-stopping data quality issues were found.

The TimesFM experiment cannot be evaluated yet; it should remain in Phase 8 pending Phase 6 (ILSA time-series feasibility).
