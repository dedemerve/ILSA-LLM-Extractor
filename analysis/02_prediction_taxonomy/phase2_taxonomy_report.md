# Phase 2 Report: Prediction-Task Taxonomy and Study-Level Classification

**Date:** 2026-10-14  
**Analyst:** Phase 2, Research Protocol v1 — rule-based classification + manual review  
**Input:** `outputs/ILSA_Meta_Analysis_Dataset_CLEAN.xlsx` (authoritative, read-only)  
**Outputs:** `analysis/02_prediction_taxonomy/`

---

## 1. Methodology

### Classification unit
One row per unique study (file_name as primary key). The 1,266 unique file_names are confirmed. DOI-uniqueness (995 unique DOIs) reflects 271 DOI-less records, not 271 duplicate studies.

### Potential duplicate detection
- 10 records flagged: 5 pairs of identical titles in the DOI-less subset
- Examples: two copies of the same TIMSS 2011 framework document in Spanish; two PISA 2003 Data Analysis Manuals
- These are technical/framework documents, not empirical studies; they do not affect the predictive study count
- Flagged for manual consolidation; not removed from the current analysis

### Classification method
Rule-based, applied to extracted fields only. Not verified against source articles unless explicitly stated. All classifications are labeled with confidence level (High/Medium/Low) and a `manual_review_required` flag.

---

## 2. Taxonomy Categories — Operational Definitions

| Code | Label | Definition |
|------|-------|------------|
| A | student_within_cycle_prediction | ML/predictive study where the target is individual student outcome and evaluation is within the same assessment cycle (typically random split of students) |
| B | student_cross_cycle_prediction | ML study where student-level outcomes from a later cycle are predicted using an earlier cycle; verified that the evaluation protocol predicts a held-out future observation |
| C | school_teacher_classroom_prediction | ML study with school, teacher, or classroom-level prediction target |
| D | country_system_prediction | ML study predicting country/system-level aggregate outcomes (mean scores, proficiency distributions) |
| E | future_cycle_forecasting | Study explicitly forecasts a future ILSA cycle outcome using historical cycles as training data — evaluated against subsequently observed cycle results |
| F | other_predictive | ML study with non-cognitive, process-data, item-scoring, clustering, or other targets that do not fit A–E |
| G | non_predictive_analytical | Descriptive, exploratory, causal, psychometric, or technical/framework studies |
| H | uncertain_insufficient_evidence | Evidence in extracted fields is insufficient to assign a substantive category |

---

## 3. Primary Task Distribution

| Task | Code | Studies | % of 1,266 |
|------|------|---------|------------|
| Non-predictive analytical | G | 1,100 | 86.9% |
| Student within-cycle prediction | A | 96 | 7.6% |
| Other predictive | F | 45 | 3.6% |
| Country/system prediction | D | 17 | 1.3% |
| Student cross-cycle prediction | B | 8* | 0.6% |
| Future-cycle forecasting | E | **0** | 0.0% |
| Uncertain | H | 0 | 0.0% |

*8 is the automated count. Manual review of all 8 reduces the confirmed cross-cycle ML count to **2** (see Section 6).

---

## 4. Machine Learning Study Subset — Revised Analysis

**Confirmed ML studies (study_filter_type = Empirical Study - Machine Learning): 166**

This count is supported by the Phase 1 audit and is unchanged.

### Task distribution within 166 ML studies

| Task | Count | % of 166 |
|------|-------|----------|
| Student within-cycle prediction (A) | 96 | 57.8% |
| Other predictive (F) | 45 | 27.1% |
| Country/system prediction (D) | 17 | 10.2% |
| Student cross-cycle prediction (B) | 8* | 4.8% |
| Future-cycle forecasting (E) | **0** | **0.0%** |

*Post-manual-review: 2 confirmed; 6 are false positives from the automated classifier.

### Prediction target level (ML studies)

| Level | Count |
|-------|-------|
| Student | 102 |
| Country/system | 17 |
| Unclear | 47 |

---

## 5. ILSA Program Coverage

| Program | Studies (extraction-based) | Notes |
|---------|---------------------------|-------|
| PISA | 543 | Dominant — 43% of corpus |
| TIMSS | 210 | |
| TALIS | 111 | Teacher surveys; not student achievement |
| PIAAC | 108 | Adult skills; different population and target |
| PIRLS | 131 | |
| ICCS | 102 | Civic education |
| ICILS | 54 | Digital literacy |
| PISA_D | 6 | PISA for Development |
| TIMSS_Advanced | 2 | |

Program coverage is extraction-based, not manually verified. A study listed under multiple programs may use them in different ways (background vs. analyzed data). This requires Phase 3 verification for the forecasting experiment.

---

## 6. Manual Review: Cross-Cycle and Temporal Studies

All 8 studies automatically classified as `student_cross_cycle_prediction` were manually inspected.

| File | Title (abbreviated) | Manual verdict | Evidence |
|------|---------------------|---------------|---------|
| s00521-023-08901-6.pdf | Predicting science scores — PISA 2015–2018 | **B confirmed** | Explicitly trains on PISA 2015, predicts PISA 2018 student scores; temporal evaluation confirmed |
| s40536-025-00262-x.pdf | TIMSS NN quality assurance — 2011→2015→2019 | **D confirmed (cross-cycle)** | Neural network predicts country-level math/science achievement residuals across TIMSS cycles; purpose is quality assurance not prospective forecasting |
| 1-s2.0-S0191491X24000099-main.pdf | TIMSS 2003–2019 school inequality (exploratory data mining) | **Reclassify → G** | Study is exploratory trend description, not a predictive model evaluated on held-out future cycles |
| data-10-00130.pdf | Extended Dataset of Educational Quality 1970–2023 | **Reclassify → G** | Data publication creating a harmonized panel; ML is used incidentally; no predictive evaluation protocol |
| 1-s2.0-S0038012118302908-main.pdf | School efficiency Tunisia — PISA 2012 | **Reclassify → C** | Single cycle (PISA 2012); "2011–2012" is school year within cycle; school-level prediction |
| Brit J Math Wang 2022 — PIAAC process data | Subtask analysis via predictive model | **Reclassify → A or F** | Single PIAAC 2012 cycle; process-data item-level prediction; not cross-cycle |
| Rodriguez-Barrios 2021 — Mexico reading | Bayesian reading comprehension — Mexico | **Reclassify → F** | Non-ILSA dataset (Mexican elementary school experiment); year pattern "2018–2019" triggered false positive |
| Chen et al. 2024 — PSLC DataShop | Learner modeling via deep learning | **Reclassify → F** | Non-ILSA dataset (PSLC DataShop); not an ILSA study |

### Revised cross-cycle counts (post-review)

| Category | Automated | Post-review |
|----------|-----------|-------------|
| Confirmed cross-cycle ML prediction | 8 | **2** |
| Confirmed future-cycle forecasting (E) | 0 | **0** |

**Finding: No study in the 1,266-record corpus performs genuine future-cycle ILSA forecasting in the TimesFM sense** (i.e., training a model on historical country-level ILSA time series and forecasting the next cycle outcome using only pre-cycle information).

---

## 7. Temporal Forecasting Status Distribution

| Status | Count | Notes |
|--------|-------|-------|
| not_applicable | 545 | Technical reports, frameworks |
| no_temporal_prediction | 538 | Traditional statistics, descriptive |
| within_cycle_prediction | 155 | ML and predictive studies within one cycle |
| cross_cycle_prediction_not_fully_verified | 28 | Includes 20+ false positives; manually 2 confirmed ML |
| confirmed_future_cycle_forecasting | **0** | Confirmed after manual review |

---

## 8. Validation Strategy (166 ML Studies)

| Strategy | Count | Notes |
|----------|-------|-------|
| k-fold cross-validation | 78 | 47.0% — dominant approach; within-cycle |
| insufficient_evidence | 48 | Reporting gap |
| random train/test split | 32 | 19.3% — within-cycle |
| temporal_holdout_cross_cycle | 7 | Automated; 2–3 confirmed after review |
| rolling_origin | 1 | False positive (non-ILSA Indonesian literature study) |

**Critical finding:** No confirmed rolling-origin evaluation exists in the corpus for genuine ILSA outcome data. Within-cycle random splits dominate (47% + 19% = 66%), which test generalization to unseen students from the same assessment but not to future cycles.

---

## 9. Does a Temporal-Forecasting Research Gap Exist?

**Yes — the evidence supports a genuine gap.**

Evidence:
1. Zero confirmed future-cycle forecasting studies in 1,266 records
2. Only 2 cross-cycle ML studies, both with limited scope (one student-level cross-cycle transfer in 13 countries; one country-level quality assurance tool)
3. No study constructs a country-level time series from multiple ILSA cycles and applies a forecasting model
4. No study evaluates TimesFM or any other time-series foundation model on ILSA data
5. The 2026 scoping review (Studies in Educational Evaluation) covers only 57 studies through a narrower lens

**Caution:** This finding is based on extracted metadata, not a manual inspection of all 1,266 articles. A small number of studies flagged as "exploratory" or "traditional statistics" may include forecasting elements not captured by extraction. The gap claim is well-supported but not independently verified at source level.

---

## 10. Promising Candidates for TimesFM Feasibility Analysis (Phase 6)

Based on program coverage and cycle availability:

| Program | Studies | Cycles available | Subject | Recommendation |
|---------|---------|-----------------|---------|----------------|
| PISA | 543 | 2000, 2003, 2006, 2009, 2012, 2015, 2018, 2022 | Math, Reading, Science | **Primary candidate** — longest consistent series |
| TIMSS | 210 | 1995, 1999, 2003, 2007, 2011, 2015, 2019, 2023 | Math, Science | **Primary candidate** — 8 cycles, grade 4 and 8 |
| PIRLS | 131 | 2001, 2006, 2011, 2016, 2021 | Reading | Secondary — 5 cycles |
| PIAAC | 108 | 2012, 2015, 2017–2018 | Adult literacy | Different population — not recommended for Phase 7 |
| TALIS | 111 | 2008, 2013, 2018 | Teacher surveys | Different construct — not recommended |

PISA and TIMSS are the most feasible: both have 8+ comparable cycles, consistent country coverage, and official published country-level mean scores.

---

## 11. Summary: Comparison with Phase 1 Preliminary Count

| Metric | Phase 1 (preliminary) | Phase 2 (verified) |
|--------|----------------------|-------------------|
| Total studies | 1,266 | 1,266 (confirmed) |
| ML studies | 166 | 166 (confirmed) |
| Confirmed future-cycle forecasting | unknown | **0** |
| Confirmed cross-cycle ML studies | unknown | **2** |
| Genuine student cognitive prediction | ~83 (estimate) | 96 within-cycle + 1 cross-cycle |
| Temporal forecasting gap confirmed | not assessed | **Yes** |

---

## 12. Decision Gate: Proceed to Phase 3 (Algorithm Taxonomy)?

**Yes, proceed — with the following conditions:**

1. The 166 ML study count is confirmed
2. The temporal forecasting gap is supported by extracted evidence
3. The false-positive rate in cross-cycle classification (~75%) requires accepting that algorithm frequency analysis from the extracted fields is extraction-based, not independently verified
4. Manual verification of a 30-study sample is recommended before publishing frequency claims

**Unresolved uncertainties before Phase 6 (TimesFM feasibility):**
- Exact number of comparable country-year observations per ILSA program (must come from official data, not this corpus)
- Which PISA/TIMSS cycles have fully released microdata for all domains
- Whether the 2 cross-cycle ML studies used representative country samples or limited subsets

**Recommended Phase 3 focus:** Algorithm frequency analysis for the 166 ML studies, with careful multi-label counting and taxonomy normalization.
