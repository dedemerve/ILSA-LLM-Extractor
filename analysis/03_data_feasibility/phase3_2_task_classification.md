# Phase 3.2 — Study Task Classification (Phase B)

**Date:** 2026-10-14  
**Protocol:** Phase 3.2 Full Specification, Phase B  
**Status:** COMPLETE — Reuses Phase 2 classification; adds multidimensional extension  
**Input:** `analysis/02_prediction_taxonomy/study_classification.csv` (1,266 rows)  
**This file does NOT overwrite any prior artifact.**

> **Scope correction (2026-10-15):** The 1,266-record classifications and within-corpus counts below remain descriptive of this fixed file. A row-level review of all 28 heuristic cross-cycle flags is now available in `phase3_14_cross_cycle_review_and_scope_correction.md`. Independent source search also verified a 2026 paper forecasting PISA 2022 country means from cycles through 2018 (DOI 10.3389/feduc.2026.1734988) that is absent from this file. Consequently, the “0 confirmed” result is conditional on the coded corpus and cannot support an exhaustive literature-wide zero; corpus search coverage is unresolved.

---

## B.1 Classification Source

The Phase 2 pipeline (committed in 9f2bf5d ancestry) produced `study_classification.csv` containing 34 fields per study. This Phase B report documents, extends, and validates that classification.

Phase 2.2.1 (analysis/02_prediction_taxonomy/phase2_2_1_addendum.md) verified the Braun et al. (2025) TIMSS QA study at source, confirming it as a retrospective analysis, not a genuine future-cycle forecasting study.

The existing classification is **reused and extended** — not replaced or overwritten.

---

## B.2 Task Distribution (Verified Counts)

### B.2.1 Research objective (Dimension 1)

| Code | Label | n | % |
|------|-------|---|---|
| G | non_predictive_analytical | 1,100 | 86.9% |
| A | student_within_cycle_prediction | 96 | 7.6% |
| F | other_predictive | 45 | 3.6% |
| D | country_system_prediction | 17 | 1.3% |
| B | student_cross_cycle_prediction | 8 | 0.6% |
| **Total** | | **1,266** | 100% |

**Verified:** Counts reproduced from `study_classification.csv` using `primary_task_code.value_counts()`.

### B.2.2 Prediction target level (Dimension 3)

| Level | n | % |
|-------|---|---|
| unclear | 554 | 43.8% |
| student | 373 | 29.5% |
| country_system | 339 | 26.8% |

Note: `country_system` (339) includes both predictive and non-predictive studies at country level (e.g., descriptive cross-national comparisons). Only 17 are classified as predictive (`primary_task_code = D`).

### B.2.3 Temporal design (Dimension 4)

| Status | n | % |
|--------|---|---|
| not_applicable (technical reports) | 545 | 43.1% |
| no_temporal_prediction | 538 | 42.5% |
| within_cycle_prediction | 155 | 12.2% |
| cross_cycle_prediction_not_fully_verified | 28 | 2.2% |
| **genuine future_cycle forecasting** | **0** | **0.0%** |

The `cross_cycle_prediction_not_fully_verified` category (n=28) consists of studies that use data from multiple cycles but where:
- Target cycle outcomes may have been used in model fitting (leakage risk)
- Temporal holdout was not verified at source
- Study descriptions were ambiguous about holdout design

**Key finding:** Zero studies have been confirmed as performing genuine future-cycle ILSA forecasting (where a held-out future cycle was not used during model development). The 28 flagged studies require source inspection to classify more precisely. The Braun et al. (2025) study — the closest candidate — was confirmed at source as a retrospective analysis (Phase 2.2.1).

---

## B.3 Operational Definition of Genuine Future-Cycle Forecasting

A study satisfies this definition if and only if:

1. It predicts an ILSA outcome for assessment cycle T.
2. The model was fit, selected, and tuned using only data from cycles < T.
3. Cycle T outcomes were not used in feature engineering, hyperparameter selection, or threshold setting.
4. Predictions were evaluated against cycle T outcomes (genuine holdout).

Studies failing any condition are classified as NOT genuine future-cycle forecasting.

The 28 `cross_cycle_prediction_not_fully_verified` studies do not meet this definition without source verification. Based on Phase 2 and 2.2.1 evidence, the confirmed count is **0 genuine future-cycle forecasting studies** in the 1,266-study corpus.

---

## B.4 Predictive Studies Detail (n=166)

### B.4.1 Student-level predictive studies (n=155 = codes A+F+B)

Most predictive studies predict individual student outcomes (achievement scores, classifications, school dropout) within a single cycle or across cycles where leakage cannot be ruled out.

| Code | Description | n |
|------|-------------|---|
| A | Student-level, within-cycle | 96 |
| F | Other predictive (unclear unit) | 45 |
| B | Student-level, cross-cycle (not verified as genuine holdout) | 8 |
| Subtotal | student-focused | **149** |

### B.4.2 Country-system predictive studies (n=17, code D)

These 17 studies predict or estimate country-level outcomes. However, "country-level prediction" in this corpus includes:
- Cross-national regression/econometric models (most common)
- Trend analyses comparing national scores across cycles
- Predictive models of national policy outcomes
- The TIMSS QA study (Braun et al. 2025: retrospective, in-sample)

**None of the 17 country-system predictive studies has been verified as genuine future-cycle forecasting.**

---

## B.5 Multidimensional Classification Summary

The following table combines all five dimensions for the 166 predictive studies:

| Dimension | Distribution |
|-----------|-------------|
| Research objective | Predictive (n=166) |
| Prediction target | Student (n=101), Country/system (n=15), Unclear (n=50) |
| Unit of analysis | Student (n=101), Country (n=15), Other/unclear (n=50) |
| Temporal design | Within-cycle (n=155), Cross-cycle unverified (n=11) |
| Evaluation design | Random split / CV (typical), Genuine temporal holdout (n=0 confirmed) |

---

## B.6 Classification Validation

Phase 2 included a 30-study validation sample (`analysis/02_prediction_taxonomy/validation_sample_30.csv`). Classification was reviewed for consistency.

**Known limitation:** All classification was performed by the same automated pipeline. No independent human rater compared labels. Inter-rater reliability cannot be computed. Classification confidence is REPORTED as high for clear-cut studies (most technical reports = G, most student-level ML studies = A), but UNRESOLVED for the 28 cross-cycle cases.

---

## B.7 Summary Answers for Phase B

| Question | Answer |
|----------|--------|
| Studies in each category | Reported in B.2 |
| Studies with multiple task labels | Not separately coded (one primary code per study) |
| Studies with insufficient evidence | 28 cross_cycle (temporal design unclear) |
| Studies with temporal evaluation details | 155 within-cycle (well-documented); 28 ambiguous |
| Studies meeting genuine future-cycle forecasting | **0 confirmed** (28 unresolved at source) |
