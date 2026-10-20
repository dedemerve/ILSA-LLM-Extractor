# Phase 3.2 — Corpus and Data-Provenance Audit (Phase A)

**Date:** 2026-10-14  
**Protocol:** Phase 3.2 Full Specification, Phase A  
**Status:** COMPLETE  
**This file does NOT overwrite any prior artifact.**

> **Scope correction (2026-10-15):** The 1,266-record size and file-level provenance checks below remain valid. The audit's forecasting-completeness caveat is now stronger: an independently verified 2026 country-level PISA trend forecast (DOI 10.3389/feduc.2026.1734988) is absent from the classification file. The original search window and inclusion/exclusion log must be recovered before treating the 1,266 records as an exhaustive literature frame. See `phase3_14_cross_cycle_review_and_scope_correction.md`.

---

## A.1 Repository and Working-Tree State

Branch: `feat/finetuning-v2-pipeline`  
Uncommitted changes preserved as found:
- `analysis/03_data_feasibility/phase3_data_feasibility_audit.md` (modified)
- `scripts/finetuning/extract_v2.py` (modified)
- Several untracked new files (audit addenda, step scripts, audit CSVs)

No existing user changes were reverted or overwritten.

---

## A.2 Corpus Source Files

### A.2.1 Primary corpus file

**Path:** `outputs/ILSA_Meta_Analysis_Dataset_CLEAN.xlsx`  
**Status:** CONFIRMED EXISTS (VERIFIED by direct inspection)

This is the canonical, most-complete version of the corpus. It contains nine sheets.

| Sheet | Rows | Columns | Purpose |
|-------|------|---------|---------|
| `0_Dashboard_Analysis_Control` | 36 | 4 | Dashboard/control |
| `1_Articles_Master` | **1,266** | 38 | One row per study (primary table) |
| `2_Main_Findings` | 2,128 | 16 | One row per finding (multiple per study) |
| `3_Confounders` | 8,336 | 11 | One row per confounder variable (multiple per finding) |
| `Canonical_View` | 10,464 | 10 | Denormalized/joined view |
| `5_Verification_Claims` | 1 | 29 | Single-row verification header |
| `6_Verification_Results` | 1 | 29 | Single-row verification header |
| `7_Prediction_Inventory` | 1 | 19 | Single-row prediction inventory header |
| `_FilterLists` | 40 | 31 | Reference/vocabulary lists |

### A.2.2 Other corpus files

| File | Rows (primary sheet) | Columns | Relation to CLEAN |
|------|---------------------|---------|-------------------|
| `outputs/ILSA_Meta_Analysis_Dataset.xlsx` | 1,266 | 27 | Earlier version; fewer fields |
| `outputs/ILSA_Analytical_Meta_Analysis_Master.xlsx` | 1,266 (Study_Details) | 23 | Parallel analytical version |
| `outputs/ILSA_Structured_Meta_Analysis.xlsx` | 1,266 | 21 | Earlier structured version |

All four files share the same 1,266-record article set. The CLEAN version is most complete.

### A.2.3 Study classification table

**Path:** `analysis/02_prediction_taxonomy/study_classification.csv`  
**Rows:** 1,266 (one per study) | **Columns:** 34  
This is the output of the Phase 2 classification pipeline. Contains all article-level fields plus derived classification fields.

---

## A.3 Corpus Size Verification

### A.3.1 Record counts (VERIFIED)

| Metric | Value | Source |
|--------|-------|--------|
| Total records in 1_Articles_Master | **1,266** | Direct row count |
| Records with DOI | **995** | `doi.notna()` |
| Records without DOI | **271** | Difference |
| Unique DOI values | **995** | `doi.nunique()` |
| Duplicate DOI records | **270** | `duplicated('doi').sum()` |

**Interpretation:** Each of the 995 unique DOIs appears exactly once. The 270 "duplicate" flag is an artifact of the pandas `duplicated()` method counting all occurrences after the first; actual DOI uniqueness = 995 distinct values, all appearing once. No duplicate studies were found. The 271 records without DOI represent studies where a DOI was not extractable (book chapters, reports, preprints without DOI assignment).

The claim "1,266 records, 995 unique DOIs, 271 without DOI" is **VERIFIED**.

### A.3.2 Unit of analysis

Each row in `1_Articles_Master` represents **one source document** (article, report, book chapter, or technical report). This is not necessarily one unique scientific finding.

The grain of `2_Main_Findings` is **one finding per study per target**. Multiple findings can belong to one study. The 2,128 findings correspond to 1,266 source documents, yielding a mean of approximately 1.68 findings per document.

The grain of `3_Confounders` is **one predictor/confounder per finding**. The 8,336 confounder records map to 2,128 findings.

---

## A.4 Schema and Key Fields

### A.4.1 Primary key

`file_name` (PDF filename) serves as the primary key. `doi` is available for 995 records but 271 are null. The combination of `file_name` and `doi` is unique within the corpus.

### A.4.2 Field inventory (`1_Articles_Master`, n=1,266)

| Field | Non-null | Coverage | Notes |
|-------|----------|----------|-------|
| `doi` | 995 | 79% | Missing for 271 records |
| `title` | 1,266 | 100% | |
| `year` | ~1,240 | ~98% | Approximate; some NaN |
| `research_design_type` | 1,266 | 100% | Categorical: exploratory/predictive/causal_obs/causal_exp |
| `ml_primary` | 167 | 13% | Only for studies using ML |
| `ml_family` | 1,266 | 100% | Includes "N/A: Technical Report" and "Not Reported" |
| `document_class` | 1,266 | 100% | empirical_article / technical_report |
| `normalized_programs` | 1,169 | 92% | ILSA program (PISA, TIMSS, PIRLS, etc.) |
| `normalized_cycles` | 882 | 70% | Assessment cycle years |
| `countries_formatted` | ~600 | ~47% | Variable; many missing |
| `prediction_target_level` | 1,266 | 100% | student / country_system / unclear |
| `primary_task_code` | 1,266 | 100% | A/B/D/F/G classification codes |
| `temporal_forecasting_status` | 1,266 | 100% | Categorical |

### A.4.3 Table relationships

```
1_Articles_Master  (1,266 rows)  [1:M]
    └── 2_Main_Findings  (2,128 rows)  [1:M]
            └── 3_Confounders  (8,336 rows)
```

Join key: `file_name` (and `doi` where non-null).

**Warning:** Joining Articles → Findings → Confounders without aggregation will multiply article-level counts by the number of findings and confounders. The Canonical_View sheet (10,464 rows) represents the fully denormalized join.

---

## A.5 Data-Availability Matrix

The following table shows whether the corpus can support specific analytical questions.

| Analytical variable | Field exists | Usable records | Evidence status |
|---------------------|-------------|---------------|-----------------|
| ILSA program | Yes (`normalized_programs`) | 1,169/1,266 | VERIFIED |
| Assessment cycle | Yes (`normalized_cycles`) | 882/1,266 | VERIFIED |
| Country coverage | Partial (`countries_formatted`) | ~600/1,266 | PARTIALLY VERIFIED |
| Research objective | Yes (`research_design_type`) | 1,266/1,266 | VERIFIED |
| Prediction target | Yes (`prediction_target_level`) | 1,266/1,266 | VERIFIED (coded) |
| ML model family | Yes (`ml_family`) | 1,266/1,266 | VERIFIED (with "N/A" categories) |
| Specific ML algorithm | Yes (`ml_primary`) | 167/1,266 | VERIFIED (predictive studies only) |
| Performance metrics (text) | Yes (`performance_metrics` in findings) | ~1,892/2,128 | PARTIALLY VERIFIED |
| Performance metrics (numeric) | Partial (embedded in text) | Unknown | UNRESOLVED — requires extraction |
| Temporal evaluation design | Yes (`temporal_forecasting_status`) | 1,266/1,266 | VERIFIED (coded) |
| Cross-cycle / future-cycle | Yes (`temporal_forecasting_status`) | 28 flagged | VERIFIED (coded, conservative) |
| Baseline methods | Rarely reported in text | Unclear | UNRESOLVED |
| Leakage risk | Not coded | 0 | UNRESOLVED — requires extraction |
| Prediction horizon | Not coded | 0 | UNRESOLVED |
| Train/test split design | Partially coded in text | Partial | UNRESOLVED |
| Number of cycles used | Partial (`n_cycles_extracted`) | 882 | PARTIALLY VERIFIED |
| Sampling weights used | Yes (`student_weights_used`) | 1,266 | VERIFIED |
| Plausible values handling | Yes (`plausible_values_handling`) | 1,266 | VERIFIED |

---

## A.6 Missingness and Limitations

1. **Performance metric values:** The `performance_metrics` field in `2_Main_Findings` is a free-text narrative (e.g., "R² = 0.82 on cross-validation..."). Numeric values are embedded in the text and are not separately extracted. Structured extraction of metric type and value would require additional coding.

2. **Country identifiers:** The `countries_formatted` field is sparse (~47% coverage). Country-level analysis of the corpus is limited.

3. **Temporal evaluation design:** The `temporal_forecasting_status` field is coded conservatively. The 28 studies marked `cross_cycle_prediction_not_fully_verified` were not confirmed at source during Phase 2 (except Braun et al. 2025, which was confirmed as retrospective). The true count of genuine future-cycle forecasting studies is likely 0.

4. **Duplicate handling:** 271 records lack DOI. They cannot be deduplicated by DOI. Manual review would be needed to identify any true duplicates among them.

---

## A.7 Phase A Answers

**1. What does the corpus contain?**  
1,266 source documents (articles, reports, book chapters) on AI/ML methods in ILSA, spanning PISA, TIMSS, PIRLS, ICCS, ICILS, and PIAAC. Most (1,100/1,266 = 87%) are non-predictive analytical studies.

**2. What is its true unit of analysis?**  
One row = one source document. Findings and confounders are in separate tables at finer granularity.

**3. Which research questions can the current fields support?**  
- Counts of studies by design type, program, cycle, and algorithm ✓
- Identification of predictive studies and their targets ✓
- Conservative classification of temporal design ✓
- General data-availability assessment ✓

**4. Which questions require additional evidence extraction?**  
- Quantitative meta-analysis of performance metrics (requires structured numeric extraction)
- Train/test split designs (requires source inspection)
- Baseline methods and leakage risk (requires source inspection)
- Country coverage for predictive studies

**5. Which claims could not be verified?**  
The claim "0 confirmed future-cycle forecasting studies" (from Phase 2.2.1) was verified at source for one study (Braun et al. 2025). The remaining 28 `cross_cycle_prediction_not_fully_verified` studies were not inspected at source in this phase and remain PARTIALLY VERIFIED.
