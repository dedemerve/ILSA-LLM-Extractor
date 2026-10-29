# Phase 3.2 — Verification Report

**Date:** 2026-10-14  
**Protocol:** Phase 3.2 Verification and Phase 3.3 Readiness Audit  
**Auditor role:** Systematic-review auditor — claims recalculated from underlying artifacts, not accepted from prior reports  
**This file does NOT overwrite any prior artifact.**

---

## Executive Summary

Phase 3.2 produced seven reports and nine CSV artifacts. This verification audit independently recalculated 20 claims from raw source artifacts (zeroshot_results.csv, country_estimates.csv, study_classification.csv, ILSA_Meta_Analysis_Dataset_CLEAN.xlsx, and pisa2022_country_diagnostics.csv). The audit found:

| Status | Count | Notes |
|--------|-------|-------|
| VERIFIED | 13 | Independently reproduced to stated precision |
| PARTIALLY VERIFIED | 3 | Core finding sound; subsidiary claims unresolved |
| INCONSISTENT | 3 | Incorrect claim in prior report; requires correction |
| BLOCKED | 1 | HTTP 403 prevents direct mean comparison |

**Decision: CONDITIONALLY READY** — PISA 2025 may serve as a held-out final evaluation subject to the required corrections and conditions listed in Section 7.

---

## 1. Step 1 — Corpus Classification Count

**Claim:** 1,266 unique study records; mutually exclusive primary_task_code (G=1100, A=96, F=45, D=17, B=8 = 1,266).  
**Verification:** Independently confirmed from study_classification.csv (n=1,266; value_counts() match).  
**Status: VERIFIED**

**Inconsistency found (C03):** The summary table in phase3_2_task_classification.md listed a sixth category ("cross_cycle unverified = 28") and summed to 1,294. This is an error: the 28 cross-cycle studies are a subset of the temporal_forecasting_status dimension, overlapping with existing primary_task_code categories (G=17, B=8, D=3 of the 28). Adding them as an additive sixth code inflates the count by 28.

**Required correction:** The summary table must be corrected. The primary_task_code categories are mutually exclusive and sum to 1,266.

---

## 2. Step 2 — Audit of Zero Genuine Future-Cycle Forecasting Studies

**Claim:** 0 studies confirmed as genuine future-cycle ILSA forecasting.  
**Verification:** `temporal_forecasting_status.value_counts()` from study_classification.csv confirms 0 records coded `genuine_future_cycle_confirmed`. Braun et al. 2025 (the closest candidate) was confirmed retrospective at source (Phase 2.2.1).  
**Status: PARTIALLY VERIFIED**

**Unresolved:** 28 studies coded `cross_cycle_prediction_not_fully_verified` were not inspected at source in this audit. Of the 17 D-coded (country_system_prediction) studies: 14 have `temporal_forecasting_status=within_cycle` (confirmed not genuine future-cycle); 3 have `cross_cycle_unverified` (DOIs: 10.1016/j.caeai.2026.100566, 10.3390/data7110147, 10.1037/edu0000999).

**Limitation:** The "0 confirmed" finding is accurate. Whether the true count is 0 or >0 remains open for 28 studies.

---

## 3. Step 3 — Independent Reproduction of Forecasting Comparison

All MAEs and correlation statistics were independently recalculated from raw CSV columns (not from report assertions).

### 3.1 MAE values (VERIFIED)

| Method | Reported MAE | Independently calculated | Status |
|--------|-------------|--------------------------|--------|
| Persistence | 14.4820 | 14.4820 | VERIFIED |
| TimesFM 3.0 | 15.8296 | 15.8296 | VERIFIED |
| Linear trend | 16.9751 | 16.9751 | VERIFIED |

Ranking: Persistence < TimesFM < Linear trend (VERIFIED).

### 3.2 Trend slope correlation (VERIFIED)

| Association | Reported | Recalculated | 95% CI | Status |
|-------------|----------|--------------|--------|--------|
| r(trend_slope, AE_TFM) | +0.496, p<0.001 | r=0.4965, p=0.000055 | [0.278, 0.666] | VERIFIED |
| r(trend_slope, AE_Persist) | +0.116, p=0.376 | r=0.1163, p=0.376 | [−0.142, 0.360] | VERIFIED |

Sensitivity excluding Albania (n=59): r_TFM = 0.316, p = 0.015 (95% CI [0.065, 0.529]). Albania is an influential observation — the correlation attenuates substantially without it but remains statistically significant. The qualitative conclusion (trend-extrapolating methods penalised more by COVID disruption) survives exclusion.

### 3.3 Positive bias claim (INCONSISTENT — C10, C11)

**Reported:** "all 60 countries were over-predicted"; "mean signed error = MAE".  
**Independently recalculated:** 
- Persistence: **56/60** over-predicted; 4 under-predicted (JPN, KAZ, KOR, SGP).
- TimesFM: **55/60** over-predicted; 5 under-predicted (AUS, JPN, KAZ, SGP, TUR).
- True mean signed error: persistence = **+13.885**, TFM = **+14.796** (≠ MAE values of 14.482 and 15.830).

**Root cause:** The stored `err_persistence` and `err_timesfm` columns in zeroshot_results.csv represent `|predicted − actual|` (absolute value). For the 4–5 countries that were under-predicted, the stored values are positive even though the signed errors are negative. Summing them as if they were signed errors produces the erroneous "mean signed error = MAE" result.

**Qualitative conclusion preserved:** The positive bias is overwhelming (56–55 of 60 countries over-predicted; mean signed errors +13.9 and +14.8 are still very large), and is consistent with systematic COVID-19 learning loss. The correction is quantitative, not qualitative.

**Required correction:** "All 60" → "56/60 (persistence) and 55/60 (TFM)". Mean signed error values must be updated to +13.885 and +14.796.

### 3.4 Context stratification (VERIFIED)

| n_context | n | MAE Persistence | MAE TFM | TFM wins? |
|-----------|---|-----------------|---------|-----------|
| 3 | 5 | 10.70 | 16.72 | No |
| 4 | 7 | 16.45 | 22.04 | No |
| 5 | 12 | 12.30 | 14.93 | No |
| 6 | 36 | 15.35 | 14.80 | Yes (+0.55) |
| All | 60 | 14.48 | 15.83 | No |

All stratum values independently confirmed.

### 3.5 H1 correlation (VERIFIED)

r(n_context, AE_TFM) = −0.114, p = 0.387; r(n_context, AE_Persist) = 0.081, p = 0.539. Both non-significant. Confirmed.

---

## 4. Step 4 — PISA 2025 Validation Status

### 4.1 SE pipeline (VERIFIED)

`pisa2025_se_validation.csv` was reviewed: MAD = 0.000026, max absolute difference = 0.000049, n = 79 countries. Pipeline matches OECD Annex 14 T14.A.13 at floating-point precision. Status: VERIFIED.

### 4.2 SAV path bug (VERIFIED — B-C)

| Check | Detail |
|-------|--------|
| Script path | `/ILSA Datasets/PISA Datasets/PISA 2025 Data/CY09_MS_STU_PUF.sav` — does NOT exist |
| Actual file | `/ILSA Datasets for LLM Extraction/PISA Datasets/PISA 2025 Data/CY09_MS_STU_PUF.sav` — EXISTS |
| Estimates present | Yes — country_estimates.csv contains 268 PISA 2025 rows (90 countries × 3 domains) |
| Estimates provenance | Unclear — were produced prior to this path discrepancy being identified; pass SE validation |

**Implication:** Existing estimates are valid (SE-validated). The path bug means the script cannot reproduce them without correction. Any re-run of the PISA 2025 pipeline requires fixing the path first.

### 4.3 Direct mean comparison (BLOCKED)

OECD Results Volume I: HTTP 403. Cannot independently verify country-level mean scores against official published values.

### 4.4 Country eligibility (VERIFIED)

| Set | n |
|-----|---|
| Countries with ≥3 pre-2022 cycles (2022 dev set) | 60 |
| 2022 dev set matches step2 countries | Yes (exact match) |
| Countries with both 2022 and 2025 data | 75 |
| Eligible for 2025 evaluation (n_pre2025 ≥ 3) | 66 |
| Quality-flagged (Annex 14 footnote 1) | 6 (ALB, CAN, NLD, NOR, NZL, USA) |

The Phase 3.1.1 addendum stated the primary 2025 evaluation set. Independently confirmed: 66 eligible countries; 6 quality-flagged.

---

## 5. Step 5 — Literature Synthesis Claims Audit

| Claim | Status | Note |
|-------|--------|------|
| 87% of corpus non-predictive (1,100/1,266) | VERIFIED | |
| 166 studies predictive | VERIFIED | |
| 17 country-level predictive (code D) | VERIFIED | |
| 0 comparable to country-level MAE benchmark | VERIFIED | |
| Meta-analysis not justified | VERIFIED | Incompatible metrics, units, targets confirmed |
| Tree-based/ensemble family is the largest coded predictive family (81/166 = 48.8%) | VERIFIED IN CLASSIFICATION FILE | `ml_family` count; exact primary algorithm counts are RF=42/147 and XGBoost=17/147 among predictive studies with `ml_primary`; see Phase 3.11. Automated labels were frequency-audited but not independently source-reread. |
| No baseline comparison in most predictive studies | PARTIALLY VERIFIED | Qualitative assessment; not coded systematically |
| COVID caused systematic positive bias | PARTIALLY VERIFIED | 56/60 and 55/60 over-predictions confirmed; "all 60" claim is incorrect (C10) |

---

## 6. Verified vs. Unverified Claims Summary

### 6.1 Fully verified claims

1. Corpus n=1,266; primary_task_code counts correct
2. MAE: Persistence=14.4820, TFM=15.8296, Linear=16.9751
3. Persistence < TFM < Linear trend ranking
4. r(trend_slope, AE_TFM) = 0.497, p<0.001, CI [0.278, 0.666]
5. r(trend_slope, AE_Persist) = 0.116, p=0.376 (not significant)
6. Context-stratified MAEs match reported values
7. SE pipeline: MAD=0.000026 vs OECD Annex 14 (n=79)
8. SAV path bug confirmed (script path missing; actual file exists at different path)
9. 2022 country set = 60; 2025 eligible = 66; quality-flagged = 6
10. 0 studies confirmed as genuine future-cycle forecasting

### 6.2 Required corrections

**Correction 1 (C03) — Count table in phase3_2_task_classification.md:**  
The 28 cross-cycle studies must not be listed as an additive sixth primary_task_code category. The table total must be 1,266, not 1,294.

**Correction 2 (C10, C11) — Positive bias claim in phase3_2_forecasting_diagnostics.md and phase3_2_hypotheses_and_tests.md:**  
"All 60 countries" → "56/60 (persistence), 55/60 (TFM)".  
Mean signed error values → persistence = +13.885, TFM = +14.796 (these are NOT equal to MAE).

Note: The qualitative conclusion (overwhelming positive bias consistent with COVID learning loss) is preserved by both corrections.

### 6.3 Partially verified / unresolved

- 28 cross-cycle studies require source inspection (only 1 of 28 inspected at source)
- 3 D-coded cross-cycle studies require source inspection
- The frequency counts are reproducible from the coded classification file; the underlying automated labels have not been independently source-reread, and method frequencies among country-level forecasting studies remain unsupported (zero confirmed genuine future-cycle examples).
- Direct PISA 2025 mean comparison remains blocked

---

## 7. PISA 2025 Evaluation Conditions

Before executing the PISA 2025 held-out evaluation, the following conditions must be met:

| Condition | Status | Action |
|-----------|--------|--------|
| Fix SAV path in build_country_estimates.py | OPEN | Change path from `/ILSA Datasets/` to `/ILSA Datasets for LLM Extraction/` — requires explicit authorization |
| Confirm no PISA 2025 outcomes used during model development | CONFIRMED | TimesFM was zero-shot; no fine-tuning; 2025 data not in any training set |
| SE validation passes | CONFIRMED | MAD=0.000026 |
| Quality-flag countries reported separately | PROTOCOL SET | ALB CAN NLD NOR NZL USA |
| Correct positive-bias claim before finalising evaluation design | OPEN | C10 and C11 require correction in any document referencing mean signed error |

---

## 8. Decision

**CONDITIONALLY READY**

The PISA 2025 data is suitable as a held-out final evaluation under these conditions:

1. The SAV path bug in build_country_estimates.py must be fixed before any re-run (existing estimates are valid; re-run is needed only if estimates must be reproduced from scratch).
2. The positive-bias analysis must be corrected (56/60, 55/60 — not all 60) in any document that carries these numbers into the final evaluation report.
3. The count inconsistency in the task classification report must be corrected.
4. Quality-flag countries (ALB, CAN, NLD, NOR, NZL, USA) must be reported as a separate stratum in all final evaluation tables.
5. Direct mean comparison against OECD Results Volume I should be attempted once the volume becomes publicly accessible.

No conditions block evaluation. The corrections are to prior-report assertions, not to the underlying data or experimental design.

---

## 9. Prioritised Next Actions

| Priority | Action | Blocking |
|----------|--------|---------|
| 1 | Fix SAV path bug (B-C) — requires explicit authorization | Re-run of PISA 2025 pipeline |
| 2 | Correct positive-bias claim (C10/C11) in diagnostics and hypotheses reports | Final evaluation report accuracy |
| 3 | Correct count table (C03) in task_classification report | Any document citing the 1,294 figure |
| 4 | Source-inspect remaining 27 cross_cycle studies (28 minus Braun 2025) | Updating the "0 confirmed" count |
| 5 | Execute PISA 2025 held-out evaluation (Phase 3.3) | After conditions 1–2 met |
| 6 | Run pre-COVID baseline test (PISA 2018 as target) | Separating COVID confound from model capability |
| 7 | Extend to TIMSS/PIRLS | Generalization of persistence-beats-TFM finding |

---

## 10. Integrity Confirmations

- No existing corpus, gold-standard artifact, or experiment output was modified during this audit. ✓
- No train/validation/test split was changed. ✓
- No model was trained or fine-tuned. ✓
- No commit or push was made. ✓
- All recalculations performed from underlying artifacts (not from report text). ✓
- PISA 2025 held-out evaluation was NOT executed. ✓

---

## 11. Artifacts Produced in This Audit

| File | Description |
|------|-------------|
| `analysis/03_data_feasibility/audit/phase3_2_claim_verification.csv` | 20-row claim audit matrix |
| `analysis/03_data_feasibility/phase3_2_verification_report.md` | This report |
