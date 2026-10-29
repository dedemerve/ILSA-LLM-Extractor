# Phase 3.3 — Final Evaluation Protocol (LOCKED)

**Date:** 2026-10-14  
**Protocol:** Phase 3.3 Final Evaluation Protocol Lock, Task 3  
**Status:** PROTOCOL LOCKED — execution not yet authorised  
**This file does NOT overwrite any prior artifact.**

> **Execution/status update (2026-10-15):** The planned work has since been carried out and expanded to all three domains and eight methods in the outcome-exposed retrospective benchmark. This document remains the historical protocol record; its statements that no TimesFM 2025 predictions exist and that the evaluation is prospective are no longer current. See `phase3_10_pisa2025_retrospective_evaluation.md` and `phase3_12_oecd_pisa2025_mean_validation.md` for executed results and current interpretation.

---

## 0. Scope and Purpose

This document defines the complete, pre-registered evaluation protocol for a future TimesFM-specific PISA 2025 final evaluation. All design decisions are based on development evidence (PISA 2022 development sample) that excludes PISA 2025 outcomes. No setting in this protocol may be changed after inspecting PISA 2025 TimesFM predictions.

---

## 1. Target Variable and Assessment Cycle

| Item | Value |
|------|-------|
| Program | PISA |
| Domain | Mathematics |
| Target cycle | **2025** |
| Target variable | Country-level mean mathematics score from `outputs/stage4/country_estimates.csv` (pipeline: BRR/FAY k=0.5, R=80, Rubin m=10 PVs) |
| Context | Pre-2025 PISA mathematics cycles from `outputs/stage4/country_estimates.csv` |

---

## 2. Eligible Country Definition

| Criterion | Value | Basis |
|-----------|-------|-------|
| Must have 2025 estimate | Yes | From country_estimates.csv |
| Minimum historical context | **≥3 pre-2025 cycles** in country_estimates.csv (identical to the 2022 MIN_CONTEXT=3 rule) | Development protocol |
| Result | **66 countries** | Verified from eligibility CSV |

The 66-country set is locked. Any country added to or removed from this set must be reported as a protocol deviation with explicit justification.

---

## 3. Treatment of Missing Values

| Rule | Value |
|------|-------|
| Countries missing 2025 estimate | Excluded from evaluation (ineligible by definition) |
| Countries with ≥3 pre-2025 cycles but no 2025 estimate | Not applicable — all 66 eligible countries have 2025 estimates |
| Partial cycle history (gap years) | Included; persistence uses the last non-missing cycle; linear trend uses all available pre-2025 observations |
| Missing replicate weights in SAV | Not applicable — PUF has full replicate weight coverage |

---

## 4. Treatment of Quality-Flagged Countries

| Country | Flag | Treatment |
|---------|------|-----------|
| ALB | Annex 14 footnote 1 caution | Included in primary analysis; reported in separate stratum |
| CAN | Annex 14 footnote 1 caution | Included in primary analysis; reported in separate stratum |
| NLD | Annex 14 footnote 1 caution | Included in primary analysis; reported in separate stratum |
| NOR | Annex 14 footnote 1 caution | Included in primary analysis; reported in separate stratum |
| NZL | Annex 14 footnote 1 caution | Included in primary analysis; reported in separate stratum |
| USA | Annex 14 footnote 1 caution | Included in primary analysis; reported in separate stratum |

**Primary n = 66 (all eligible including quality-flagged). Secondary n = 60 (quality-flagged excluded). Both strata reported in every table.**

Albania must be highlighted individually in all tables given its anomalous 2022 result.

---

## 5. Methods

### 5.1 Persistence baseline

Prediction = last observed PISA mathematics cycle score before 2025 (= 2022 for countries with 2022 data; 2018 or earlier for any country without 2022).  
**For all 66 eligible countries: last observed cycle before 2025 will be 2022 (all 66 have 2022 estimates — confirmed by S6 = 60 countries ∩ S4, plus the 6 new countries DOM/KSV/LUX/MKD/TAP/VNM which have 2022 data too).**

### 5.2 Linear trend baseline

OLS regression of pre-2025 cycle scores on cycle year (numeric); predict at year 2025. Requires ≥2 pre-2025 observations (all 66 meet this). Implementation: identical to `analysis/03_data_feasibility/audit/linear_trend_baseline_2022.csv` generation.

### 5.3 TimesFM 3.0 zero-shot

Model: `google/timesfm-3.0-pytorch`, zero-shot, horizon=1.  
Input: all pre-2025 cycle scores (not scaled; same series format as step2).  
Output: single point prediction for 2025.  
**No fine-tuning. No parameter updates. Zero-shot only.**  
Implementation: adapt `scripts/step2_timesfm_zeroshot.py` with `TEST_CYCLE = 2025`.

---

## 6. Error Metrics

| Metric | Formula | Reported for |
|--------|---------|-------------|
| MAE | mean(|ŷ − y|) | All three methods × full 66 × clean 60 |
| Mean signed error (MSE) | mean(ŷ − y) | All three methods × full 66 × clean 60 |
| n over-predicted | count(ŷ > y) | All three methods |
| n under-predicted | count(ŷ < y) | All three methods |
| Country-level errors | (ŷ − y) per country | CSV output, all three methods |

**Error sign convention:** positive = overprediction (predicted > actual). Both MAE and signed error computed from raw prediction columns, not from stored error columns. This avoids the sign-convention bug found in the 2022 experiment outputs.

---

## 7. Common-Country Comparison Rule

If any method produces predictions for a different set of countries (e.g., TimesFM fails on a specific country), the comparison is computed on the **intersection** of all three method country sets. The original per-method n and any excluded countries are reported explicitly.

---

## 8. Uncertainty and Sensitivity Analyses

All of the following are **pre-specified** and must be reported regardless of outcome:

| Analysis | Definition |
|---------|-----------|
| S1: Exclude quality-flagged | n=60; compare MAE vs primary n=66 |
| S2: Exclude Albania | n=65 (or 59 for clean); compare MAE |
| S3: n_context stratification | Stratify by number of pre-2025 cycles (3, 4, 5, 6, 7+); report n and MAE per stratum |
| S4: Trend slope vs. AE_TFM | Pearson r AND Spearman r; report both; 95% CI; label as exploratory regardless of p-value |
| S5: COVID-cycle comparison | If PISA 2018 evaluation is available by this time, compare 2025 error rates to 2018 to assess whether COVID confound persists |

> **Arithmetic correction (2026-10-15):** Albania (ALB) is included among the six quality-flagged countries above. Therefore the clean sample is n=60, and removing Albania after excluding those six still leaves n=60. The parenthetical “59 for clean” in S2 is incorrect. The later Phase 3.13 sensitivity report applies the consistent counts.

---

## 9. Reporting Requirements

Every table must include:
- n (total eligible, quality-flagged stratum, clean stratum)
- MAE and MSE for all three methods
- Head-to-head comparison: which method wins per country (count)
- Explicit statement of exclusions and reasons

The evaluation report must include:
- Statement that this is the **single pre-specified held-out evaluation** for TimesFM on PISA 2025
- Statement that PISA 2025 was used as a LOCO test set in a prior Ridge regression analysis (see Section 10 below)
- Caveat that conclusions about model capability remain confounded if any post-2019 shock affected 2025 outcomes

---

## 10. PISA 2025 Prior Exposure — Required Disclosure

**PISA 2025 country-level mathematics outcomes are not project-blind.** The LOCO backtesting pipeline (`outputs/stage5/loco_results.csv`, `outputs/stage5/loco_predictions.csv`) used PISA 2025 as a test cycle for Ridge regression models (M0, M0A, M1, M2, M3). Aggregate performance and row-level predictions/residuals are available in the project.

**TimesFM specifically** has not been run on PISA 2025. No TimesFM 2025 predictions exist anywhere in the project. The TimesFM evaluation on 2025 is genuinely prospective.

**Consequence for interpretation:** The evaluation is prospective for TimesFM only. It cannot be claimed that the overall evaluation design is "blind" to PISA 2025 outcomes, because Ridge/persistence LOCO results were available before the TimesFM protocol was finalised. Any finding that "TimesFM does better or worse than Ridge" should acknowledge this asymmetry: Ridge was evaluated on 2025 in a prior analysis; TimesFM is being evaluated for the first time.

---

## 11. Criteria for Declaring One Method Better

A method is declared better than another if and only if:
1. Its MAE is lower on the **primary evaluation set** (n=66), AND
2. Its MAE is lower on the **clean evaluation set** (n=60, no quality-flagged), AND
3. The direction is consistent across at least 3 of the 4 n_context strata, OR  
   the method wins on the dominant stratum (n_context=6 if it is the largest, as in 2022).

If MAE rankings conflict across primary and clean sets, both results are reported without a winner declaration.

**Effect size:** A difference of <1.0 MAE score points is considered negligible (based on the 2022 development finding that TFM and persistence differed by 1.35 points). Differences of 1–3 points are moderate; >3 points are substantial.

**No p-value threshold is used** for declaring a winner on this single evaluation.

---

## 12. Execution Prerequisite Checklist

Before running the evaluation, confirm all of the following:

- [ ] SAV path bug fixed in `build_country_estimates.py` (or confirmed not needed if existing estimates are used)
- [ ] Existing PISA 2025 estimates confirmed sufficient (SE-validated; no re-run needed)
- [ ] Linear trend script adapted for 2025 target (TEST_CYCLE = 2025)
- [ ] TimesFM script adapted for 2025 target (TEST_CYCLE = 2025)
- [ ] 66-country eligible set written to a lock file before execution
- [ ] Positive-bias claim correction documented
- [ ] This protocol reviewed and approved by user
- [ ] No PISA 2025 TimesFM outputs exist before execution (confirmed: None found)
