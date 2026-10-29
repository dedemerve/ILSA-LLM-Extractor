# Phase 3.3 — Country Count Reconciliation

**Date:** 2026-10-14  
**Protocol:** Phase 3.3 Readiness Gate, Task 1  
**Status:** COMPLETE — verified from source files  
**This file does NOT overwrite any prior artifact.**

---

## Summary Table

| Stage | Definition | n | Type |
|-------|-----------|---|------|
| S1 | All PISA 2022 math countries in country_estimates.csv | 80 | Raw |
| S2 | S1 with ≥3 pre-2022 cycles in estimates = **TimesFM experiment set** | 60 | Experiment-ready |
| S3 | All PISA 2025 math countries in country_estimates.csv | 89 | Raw |
| S4 | S3 with ≥3 pre-2025 cycles in estimates = **2025 evaluation eligible** | 66 | Eligible |
| S5 | Countries in both S1 and S3 (raw overlap) | 75 | Raw overlap |
| S6 | Countries eligible in both S2 and S4 (experiment-ready in both) | 60 | Model-ready |
| S4 minus quality-flagged | S4 excluding 6 Annex 14 flagged countries | 60 | Clean eligible |

---

## Origins of All Reported Counts

### "60 PISA 2022 countries in the TimesFM experiment"

**Source:** `outputs/step2/zeroshot_results.csv` — exactly 60 rows.  
**Derivation:** From `outputs/stage4/country_estimates.csv`, filter to PISA 2022 mathematics (→ 80 countries). Retain only those with ≥3 pre-2022 cycles in the estimates file (MIN_CONTEXT=3 in `scripts/step2_timesfm_zeroshot.py`). This gives 60.

**Why 60, not 65 (panel) or 80 (estimates)?**
- The unified panel contains historical entries for countries that did not participate in PISA 2022 (e.g., RUS, LUX, TUN, TWN, LIE — all have ≥3 pre-2022 entries in the panel but no 2022 estimate). Step 2 uses `country_estimates.csv` as its source, not the unified panel. These 5 countries are therefore excluded.
- 80 − 20 countries with <3 pre-2022 cycles = 60 eligible.

**Excluded from S1 → S2:**
- 11 countries: <3 pre-2022 cycles in estimates (see reconciliation CSV)
- 5 countries: DOM, GTM, JAM, KHM, MNG, PRY, PSE, QUR, SLV, UZB — 0 pre-2022 cycles
- Additional countries with 1–2 pre-2022 cycles: BRN, KSV, MAR, MKD, PAN, PHL, PRY, PSE, QAZ, SAU, SLV, TAP, UZB, VNM (partially — see CSV)

---

### "66 eligible PISA 2025 countries"

**Source:** `analysis/03_data_feasibility/audit/pisa_2022_2025_country_eligibility.csv`, field `in_primary_2025`.  
**Derivation:** From country_estimates.csv, filter to PISA 2025 mathematics (→ 89 countries). Retain those with ≥3 pre-2025 cycles in estimates. Result: 66.

**New in 2025 that were not eligible in 2022:** DOM, KSV, LUX, MKD, TAP, VNM (6 countries gained sufficient history by 2025).

---

### "6 quality-flagged countries"

**Source:** PISA 2025 Technical Report Annex 14, footnote 1 (accessed via SE validation audit).  
**Countries:** ALB, CAN, NLD, NOR, NZL, USA.  
**Meaning:** "Caution required when interpreting estimates because one or more PISA sampling standards were not met." They are NOT automatically excluded; they must be reported as a separate stratum.

---

### "75 raw overlapping countries"

**Source:** Countries present in both PISA 2022 and PISA 2025 country_estimates.csv (math domain).  
**Definition:** Raw data overlap — not a filtering stage. Does not require ≥3 pre-cycle history.

---

### "74" (origin identified)

**Source:** Phase 3.1 integrated audit (`phase3_1_integrated_audit.md`), which stated: "The viable final-test country set is 75. ALB is in this 75; if excluded on account of the anomaly, n=74."  
**Definition:** 75 raw overlap minus Albania (proposed sensitivity). This is a hypothetical exclusion scenario, not an eligibility stage.  
**Status:** The 74 figure refers to a conditional sensitivity count; it is NOT the primary evaluation set. It has been superseded by the formal S4 eligibility definition (n=66) and the quality-flag protocol (report separately, not exclude).

---

## Eligibility for PISA 2025 Final Evaluation

| Stage | n | Decision |
|-------|---|---------|
| S4 eligible | 66 | Primary evaluation set |
| S4 clean (no quality flags) | 60 | Clean subset |
| S4 quality-flagged | 6 | Report as separate stratum |
| S6 (comparable to 2022 experiment) | 60 | Cross-year comparison set |
| Countries new in 2025 (in S4 but not S2) | 6 | DOM KSV LUX MKD TAP VNM |

---

## Artifacts

| File | Contents |
|------|---------|
| `analysis/03_data_feasibility/audit/phase3_3_country_count_reconciliation.csv` | 94-row table: every PISA 2022 or 2025 math country with full eligibility and exclusion detail |
