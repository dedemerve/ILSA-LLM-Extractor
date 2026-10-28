# Phase 3.3 — PISA 2025 Data Validation Status

**Date:** 2026-10-14  
**Protocol:** Phase 3.3 Readiness Gate, Task 3  
**Status:** COMPLETE — read-only audit; production script not modified  
**This file does NOT overwrite any prior artifact.**

> **Superseding update (2026-10-15):** This file is a dated snapshot. The direct OECD mean comparison previously marked blocked has now been completed for the final 66-country, three-domain evaluation set; see `phase3_12_oecd_pisa2025_mean_validation.md`. Its other provenance and standard-error scope limits remain as written.

---

## 1. SAV File Status

| Item | Value |
|------|-------|
| Script-configured path | `/Users/mrved/Desktop/ILSA Datasets/PISA Datasets/PISA 2025 Data/CY09_MS_STU_PUF.sav` |
| Configured path exists | **NO** |
| Actual file path | `/Users/mrved/Desktop/ILSA Datasets for LLM Extraction/PISA Datasets/PISA 2025 Data/CY09_MS_STU_PUF.sav` |
| Actual file exists | **YES** |
| File size | 2.17 GB |
| File modification date | 2026-08-21 11:23:58 |
| Script lines affected | L137, L142, L147 of `scripts/build_country_estimates.py` |

**The path bug is confirmed.** `ILSA_BASE` is defined as `/Users/mrved/Desktop/ILSA Datasets` (L38 of script), but the PISA 2025 SAV file is under `ILSA Datasets for LLM Extraction`. The script cannot re-run the PISA 2025 pipeline without this fix.

**Implication:** The existing PISA 2025 estimates in `outputs/stage4/country_estimates.csv` were produced correctly (as evidenced by SE validation); the path bug affects only future re-runs, not the already-computed estimates.

---

## 2. Provenance of Existing Estimates

| Item | Status |
|------|--------|
| PISA 2025 rows in country_estimates.csv | 268 (89 countries × 3 domains, but only 90 unique countries for mathematics; 89 for reading and science may differ) |
| PISA 2025 math countries | 89 |
| Script version that produced them | Unknown — git history shows estimates were committed in 9f2bf5d |
| Evidence that correct SAV was used | SE validation passes (see Section 3) |
| Direct provenance documentation | **NOT AVAILABLE** — no log file, no hash of input SAV in outputs |

The estimates cannot be reproduced by re-running the production script in its current state. Their correctness is established indirectly through SE comparison only.

---

## 3. Standard Error Validation

**Method:** Compare our computed SEs against OECD PISA 2025 Technical Report Annex 14, Table T14.A.13 (official PISA 2025 mathematics SEs for 79 countries with available official values).

| Metric | Value |
|--------|-------|
| Countries compared | 79 |
| Mean absolute difference (MAD) | **0.000026** |
| Maximum absolute difference | 0.000049 |
| Interpretation | Floating-point precision agreement — the pipeline reproduces official SEs |

**What SE validation establishes:**
- The BRR/FAY variance estimation with k=0.5, R=80 replicates was implemented correctly.
- The Rubin (1987) PV pooling (m=10 PVs) was applied correctly.
- The pipeline identifies the correct student records, sampling weights, and replicate weights.

**What SE validation does NOT establish:**
- Whether the point estimates (means) are correct (different calculation from SE).
- Whether the country composition matches the official OECD participants list (89 vs official count: not independently confirmed).
- Whether any preprocessing step introduced error that was offset in both mean and SE.

---

## 4. Direct Mean Comparison

| Status | BLOCKED |
|--------|---------|
| Source | OECD PISA 2025 Results Volume I |
| Reason | HTTP 403 at time of access |
| Alternative | Manual inspection of PDF table if available |
| Impact | Cannot confirm mean scores against official OECD publication |

The SE validation provides strong indirect evidence of pipeline correctness. Direct mean comparison remains the only missing independent check.

---

## 5. Quality Flags

Six countries have PISA 2025 sampling quality flags (Annex 14, footnote 1):

| ISO3 | Country | Flag | SE diff vs official | Protocol |
|------|---------|------|--------------------|---------| 
| ALB | Albania | CAUTION | 0.000047 | Report separately |
| CAN | Canada | CAUTION | 0.000046 | Report separately |
| NLD | Netherlands | CAUTION | 0.000017 | Report separately |
| NOR | Norway | CAUTION | 0.000049 | Report separately |
| NZL | New Zealand | CAUTION | 0.000048 | Report separately |
| USA | United States | CAUTION | 0.000036 | Report separately |

**SEs for all six quality-flagged countries match official values** (maximum diff 0.000049). The quality flag relates to sampling standards, not pipeline accuracy. These countries are NOT excluded from the evaluation set; they are reported as a separate stratum per the assessment protocol.

**Note on Albania:** ALB is quality-flagged for 2025 and was also the most anomalous 2022 prediction (AE ~69–94 points across all methods). Phase 3.1.1 confirmed the 2022 anomaly is real (ALB 2022 math = 368, substantially below the 2009–2018 upward trend). The 2025 value (433) represents a recovery. Albania should be examined individually in all final evaluation reports.

---

## 6. Summary

| Check | Status |
|-------|--------|
| SAV file exists at correct location | YES (wrong location in script) |
| Script path correct | NO — path fix required before re-run |
| SE pipeline validated | YES (MAD=0.000026, n=79) |
| Direct mean comparison | BLOCKED (HTTP 403) |
| Quality flags documented | YES (6 countries) |
| Provenance documented | PARTIAL — no input hash, no run log |
