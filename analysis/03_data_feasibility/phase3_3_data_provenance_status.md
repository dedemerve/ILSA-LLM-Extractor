# Phase 3.3 — PISA 2025 Data Provenance Status

> **Superseding update (2026-10-15):** The point-mean check marked blocked below was completed for the 66-country three-domain retrospective evaluation sample against OECD Volume I StatLink. See `phase3_12_oecd_pisa2025_mean_validation.md`. The historical file-provenance, reproducibility, and standard-error limitations below remain separate and are not erased by that mean comparison. The statement below that TimesFM had not yet been run on PISA 2025 is also historical; see `phase3_10_pisa2025_retrospective_evaluation.md`.

**Date:** 2026-10-14  
**Protocol:** Phase 3.3 Final Evaluation Protocol Lock, Task 2  
**Scope:** PISA 2025 country-level mathematics estimates only  
**This file does NOT overwrite any prior artifact.**

---

## A. Input File Identity and Hash

| Item | Value |
|------|-------|
| Filename | `CY09_MS_STU_PUF.sav` |
| Expected path (per script) | `/Users/mrved/Desktop/ILSA Datasets/PISA Datasets/PISA 2025 Data/CY09_MS_STU_PUF.sav` |
| Actual path | `/Users/mrved/Desktop/ILSA Datasets for LLM Extraction/PISA Datasets/PISA 2025 Data/CY09_MS_STU_PUF.sav` |
| File exists at actual path | **YES** |
| File size | 2,168,666,883 bytes (2.169 GB) |
| File modification date | 2026-08-21 11:23:58 |
| SHA-256 hash | `6f6fa1e1ecce23d39183d44ec82ba05b245fe88a2416c8d8dc1b016d556fc379` |
| Source | OECD PISA 2025 Public Use File (CY09_MS_STU_PUF = Cycle 9, Main Study, Student, Public Use File) |

**Path bug note:** The script `build_country_estimates.py` defines `ILSA_BASE = Path("/Users/mrved/Desktop/ILSA Datasets")` (L38). Git history shows this was already the value in commit `3a84843` (feat: PISA 2025 entegrasyonu). The path was therefore incorrect at the time of the original run. How the estimates were nevertheless produced is **unknown** — possible explanations: (a) the directory was renamed after the run; (b) a different script invocation was used; (c) ILSA_BASE was temporarily changed. No run log or shell history is available to confirm.

---

## B. Country-Estimate Generation Script

| Item | Value |
|------|-------|
| Script | `scripts/build_country_estimates.py` |
| Git commit of current version | HEAD (modified; unstaged changes per `git status`) |
| Commit that added PISA 2025 | `3a84843` ("feat: PISA 2025 entegrasyonu ve tam LOCO backtesting") |
| Commit that last touched estimates file | `9f2bf5d` ("data: integrate PISA 2018 and update Phase 3 feasibility audit") |
| Script version used for existing estimates | **UNKNOWN** — no hash of script recorded at run time |

---

## C. Weighting and Variance Estimation Method

**Known (from script inspection):**

| Item | Value |
|------|-------|
| Student weight | `W_FSTUWT` |
| Replicate weights | `W_FSTURWT001` – `W_FSTURWT080` (R=80) |
| Variance method | BRR/FAY, k=0.5 |
| Denominator | R/2 = 20 (per FAY BRR formula) |
| PV columns | `PV1MATH` – `PV10MATH` (m=10 plausible values) |
| PV pooling | Rubin (1987): T = Ū + (1+1/m)·B |

---

## D. Country Eligibility Rules

**Known:**
- All countries with valid `W_FSTUWT` and at least one PV column in the SAV file are included.
- No minimum student count filter is applied beyond what the OECD PUF already provides.
- Result: 89 unique countries for PISA 2025 mathematics.

**Unknown:** whether any outlier-detection or exclusion rule was applied beyond standard PUF coverage.

---

## E. Quality Flags

Six countries have PISA 2025 Annex 14 footnote 1 caution flags: ALB, CAN, NLD, NOR, NZL, USA. Confirmed via SE comparison (`pisa2025_se_validation.csv`). Protocol: report as separate stratum, not excluded.

---

## F. SE Validation Evidence

| Item | Value |
|------|-------|
| Countries compared | 79 |
| Comparison source | OECD PISA 2025 Technical Report Annex 14, Table T14.A.13 |
| MAD (SE difference) | 0.000026 |
| Max abs difference | 0.000049 |
| Conclusion | Pipeline SE is correct to floating-point precision |

**What SE validation establishes:** The BRR/FAY and Rubin pooling were implemented correctly; the correct SAV columns were used; weights and PVs are being read correctly.  
**What it does NOT establish:** Whether point estimates (means) are correct. SE and mean are computed by different code paths; a bug can affect one without the other.

---

## G. Direct Official-Mean Validation

| Item | Value |
|------|-------|
| Source | OECD PISA 2025 Results Volume I |
| Access status | **BLOCKED** (HTTP 403 at time of audit) |
| Consequence | Cannot independently confirm country means against official OECD publication |
| Alternative | OECD Education at a Glance 2026 may carry aggregate figures; individual country data tables in PISA 2025 summary PDFs (not verified) |

---

## H. PISA 2025 Exposure in Prior Analyses

**CONFIRMED: PISA 2025 outcomes appeared in the LOCO backtesting pipeline.**

| Analysis | File | Exposure |
|----------|------|---------|
| LOCO test set (test_cycle=2025) | `outputs/stage5/loco_results.csv` | Aggregate MAE/RMSE/R² for M0, M0A, M1, M2, M3 |
| LOCO predictions for 2025 | `outputs/stage5/loco_predictions.csv` | Row-level: y_true, y_M0, y_M1, y_M2, residuals (n=75–76 per domain) |

**What these analyses are:** M0, M0A, M1 = RidgeCV regression; M2 = persistence; M3 = AR(1). These are entirely separate from TimesFM. The LOCO framework holds out cycle T (here, 2025) and trains on all other cycles — it is a valid retrospective evaluation design, not a training exposure.

**TimesFM status:** `scripts/step2_timesfm_zeroshot.py` has `TEST_CYCLE = 2022`. TimesFM was run ONLY on PISA 2022. No TimesFM output for PISA 2025 exists anywhere in the project. **TimesFM has not seen PISA 2025 outcomes.**

**Consequence for interpretation:** PISA 2025 is not a "project-blind" holdout — the Ridge/persistence LOCO results (LOCO M2 persistence MAE ≈ 11.2, M0/M1 Ridge ≈ 12.1 on n=75) are available in the output files and may have influenced earlier design decisions. Specifically:
1. The LOCO evaluation was run before the TimesFM evaluation was designed.
2. It is not possible to confirm post hoc that the TimesFM evaluation protocol (MIN_CONTEXT=3, choice of baselines, country-set definition) was not influenced by prior inspection of LOCO 2025 results.
3. A future TimesFM evaluation report should explicitly acknowledge that PISA 2025 was used as a LOCO test set in a separate prior analysis, and that TimesFM specifically has not been evaluated on 2025 data.

---

## Provenance Summary Table

| Provenance Item | Status |
|----------------|--------|
| Input SAV identity (filename + hash) | KNOWN: SHA-256 = 6f6fa...c379 |
| Script version used for existing estimates | UNKNOWN — no run log |
| Command or procedure used | UNKNOWN — no run log |
| Weighting method | KNOWN (from script inspection) |
| PV pooling | KNOWN (from script inspection) |
| Country eligibility rules | KNOWN |
| Quality flags | KNOWN (6 countries) |
| SE validation | PASS (MAD=0.000026) |
| Point-estimate validation | BLOCKED (HTTP 403) |
| Prior PISA 2025 exposure | CONFIRMED (LOCO; NOT TimesFM) |
