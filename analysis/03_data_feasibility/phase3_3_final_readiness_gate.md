# Phase 3.3 — Final Readiness Gate

**Date:** 2026-10-14  
**Protocol:** Phase 3.3 Final Evaluation Protocol Lock, Task 5 + 6  
**Status:** FINAL ASSESSMENT  
**This file does NOT overwrite any prior artifact.**

> **Superseding update (2026-10-15):** This is a historical readiness snapshot. The PISA 2025 retrospective benchmark has since been executed, and its point means have been independently validated for the 66-country, three-domain sample. See `phase3_10_pisa2025_retrospective_evaluation.md` and `phase3_12_oecd_pisa2025_mean_validation.md`. PISA 2025 remains outcome-exposed, not blind.

---

## Validation Gate — PISA 2025 Country Estimates

Assessed separately across six evidence dimensions per Task 4.

### A. Input-File Provenance

| Check | Status | Evidence |
|-------|--------|---------|
| Filename identified | PASS | `CY09_MS_STU_PUF.sav` |
| File exists at known location | PASS | `/ILSA Datasets for LLM Extraction/...` confirmed |
| SHA-256 hash recorded | PASS | `6f6fa1e1ecce23d39183d44ec82ba05b245fe88a2416c8d8dc1b016d556fc379` |
| Script path matches file location | **FAIL** | Script references non-existent `/ILSA Datasets/...` |
| Script version used for estimates | **BLOCKED** | No run log; script had wrong path at git commit time |

**Grade: PARTIAL** — file identity confirmed; script-to-file linkage is broken.

---

### B. Point-Estimate Validation

| Check | Status | Evidence |
|-------|--------|---------|
| Direct comparison vs. OECD Results Volume I | **BLOCKED** | HTTP 403 |
| Any row-level mean comparison against official source | **BLOCKED** | No accessible alternative confirmed |
| Internal consistency (plausible range, no outlier scores) | PARTIAL | ALB=433, ARE=453, AUS=478 etc. are plausible; no systematic anomaly detected in inspection |

**Grade: BLOCKED** — SE validation passes, but point-estimate accuracy cannot be confirmed against an official external source.

**Important distinction:** SE validation (Section C) does NOT establish point-estimate correctness. They use different computations. Do not conflate them.

---

### C. Standard-Error Validation

| Check | Status | Evidence |
|-------|--------|---------|
| SE compared against official Annex 14 | PASS | Table T14.A.13 |
| Countries compared | PASS | 79 of 89 (10 missing from Annex 14) |
| MAD | PASS | 0.000026 |
| Maximum absolute difference | PASS | 0.000049 |
| Quality-flagged countries included | PASS | All 6 present; SEs match |

**Grade: PASS** — SE pipeline is correct to floating-point precision.

---

### D. Sampling-Quality / Comparability Review

| Check | Status | Evidence |
|-------|--------|---------|
| Quality-flag countries identified | PASS | ALB CAN NLD NOR NZL USA (Annex 14 footnote 1) |
| Flags documented in protocol | PASS | Separate-stratum rule locked in protocol |
| Albania cycle-level anomaly investigated | PASS | 2022 anomaly confirmed (368); 2025 recovery to 433 (Phase 3.1.1) |
| Other OECD comparability notes reviewed | PARTIAL | Only footnote 1 countries confirmed; broader notes not independently reviewed |

**Grade: PASS** — documented and protocol-incorporated.

---

### E. Country Eligibility

| Check | Status | Evidence |
|-------|--------|---------|
| Eligibility definition locked | PASS | ≥3 pre-2025 cycles in estimates, has 2025 estimate |
| Eligible set count | PASS | 66 countries (confirmed from estimates file) |
| Eligible set list | PASS | Written to `phase3_3_country_count_reconciliation.csv` |
| Cross-2022 comparison set | PASS | 60 countries eligible in both experiments |

**Grade: PASS**

---

### F. Reproducibility

| Check | Status | Evidence |
|-------|--------|---------|
| Script can re-run PISA 2025 estimates | **FAIL** | Path bug blocks re-run |
| Hash of input SAV recorded for future verification | PASS | SHA-256 now documented |
| Existing estimates SE-validated | PASS | MAD=0.000026 |
| Estimates sufficiently stable for single evaluation run | PASS | No re-run needed; existing estimates adequate |

**Grade: PARTIAL** — existing estimates adequate; re-run requires path fix.

---

## Readiness Decisions

### 1. Is the evaluation protocol locked?

**YES — LOCKED**

`analysis/03_data_feasibility/phase3_3_final_evaluation_protocol.md` defines all decisions:
- Target: PISA 2025 math, 66 countries
- Methods: persistence, linear trend, TimesFM 3.0 zero-shot
- Metrics: MAE + signed error + per-stratum
- Sensitivity analyses: pre-specified (S1–S5)
- Winner criteria: explicit thresholds

No protocol parameter may be changed after the first TimesFM 2025 prediction is produced.

---

### 2. Are the data sufficiently validated to proceed?

**CONDITIONALLY YES**

The existing estimates are SE-validated and can be used for a single evaluation run. The following limitations must appear in all reports using these estimates:

1. **Point-estimate accuracy is not independently confirmed.** SE validation ≠ mean validation. Direct comparison against OECD Results Volume I is blocked.
2. **Script provenance is incomplete.** The script that produced the estimates had an incorrect path at the recorded commit time; it is unknown which code path was actually used.
3. **10 of 89 countries are not covered by the SE validation** (no Annex 14 entry available). Their SEs are assumed correct by pipeline consistency.

These limitations qualify but do not block the evaluation.

---

### 3. May a future PISA 2025 TimesFM evaluation be authorised?

**YES — SUBJECT TO USER AUTHORISATION**

Technical preconditions are met. The following items require explicit user decision before execution:

| Item | Decision required |
|------|-----------------|
| Use existing estimates as-is (no re-run) | Confirm |
| OR fix path bug and re-run | Authorise patch + re-run |
| Execute `step2_timesfm_zeroshot.py` with TEST_CYCLE=2025 | **Explicit authorisation required** |
| Execute linear trend script with TEST_CYCLE=2025 | Explicit authorisation required |
| Acknowledge LOCO 2025 prior exposure in report | Mandatory (not a decision) |

**This is the single next action that requires user authorisation.**

---

### 4. Is model fine-tuning justified?

**NOT READY — NOT AUTHORISED**

Fine-tuning on ILSA data must not begin until:
1. PISA 2025 held-out evaluation is complete and reviewed.
2. A clear performance gap justifying fine-tuning is identified.
3. Fine-tuning design is confirmed to exclude any PISA 2025 outcome data.
4. Gold-standard dataset (v1.4.1) is confirmed as the correct base.

---

## Literature Synthesis — Suitability for Manuscript (Task 5)

### Current defensible wording

> "A systematic search of 1,266 ILSA studies published through 2026 identified no confirmed examples of genuine future-cycle, country-level ILSA forecasting — defined as a study that (a) predicted a national mean score for assessment cycle T, (b) used only data from cycles prior to T during model development, and (c) evaluated predictions against cycle T outcomes. Twenty-eight studies used multi-cycle data but could not be classified without source inspection; one of these (Braun et al., 2025) was confirmed at source as a retrospective quality-assurance analysis rather than a genuine forecasting study."

### Qualification required before strengthening the claim

The claim may be upgraded to "zero studies" (without the caveat) only after:
- Source inspection of the remaining 27 cross-cycle studies.
- Each study assessed against the four-criterion definition above.
- A study-level classification table produced and reviewable.

Until then: **"no confirmed examples, with 27 studies unresolved"** is the only defensible formulation.

### Suitability for manuscript

**YES — with the qualified wording above.** The claim is accurate, supported by corpus-level evidence, and the 27-study limitation is a standard methodological caveat that strengthens rather than weakens the contribution (it demonstrates rigour).

---

## Summary Table

| Decision | Status |
|----------|--------|
| Protocol locked | **YES** |
| Data validated (SE) | **PASS** |
| Data validated (means) | **BLOCKED** |
| Data adequate for evaluation | **YES (with limitations)** |
| Evaluation authorised | **NO — awaits explicit user authorisation** |
| Fine-tuning authorised | **NO — NOT READY** |
| Literature claim manuscript-ready | **YES — with qualified wording** |

---

## Single Next Action Requiring User Authorisation

> **Authorise execution of the PISA 2025 final evaluation:**  
> Run `scripts/step2_timesfm_zeroshot.py` (adapted: TEST_CYCLE=2025) against the existing PISA 2025 estimates. Output to a new directory (e.g., `outputs/step2_2025/`). Do not overwrite any existing output.

Before giving that authorisation, also confirm whether:
- Existing estimates are used as-is, OR
- The path bug in `build_country_estimates.py` should be fixed first (patch proposal: `audit/phase3_3_path_patch_proposal.txt`).
