# Phase 3.3 — Readiness Gate

**Date:** 2026-10-14  
**Protocol:** Phase 3.3 Readiness Gate, Task 5  
**Status:** FINAL ASSESSMENT  
**This file does NOT overwrite any prior artifact.**

---

## Readiness Decisions

### A. Literature Synthesis

**Decision: READY**

The 1,266-study corpus has been classified, audited, and synthesised. The primary task-code distribution is verified (G=1,100; A=96; F=45; D=17; B=8). The claim of 0 confirmed genuine future-cycle forecasting studies is accurate with a required caveat (27 of 28 cross-cycle studies unresolved at source). The classification count error (1,294 vs 1,266) is documented. Meta-analysis is not justified (incompatible metrics, targets, units — confirmed). No further work is required to proceed.

**Residual task:** Source-inspection of remaining 27 cross-cycle studies — does not block any subsequent phase.

---

### B. PISA 2022 Forecasting Diagnostics

**Decision: CONDITIONALLY READY**

All MAEs and ranking are independently verified. The three baselines use identical country sets (n=60), identical target variables, and identical error definitions. The comparison is fair.

**Conditions:**
1. Correct the positive-bias claim ("56/60" and "55/60", not "all 60") in any document carried forward into Phase 3.3 or beyond.
2. Label the trend-slope–TFM correlation (r=0.497) as **exploratory** in all future reports. The Pearson correlation is reproduced but is not robust (Spearman r=0.18, p=0.17; driven by Albania).
3. All other claims are either verified or carry clearly documented limitations.

**Does not block Phase 3.3:** The corrections are to assertion wording, not to the underlying experiment.

---

### C. PISA 2025 Country-Estimate Validity

**Decision: CONDITIONALLY READY**

The existing estimates in `country_estimates.csv` are SE-validated (MAD=0.000026, n=79 countries). The six quality-flagged countries (ALB, CAN, NLD, NOR, NZL, USA) are documented. The estimates were produced from the correct SAV file (evidenced by SE match); the path bug affects only future re-runs.

**Conditions:**
1. Use the existing estimates as-is for evaluation. Do not re-run the pipeline without fixing the path first.
2. Quality-flagged countries must be reported as a separate stratum.
3. Direct mean comparison against OECD Results Volume I remains pending (HTTP 403). This is a desirable verification step but does not block evaluation of the existing estimates.

---

### D. PISA 2025 Final Evaluation (held-out test)

**Decision: CONDITIONALLY READY**

The evaluation may proceed once the following are explicitly documented:

| Requirement | Status |
|------------|--------|
| **Country set defined** | 66 eligible (≥3 pre-2025 cycles); 60 clean; 6 quality-flagged separately. Confirmed. |
| **Target estimates** | PISA 2025 math means from country_estimates.csv; SE-validated. Confirmed. |
| **Evaluation protocol** | Use same protocol as 2022 development: MAE, mean signed error, context-length stratification, quality-flag stratum. DEFINED. |
| **Leakage caveat documented** | TimesFM was zero-shot; no fine-tuning on any PISA data including 2025. No 2025 outcomes were used in model selection or hyperparameter tuning. CONFIRMED. |
| **Positive-bias claim corrected** | 56/60 and 55/60 (not "all 60"). REQUIRES CORRECTION before evaluation report is written. |
| **Trend-slope correlation labelled exploratory** | REQUIRES ACTION before final report. |
| **Path bug fixed if re-run needed** | Requires explicit user authorization. Fix proposal in `audit/phase3_3_path_patch_proposal.txt`. |

**The evaluation itself (running `step2_timesfm_zeroshot.py` on 2025 data) is NOT authorised in this task.** It must be authorised explicitly.

---

### E. Future Model Fine-Tuning

**Decision: NOT READY**

Fine-tuning is not authorised in this task and must not proceed until:
1. The PISA 2025 held-out evaluation is complete and results are reviewed.
2. The gold-standard dataset (v1.4.1) is re-confirmed as the correct base.
3. An explicit fine-tuning protocol is approved.
4. No PISA 2025 outcomes may influence fine-tuning design.

---

## Summary

| Component | Decision | Blocking conditions | Can proceed without auth? |
|-----------|----------|--------------------|--------------------------| 
| A. Literature synthesis | **READY** | None | Yes |
| B. 2022 diagnostics | **CONDITIONALLY READY** | Correct 3 claims in forward-facing documents | Yes — corrections to wording only |
| C. 2025 estimates | **CONDITIONALLY READY** | Use existing; fix path before re-run | Yes — existing estimates usable |
| D. 2025 final evaluation | **CONDITIONALLY READY** | Document 5 items; do not run before authorisation | **No — requires explicit authorisation** |
| E. Fine-tuning | **NOT READY** | Multiple conditions; blocked | No |

---

## Next Action Requiring Explicit User Authorisation

**PISA 2025 final evaluation:**  
Authorise the execution of the held-out evaluation (`step2_timesfm_zeroshot.py` or equivalent) on PISA 2025 data. Before execution, confirm: (1) path patch applied; (2) claim corrections incorporated into evaluation design document; (3) evaluation protocol matches the 2022 development protocol.

**SAV path fix:**  
Authorise applying Option A or Option B from `audit/phase3_3_path_patch_proposal.txt` to `scripts/build_country_estimates.py`.
