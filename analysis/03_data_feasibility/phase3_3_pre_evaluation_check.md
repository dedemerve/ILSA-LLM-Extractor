# Phase 3.3 — Pre-Evaluation Check

**Date:** 2026-10-14  
**Preceding documents inspected:**  
- `phase3_3_final_evaluation_protocol.md`  
- `phase3_3_data_provenance_status.md`  
- `phase3_3_final_readiness_gate.md`  
**This file does NOT overwrite any prior artifact.**

---

## 1. Protocol Internal Consistency Audit

### 1.1 Target Variable and Domain Name

| Item | Protocol states | Actual in `country_estimates.csv` | Status |
|------|----------------|----------------------------------|--------|
| Domain column value | "mathematics" | `mathematics` | **CONSISTENT** |
| Script constant `DOMAIN` in `step2_timesfm_zeroshot.py` (L38) | — | `DOMAIN = "mathematics"` | **CONSISTENT** |

No inconsistency. Earlier audit notes mentioning "math" were informal shorthand; the actual data and script both use `mathematics`.

### 1.2 Country Eligibility Counts

| Item | Protocol states | Independently verified | Status |
|------|----------------|----------------------|--------|
| PISA 2025 math estimates in `country_estimates.csv` | 89 | **89** (domain=`mathematics`) | CONSISTENT |
| Eligible (≥3 pre-2025 cycles, has 2025 estimate) | 66 | **66** from eligibility CSV | CONSISTENT |
| Quality-flagged stratum | 6 (ALB/CAN/NLD/NOR/NZL/USA) | **6** (same list) | CONSISTENT |
| Clean stratum | 60 | **60** | CONSISTENT |
| Cross-2022 comparison set | 60 | **60** | CONSISTENT |

### 1.3 Quality-Flag Rule

Protocol: include in primary (n=66), report as separate stratum, highlight Albania individually.  
Eligibility file: all 6 flagged countries are present in the eligible-66 set.  
**CONSISTENT.**

### 1.4 Missing-Data Rule

Protocol: persistence uses last non-missing pre-2025 cycle; linear trend uses all available pre-2025 observations; MIN_CONTEXT=3.  
Script `step2_timesfm_zeroshot.py` L40: `MIN_CONTEXT = 3`.  
For countries without 2022 data, last non-missing cycle would be earlier; no such country is in the eligible-66 (all 66 have both 2022 and 2025 estimates per the eligibility file).  
**CONSISTENT.**

### 1.5 Error Sign Convention

Protocol (Section 6): "positive = overprediction (predicted − actual); compute from raw prediction columns, not stored error columns."  
The prior sign-convention bug in `outputs/step2/zeroshot_results.csv` (stored absolute errors, not signed) was identified in the 2022 development results but does NOT affect the 2025 evaluation because:
(a) the 2025 evaluation will use raw predictions minus actual values, per the protocol;  
(b) the 2022 stored-error bug is in the development set only and has been documented.  
**CONSISTENT** — provided the 2025 evaluation follows the protocol instruction.

### 1.6 Comparison and Winner Criteria

Protocol: winner declared only if lower MAE on both primary (n=66) and clean (n=60) sets, consistent across ≥3 of 4 n_context strata. Effect size threshold: <1.0 point = negligible.  
No contradiction found in the protocol document.  
**CONSISTENT.**

### 1.7 LOCO 2025 Disclosure

Protocol Section 10 states: "PISA 2025 was used as a LOCO test set for Ridge/persistence (M0, M0A, M1, M2, M3). TimesFM has not been evaluated on 2025."  
Independently confirmed: `outputs/stage4/loco_results.csv` contains test_cycle=2025 rows for M0, M1, M2 with MAE = 9.68 (Ridge), 11.15 (persistence); `outputs/stage4/loco_predictions.csv` contains row-level y_true for 75 countries.  
**CONSISTENT and CONFIRMED.**

---

## 2. New Finding — LOCO Country Set vs. Evaluation Country Set

**Finding:** The LOCO 2025 analysis used n_test=75 countries, not 66. The 10 extra countries (BRN, GTM, KHM, MAR, MNG, PHL, PRY, PSE, SAU, SLV) all have n_pre2025 < 3 (range: 1–2) and are therefore ineligible under the protocol's MIN_CONTEXT=3 rule. Additionally, LUX is in the eligible-66 set but was **absent** from the LOCO 2025 predictions.

| Item | Value |
|------|-------|
| LOCO 2025 n_test | 75 |
| Protocol evaluation n | 66 |
| Countries in LOCO but below MIN_CONTEXT | 10 (n_pre2025 = 1 or 2) |
| Country in eligible-66 but absent from LOCO | LUX (n_pre2025=6) |

**Impact on evaluation:** None. LOCO used a different eligibility rule (MIN_CONTEXT not applied consistently). The final TimesFM evaluation will use the protocol-defined 66-country set. This discrepancy is a LOCO implementation quirk, not a protocol inconsistency. It must be noted in any report comparing LOCO Ridge performance to TimesFM performance.

---

## 3. New Finding — n_pv Column Empty for All PISA 2025 Estimates

**Finding:** All 89 PISA 2025 entries in `country_estimates.csv` have `n_pv = ""` (empty). The same is true for all PISA 2022 (n=80) and 2018 (n=79) entries. The Rubin (1987) pooling intermediate columns (T, U_bar, B) are also empty for PISA 2025.

**Interpretation:** The n_pv column is not populated by the script for any PISA cycle, not only 2025. This is consistent with a single-pass pipeline that computes the Rubin-pooled final mean and SE without storing intermediates. The SEs are verified correct (MAD=0.000026 vs Annex 14). The empty n_pv column therefore does NOT indicate that PV pooling was skipped; it indicates that the intermediate was not written to output.

**Status:** Not a protocol violation. Documented for completeness.

---

## 4. Alternative Sources for Official Mean Validation (Task 3)

The protocol and provenance report document that direct comparison against OECD PISA 2025 Results Volume I was blocked (HTTP 403). This section reports all alternative local and accessible sources investigated.

### 4.1 SE Validation File (`audit/pisa2025_se_validation.csv`)

Columns: iso3, country_name, **our_mean**, our_se, **official_se**, se_diff, se_abs_diff, n_official, quality_flag.

The file stores `our_mean` and `official_se` but **no `official_mean` column**. The official SE was extracted from OECD Annex 14 Table T14.A.13 for SE comparison purposes only. Official means were not recorded.

**Conclusion:** This file cannot be used for mean validation.

### 4.2 Eligibility File (`audit/pisa_2022_2025_country_eligibility.csv`)

Contains `mean_2025` and `se_2025` columns, but these are sourced from `country_estimates.csv` (our own pipeline). They are not independent.

**Conclusion:** Not an independent source.

### 4.3 `outputs/stage4/loco_predictions.csv`

Contains `y_true` for 75 countries at test_cycle=2025. These values are sourced from `country_estimates.csv` (used as the truth vector in LOCO). They are not independent.

**Conclusion:** Not an independent source.

### 4.4 No Other Local Official Source Found

A search of the project filesystem found no locally stored OECD official country mean table, no PISA 2025 summary PDF, no `education_at_a_glance_2026.csv`, and no OECD data extract. The `ilsa_official_variable_registry.json` and `semantic_knowledge_base_v4.json` files contain variable metadata, not country-level scores.

**Conclusion:** No locally available official source can validate point estimates. The HTTP 403 barrier documented in the provenance report remains the only known obstacle. This limitation must appear in the evaluation report.

---

## 5. Provenance Gaps — Definitive List

The following gaps cannot be closed without additional evidence:

| Gap | Type | Severity | Can it block evaluation? |
|-----|------|----------|-------------------------|
| **G1: Script version used for existing estimates** | Provenance | Moderate | No — SE match provides indirect validation |
| **G2: Which code path produced the estimates** (path bug present at commit time) | Provenance | Moderate | No — same indirect validation argument |
| **G3: Official country means not validated** (HTTP 403, no local alternative) | Validation | Serious | No — SE validation is a strong proxy; means are plausible; see note below |
| **G4: 10 of 89 countries not covered by SE validation** (CZE, HKG, MAC, PSE, QAT, QCI, QKI, QTJ, QUA, TUR) | Validation | Low–Moderate | No — 7 of 10 are ineligible (n_pre2025 < 3); 3 are in eligible-66 |
| **G5: n_pv column empty** | Metadata | Low | No — artefact of output schema, not of computation |
| **G6: LUX absent from LOCO** | LOCO implementation | Low | No — affects only LOCO-to-TimesFM comparison note |

**Note on G3 severity:** SE validation is not mean validation (the protocol correctly separates them). However, the SE is a function of the same replicate weights and PV values that produce the mean. A pipeline that correctly replicates 79 SEs to within 5×10⁻⁵ is almost certain to have produced correct means from the same inputs. This does not formally prove mean correctness, but it substantially reduces the probability of a systematic mean error. The gap is **serious but not fatal**.

**G4 detail:** Of the 10 countries not in the SE validation:
- PSE is in eligible-66 (n_pre2025=6 per eligibility file).
- QCI, QKI, QTJ, QUA appear in country_estimates but not in the eligibility file's 66 set — they are sub-national units or adjudicated regions with n_pre2025 < 3.
- CZE, HKG, MAC, QAT, TUR are not in eligible-66.
- Only PSE contributes to the primary evaluation without independent SE confirmation.

---

## 6. Summary of Protocol Consistency

| Check | Result |
|-------|--------|
| Domain name (`mathematics`) | CONSISTENT |
| Eligible n=66 | CONFIRMED |
| Quality-flag stratum (6 countries) | CONFIRMED |
| Clean stratum n=60 | CONFIRMED |
| MIN_CONTEXT=3 | CONSISTENT with script |
| Error sign convention rule | CORRECTLY STATED (avoid stored-error columns) |
| Winner criteria | INTERNALLY CONSISTENT |
| LOCO 2025 disclosure | CONFIRMED AND PRESENT |
| LOCO n_test=75 ≠ evaluation n=66 | DOCUMENTED (new finding; no protocol conflict) |
| n_pv empty | DOCUMENTED (pipeline artefact; not a gap) |

---

## 7. Decision

### Evidence supporting readiness

1. **Estimates are SE-validated to floating-point precision** (79/89 countries; MAD=0.000026).
2. **All eligibility counts verified** (n=66, n=60, 6 quality flags, 60 cross-2022).
3. **Protocol is internally consistent** across domain, eligibility, error convention, and winner criteria.
4. **LOCO 2025 disclosure is present and correct** in the protocol.
5. **No local authoritative mean source exists** — this is a ceiling on achievable validation, not a gap that can be closed before evaluation.
6. **The 2022 development experiment** used the same pipeline and was the basis for the protocol's design; it provided a validated reference point for assessing TimesFM performance in this context.

### Evidence qualifying readiness

1. **G3 (official means not validated):** Cannot be confirmed against OECD publication. Probability of systematic error is low given SE match, but not formally excluded.
2. **G1/G2 (script provenance):** Unclear which code path produced the estimates. The SE match provides an indirect consistency check.
3. **PSE SE unvalidated:** One eligible country (PSE) is not covered by the SE validation.
4. **LOCO n_test ≠ evaluation n:** LOCO results (MAE=9.68 Ridge, 11.15 persistence on n=75) include 10 ineligible countries and exclude LUX; they should not be directly compared to the final TimesFM result on n=66 without adjustment.

### Conclusion

The project cannot achieve full point-estimate validation before evaluation — no local official source exists and the web source is blocked. Waiting for that validation indefinitely is not a viable path. The SE validation is the strongest available proxy and passes. All protocol parameters are consistent and internally coherent. The critical LOCO 2025 exposure is disclosed.

---

## **READY FOR PROVISIONAL EVALUATION ONLY**

**Rationale:** The evaluation may proceed, but the report must carry the following mandatory caveats, which cannot be removed before the evaluation runs:

1. Country-level point estimates are SE-validated (MAD=0.000026, n=79) but not independently validated against OECD Results Volume I. Mean accuracy is inferred from SE accuracy, not directly confirmed.
2. Script provenance for the existing estimates is incomplete (path bug documented; run log unavailable).
3. PISA 2025 is not a project-blind holdout for the Ridge/persistence methods (LOCO results available). TimesFM specifically is prospective.
4. LOCO MAE figures (Ridge: 9.68, Persistence: 11.15 on n=75) are not directly comparable to the final evaluation on n=66 without adjustment for the different country sets.

**The label "PROVISIONAL" may be removed** if, before or after evaluation, the OECD Results Volume I becomes accessible and direct mean comparison confirms the existing estimates. No re-run is needed; only verification.

**Execution prerequisite:** Explicit user authorisation to run `step2_timesfm_zeroshot.py` with TEST_CYCLE=2025 (and confirmation of whether to use existing estimates as-is or fix the path bug and re-run first).
