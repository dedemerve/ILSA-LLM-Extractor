# Phase 3.3 — Claim Corrections

**Date:** 2026-10-14  
**Protocol:** Phase 3.3 Readiness Gate, Task 4  
**Status:** PROPOSED CORRECTIONS — prior reports not modified  
**This file does NOT overwrite any prior artifact.**

---

## Correction Table

Each claim is assessed along four dimensions:
- **Observed result** — what the data actually show
- **Interpretation** — what this result means
- **Causal explanation** — what is offered as a cause
- **Remaining uncertainty** — what is not established

---

### Claim 1 — Classification count

| Field | Content |
|-------|---------|
| Source report | phase3_2_task_classification.md (B.2 summary table) |
| Existing wording | Categories sum to 1,294 (G=1,100 + A=96 + F=45 + D=17 + B=8 + cross_cycle=28 = 1,294) |
| Evidence | study_classification.csv: primary_task_code value_counts() = G+A+F+D+B = 1,266 exactly |
| Corrected wording | "The five primary_task_code categories are mutually exclusive and exhaustive: G=1,100; A=96; F=45; D=17; B=8; total=1,266. The 28 cross_cycle_prediction_not_fully_verified studies are a subset of the temporal_forecasting_status dimension and overlap with existing primary codes (G=17, B=8, D=3 of the 28). They are NOT an additive sixth category." |
| Status | INCONSISTENT → REQUIRES CORRECTION |
| Observed result | Corpus n=1,266; five mutually exclusive codes sum to 1,266 |
| Interpretation | Classification is complete and consistent |
| Causal explanation | N/A |
| Remaining limitation | Cross-cycle studies are classified by primary code but their temporal design remains unverified at source for 27 of 28 |

---

### Claim 2 — Zero confirmed future-cycle forecasting studies

| Field | Content |
|-------|---------|
| Source report | phase3_2_task_classification.md, phase3_2_predictive_evidence.md |
| Existing wording | "0 confirmed genuine future-cycle ILSA forecasting studies" |
| Evidence | temporal_forecasting_status = genuine_future_cycle_confirmed: 0 records. Braun et al. 2025 (closest candidate) confirmed retrospective at source (Phase 2.2.1). |
| Corrected wording | "0 confirmed as of 2026-10-14. 28 studies are classified as cross_cycle_prediction_not_fully_verified and require source inspection to determine whether any meet the definition of genuine future-cycle forecasting. Only Braun et al. 2025 has been inspected at source; the remaining 27 are unresolved." |
| Status | PARTIALLY VERIFIED — correct but requires caveat |
| Observed result | 0 records with genuine_future_cycle_confirmed status in corpus |
| Interpretation | No published precedent for this exact forecasting task |
| Causal explanation | N/A |
| Remaining limitation | 27 of 28 cross-cycle studies not yet inspected at source |

---

### Claim 3 — Positive forecast bias across countries

| Field | Content |
|-------|---------|
| Source report | phase3_2_forecasting_diagnostics.md, phase3_2_hypotheses_and_tests.md |
| Existing wording | "All 60 countries were over-predicted by both methods. Mean signed error = MAE for both methods (virtually all overpredictions)." |
| Evidence | Recalculated signed errors: persistence − true_score < 0 for 4 countries (JPN KAZ KOR SGP); timesfm − true_score < 0 for 5 countries (AUS JPN KAZ SGP TUR). Mean signed errors: persistence = +13.885, TFM = +14.796. |
| Corrected wording | "56 of 60 countries were over-predicted by persistence; 55 of 60 by TimesFM. The mean signed errors are +13.885 (persistence) and +14.796 (TFM), which are not equal to the respective MAEs (14.482 and 15.830). The bias is overwhelmingly positive and consistent with systematic COVID-19 learning loss, but 4–5 countries were under-predicted." |
| Status | INCONSISTENT → REQUIRES CORRECTION |
| Observed result | 56/60 (persistence) and 55/60 (TFM) over-predicted; 4–5 under-predicted |
| Interpretation | Strong positive bias overall |
| Causal explanation | Proposed: COVID-19 learning loss (see Claim 4) |
| Remaining limitation | Countries under-predicted are not explained by the COVID narrative; individual-country factors not investigated |

---

### Claim 4 — COVID-19 as explanation for forecast bias

| Field | Content |
|-------|---------|
| Source report | phase3_2_forecasting_diagnostics.md, phase3_2_integrated_synthesis.md |
| Existing wording | "Consistent with COVID-19 learning loss — structurally unpredictable" |
| Evidence | (a) All 2022 targets use last_cycle=2018 context → 3-year gap spanning COVID. (b) Trend slope r(slope, AE_TFM) = 0.497 supports mechanism. (c) 56/60 over-prediction pattern is inconsistent with random noise. (d) No counter-evidence. |
| Corrected wording | "The positive prediction bias is consistent with COVID-19 learning loss in 2022. However, COVID-19 is the most plausible explanation, not a confirmed cause. The 4–5 under-predicted countries (countries that improved in 2022 despite COVID) are not explained by this narrative and require separate investigation. The relationship cannot be confirmed causally from this data." |
| Status | INTERPRETATION — plausible but not confirmed causal |
| Observed result | Large positive bias; 56/60 over-predictions |
| Interpretation | Consistent with COVID learning loss |
| Causal explanation | PROPOSED, NOT ESTABLISHED — observational data only |
| Remaining limitation | No counterfactual; no non-COVID year comparison; 4–5 exceptions unexplained |

---

### Claim 5 — Trend slope vs. TimesFM absolute error (r = 0.497)

| Field | Content |
|-------|---------|
| Source report | phase3_2_hypotheses_and_tests.md (H4), phase3_2_forecasting_diagnostics.md (E.3.3) |
| Existing wording | "H4 STRONGLY SUPPORTED. r = +0.496, p<0.001 (n=60). Countries with steeper upward trends experienced larger TimesFM errors." |
| Evidence | Pearson r reproduced: 0.4965, p=0.000055, 95% CI [0.278, 0.666]. However: (a) Spearman r = 0.178, p=0.173 (non-significant). (b) Without Albania: r=0.316, p=0.015. (c) Without Albania + Malaysia: r=0.227, p=0.087 (non-significant). Albania alone explains 0.181 of the Pearson correlation. |
| Corrected wording | "Pearson r = 0.4965 (p<0.001) is reproduced. However, this result is not robust: (a) Spearman correlation is 0.178 (p=0.17, non-significant), indicating the Pearson result is sensitive to outliers. (b) Albania is the single most influential observation; without it, r drops from 0.497 to 0.316 and without the two most influential countries to 0.227 (p=0.087). This relationship is an **exploratory diagnostic observation**, not an established finding. It is consistent with the COVID disruption narrative but cannot be confirmed as such." |
| Status | EXPLORATORY — reproduced but not robust |
| Observed result | Pearson r=0.497 reproduced from source columns |
| Interpretation | Countries with steeper upward trends tended to have larger TFM errors |
| Causal explanation | PROPOSED: TFM extrapolates trends; COVID reversed them; ergo steeper trends → larger errors |
| Remaining limitation | Driven by Albania; Spearman non-significant; single year, single program |

---

### Claim 6 — Persistence vs. TimesFM vs. Linear trend ranking

| Field | Content |
|-------|---------|
| Source report | phase3_2_forecasting_diagnostics.md, phase3_2_integrated_synthesis.md |
| Existing wording | "Persistence (14.48) beats TimesFM (15.83) and linear trend (16.98)" |
| Evidence | Independently reproduced: Persistence MAE=14.4820, TFM MAE=15.8296, Linear MAE=16.9751. Country sets identical (60). Target variable identical. Error definition identical. |
| Corrected wording | Same — MAE ranking is verified. **Caveat to add:** "This is a single development sample (PISA 2022 math, n=60). Linear trend also shows r=0.688 with trend_slope — a stronger relationship than TFM (0.497), consistent with linear trend extrapolating the pre-2022 slope most aggressively. Neither TimesFM nor linear trend has been evaluated on a held-out test set." |
| Status | VERIFIED — with caveat about scope |
| Observed result | MAE ranking: Persistence < TFM < Linear trend (n=60, PISA 2022 math) |
| Interpretation | Persistence is the best single-method baseline in this development sample |
| Causal explanation | Linear trend and TFM both extrapolate trends; COVID reversed trends; persistence anchors to last observation |
| Remaining limitation | One target cycle, one program, one domain; COVID confound; not yet tested on 2025 holdout |

---

### Claim 7 — PISA 2025 readiness

| Field | Content |
|-------|---------|
| Source report | phase3_1_1_validation_addendum.md, phase3_2_integrated_synthesis.md |
| Existing wording | "CONDITIONALLY READY" |
| Evidence | (a) SE validation passes (MAD=0.000026). (b) Path bug documented. (c) Direct mean comparison blocked. (d) Quality flags documented. (e) Provenance of existing estimates not independently reproducible. |
| Corrected wording | "CONDITIONALLY READY — subject to: (1) path fix applied and verified before any re-run; (2) existing estimates used as-is for evaluation (proven SE-accurate; not reproducible by re-run without fix); (3) quality-flagged countries (ALB CAN NLD NOR NZL USA) reported as separate stratum; (4) positive-bias claim corrected in evaluation design document; (5) direct mean comparison attempted when OECD Results Volume I becomes accessible." |
| Status | CONDITIONALLY READY |
| Observed result | Estimates exist; SE validated; path bug present |
| Interpretation | Data suitable for evaluation; reproducibility limited |
| Causal explanation | N/A |
| Remaining limitation | No direct mean comparison; path bug means re-run requires fix; provenance not fully documented |
