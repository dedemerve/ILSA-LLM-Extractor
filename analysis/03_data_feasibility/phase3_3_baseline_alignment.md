# Phase 3.3 — Baseline Alignment Verification

**Date:** 2026-10-14  
**Protocol:** Phase 3.3 Readiness Gate, Task 2  
**Status:** COMPLETE — all values independently recalculated from raw columns  
**This file does NOT overwrite any prior artifact.**

---

## 1. Alignment Verification

All three methods (persistence, TimesFM 3.0 zero-shot, linear trend) were verified against the following alignment criteria.

| Criterion | Status | Detail |
|-----------|--------|--------|
| Same target cycle | PASS | 2022 for all three |
| Same target variable | PASS | PISA mathematics mean score from country_estimates.csv |
| Same country set | PASS | Exactly 60 countries; set equality confirmed |
| Same observed target values | PASS | true_score (step2) = actual (linear_baseline) = mean (estimates); max diff = 0.000000 |
| Same persistence values | PASS | persistence column in step2 = persistence column in linear_baseline; max diff = 0.000000 |
| Missing-data rule | PASS | No missing values; MIN_CONTEXT=3 applied identically |
| Error definition | PASS | MAE = mean(|predicted − actual|); independently confirmed for all three |

**Conclusion: All three baselines use identical country sets, identical target values, and a consistent error definition. The comparison is fair.**

---

## 2. Independently Recalculated Metrics

Recalculated from raw columns (`persistence`, `timesfm_zeroshot`, `linear_trend`, `true_score`) — not from stored error columns.

| Method | n | MAE | Mean signed error | Over-predictions | Under-predictions |
|--------|---|-----|-------------------|-----------------|------------------|
| Persistence | 60 | **14.4820** | +13.8845 | 56/60 | 4 (JPN KAZ KOR SGP) |
| TimesFM 3.0 | 60 | **15.8296** | +14.7963 | 55/60 | 5 (AUS JPN KAZ SGP TUR) |
| Linear trend | 60 | **16.9751** | +16.2714 | 55/60 | 5 (JPN KAZ KOR MYS SGP) |

**Ranking: Persistence < TimesFM < Linear trend (confirmed).**

Note: Mean signed error ≠ MAE for all three methods because a small number of countries (4–5) were under-predicted. The stored error columns in prior artifacts used `|predicted − actual|` (absolute value) in all cases; their sums are identical to MAE but should not be treated as signed errors.

---

## 3. Trend Slope vs. Absolute Error Correlation

### 3.1 Reproduced correlation

| Measure | Pearson r | p-value | 95% CI |
|---------|-----------|---------|--------|
| r(trend_slope, AE_TFM) | **0.4965** | 0.000055 | [0.278, 0.666] |
| r(trend_slope, AE_Persist) | 0.1163 | 0.376 | [−0.142, 0.360] |
| r(trend_slope, AE_Linear) | **0.6882** | <0.0001 | — |

Pearson r = 0.4965 is independently reproduced from source columns. The reported confidence interval is confirmed.

### 3.2 Sensitivity analyses conducted

| Analysis | r (AE_TFM) | p | Interpretation |
|----------|-----------|---|---------------|
| Full sample (n=60) | 0.4965 | 0.000055 | Statistically significant |
| Excluding Albania (n=59) | 0.3156 | 0.015 | Still significant; attenuated |
| Excluding ALB + MYS (n=58) | 0.2269 | 0.087 | Not significant |
| Spearman (n=60) | 0.1784 | 0.173 | Not significant |

### 3.3 Leverage assessment

Albania is the single most influential observation. Removing Albania alone reduces Pearson r from 0.497 to 0.316 (change = 0.181). The next most influential country (Malaysia) contributes 0.038 to the correlation. The Spearman rank correlation (robust to outliers) is 0.178 (p=0.17, not significant).

### 3.4 Classification of the relationship

**EXPLORATORY — not established.**

The Pearson correlation is reproduced and statistically significant in the full sample. However:
- It is not robust to exclusion of Albania (the most anomalous country)
- The Spearman correlation is non-significant
- The relationship is observational; no causal mechanism is established
- Albania's 2022 score is the most anomalous prediction in the entire sample

The correlation should be treated as an exploratory diagnostic observation, not as an established finding. It is consistent with (but does not prove) the hypothesis that COVID disrupted upward-trending countries more than flat or declining-trending countries.

---

## 4. Additional Diagnostic Finding

r(trend_slope, AE_Linear) = 0.688 (p<0.0001) — stronger than for TFM. This is expected: the linear trend method extrapolates the pre-2022 trend directly; countries with steep upward trends will receive the largest linear extrapolations, and therefore the largest errors when the 2022 outcome falls below trend. This finding is NEW (not reported in Phase 3.2) and supports the mechanistic interpretation.

---

## Artifact

`analysis/03_data_feasibility/audit/phase3_3_baseline_alignment.csv` — 60 rows, one per country. Columns: country, target_cycle, target_variable, true_score, persistence_pred, tfm_pred, linear_pred, ae_persistence, ae_tfm, ae_linear, se_persistence, se_tfm, se_linear, trend_slope, n_context.
