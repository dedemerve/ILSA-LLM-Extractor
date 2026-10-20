# Phase 3.2 — Evidence-Linked Hypotheses (Phase D)

**Date:** 2026-10-14  
**Protocol:** Phase 3.2 Full Specification, Phase D  
**Status:** COMPLETE  
**This file does NOT overwrite any prior artifact.**

---

## D.1 Hypothesis Generation Framework

Hypotheses are categorized by evidence basis:

- **[LIT]** — Supported by direct evidence in the literature corpus
- **[GAP]** — Suggested by gaps or patterns in the literature
- **[THEORY]** — Theory-driven; limited direct empirical support
- **[TESTABLE-NOW]** — Can be tested with current project data (PISA math, n=60, 2022)
- **[TESTABLE-FUTURE]** — Requires additional data (2025, more programs, etc.)
- **[NOT-TESTABLE]** — Requires external data not currently available

---

## D.2 Hypotheses

### H1: Context length and forecasting error

**ID:** H1  
**Hypothesis:** Countries with more historical ILSA observations (longer time series) have lower forecasting error than countries with fewer observations.  
**Basis:** [THEORY] + [GAP] — No direct literature evidence for country-level ILSA forecasting. General time-series forecasting theory: longer series support better model estimation.  
**Evidence from corpus:** n_context in `outputs/step2/zeroshot_results.csv` available for all 60 countries.  
**Empirical test performed (Phase E):** Pearson r between n_context and AE_TimesFM = **−0.114, p=0.387**; r between n_context and AE_Persistence = **0.081, p=0.539**. Neither is statistically significant.  
**Result:** H1 NOT SUPPORTED in the current sample (n=60, PISA 2022 math).  
**Limitation:** Small sample; n_context ranges only from 3 to 6, limiting power.  
**Status:** TESTED; result inconclusive at n=60.

---

### H2: Temporal evaluation design inflates reported performance

**ID:** H2  
**Hypothesis:** Studies using random or within-cycle train/test splits report substantially higher predictive performance than studies using genuine temporal holdout designs.  
**Basis:** [LIT] — The 28 `cross_cycle_prediction_not_fully_verified` studies lack verified temporal holdout. General ML methodology: random splits ignore temporal autocorrelation and produce optimistically biased performance estimates when applied to time-structured data. [GAP] — The corpus lacks enough verified temporal holdout studies to compute the difference directly.  
**Evidence from corpus:** Not directly testable (no confirmed future-cycle holdout studies to compare against random-split studies on the same task).  
**Empirical test proposed:** Compare rolling-origin evaluation on PISA panel against single-cycle holdout.  
**Status:** HYPOTHESIS — cannot be tested with current data without additional future-cycle results. [TESTABLE-FUTURE]

---

### H3: Simple baselines competitive for short country-level time series

**ID:** H3  
**Hypothesis:** For ILSA country-level data with ≤6 historical observations, persistence and linear trend baselines perform comparably to or better than more complex forecasting models.  
**Basis:** [GAP] — No directly comparable literature evidence. General forecasting literature (Makridakis et al. 2018): simple methods outperform complex methods on short, noisy time series. [THEORY] — With ≤6 observations, fitting complex temporal patterns is unreliable.  
**Evidence from corpus:** Zero comparable studies in the 1,266-study corpus.  
**Empirical test performed (Phase E):** MAE Persistence=14.48, MAE TimesFM=15.83, MAE Linear Trend=16.98 (n=60, PISA 2022 math). Persistence beats both alternatives.  
**Result:** H3 SUPPORTED — persistence is the best baseline; TimesFM and linear trend perform worse.  
**Limitation:** Single target cycle, single program, single domain. Does not establish which model is better in the long run.  
**Status:** PARTIALLY TESTED; supported in the current sample.

---

### H4: Historical trend slope predicts TimesFM error

**ID:** H4  
**Hypothesis:** Countries with steeper historical upward trends will have larger TimesFM absolute errors for the 2022 target cycle, due to COVID-19 disruption causing a reversal of the trend.  
**Basis:** [THEORY] — Countries improving faster pre-2022 would experience more expected-vs-actual divergence when COVID disrupted the trend. TimesFM extrapolates recent patterns; it should therefore over-predict more strongly for countries with steeper upward trends.  
**Evidence from corpus:** No direct literature evidence. Empirical test designed from first principles.  
**Empirical test performed (Phase E):** Pearson r between trend_slope and AE_TimesFM = **+0.496, p<0.001** (n=60).  
**Result:** H4 STRONGLY SUPPORTED. Historical trend slope is significantly associated with TimesFM absolute error (r=0.50). Countries with steeper upward trends experienced larger TimesFM errors in 2022.  
**Caveat:** Observational association; cannot establish that trend extrapolation caused the error. COVID disruption is a confound.  
**Additional note:** Trend slope does NOT predict persistence error (r=0.116, p=0.376), which is consistent with persistence anchoring to the last observation rather than extrapolating the trend.  
**Status:** TESTED; result: CONFIRMED association.

---

### H5: Systematic positive bias in 2022 predictions

**ID:** H5  
**Hypothesis:** Both persistence and TimesFM over-predict 2022 PISA mathematics scores on average, consistent with a systematic COVID-19 achievement decline that is unpredictable from pre-2022 data.  
**Basis:** [THEORY] — COVID-19 caused learning loss in most countries. Pre-2022 data cannot inform models about this disruption.  
**Empirical test performed (Phase E):**  
- Mean signed error (persistence): +14.48 score points (all positive bias)  
- Mean signed error (TimesFM): +15.83 score points (all positive bias)  
**Result:** H5 SUPPORTED. Both methods have large positive bias, consistent with unpredictable COVID-19 learning loss. The mean signed error equals the MAE, meaning **almost every prediction is above the actual value** (overprediction).  
**Status:** TESTED; result: CONFIRMED.

---

### H6: Performance varies by context length stratum

**ID:** H6  
**Hypothesis:** TimesFM gains advantage over persistence in the higher-context (n_context=6) stratum because more historical observations help the model identify patterns.  
**Basis:** [THEORY] — More context should help TimesFM more than persistence.  
**Empirical test performed (Phase E):**

| n_context | n | MAE Persistence | MAE TimesFM |
|-----------|---|-----------------|-------------|
| 3 | 5 | 10.70 | 16.72 |
| 4 | 7 | 16.45 | 22.04 |
| 5 | 12 | 12.30 | 14.93 |
| 6 | 36 | 15.35 | **14.80** |

**Result:** H6 PARTIALLY SUPPORTED. TimesFM outperforms persistence only in the n_context=6 stratum (36 countries), with MAE of 14.80 vs. 15.35. For n_context=3 and 4, TimesFM is substantially worse. The aggregate result (TimesFM worse) is dominated by the small-context strata where TimesFM underperforms badly.  
**Limitation:** Very small n in n_context=3 (n=5) and n_context=4 (n=7). Results are noisy.  
**Status:** TESTED; H6 has partial support.

---

### H7: Random evaluation splits overstate model performance

**ID:** H7  
**Basis:** [LIT+THEORY] — The 28 cross-cycle studies in the corpus likely used random splits or within-sample evaluation. Studies using random splits ignore temporal autocorrelation.  
**Status:** NOT TESTABLE with current data. Requires additional studies with documented split designs. [NOT-TESTABLE-NOW]

---

### H8: Metric and target definition affect comparability

**ID:** H8  
**Hypothesis:** Reported performance values in the corpus are not comparable across studies because they use different metrics, targets, and evaluation units.  
**Basis:** [LIT] — The 166 predictive studies use R², accuracy, RMSE, MAE, correlation, and other metrics across student-, school-, and country-level targets. Direct pooling would combine incompatible quantities.  
**Result:** CONFIRMED by Phase C analysis. Quantitative meta-analysis is not justified.  
**Status:** CONFIRMED — no empirical test needed; directly observable from corpus schema.

---

## D.3 Hypothesis-to-Evidence Matrix Summary

| Hypothesis | Literature evidence | Testable now | Test performed | Result |
|------------|--------------------|--------------|--------------------|--------|
| H1: Context length → error | None | Yes (n=60) | Yes | Not supported |
| H2: Random splits inflate performance | Gap/theory | No | No | Untested |
| H3: Simple baselines competitive | Gap/theory | Yes (n=60) | Yes | Supported |
| H4: Trend slope → TFM error | Theory | Yes (n=60) | Yes | Strongly supported (r=0.50) |
| H5: Positive bias from COVID | Theory | Yes (n=60) | Yes | Supported |
| H6: Context=6 → TFM advantage | Theory | Yes (n=60) | Yes | Partial support |
| H7: Random splits overstate | Lit+theory | No | No | Untested |
| H8: Metric incomparability | Lit direct | Not needed | Confirmed | Confirmed |

---

## D.4 Chain of Evidence

```
CORPUS RECORDS (1,266 studies)
  → 0 confirmed future-cycle forecasting studies
  → Absence of comparable performance benchmarks
  → SYNTHESIS: No external performance benchmark for country-level PISA forecasting

CORPUS RECORDS (166 predictive studies)
  → Tree-based/ensemble family: 81/166 predictive studies (48.8%)
  → Exact primary methods among 147 algorithm-coded predictive studies: Random Forest=42, XGBoost=17
  → Country-level prediction subset (n=17): SVM=4, Random Forest=3, XGBoost=2; no confirmed genuine future-cycle forecast
  → No verified temporal holdout
  → SYNTHESIS: Performance cannot be compared across studies or to this project

EXPERIMENTAL DATA (PISA 2022, n=60)
  → Persistence MAE=14.48, TimesFM MAE=15.83, Linear trend MAE=16.98
  → Trend slope r=0.50 with TFM error (p<0.001)
  → Positive bias in both methods (COVID learning loss)
  → TimesFM advantage only at n_context=6
  → SYNTHESIS: Persistence is currently the best single-baseline; COVID disruption
    is a major confound that neither model can handle
```
