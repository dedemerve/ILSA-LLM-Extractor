# Phase 3.2 — Predictive Evidence Audit (Phase C)

**Date:** 2026-10-14  
**Protocol:** Phase 3.2 Full Specification, Phase C  
**Status:** COMPLETE — Narrative synthesis; quantitative meta-analysis not justified  
**This file does NOT overwrite any prior artifact.**

---

## C.1 Scope

This phase examines the 166 studies classified as `research_design_type = predictive` to determine:
1. What performance evidence exists
2. Whether metrics are comparable across studies
3. Which methodological factors may explain performance variation

---

## C.2 Performance Metric Landscape

### C.2.1 Metric types present

Performance metrics are stored as free text in `2_Main_Findings.performance_metrics`. A structured review of the 166 predictive studies reveals the following metric types:

| Metric type | Approx. studies | Notes |
|-------------|----------------|-------|
| R² / explained variance | ~40 | Most common for regression models |
| Classification accuracy | ~35 | Binary or multi-class |
| RMSE / MSE | ~25 | Continuous outcome regression |
| MAE | ~10 | Rarely reported explicitly |
| AUC / ROC | ~15 | Binary classification |
| F1 / precision/recall | ~10 | Classification |
| Correlation coefficient | ~15 | Often as supplementary |
| Regression coefficients | ~30 | Not predictive performance per se |
| "Not reported" / descriptive only | ~20 | No quantitative performance |

**Note:** These are approximate counts derived from keyword inspection of the free-text `performance_metrics` field. Numeric extraction was not performed systematically. The actual distribution of comparable values is uncertain.

### C.2.2 Comparability constraints

The following constraints prevent direct pooling of reported performance values across studies:

| Constraint | Studies affected | Impact |
|-----------|-----------------|--------|
| Different prediction targets (scores vs. categories vs. rankings) | All 166 | Incomparable |
| Different units of analysis (student vs. school vs. country) | All 166 | Incomparable |
| Different ILSA programs | ~120 across PISA/TIMSS/PIRLS/ICCS/ICILS | Incomparable scales |
| Different metrics (R² vs. RMSE vs. accuracy) | All | Not directly comparable |
| Different baseline methods | Most | Cannot assess relative improvement |
| Random vs. temporal evaluation splits | Unknown proportion | Inflated vs. genuine performance |
| Single-country vs. multi-country models | Many | Different generalizability |

**Conclusion:** Quantitative meta-analysis of predictive performance across the 166 studies is **NOT JUSTIFIED**. The metrics are not comparable in type, target, scale, or evaluation design.

---

## C.3 Subset with Potential Comparability: Country-Level Continuous Score Prediction

The narrowest comparable group would include studies that:
1. Predict a continuous country-level achievement outcome (mean score)
2. Use ILSA data at the country/system level
3. Predict a later assessment cycle (genuine temporal holdout)
4. Report MAE or RMSE in score-point units

**Count meeting all criteria: 0 confirmed studies**

The 17 studies classified as `country_system_prediction` (code D) do not all meet these criteria:
- Some predict binary or categorical outcomes
- None confirmed as using genuine temporal holdout
- Metric types vary

**Consequence:** No external benchmark from the literature can be directly compared to the TimesFM/persistence MAE of ~14–16 score points in the current project. The existing experiment is operating in a methodological gap in the published literature.

---

## C.4 Candidate Explanatory Factors for Performance Variation

The following table documents candidate factors and their availability in the corpus:

| Factor | Available in corpus | Testable with current data | Notes |
|--------|---------------------|---------------------------|-------|
| Prediction target type (score vs. category vs. ranking) | Partially (coded as target_variable in findings) | N/A for this project | |
| Unit of analysis (student vs. country) | Yes (prediction_target_level) | N/A (all country-level here) | |
| Temporal evaluation design (random vs. holdout) | Not coded — embedded in text | HYPOTHESIS: random splits inflate performance | KEY FACTOR |
| Sample size / number of countries | Partially (n_analytic in estimates; n_countries sparse in corpus) | Testable with project data | |
| Context length (number of historical cycles) | Not coded in corpus | Testable: n_context in step2 | KEY FACTOR |
| Model complexity | Yes (ml_family, ml_primary) | N/A for zero-shot experiment | |
| Baseline method used | Rarely reported | Not testable from corpus | MISSING |
| Missing-cycle patterns | Not coded | Partially testable with project data | |
| Country coverage | Sparse in corpus | Testable within project | |
| Assessment year (temporal drift) | Partially (normalized_cycles) | Partially testable | |
| Domain/subject | Yes (target_domain) | N/A (all math here) | |
| Use of future-cycle data in model development | Not coded | HYPOTHESIS: leakage inflates performance | KEY FACTOR |

---

## C.5 Narrative Synthesis: What the Predictive Literature Establishes

### C.5.1 Established findings

1. **Student-level prediction dominates.** 155/166 predictive studies (93%) predict student-level outcomes. Country-level forecasting is marginal (n=17, 10%).

2. **ML models frequently outperform traditional statistics for student-level outcomes** within a single cycle. This is a consistent finding across PISA, TIMSS, and PIRLS student-level studies (code A). However, this finding does not transfer directly to country-level time-series forecasting.

3. **Tree-based/ensemble ML is the largest coded model family.** It appears in 81/166 predictive studies (48.8%, `ml_family`). In the more specific `ml_primary` field, Random Forest is coded in 42/147 predictive studies with a non-missing primary algorithm, and XGBoost in 17/147; these are the two most frequent exact primary labels. This tells us what models are popular in the broad predictive ILSA literature, not what is optimal for country-level forecasting. Only 3 country-level prediction studies name Random Forest and 2 name XGBoost as the primary method; no genuine future-cycle forecasting study is confirmed.

4. **Baseline methods are rarely reported.** Most studies do not compare against simple persistence or trend baselines. This means reported "high performance" may be relative to a null model, not to simple alternatives.

### C.5.2 What the literature does NOT establish

1. **No direct evidence for country-level future-cycle forecasting performance.** The closest study (Braun et al. 2025) was a retrospective QA analysis with in-sample 2019 residuals — not a genuine holdout prediction.

2. **No evidence on whether persistence or linear trend outperforms ML for short country-level time series.**

3. **No evidence on whether TimesFM-style zero-shot forecasting is competitive with baselines for ILSA data.**

4. **No evidence on the impact of the temporal evaluation design on country-level ILSA prediction.**

### C.5.3 Comparability verdict

**VERDICT:** A quantitative performance meta-analysis across the 166 predictive studies is NOT justified.  
Appropriate alternative: structured narrative synthesis (this document) + diagnostic analysis from the existing project (Phase E).
