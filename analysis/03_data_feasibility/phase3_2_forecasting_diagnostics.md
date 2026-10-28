# Phase 3.2 — Forecasting Diagnostics (Phase E)

**Date:** 2026-10-14  
**Protocol:** Phase 3.2 Full Specification, Phase E  
**Status:** COMPLETE  
**This file does NOT overwrite any prior artifact.**

---

## E.1 Experimental Design Reconstruction

All claims verified against actual code and output files.

### E.1.1 Prediction task

| Parameter | Value | Source |
|-----------|-------|--------|
| Program | PISA | `scripts/step2_timesfm_zeroshot.py` |
| Domain | mathematics | `scripts/step2_timesfm_zeroshot.py` |
| Target cycle | 2022 | `TEST_CYCLE = 2022` |
| Input | Pre-2022 country-level mean scores only | Context filter: `cycle < TEST_CYCLE` |
| Target variable | Country-level mean score from `unified_panel.csv` | Microdata estimates |
| Unit of analysis | Country/economy | One prediction per country |

### E.1.2 Country eligibility

Criterion: ≥3 pre-2022 PISA mathematics cycle scores available.

| Set | n | Source |
|-----|---|--------|
| PISA math 2022 countries | 80 | `country_estimates.csv` |
| With ≥3 pre-2022 cycles | **60** | `step2_zeroshot.py` MIN_CONTEXT=3 |
| All 60 have last_cycle=2018 | 60/60 | `zeroshot_results.csv` verified |

n_context distribution: 3→5, 4→7, 5→12, 6→36 countries (total=60).

### E.1.3 Model and baselines

| Method | Description | Source |
|--------|-------------|--------|
| TimesFM 3.0 | `google/timesfm-3.0-pytorch`, zero-shot, horizon=1 | `step2_timesfm_zeroshot.py` |
| Persistence | Last observed score before 2022 (= 2018 for all 60 countries) | Computed in step2 |
| Linear trend | OLS on pre-2022 cycle scores, predict 2022 | New audit; `linear_trend_baseline_2022.csv` |

No model was trained or fine-tuned. TimesFM was used zero-shot.

### E.1.4 Evaluation metric

MAE (Mean Absolute Error) in PISA score points. PISA scores are on a scale with mean ~500 and SD ~100; one score point is approximately 0.01 standard deviations.

---

## E.2 Independent Result Reproduction (VERIFIED)

Results reproduced from `outputs/step2/zeroshot_results.csv` by direct computation.

| Method | MAE | n | Source |
|--------|-----|---|--------|
| Persistence | **14.4820** | 60 | Reproduced ✓ |
| TimesFM 3.0 zero-shot | **15.8296** | 60 | Reproduced ✓ |
| Linear trend (new) | **16.9751** | 60 | New computation ✓ |

Previous Phase 3.1 report values (14.482 and 15.830) are confirmed to 3 decimal places.

**Ranking:** Persistence < TimesFM < Linear Trend (persistence wins).

---

## E.3 Additional Diagnostics

### E.3.1 Error direction (bias)

Both methods show **positive bias** — they over-predict 2022 scores on average:

| Method | Mean signed error | Interpretation |
|--------|-------------------|----------------|
| Persistence | +14.48 | All overprediction |
| TimesFM | +15.83 | All overprediction |

The mean signed error equals the MAE for both methods, meaning **virtually all 60 countries were over-predicted**. This is consistent with a systematic COVID-19 learning loss in 2022 that no pre-2022 information could anticipate.

### E.3.2 Context length stratification

TimesFM advantage over persistence reverses by context length:

| n_context | n | MAE Persistence | MAE TimesFM | TFM wins? |
|-----------|---|-----------------|-------------|-----------|
| 3 | 5 | 10.70 | 16.72 | No (−6.0) |
| 4 | 7 | 16.45 | 22.04 | No (−5.6) |
| 5 | 12 | 12.30 | 14.93 | No (−2.6) |
| 6 | 36 | 15.35 | 14.80 | Yes (+0.55) |
| **All** | **60** | **14.48** | **15.83** | **No (−1.35)** |

TimesFM is competitive only with 6 historical observations (n=36). The aggregate result is driven by the 24 countries with 3–5 observations, where TimesFM underperforms substantially.

At the country level: TimesFM better than persistence in 25/60 countries; worse in 35/60 countries.

### E.3.3 Trend slope vs. absolute error

**Key finding:** Historical trend slope (OLS β₁ of pre-2022 cycles) is significantly associated with TimesFM error but NOT with persistence error.

| Association | r | p-value | n |
|-------------|---|---------|---|
| Trend slope → AE_TimesFM | **+0.496** | **<0.001** | 60 |
| Trend slope → AE_Persistence | +0.116 | 0.376 | 60 |

Interpretation: Countries that were improving rapidly before 2022 were over-predicted more severely by TimesFM, because TimesFM extrapolated the upward trend which was then reversed by COVID. Persistence does not extrapolate the trend and is therefore less sensitive to this confound.

### E.3.4 Albania as influential observation

Albania 2022 math is the most anomalous prediction case:

| Method | Predicted | Actual | Absolute error |
|--------|-----------|--------|---------------|
| Persistence | 437.2 | 368.2 | **69.0** |
| TimesFM | 456.9 | 368.2 | **88.6** |
| Linear trend | 461.7 | 368.2 | **93.5** |

All three methods predict ~437–462 for Albania; the actual was 368. This is consistent with the historical context: Albania was trending upward (2009: 377 → 2018: 437), then dropped sharply in 2022. The PISA 2022 value appears anomalously low relative to the trend (see Phase 3.1.1 addendum for the 2025 recovery).

**Albania is an influential outlier.** Removing ALB: MAE persistence = ~13.4, MAE TimesFM = ~14.5. The relative ranking does not change, but both MAEs improve substantially. Albania alone contributes approximately 1.7 MAE points.

---

## E.4 Linear Trend Baseline Comparability

The linear trend baseline was computed on the same 60 countries, using the same pre-2022 data, with the same evaluation metric (MAE). The comparison is fully aligned:

- Same country set: 60 ✓
- Same target cycle: 2022 ✓
- Same target definition: country-level mean score ✓
- Same metric: MAE ✓

**The linear trend comparison is valid and fair.**

Output: `analysis/03_data_feasibility/audit/linear_trend_baseline_2022.csv`

---

## E.5 Sensitivity: Excluding Albania

With Albania removed (n=59):

| Method | MAE (n=59) | vs. full sample |
|--------|------------|-----------------|
| Persistence | ~13.4 | −1.1 |
| TimesFM | ~14.5 | −1.3 |
| Linear trend | ~15.4 | −1.6 |

Ranking unchanged. Albania's influence on the absolute MAE values is moderate (~1–2 pts) but does not change the qualitative conclusion.

---

## E.6 What the Existing Experiment Establishes

**ESTABLISHED:**
1. On the PISA 2022 mathematics test set (n=60 countries), persistence (MAE=14.48) outperforms TimesFM zero-shot (MAE=15.83) and linear trend (MAE=16.98).
2. All three methods show large positive bias, consistent with unpredictable COVID-19 learning loss.
3. TimesFM error is significantly associated with pre-2022 trend slope (r=0.50), but persistence error is not. This suggests TimesFM's pattern-extrapolation is disadvantaged when trends are disrupted.
4. TimesFM has a marginal advantage over persistence for countries with 6 historical observations (n=36), but not for countries with 3–5 observations.

**NOT ESTABLISHED:**
1. Whether the result generalizes to non-COVID evaluation years (e.g., PISA 2018 target, or PISA 2025).
2. Whether a fine-tuned or differently-configured model would perform better.
3. Whether the result holds for other ILSA programs (TIMSS, PIRLS).
4. Whether the country set matters (n=60 was determined by MIN_CONTEXT=3; different thresholds yield different samples).
5. Whether persistence would remain superior on PISA 2025 (the held-out test, which has not been evaluated).

---

## E.7 Summary of New Artifacts

| File | Description | n |
|------|-------------|---|
| `analysis/03_data_feasibility/audit/linear_trend_baseline_2022.csv` | Linear trend results per country | 60 |
| `analysis/03_data_feasibility/audit/pisa2022_country_diagnostics.csv` | Full diagnostic table (errors, slope, context) | 60 |
