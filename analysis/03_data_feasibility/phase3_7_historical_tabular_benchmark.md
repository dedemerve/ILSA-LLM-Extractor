# Phase 3.7 — Historical PISA Tabular Forecasting Benchmark

**Date:** 2026-10-15  
**Status:** First fixed-configuration rolling-origin comparison complete  
**Prediction output:** `outputs/pisa_tabular_benchmark_v1/historical_predictions.csv`  
**Metric output:** `outputs/pisa_tabular_benchmark_v1/historical_metrics.csv`  
**Run manifest:** `outputs/pisa_tabular_benchmark_v1/run_manifest.json`

---

## 1. Design

Five approaches were evaluated for PISA mathematics, reading, and science:

- Persistence (last observed score)
- Country-specific linear trend
- Pooled Ridge regression
- Pooled Random Forest
- Pooled XGBoost

The temporal folds target 2012, 2015, 2018, and 2022. Each fold uses the frozen intersection of countries eligible in all three domains, with at least three earlier score observations. Test-set sizes are 40, 51, 57, and 60 countries, respectively.

For each outer target cycle `t`, supervised training targets satisfy `cycle < t`. Model features contain only the same country's earlier scores: latest score, second-latest score, score change, target year, gap from last observed cycle, and number of prior observations. Median imputation and scaling are fit within each fold's training set. The historical metric table contains 3,120 country predictions and 60 fold/domain/model rows; all predictions are present and prediction keys are unique.

The model hyperparameters were fixed before this run and were not selected using PISA 2025. This is a first benchmark, not a tuned or independently confirmed model-selection result.

## 2. Mean Absolute Error by Fold

| Target cycle | Domain | Persistence | Linear trend | Ridge | Random Forest | XGBoost |
|---:|---|---:|---:|---:|---:|---:|
| 2012 | Mathematics | 8.29 | 10.05 | 9.62 | 9.13 | 8.97 |
| 2012 | Reading | 9.60 | 11.50 | 9.40 | 9.94 | 9.49 |
| 2012 | Science | 6.95 | 9.52 | 8.36 | 8.40 | 7.37 |
| 2015 | Mathematics | 8.18 | 9.63 | 8.80 | 8.07 | 7.87 |
| 2015 | Reading | 10.43 | 12.59 | 13.10 | 11.38 | 11.23 |
| 2015 | Science | 10.87 | 12.30 | 12.31 | 12.44 | 11.68 |
| 2018 | Mathematics | 6.35 | 8.54 | 6.71 | 6.64 | 6.76 |
| 2018 | Reading | 8.25 | 12.35 | 9.53 | 8.41 | 8.31 |
| 2018 | Science | 7.11 | 10.03 | 6.01 | 6.82 | 6.00 |
| 2022 | Mathematics | 14.48 | 16.98 | 14.71 | 15.40 | 14.65 |
| 2022 | Reading | 12.35 | 16.90 | 11.31 | 13.30 | 12.48 |
| 2022 | Science | 8.83 | 11.35 | 9.82 | 9.01 | 9.21 |

## 3. Equal-Fold Average MAE by Domain

| Domain | Persistence | Linear trend | Ridge | Random Forest | XGBoost |
|---|---:|---:|---:|---:|---:|
| Mathematics | 9.33 | 11.30 | 9.96 | 9.81 | 9.56 |
| Reading | 10.16 | 13.33 | 10.83 | 10.76 | 10.38 |
| Science | 8.44 | 10.80 | 9.12 | 9.17 | 8.56 |

Across all 624 country-domain-fold cases, pooled MAE is: Persistence **9.42**, XGBoost **9.59**, Random Forest **9.99**, Ridge **10.03**, and linear trend **11.98**.

## 4. Interpretation

- Persistence has the lowest equal-fold average MAE in each domain and the lowest pooled MAE in this fixed-configuration benchmark.
- XGBoost is the strongest learned tree model overall and is close to persistence in pooled MAE; it wins individual domain-fold cells, but not the equal-fold domain averages.
- Random Forest is better than the linear trend on average in all domains, but does not beat persistence overall.
- Ridge's performance is mixed; it has the lowest MAE in PISA 2022 reading among the five approaches.
- PISA 2022 mathematics persistence MAE reproduces the prior project result (14.48), which provides a cross-check on that historical baseline.

These results do not prove that persistence is universally superior. There are only four target cycles, scores are country aggregates, and the 2022 cycle includes the pandemic-era disruption. No inferential significance claim is made from these fold averages. Hyperparameter sensitivity, uncertainty intervals, survey-score standard errors, and PISA 2025 remain for subsequent stages.

## 5. Leakage and Exposure Notes

- No PISA 2025 target is included in the training folds or this benchmark.
- Training target years are strictly earlier than each outer test cycle.
- Historical target cycles are used for model assessment and may inform later development decisions; consequently, report performance as temporal validation, not as a pristine final test.
- PISA 2025 is already outcome-exposed in the repository and is reserved for a pre-specified retrospective benchmark only. It cannot be represented as an untouched blind test.
