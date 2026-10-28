# Phase 3.24 — Literature-motivated linear/quadratic trend benchmark

**Date:** 2026-10-15

**Status:** Historical rolling-origin benchmark extended with a pre-target linear-versus-quadratic trend selection baseline. The external paper is confirmed as a future-cycle PISA country-score forecast precedent; its eligibility under the unavailable original search protocol remains unresolved.

**Script:** [`scripts/run_pisa_quadratic_trend_benchmark.py`](../../scripts/run_pisa_quadratic_trend_benchmark.py)

**Outputs:** [`outputs/pisa_tabular_benchmark_v2/`](../../outputs/pisa_tabular_benchmark_v2/)

**Paired uncertainty extension:** [`scripts/summarize_historical_country_cluster_bootstrap_v2.py`](../../scripts/summarize_historical_country_cluster_bootstrap_v2.py), [`outputs/historical_country_cluster_bootstrap_v2/`](../../outputs/historical_country_cluster_bootstrap_v2/)

**Candidate screen:** [`audit/external_forecast_candidate_screen_v1.csv`](audit/external_forecast_candidate_screen_v1.csv)

## Why this extension was made

The external study by Becker and Coyle forecasts country PISA 2022 performance from country PISA data available through 2018. It fits linear and quadratic time trends per country and compares them by an F test; the quadratic forecast is selected only when its improvement is significant at p <= .05. The data include country mean scores in mathematics, science, and reading, alongside additional score summaries. This is a directly relevant historical future-cycle forecasting precedent and supports testing a literature-motivated trend-selection benchmark alongside persistence, linear trend, Ridge, Random Forest, and XGBoost.

The article was published in 2026, after its historical forecast target was observed. It is therefore evidence of a retrospective held-out-cycle analysis, not a prospective real-time forecast publication. Its DOI is absent from the frozen 1,266-record corpus and inventoried Scopus/WoS JSONs. The original search and eligibility protocol is not recoverable from current project files, so inclusion eligibility is **unresolved**. The frozen corpus and its algorithm frequency denominators were not changed. A versioned supplemental screen records these distinctions.

## Implementation

- Each country/domain/outer fold is treated separately; the target cycle and later values are excluded from both model fitting and linear-versus-quadratic selection.
- A nested F test evaluates whether the quadratic time term improves fit over a linear trend. The quadratic is selected iff p < .05; otherwise the linear model is selected.
- With exactly three prior cycles, the quadratic is not fitted or selected because it would interpolate the training points; the linear trend is used. Quadratic testing begins at four prior cycles.
- Calendar years are centered at the training-window mean for numerical stability. This does not change fitted values relative to an uncentered polynomial in calendar year.
- PISA 2025 is excluded from these historical folds and is not used for model fitting or selection.
- Existing v1 predictions, metrics, panel, fold manifest, and score inputs were read but not modified. The v2 run is a separate output directory.

This is a transparent operationalization of the article's stated R²/F-test selection principle, not an exact reproduction of its country forecasts: our panel, common-country folds, and target cycles differ. The paper states p <= .050, whereas this run uses p < .05; none of the eligible fits had p exactly .05, so this boundary convention did not change any selection. The article points to supplementary country-level fits and tests; those supplementary tables have not yet been checked against our implementation.

## Verification and results

The v2 run reproduced the v1 target keys, observed scores, and country/context metadata exactly. It generated 624 new trend forecasts, one for each of the 624 domain-country-fold targets. The combined output has 3,744 rows across six methods, unique prediction keys, and finite forecasts. The quadratic term was selected in 51 cases; 151 cases used the linear model because there were fewer than four prior cycles. Other cases were tested and retained linear because the quadratic improvement did not meet p < .05.

| Domain | Persistence equal-fold MAE | Linear trend | Linear/quadratic selection | RF | XGBoost |
|---|---:|---:|---:|---:|---:|
| Mathematics | 9.328 | 11.297 | 11.564 | 9.811 | 9.563 |
| Reading | 10.156 | 13.333 | 14.099 | 10.759 | 10.379 |
| Science | 8.442 | 10.799 | 11.671 | 9.168 | 8.565 |

Across all 624 cases, selected linear/quadratic trend MAE is 12.659, versus 11.984 for the fixed linear trend and 9.415 for persistence. Thus reproducing this literature-motivated selection rule does not improve the current historical benchmark. It is a useful comparator, not a selected production forecaster. Only four temporal target cycles are available, so these fold averages are descriptive and not evidence of universal method superiority.

The paired country-cluster bootstrap was extended to include this method (20,000 replicates; country clusters preserved across the four folds). The table reports `MAE(LinearQuadraticTrend) - MAE(Persistence)`; positive values favor Persistence.

| Domain | Δ equal-fold MAE | Exploratory 95% country-cluster interval |
|---|---:|---:|
| Mathematics | +2.236 | [+1.237, +3.325] |
| Reading | +3.942 | [+2.495, +5.488] |
| Science | +3.228 | [+2.234, +4.259] |

These intervals are conditional on the four observed historical target cycles, fixed predictions, and published point-score estimates. They do not represent uncertainty over future assessment cycles, PISA sampling/measurement, or a random population of countries; treat them as exploratory comparisons, not confirmatory tests.

## Interpretation and next checks

This result strengthens the rationale for including RF and XGBoost as general predictive-literature comparators while correcting the earlier gap in conventional country-level forecast precedent. It does not establish RF or XGBoost as the most frequent methods among true ILSA future-cycle forecasts: the verified forecast precedent uses linear/quadratic trend regression. The 1,266-record frequency result remains a descriptive property of the frozen coded corpus, not an exhaustive ranking of forecasting algorithms.

The paper's strict project eligibility as a forecasting precedent is confirmed from the publisher's Methods section. Its inclusion in the original corpus remains undecidable until the original query and screening rules are recovered. The next literature gate is a second-reviewer screen and decision after those rules are recovered, alongside search recall work and comparison with the article's supplementary fits. Separately, all benchmark results still need final end-to-end reproduction audit and manuscript synthesis; PISA 2025 remains outcome-exposed and cannot provide prospective confirmation.

## Reproduction

Run `python3 scripts/run_pisa_quadratic_trend_benchmark.py` to reproduce the added trend forecasts, then `python3 scripts/summarize_historical_country_cluster_bootstrap_v2.py` for the extended paired uncertainty analysis. The first script verifies panel/fold hashes and target-key/score agreement with v1; the second verifies that existing tabular predictions match the TimesFM all-model panel and adds only the new model. Both refuse to overwrite existing output directories and record input/code/output hashes. All recorded checks passed; the existing v1 benchmark and bootstrap outputs and frozen panel were left untouched.

## Source

Becker, D., & Coyle, T. R. (2026). *Education during the COVID-19 pandemic: Does national culture predict school closures and changes in student performances?* Frontiers in Education, 11. DOI: 10.3389/feduc.2026.1734988. Publisher full text: <https://www.frontiersin.org/journals/education/articles/10.3389/feduc.2026.1734988/full>.
