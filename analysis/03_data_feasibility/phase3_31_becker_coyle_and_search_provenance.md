# Phase 3.31 — Becker–Coyle reconstruction and search-provenance recovery

**Date:** 2026-10-20  
**Status:** Becker–Coyle's published R procedure reconstructed and compared on the 2022 common sample; original Scopus/WoS query and screening records remain unavailable after a workspace-wide targeted file search.  
**Scope:** Complete the two literature-related follow-ups from Phase 3.28. The coded 1,266-record corpus and prior outputs were not changed.

## Becker–Coyle supplement and comparison

**Source verified:** The Frontiers article links a supplementary ZIP (`Data Sheet 1.zip`). It was retrieved from the publisher API endpoint recorded in the comparison manifest. The ZIP contains `DATA/DATA_SAS_PIS.CSV` and R scripts `R_SYN_MTH_MEA.txt`, `R_SYN_REA_MEA.txt`, and `R_SYN_SCI_MEA.txt`. The R files calculate country/domain trend models; they are code, not a ready-made published forecast table. The source R scripts fit linear and quadratic regressions to available PISA cycle means through 2018, predict 2022, calculate an F statistic from the difference in R-squared values, and select the quadratic forecast when the reported p-value is at most .05. The article describes the forecasts as 2022 estimates based on earlier cycles.

I reconstructed that procedure from the included historical means and compared the selected 2022 predictions with the project's 2022 historical target observations. This is a computational reconstruction of the supplement code, not extraction of a publisher-supplied prediction table. Project and supplement records were joined on ISO3 country code and domain. The common set contains 60 countries in each domain (180 country-domain rows). All matched rows had enough historical scores to produce the Becker–Coyle forecast. As a scale/target alignment check, the supplement's 2022 observed means differ from project target scores by at most 0.0371 points across these 180 rows. Metrics are calculated on that common set; lower MAE/RMSE is better.

| Domain | Becker–Coyle reconstruction MAE | RMSE | Persistence MAE | Ridge MAE | RF MAE | XGBoost MAE | Paired MAE difference vs Persistence |
|---|---:|---:|---:|---:|---:|---:|---:|
| Mathematics | 19.70 | 27.50 | 14.48 | 14.71 | 15.40 | 14.65 | +5.22 |
| Reading | 20.07 | 27.25 | 12.35 | 11.31 | 13.30 | 12.48 | +7.73 |
| Science | 18.53 | 29.54 | 8.83 | 9.82 | 9.01 | 9.21 | +9.70 |

In this common-sample retrospective comparison, the reconstructed forecasts have higher MAE than every listed project model in each domain. This is a descriptive comparison, not an independent validation: Becker–Coyle published the method after the forecast target was observed, and the project PISA 2022 fold is one historical development fold. Their source script's unusual R-squared F-test formula and p-value degrees of freedom were followed as published; the project `LinearQuadraticTrend` uses a standard nested-regression F test, so the two model-selection rules are not identical. The comparison is therefore of the published procedure's reconstruction with the project methods, not a like-for-like implementation of one identical test statistic.

Reproduction artifacts:

- [`compare_becker_coyle_2022.py`](../../scripts/compare_becker_coyle_2022.py)
- [`outputs/becker_coyle_comparison_v4/`](../../outputs/becker_coyle_comparison_v4/), including reconstructed forecasts, paired common-country comparisons, metrics, and a SHA-256 run manifest
- Supplement URL and ZIP/member hashes are in `run_manifest.json`; the source ZIP was not copied into the repository.

## Scopus/WoS search-record recovery

I rechecked the project search inventory, folder labels, Scopus processing log, and workspace files for database-native export or query-history formats (RIS, BibTeX, EndNote, export CSV, saved query, and search-history names). The only recoverable records remain the 725 enriched extraction JSONs (423 Scopus and 302 Web of Science), their inventory/hash manifest, and the Scopus extraction-processing log. The JSONs are not raw database result exports. No original query execution history, raw exports, query date, native search fields/filters, result hit totals, export completeness, deduplication log, or title/abstract/full-text screening decisions were located.

Accordingly, the 1,266-row coded corpus remains a fixed project corpus and must not be described as a reproducible, exhaustive Scopus/WoS search sample. No corpus rows or frequency denominators were changed. To establish systematic-search reproducibility, the original database exports/history and screening decisions must be obtained from the researcher or database accounts; missing history cannot be recreated retrospectively from article JSONs or folder labels.

## Remaining literature limitation

The numeric Becker–Coyle comparison is complete as a reconstruction. The database-search provenance remains unresolved. A second human reviewer is also still needed for the borderline-study eligibility coding; this computational audit is not a substitute for that review.
