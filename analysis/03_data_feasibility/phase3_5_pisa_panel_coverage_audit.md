# Phase 3.5 — PISA Panel Coverage and Country-Key Audit

**Date:** 2026-10-15  
**Input:** `outputs/stage5/unified_panel.csv` (read-only)  
**Status:** Coverage verified; canonical panel and historical fold manifest generated  
**Integrity:** No source panel or existing result was modified.

---

## 1. Cycle Coverage in the Current Panel

Counts below are non-missing country scores in the current panel. PISA 2025 is shown for inventory only; it has been outcome-exposed in this project and is not eligible as a blind final set.

| PISA domain | 2000 | 2003 | 2006 | 2009 | 2012 | 2015 | 2018 | 2022 | 2025 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Mathematics | — | 41 | 57 | 74 | 65 | 73 | 79 | 80 | 89 |
| Reading | 43 | 41 | 56 | 74 | 65 | 73 | 79 | 80 | 89 |
| Science | — | 41 | 57 | 74 | 65 | 73 | 79 | 80 | 90 |

The current panel begins mathematics and science in 2003. Reading includes 2000, but its country key is not harmonized with later rounds.

## 2. Country-Identifier Finding

PISA 2000 reading rows store `country_iso3` as numeric ISO 3166-1 identifiers (`008`, `032`, `036`, etc.). Later cycles use alpha-3 values (`ALB`, `ARG`, `AUS`, etc.). The current panel therefore treats the same country as different identifiers across 2000 and later years.

The upstream cause is visible in `scripts/build_pisa_legacy_estimates.py`: `cnt_to_iso3()` handles a small alpha-code exception map but passes numeric PISA 2000 `COUNTRY` values through unchanged. `scripts/build_unified_panel.py` then preserves those values when it constructs the panel.

The source labels were checked in the original `PISA2000_SPSS_student_reading.txt` under `VALUE LABELS COUNTRY`; they identify the country for every numeric value. Read-only crosswalk check with the installed `pycountry` ISO numeric table:

- All 43 numeric identifiers map to an alpha-3 identifier and match the corresponding SPSS country label. `807` is labeled `MACEDONIA` in the historical source and maps to current alpha-3 code `MKD`.
- 42 of the mapped 2000 identifiers occur in the 2009 reading target set.
- Re-keying in memory (without editing the panel) changes the eligible count for the three-pre-cycle reading forecast from 0 to 34 countries for the 2009 target.

The 2009 result is a diagnostic of key harmonization, not a model result. The verified 43-row mapping is recorded in `analysis/03_data_feasibility/audit/pisa_2000_reading_country_crosswalk.csv`. The main panel remains unchanged. Any corrected panel must be generated as a separate versioned artifact with collision checks.

The verified crosswalk has been applied to a separate model panel at `outputs/stage5/pisa_forecasting_panel_v2.csv`; all original identifiers are retained in `country_iso3_source`. A hash manifest is stored alongside the panel. The 43-row mapping is also added to `scripts/build_pisa_legacy_estimates.py` for future legacy-estimate builds; that production estimator was not run, so existing estimates remain untouched.

## 3. Historical-Fold Eligibility

Eligible means the country has a non-missing score in the target cycle and at least three earlier non-missing cycles. Counts below apply the numeric-to-alpha-3 mapping in memory for PISA 2000 reading only.

| Target cycle | Math eligible | Reading eligible | Science eligible | Common across all three domains |
|---|---:|---:|---:|---:|
| 2009 | 0 | 34 | 0 | 0 |
| 2012 | 40 | 45 | 40 | 40 |
| 2015 | 51 | 53 | 51 | 51 |
| 2018 | 57 | 57 | 57 | 57 |
| 2022 | 60 | 61 | 60 | 60 |
| 2025 | 66 | 66 | 66 | 66 |

This supports four common historical evaluation folds (2012, 2015, 2018, 2022), with a fixed three-domain country set per fold. 2009 can be included as a reading-only supplemental fold, but not in the balanced three-domain comparison under the same minimum-context rule.

## 4. Panel-Shape and Modeling Implications

- The source is long-form: the correct observation key includes at least `program`, `domain`, `country_iso3`, and `cycle`. Country-cycle alone repeats across PISA domains and is not a unique key.
- Apply the country crosswalk before forming country time series, with a collision check to ensure no two source rows map to the same canonical country-domain-cycle key.
- Preserve original country identifiers and add a canonical key in a derived panel; never overwrite the source identifier.
- Keep historical fold eligibility explicit. Do not silently let a model use fewer than the registered three pre-target observations.
- For the balanced three-domain comparison, use common country IDs within each target fold; also report domain-specific coverage as a secondary analysis.
- The 2025 target has 66 eligible countries per domain under current panel identifiers, but it is an exposed retrospective benchmark, not a new blind test.

## 5. Next Required Work

1. Review the candidate feature arm against the fold-specific availability recorded in `phase3_6_covariate_feasibility.md`.
2. Use the frozen country sets and cutoffs in `analysis/03_data_feasibility/audit/pisa_historical_folds_v1.json` for historical development and temporal validation.
3. Use 2012, 2015, 2018, and 2022 for comparative development/evaluation; keep 2025 excluded from tuning and disclose its outcome exposure.
