# Phase 3.12 — Independent OECD Validation of PISA 2025 Country Means

**Date:** 2026-10-15  
**Status:** PASS for the 66-country, three-domain evaluation sample  
**Scope:** Point estimates only; this does not validate all PISA 2025 countries, standard errors in all domains, or remove outcome exposure.

## Source and reproducibility

The OECD's PISA 2025 Results Volume I StatLink workbook was downloaded from [StatLink mrq53f](https://stat.link/mrq53f). It contains the published country/economy tables for mean science, reading, and mathematics performance (Tables I.B1.2a.1, I.B1.2a.2, and I.B1.2a.3; listed on the [OECD country results page](https://www.oecd.org/en/publications/pisa-2025-results-volume-i_73451bc5-en/full-report/results-for-countries-and-economies_33854635.html)). The exact source workbook is archived with this audit at `outputs/pisa2025_oecd_mean_audit_v2/oecd_pisa2025_performance_tables.xlsx`.

The audit compares the unrounded country means in the hash-locked `pisa_forecasting_panel_v2.csv` with the unrounded OECD StatLink values. Country labels are mapped using the project's PISA 2025 country-name file plus five explicit panel-code overrides (CZE, HKG, MAC, QAT, TUR). No country was silently dropped. The evaluation's `true_score` values were also checked against the panel before the OECD comparison.

Reproduction command:

```bash
python3 scripts/audit_pisa2025_oecd_means.py \
  --workbook outputs/pisa2025_oecd_mean_audit_v3/oecd_pisa2025_performance_tables.xlsx \
  --output outputs/pisa2025_oecd_mean_audit_reproduction
```

The script refuses to overwrite an existing output directory. Machine-readable country results, unmatched-key file, source workbook, hashes, and software version are stored in the audit directory. Its output is also summarized in `outputs/pisa2025_oecd_mean_audit_v3/audit_manifest.json`.

## Results

Difference is project panel mean minus OECD unrounded mean, in score points. MAE/RMSE here describe source-estimate agreement, not model forecast error.

| Domain | Matched countries | Mean signed difference | MAE difference | RMSE difference | Maximum absolute difference | Within 0.5 points |
|---|---:|---:|---:|---:|---:|---:|
| Mathematics | 66 | −0.000051 | 0.000072 | 0.000397 | 0.00322 | 66/66 |
| Reading | 66 | +0.000177 | 0.000202 | 0.001422 | 0.01155 | 66/66 |
| Science | 66 | +0.000048 | 0.000079 | 0.000431 | 0.00349 | 66/66 |

All 198 country-domain keys matched; the unmatched-key file has zero rows. The evaluation CSV's target values match the canonical panel exactly (maximum absolute difference 0.0). The largest difference against the OECD's unrounded workbook values is 0.01155 points in reading, consistent with precision/rounding differences and negligible relative to PISA score units.

## Decision and limits

The country-level PISA 2025 **point means used to score the 66-country retrospective model comparison are independently corroborated** against the official OECD results tables. The previous reports' statement that direct OECD mean validation was blocked is superseded by this audit.

This validation does not establish provenance for every step that created the panel, validate all 89/90 countries outside the evaluation sample, or independently validate reading/science standard errors. The existing mathematics standard-error comparison remains separate evidence. PISA 2025 also remains outcome-exposed in this project; external source agreement improves outcome validity but does not turn the benchmark into a blind test.

The prediction metrics in the Phase 3.10 report do not need recalculation because the audited target values match the evaluation targets exactly. Continue to describe the study as a country-level, outcome-exposed retrospective comparison.
