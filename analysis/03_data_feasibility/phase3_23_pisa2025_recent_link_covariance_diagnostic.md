# Phase 3.23 — Recent PISA 2025 link-covariance consistency diagnostic

**Date:** 2026-10-15  
**Status:** A common-offset covariance interpretation is feasible for math and reading, but fails a necessary Cauchy bound for science. No candidate covariance was used in model uncertainty results.  
**Script:** [`scripts/audit_pisa2025_recent_link_covariance_consistency.py`](../../scripts/audit_pisa2025_recent_link_covariance_consistency.py)  
**Outputs:** [`outputs/pisa2025_recent_link_covariance_diagnostic_v1/`](../../outputs/pisa2025_recent_link_covariance_diagnostic_v1/)

## Question

Can the OECD-published link-error SEs for the 2018→2025 and 2022→2025 comparisons be represented as two correlated, additive cycle-level score offsets whose covariance also reproduces the OECD link-error SE for the 2018/2022/2025 linear decennial trend?

This is a narrow consistency test for the two historical inputs used by that three-cycle trend. It is not an attempt to build the full covariance matrix needed for every historical PISA input to the forecasting models.

## Method and validation

The OECD 2025 performance tables give means by cycle and the linear decennial trend. Calendar-year OLS weights were verified independently: applying calendar-year weights to the OECD-35 science means for 2015, 2018, 2022 and 2025 reproduces the published trend (calculated −6.738629; published −6.738620; absolute difference 0.000009 score points per decade). Thus the comparison uses the same calendar-time convention as the published point trend.

For the three-cycle window, 2025 was chosen as the reference scale, so its link offset is zero. The link-error part of the trend has the form:

```text
Var(T) = w2018² Var(L2018) + w2022² Var(L2022)
         + 2 w2018 w2022 Cov(L2018, L2022)
```

The marginal SEs for 2018→2025 and 2022→2025 come from Phase 3.22’s country-level reconstruction of the OECD score and difference SEs. The trend-link SEs come from OECD Table I.A3.1 for the 2018, 2022 and 2025 combination. Solving the expression yields the covariance implied by the trend SE. Any valid covariance must satisfy `|Cov| ≤ SE2018 × SE2022` (Cauchy–Schwarz).

OECD states that cross-cycle 2025 standard errors include link uncertainty and that multi-cycle trend calculations account for link-error covariance. The test therefore checks whether these reported marginal and trend quantities can share the simple common-offset covariance representation used for a model-input perturbation.

## Results

| Domain | Link SE 2018→2025 | Link SE 2022→2025 | Trend-link SE, 2018/22/25 | Implied covariance | Maximum absolute covariance | Feasible trend-SE range | Feasible? |
|---|---:|---:|---:|---:|---:|---:|---|
| Mathematics | 2.551 | 1.220 | 3.652 | 2.662 | 3.112 | 3.627–3.957 | Yes |
| Reading | 1.832 | 1.094 | 2.617 | 1.467 | 2.004 | 2.575–2.871 | Yes |
| Science | 3.507 | 3.116 | 4.741 | 12.139 | 10.928 | 4.792–5.634 | **No** |

For science, the trend table implies covariance 12.139 score-points², exceeding the maximum possible 10.928. Equivalently, given the two marginal SEs, any covariance would yield a trend-link SE of at least 4.792, while Table I.A3.1 reports 4.741. The difference is about 0.051 points per decade, greater than the published three-decimal rounding margin. The analogous calculations are feasible for mathematics and reading.

## Interpretation

The result **does not establish an OECD table error**. It shows that the values currently available cannot all be represented as one additive two-offset covariance model for science under the verified calendar-year OLS weights. Possible explanations include a difference in the estimands or covariance definitions behind the pairwise and trend tables, or another part of the OECD trend simulation that is not recoverable from the marginal SE table alone. The exact construction must be checked against the PISA 2025 Technical Report's Chapter 15 and its underlying covariance/simulation inputs.

Do not use the covariance implied by each trend row as a production value: the science value violates Cauchy–Schwarz. Do not project it onto a positive-semidefinite matrix or substitute an independence assumption. The previous 20,000-draw uncertainty outputs remain conditional on their documented inputs and still omit link error.

## Next technical work

1. Compare the definitions, scale direction, link-error components and trend weights in OECD's PISA 2025 Chapter 15 and PISA 2022 Annex A7.
2. Recover or request the covariance/simulation inputs used to produce Table I.A3.1; reconcile the 2018/22/25 science entry before constructing any joint matrix.
3. Extend the consistency checks to the other trend windows and domains. Require positive semidefiniteness and reproduction of every reported trend-link SE, not only marginal SE agreement.
4. If no common covariance structure can reproduce the official values, report the model link-error sensitivity as unidentified from public marginal tables, or use transparent scenario bounds. Do not claim a fully link-adjusted interval.
5. Only after these checks pass, add link perturbations to a new conditional saved-prediction uncertainty run. Model-input resampling and model refitting remain a separate later uncertainty gate.

## Reproduction

The script reads the Phase 3.22 pairwise-link summary and the existing OECD workbook without modifying either. It verifies the OLS time convention against the published OECD-35 science point trend, calculates the implied covariance and Cauchy bound by domain, records hashes, and refuses to overwrite an existing output directory. The run verified the OLS point-trend check, found feasible covariance values for mathematics and reading, and rejected science. It did not estimate or apply a covariance matrix.

## Sources

- OECD PISA 2025 Annex A3, cross-cycle standard errors and link-error trend table reference: <https://www.oecd.org/en/publications/pisa-2025-results-volume-i_73451bc5-en/full-report/technical-notes-on-analyses-in-this-volume_46a7001c.html>
- OECD Table I.A3.1 StatLink: <https://stat.link/p7oc9v>
- OECD PISA 2022 Annex A7, multi-cycle link covariance and trend method: <https://www.oecd.org/en/publications/2023/12/pisa-2022-results-volume-i_76772a36/full-report/comparing-mathematics-reading-and-science-performance-across-pisa-assessments_1c417d6f.html>
- NCES PISA 2025 methodology notes, pairwise link-error method and values: <https://nces.ed.gov/surveys/pisa/pisa2025/technical-notes/index.asp>
- Local source workbook: [`outputs/pisa2025_oecd_mean_audit_v3/oecd_pisa2025_performance_tables.xlsx`](../../outputs/pisa2025_oecd_mean_audit_v3/oecd_pisa2025_performance_tables.xlsx).
