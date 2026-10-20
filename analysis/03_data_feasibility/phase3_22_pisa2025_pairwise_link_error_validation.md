# Phase 3.22 — PISA 2025 pairwise scale-link error reconstruction

**Date:** 2026-10-15
**Status:** Pairwise marginal link-error SEs reconstructed and externally matched; joint cross-cycle covariance remains open.
**Script:** [`scripts/audit_pisa2025_pairwise_link_errors.py`](../../scripts/audit_pisa2025_pairwise_link_errors.py)
**Outputs:** [`outputs/pisa2025_pairwise_link_error_v1/`](../../outputs/pisa2025_pairwise_link_error_v1/)

## Why this was needed

The Phase 3.20 uncertainty analysis propagated published country-score sampling SEs and country-cluster resampling, but omitted cross-cycle PISA scale-link uncertainty. Phase 3.21 had established that this uncertainty is distinct from sampling uncertainty and that a valid covariance matrix could not be obtained by mechanically combining selected historical pairwise values.

The OECD PISA 2025 Annex A3 says that standard errors and p-values for cross-cycle 2025 estimates account for link error and directs readers to the PISA 2025 Technical Report (Chapter 15, forthcoming) and PISA 2022 Annex A7 for methods. The OECD performance-tables workbook includes each previous cycle's score SE, the 2025 score SE, and the SE of the 2025-minus-previous score difference for all three domains. The OECD page for PISA data and methodology currently identifies the downloadable 2025 Technical Report file as draft chapters and says the final report is due in 2027.

## Method

For every country/economy with all three standard errors, the pairwise link-error SE was reconstructed as:

```text
link_SE(2025, t) = sqrt(SE(2025 - t)^2 - SE(t)^2 - SE(2025)^2)
```

This decomposition treats the cycle-specific score-estimation errors as independent across PISA samples and the linking error as a separate variance component. The OECD's technical notes state that published cross-cycle 2025 standard errors include link uncertainty. The result was checked for constancy across country/economy rows for a given domain and comparison, then matched against Table 4 of the US National Center for Education Statistics PISA 2025 technical notes. The NCES notes also describe how links from pre-2022 cycles to 2025 are composed from the 2022–2025 and earlier-cycle–2022 links, treating those two components independently.

## Results

The workbook yielded 1,280 country/economy-level reconstructions across 21 domain-by-cycle comparisons. For each comparison, reconstructed values varied by no more than 0.00004 points across systems, and all 21 median values matched the NCES published values rounded to three decimals.

| Previous cycle | Mathematics | Reading | Science |
|---:|---:|---:|---:|
| 2000 | — | 6.759 | — |
| 2003 | 5.673 | 5.363 | — |
| 2006 | 4.268 | 8.630 | 4.822 |
| 2009 | 4.450 | 4.787 | 6.690 |
| 2012 | 3.782 | 6.109 | 6.062 |
| 2015 | 2.999 | 3.791 | 3.408 |
| 2018 | 2.551 | 1.832 | 3.507 |
| 2022 | **1.220** | **1.094** | **3.116** |

All values are score-point standard errors. A dash means the comparison is not available because the domain was not yet a major PISA domain.

The more precise 2022–2025 science link-error SE recovered from the published comparison table is 3.116. This differs from the rounded 2.9-point example cited in the OECD Reader's Guide and in the earlier Phase 3.19 note. The table-based reconstruction matches the separate NCES technical note to three decimals; future reporting should cite the exact tabular value and document the Reader's Guide discrepancy rather than silently mixing the two figures.

## What this closes and what it does not

This resolves the **marginal pairwise link-error SE inventory for comparisons of each earlier cycle against 2025**. It does not provide the covariance among link errors for multiple historical cycles used together by a model. It is therefore not yet valid to inject independent draws for each historical cycle, assume one common offset, or describe the current model-ranking intervals as fully link-error-adjusted.

The OECD 2022 Annex A7 describes multi-cycle trend-link uncertainty as a multivariate normal simulation whose variance/covariance structure includes earlier reporting-scale links. The available 2025 pairwise table gives marginal SEs, while the joint historical covariance needed to perturb a multi-cycle model input vector remains to be verified from the cited technical-report material or direct re-scaling inputs. The previous Phase 3.21 attempt yielded non-positive-semidefinite reading/science covariance matrices under a simple variogram reconstruction; those matrices remain rejected and were not used here.

## Next technical steps

1. Recover the exact 2012/2015/2018/2022 link-error covariance inputs or simulation specification referenced by OECD Annex A7. Do not infer covariance from marginal SEs alone.
2. Combine that historical covariance with the 2022–2025 link component, which the NCES 2025 notes treat as independent of links between 2022 and earlier cycles.
3. Require symmetry, positive semidefiniteness, matching published marginal variances, and reproduction of OECD's reported multi-cycle trend-link SEs before using a candidate matrix.
4. If a fully supported covariance matrix cannot be recovered, keep link uncertainty separate and report scenario/bounds analyses rather than claiming a joint uncertainty interval.
5. Once validated, add link uncertainty to a new version of the saved-prediction country-bootstrap/target-SE analysis. Keep this conditional analysis distinct from the later full model-refit/input-uncertainty analysis.

## Reproduction checks

`audit_pisa2025_pairwise_link_errors.py` reads the existing OECD workbook without modifying it and refuses to overwrite a pre-existing output directory. It writes country-level reconstructions, a comparison summary, and a manifest with input/script hashes. The recorded checks passed:

- 21 of 21 domain-cycle comparisons were reconstructed.
- All values were country/economy-invariant within the stated tolerance.
- All values matched the NCES reference rounded to three decimals.
- No covariance matrix was estimated.

## Sources

- OECD PISA 2025 technical notes, Annex A3, section “Standard errors, confidence intervals, significance test and p-values”: <https://www.oecd.org/en/publications/pisa-2025-results-volume-i_73451bc5-en/full-report/technical-notes-on-analyses-in-this-volume_46a7001c.html>
- OECD PISA data and methodology, report availability note (draft 2025 technical-report chapters; final report expected 2027): <https://www.oecd.org/en/about/programmes/pisa/pisa-data.html>
- NCES, PISA 2025 Methodology and Technical Notes, link-error explanation and Table 4: <https://nces.ed.gov/surveys/pisa/pisa2025/technical-notes/index.asp>
- OECD PISA 2022 Results (Volume I), Annex A7, multi-cycle link-error covariance method: <https://www.oecd.org/en/publications/2023/12/pisa-2022-results-volume-i_76772a36/full-report/comparing-mathematics-reading-and-science-performance-across-pisa-assessments_1c417d6f.html>
- Local OECD workbook and provenance: [`outputs/pisa2025_oecd_mean_audit_v3/oecd_pisa2025_performance_tables.xlsx`](../../outputs/pisa2025_oecd_mean_audit_v3/oecd_pisa2025_performance_tables.xlsx), [`outputs/pisa2025_oecd_mean_audit_v3/audit_manifest.json`](../../outputs/pisa2025_oecd_mean_audit_v3/audit_manifest.json).
