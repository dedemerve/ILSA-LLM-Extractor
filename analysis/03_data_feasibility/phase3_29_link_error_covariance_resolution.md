# Phase 3.29 — PISA link-error covariance: method and remaining identification gap

**Date:** 2026-10-20  
**Status:** Official multi-cycle method identified; table references now reconciled against published table titles; no defensible full historical covariance matrix identified or applied.
**Scope:** Prompt 2. Existing analysis files, model predictions, and uncertainty outputs were not modified.

## Executive finding

- **VERIFIED:** OECD describes multi-cycle trend link uncertainty as a multivariate-normal simulation with 2,000 draws. The covariance structure is not an independence assumption and is not a single common-offset model. It combines the current cycle-pair link-error information with covariances among earlier reporting scales.
- **VERIFIED:** The PISA 2022 Technical Report is published. The relevant link/scaling material is in Chapter 11 (method) and Chapter 14 (scaling outcomes/linking results); Chapter 15 is not the linking chapter. The multi-cycle trend procedure is described directly in *PISA 2022 Results, Volume I, Annex A7*.
- **VERIFIED:** The Phase 3.23 science Cauchy–Schwarz failure rejects the particular two-additive-offset representation tested there. It does not, by itself, identify which OECD quantity or modeling assumption differs, and does not establish an OECD table error.
- **UNRESOLVED:** The exact, complete, machine-usable covariance input needed to jointly perturb every historical PISA score series used by this project has not been reconstructed. Several historical-table citations in OECD results annexes do not appear to identify covariance tables consistently when checked against the technical-report contents.
- **Decision:** Do not add link-error draws to the project’s model uncertainty intervals yet. Keep those intervals explicitly conditional on the saved linked scores and state that PISA scale-link uncertainty is omitted.

## Sources checked and chapter correction

The PISA 2022 Technical Report is available from OECD and is no longer forthcoming. Its table of contents places scaling methodology in Chapter 11 and scaling outcomes, including link-error estimates, in Chapter 14. Chapter 15 concerns database material. The project prompt’s reference to “PISA 2022 Technical Report Chapter 15” is therefore not the correct location for the multi-cycle trend method.

The operative published description is OECD, *PISA 2022 Results, Volume I*, Annex A7. OECD says the multi-cycle trend link error is estimated by simulating 2,000 multivariate-normal draws with zero means. The stated variance/covariance structure combines the current pairwise link errors with historical between-reporting-scale covariance information. Thus it is neither independent errors for all cycles nor one common offset applied to all cycles. OECD also describes the simulations as scenarios where the real trend is zero and the estimated trend arises from scale comparability uncertainty.

**Source-table check (reconciled):** PISA 2012 Technical Report Table 12.31 is titled “Link error estimates”; Table 12.32 is “Link error covariances” and publishes selected domain-specific covariance entries. PISA 2015 Technical Report Table 12.8 is titled “Robust link error ... for comparisons of performance between PISA 2015 and previous assessments.” OECD PISA 2018 Results Annex A7 nevertheless says its multi-cycle simulation covariance structure is identified by PISA 2018 Table I.A7.1 and “those between previous PISA reporting scales” published in PISA 2012 Table 12.31 and PISA 2015 Table 12.8. The cited table titles do not themselves identify covariance matrices. This is now a verified citation/table-title mismatch in the explanatory text, rather than uncertainty about what those tables contain. PISA 2012 Table 12.32 supplies selected covariance entries, but not an evident complete covariance matrix for all domains and cycles in this project. The official 2018 Results Annex also provides 2,000-draw trend-link summaries, which are useful checks but do not expose the full simulation draws or all cross-cycle covariances. The OECD-linked 2018 technical-report ZIP returned HTTP 403 during this follow-up, so it could not supply further machine-usable details.

The current OECD PISA data-methodology page says that the downloadable PISA 2025 Technical Report contains draft chapters and that the final report is due in 2027. PISA 2025 Annex A3 points to the final technical-report Chapter 15 for details and to PISA 2022 Results Annex A7 for the established multi-cycle trend method. Accordingly, the full 2025 implementation is still not publicly verifiable from a final technical report as of this audit.

## Reassessment of the science Cauchy–Schwarz result

Phase 3.23 used the verified calendar-year OLS weights for the 2018/2022/2025 trend and the published marginal link-error SEs for 2018→2025 and 2022→2025. Under its specific representation—two additive random cycle-level offsets relative to 2025, with those exact SEs as the offsets’ marginal standard deviations—the implied science covariance is 12.139 score-points², while Cauchy–Schwarz permits at most 10.928. The smallest trend-link SE attainable under that representation is 4.792; the OECD table reports 4.741. Mathematics and reading satisfy the same bound.

**VERIFIED interpretation:** the published values cannot all be reproduced by that particular two-offset model with the Phase 3.23 weights and marginal inputs. **UNRESOLVED interpretation:** the public sources inspected do not reveal which detailed link-error definition, decomposition, scale direction, or simulation input accounts for the difference. The official method uses a richer multivariate covariance structure, but the Cauchy failure alone does not prove that “richer covariance” is the explanation: for two fixed variables with those exact marginals and weights, no admissible covariance can produce the smaller reported science SE. At least one element of that two-variable specification is not aligned with the official trend calculation, or the public rounded summaries are not sufficient to identify the underlying quantities. The roughly 0.051-point gap is larger than the three-decimal rounding allowance reported in Phase 3.23.

This finding is not evidence that the OECD value is erroneous. No covariance has been forced to fit by clipping, positive-semidefinite projection, or an independence assumption.

## What is available and what is still needed

**VERIFIED available inputs:**

1. Phase 3.22 reconstructs 21 marginal pairwise link-error SEs for 1,280 country/economy rows from the OECD score and difference SEs. All 21 summaries match the NCES reference table to three decimals. The audit reruns byte-identically in `outputs/reproducibility_check_v1/`.
2. OECD 2012 Technical Report Table 12.32 publishes selected historical link-error covariances. It is not, by itself, an evident complete matrix for all model input cycles and domains.
3. NCES PISA 2025 notes describe composing pre-2022-to-2025 errors from the 2022–2025 component and the earlier-to-2022 component under an independence treatment. This supports a marginal comparison construction; it does not identify all cross-covariances among the historical inputs used jointly by a model.
4. OECD 2022 Results Annex A7 provides the multi-cycle simulation principle and trend-level link-error outputs. These outputs are useful validation targets but do not expose a complete covariance matrix for the project’s full input vector.

**Needed before a project covariance can be used:**

1. Obtain the underlying covariance definitions and inputs from OECD’s cited 2012, 2015 and 2018 materials, including an accessible copy of the 2018 technical-report material; request clarification from OECD if the Annex A7 table references are editorial errors or refer to transformed data not described in those tables.
2. Establish for every entry its domain, pair of cycle links, reference scale/direction, variance versus covariance status, score metric, and whether it is an estimate or a transformed/simulation quantity.
3. Build the full covariance for exactly the link-error variables in the project’s historical feature vector. Do not infer off-diagonal entries from marginal SEs alone. Identify how 2022–2025 linking enters and verify any independence assumption against official definitions.
4. Check symmetry, diagonal variances, positive semidefiniteness, and correct transformation to the chosen reference scale.
5. Reproduce OECD-published pairwise and multi-cycle trend link-error values for all available domains/windows, including the 2018/2022/2025 science diagnostic, within documented rounding tolerance.
6. Only after these gates pass, propagate draws through a new, separately versioned uncertainty run. Keep link uncertainty distinct from target-score sampling uncertainty, country-cluster uncertainty, and model-input/refitting uncertainty.

## Recommended limitation language for current reports

> The reported uncertainty analysis propagates the documented country-level target-score sampling uncertainty and country-cluster resampling, conditional on the observed PISA cycle scores used as model inputs. It does not include uncertainty from linking PISA scales across cycles. Although OECD describes a multivariate simulation for multi-cycle trend-link uncertainty, the complete historical covariance inputs required for the full project feature vector could not be recovered and validated from the inspected published tables. We therefore did not assume independent cycle-link errors, impose a common-offset structure, or repair an infeasible covariance matrix. The resulting intervals are not fully link-error-adjusted.

The Phase 3.23 science finding may be added as a technical note: the published marginal and trend summaries failed a necessary Cauchy bound under the tested two-offset representation, with the reason for the discrepancy unresolved. It should not be stated as an OECD inconsistency.

## Source trail

- OECD, [PISA 2022 Results, Volume I, Annex A7](https://www.oecd.org/en/publications/2023/12/pisa-2022-results-volume-i_76772a36/full-report/comparing-mathematics-reading-and-science-performance-across-pisa-assessments_1c417d6f.html) — multi-cycle trend simulation and described historical inputs.
- OECD, [PISA 2022 Technical Report](https://www.oecd.org/en/publications/pisa-2022-technical-report_01820d6d-en.html) — published report; see Chapter 11 and Chapter 14. The direct PDF is [here](https://www.oecd.org/content/dam/oecd/en/publications/reports/2024/03/pisa-2022-technical-report_599753f0/01820d6d-en.pdf).
- OECD, [PISA 2012 Technical Report, Table 12.31 and Table 12.32](https://www.oecd.org/content/dam/oecd/en/about/programmes/edu/pisa/publications/technical-report/PISA-2012-technical-report-final.pdf) — pairwise error estimates versus selected link-error covariances.
- OECD, [PISA 2015 Technical Report, Table 12.8](https://www.oecd.org/content/dam/oecd/en/about/programmes/edu/pisa/publications/technical-report/PISA2015_TechRep_Final.pdf) — robust pairwise link-error estimates.
- OECD, [PISA 2018 Results Volume I, Annex A7](https://www.oecd.org/en/publications/pisa-2018-results-volume-i_5f07c754-en/full-report/component-24.html) — describes the 2,000-draw multivariate-normal trend-link simulation and cites prior-cycle tables whose titles do not match the description as covariance tables.
- OECD, [PISA 2015 Technical Report, Table 12.8](https://www.oecd.org/content/dam/oecd/en/about/programmes/edu/pisa/publications/technical-report/PISA2015_TechRep_Final.pdf) — table caption identifies robust pairwise link errors.
- OECD, [PISA 2018 Results, Volume I, Annex A7](https://www.oecd.org/en/publications/pisa-2018-results-volume-i_5f07c754-en/full-report/component-24.html) — multivariate-normal method and source-table references.
- OECD, [PISA 2025 Results, Volume I, Annex A3](https://www.oecd.org/en/publications/pisa-2025-results-volume-i_73451bc5-en/full-report/technical-notes-on-analyses-in-this-volume_46a7001c.html) — 2025 pairwise/trend-link notes and reference to the pending technical report.
- OECD, [PISA data and methodology](https://www.oecd.org/en/about/programmes/pisa/pisa-data.html) — status of the draft 2025 technical-report chapters and 2027 final-report schedule.
- NCES, [PISA 2025 Methodology and Technical Notes](https://nces.ed.gov/surveys/pisa/pisa2025/technical-notes/index.asp) — 2025 link-error notes and table.
- Local validated evidence: [`phase3_22_pisa2025_pairwise_link_error_validation.md`](phase3_22_pisa2025_pairwise_link_error_validation.md), [`phase3_23_pisa2025_recent_link_covariance_diagnostic.md`](phase3_23_pisa2025_recent_link_covariance_diagnostic.md), and `outputs/pisa2025_pairwise_link_error_v1/`.
