# Phase 3.16 — Targeted Country-Level ILSA Forecast Candidate Search (Interim)

**Date:** 2026-10-15  
**Status:** Exploratory targeted search completed; not an exhaustive systematic update.  
**Purpose:** Identify country-level achievement-mean prediction/forecast candidates missed by the 28-row cross-cycle heuristic, including conventional statistical methods.

## Search scope and reproducibility note

An exploratory web search was run on 2026-10-15 using combinations of:

- PISA / TIMSS / PIRLS / ILSA program names;
- country mean, national score, country-level performance;
- forecasting, forecast, projection, extrapolation, future cycle, and time-series prediction.

Queries included:

1. `PISA country mean score forecasting country-specific trend 2022 forecast national means PISA`
2. `TIMSS country mean score forecast projection future cycle country-level assessment score forecast`
3. `PIRLS country mean achievement score projection forecast assessment cycle regression trend`
4. `international large scale assessment PISA forecasting country performance national scores time series prediction country level`
5. Follow-up searches for the OECD 2010 PISA–TIMSS regression paper and for PISA country-level held-out/future-cycle forecasts.

This search was exploratory and search-engine based. It did not have a preregistered protocol, database export, independent screening, or a validated recall denominator; it cannot support an exhaustive count.

## Candidate dispositions

| Candidate | Evidence found | Strict project disposition | Corpus status |
|---|---|---|---|
| Becker & Coyle (2026), DOI `10.3389/feduc.2026.1734988` | Country-specific linear/quadratic models fit PISA country means through 2018 and forecast PISA 2022; target-cycle deviations are then calculated. | **Relevant historical future-cycle forecast candidate.** Ex-post publication does not make its temporal training/target split prospective in real time, but it meets the project's operational historical holdout definition as reported in the paper. Inspect complete methods and inclusion scope before any corpus update. | DOI absent from the fixed 1,266 records and from the inventoried Scopus/WoS JSON set. Publisher full text: https://www.frontiersin.org/journals/education/articles/10.3389/feduc.2026.1734988/full |
| Wu & Stacey / OECD (2010), *Comparing the Similarities and Differences of PISA 2003 and TIMSS*; DOI `10.1787/5km4psnm13nx-en` | Regressions predict PISA 2003 country mathematics means using same-period TIMSS and related predictors. The report states the purpose is to explain PISA–TIMSS differences, not predict future PISA scores. | **Country-level score-prediction precedent, not future-cycle forecast.** Same-cycle outcome reconstruction/comparison must not be counted as held-out forecasting. | Present in the 1,266-row classifier table; coded as a technical report/non-predictive analytical study. OECD publication: https://www.oecd.org/en/publications/comparing-the-similarities-and-differences-of-pisa-2003-and-timss_5km4psnm13nx-en.html |
| OECD PISA 2022 Volume I trend calculations | OECD fits linear regression over a country's available cycles for average decennial trends, and quadratic regression for curvilinear trends; curvilinear trends are limited to countries with comparable data in more than four assessments. | **Descriptive trend-estimation precedent, not a clean held-out future-cycle forecast.** A regression trend can serve as a baseline, but the published trend estimate uses observed data through its reporting endpoint. | Official methods section: https://www.oecd.org/en/publications/2023/12/pisa-2022-results-volume-i_76772a36/full-report/comparing-mathematics-reading-and-science-performance-across-pisa-assessments_1c417d6f.html |

## Interim methodological conclusions

1. “Country-level prediction” is broader than forecasting a future assessment. A same-cycle PISA score reconstructed from TIMSS or covariates is not evidence of out-of-time performance.
2. Published PISA trend estimates provide a relevant conventional baseline and account for sampling/measurement uncertainty and scale-link error in the official methodology. Our own country-level trend benchmark should be described as a forecasting model only when it is refit using data strictly earlier than each held-out target cycle.
3. The exploratory search found a second, in-corpus same-cycle prediction precedent and confirms that simple algorithm-frequency rankings are not sufficient to classify forecasting evidence.
4. The Becker–Coyle paper remains an important out-of-corpus future-cycle forecast candidate. Its eligibility under the original corpus search criteria remains unknown until that protocol is recovered.
5. No exhaustive literature-wide zero/non-zero conclusion is established by this exploratory search. The defensible statement is that a specific future-cycle PISA country-mean forecast has been independently verified outside the coded corpus and that the corpus/search coverage question remains open.

## Next required literature actions

1. Add the three candidates above to a versioned supplemental screening register with reviewer, date, source inspected, decision, and rationale.
2. Retrieve the full Becker–Coyle text and supplementary materials; record forecast origin, model selection, target construction, country coverage, and exact inclusion/exclusion decision.
3. Expand search to OECD iLibrary, ERIC, Crossref/OpenAlex, and IEA repositories using explicit per-source queries and date-stamped exports; include older publication years because forecasting precedent may predate the project's nominal 2020–2026 directory labels.
4. Search related terms for “nowcasting”, “trend projection”, “score projection”, “national performance”, and “assessment results” and screen their outcomes against the strict target definition.
5. Independently screen records outside the original 28 flags, especially trend/forecast terms, country-system targets, and traditional-statistics classifications.
6. Only after eligibility rules are reconciled, issue a separately versioned corpus release and recalculate method counts.
