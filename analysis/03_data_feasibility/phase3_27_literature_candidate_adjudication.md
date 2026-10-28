# Phase 3.27 — Literature Candidate Adjudication and Search-Scope Recovery

**Date:** 2026-10-20  
**Status:** Five candidate-level source reviews recorded; independent second review and original search-scope recovery remain open.  
**Locked corpus:** `analysis/02_prediction_taxonomy/study_classification.csv` (1,266 rows; unchanged)  
**Register:** `analysis/03_data_feasibility/audit/literature_candidate_adjudication_v1.csv`

## Purpose and limits

This phase follows the saved-search inventory audit (Phase 3.15) and targeted search (Phase
3.16). It resolves the two DOI records that were present in saved Scopus/WoS extraction JSONs
but absent from the fixed corpus, rechecks the external Becker–Coyle PISA forecast, records
an external PISA forecasting-method candidate and an adjacent proficiency-threshold projection
found in the supplemental search. It is a source-level adjudication of five records, not a new
systematic review or an estimate of literature-wide recall. The frozen corpus and algorithm
frequency outputs were not changed.

## Findings

| Record | Finding | Disposition for this project |
|---|---|---|
| Becker & Coyle (2026), DOI `10.3389/feduc.2026.1734988` | The publisher methods state that country-specific linear and quadratic trend regressions use PISA waves through 2018 to forecast 2022 scores; an F-test selects between trend forms and observed-minus-forecast deviations are analyzed. The publisher links supplement containing trend estimates and significance tests. | Keep as a verified external future-cycle country-score forecast precedent. Its membership under the original corpus search protocol remains unresolved. |
| Kaplan et al. (2025), DOI `10.1017/psy.2025.2` | Uses country-level PISA mathematics trajectories from 2009–2022, estimates country-level pace of progress with Bayesian growth models and stacking, and plots one-cycle-ahead 2025 mathematics trajectories for boys and girls. Predictive weighting for pace is assessed with leave-one-country-out validation. Published online 13 February 2025. | Keep as a strong external methodological forecast precedent, with a target mismatch: the displayed 2025 forecast is gender-stratified pooled mathematics performance, not our unstratified country-by-domain target. The article does not evaluate the 2025 prediction against observed 2025 results. Corpus eligibility remains unresolved. |
| Kaplan et al. (2025), arXiv:2511.06107 | Uses PISA 2009–2022 plus country indicators to build Bayesian latent growth and Bayesian model-averaged probabilistic projections of sex-specific country-level shares meeting minimum proficiency thresholds in reading and mathematics. | Keep as adjacent projection-method literature only. The outcome is a threshold percentage, not a mean score. The item is an arXiv preprint; journal status and original corpus eligibility remain unresolved. |
| Martín Ezpeleta et al. (2024), DOI `10.5209/rced.86027` | The publisher record and saved extraction JSON describe 438 Spanish teacher trainees completing creativity and metaphor-generation measures. PISA 2022 appears as background motivation; no PISA achievement data or forecast is analyzed. | Exclude from the ILSA forecasting corpus on study-scope grounds. |
| van Rijn & Ali (2023), DOI `10.1007/978-3-031-27781-8_11` | The publisher abstract describes a psychometric speed-accuracy response model, illustrated using listening-test data. No ILSA country score or future-cycle forecast is involved. | Exclude from the ILSA forecasting corpus on study-scope grounds. |

The two unmatched saved records appear under folders labeled PISA 2022 and PIRLS, respectively,
but those folder labels are query provenance hints, not proof that a record uses those assessment
data. They demonstrate why folder-level query labels cannot substitute for record-level eligibility
screening.

## Supplemental search update (2026-10-20)

A new, explicitly exploratory web search looked for conventional statistical or Bayesian
country-level PISA/TIMSS/PIRLS projections beyond the previously known Becker–Coyle paper. The
searches used these exact strings:

1. `PISA country level mean score forecast future cycle country-specific trend PISA 2022 forecast`
2. `TIMSS national country mean score forecast prediction next assessment cycle future cycle`
3. `PIRLS country mean achievement score forecast projection future assessment cycle`
4. `international large scale assessment country-level score forecasting PISA TIMSS PIRLS trend extrapolation`

The search surfaced the Kaplan et al. Psychometrika paper and a related arXiv preprint. The
Psychometrika publisher-hosted PDF states that
the study models country-level PISA mathematics proficiency over 2009–2022, predicts the pace of
progress, and presents a one-cycle-ahead forecast through PISA 2025. Predictive evaluation for
pace uses leave-one-country-out cross-validation; the 2025 score projection is displayed for boys
and girls, and the paper does not report verification against observed 2025 results. This is a
highly relevant methodological precedent, but its displayed forecast estimand is not identical
to a set of unstratified country-specific mean forecasts. Search results also surfaced OECD
descriptive trend tables, which are not themselves held-out forecasts. Search-engine results are
discovery leads only; this four-query search does not establish exhaustive coverage.

The preprint projects the country-level percentage at or above PISA minimum-proficiency
thresholds, separately by sex, for reading and mathematics, using PISA 2009–2022 and external
indicators. This is adjacent evidence, not the same outcome as mean-score forecasting. The
Psychometrika paper points to the authors' publication/resources page for data and analysis
materials. That page did not list the 2025 paper or a corresponding replication archive at audit
time; the article says its analysis code is available there but marks the specific stacking code
link as blinded for review. Thus a complete, directly downloadable replication package was not
verified in this pass. This does not establish that the authors cannot provide code on request.

## Original search protocol: what is recoverable

The local search inventory confirms 725 extraction JSON files (423 Scopus, 302 Web of Science),
665 distinct normalized DOIs across the two sources, 663 exact DOI matches to the corpus, and 332
corpus DOI records absent from those saved JSONs. The nominal folder names preserve a 2020–2026
range and broad terms for AI, ML, time series, prediction, or correlation combined with each ILSA
program. They do not preserve a verifiable database execution date, exact platform syntax and
field restrictions, result counts at execution, export completeness, deduplication log, or
record-level inclusion/exclusion decisions. The extraction JSONs are not raw database exports.

No local source located in this pass supplies the missing execution or screening history. Therefore
the original eligibility of either external study cannot be decided from DOI absence alone. Becker
and Coyle's June 2026 publication falls within the nominal folder year range; Kaplan et al. was
published in February 2025. Actual search timing, database query behavior, and project inclusion
criteria have not been recovered. Do not add either study to the 1,266 records or recalculate any
denominator until a versioned inclusion protocol is agreed and both candidates have an independent
second review.

## Consequences for current claims

1. The two unmatched DOI records do not represent missed qualifying ILSA forecasts; both are
   out of scope after source inspection.
2. Becker–Coyle is a verified external country-specific future-cycle PISA forecast. Kaplan et al.
   is an external one-cycle-ahead PISA forecast-method precedent with a different displayed
   estimand and no evaluation against observed 2025 results. A separate preprint projects
   threshold proficiency shares, not mean scores. Together these records rule out an
   unqualified claim that no relevant future-cycle PISA forecasting work exists.
3. The exact count and method frequencies in the 1,266-row corpus remain usable only as
   descriptions of that fixed coded corpus. Search completeness and the external studies'
   eligibility under the original protocol remain unresolved.
4. The preprint projects threshold percentages, not average scores, and is therefore adjacent
   evidence rather than a strict match. No second reviewer has independently adjudicated these
   five records yet. This file records
   the primary source review and preserves the second-review status as pending.

## Next actions

1. Request/recover the original Scopus and Web of Science exports, search-history screenshots or
   logs, execution dates, filters, and record-screening decisions. If unavailable, state that
   recovery failed and establish a new, date-stamped supplemental search protocol.
2. Have an independent reviewer screen the five rows in the CSV without seeing the primary
   dispositions; reconcile and record disagreements before using the candidate to amend corpus
   claims.
3. Run the separately versioned supplemental search using explicit per-source queries and
   preserved exports; screen against the strict outcome (country-level ILSA mean), forecast
   origin (only prior cycles available), and held-out later-cycle target criteria.
4. Only after eligibility rules and deduplication are fixed, decide whether to publish a new
   corpus release and recalculate method-frequency denominators.

## Sources inspected

- Becker & Coyle publisher full text, Methods / “Trends and forecasts”: 2022 PISA country means
  forecast from waves through 2018; model choice and forecast deviations described. The paper
  states the trend estimates and significance tests are in the linked [supplementary material](https://www.frontiersin.org/journals/education/articles/10.3389/feduc.2026.1734988/full).
- Kaplan et al. (2025) publisher-hosted [Psychometrika article PDF](https://www.cambridge.org/core/services/aop-cambridge-core/content/view/D85FD94C994DB13028209C06CA323169/S003331232500002Xa.pdf/stacking_models_of_growth_a_methodology_for_predicting_the_pace_of_progress_to_the_education_sustainable_development_targets_using_international_largescale_assessments.pdf): PISA 2009–2022 country-level mathematics data, LOO prediction of pace, and a one-cycle-ahead 2025 forecast shown by gender.
- Kaplan et al. (2025) [arXiv preprint](https://arxiv.org/abs/2511.06107): country-level probabilistic projections of PISA reading/mathematics minimum-proficiency percentages; adjacent to, but distinct from, mean-score forecasting.
- Authors' [BMER publication/resources page](https://bmer.wceruw.org/publications.html): checked for a downloadable replication archive for the Psychometrika paper; none was listed there at audit time.
- Martín Ezpeleta et al. publisher record: the abstract describes a teacher-trainee creativity
  study and identifies the PISA 2022 reference as background context.
- Springer chapter record and abstract for van Rijn & Ali: the model is illustrated on a listening
  test.
- Local saved-search inventory report and row-level inventory:
  `analysis/03_data_feasibility/audit/literature_search_inventory_v1/`.
