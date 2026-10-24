# Phase 3.33 — Research-objective and evidence-alignment audit

**Date:** 2026-10-24  
**Status:** Local evidence audit complete; external and prospective validation gates remain open.  
**Scope:** Check whether the executed corpus, PISA forecasts, TimesFM adaptation, uncertainty analyses, reproducibility evidence, and final claims match the research objective described by the researcher. No source panel, frozen fold, forecast, or existing output was overwritten.

## 1. Audit verdict

The executed experiments substantially match a carefully scoped version of the research objective: forecast **country/economy-level PISA mean scores** in mathematics, reading, and science one assessment cycle ahead; compare simple baselines, tabular methods, and TimesFM; and test whether ILSA adaptation helps. The work does **not** estimate individual student achievement.

The main methodological distinction is between target leakage and outcome exposure. The 2025 forecast code uses only cycles before 2025 for contexts and training targets, so the inspected scripts do not feed the 2025 target score to the models. However, 2025 outcomes and earlier model results were already accessible to the research team. The 2025 comparison is therefore a valid **outcome-exposed retrospective comparison**, not a blind independent test.

The strongest result is conditional, not universal: Persistence had the lowest equal-fold historical MAE in all three domains, while the 2025 point-MAE leader differed by domain (XGBoost in mathematics, Ridge in reading, Persistence in science). The evidence does not establish that Persistence will win in a future cycle or that complex models never help.

## 2. Research question and falsification frame

**Audit question:** On the frozen, eligible country/economy samples, do the candidate methods improve one-cycle-ahead forecasts of PISA domain means over Persistence, and does ILSA adaptation improve TimesFM?

| Evidence pattern | Warranted conclusion |
|---|---|
| A complex method beats Persistence across the registered historical folds and domains, with paired uncertainty supporting the difference | Historical evidence of an advantage under this protocol; still not proof of future-cycle superiority. |
| A method leads only in one domain or the exposed 2025 comparison | Domain- and sample-specific retrospective finding. |
| A method fails to improve on Persistence in the historical folds | No evidence in these data that added complexity improved the tested forecasts. |
| A future target is unavailable, exposed before protocol lock, or cannot be scored on a comparable sample | Independent prospective superiority is unanswerable from that evaluation. |

This frame is an audit interpretation, not a preregistration. The project has a prepared PISA 2029 lock protocol, but its date, access controls, hashes, and sealed predictions are not yet complete.

## 3. Data and target checks

**What data support the comparisons, and are they aligned to the target?** The audited panel contains PISA cycles 2000–2025, with mathematics and science beginning in 2003 and reading including 2000 after the numeric-to-ISO3 crosswalk. The primary historical comparison uses the frozen, common three-domain samples: 40 systems in 2012, 51 in 2015, 57 in 2018, and 60 in 2022. The outcome-exposed 2025 comparison uses 66 common systems in each domain. PISA 2025 project target means match the OECD source for all 198 country/domain rows in the evaluation panel; this confirms target-score alignment for those rows, not that 2025 was blind.

The estimand is a country/economy mean score. Student-level predictions, a census of all countries, and causal effects are outside the evidence. A 2009 reading-only fold is possible after the crosswalk, but it is not part of the balanced three-domain primary comparison because mathematics and science do not meet the same eligibility rule.

## 4. Objective-to-evidence traceability

| Research objective or audit criterion | Executed evidence and validation | Status | Interpretation / remaining condition |
|---|---|---|---|
| Identify methods used in ILSA forecasting literature | The 1,266-row coded corpus has RF 42 and XGBoost 17 among 147 predictive studies with a coded primary method. In the narrower 17 coded country-level prediction records, SVM leads with 4, followed by RF 3 and XGBoost 2. | **Partly aligned** | RF/XGBoost are defensible general predictive comparators, not proven to be the two most common methods in verified future-cycle country-level ILSA forecasting. Original Scopus/WoS query, export, and screening provenance is missing; corpus completeness is unverified. |
| Forecast future PISA math, reading, and science performance | The experiment forecasts country/domain/cycle means, not student scores. The 2025 target panel contains 66 systems per domain and all three domains. | **Aligned after estimand clarification** | Use “country/economy mean score” consistently in the research question, tables, and conclusions. |
| Use PISA history from 2000–2022 | Historical rolling-origin targets are 2012, 2015, 2018, and 2022. For each target, all model contexts are cut off before the target; the 2025 target is absent from historical folds. TimesFM 3.0 receives the full available ordered PISA history for each country/domain before each target. | **Aligned, with representation differences** | The four years are test origins, not the only four input cycles. RF/XGBoost receive pooled prior transition examples with compact lag features (latest score, second-latest score, change, target year, gap, and count); country trend and TimesFM use the available ordered history. This is a reasonable benchmark but not identical feature encoding across methods. |
| Compare methods on consistent outcomes | Historical combined panel has 5,616 rows (9 methods × 624 country/domain/fold targets) and unique keys; 2025 panel has 1,782 rows (9 × 66 × 3). MAE is calculated on common country/domain/target keys; historical headline scores give each of four folds equal weight. | **Aligned** | Report per-fold results beside the equal-fold average. The folds reuse countries and represent only four temporal origins. |
| Evaluate 2025 as a forecast target | In the inspected scripts/manifests, tabular and trend inputs are restricted to cycles `< 2025`; TimesFM contexts exclude 2025, and adaptation training windows have target cycles `< 2025`. The observed 2025 values enter scoring. The run manifest also records prior project availability of the outcomes/results. | **Forecast cutoff passes; blind-test claim fails** | No evidence here that the model input included the 2025 answer. Researcher/project exposure means this remains retrospective and cannot be presented as an untouched prospective test. |
| Test TimesFM zero-shot and ILSA adaptation | TimesFM 3.0 was tested zero-shot. TimesFM 2.5 LoRA was adapted on fold-eligible country-level PISA/TIMSS/TIMSS Grade 4/PIRLS series, selected using earlier temporal validation, then compared with its own zero-shot arm on matched folds. Independent reruns reproduced TimesFM outputs byte-for-byte. | **Aligned, with modest result** | LoRA reduced matched TimesFM 2.5 equal-fold MAE by less than 0.2 points per domain; it did not replace Persistence as the historical leader. The inputs are national score series, not student microdata; time gaps and program/domain identities are not passed as model covariates. |
| Quantify uncertainty | Phase 3.25 varies target score estimates and country clusters while keeping forecasts fixed. Phase 3.26 samples historical input/label scores and refits Ridge, RF, and XGBoost over 2,000 draws; TimesFM refitting is excluded. | **Partly complete** | These are separate conditional sensitivity analyses, not one joint predictive interval. Phase 3.25’s headline paired contrast is LinearQuadraticTrend versus Persistence. Phase 3.26 uses marginal score SE draws and omits cross-cycle/cross-country link covariance and model specification uncertainty. |
| Include PISA scale-link uncertainty | Pairwise 2025 link-error SE reconstruction was validated, but a full historical covariance for all model input cycles was not identified; the science two-offset diagnostic did not pass its necessary bound. | **Unresolved; correctly omitted** | Do not assume independence, clip/project a non-PSD matrix, or call current uncertainty intervals fully link-adjusted. Reopen only when the required official covariance definitions/inputs are available and validation gates pass. |
| Reproduce reported numbers and preserve provenance | Phase 3.32 reports independent reruns: historical tabular v2 and TimesFM CSVs byte-identical; historical tabular v1 and 2025 tabular numerically equivalent; Phase 3.26 mixed-environment tree checkpoints were diagnosed and replaced in the synthesis by a uniform-version 2,000-draw run. The new validator passed 68 read-only checks. | **Substantially complete for reported core branches** | Some earlier Phase 3.30 statuses are historical snapshots and are superseded by Phase 3.32. The validator checks current panel metrics, contexts, selected output hashes, OECD row counts, uncertainty outputs, and literature frequencies; it is not a full rerun of every model. |
| Support final conclusions | Nine-model historical and 2025 tables, uncertainty sections, literature limits, and reproduction limits are in the synthesis report. The report now separates target-score exclusion from researcher outcome exposure and links this audit. | **Aligned with qualifications** | Findings are sample/protocol-specific. No universal winner, individual-level claim, causal claim, or independent 2025 validation claim is warranted. |

## 5. Confirmed numerical findings

**Which models lead under the primary point-error summaries?** The reproduced historical equal-fold MAE table gives Persistence the lowest value in mathematics (9.328), reading (10.156), and science (8.442). In the 2025 outcome-exposed table, the lowest MAE is XGBoost for mathematics (9.318), Ridge for reading (11.357), and Persistence for science (9.378). These values are PISA score points, not student-level performance metrics.

**How stable is Persistence under the model-input perturbation?** In 2,000 marginal-SE input draws with refitting for Ridge, RF, and XGBoost, Persistence had the lowest equal-fold MAE in 97.45% of mathematics draws, 98.55% of reading draws, and 99.75% of science draws. These are frequencies under the specified simulation—not probabilities of winning future PISA cycles.

**What did the target-score/country-cluster sensitivity show?** For the registered LinearQuadraticTrend-minus-Persistence contrast, historical joint intervals were positive in all three domains. In the 2025 all-66 retrospective analysis, the joint interval excluded zero in reading but included zero in mathematics and science. This is conditional uncertainty analysis for that contrast and does not establish an overall model ranking.

## 6. Problems found and their practical effect

1. **Literature selection claim can be overstated.** The fixed corpus supports descriptive frequencies only; no complete search frame or human second review is documented. This limits how strongly the literature justifies “most common ILSA forecasting methods,” but does not invalidate RF/XGBoost as comparison models.
2. **2025 forecast and 2025 validation are different claims.** The code uses pre-2025 histories, but outcome exposure prevents a blind-test claim. This qualifies inference, not the numerical forecast-versus-score calculation.
3. **The target is aggregated.** Conclusions do not describe individual students.
4. **The methods do not encode history identically.** TimesFM sees the full ordered sequence, while RF/XGBoost use compact lag features learned from pooled prior transitions. Comparative error is still meaningful on shared targets, but an input-representation ablation would answer a different question.
5. **Four folds are limited temporal evidence.** Equal weighting avoids allowing the largest fold to dominate, but repeated countries and only four target origins limit generalization.
6. **Uncertainty remains incomplete.** Link covariance, one joint propagation of target/input/model uncertainty, TimesFM input/refit uncertainty, and hyperparameter/model-specification uncertainty are not covered.
7. **The 2029 prospective protocol is a runbook, not a sealed forecast.** Freeze timestamp, data hashes, access controls, software/model hashes, and forecast archive still need completion before outcomes are accessible.

## 7. Ordered closure plan

| Order | Required work | Completion evidence |
|---:|---|---|
| 1 | Obtain original Scopus/WoS searches, raw exports, query dates/filters/hit counts, deduplication, and screening records; if unrecoverable, run a clearly labeled supplemental search without silently altering the 1,266-row corpus. | Archived native exports and search logs, or a documented recovery failure plus a new dated search protocol and versioned corpus decision. |
| 2 | Have an independent human reviewer screen the five borderline candidates without seeing the primary dispositions. | Reviewer-coded file, disagreement log, and reconciled decision. |
| 3 | Continue OECD link-covariance work only with identifiable, machine-usable official definitions/entries; validate symmetry, PSD, scale direction, and reproduction of official pairwise/trend checks. | Versioned matrix/source manifest and passed numeric checks; otherwise retain the explicit limitation. |
| 4 | Complete the PISA 2029 freeze runbook before outcomes are accessible. | Timestamped protocol, data/code/environment/checkpoint hashes, eligible sample, sealed predictions, and access-control record. |
| 5 | After the next cycle results release, score the sealed forecast without changing it and publish all registered metrics and deviations. | Independent prospective evaluation with the sealed artifact and reproducible scoring output. |

The literature files and human-review decision require external records or a second person; the OECD covariance gate depends on official source availability; the prospective gate cannot be completed before a future outcome. These are not tasks that can be resolved by modifying current local forecasts.

## 8. Validation record

Run from the project root:

```bash
python3 scripts/audit_pisa_evidence_alignment.py
```

Result on 2026-10-24: **1,926 read-only checks passed**. The checks include prediction-key uniqueness and row counts, fold and country counts, tabular training cutoffs, saved context cutoffs for each forecast arm, exact agreement between saved context-cycle lists and all available pre-target PISA cycles, adaptation-validation chronology, reproduction of all 27 historical MAEs and the three 2025 point-MAE leaders, combined-panel output hashes, the 198-row OECD target audit, Phase 3.26 output hashes/draw/checkpoint/row counts, joint uncertainty output hashes, and the two literature-frequency scopes. The validator does not rerun the forecasting models or prove the completeness of external literature searches.

`python3 -m py_compile scripts/audit_pisa_evidence_alignment.py` and `git diff --check` also passed. The full project test suite was not run because this audit changed no forecasting/model implementation and the repository has no report-specific automated test suite.

## 9. Audit gates and closing test

- **Research question:** available from the researcher's brief and operationalized above; a final manuscript may still refine wording.
- **Audience:** this is an internal research-team audit; no journal or external audience has been specified.
- **Falsification plan:** the audit records an evidence-to-conclusion frame, but it was not prospectively registered before the 2025 results were available.
- **Data sanity:** passed for the common-panel counts, score target alignment, and prediction key/cutoff checks described above; external search completeness remains unvalidated.
- **Verdict:** historical and retrospective forecasting objective is substantially executed; broad literature coverage, full uncertainty propagation, and prospective confirmation remain unproven or open.
- **If all tables were deleted, would the argument still stand?** Yes: historical evidence favors Persistence on the registered equal-fold metric; 2025 is a separate outcome-exposed retrospective comparison with domain-specific leaders; missing literature and uncertainty evidence limits broader claims.

**Unmet gates:** External audience not confirmed; falsification criteria were not preregistered before outcome exposure; original search provenance; independent human second review; full historical link-error covariance; complete uncertainty propagation; sealed prospective PISA 2029 forecast and later evaluation.
