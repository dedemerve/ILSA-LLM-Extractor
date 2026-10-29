# Literature frequency and country-level forecasting scope

## What can the frequency counts support?

### Takeaway

The defensible quantitative statement is descriptive and corpus-bounded: in the project's frozen 1,266-record classification, Random Forest and XGBoost are the most frequent **coded primary algorithm labels** among the 147 predictive records with a coded primary method (42/147, 28.6%; 17/147, 11.6%). This ranking must not be carried over to genuine future-cycle country-mean forecasting: within the narrower coded country-level prediction-task-D subset, only 17 records have a primary algorithm and the leading labels are SVM (4), Random Forest (3), then XGBoost (2); the local review also shows at least several external cross-cycle/future-cycle papers absent from the frozen corpus, so no exhaustive frequency ranking is currently supportable for that narrow literature.

### Cited findings

- The project audit reports 1,266 unique coded records; 166 are tagged predictive and 147 of those have a coded primary method. Primary-label counts in that predictive subset: Random Forest 42, XGBoost 17, Neural Network 16, SVM 14. RF and XGBoost therefore rank first and second **within this dataset and coding scheme**, not in all published ILSA prediction research. Source: local project audit, `analysis/03_data_feasibility/phase3_11_corpus_method_frequency.md` and its CSV `analysis/03_data_feasibility/audit/ilsa_ml_method_frequency_v1.csv`.
- The same frequency CSV gives the country-level prediction-task-D subset as 17 primary-coded records: SVM 4 (23.5%), Random Forest 3 (17.6%), XGBoost 2 (11.8%). This is task-D's coded country-level prediction scope and is not equivalent to a verified set of prospective country-mean forecasts.
- In the separate any-technique count, methods are counted once per study and a study can contribute to multiple methods; do not blend those totals with the one-primary-label denominator. The local audit normalizes a limited set of RF/XGBoost aliases and explicitly notes that classification was automated, not independently re-read for every source.
- The original database search cannot be reconstructed from local records: the saved inventory contains 725 extraction JSON files (423 Scopus, 302 Web of Science), not raw database exports; execution dates, native queries/fields, filters, hit counts, export completeness, and screening decisions are unavailable. The inventory's 665 distinct DOI / 663 corpus matches are not a recall estimate. Thus the fixed corpus can describe its own recorded membership but cannot establish exhaustive literature prevalence. Source: local `analysis/03_data_feasibility/phase3_28_literature_scope_audit.md` and `phase3_31_becker_coyle_and_search_provenance.md`.

### Inferences

- Recommended RQ1 frequency wording: “Within our coded corpus, Random Forest (42/147; 28.6%) and XGBoost (17/147; 11.6%) were the two most common primary algorithm labels among predictive studies with a coded primary method.” Add the denominator, codebook/corpus version, and a coverage caveat in the manuscript.
- Do not state that RF/XGBoost are the most common approaches for country-level ILSA forecasting. The narrower coded task-D order is SVM > RF > XGBoost, while the prospectively relevant papers found in this audit are few and methodologically heterogeneous.
- Do not present a percentage/frequency ranking for “genuine future-cycle forecasts” until a new documented search, record-level eligibility screening, deduplication, and independent second review are complete. These records are studies, not independent model runs or country-domain forecasts.

### Gaps

- No reproducible search history or raw Scopus/WoS exports; consequently, no defensible literature-wide numerator/denominator or recall estimate.
- No source-level second reviewer for borderline temporal and outcome eligibility in the local candidate register.
- A complete, updated primary-method coding of all eligible country-level future-cycle studies has not been conducted; it would require a new versioned corpus and explicit treatment of country-level means vs country aggregates of student predictions, future vs retrospective cross-cycle estimates, and forecasts vs QC/anomaly detection.

## Which papers are methodologically in scope for the narrow target?

### Takeaway

Primary sources establish that the narrow area is not simply empty and that it includes conventional ML and trend/regression methods, but its boundaries matter. Acisli-Celik & Yesilkanat explicitly use PISA 2015 to predict PISA 2018 science achievement and country averages; Braun et al. use TIMSS cross-administration models to predict country-level Grade 8 means for QC; Kaplan et al. show a one-cycle-ahead PISA 2025 mathematics trajectory, but their principal validated outcome is pace of progress and displayed forecasts are sex-stratified; Becker & Coyle reconstruct PISA 2022 country means from prior cycles with linear/quadratic trends. These are not interchangeable evaluation designs, and raw paper counts should not be interpreted as a fair RF/XGBoost-versus-time-series model comparison.

### Cited Findings

| Source | Evidence type / method | Outcome and temporal status | Fit to strict future-cycle country-domain mean forecast | Weight / caveat |
|---|---|---|---|---|
| Acisli-Celik & Yesilkanat (2023), *Neural Computing and Applications*, DOI 10.1007/s00521-023-08901-6. [Publisher DOI](https://doi.org/10.1007/s00521-023-08901-6) | Peer-reviewed primary empirical study; multiple linear regression, SVR, Random Forest, XGBoost. The abstract reports 67,329 PISA 2015 students from 13 countries and says XGBoost performed best for estimating student science results and country mean science scores. | Explicit PISA 2015 → PISA 2018 science prediction/estimate. | **Strongest ML precedent found** for a cross-cycle PISA science target and country averages. | Abstract is available through ILSA Gateway [record](https://ilsa-gateway.org/papers/predicting-science-achievement-scores-machine-learning-algorithms); publisher DOI resolves to primary article. However, the reported training/test split within PISA 2015 assesses within-cycle performance, whereas 2018 is a separate external cycle; do not conflate metrics or infer prospective deployment from abstract alone. |
| Braun, von Davier & Chen (2025), *Large-scale Assessments in Education*, DOI 10.1186/s40536-025-00262-x. [Publisher full text](https://link.springer.com/article/10.1186/s40536-025-00262-x) | Peer-reviewed primary methods/QC study; neural network at student level, country-level aggregation, then regression bias correction. | TIMSS administrations predict later-cycle Grade 8 country means: 2011/2015 and 2015/2019, with cross-grade and same-grade/different-cohort specifications. The paper says country-level predicted means were compared against observed means and explores QC residuals. | **Relevant cross-cycle country-mean prediction**, including same-grade later-administration forecasting; include in broad narrow-target mapping, but label as QC/anomaly-screening rather than conventional forecasting. | The authors state that all countries are pooled for training and leave-one-country-out validation would be preferable but computationally costly. The target-cycle observations are used retrospectively to evaluate residuals; it is not a deployed forecast. No RF/XGBoost or conventional time-series comparison. |
| Kaplan, Harra, Stampka & Jude (2025), *Psychometrika*, DOI 10.1017/psy.2025.2. [Publisher article PDF](https://www.cambridge.org/core/services/aop-cambridge-core/content/view/D85FD94C994DB13028209C06CA323169/S003331232500002Xa.pdf/stacking_models_of_growth_a_methodology_for_predicting_the_pace_of_progress_to_the_education_sustainable_development_targets_using_international_largescale_assessments.pdf) | Peer-reviewed primary modeling study; Bayesian growth models and stacking to predict pace of progress. | PISA mathematics country trajectories 2009–2022; paper displays one-cycle-ahead 2025 trajectories by sex. LOO country validation is for predicting pace, and 2025 forecasts are not validated against observed 2025 results in the paper. | **Near match / methodological precedent**, but not a directly validated unstratified country-domain mean forecast comparison. | Keep as a separate projection design; avoid treating predicted pace validation as forecast-score validation. |
| Becker & Coyle (2026), DOI 10.3389/feduc.2026.1734988. [Publisher article and supplement](https://www.frontiersin.org/journals/education/articles/10.3389/feduc.2026.1734988/full) | Peer-reviewed primary paper; country/domain linear and quadratic trend regressions, with model choice from an F-test. Supplement contains source data and R code. | Uses PISA waves through 2018 to forecast 2022 country means across mathematics, reading, and science; retrospective forecast target is observed. | **Direct outcome/method precedent** for country/domain future-cycle-style prediction using trend models. | The project reconstructed the published procedure rather than extracting a publisher forecast table and compared on 60 common countries/domain (180 rows). Local Phase 3.31 reports reconstructed MAE 19.70 math, 20.07 reading, 18.53 science versus persistence 14.48, 12.35, 8.83, respectively. This is a descriptive historical comparison; publication postdates target and the project's same fold is not independent prospective validation. |
| Kaplan et al. (2025), arXiv:2511.06107. [Preprint](https://arxiv.org/abs/2511.06107) | Preprint / primary projection analysis; Bayesian latent growth and model averaging. | Projects sex-specific country shares reaching minimum proficiency in reading/math from PISA 2009–2022 plus covariates. | **Exclude from strict mean-score outcome count**; adjacent ILSA indicator projection only. | Threshold shares are not country mean scores; preprint status differs from peer-reviewed articles. |
| Erdoğan & Taştan (2024), “Predicting Student Achievement via Machine Learning: Evidence from Turkish Subset of PISA,” [journal/repository record](https://ideas.repec.org/a/aye/journl/v10y2024i1p7-27.html) | Primary empirical PISA 2018 study; boosted regression trees compared with bagging and RF regression trees. | Predicts Turkish students' PISA mathematics, science, and reading performance within a cycle. | **Exclude from strict future-cycle country-mean set**; include in broad student-level predictive literature frequency only if the corpus coding rules qualify it. | Useful illustration that general PISA achievement prediction counts include within-cycle student-level models, not only country forecasting. The abstract's BRT result is not automatically identical to an XGBoost label. |

### Inferences

- For manuscript organization, report two distinct evidence universes: (1) broad achievement-prediction papers, where the local coded-corpus RF/XGBoost counts apply; (2) strict cross-cycle country-mean forecast/projection precedents, where eligibility must be adjudicated at paper level and counts remain exploratory/non-exhaustive.
- The candidate set includes RF/XGBoost, NN+regression aggregation, Bayesian growth/stacking, and linear/quadratic trend models, but differences in outcome, temporal split, validation unit, and purpose make paper-level performance numbers non-comparable. A common project benchmark with shared country/domain/cycle folds is needed for RQ1's empirical performance comparison.
- Braun et al. should be checked against the user's intended definition: if “forecast” means a score forecast produced before target-cycle data exist, the paper's retrospective QC design is not a prospective forecast, although it predicts the later cycle from earlier administration data.
- The 2023 paper is a key potential missing record because the existing local audit describes 28 cross-cycle flags and the fixed corpus as not exhaustive; whether it belongs to the original database-search inclusion set is unknown.

### Gaps

- Candidate discovery was exploratory and did not preserve result counts; it cannot produce a PRISMA-complete search or prove that these are all qualifying papers.
- Full text/reproducibility details for the 2023 PISA paper were not retrieved in this pass; primary DOI and abstract-level details are verified, detailed model fit and country-average validation design need full-text review.
- No verified directly comparable TIMSS/PIRLS published future-cycle country-domain mean forecast was located in this focused pass beyond Braun et al.'s QC framing. This is “not found in this search,” not proof of absence.
- The PISA 2025 observed-cycle release exists now, but Kaplan et al.'s 2025 predictions were not checked against it in the underlying paper; a post hoc forecast verification would require recovering exact forecast values and alignment rules.

## Search strategy and claim boundaries

### Takeaway

This was a focused, source-led audit of the local corpus and candidate registers, supplemented by targeted web searches for primary papers. It supports a careful scoped claim and identifies likely additions; it is not an exhaustive systematic search, and no revised denominator should be published from it.

### Cited findings

- **Local materials inspected:** `phase3_11_corpus_method_frequency.md`; `phase3_27_literature_candidate_adjudication.md`; `phase3_28_literature_scope_audit.md`; `phase3_31_becker_coyle_and_search_provenance.md`; `audit/ilsa_ml_method_frequency_v1.csv`; `audit/literature_candidate_adjudication_v1.csv`; `audit/literature_search_inventory_v1/`.
- **Web search strings used in this focused pass:**
  - `PISA student achievement prediction machine learning random forest XGBoost primary study`
  - `TIMSS achievement prediction machine learning Random Forest student achievement`
  - `PISA country level forecast mean score future cycle Becker Coyle 2026`
  - `PISA future score forecast country level machine learning`
  - `"Predicting science achievement scores with machine learning algorithms" DOI`
  - `"Stacking models of growth" PISA 2025 Psychometrika DOI`
  - `site:frontiersin.org "country" "PISA 2022" forecast mean score 2025`
  - `TIMSS country mean score forecast machine learning future cycle achievement`
- **Inclusion for the focused candidate map:** primary studies/method papers using PISA/TIMSS/PIRLS achievement data; country-level mean-score outcome or explicitly country-aggregated student score prediction; a temporal relationship across assessment administrations or an explicit later-cycle projection. The map retains borderline temporal/forecast designs but labels them.
- **Exclusions:** studies predicting individual achievement only within one administration from the strict country-mean cross-cycle set; threshold-share projections from the strict mean-score set; descriptive trend tables without a prediction model; background mentions of ILSA without ILSA forecast outcome; generic education ML papers without ILSA data.
- **Search boundaries:** local project search inventory and audit records plus targeted web search on 2026-10-25; English-language search strings; primary publisher pages prioritized. This pass did not query or preserve fresh Scopus, Web of Science, ERIC, or OpenAlex result exports, and did not conduct citation chaining systematically.

### Inferences

- For a defensible final paper, phrase the method-frequency result as an **internal coded-corpus result**, include both 42/147 and 17/147 with the exact scope, and state search coverage is unresolved.
- Any claim that RF and XGBoost are the most frequent models “in ILSA research” without the corpus qualifier overstates the available evidence. Any claim they are most frequent in prospective country-level future-cycle forecasts is contradicted by the narrower coded subset's frequency order and unsupported by the small, incompletely searched external candidate set.
- Preserve the distinction between method-use frequency (literature coding) and comparative forecasting performance (shared benchmark). One cannot substitute for the other.

### Gaps

- Exact executed Scopus/WoS queries, dates, limits, exports, deduplication, and screening logs remain unrecovered per local audit.
- No new supplemental database search with saved raw exports was conducted in this audit.
- No full bibliographic screening, duplicate resolution, or second-review adjudication of all eligible country-level cross-cycle papers was done.
