# Phase 2.2 Report: Methodological Issues and Refined Zero-Forecasting Claim

**Date:** 2026-10-14  
**Protocol:** Research Protocol v1, Phase 2.2 (addendum to Phase 2.1)  
**Preceding artifacts:** phase2_taxonomy_report.md, phase2_1_report.md, validation_sample_30.csv  
**This file does not overwrite any prior artifact.**

---

## Task 1: TIMSS QA Study — Full Classification Audit

### Study
Braun, von Davier & Chen (2025). "Rethinking TIMSS quality assurance: utilizing neural network models with regression-based bias mitigation strategies for validating country-level math and science achievement scores." *Large-scale Assessments in Education*, 13. DOI: 10.1186/s40536-025-00262-x  
File: `s40536-025-00262-x.pdf`

### Methodological evidence from extraction record

**Cycles used:** TIMSS 2011, 2015, and 2019 (all Grade 8; Method 1 also uses Grade 4).

**Training and prediction structure:**
- **Method 1 (cohort linkage):** Grade 4 predictors from cycle *t* → Grade 8 achievement in cycle *t+4* years. Two pairs fitted: 2011 G4 → 2015 G8, and 2015 G4 → 2019 G8. A neural network is trained on the earlier pair and applied to predict the later pair's country means.
- **Method 2 (G8-to-G8 cross-cohort):** Grade 8 predictors from cycle *t* → Grade 8 achievement in cycle *t+4*. Two pairs: 2011 G8 → 2015 G8, and 2015 G8 → 2019 G8.

**Evaluation target:** Country-level residuals — the difference between the neural network's predicted country mean and the observed country mean. Residuals above the 80th percentile flag countries for quality screening.

**Answers to the five methodological questions:**

| Question | Finding |
|----------|---------|
| Which cycles were used for training? | 2011 data (G4 or G8) trains the model predicting 2015 G8; 2015 data (G4 or G8) trains the model predicting 2019 G8. |
| Which cycle(s) for validation/testing? | 2015 G8 and 2019 G8 are the prediction targets; residuals computed and used for screening. |
| Were 2019 outcomes unavailable during model development? | **No.** TIMSS 2019 data was publicly released in December 2020; the paper was published in 2025. All three cycles were available simultaneously during the study's development. |
| Is the target country-level achievement? | Yes — country-level Grade 8 mean scores and residuals. Not student-level. |
| Does the study evaluate predictions against subsequently observed outcomes? | **No — retrospective.** All cycle data (2011, 2015, 2019) was known when the analysis was conducted. The study does not withhold a future cycle during model fitting; it fits each model on earlier-cycle data and evaluates on later-cycle data, but both sets were available at the same time. |
| Does the design satisfy the operational definition of future-cycle forecasting? | **No.** The operational definition requires: (a) historical time-series training, (b) forecasting a future cycle outcome, AND (c) evaluation against a subsequently observed cycle. Condition (c) is not met: no TIMSS cycle was prospectively held out. |

**Purpose of the study:** The explicit purpose is quality assurance screening — detecting anomalous country-level score changes that may indicate data-quality or population-definition problems. The residual distribution is used as a triage signal, not a forecast.

### Classification

| Field | Current value | Verdict |
|-------|--------------|---------|
| primary_task_code | B (automated), D (Phase 2 manual) | **Retain D** — country/system level, cross-cycle |
| temporal_forecasting_status | cross_cycle_prediction_not_fully_verified | **Retain** — genuinely cross-cycle design, but purpose is retrospective QA not prospective forecasting |
| Satisfies future-cycle forecasting definition? | — | **No** |

**No reclassification required.** The Phase 2 manual verdict ("D confirmed, cross-cycle; purpose is quality assurance not prospective forecasting") is supported by the full extraction record.

### Revised zero-forecasting conclusion after Task 1

The TIMSS QA study was already excluded from future-cycle forecasting in Phase 2. This Task 1 audit provides the methodological basis for that exclusion:
1. All TIMSS cycles used were retrospectively available
2. The design is cross-cycle retrospective validation for quality assurance, not prospective forecasting
3. No held-out future cycle exists in the evaluation design

**The zero confirmed future-cycle forecasting conclusion is unchanged.**

---

## Task 2: Validation Sample Size Discrepancy

### Finding

The Phase 2.1 report states a "30-study stratified validation sample" but the actual CSV (`validation_sample_30.csv`) contains **28 unique studies**.

### Root cause

The sampling design specified a stratum `G_ml` (ML studies with non-predictive task code G) targeting 2 studies. However, this stratum has a structural pool size of **0**: the classification script assigns task code G exclusively to non-ML studies (those without `study_filter_type = 'Empirical Study - Machine Learning'`). No ML study can receive task code G under the current classifier logic. The stratum was never populated; the 2 targeted records were never drawn.

| Stratum | Target | Actual | Shortfall | Reason |
|---------|--------|--------|-----------|--------|
| A | 8 | 8 | 0 | |
| D | 5 | 5 | 0 | |
| F | 5 | 5 | 0 | |
| **G_ml** | **2** | **0** | **2** | Structurally empty: classifier logic precludes ML + G |
| G_trad | 5 | 5 | 0 | |
| B_confirmed | 2 | 2 | 0 | |
| cross_cycle_other | 3 | 3 | 0 | |
| **Total** | **30** | **28** | **2** | |

### Corrected statements

- **Actual unique studies reviewed: 28** (not 30)
- **Stratum G_ml: 0 studies reviewed** (stratum was empty; this is a sampling design error, not a study-level error)
- The "30" in the Phase 2.1 report title was an intended target that was not achieved
- No study was excluded during deduplication that reduces the count from 30 to 28; the shortfall arises entirely from the empty G_ml stratum

### Limitation on corpus-wide inference

The 28-study sample was drawn from a non-probability stratified design targeting specific task-code strata. It cannot support population-level precision or recall estimates for the full 1,266-study corpus. Specifically:

- The sample does not cover the non-ML G stratum representatively (only 5 of 538 non-predictive traditional-statistics studies were reviewed)
- The sample includes all 2 confirmed cross-cycle ML studies and 3 from the cross-cycle false-positive pool — this is a purposive design, not a random one
- **The absence of false negatives in 28 reviewed studies does not establish that no false negatives exist in the 1,266-study corpus.** A missed future-cycle forecasting study in the non-ML, exploratory, or grey-literature portion of the corpus would not have been detected by this sample.

---

## Task 3: Disaggregated Zero-Forecasting Claims

The Phase 2.1 report used "zero future-cycle forecasting" as a single claim. Phase 2.2 separates this into three distinct claims with independent evidence bases.

### Claim A: No confirmed country-level future-cycle forecasting study was identified in the current corpus

**Verdict: Confirmed, with stated caveats.**

Evidence: All 38 keyword candidates from outcome_summary, 0 from target_variable, and all 8 automated cross-cycle candidates were manually reviewed. None satisfies the operational definition: using historical ILSA country-level time-series data to train a model that forecasts a future cycle outcome evaluated against subsequently observed results.

Caveat: The search was extraction-based. Studies classified as "traditional statistics" or "non-predictive" based on extraction fields were not exhaustively inspected at source level. A study that uses ML for country-level forecasting but whose extraction record does not surface forecasting keywords cannot be ruled out.

### Claim B: No study of any kind performs cross-cycle prediction

**Verdict: FALSE. This claim is too broad and is NOT supported.**

Confirmed cross-cycle ML studies in the corpus: **2**
- `s00521-023-08901-6.pdf`: Trains Random Forest on PISA 2015 student data, evaluates on PISA 2018 student data (13 OECD countries) — genuine student-level cross-cycle prediction
- `s40536-025-00262-x.pdf`: Neural network predicts country-level TIMSS G8 means using prior-cycle G4 data (2011→2015, 2015→2019) — genuine country-level cross-cycle ML

Additionally, the longitudinal traditional-statistics literature (e.g., PISA 2009–2018 pooled fixed-effects, PISA 2006 + LSAY follow-up cohort) uses data from multiple cycles, though not for machine-learning cross-cycle prediction.

### Claim C: No study uses machine learning to predict future ILSA outcomes

**Verdict: Requires precision about "future."**

- No study uses ML to predict a prospectively held-out ILSA cycle outcome (i.e., a cycle the model did not see during development)
- **Two studies use ML in a cross-cycle design** (see Claim B), but both are retrospective: all cycle data was available simultaneously during model development
- Zero studies evaluate ML predictions against a cycle that was administered after the model was trained and not available to the researchers at the time

**Summary table**

| Claim | Verdict |
|-------|---------|
| A: No confirmed country-level future-cycle forecasting | **Confirmed** (with extraction-based caveats) |
| B: No study performs cross-cycle prediction | **False** — 2 confirmed cross-cycle ML studies exist |
| C: No ML study predicts future ILSA outcomes prospectively | **Confirmed** — no prospective held-out future-cycle evaluation found |

---

## Task 4: Artifact Preservation

All Phase 1, Phase 2, and Phase 2.1 artifacts are preserved unchanged:
- `analysis/01_dataset_audit/dataset_audit.md`
- `analysis/01_dataset_audit/audit_summary.json`
- `analysis/02_prediction_taxonomy/classify_studies.py`
- `analysis/02_prediction_taxonomy/study_classification.csv`
- `analysis/02_prediction_taxonomy/phase2_taxonomy_report.md`
- `analysis/02_prediction_taxonomy/phase2_1_report.md`
- `analysis/02_prediction_taxonomy/validation_sample_30.csv`
- `analysis/02_prediction_taxonomy/ambiguous_studies.csv`
- `analysis/02_prediction_taxonomy/program_normalization.csv`
- `analysis/02_prediction_taxonomy/phase2_audit.json`

This Phase 2.2 report is a new, versioned addendum. No prior artifact has been modified.

---

## Task 5: Phase 3 Readiness — Decision Report

### Verified counts

| Item | Count |
|------|-------|
| Confirmed future-cycle forecasting studies (Claim A) | **0** |
| Confirmed prospective held-out ML cross-cycle studies (Claim C) | **0** |
| Confirmed retrospective cross-cycle ML studies | **2** |
| Unresolved candidates (require source-level inspection) | **1** |

### Unresolved candidate

**German ICT / PISA 2015→2018 study** (`Combining statistical and machine learning methods to explore German students' attitudes towards ICT in PISA.pdf`): The findings record states "When evaluated on 2018 data: multiclass AUC..." after training on PISA 2015 data. Whether this constitutes a genuine 2015→2018 cross-cycle transfer or two independent within-cycle models is ambiguous from the extraction record alone. Source article inspection is required.

- If it is a 2015→2018 transfer: confirmed cross-cycle ML count rises from 2 to 3; zero-forecasting conclusion unaffected.
- If it is two independent within-cycle models: current classification stands.

### Corrected validation sample size

- **Intended:** 30 studies
- **Actual:** 28 unique studies reviewed
- **Shortfall reason:** G_ml stratum structurally empty (classifier logic precludes ML + G combination)

### Remaining limitations of the zero-forecasting claim

1. **Extraction-based search.** The keyword search covered outcome_summary and target_variable fields. Forecasting elements buried in methodology sections, appendices, or supplementary materials of studies classified as "non-predictive" would not be detected.
2. **Non-ML grey literature gap.** 271 studies (21.4%) have no DOI; these include technical reports and grey literature that may not have been fully extracted.
3. **Sample inference limitation.** The 28-study validation sample does not support population-level false-negative rate estimates. The absence of missed forecasting studies in the sample does not prove no such studies exist.
4. **One unresolved candidate** (German ICT paper; does not affect the forecasting conclusion either way).

### Recommendation

**Proceed to ILSA data-feasibility analysis (Phase 6 in the protocol).**

Rationale:
- The zero-future-cycle-forecasting finding is robust under all verified evidence
- The research gap claim meets the evidential standard for justifying a feasibility analysis
- Phase 3 (algorithm taxonomy depth) and Phase 4 (evaluation design audit) can proceed in parallel with, or after, Phase 6 data-feasibility work, as they do not gate the TimesFM experiment design
- The one unresolved candidate (German ICT paper) should be resolved at source level during Phase 3; it does not affect the go/no-go decision

**Do not run TimesFM or select final forecast cut-off years during Phase 2.2 or Phase 3.**
