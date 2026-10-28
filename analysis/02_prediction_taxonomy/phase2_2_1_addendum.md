# Phase 2.2.1 Addendum: Source-Level Verification of Braun et al. (2025)

**Date:** 2026-10-14  
**Protocol:** Research Protocol v1, Phase 2.2.1  
**Preceding artifact:** phase2_2_report.md (Task 1 relied on extraction record only)  
**Source accessed:** https://link.springer.com/article/10.1186/s40536-025-00262-x  
  (Open access, full HTML article retrieved 2026-10-14)  
**This file does not overwrite any prior artifact.**

> **Coverage qualification (2026-10-15):** This addendum's source review of Braun et al. remains valid. Its “zero confirmed in the 1,266 records” statement is conditional on corpus membership, not an exhaustive statement about the literature. A separate 2026 source-level audit verified a PISA 2022 country-mean trend forecast based only on PISA through 2018 (Becker & Coyle, DOI 10.3389/feduc.2026.1734988) that is not present in the corpus. Recover the original search scope and assess that record before making a field-wide zero claim; see `analysis/03_data_feasibility/phase3_14_cross_cycle_review_and_scope_correction.md`.

---

## Study

Braun, H. I., von Davier, M., & Chen, J. (2025). Rethinking TIMSS quality assurance: utilizing neural network models with regression-based bias mitigation strategies for validating country-level math and science achievement scores. *Large-scale Assessments in Education*, 13, article 29. DOI: 10.1186/s40536-025-00262-x  
Local file: `s40536-025-00262-x.pdf`

---

## Source-Level Answers to the Six Protocol Questions

### Q1. Which TIMSS cycles were used for model training?

**Section: Methodology / RQ1 — Stage 1: neural networks**

> "For this study, we initially began by training a NN to predict **both 2011 4th-grade and 2015 8th-grade student-level math scores simultaneously**, using the observed 2011 4th-grade and 2015 8th-grade student background variables, respectively."

> "After completing the analysis for predicting 2015 Grade 8 mean country-level math scores and computing the corresponding residuals, we employed the same procedure described above to train three additional neural network models: one for predicting student-level science scores for the 2015 8th-grade cohort (using 2011 4th-grade cohort scores), one for **predicting 2019 8th-grade math scores (using 2015 4th-grade cohort scores)**, and one for the **2019 8th grade science scores (using 2015 4th grade cohort scores)**."

**Finding:** For the model that generates "2019 predictions":
- Input features: individual-level background variables from TIMSS 2015 Grade 4
- Criterion (label): individual-level achievement scores from TIMSS **2019 Grade 8**

The 2019 Grade 8 achievement outcomes are the supervised learning target. They are in the training data.

---

### Q2. Was TIMSS 2019 excluded from model fitting, feature selection, hyperparameter tuning, and model selection?

**Section: Methodology / RQ1 — training note (verbatim)**

> "Note: The data from all countries are incorporated into the analytic data set on which the neural network was trained. With a substantial number of countries contributing data, the impact of any one country on the final results is rather small. **Ideally, a 'leave-out-one' strategy akin to cross-validation would be preferred, but would require substantially more computation. This would constitute an interesting extension of the present study.**"

**Finding:** The authors explicitly state that no leave-one-out or held-out validation was used. All country data — including 2019 G8 outcomes — were used in the training set. The absence of a holdout is acknowledged as a limitation. The 2019 G8 outcomes were **not** excluded from model fitting.

**Answer: No.**

---

### Q3. Did the authors generate predictions for TIMSS 2019 using a model developed without access to its outcomes?

**Section: Methodology / RQ1 — prediction procedure (verbatim)**

> "Leveraging the fitted model, we exploited the fact that—for each country—the 4th-grade and 8th-grade samples represent (approximately) probability samples of the same cohort. Specifically, we predicted 2015 8th-grade math scores for each country by inputting the **2011 4th-grade background and contextual data while modifying the grade indicator from 0 (4th grade) to 1 (8th grade)**."

The same procedure was applied to 2019: a model trained on (2015 G4 inputs, 2019 G8 criterion) generates a "prediction" for each country by feeding its 2015 G4 data through the model with the grade indicator set to 1.

**Finding:** The "prediction" for 2019 G8 is produced by a model whose training criterion was 2019 G8 outcomes. This is an in-sample fitted value, not an out-of-sample prediction. The model had access to 2019 outcomes during fitting.

**Answer: No.**

---

### Q4. Were those predictions evaluated against the observed TIMSS 2019 outcomes?

**Section: Results for RQ1**

> "First, for 2015 TIMSS Grade 8 Mathematics, we compare the distributions of residuals for two parallel analyses (2011–2015 and **2015–2019**), limited to the 29 countries that participated in both pairs of administrations."

Yes — residuals (model output minus observed 2019 G8 score) were computed and the distribution was used for quality screening.

**Finding:** The residuals were computed, but they are **in-sample residuals** from a model trained on the same 2019 G8 outcomes. This is structurally equivalent to computing training-set residuals after fitting a regression — not an out-of-sample evaluation.

**Answer: Technically yes, but the evaluation is in-sample, not out-of-sample.**

---

### Q5. Was the target a country/system-level achievement outcome?

**Section: Abstract**

> "The basic methodology consists of a combination of neural networks and classical regression analysis to generate **model-based predictions of individual-level scores which are then aggregated to obtain country-level, average achievement scores** for all participating jurisdictions."

**Answer: Yes — country-level aggregate mean scores for Grade 8 Math and Science. Not student-level.**

---

### Q6. Did the study evaluate a genuinely held-out future cycle, or merely compare data from multiple cycles retrospectively?

**Synthesis from Q1–Q4:**

1. The 2019 G8 achievement outcomes were the supervised criterion in the NN used to generate "2019 predictions"
2. No cycle was prospectively held out — the authors explicitly acknowledge this and flag it as a limitation
3. The residuals for 2019 are in-sample: computed from a model that saw the 2019 outcomes during training
4. The purpose is quality-screening triage, not forecasting

**Answer: The study evaluated data from multiple cycles retrospectively. No future cycle was genuinely held out. The 2019 outcomes were in the training criterion.**

---

## Classification

**Assigned classification: `RETROSPECTIVE_MULTI_CYCLE_ANALYSIS`**

This classification is more precise than `CROSS_CYCLE_PREDICTION_OTHER` because the 2019 outcomes were not merely "available" during model development (the Phase 2.2 inference) — they were the training criterion (the Phase 2.2.1 source-level finding). The study cannot be correctly described as "predicting 2019 using only pre-2019 data." The 2019 G8 scores were in the supervised training signal.

| Field | Phase 2 value | Phase 2.2 value | Phase 2.2.1 (source-verified) |
|-------|--------------|-----------------|-------------------------------|
| primary_task_code | D | D | **D — retain** |
| temporal_forecasting_status | cross_cycle_prediction_not_fully_verified | cross_cycle_prediction_not_fully_verified | **Resolved: retrospective; 2019 G8 outcomes in training criterion** |
| Satisfies future-cycle forecasting definition? | No | No | **No — confirmed at source** |
| Source inspected? | No | No | **Yes — full HTML article** |

**Classification note for the controlled vocabulary:** The existing schema does not have a `retrospective_multi_cycle` or `in_sample_residual` label. The most accurate description is `cross_cycle_retrospective_no_holdout`. The study should not be reclassified to `no_temporal_prediction` because the cross-cycle design (training on earlier cycle, generating residuals against later cycle) is genuine. Recommend adding a resolved status annotation in Phase 3 schema work.

---

## Decision Note: Does "zero confirmed future-cycle forecasting studies" Remain Defensible?

**Yes — and the claim is now more precisely grounded.**

The Phase 2 and Phase 2.2 determinations were correct in conclusion but based on inference from publication date and data availability. This source-level verification provides a stronger foundation:

1. The TIMSS QA study — the only study whose status was previously "not fully verified" — has been confirmed at source as a `RETROSPECTIVE_MULTI_CYCLE_ANALYSIS`. The 2019 G8 outcomes were in the training criterion; there was no held-out future cycle.

2. No other study in the 1,266-record corpus was confirmed at any level of verification to satisfy the operational definition of future-cycle forecasting.

**Revised claim (precise version):**

> *No study in the 1,266-record corpus was confirmed — at extraction level or verified at source level — to perform future-cycle ILSA forecasting with a genuinely held-out future cycle, at either country or student level.*

**Remaining limitations of this claim:**

1. Source-level verification was performed for only one study (Braun et al., 2025). The 37 other keyword candidates from the Phase 2.1 broad search were assessed from extraction records only.
2. The 28-study validation sample does not support a population-level false-negative rate estimate.
3. Studies classified as "traditional statistics" or "non-predictive" based on extraction fields were not inspected at source. A forecasting study whose extraction record does not surface forecasting keywords would not be detected.
4. The German ICT paper (PISA 2015/2018) remains an unresolved candidate for cross-cycle status (does not affect the forecasting conclusion).

**These limitations are known and proportionate.** The zero-forecasting claim is well-supported for the ML literature as characterized in the extraction corpus. It is a corpus-level claim, not a claim about every paper ever published on ILSA.
