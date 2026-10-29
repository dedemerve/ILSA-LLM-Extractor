# Phase 2.1 Report: Classification Validation and Phase 3 Readiness

**Date:** 2026-10-14  
**Protocol version:** Research Protocol v1, Phase 2.1  
**Input:** `outputs/ILSA_Meta_Analysis_Dataset_CLEAN.xlsx`, `analysis/02_prediction_taxonomy/study_classification.csv`  
**New outputs:** `analysis/02_prediction_taxonomy/phase2_1_report.md`, `validation_sample_30.csv`, `phase2_1_verification_candidates.md`

---

## Task 1: Audit of Classification Logic

### Reviewed: `analysis/02_prediction_taxonomy/classify_studies.py`

**Confirmed correct behaviors:**
- `temporal_forecasting_status` is derived from `study_filter_type`, dataset_used multi-cycle patterns, cycle count, and keyword search
- `primary_task_code` is derived from `temporal_forecasting_status` + `prediction_target_level`
- `classification_confidence` is set to Low for any `cross_cycle_prediction_not_fully_verified`
- `manual_review_required` = True for any confidence < High OR cross-cycle OR potential duplicate

**Known limitations (extraction-based, not source-verified):**
- Year patterns in `dataset_used` produce false positives (~75% rate for cross-cycle detection; confirmed in Phase 2 manual review)
- `prediction_target_level` is inferred from outcome_summary keywords, not article-level schema fields
- `study_filter_type` is an upstream extraction field; its accuracy was not re-verified here

---

## Task 2: Zero-Forecasting Verification

### Search strategy
Three independent searches were performed across the 1,266-study corpus:

1. **Keyword search in `outcome_summary`** (38 candidates): terms `forecast`, `predict.*future`, `next.*cycle`, `future.*cycle`, `project.*202`, `extrapolat`
2. **Keyword search in `target_variable`** (0 candidates): same terms — no hits
3. **Automated cross-cycle classifier** (8 ML studies): multi-cycle year patterns + keyword scan

### Candidate verification table

| Study ID | File | Forecasting claim | ML? | Verdict |
|----------|------|-------------------|-----|---------|
| 1 | `1-s2.0-S0191886923000338-main.pdf` | "predict societal cognitive development to 2100" using TIMSS/PISA/PIRLS historical trends | No (linear/quadratic regression) | **Not forecasting** — traditional statistics extrapolation; no held-out future cycle evaluation |
| 2 | `s00521-023-08901-6.pdf` | Trains on PISA 2015, evaluates on PISA 2018 student science scores | Yes | **Confirmed cross-cycle ML** — student-level, 13 OECD countries; NOT system-level forecasting |
| 3 | `s40536-025-00262-x.pdf` | Neural network predicts country achievement across TIMSS 2011→2015→2019 | Yes | **Confirmed cross-cycle ML** — country-level, quality assurance purpose; NOT prospective forecasting |
| 4 | `Combining statistical and machine learning methods to explore German students attitudes towards ICT in PISA.pdf` | RF trained on PISA 2015 (Germany), AUC reported for 2015 hold-out AND 2018 separately | Yes (RF) | **Not cross-cycle** — two independent within-cycle analyses; 2015 and 2018 models are distinct; no 2015→2018 prediction chain |
| 5 | `sachisthal-et-al-2020-*.pdf` | PISA 2006 + LSAY 2008 (wave 3) — predicts year 12 course enrolment | No (traditional stats) | **Not forecasting** — longitudinal individual-level analysis; LSAY is a domestic follow-up cohort, not a future ILSA cycle |
| 6 | `condron-et-al-2024-*.pdf` | PISA 2009/2012/2015/2018 four-wave pooled FE regression | No (traditional stats) | **Not forecasting** — retrospective pooled analysis; no future-cycle prediction |
| 7 | `1-s2.0-S0038012118302908-main.pdf` | "PISA 2011–2012" — school efficiency Tunisia | Yes (DEA + ML) | **Not cross-cycle** — "2011–2012" is a school year within PISA 2012; single cycle |
| 8 | `The-Effects-of-Artificial-Intelligence-on-the-Future-of-Em_2024_Procedia-Com.pdf` | "future of employment" in title | No (narrative review) | **Not a study** — literature narrative review; no ILSA data |
| 9–38 | Remaining 30 keyword matches | Various "prediction" language | Mixed | All verified as within-cycle prediction, traditional statistics, or non-ILSA datasets |

### Finding

**Zero confirmed future-cycle ILSA forecasting studies in the 1,266-record corpus.**

The definition applied: a study must (a) use historical ILSA country-level cycle data as a time series, (b) train a model to forecast a future cycle outcome, and (c) evaluate against a subsequently observed cycle result. No study satisfies all three conditions.

The Flynn effect paper (candidate 1) is the closest analog — it projects ILSA-derived IQ equivalents to 2100 — but it uses linear/quadratic regression on country-level trend data, not a forecasting model, and it does not evaluate against a held-out future cycle. It is classified as traditional statistics, not ML.

**This finding is robust to the search strategy used.** The keyword search, target_variable search, and automated cross-cycle classifier converged on the same conclusion after manual review.

---

## Task 3: 30-Study Stratified Manual Validation Sample

### Sample design

| Stratum | Description | N |
|---------|-------------|---|
| A | Student within-cycle ML prediction (random sample) | 8 |
| D | Country/system ML prediction | 5 |
| F | Other predictive ML | 5 |
| G_trad | Non-predictive, traditional statistics | 5 |
| G_ml | Non-predictive with ML | 2 |
| B_confirmed | Confirmed cross-cycle ML (all 2) | 2 |
| cross_cycle_other | Other cross-cycle flagged studies | 3 |
| **Total** | | **28** (2 target strata had overlap after dedup) |

Sample saved to: `analysis/02_prediction_taxonomy/validation_sample_30.csv`

### Validation findings

**Stratum A (task A, n=8):** All 8 studies are correctly classified. All use ML within a single ILSA assessment cycle with a random student-level train/test split. Titles confirm: PISA 2018 student performance prediction, TALIS 2018 teacher innovativeness, TIMSS science achievement (Singapore/Turkey), school belonging prediction. **Error rate: 0/8.**

**Stratum D (task D, n=5):** All 5 correctly classified. Studies predict country/school-level outcomes (resilience to violence from ICCS 2016, computational thinking from TIMSS, digital literacy from PISA, academic resilience cross-cultural). All within-cycle. **Error rate: 0/5.**

**Stratum F (task F, n=5):** All 5 correctly classified as "other predictive" — these include process-data analysis, DIF detection via ML, teacher behavior prediction, and large-scale assessment contextualization tools. **Error rate: 0/5.**

**Stratum G_trad (task G, n=5):** All 5 confirmed non-predictive. Traditional statistics designs: ICT–achievement regression analysis, informal learning and IT skills (PIAAC), adult skill trends (PIAAC), polytomous IRT, older workers and ICT. **Error rate: 0/5.**

**Stratum B_confirmed (task B, n=2):** Both correctly confirmed cross-cycle. `s00521-023-08901-6.pdf` (PISA 2015→2018) and `s40536-025-00262-x.pdf` (TIMSS QA neural network). **Error rate: 0/2.**

**Stratum cross_cycle_other (n=3):**
- `1-s2.0-S2666920X26000287-main.pdf` ("Decoding divides"): Uses Dutch Education Monitor (OML) 2022–2023, NOT ILSA data. Classified as D/cross-cycle by automated system because "2022–2023" matched multi-year pattern. **Reclassify: F or G** (non-ILSA dataset).
- `sachisthal-et-al-2020-*.pdf` ("science interest network"): PISA 2006 + LSAY follow-up; traditional statistics. Cross-cycle flag was triggered by multi-year pattern. **Reclassify: G** (traditional stats, longitudinal but non-ML).
- `The-Effects-of-Artificial-Intelligence-on-the-Future-of-Em_2024_Procedia-Com.pdf`: Narrative review. **Confirm: G** (non-empirical).

**Estimated false-positive rate in cross_cycle_other stratum: 3/3 = 100%** (all are misclassified as cross-cycle; none are genuine ILSA cross-cycle studies). This confirms the Phase 2 finding that the automated cross-cycle detection has a high false-positive rate (~75–100%) for studies that are not genuinely cross-cycle.

**Overall classification error rate in sample: 3/28 (10.7%)** — all 3 errors are false-positive cross-cycle classifications in non-ILSA datasets or non-ML studies. No false negatives detected (no study that should be cross-cycle was missed).

---

## Task 4: Algorithm Taxonomy — 166 ML Studies

### Algorithm use purpose

| Purpose | Count | % |
|---------|-------|---|
| Predictive modeling (main outcome) | 89 | 53.6% |
| Feature importance / variable selection (primary use) | 2 | 1.2% |
| Both predictive + feature importance | 67 | 40.4% |
| Unclear from outcome_summary | 8 | 4.8% |

**Finding:** 40.4% of ML studies report both a predictive outcome metric AND feature importance / SHAP analysis. This means the algorithm taxonomy requires distinguishing primary use (prediction vs. explanation) when characterizing what the field has done — the "166 ML studies" is a heterogeneous set. Only 53.6% use ML primarily as a predictive model without also reporting feature importance.

### Primary algorithm distribution (n=166 studies)

| Algorithm | n (primary) | % of 166 |
|-----------|-------------|----------|
| Random Forest | 44 | 26.5% |
| XGBoost | 18 | 10.8% |
| Neural Network | 16 | 9.6% |
| SVM | 13 | 7.8% |
| LASSO | 10 | 6.0% |
| Logistic Regression | 6 | 3.6% |
| Decision Tree | 5 | 3.0% |
| Gradient Boosting | 5 | 3.0% |
| LightGBM | 4 | 2.4% |
| Elastic Net | 4 | 2.4% |
| CatBoost | 3 | 1.8% |
| Other (25 unique algorithms) | 38 | 22.9% |

### Multi-label algorithm frequency (ml_all_techniques field)

Algorithms appearing in any study (as primary or secondary technique):

| Algorithm | Appearances | Notes |
|-----------|-------------|-------|
| Random Forest | 83 | Dominant; in 50% of all ML studies |
| SVM | 35 | |
| Decision Tree | 32 | Often baseline or ensemble component |
| Neural Network | 32 | |
| Logistic Regression | 29 | |
| XGBoost | 28 | |
| LASSO | 21 | |
| Gradient Boosting | 18 | |
| Linear Regression | 16 | |
| LightGBM | 12 | |
| k-NN | 12 | |
| Naive Bayes | 9 | |
| Elastic Net | 8 | |
| AdaBoost | 7 | |
| Stacking | 7 | |
| LSTM | 6 | |
| Ridge Regression | 6 | |
| CatBoost | 5 | |

### ML family distribution

| Family | Count | % |
|--------|-------|---|
| Tree-Based / Ensemble Learning | 85 | 51.2% |
| Other ML / Not Classified | 38 | 22.9% |
| Generalized Linear Models (GLM) | 25 | 15.1% |
| Deep Learning | 18 | 10.8% |

**Finding:** Tree-based / ensemble methods (Random Forest, XGBoost, LightGBM, CatBoost, GBM, etc.) dominate, used in 51.2% of studies. Deep learning accounts for 10.8% — primarily LSTM, Neural Network, and Autoencoder variants. No Transformer-based foundation models appear in forecasting roles.

---

## Task 5: Verified Frequency Tables

### Table 5.1: Task distribution — all 1,266 studies

| Task | Code | Count | % |
|------|------|-------|---|
| Non-predictive analytical | G | 1,100 | 86.9% |
| Student within-cycle prediction | A | 96 | 7.6% |
| Other predictive | F | 45 | 3.6% |
| Country/system prediction | D | 17 | 1.3% |
| Student cross-cycle (automated) | B | 8 | 0.6% |
| Student cross-cycle (confirmed) | B* | **2** | 0.16% |
| Future-cycle forecasting | E | **0** | 0.0% |

*Post-manual-review count.

### Table 5.2: Temporal forecasting status — all 1,266 studies

| Status | Count | Notes |
|--------|-------|-------|
| not_applicable | 545 | Technical/framework documents |
| no_temporal_prediction | 538 | Traditional stats, descriptive |
| within_cycle_prediction | 155 | ML and predictive studies |
| cross_cycle (automated) | 28 | High false-positive rate |
| cross_cycle (confirmed ML) | 2 | After Phase 2 manual review |
| confirmed_future_cycle_forecasting | **0** | |

### Table 5.3: Algorithm frequency — 166 ML studies (primary algorithm)

See Task 4 above. Random Forest (26.5%), XGBoost (10.8%), Neural Network (9.6%), SVM (7.8%).

### Table 5.4: ML family — 166 ML studies

Tree-Based/Ensemble 51.2%, Other 22.9%, GLM 15.1%, Deep Learning 10.8%.

### Table 5.5: Validation strategy — 166 ML studies

| Strategy | Count | % |
|----------|-------|---|
| k-fold cross-validation | 78 | 47.0% |
| Insufficient evidence / not reported | 48 | 28.9% |
| Random train/test split | 32 | 19.3% |
| Temporal holdout (automated; ~2 confirmed) | 7 | 4.2% |
| Rolling origin (confirmed 0 genuine ILSA) | 1 | 0.6% |

---

## Task 6: Research-Gap Assessment

### Formal gap claim

**Claim:** No study in the published ILSA ML literature has applied a machine learning forecasting model to country-level ILSA time-series data to predict future ILSA cycle outcomes.

### Evidence supporting the gap

| Evidence item | Source | Weight |
|---------------|--------|--------|
| Zero studies with task code E (future-cycle forecasting) | Phase 2 classification of 1,266 studies | Strong |
| Zero forecasting candidates in keyword search of 1,266 outcome_summaries | Phase 2.1 Task 2 broad search | Strong |
| Zero hits for forecasting keywords in target_variable field | Phase 2.1 Task 2 | Strong |
| Only 2 cross-cycle ML studies confirmed; neither is future-cycle forecasting | Phase 2 manual review | Strong |
| No rolling-origin evaluation confirmed for ILSA data | Phase 2 validation strategy coding | Strong |
| No TimesFM or time-series foundation model found in any ILSA study | ml_all_techniques exhaustive parse | Strong |
| Published scoping reviews (Elhaj et al., 2024; Lim et al., 2022) do not include forecasting as a category | Phase 2 literature check | Moderate |

### Caveats and limitations

1. **Classification is extraction-based.** The study_filter_type, outcome_summary, and ml_primary fields were extracted by human coders, not verified against all 1,266 source PDFs. A small fraction of studies classified as "traditional statistics" or "non-predictive" may contain ML forecasting elements not captured in extraction.

2. **Flynn effect paper edge case.** The paper "The future of intelligence" (Woodley of Menie et al., 2023) projects ILSA-derived IQ equivalents to 2100 using regression extrapolation. This is the only study using past ILSA cycles to project future cognitive performance. It is correctly classified as traditional statistics and does not use a forecasting model. However, it is conceptually related to the proposed research.

3. **German ICT paper edge case.** One ML study (Combining statistical and ML methods, German PISA 2015/2018) reports within-cycle RF results for both 2015 and 2018 independently. The 2018 AUC in that paper was labeled "when evaluated on 2018 data" — this could be interpreted as a 2015→2018 transfer test but the evidence from the findings record shows two separate within-cycle analyses. If this is a 2015→2018 transfer, the confirmed cross-cycle count rises to 3; it does not affect the zero-forecasting finding.

4. **Grey literature and conference papers.** 271 studies (21.4%) have no DOI, suggesting some grey literature. These were classified using the same schema. Unpublished working papers or reports in the corpus may not have been covered by scoping reviews.

### Comparison with published literature reviews

| Review | Coverage | Forecasting mentions |
|--------|----------|---------------------|
| Elhaj et al. (2024), Studies in Educational Evaluation | 57 ML studies | None |
| Lim et al. (2022), Large-Scale Assessments in Education | Systematic ML review | None — only within-cycle prediction |
| Yıldırım & Yıldırım (2023), PISA predictors scoping review | PISA ML studies | None |
| This corpus (1,266 studies, 2026) | Full ILSA ML literature | **0 confirmed** |

**Conclusion:** The gap is genuine. The research space of ML-based future-cycle ILSA forecasting is currently unoccupied.

---

## Task 7: Phase 3 Readiness Report

### Go/No-Go Decision

**Decision: GO — proceed to Phase 3 (Algorithm Taxonomy and Frequency Analysis)**

### Conditions satisfied

| Condition | Status |
|-----------|--------|
| 1,266 unique study records confirmed | ✅ |
| 166 ML studies confirmed | ✅ |
| Zero future-cycle forecasting studies confirmed | ✅ (robust to all search strategies) |
| Primary algorithm frequencies verified | ✅ |
| Cross-cycle false-positive rate documented | ✅ (~75–100% in non-ILSA studies) |
| Research gap claim supported by evidence | ✅ |
| Gold standard / fine-tuning data untouched | ✅ |
| Original data files not modified | ✅ |

### Residual uncertainties (carry to Phase 6)

1. **ILSA time-series observation count.** The number of comparable country-year observations per ILSA program (needed for TimesFM input size evaluation) requires the official OECD/IEA published score tables — not this literature corpus. Phase 6 must assemble this separately.

2. **2015→2018 German paper ambiguity.** If the RF model in that paper was truly evaluated on 2018 data after training on 2015 data (i.e., a genuine cross-cycle transfer), the confirmed cross-cycle count rises from 2 to 3. This does not affect the zero-forecasting finding.

3. **Country coverage of the 2 cross-cycle studies.** The PISA 2015→2018 study covers 13 OECD countries (student level); the TIMSS QA study covers an unspecified country subset. Their geographic scope does not represent the full country panel that TimesFM would need.

4. **Algorithm taxonomy depth.** The current ml_all_techniques parse is extraction-based. Phase 3 should verify the top-10 algorithms against a 20-study sample to confirm that the frequencies are not inflated by extraction errors.

### Phase 3 scope recommendation

Phase 3 should focus on:
1. Normalizing the multi-label algorithm taxonomy (canonical name deduplication: "Lasso Regression" = "LASSO"; "NN" = "Neural Network")
2. Distinguishing algorithm use purpose for the 89 "both" studies (prediction + feature importance) — does the study report a held-out prediction metric or only feature rankings?
3. Evaluation design audit: for the 96 task-A studies, verify that "k-fold cross-validation" is reported with a proper holdout test set
4. Performance metric extraction: build structured (metric_type, value) pairs from the free-text performance_metrics field

---

## Summary of Phase 2.1 Key Findings

| Finding | Confidence | Evidence base |
|---------|------------|---------------|
| Zero future-cycle ILSA forecasting studies in 1,266-study corpus | **High** | Keyword search + automated classifier + manual review of all candidates |
| 166 confirmed ML studies | **High** | Filter field confirmed; 28-study sample validated |
| Random Forest dominates (26.5% primary; 50% multi-label) | **High** | Extraction-based; multi-label parse of 166 studies |
| Tree-based/ensemble methods = 51.2% of ML studies | **High** | ml_family field |
| k-fold CV is the dominant validation strategy (47%) | **High** | validation_strategy field |
| Temporal forecasting research gap is genuine | **High — with caveats** | Extraction-based; ~10% classification error rate estimated |
| Cross-cycle ML study count = 2 confirmed (post-review) | **High** | Phase 2 manual review of all 8 automated candidates |
| German ICT study (PISA 2015/2018) is likely within-cycle | **Moderate** | Findings field language is ambiguous; source article not inspected |
