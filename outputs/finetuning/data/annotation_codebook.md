# ILSA Fine-Tuning Annotation Codebook

**Version:** 1.0  
**Date:** 2026-10-05  
**Branch:** experiment/ilsa-domain-finetuning  
**Status:** Active — revised from Phase 1 protocol after pilot annotation

---

## PRINCIPLE: SOURCE TEXT FIRST

The human annotator must always start from the source text and form an independent judgment.

The LLM extraction is a **candidate label only**. Never accept a LLM label without confirming it from the source text.

For each field: SOURCE TEXT → HUMAN JUDGMENT → FINAL GOLD LABEL.

---

## Tier-1 Fields (Forecasting-critical)

### 1. program

**Definition:** The ILSA program whose microdata was used.  
**Allowed values:** PISA | TIMSS | TIMSS_G4 | PIRLS | TALIS | ICCS | ICILS | PIAAC  
**Multiple values:** Use | separator for multi-program studies.  
**Source of truth:** Study abstract and methods section. Must be explicitly stated.  
**Common errors:**
- IEA publications may describe multiple cycles of TIMSS/PIRLS — list all
- Some studies use non-standard ILSA data (e.g., "TIMSS-like national assessments") — do NOT code as TIMSS unless the standard TIMSS microdata was used
- ICCS/ICILS are sometimes confused with each other

### 2. cycle

**Definition:** The year(s) of the ILSA data used.  
**Format:** "PISA 2018" or "TIMSS 2019" or "PISA 2012; PISA 2015" for multi-cycle.  
**Source of truth:** Methods section ("We used PISA 2018 data from...").  
**Common errors:**
- Year of publication ≠ year of data cycle
- Some studies use pooled cycles — list all

### 3. target_variable

**Definition:** The outcome variable the model predicts or the dependent variable in the regression.  
**Format:** Descriptive text, 5–30 words.  
**Rule:** Do NOT use the LLM placeholder "Primary study outcome (heuristic migration)" — this is an LLM failure mode. Find the actual outcome from the source text.  
**Common errors:**
- Confusing multiple outcomes (e.g., math + reading + science achievement)
- Treating the ILSA assessment score as the only possible outcome (some studies predict attitudes, participation, teacher outcomes)

### 4. predictor

**Definition:** The variable used as a predictor/feature in the ML model or the independent variable in regression.  
**Rules:**
- Record the exact term used in the source (e.g., "ESCS index" not "SES")
- For multiple predictors, create separate rows
- Do NOT include the outcome variable as a predictor
- Confounders that appear only in Table 1 are NOT predictors
- The top-ranked predictors (by feature importance, beta magnitude, or SHAP value) take precedence

### 5. canonical_construct

**Definition:** The canonical predictor from the 19-construct vocabulary.  
**Allowed values:** SES_COMPOSITE | GENDER | PARENTAL_EDUCATION | HOME_RESOURCES | IMMIGRANT_STATUS | LANGUAGE_BACKGROUND | SCHOOL_BELONGING | ANXIETY | MOTIVATION | SELF_CONCEPT | SELF_REGULATION | HOME_LITERACY_ACTIVITIES | DISCIPLINARY_CLIMATE | INSTRUCTIONAL_TIME | CURRICULUM_TYPE | SCHOOL_RESOURCES | EMOTIONAL_SUPPORT | ICT_HOME_ACCESS | GDP_EXPENDITURE | TEACHER_SELF_EFFICACY_CLS_MGMT | UNMATCHED  
**Rules:**
- UNMATCHED: use when no canonical construct fits (record the original predictor_raw exactly)
- Do NOT map teacher-level ICT variables → ICT_HOME_ACCESS (ICT_HOME_ACCESS is a student home-access construct)
- Do NOT map "Gender gap" (aggregated) → GENDER (individual sex/gender variable)
- ITSEX as individual variable → GENDER ✓
- ESCS → SES_COMPOSITE ✓
- BSDGEDUP (inverted scale) → PARENTAL_EDUCATION (document direction reversal)

### 6. direction

**Definition:** Direction of the association between predictor and outcome.  
**Allowed values:** + | - | 0 | not_reported  
**Rules:**
- + : positive association (higher predictor → higher outcome)
- - : negative association (higher predictor → lower outcome)
- 0 : statistically null/non-significant result explicitly reported
- not_reported: direction not determinable from source text
- For reversed scales: first record direction_original (as in source), then direction_canonical (after reversal)
- NaN in the evidence matrix means not_reported — **do NOT interpret NaN or 0 as a meaningful null result without source confirmation**
- Do NOT infer direction from feature importance scores without sign information

### 7. effect_type

**Definition:** The type of evidence presented.  
**Allowed values:** predictive | associative | descriptive | unspecified  
**Rules:**
- predictive: study explicitly frames the analysis as prediction/classification (uses ML, reports accuracy/AUC/RMSE)
- associative: study reports correlation, regression coefficients, or association without prediction framing
- descriptive: descriptive statistics, factor analysis, measurement invariance — NOT a prediction
- A study using SHAP/LIME does NOT automatically establish causality
- A study using "random forest for prediction" IS predictive even if it reports variable importance

---

## Tier-2 Fields (Evidence strength)

### 8. performance_metrics

**Definition:** The model performance statistics reported.  
**Format:** Exact values with metric name, e.g., "Accuracy=0.826, AUC=0.739"  
**Rule:** Copy from source text. Do not compute derived metrics.  
**If absent:** "not_reported"

### 9. standardized_conclusion

**Definition:** 2–3 sentence summary of the key finding, in standardized language.  
**Format:** "Using [DATASET], the study predicted [OUTCOME] using [ML method]. Top predictors were [X, Y, Z]. [CONSTRUCT] showed the strongest positive/negative association."  
**Rule:** Do NOT reproduce LLM verbatim — rephrase to remove hallucinations and placeholder language.

### 10. ml_techniques

**Definition:** The primary ML method(s) used.  
**Examples:** deep learning (Jordan NN) | gradient boosting | random forest | logistic regression | MCA | SEM  
**Rule:** Non-ML statistical methods (HLM, IRT, CFA) → code the actual method, note as not_ml=True if needed.

---

## Tier-3 Fields (Methodology)

### 11. plausible_values_handling

**Allowed values:** all_pvs | single_pv | not_applicable | not_reported  
**Rules:**
- all_pvs: study uses all 5–10 PVs with Rubin's rules or equivalent pooling procedure (rubin_rules → all_pvs)
- single_pv: study uses only 1 PV (PV1MATH etc.) without pooling
- not_applicable: study does NOT use ILSA continuous achievement outcomes (e.g., binary classification, teacher outcomes, attitude scales)
- not_reported: PV handling not described in source
- irt_theta / wle: non-standard; code as single_pv and flag

### 12. student_weights_used

**Allowed values:** True | False | not_reported  
**Rule:** Check methods section for mention of TOTWGT, SENWGT, or "student final weight"

### 13. replicate_weights_used

**Allowed values:** True | False | not_reported  
**Rule:** Check methods section for mention of BRR, jackknife, or "balanced repeated replication"

### 14. missing_data_handling

**Allowed values:** listwise_deletion | multiple_imputation | mean_imputation | not_reported | not_applicable  
**Rules:**
- knn_imputation → multiple_imputation (HIGH confidence normalization)
- single_imputation → mean_imputation (MEDIUM confidence — verify from text)

### 15. research_design_type

**Allowed values:** cross_sectional | longitudinal | experimental | not_reported  
**Rules:**
- predictive / exploratory / causal_observational → cross_sectional (these are LLM drift values)
- longitudinal: explicitly uses data from multiple time points with the same participants
- experimental: intervention with random assignment (rare in ILSA research)

---

## Additional Evidence Fields

### 16. source_supported_claim

**Allowed values:** True | False | partial  
**Definition:** Does the source text support the extraction claim?  
**Rule:** False if the LLM extracted something not present in the source. Partial if the extraction is approximately correct but imprecise.

### 17. source_location

**Format:** "Section + page/sentence reference", e.g., "Methods, p.3, line 12" or "Abstract, paragraph 2"  
**Rule:** Required for all Tier-1 and Tier-2 fields in gold-verified records.

### 18. evidence_type

Same as effect_type — kept separate for legacy compatibility.

### 19. study_design_context

**Format:** Free text, max 100 chars.  
**Examples:** "Multilevel regression, cross-sectional, single-country", "Tiered ML, binary classification, SMOTE balancing"

### 20. ambiguity_flag

**Allowed values:** none | canonical_mapping_error | canonical_mapping_ict_teacher | outcome_placeholder | pv_handling_mixed | direction_encoding_inconsistency | direction_zero_ambiguous | self_referential_predictor | inappropriate_study_type | methodological_paper | descriptive_report | effect_type_wrong | other  
**Rule:** Always set. Multiple flags use | separator.

### 21. adjudication_note

**Format:** Free text, max 200 chars.  
**Rule:** Required for gold_verified_after_adjudication records. Must explain what changed and why.

---

## Gold Status Definitions

| Status | Meaning |
|--------|---------|
| gold_verified | Source text read; all Tier-1/2 fields confirmed; no disagreements |
| gold_verified_after_adjudication | Source text read; labels corrected/adjusted; adjudication recorded |
| ambiguous | Source text read; fields cannot be reliably assigned from text |
| excluded | Source text confirms this record is NOT an ML prediction study |
| silver_only | Source text not available |
| annotation_pending | Not yet annotated (Phase 3 in-progress) |

---

## Exclusion Criteria (after source text inspection)

Exclude if the source text confirms:
1. Descriptive country report / OECD country note (no ML model)
2. Conceptual / theoretical paper (no empirical ILSA analysis)
3. Macroeconomic aggregate study (GDP growth as outcome, national-level)
4. Data harmonization / methodology paper (no student outcome prediction)
5. Measurement invariance study (CFA/MGCFA — effect_type = descriptive)
6. Review article / meta-analysis (secondary analysis only)

**Do NOT exclude** just because the ML method is simple or the study is older.

---

## Canonical Construct Clarifications (from Pilot)

### ICT_HOME_ACCESS
**Applies to:** Student-level ICT access at home (PISA/TIMSS/ICILS: ICTHOME, HOMCON, etc.)  
**Does NOT apply to:** Teacher ICT professional development, ICT in classroom (TALIS variables)  
**For teacher ICT variables in TALIS:** Use UNMATCHED_TEACHER_ICT (pending canonical vocabulary update) or UNMATCHED

### SELF_CONCEPT
**Applies to:** Student academic self-concept, creative self-efficacy, reading self-concept  
**Also applies to:** Teacher self-efficacy (TALIS TSEFF) — acceptable approximation  
**Does NOT apply to:** Confidence in a specific subject (too specific; use UNMATCHED if not in canonical)

### direction = 0
**Meaning in evidence matrix:** Used inconsistently. May mean (a) null result or (b) not_reported.  
**Gold annotation rule:** If source text reports a null result → use 0. If direction simply not extractable → use not_reported. Do NOT copy 0 from the evidence matrix without source confirmation.

---

*Codebook version 1.0 — established 2026-10-05 from pilot annotation of 18 records across 7 programs*
