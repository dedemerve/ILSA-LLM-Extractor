# Few-Shot Rendering Review — v1

**Generated:** 2026-10-06  
**Branch:** experiment/ilsa-domain-finetuning  
**Source file:** `outputs/finetuning/baseline/few_shot_demonstrations_v1.json`  
**Gold source (frozen):** `outputs/finetuning/data/gold_standard_full_corpus_v1.4.csv`  
**Evidence matrix:** `outputs/finetuning/evidence_matrix_v1.3.csv`

---

## Summary

| Demo | Study ID | Program | Records | Status |
|------|----------|---------|---------|--------|
| demo_1 | Teacher-job-satisfaction--TIMSS-2025.pdf | TIMSS 2019 | G0184–G0190 (7) | VERIFIED_WITH_FLAGS |
| demo_2 | thinking-skills-in-teaching-practices.pdf | TIMSS+PISA 2011/12 | G0162–G0165 (4) | VERIFIED_WITH_NULLS |
| demo_3 | jwl-04-2021-0047.pdf | PIAAC Cycle 1 | G0197–G0200 (4) | VERIFIED_WITH_NULLS |
| demo_4 | 1-s2.0-S0022096522001059-main.pdf | PIRLS 2006 | G0069–G0074 (6) | VERIFIED_WITH_NULLS |

**Overall FEW_SHOT_RENDERING_STATUS:** MANUAL_REVIEW_REQUIRED

Reason: Two direction annotation discrepancies in demo_1 (G0186, G0188) require researcher adjudication before the demonstrations can be declared clean. The discrepancies are documented below. All other records are verified from source texts.

---

## Demo 1 — TIMSS 2019 Teacher Job Satisfaction

**Source text:** `Teacher-job-satisfaction--International-evidence-on-the_2025_Studies-in-Educ.txt`  
**Key table:** Table 4 (Standardized regression estimates for pooled international data, N ≈ 13,048 teachers, 46 TIMSS 2019 education systems)

### Per-Record Status

| Record | Predictor | ilsa_variable | effect_size | student_weights | evidence_text | Status |
|--------|-----------|--------------|-------------|-----------------|---------------|--------|
| G0184 | Leadership support | BTBG06L | β=0.266***, SE=0.013 | false | VERIFIED from lines 1195–1197 | VERIFIED |
| G0185 | Student discipline | BTBG07D/E/F/G/H | β=0.356***, SE=0.013 | false | VERIFIED from lines 1200–1202 | VERIFIED |
| G0186 | Teacher workload | BTBG09B/C/D/E/H | β=−0.113***, SE=0.012 | false | VERIFIED from lines 1206–1208 | MANUAL_REVIEW_REQUIRED |
| G0187 | Gender | BTBG02 | β=0.071**, SE=0.022 | false | VERIFIED from lines 1289, 1296 | VERIFIED |
| G0188 | Teaching years | BTBG01 | β=−0.006***, SE=0.001 | false | VERIFIED from lines 1301–1303 | MANUAL_REVIEW_REQUIRED |
| G0189 | Math major | BTBG05A | β=0.023, ns | false | VERIFIED from lines 1354–1355 | VERIFIED |
| G0190 | PD | BTBM23 | β=0.084***, SE=0.008 | false | VERIFIED from lines 1299–1301 | VERIFIED |

### MANUAL_REVIEW_REQUIRED Flags

**G0186 — Teacher workload — Direction annotation discrepancy:**

- Gold annotation: `direction_canonical = "positive"`
- Paper source: Table 4 reports `β = −0.113***` (negative pooled standardized coefficient, SE = 0.012, p < 0.001)
- Narrative evidence: "Teacher workload was negatively related to teacher job satisfaction in 37 countries." (lines 1206–1208)
- Assessment: The direction in the gold record is inconsistent with the pooled regression coefficient and the paper's narrative. The effect is clearly NEGATIVE in the source.
- In the rendered demonstration, `direction` is set to `"positive"` faithfully reflecting the frozen gold value.
- Action required: Researcher must decide whether to (A) accept the demonstration with the incorrectly annotated direction rendered as-is, or (B) revise the gold annotation before rendering. **Gold cannot be modified without explicit researcher authorization.**

**G0188 — Teaching years — Direction annotation discrepancy:**

- Gold annotation: `direction_canonical = "positive"`
- Paper source: Table 4 reports `β = −0.006***` (negative pooled standardized coefficient, SE = 0.001, p < 0.001)
- Narrative evidence: "The relationships between years of teaching, mathematics as a major, and teacher job satisfaction were less consistent, suggesting that these factors may be more context-dependent." (lines 1354–1355)
- Assessment: The direction in the gold record is inconsistent with the pooled regression coefficient. β = −0.006 is statistically significant (p < 0.001) and negative, though the magnitude is very small. The country-level results are described as "inconsistent" (some positive, some negative), but the pooled estimate is negative.
- In the rendered demonstration, `direction` is set to `"positive"` faithfully reflecting the frozen gold value.
- Action required: Same as G0186. Researcher must adjudicate.

### Student Weights Note

Justification explicitly stated in source at lines 865–869: "we made a deliberate choice not to employ teacher weights in our analysis. The use of teacher weights is typically relevant when considering student-level outcomes or characteristics, which fall outside the scope of our investigation." Rendered as `student_weights_used: false`. ✓

---

## Demo 2 — TIMSS 2011 + PISA 2012 Thinking Skills

**Source text:** `thinking-skills-in-teaching-practices-relationship-with-students-achievement-in-.txt`  
**Key tables:** Table 4 (impact on TIMSS and PISA scores), Table 5 (PISA with TIMSS control)

### Per-Record Status

| Record | Predictor | ilsa_variable | effect_size | student_weights | evidence_text | Status |
|--------|-----------|--------------|-------------|-----------------|---------------|--------|
| G0162 | LO practices | null | Mixed (see JSON) | null | VERIFIED from lines 538–540 | VERIFIED_WITH_NULLS |
| G0163 | HO practices | null | Mixed (see JSON) | null | VERIFIED from lines 534–537 | VERIFIED_WITH_NULLS |
| G0164 | Student SES | null | ≈0.15/0.11*** | null | VERIFIED from lines 283–287 | VERIFIED_WITH_NULLS |
| G0165 | School/teacher chars | null | null | null | VERIFIED from lines 288–292 | VERIFIED_WITH_NULLS |

**ilsa_variable = null for all records:** The paper derives LO/HO practice scales from TIMSS 2011 teacher questionnaire Question 19 items and measures SES via binary student questionnaire items, but does not report explicit TIMSS/PISA variable codes. Null is the correct value.

**student_weights_used = null for all records:** Paper does not state whether sampling weights were applied. Gold annotation is "not_reported". Rendered as null.

---

## Demo 3 — PIAAC Cycle 1 Informal Learning Behaviors

**Source text:** `jwl-04-2021-0047.txt`  
**Key table:** Table 4 (hierarchical multiple regression, n = 122,897)

### Per-Record Status

| Record | Predictor | ilsa_variable | effect_size | student_weights | evidence_text | Status |
|--------|-----------|--------------|-------------|-----------------|---------------|--------|
| G0197 | Workplace autonomy | D_Q11a/b/c | b=0.15, p<0.001 | null | VERIFIED from lines 854–855 | VERIFIED |
| G0198 | Interaction in workplace | F_Q02b/c/e, F_Q03b, F_Q04a/b | b=0.31, p<0.001 | null | VERIFIED from lines 471–472 | VERIFIED |
| G0199 | Readiness to learn | I_Q04b/d/h/j/l/m | null | null | VERIFIED from lines 854–855 | VERIFIED_WITH_NULLS |
| G0200 | Demographic controls | null | null | null | VERIFIED from lines 472–475 | VERIFIED_WITH_NULLS |

**G0199 effect_size = null:** The standardized coefficient for Readiness to Learn could not be extracted with sufficient confidence from the PDF-extracted table layout. The positive direction and significance are confirmed by the narrative (lines 854–855). Null is reported rather than a potentially incorrect value.

**G0200 ilsa_variable = null:** Demographic controls (age, education, sex, contract type, sector, occupation, training) are standard background variables, not a single PIAAC instrument scale with specific variable codes relevant to the evidence row. Null is correct.

---

## Demo 4 — PIRLS 2006 Reading and Theory of Mind

**Source text:** `1-s2.0-S0022096522001059-main.txt`  
**Key tables:** Table 3 (concurrent model, Time 3), Table 4 (longitudinal model, Time 1 → Time 3)

### Per-Record Status

| Record | Predictor | Model | ilsa_variable | effect_size | student_weights | evidence_text | Status |
|--------|-----------|-------|--------------|-------------|-----------------|---------------|--------|
| G0069 | Reading motivation | Concurrent (T3) | null | b=−.22*,SE=.10 | true | VERIFIED from lines 530–531 | VERIFIED_WITH_NULLS |
| G0070 | Gender | Concurrent (T3) | null | b=−.17**, SE=.07 | true | VERIFIED from lines 524–527 | VERIFIED_WITH_NULLS |
| G0071 | Reading motivation | Longitudinal (T1→T3) | null | b=.10 (ns) | true | VERIFIED from lines 534–536 | VERIFIED_WITH_NULLS |
| G0072 | Gender | Longitudinal (T1→T3) | null | b=−.17**, SE=.07 | true | VERIFIED from lines 535–536 | VERIFIED_WITH_NULLS |
| G0073 | Parental education | Longitudinal (T1→T3) | null | b=.17*, SE=.08 | true | VERIFIED from lines 535–536 | VERIFIED_WITH_NULLS |
| G0074 | SES | Concurrent (T3) | null | b=.16*, SE=.08 | true | VERIFIED from lines 638–639 | VERIFIED_WITH_NULLS |

**ilsa_variable = null for all records:** This is a UK longitudinal cohort study that uses PIRLS 2006 student questionnaire items as an instrument (not a PIRLS re-analysis). The paper does not provide explicit PIRLS variable codes (e.g., ASBREADM). Null is correct.

**direction = not_reported for all records:** Gold annotation uses `direction_zero_ambiguous` flag for all 6 records. The annotator originally recorded direction = 0 (null/not_applicable), which codebook maps to `not_reported`. The paper reports both positive (parental education, SES, concurrent fiction) and negative (gender, reading motivation) effects; the not_reported annotation reflects annotator uncertainty about which direction governs the ILSA-relevant construct, not absence of data.

**effect_size sign convention:** Effect sizes show negative b values for Gender and Reading motivation (concurrent model) due to PDF text extraction stripping minus signs from CIs. Effect sizes have been reconstructed from CI inspection and reported with correct signs in the demonstration JSON.

---

## Cross-Cutting Notes

1. **No fabricated values.** All ilsa_variable codes, effect sizes, and evidence_text strings are sourced from the actual paper texts. Fields that could not be confirmed are set to null.

2. **Verbatim evidence_text.** All evidence_text values are verbatim excerpts from the source text files. No paraphrasing was applied.

3. **Direction rendering rule applied (demo_1).** demo_1 gold records have `direction_canonical = "positive"` (string, not "1"), so no numeric-to-string transformation was required for this study. The rendering rule (1 → positive) was not triggered for any record in these 4 demonstrations.

4. **plausible_values_handling normalization.** demo_2 gold stores "rubin_rules (5 PVs via Rubin)"; rendered as "rubin_rules" per canonical schema vocabulary.

5. **effect_type normalization.** demo_4 gold stores "predictive"; rendered as "E_predictive" per §3.2 schema vocabulary.

---

## Required Researcher Actions

| # | Item | Urgency | Description |
|---|------|---------|-------------|
| 1 | G0186 direction discrepancy | HIGH | Gold says positive; paper says β=−0.113***. Decide: (A) accept demonstration as-is with mismatched direction, or (B) authorize gold correction before freezing. |
| 2 | G0188 direction discrepancy | HIGH | Gold says positive; paper says β=−0.006***. Same decision needed as G0186. |
| 3 | demo_1 overall usability | MODERATE | With 2 direction annotation errors, does demo_1 remain suitable as a training demonstration? Alternative: replace with a different TIMSS study or proceed with annotation noting the flags. |

**The demonstrated direction values (positive) are rendered faithfully from the gold. The model will learn that workload and teaching years are positively associated with job satisfaction — the opposite of what the paper reports. This may introduce noise into the baseline's learning signal for these predictors.**

---

## What Was NOT Done

Per global pipeline invariants:
- Gold standard files were NOT modified.
- Baseline inference was NOT run.
- Validation or test set files were NOT accessed.
- Stage 5 outputs were NOT modified.
