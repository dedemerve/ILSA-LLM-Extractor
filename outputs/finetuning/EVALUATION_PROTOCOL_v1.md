# Evaluation Protocol — LLM Extraction for ILSA Evidence

**Version:** 1.0  
**Date:** 2026-10-06  
**Status:** FROZEN DESIGN — do not modify after 2026-10-12 reliability freeze  
**Author:** Merve Dede

---

## 1. Motivation

The thesis pipeline is:

```
Source PDFs
    ↓
LLM extraction layer (annotation)
    ↓
Evidence matrix (predictor → construct → effect)
    ↓
Predictor weight construction (W_j, W_jpd)
    ↓
Forecasting (PISA score prediction)
```

Fine-tuning affects only the first step. To evaluate whether fine-tuning is actually useful, we must answer:

> Does fine-tuning improve extraction quality, and does that improvement propagate to predictor weights and ultimately to forecasting accuracy?

Field-level F1 alone cannot answer this. This protocol defines how both questions will be evaluated.

---

## 2. Evaluation sequence (frozen)

```
Now (2026-10-06):   Few-shot baseline design + data preparation (this document)
2026-10-12+:        Second-pass intra-rater reliability
Post-reliability:   Gold standard freeze (evidence_matrix_v1.3 → frozen)
                    Split freeze (train=30 / val=6 / test=6 studies, seed=2026)
                    Few-shot baseline run
                    Results saved → baseline_results_frozen.json
                    LoRA fine-tuning
                    Evaluation on same frozen test set
                    Downstream comparison (W → forecasting)
If needed:          SFT only if LoRA shows no improvement over baseline
```

**Critical rule:** The gold test set (6 studies, 35 rows) must not be used for prompt engineering, example selection, or hyperparameter tuning before the baseline run is recorded.

---

## 3. Annotation input/output format

### 3.1 Input

```
ILSA ANNOTATION TASK

PROGRAM: {program}
CYCLE: {cycle}
SOURCE: {file_name}

SOURCE TEXT (abstract + methods + results excerpt):
{source_text}

TASK: Extract all predictor–outcome evidence pairs from this text.
For each pair, provide the fields below.
```

### 3.2 Output schema (per evidence pair)

```json
{
  "predictor": "raw predictor string as it appears in the source text",
  "ilsa_variable": "ILSA survey variable code if identifiable (e.g. BSDGEDUP, TOTWGT), else null",
  "canonical_construct": "canonical construct label from codebook v1.3",
  "outcome": "target variable as described in the source text",
  "effect_type": "E_assoc | E_predictive | E_cross_cycle | E_cross_country",
  "direction": "positive | negative | mixed | null | not_reported",
  "effect_size": "reported effect size or coefficient, else null",
  "plausible_values_handling": "rubin_rules | average_pv | single_pv | all_pvs | not_applicable | not_reported",
  "student_weights_used": true,
  "evidence_text": "verbatim sentence(s) from source text supporting this extraction"
}
```

### 3.3 Design rationale

The chain preserved from thesis framework:
```
study → predictor → ILSA variable → construct → outcome → effect_type → effect_size → direction
```

`evidence_text` is mandatory — enables evidence attribution accuracy measurement and prevents unsupported inference.

`ilsa_variable` is nullable — not all studies report variable codes.

---

## 4. Evaluation metrics

### 4.1 Field-level metrics (per Tier-1 field)

For each field in `{program, cycle, canonical_construct, direction, effect_type, plausible_values_handling}`:

| Metric | Definition |
|--------|-----------|
| Precision | TP / (TP + FP) |
| Recall | TP / (TP + FN) |
| F1 | harmonic mean of P and R |
| Exact match | predicted == gold (string equality, lowercased) |

### 4.2 Construct-level metrics

| Metric | Definition |
|--------|-----------|
| Construct mapping accuracy | % rows where predicted canonical_construct == gold |
| Construct confusion matrix | per-construct precision/recall |
| NEW: UNMATCHED rate | % rows where model outputs an unrecognized construct |
| NEW: Hallucination rate | % rows with a construct not in codebook v1.3 |

### 4.3 Evidence attribution

| Metric | Definition |
|--------|-----------|
| Attribution present | % rows where evidence_text is non-null |
| Attribution accuracy | % rows where evidence_text is a verbatim or paraphrase of source text (human-scored subset) |
| Unsupported inference rate | % rows where canonical_construct is assigned without supporting evidence_text |

### 4.4 Construct frequency strata (report separately)

| Stratum | Constructs | Current rows |
|---------|-----------|-------------|
| HIGH | SES_COMPOSITE, GENDER, SELF_CONCEPT, ICT_HOME_ACCESS, DISCIPLINARY_CLIMATE, IMMIGRANT_STATUS, SCHOOL_ICT_INFRASTRUCTURE | 9 constructs, ≥8 rows each |
| MID | EDUCATIONAL_ATTAINMENT, TEACHER_ICT_USE, MOTIVATION, ANXIETY, NUMERACY_OUTCOME, PROCESS_DATA_BEHAVIOR | 6 constructs, 4–7 rows |
| LOW | TEACHER_SELF_EFFICACY_CLS_MGMT, INSTRUCTION_TIME, TEACHING_QUALITY, SCHOOL_BELONGING, PARENTAL_EDUCATION, HOME_LITERACY_ACTIVITIES, SCHOOL_LOCATION, PRIVATE_TUTORING, TEACHER_EXPERIENCE, TEACHER_PD, JOB_TRAINING, AGE_GROUP, LITERACY_OUTCOME | 13 constructs, 2–3 rows |
| SINGLE | HOMEWORK_TIME, SCHOOL_LEADERSHIP, WORKPLACE_AUTONOMY | 3 constructs, 1 row |

Reporting construct mapping accuracy aggregated across all constructs conceals systematic failures in LOW/SINGLE strata. Report per-stratum F1.

### 4.5 Independent training example count

169 FORECASTABLE rows ≠ 169 independent training examples.

| Split | Studies | Rows | FORECASTABLE rows | Mean rows/study |
|-------|---------|------|-------------------|-----------------|
| train | 30 | 187 | 153 | 6.2 |
| val | 6 | 24 | 16 | 4.0 |
| test | 6 | 35 | 29 | 5.8 |

**Effective sample size note:** For LoRA training, report results at both row level and study level. Study-level aggregation (mean construct-mapping accuracy per study) is the more conservative and scientifically defensible estimate.

### 4.6 SCHOOL_BELONGING split note

`SCHOOL_BELONGING` appears in the test set (2 rows) but not in the train set. This is a known split artifact from the stratified-by-program allocation. Report its performance separately; do not use it to claim generalization failure.

---

## 5. Downstream evaluation (W → forecasting comparison)

### 5.1 Two conditions

| Condition | Extraction | Rest of pipeline |
|-----------|-----------|-----------------|
| BASE | Few-shot extraction (no fine-tuning) | Identical: W_j construction → forecasting |
| FINETUNED | LoRA fine-tuned extraction | Identical: W_j construction → forecasting |

**Critical rule:** W_j and W_jpd construction, LOCO folds, and forecasting results from Stage 5 are FROZEN. The comparison uses the Δ between base-model-derived weights and fine-tuned-model-derived weights applied to the same forecasting framework.

Do NOT modify Stage 5 outputs. Do NOT re-run LOCO.

### 5.2 Downstream metrics

| Metric | Definition |
|--------|-----------|
| ΔW_j | Change in predictor weights between BASE and FINETUNED conditions |
| ΔMAE | Change in country-level forecasting MAE |
| ΔPISA coverage | Change in % of PISA 2022 countries with complete predictor coverage |
| Construct alignment | Overlap between extraction-derived constructs and Stage 5 active predictors |

### 5.3 Research question answered

> "Does fine-tuning improve extraction accuracy alone, or does it also improve literature-derived predictor representation and downstream forecasting?"

If ΔMAE ≈ 0 despite improved extraction F1, the fine-tuning effect is annotation-quality only (useful for replication) but not forecasting-critical.

If ΔMAE > 0, fine-tuning contributes directly to the thesis's primary claim.

---

## 6. Few-shot baseline design

### 6.1 Shot selection strategy

- Do NOT use test set examples as few-shot demonstrations.
- Select 3–5 demonstrations from the train set, stratified:
  - 1 PISA study (HIGH construct, clear direction)
  - 1 TIMSS study (MID construct, rubin_rules PV handling)
  - 1 PIAAC study (labor market outcome, NOT_FORECASTABLE example included)
  - Optional: 1 PIRLS or TALIS study for diversity

- Demonstrations must be finalized before the baseline run and frozen in `few_shot_examples_frozen.json`.

### 6.2 Prompt engineering constraint

Prompt engineering (rewording system prompt, adjusting output format, adding constraints) is permitted only using the val set.

The test set is locked until:
1. Reliability ≥ 0.61 confirmed (post 2026-10-12)
2. Baseline prompt is frozen (committed to git)
3. Single baseline run is executed and results saved

No second run on the test set to "fix" the prompt.

### 6.3 Baseline output storage

```
outputs/finetuning/baseline/
  few_shot_examples_frozen.json    ← frozen shot selection
  baseline_prompt_frozen.txt       ← frozen system + user prompt template
  baseline_predictions_test.json   ← model outputs on test set (6 studies)
  baseline_results_frozen.json     ← field-level metrics, stratified results
```

---

## 7. LoRA design (post-reliability)

To be specified after baseline results are recorded. Provisional:

- Base model: TBD (Claude API fine-tuning / Llama-3 8B / Mistral 7B)
- Adapter rank: r=8 initially, r=16 if underfitting
- Target modules: q_proj, v_proj (attention only)
- Learning rate: 1e-4
- Epochs: 3–5 with early stopping on val set
- Training examples: FORECASTABLE rows only (153 train rows)
- Excluded: UNMATCHED, NOT_FORECASTABLE rows

**Class weighting:** Do not oversample LOW/SINGLE constructs to match HIGH constructs. The sparsity is a real property of the literature. Instead:
- Report per-stratum performance separately
- Use label smoothing if training loss is dominated by HIGH constructs
- Consider curriculum learning: HIGH constructs first, then MID, then LOW

---

## 8. Pre-execution checklist (gate before any model run)

- [ ] Second-pass reliability ≥ 0.61 (post 2026-10-12)
- [ ] evidence_matrix_v1.3.csv frozen (no further construct remapping)
- [ ] Split frozen (30/6/6, seed=2026)
- [ ] few_shot_examples_frozen.json committed
- [ ] baseline_prompt_frozen.txt committed
- [ ] Test set not inspected for prompt engineering
- [ ] Stage 5 outputs confirmed unchanged
- [ ] W_j, W_jpd confirmed unchanged

---

## 9. What NOT to do

- Do NOT run the baseline before reliability is confirmed
- Do NOT use the test set for prompt engineering
- Do NOT oversample LOW constructs to inflate F1
- Do NOT report aggregate F1 without per-stratum breakdown
- Do NOT treat 169 rows as 169 independent examples
- Do NOT interpret ΔMAE = 0 as fine-tuning failure without checking extraction F1
- Do NOT start SFT before LoRA is evaluated
- Do NOT modify Stage 5 forecasting outputs

---

*Protocol v1.0 — frozen design — 2026-10-06*
