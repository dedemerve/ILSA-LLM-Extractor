# Step 3 Readiness Report — Few-Shot Baseline

**Generated:** 2026-10-06  
**Branch:** experiment/ilsa-domain-finetuning  
**Commit:** f0845198f622a170175e3aa9f08e3dee295b7839

---

## Gate Status

| Gate | Status | Evidence |
|------|--------|----------|
| TRAINING_POPULATION | READY | See below |
| FEW_SHOT_EXAMPLES | NOT_FROZEN | few_shot_freeze_audit_v1.4.json |
| BASELINE_PROMPT | NOT_FROZEN | baseline_prompt_audit_v1.4.json |
| VALIDATION_SOURCES | COMPLETE | validation_source_audit_v1.4.json |

---

## Decision

**STOP — Do not proceed to Step 3 (few-shot baseline).**

The gate condition requires ALL four items to be READY/FROZEN/COMPLETE. Two blockers remain:

1. **FEW_SHOT_EXAMPLES = NOT_FROZEN**
2. **BASELINE_PROMPT = NOT_FROZEN**

---

## 1. TRAINING_POPULATION — READY

The three counts (169, 153, 131) have been resolved:

| Count | Source | Definition | Verdict |
|------:|--------|-----------|---------|
| 169 | EVALUATION_PROTOCOL_v1.md §4.5 heading; pipeline execution prompt | train FORECASTABLE (153) + val FORECASTABLE (16) = 169 rows available for model development. NOT a training-only count. Also coincidentally equals total gold_verified rows. | PROTOCOL HEADING AMBIGUITY — not a separate training figure |
| 153 | EVALUATION_PROTOCOL_v1.md §4.5 table and §7; evidence_matrix_v1.3.csv construct_status | Training rows with construct_status == 'FORECASTABLE' per v1.3 explicit column. Includes UNMATCHED_TEACHER_ICT rows designated FORECASTABLE. Excludes 34 NOT_FORECASTABLE train rows. | AUTHORITATIVE PER PROTOCOL |
| 131 | preflight_v1.4_report.json | canonical_construct != 'UNMATCHED' (wrong filter — missed UNMATCHED_TEACHER_ICT, included NOT_FORECASTABLE rows) | INCORRECT — discard |

**V1.4 note:** G0244 (train) was FORECASTABLE in v1.3 but was adjudicated to UNMATCHED in v1.4. V1.4-consistent count = 152. This 1-row change is a documented consequence of the freeze adjudication. No methodological conflict.

**Authoritative training population for LoRA:**
- Studies: 30
- Total rows: 187
- FORECASTABLE rows (protocol figure, v1.3): **153**
- FORECASTABLE rows (v1.4-adjusted, excluding G0244): **152**
- NOT_FORECASTABLE rows: 34 (exclude from LoRA training)

---

## 2. FEW_SHOT_EXAMPLES — NOT_FROZEN

**Audit file:** `outputs/finetuning/audit/few_shot_freeze_audit_v1.4.json`

All 4 shots (PISA, TIMSS, PIAAC, PIRLS) are from the training split — no contamination.  
All 4 carry status: `"CANDIDATE — requires human review before freezing (post-reliability 2026-10-12)"`

Reliability precondition MET (κ=0.871 ≥ 0.61, frozen 2026-10-06).  
No explicit human freeze decision or approval record found in the repository.

**Required action:** Review the 4 shots in `outputs/finetuning/baseline/few_shot_examples_frozen.json`, confirm they are acceptable as demonstration examples, then commit a version with `status: "FROZEN"` for each shot.

---

## 3. BASELINE_PROMPT — NOT_FROZEN

**Audit file:** `outputs/finetuning/audit/baseline_prompt_audit_v1.4.json`

The required file `outputs/finetuning/baseline/baseline_prompt_frozen.txt` does not exist.

The general extraction prompt at `prompts/extraction_system_prompt.txt` is NOT equivalent — it covers the full metadata schema, not the 10-field evidence-pair extraction schema specified in EVALUATION_PROTOCOL_v1.md §3.

**Required action:** Create `outputs/finetuning/baseline/baseline_prompt_frozen.txt` implementing:
- System message (role, task definition, construct vocabulary, few-shot demonstration format)
- User template with slots: `{program}`, `{cycle}`, `{file_name}`, `{source_text}`
- Output schema: 10 fields per evidence pair (predictor, ilsa_variable, canonical_construct, outcome, effect_type, direction, effect_size, plausible_values_handling, student_weights_used, evidence_text)
- Model, temperature, max_tokens specifications
- Freeze and commit

---

## 4. VALIDATION_SOURCES — COMPLETE

**Audit file:** `outputs/finetuning/audit/validation_source_audit_v1.4.json`

All 6 validation study source texts are available at `outputs/finetuning/source_recovery/texts/`:

| Study | Text available | FORECASTABLE rows |
|-------|---------------|-------------------|
| 1-s2.0-S0191886926001704-main.pdf | YES (45 KB) | 1 |
| 5.pdf (Song & Cutumisu 2024) | YES (114 KB) | 3 |
| InfocomJournal_2024_SpecISS_CogInf_CogAspVR_2.pdf | YES (97 KB) | 0 |
| Patterns-of-teacher-collaboration-2022.pdf | YES (49 KB) | 5 |
| Review of Income and Wealth - 2024 - Cammeraat.pdf | YES (106 KB) | 3 |
| lopez-agudo-marcenaro-gutierrez-2025.pdf | YES (60 KB) | 4 |

Note: 2 filename mismatches (Cammeraat, Lopez-Agudo) between split manifest and actual text files — documented, does not block extraction.

Note: InfocomJournal study has 0 FORECASTABLE rows in v1.3. Include in evaluation regardless — contributes to FN counting.

---

## Summary: PIPELINE_REMAINS_BLOCKED

**Next required actions (in order):**

1. **Review and freeze the few-shot examples** — inspect `outputs/finetuning/baseline/few_shot_examples_frozen.json`, confirm the 4 shots, update status to FROZEN, commit.

2. **Create and freeze the baseline prompt** — draft `outputs/finetuning/baseline/baseline_prompt_frozen.txt` with the exact system message, few-shot format, user template, model, temperature, and output schema. Commit.

3. **Only then**: run the few-shot baseline on the 6 validation studies.

Do not access the test set. Do not train LoRA. Do not perform SFT.
