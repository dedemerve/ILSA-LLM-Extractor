# Step 3 Freeze Readiness — v1.4b (Round 3 Resolved)

**Generated:** 2026-10-06  
**Branch:** experiment/ilsa-domain-finetuning  
**Supersedes:** step3_freeze_readiness_v1.4.md  
**Protocol source:** EVALUATION_PROTOCOL_v1.md (frozen, sha256=214830a18e3aed43ecf7849d056b622b5fcf307265e94a145d0859364c068be9)  
**Decision register:** round3_decision_register_v1.csv

---

## Final Gate Summary

| Gate | Status | Category |
|------|--------|----------|
| TRAINING_POPULATION | DEFERRED_TO_LORA_STAGE | C — Deferred |
| FEW_SHOT_SHOT_1 | REJECT RECOMMENDED | B — Requires user approval |
| FEW_SHOT_SHOT_2 | RECOMMEND_APPROVE | B — Requires user approval |
| FEW_SHOT_SHOT_3 | RECOMMEND_APPROVE | B — Requires user approval |
| FEW_SHOT_SHOT_4 | RECOMMEND_APPROVE with note | B — Requires user approval |
| BASELINE_MODEL | UNRESOLVED | D — Blocking baseline |
| TEMPERATURE | UNCONFIRMED (0.0 pending confirmation) | B — Requires user approval |
| MAX_TOKENS | UNCONFIRMED (4096 pending confirmation) | B — Requires user approval |
| DIRECTION_NULL_POLICY | AMBIGUOUS | B — Requires user approval |
| VALIDATION_SOURCES | COMPLETE | A — Resolved |

**Overall: STEP_3_BLOCKED**

---

## A. RESOLVED

### A1. Validation Sources — COMPLETE

All 6 validation study source texts are available at `outputs/finetuning/source_recovery/texts/`. Two filename mismatches (Cammeraat, Lopez-Agudo) are documented; they do not block extraction. No action required.

### A2. Training Population Discrepancy (169 / 153 / 131) — DOCUMENTED

Full provenance in `training_population_audit_v1.4.json`.

| Count | Definition | Verdict |
|------:|-----------|---------|
| 169 | train(153) + val(16) FORECASTABLE — development-set total, NOT training-only | PROTOCOL HEADING AMBIGUITY |
| 153 | Training rows with construct_status=FORECASTABLE per v1.3 (protocol §4.5 table + §7) | AUTHORITATIVE |
| 131 | Wrong filter (canonical_construct != UNMATCHED) — initial preflight error | DISCARD |

The 153 vs 152 question is deferred to LoRA stage (see Section C).

---

## B. REQUIRES USER APPROVAL

### B1. Shot 1 — PISA — REJECT RECOMMENDED

**Study:** `1-s2.0-S2666374025001359-main.pdf` (PISA 2022, 27 rows)

**Finding 1 — direction encoding is non-canonical (CRITICAL):**

All 27 rows have `direction_canonical = "1"` in evidence_matrix_v1.3.csv. Confirmed by direct audit of v1.3 data:

```
G0013 | direction_original=1 | direction_canonical=1 | transformation=none
G0014 | direction_original=1 | direction_canonical=1 | transformation=none
... (27 identical entries)
```

The §3.2 output schema requires: `"direction": "positive | negative | mixed | null | not_reported"`

The value `"1"` is NOT in this vocabulary. It is the raw numeric encoding from the original annotation phase (codebook v1.0 used +/-/0/not_reported). This was never canonicalized to `"positive"` for this study.

**This is a gold-label encoding inconsistency in v1.3.** 44 training rows total across 4 studies have `direction_canonical` as numeric (1/-1) rather than string (positive/negative). The researcher must decide whether to:

- (Option A) Canonicalize 1→positive at rendering time when constructing the demonstration, with explicit approval recorded — gold record is NOT changed, only the formatted shot uses "positive"
- (Option B) Reject and replace

**If Option A is chosen**, the researcher must explicitly state: "I authorize rendering direction_canonical='1' as 'positive' in the formatted few-shot demonstration for study `1-s2.0-S2666374025001359-main.pdf`."

**Finding 2 — volume (SERIOUS):**
27 rows for a single shot. The 4 unique predictors (SES_COMPOSITE, GENDER, IMMIGRANT_STATUS, SELF_CONCEPT) each appear 6–9 times with the same target_variable and direction. This is highly repetitive and will consume disproportionate context window space. The protocol says "3–5 demonstrations" meaning 3–5 studies; it does not specify per-study row count, but 27 is far above typical.

**Finding 3 — missing fields (cross-cutting, applies to all shots):**
The JSON evidence rows are missing 4 of the 10 §3.2 schema fields: `ilsa_variable`, `effect_size`, `student_weights_used`, `evidence_text`. These must be filled when constructing the formatted demonstration. `evidence_text` is MANDATORY per §3.3. This is not a shot-selection issue — it requires rendering work.

**Replacement options (if REJECT):**

No pure-PISA training study with canonical direction values and ≥3 forecastable rows exists in the training pool. Available alternatives:

| Study | Program | n | Forecastable | Direction | Constructs | Concern |
|-------|---------|---|-------------|-----------|-----------|---------|
| `Teacher-job-satisfaction--TIMSS-2025.pdf` | TIMSS | 7 | 6 | positive (canonical) | GENDER(HIGH), DISCIPLINARY_CLIMATE(HIGH), TEACHER_PD(LOW) | 2nd TIMSS shot alongside shot_2 |
| `An_Evaluation_of_Tiered_ML_Framework...pdf` | TIMSS | 5 | 5 | not_reported | SES_COMPOSITE(HIGH), SELF_CONCEPT(HIGH) | HIGH constructs ✓ but direction not_reported (same as shot_4) |
| `1-s2.0-S0272775725001001-main.pdf` | PIAAC+PISA+TIMSS | 4 | 1 | positive (canonical) | G0244 adjudicated to UNMATCHED in v1.4 | Only 1 forecastable row; G0244 = UNMATCHED |
| Reduce to 3 shots | — | — | — | — | shots 2+3+4 still meet minimum (protocol: 3–5) | No PISA representation |

**Recommended action:** Present these options to researcher. No automatic replacement.

---

### B2. Shot 2 — TIMSS+PISA — RECOMMEND APPROVE

**Study:** `thinking-skills-in-teaching-practices.pdf` (TIMSS 2011 + PISA 2012, 4 rows)

All direction values are canonical (mixed). rubin_rules criterion met exactly. 4 rows is appropriate. One NOT_FORECASTABLE row (G0165 = UNMATCHED school/teacher characteristics) — provides good demonstration of the UNMATCHED case. Minor deviation: cross-program study, not pure TIMSS.

The 4 missing fields (ilsa_variable, effect_size, student_weights_used, evidence_text) must be filled when rendering the formatted demonstration from the paper text.

**Awaiting:** APPROVE / REJECT from researcher.

---

### B3. Shot 3 — PIAAC — RECOMMEND APPROVE

**Study:** `jwl-04-2021-0047.pdf` (PIAAC Cycle 1 Rounds 1–3, 4 rows)

All direction values are canonical (positive). not_applicable PV correct for PIAAC. Labor market outcome (informal learning behaviors) ✓. NOT_FORECASTABLE examples included (G0198 demographic controls, G0200 workplace interaction) ✓. WORKPLACE_AUTONOMY as SINGLE-stratum construct demonstrates low-frequency case ✓.

The 4 missing fields must be filled when rendering.

**Awaiting:** APPROVE / REJECT from researcher.

---

### B4. Shot 4 — PIRLS — RECOMMEND APPROVE with note

**Study:** `1-s2.0-S0022096522001059-main.pdf` (PIRLS 2006, 6 rows)

All direction values are canonical (not_reported). direction_zero_ambiguous ambiguity flag correctly resolved per codebook: 0 → not_reported. 6 rows with 4 distinct constructs (MOTIVATION, GENDER, PARENTAL_EDUCATION, SES_COMPOSITE) provides good diversity.

**Note for researcher:** The outcome variable (Theory of Mind at Time 3) is unusual for PIRLS (typically reading literacy). This is a longitudinal follow-up study using PIRLS 2006 as a baseline. If you prefer a more standard PIRLS study, the following training alternatives exist: `3.pdf` (PIRLS, 7 rows, SES/GENDER/ICT/SCHOOL_LOCATION, direction=mixed) and `s10639-023-11881-8.pdf` (PIRLS, 6 rows, HOME_LITERACY/MOTIVATION/SELF_CONCEPT, direction=not_reported). Both have canonical direction values.

The 4 missing fields must be filled when rendering.

**Awaiting:** APPROVE / REJECT (or APPROVE with alternative study) from researcher.

---

### B5. Inference Parameters

**Temperature — UNCONFIRMED:**

The protocol (EVALUATION_PROTOCOL_v1.md) does not specify temperature anywhere. The value 0.0 comes from the pre-v1.4 `training_configuration.json` template.

Recommendation: CONFIRM 0.0. Rationale: deterministic extraction required for reproducibility of the BASE condition. Any non-zero temperature introduces run-to-run variance that would confound the BASE vs FINETUNED comparison. Confirm or override.

**max_tokens — UNCONFIRMED:**

Protocol does not specify. Template says 4096. Longest validation study is 114 KB; a study with 20–30 evidence pairs each with 50–100 token evidence_text fields could approach or exceed 4096 output tokens. Recommendation: CONFIRM 4096 OR INCREASE to 8192 as a safety margin, especially if some validation studies are dense.

**Direction null policy — AMBIGUOUS:**

The §3.2 schema lists: `"direction": "positive | negative | mixed | null | not_reported"`

`"null"` appears as a direction value. However, the evidence_matrix_v1.3.csv contains no `direction_canonical = "null"` records — rows with original direction=0 are stored as `not_reported` per the codebook rule. The system prompt rule 4 in the draft lists: `positive | negative | mixed | not_reported` (omits `null`).

Two possible interpretations:
- (Interpretation A) `"null"` in §3.2 means the JSON `null` value (field is absent/inapplicable) — distinct from the string `"not_reported"`. In this case, `null` ≠ `not_reported`.
- (Interpretation B) `"null"` in §3.2 is a drafting artifact/redundancy with `"not_reported"` and should not be treated as a separate valid string value.

**Researcher must state which interpretation is correct.** The system prompt must be updated before freezing to remove ambiguity. In either case, the gold data uses only `not_reported` (never the string `"null"`).

---

## C. DEFERRED TO LORA STAGE

### C1. Training Population — 153 vs 152

The 153 vs 152 discrepancy does NOT affect the few-shot baseline. It only matters when constructing the LoRA training batch.

| Figure | Source | Status |
|--------|--------|--------|
| 153 | Protocol §4.5 table + §7 (v1.3) | Authoritative protocol figure |
| 152 | v1.4-adjusted (153 − G0244 adjudicated to UNMATCHED) | v1.4-consistent |

Deferred decision: for LoRA, use **152** (exclude G0244 per v1.4 adjudication) and update protocol §7 note accordingly. G0244 exclusion rationale: predictor is "Assessment cycle" (temporal year variable), not an SES composite; UNMATCHED is the correct v1.4 label.

**Mark:** DEFERRED_TO_LORA_STAGE

---

## D. BLOCKING BASELINE

### D1. Baseline Model Selection — BLOCKING

The evaluation protocol §7 explicitly says `"Base model: TBD (Claude API fine-tuning / Llama-3 8B / Mistral 7B)"`. This has NOT been resolved.

**This is a hard blocker.** The baseline prompt cannot be frozen without a model selection. The choice has architectural consequences:

| Option | Context window | LoRA compatible | Notes |
|--------|---------------|----------------|-------|
| Claude API | Up to 200K tokens | No — Claude API fine-tuning is a different mechanism | Highest capability baseline; incompatible with standard LoRA |
| Llama-3 8B | ~8K tokens | Yes — HuggingFace PEFT | Open-weight; matches LoRA planned method; smaller context |
| Mistral 7B | ~32K tokens | Yes — HuggingFace PEFT | Open-weight; larger context than Llama-3 8B; matches LoRA |

**If the thesis plan is BASE (few-shot) → FINETUNED (LoRA), the baseline model must be the SAME architecture as the LoRA model.** Using Claude API for baseline and Llama-3 for LoRA would conflate model capability differences with fine-tuning effects — this would make the comparison scientifically indefensible.

**Recommendation (not a decision):** If LoRA is the planned fine-tuning method, select Llama-3 8B or Mistral 7B for the baseline. Mistral 7B has a larger context window which is safer for long papers.

**Awaiting researcher's decision.**

---

## E. BLOCKING LORA (not blocking baseline)

### E1. Training Population 153 vs 152

Documented in Section C. Does not block few-shot baseline. Must be resolved before configuring the LoRA training batch.

### E2. Few-shot Example Format

Regardless of which shots are approved, the current JSON evidence rows are missing 4 of the 10 §3.2 schema fields (ilsa_variable, effect_size, student_weights_used, evidence_text). These fields must be sourced from the paper texts and added to the formatted demonstration pairs. This is a rendering step, not a gold-label change. It must be completed and reviewed before the prompt is frozen.

---

## Required User Decisions (in order)

| # | Decision | Options | Baseline blocker | LoRA blocker |
|---|---------|---------|-----------------|-------------|
| 1 | shot_1: REJECT or CONDITIONAL_APPROVE | Reject (choose replacement or reduce to 3 shots) / Conditional approve with direction canonicalization authorization | YES | YES |
| 2 | shot_2: APPROVE or REJECT | Approve (recommended) / Reject | YES | YES |
| 3 | shot_3: APPROVE or REJECT | Approve (recommended) / Reject | YES | YES |
| 4 | shot_4: APPROVE or REJECT (or alternative) | Approve (recommended) / Reject / Choose alternative PIRLS study | YES | YES |
| 5 | Baseline model selection | Claude API / Llama-3 8B / Mistral 7B | YES | YES |
| 6 | temperature: CONFIRM 0.0 or override | Confirm 0.0 (recommended) / Set other value | No (0.0 is stated default) | No |
| 7 | max_tokens: CONFIRM 4096 or increase | Confirm 4096 / Increase to 8192 | No (4096 is stated default) | No |
| 8 | direction null policy: interpretation A or B | JSON null = absent field / String 'null' is a valid direction value | No (not_reported is used in all gold data) | No |

Decisions 1–5 must be made before baseline_prompt_frozen.txt can be committed.  
Decisions 6–8 are confirmations that refine the prompt but do not block freezing if the stated defaults are acceptable.

---

## What Must NOT Happen

Per the global invariants in effect for this pipeline:

- DO NOT run few-shot extraction on validation studies.
- DO NOT calculate validation F1.
- DO NOT train LoRA or SFT.
- DO NOT inspect or evaluate the test set.
- DO NOT rename baseline_prompt_draft_v1.txt to baseline_prompt_frozen.txt without researcher approval.
- DO NOT modify: gold_standard_full_corpus_v1.4.csv, gold_freeze_manifest_v1.4.json, reliability form, kappa JSON, train/val/test split, annotation codebook, construct vocabulary, EVALUATION_PROTOCOL_v1.md, few_shot_examples_frozen.json, test-set files, literature weights, Stage 5 outputs.
- DO NOT silently canonicalize direction='1' to 'positive' in any gold record.
- DO NOT silently select a model.
