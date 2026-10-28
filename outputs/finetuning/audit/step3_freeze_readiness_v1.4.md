# Step 3 Freeze Readiness — v1.4

**Generated:** 2026-10-06  
**Branch:** experiment/ilsa-domain-finetuning  
**Protocol source:** EVALUATION_PROTOCOL_v1.md (frozen, sha256=214830a18e3aed43ecf7849d056b622b5fcf307265e94a145d0859364c068be9)

---

## Final Gate Status

| Gate | Status | Audit file |
|------|--------|-----------|
| TRAINING_POPULATION | TRAINING_POPULATION_DECISION_PENDING | training_population_audit_v1.4.json |
| FEW_SHOT_EXAMPLES | NOT_FROZEN | few_shot_human_review_sheet.md, few_shot_freeze_audit_v1.4.json |
| BASELINE_PROMPT | DRAFTED (not frozen) | baseline_prompt_protocol_check_v1.json |
| VALIDATION_SOURCES | COMPLETE | validation_source_audit_v1.4.json |

**Overall:** STEP_3_BLOCKED

---

## 1. TRAINING_POPULATION — TRAINING_POPULATION_DECISION_PENDING

Three counts exist; their provenance is fully documented in `training_population_audit_v1.4.json`.

| Count | Definition | Verdict |
|------:|-----------|---------|
| 169 | train FORECASTABLE (153) + val FORECASTABLE (16) — development-set total | PROTOCOL HEADING AMBIGUITY — not training-only |
| 153 | Training rows with construct_status == FORECASTABLE per v1.3 (authoritative protocol table) | AUTHORITATIVE PER PROTOCOL v1.3 |
| 152 | v1.4-adjusted: 153 minus G0244 (adjudicated to UNMATCHED in v1.4 freeze) | v1.4-CONSISTENT |
| 131 | Wrong filter (canonical_construct != UNMATCHED) from initial preflight | INCORRECT — discard |

**Decision pending:** The protocol table in EVALUATION_PROTOCOL_v1.md §4.5 reports 153 (v1.3). The v1.4 freeze adjudicated G0244 to UNMATCHED, making the v1.4-consistent count 152. This 1-row difference must be resolved before LoRA training:

- **Option A:** Use 153 (protocol figure) — treat G0244 as FORECASTABLE for LoRA training per the frozen protocol.
- **Option B:** Use 152 (v1.4-adjusted) — treat G0244 as UNMATCHED/NOT_FORECASTABLE per the current frozen gold.

The protocol document has not been updated to reflect this 1-row adjudication. **Researcher must confirm which count governs LoRA training.** No modification to the protocol or gold standard is required for the few-shot baseline (Steps 3–4); this decision is only load-bearing for LoRA training (Step 5+).

---

## 2. FEW_SHOT_EXAMPLES — NOT_FROZEN

All 4 shots (PISA, TIMSS, PIAAC, PIRLS) have been audited:
- All are from the training split — no data contamination.
- Second-pass reliability precondition: CLEARED (κ = 0.871 ≥ 0.61, frozen 2026-10-06).
- Human review sheet: `outputs/finetuning/audit/few_shot_human_review_sheet.md`

**Current status:** CANDIDATE (all 4 shots). No APPROVE or REJECT decisions recorded.

**Cross-cutting issue documented in review sheet:** The JSON evidence rows use different field names than the §3.2 output schema (e.g., `target_variable` vs `outcome`, `direction_canonical` vs `direction`) and are missing 4 fields (`ilsa_variable`, `effect_size`, `student_weights_used`, `evidence_text`). These rows are gold reference labels, not formatted demonstration pairs. The prompt must specify how to render them as (source_text → extraction) examples.

**Required action:** Researcher must supply APPROVE / REJECT decision for each shot in `few_shot_human_review_sheet.md`. Claude will not auto-freeze.

---

## 3. BASELINE_PROMPT — DRAFTED (not frozen)

Draft file: `outputs/finetuning/baseline/baseline_prompt_draft_v1.txt`  
Protocol check: `outputs/finetuning/audit/baseline_prompt_protocol_check_v1.json`

Protocol compliance summary (20 checks):

| Result | Count | Items |
|--------|------:|-------|
| PASS | 13 | §3.1 input template (4 slots + task instruction), §3.2 schema (all 10 fields), output format |
| PASS_WITH_NOTE | 4 | direction null vs not_reported inconsistency; temperature 0.0; max_tokens 4096; ilsa_variable nullable |
| FAIL_BLOCKED | 1 | PE-16: Few-shot demonstrations — placeholder only, blocked by NOT_FROZEN few-shot status |
| FAIL_TBD | 1 | PE-17: Model not selected — protocol §7 marks as TBD |

**Hard blockers preventing prompt freeze:**

1. **Few-shot demonstrations not approved** — shots cannot be inserted until researcher approves them.
2. **Model not selected** — must choose between Claude API / Llama-3 8B / Mistral 7B (or other) before freezing.

**Researcher confirmations needed before freeze:**

- Is `null` a valid direction value distinct from `not_reported`? (PE-11)
- Confirm temperature = 0.0. (PE-18)
- Confirm max_tokens = 4096 sufficient. (PE-19)

**Required action:** Do NOT rename baseline_prompt_draft_v1.txt to baseline_prompt_frozen.txt. Researcher must resolve the two hard blockers, confirm the three items, and provide explicit freeze approval.

---

## 4. VALIDATION_SOURCES — COMPLETE

All 6 validation study source texts are available at `outputs/finetuning/source_recovery/texts/`.

| Study | Source text | FORECASTABLE rows (v1.3) |
|-------|------------|------------------------:|
| 1-s2.0-S0191886926001704-main.pdf | AVAILABLE (45 KB) | 1 |
| 5.pdf (Song & Cutumisu 2024) | AVAILABLE (114 KB) | 3 |
| InfocomJournal_2024_SpecISS_CogInf_CogAspVR_2.pdf | AVAILABLE (97 KB) | 0 |
| Patterns-of-teacher-collaboration-2022.pdf | AVAILABLE (49 KB) | 5 |
| Review of Income and Wealth - 2024 - Cammeraat.pdf | AVAILABLE (106 KB) | 3 |
| lopez-agudo-marcenaro-gutierrez-2025.pdf | AVAILABLE (60 KB) | 4 |

Total validation FORECASTABLE rows (v1.3): 16  
Note: InfocomJournal study contributes 0 FORECASTABLE rows. Include in evaluation (contributes to FN counting).  
Note: 2 filename mismatches (Cammeraat, Lopez-Agudo) between split manifest and actual files — documented, does not block extraction.

---

## Required Actions Before Step 3 Can Proceed

Actions are listed in dependency order. Complete them in sequence.

| # | Action | Owner | Blocker for |
|---|--------|-------|-------------|
| 1 | Review 4 shots in `few_shot_human_review_sheet.md`, supply APPROVE/REJECT for each | Researcher | PE-16, prompt freeze, Step 3 |
| 2 | Select baseline model (Claude API model / Llama-3 8B / Mistral 7B / other) | Researcher | PE-17, prompt freeze, Step 3 |
| 3 | Confirm temperature=0.0 and max_tokens=4096 | Researcher | PE-18, PE-19, prompt freeze |
| 4 | Confirm null vs not_reported direction semantics (PE-11) | Researcher | PE-11, prompt freeze |
| 5 | Insert approved shots into prompt draft, update model/temperature | Claude (after 1–4) | prompt freeze |
| 6 | Researcher reviews and approves final prompt — rename to frozen | Researcher | Step 3 execution |
| 7 | Decide training population: 153 (v1.3 protocol) or 152 (v1.4-adjusted) | Researcher | LoRA training (Step 5+) only |

**Step 3 (few-shot baseline extraction) is BLOCKED until actions 1–6 are complete.**  
Action 7 is only required before LoRA training; it does not block Step 3.

---

## What Must NOT Happen

Per the global safety rules in effect:

- DO NOT run few-shot extraction on validation studies.
- DO NOT calculate validation F1.
- DO NOT train LoRA.
- DO NOT run SFT.
- DO NOT inspect test set labels.
- DO NOT evaluate the test set.
- DO NOT rename baseline_prompt_draft_v1.txt to baseline_prompt_frozen.txt without researcher approval.
- DO NOT modify: gold_standard_full_corpus_v1.4.csv, gold_freeze_manifest_v1.4.json, reliability form, kappa JSON, train/val/test split, annotation codebook, construct vocabulary, EVALUATION_PROTOCOL_v1.md, few_shot_examples_frozen.json, test-set files, literature weights, Stage 5 outputs.
