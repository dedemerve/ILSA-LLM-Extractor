# Phase 3C Status

**Date:** 2026-10-06  
**Branch:** experiment/ilsa-domain-finetuning  
**Status:** WAITING — gold freeze pending reliability (2026-10-12)

---

## Current state

| Item | Status | Commit |
|------|--------|--------|
| Gold v4 (42 studies, 246 rows) | Frozen | 54858d9 |
| evidence_matrix_v1.3 | Frozen | 92c8c5c |
| Codebook v1.2 + v1.3 | Frozen | 491c6b7 / 92c8c5c |
| Train/val/test split (30/6/6, seed=2026) | Frozen | 8a2dde7 |
| Annotation I/O schema | Frozen | 287d2f2 |
| BASE vs FINETUNED protocol | Frozen | 287d2f2 |
| Few-shot examples (4 studies, train-only) | Candidate | 287d2f2 |
| Test set | Untouched | — |
| Second-pass reliability | **Pending 2026-10-12** | — |
| Gold freeze | After reliability | — |
| Few-shot baseline | After gold freeze | — |
| LoRA | After baseline | — |

---

## Next milestone: 2026-10-12

1. Re-annotate 13-study subset (second_pass_reliability_form.csv) without first-pass labels
2. Run: `python3 outputs/finetuning/audit/second_pass_kappa_calculator.py`
3. If macro-kappa ≥ 0.61 → gold v4 freeze confirmed
4. If disagreements found → disagreement analysis → codebook clarification → adjudication → then freeze
5. Freeze few_shot_examples_frozen.json
6. Run few-shot baseline on frozen test set (6 studies, 35 rows)
7. Save baseline_results_frozen.json
8. Then design LoRA

---

## Do NOT do before reliability

- Run the few-shot baseline
- Modify evidence_matrix_v1.3
- Change construct mappings
- Alter train/val/test split
- Run LoRA or SFT
- Inspect test set for prompt engineering
- Run forecasting or recompute W_j

---

## Key methodological decisions (frozen)

- 169 FORECASTABLE rows ≠ 169 independent training examples (30 paper-level)
- Per-stratum construct F1 (HIGH/MID/LOW/SINGLE) required — aggregate F1 insufficient
- Do NOT oversample LOW constructs (sparsity reflects real literature distribution)
- SCHOOL_BELONGING in test but not train — known split artifact, report separately
- Downstream: BASE→W→forecast vs FINETUNED→W→forecast comparison is the primary thesis claim

