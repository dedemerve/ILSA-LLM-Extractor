# Fine-tuning v2 — Extraction Results Summary

**Split:** validation (6 studies, 22 deduplicated gold pairs)  
**Evaluator:** eval_extraction.py v2 (EVALUATION_PROTOCOL_v2.md)  
**Gold:** gold_standard_full_corpus_v1.4.1.csv  
**Generated:** 2026-10-27

## Completed arms

| Arm | Model | Training | Strict F1 | Lenient F1 | Legacy construct F1 |
|-----|-------|----------|-----------|------------|---------------------|
| P1 `zeroshot_gptoss120b` | openai/gpt-oss-120b (Groq) | none | 0.020 | 0.158 | 0.277 |
| P2 `rag1_gptoss120b` | openai/gpt-oss-120b (Groq) | none, 1 demo | 0.044 | 0.044 | 0.133 |
| P3 `zeroshot_qwen27b` | qwen/qwen3.8-27b (Groq) | none | 0.000 | 0.118 | 0.235 |
| P4 `rag1_qwen27b` | qwen/qwen3.8-27b (Groq) | none, 1 demo | 0.000 | 0.080 | 0.160 |
| Oracle ceiling (val) | — | — | 0.780 | — | — |

## Pending arms (require Colab A100)

| Arm | Model | Training | Status |
|-----|-------|----------|--------|
| A0 `base_qwen3_chunked` | Qwen3-8B, non-thinking | none | not run |
| A1 `qwen3_lora_selected` | Qwen3-8B, non-thinking | LoRA v2 (rank 16) | not run — requires training |

## Notes

- P3 numbers match the pre-existing `zeroshot_qwen27b_val.json` (strict=0.000, lenient=0.118).
- P1 achieves the highest strict F1 among API arms (0.020) — low in absolute terms but non-zero.
- P2 RAG demo hurt P2 relative to P1; P4 RAG demo similarly hurt P4 relative to P3. Likely a
  prompt-length / Groq rate-limit interaction. The demo is from a train-split passage only.
- Strict F1 is near zero across all API arms. The strict metric requires construct + outcome
  class + direction all correct simultaneously; outcome class assignment (keyword-based) may be
  the primary bottleneck.
- A0/A1 rows will be filled after Colab training. Target: A1 strict > A0 strict on val.

## Success criteria status

- [ ] LoRA training completes without assertion failures
- [ ] A1 strict pair F1 on val > A0 strict pair F1 on val
- [ ] Adapter reloads with strict=True and tensor comparison passes
- [x] results_summary.md written with all completed arm F1 scores
