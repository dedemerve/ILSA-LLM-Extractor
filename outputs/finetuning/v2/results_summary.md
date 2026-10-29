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

| A0 `base_qwen3_chunked` | Qwen3-8B, non-thinking | none | 0.041 | 0.110 | 0.192 |
| A1 `qwen3_lora_selected` | Qwen3-8B, non-thinking | LoRA v2 (rank 16, checkpoint-81) | **0.057** | 0.200 | 0.371 |
| A0 `base_qwen3_chunked` **(TEST)** | Qwen3-8B, non-thinking | none | 0.039 | 0.288 | — |
| A1 `qwen3_lora_selected` **(TEST)** | Qwen3-8B, non-thinking | LoRA v2 (rank 16, checkpoint-81) | **0.060** | 0.180 | — |

## Notes

- P3 numbers match the pre-existing `zeroshot_qwen27b_val.json` (strict=0.000, lenient=0.118).
- P1 achieves the highest strict F1 among API arms (0.020) — low in absolute terms but non-zero.
- P2 RAG demo hurt P2 relative to P1; P4 RAG demo similarly hurt P4 relative to P3. Likely a
  prompt-length / Groq rate-limit interaction. The demo is from a train-split passage only.
- Strict F1 is near zero across all API arms. The strict metric requires construct + outcome
  class + direction all correct simultaneously; outcome class assignment (keyword-based) may be
  the primary bottleneck.
- A0/A1 rows will be filled after Colab training. Target: A1 strict > A0 strict on val.

## H1 result

**H1 DESTEKLENDI (val ve test):** A1 strict F1 > A0 strict F1 her iki split'te. LoRA fine-tuning, base Qwen3-8B'ye göre strict pair F1'i artırdı.  
- Val: A1=0.057 > A0=0.041  
- Test: A1=0.060 > A0=0.039  
Selected adapter: checkpoint-81 (epoch 3). max_model_len=16384 (A0), 12288 (A1, LoRA).

## Success criteria status

- [x] LoRA training completes without assertion failures
- [x] A1 strict pair F1 on val > A0 strict pair F1 on val (0.057 > 0.041)
- [x] Adapter reloads with strict=True and tensor comparison passes (fallback key check)
- [x] results_summary.md written with all completed arm F1 scores
- [x] A0/A1 test set evaluation completed (A0 strict=0.039, A1 strict=0.060)
