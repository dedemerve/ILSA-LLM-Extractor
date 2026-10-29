# Alternative open-weight small LLMs (3-14B) for long-context (16k-32k) structured extraction fine-tuning

Scope: replace Mistral-7B-Instruct-v0.3 for extracting JSON arrays of evidence pairs from ~18k-token excerpts of PISA/TIMSS papers; 169 training examples; QLoRA/LoRA on a single A100-40GB, or Together AI fine-tuning; deploy on HF or Together. Research date: 2026-10-09.

## Q1. Candidate models: native context, architecture, license

### Takeaway
The families with strong native 32k+ context and permissive licenses are Qwen3 (4B/8B/14B, 32k native, Apache-2.0), Qwen3.5 (4B/9B, 262k native, Apache-2.0, hybrid Gated-DeltaNet), Llama-3.1-8B (128k) and Gemma 3 (4B/12B, 128k via RoPE rescaling from 32k pretraining). Gemma-2-9B (8k) and Ministral-8B (research-only license) should be excluded; Mistral-7B v0.x has an effective context of only ~16k, which plausibly explains the poor results.

### Comparison table

| Model | Params | Native ctx | Extension | Attention / KV notes | License | RULER 16k / 32k | Together FT (max LoRA SFT ctx) |
|---|---|---|---|---|---|---|---|
| Mistral-7B-Instruct v0.2 (v0.3 not measured) | 7B | 32k | none | GQA | Apache-2.0 | 87.2 / 75.4 (effective 16k) | not listed |
| Llama-3.1-8B-Instruct | 8B | 128k | native | GQA | Llama 3.1 Community | 91.6 / 87.4 (effective 32k) | yes, 131,072 |
| Qwen2.5-7B-Instruct | 7B | 32k | YaRN to 128k | GQA | Apache-2.0 | 93.7 / 89.4 | not listed |
| Qwen2.5-14B-Instruct | 14B | 32k | YaRN | GQA | Apache-2.0 | 95.9 / 93.4 | not listed |
| Qwen3-4B (non-thinking) | 4B | 32k | YaRN to 128k | GQA | Apache-2.0 | 91.0 / 87.8 | not listed |
| Qwen3-8B (non-thinking) | 8B | 32k | YaRN to 131k | 36 layers, 32 Q / 8 KV heads | Apache-2.0 | 91.8 / 91.2 | not listed |
| Qwen3-14B (non-thinking) | 14B | 32k | YaRN to 128k | GQA | Apache-2.0 | 96.4 / 96.1 | not listed |
| Qwen3.5-4B / 9B | 4B / 9B | 262k | YaRN to ~1M | hybrid: per block 3 Gated DeltaNet + 1 gated attention (9B: 32 layers, attn 16 Q / 4 KV heads); multimodal (vision encoder) | Apache-2.0 | not reported (9B: LongBench v2 55.2, AA-LCR 63.0) | yes, 4B: 131,072; 9B: 65,536 |
| Gemma 3 4B IT | 4B | 128k (RoPE-rescaled from 32k) | - | 5 local (1024-token sliding window) : 1 global | Gemma terms | IT 32k: 61.4 | not listed |
| Gemma 3 12B IT | 12B | 128k (RoPE-rescaled from 32k) | - | 5:1 local:global, 1024 window | Gemma terms | IT 32k: 80.3 (PT 90.6) | not listed |
| Gemma-2-9B | 9B | 8k | - | alternating sliding window | Gemma terms | not measured | not listed (disqualified: context < 18k) |
| Phi-3-mini-128k (proxy for Phi-3.5-mini) | 3.8B | 128k | - | - | MIT | 90.7 / 87.5 (effective 32k) | not listed |
| Phi-4-mini-instruct | 3.8B | 128k | - | - | MIT (NVIDIA listing adds NVIDIA terms) | not found | not listed |
| GLM-4-9B (reference point) | 9B | 1M (variant) | - | - | GLM license | 92.1 / 89.9 | not listed |
| Ministral-8B-2410 | 8B | 128k (32k on vLLM at launch) | - | interleaved sliding window | Mistral Research License (non-commercial) | not found | not listed |
| Granite 4.0-H (Micro/Tiny/Small) | 3B-32B | 128k-131k | - | hybrid Mamba-2 + transformer (~9:1) | Apache-2.0 | not found | not listed |

### Cited Findings
- Qwen3-8B: native 32,768 tokens; YaRN (factor 4) validated to 131,072; 36 layers, 32 Q / 8 KV heads; Apache-2.0; card warns static YaRN "potentially impacting performance on shorter texts" and says not to enable YaRN if contexts stay within 32k; thinking mode is the default, `enable_thinking=False` is a hard switch — [Qwen3-8B model card](https://huggingface.co/Qwen/Qwen3-8B)
- Qwen3.5-9B: hybrid layout of 8 blocks each with 3 Gated DeltaNet layers + 1 gated attention layer, 32 layers; gated attention 16 Q / 4 KV heads; native 262,144 context, YaRN to ~1,010,000; Apache-2.0; multimodal (image/video) with a text-only vLLM serving option; thinking mode on by default (disable via `enable_thinking: False`; no `/think` soft switches); LongBench v2 55.2, AA-LCR 63.0; released ~Feb 2026 — [Qwen3.5-9B model card](https://huggingface.co/Qwen/Qwen3.5-9B)
- Gemma 3: 5 local : 1 global layers, local sliding window 1,024 tokens; pretrained on 32k sequences and extended to 128k by RoPE rescaling (factor 8); "degrade quickly" beyond 128k — [Gemma 3 Technical Report](https://arxiv.org/html/2503.19786)
- Ministral 8B: interleaved sliding-window attention, 128k context ("currently 32k on vLLM" at launch), Mistral Research License, commercial license on request; one secondary source says Mistral deprecated the 2410 models on its platform in Dec 2025 (unverified) — [OpenRouter Ministral 8B](https://openrouter.ai/models/mistralai/ministral-8b); [iThome](https://www.ithome.com.tw/news/165534)
- Granite 4.0: Mamba-2 + transformer hybrid, ~9:1 ratio; 128k-131k context; Apache-2.0; IBM claims >70% RAM reduction for long inputs — [IBM announcement](https://www.ibm.com/new/announcements/ibm-granite-4-0-hyper-efficient-high-performance-hybrid-models); [The Decoder](https://the-decoder.com/ibms-granite-4-0-family-of-hybrid-models-uses-much-less-memory-during-inference/)
- Phi-4-mini-instruct: 128k context; MIT per most listings, but NVIDIA's listing references the NVIDIA Community Model License — [NVIDIA NIM model card](https://build.nvidia.com/microsoft/phi-4-mini-instruct/modelcard)

### Inferences
- Mistral-7B v0.2 has RULER effective length 16k and drops to 75.4 at 32k (see Q2). If v0.3 behaves similarly, the ~18k-token inputs plus prompt and output sit right at or past its reliable range. This is a likely structural cause of the poor results, separate from the small training set.
- For an ~18-20k total sequence, Qwen3 models run inside their native 32k window, so no YaRN is needed. That avoids the warned short-text degradation and keeps training and inference config identical.
- Qwen3.5 is architecturally new (linear-attention hybrid, vision tower). That means more tooling risk (vLLM version requirements, QLoRA discouraged; see Q3) but in exchange it has very long native context and Together fine-tuning support.

### Gaps
- Exact layer/KV-head counts for Llama-3.1-8B, Qwen2.5-7B, Qwen3-4B/14B and Gemma 3 were not re-verified from config.json in this session (commonly cited values: Llama-3.1-8B 32 layers / 8 KV heads; Qwen2.5-7B 28 layers / 4 KV heads).
- OLMo (2/3), Llama 3.2/3.3 small models and Mistral Small 3.x were not researched in depth. Llama 3.2 1B/3B are below the target size; Mistral Small 3.x is ~24B, above the range.
- Phi-3.5-mini itself has no RULER row; Phi-3-mini-128k is used as a proxy.

## Q2. Long-context retrieval/IE quality at 16k-32k

### Takeaway
On RULER at 16k/32k, Qwen3-14B (96.4/96.1) leads the small models, followed by Qwen2.5-14B (95.9/93.4), Qwen3-8B (91.8/91.2), Qwen2.5-7B (93.7/89.4), Llama-3.1-8B (91.6/87.4) and Qwen3-4B (91.0/87.8). Gemma 3 12B IT is notably weaker at 32k (80.3), and Mistral-7B v0.2 is weakest (75.4). Thinking mode hurts Qwen3 retrieval.

### Cited Findings
- RULER (NVIDIA) main table: Llama3.1-8B 4k-128k = 95.5/93.8/91.6/87.4/84.7/77.0, effective length 32k; Phi3-mini 3.8B 92.2/91.5/90.7/87.5/80.6/66.7, effective 32k; Mistral-7B v0.2 93.6/91.2/87.2/75.4/49.0/13.8, effective 16k; Mistral-Nemo 16k effective (32k: 69.0); GLM4-9B 32k: 89.9; Qwen2.5-7B-Instruct-1M 32k: 91.1 — [RULER GitHub](https://github.com/NVIDIA/RULER)
- Qwen3 Technical Report Table 23 (RULER, non-thinking): Qwen3-4B 16k 91.0 / 32k 87.8; Qwen3-8B 91.8 / 91.2; Qwen3-14B 96.4 / 96.1; Qwen2.5-7B-Instruct 93.7 / 89.4; Qwen2.5-14B-Instruct 95.9 / 93.4. Thinking mode is lower (Qwen3-8B thinking 32k: 80.8), and the authors hypothesize thinking "may interfere with the retrieval process". RULER was run with YaRN factor 4 — [Qwen3 Technical Report](https://arxiv.org/html/2505.09388)
- Gemma 3 Table 15: RULER 32k IT 4B 61.4, IT 12B 80.3, IT 27B 91.1 (PT 12B 90.6); MRCR 32k IT 12B 53.7 — [Gemma 3 Technical Report](https://arxiv.org/html/2503.19786)
- HELMET authors report that NIAH is saturated and that RULER/∞Bench rankings can be inconsistent (e.g., 8B beating 70B), and argue for HELMET's application-like tasks; its main comparison is at 128k and covers Llama-3.1-8B but not Qwen3/Gemma 3 — [HELMET paper](https://arxiv.org/html/2410.02694v3); [HF blog](https://huggingface.co/blog/helmet)

### Inferences
- Use Qwen3 in non-thinking mode (or train on direct-answer JSON targets) for this extraction task. Thinking costs RULER points and adds output tokens to the training sequence.
- Gemma 3 12B IT's 32k RULER drop (80.3 vs 90.6 for PT) plus its 1,024-token local window make it a weaker fit than Qwen3 at the same size, although it is not disqualified the way Gemma-2-9B is.
- RULER is synthetic retrieval. Evidence-pair extraction is closer to multi-hop QA / aggregation, so these scores are an upper-bound proxy. Validate on the project's own held-out set.

### Gaps
- No HELMET or LongBench v2 numbers at 16k-32k were found for Qwen3-8B/14B, Gemma 3 12B or Llama-3.1-8B in one comparable table.
- No standardized structured-output/JSON-schema benchmark (e.g., JSONSchemaBench) results were gathered for these specific models. JSON validity after fine-tuning is usually near 100% and can be enforced with constrained decoding (vLLM/SGLang guided decoding).
- RULER for Qwen3.5-4B/9B, Phi-4-mini, Granite 4.0 and Ministral 8B was not found.

## Q3. QLoRA/LoRA at 16k-32k on a single A100-40GB

### Takeaway
With Unsloth, Llama-3.1-8B QLoRA reaches ~154k tokens on 40GB, so 16k-32k is comfortable. Plain HF + FlashAttention-2 tops out at ~12k on 40GB, which is below the ~18-20k needed. Qwen3-8B (similar size, 8 KV heads) should also fit at 32k under Unsloth. Qwen3-14B is plausible but unpublished. For Qwen3.5, Unsloth explicitly advises against QLoRA; use bf16 LoRA instead (9B ≈ 22GB at base context).

### Cited Findings
- Unsloth benchmark, Llama-3.1-8B 4-bit QLoRA, max context: 24GB 78,475; 40GB 153,977; 48GB 191,728; 80GB 342,733. HF + FA2: 24GB 5,789; 40GB 12,264; 48GB 15,502; 80GB 28,454. Conditions: rank 32, all linear layers, batch size 1, padded to max length; Unsloth gradient checkpointing + Apple CCE — [Unsloth benchmarks](https://unsloth.ai/docs/basics/unsloth-benchmarks)
- Llama-3.3-70B QLoRA: 48GB 12,106; 80GB 89,389 (out of range, for reference) — [Unsloth benchmarks](https://unsloth.ai/docs/basics/unsloth-benchmarks)
- Unsloth Qwen3 guide: claims 70% less VRAM and 8x longer context; Qwen3-14B fits a 16GB T4 at short context; no per-context table — [Unsloth Qwen3 guide](https://unsloth.ai/docs/models/qwen)
- Unsloth Qwen3.5 guide: bf16 LoRA VRAM 4B 10GB, 9B 22GB, 27B 56GB; "It is not recommended to do QLoRA (4-bit) training on the Qwen3.5 models, no matter MoE or dense" (higher quantization error); use `use_gradient_checkpointing="unsloth"` for long context; vLLM 0.16.0 did not support Qwen3.5 (needs ≥0.17 or nightly) — [Unsloth Qwen3.5 fine-tune](https://unsloth.ai/docs/models/qwen3.5/fine-tune)

### Inferences
- The A100-40GB is only viable for 18-20k sequences with Unsloth (or an equivalent stack: FA2 + gradient checkpointing + chunked/fused cross-entropy). The plain HF/PEFT stack OOMs above ~12k on Llama-8B. This may also have contributed to the earlier Mistral run, if truncation was involved.
- Qwen3-8B vs Llama-3.1-8B: same KV-head count (8), 36 vs 32 layers, larger vocab (~151k vs ~128k), so somewhat higher memory per token. Still far below the 154k headroom, so 32k should fit.
- Qwen3-14B on 40GB at ~20k: 4-bit weights are ~9-10GB, and with Unsloth checkpointing this is likely feasible but tight at 32k. Treat it as needing an empirical test.
- Qwen3.5-9B bf16 LoRA (22GB base) plus ~20k-token activations on 40GB is uncertain. The linear-attention layers should help memory, but there is no published number.
- With 169 examples × ~19k tokens ≈ 3.2M tokens per epoch, a 3-epoch run is small: a few GPU-hours on an A100.

### Gaps
- No Unsloth max-context-per-VRAM table for Qwen3, Qwen3.5 or Gemma 3 was found. Only the Llama-3.1-8B/3.3-70B table exists.
- No measured A100-40GB results for Qwen3-14B QLoRA at 32k were found.

## Q4. Deployment and Together AI fine-tuning

### Takeaway
Together AI has discontinued serverless (multi-)LoRA inference, so adapters must now be served on dedicated endpoints. Its fine-tuning service supports Llama-3.1-8B (131k LoRA SFT context) and Qwen3.5-4B/9B at $0.34 per 1M training tokens ($4 minimum), which would cost the minimum ~$4 for this dataset. Qwen3, Qwen2.5 and Gemma 3 are not on Together's fine-tune list. On Hugging Face, Inference Endpoints with vLLM (`enable_lora`) can serve adapters, but Inference Providers do not document text-generation LoRA serving, and TGI is in maintenance mode.

### Cited Findings
- Together fine-tuning models (LoRA SFT max context): Meta-Llama-3.1-8B-Instruct-Reference 131,072; Qwen3.5-4B 131,072; Qwen3.5-9B 65,536; Qwen3.5-27B 131,072; gpt-oss-20b 131,072; Gemma 4 26B-A4B / 31B 49,152; Llama-3.3-70B 24,576. Qwen2.5-7B, Qwen3 4B/8B/14B, Gemma 3, Phi, Ministral, Granite and OLMo are not listed — [Together fine-tuning models](https://docs.together.ai/docs/fine-tuning-models)
- Together pricing per 1M training tokens (dataset tokens × epochs, plus validation): small models (Qwen3.5 0.8B-9B, Llama 3.1 8B) SFT $0.34, DPO $0.84, $4.00 minimum per job; gpt-oss-20B $0.40; ~27-35B tier $1.05. A second, unlabeled higher-rate table also appears on the page (ambiguous whether it is full fine-tuning) — [Together pricing](https://www.together.ai/pricing)
- "Serverless LoRA inference (including Serverless Multi-LoRA) has been discontinued"; fine-tuned adapters are "served on dedicated endpoints"; multiple adapters can share one LoRA-enabled dedicated endpoint — [Together LoRA inference docs](https://docs.together.ai/docs/lora-inference)
- Earlier (2025) Together docs and support pages listed serverless LoRA for Llama-3.1-8B/70B, Qwen2.5-14B/72B and Llama-4-Maverick only. That information is now superseded — [Together 503 support article](https://support.together.ai/articles/7010438063-why-is-my-fine-tuned-model-returning-a-503-error)
- Together adapter upload: adapters trained elsewhere (HF Hub or S3, with `adapter_config.json` + `adapter_model.safetensors`) can be imported, but the base model must be one supported for dedicated inference; serving is via dedicated endpoints — [Together adapter upload](https://docs.together.ai/docs/adapter-upload)
- HF Inference Endpoints vLLM engine accepts any vLLM engine argument as a container argument, with `enable_lora=true` shown as the example; supported models follow vLLM's list — [HF Inference Endpoints vLLM](https://huggingface.co/docs/inference-endpoints/en/engines/vllm)
- TGI multi-LoRA via the `LORA_ADAPTERS` env var (since ~2.0.6, PEFT adapters) — [TGI LoRA docs](https://huggingface.co/docs/text-generation-inference/main/en/conceptual/lora). Secondary sources report that TGI went into maintenance mode around Dec 2025, with HF pointing new endpoints to vLLM/SGLang; new architectures will not be added — [Spheron migration guide](https://www.spheron.network/blog/migrate-tgi-to-vllm-sglang-2026/); [PremAI comparison](https://blog.premai.io/llm-inference-servers-compared-vllm-vs-tgi-vs-sglang-vs-triton-2026)
- HF Inference Providers' documented LoRA support is for text-to-image (fal.ai, Replicate); no text-generation LoRA support was found — [huggingface_hub 0.31.0 release notes](https://newreleases.io/project/pypi/huggingface-hub/release/0.31.0)

### Inferences
- Cost estimate on Together: 169 × ~19k tokens ≈ 3.2M tokens/epoch; 3 epochs ≈ 9.6M tokens × $0.34/M ≈ $3.3, so the $4 minimum applies. Together fine-tuning is effectively free compared with A100 rental, but it locks the choice to Llama-3.1-8B or Qwen3.5-4B/9B among small models.
- The deployment cost is now the dedicated endpoint, billed per hour, not per token. For a batch extraction research workload, the cheapest pattern is: download or merge the adapter and run batch inference locally on the A100 with vLLM, or spin up an HF Inference Endpoint with vLLM + `enable_lora` (or merged weights) only during runs, with scale-to-zero.
- Because hosted serverless LoRA is gone at Together and absent on HF Inference Providers for text, base-model choice is no longer constrained by a serverless LoRA whitelist. The constraint is now vLLM support, which Qwen3, Llama-3.1, Gemma 3 and (vLLM ≥0.17) Qwen3.5 all have.

### Gaps
- Together dedicated endpoint hourly prices for 8B-class models were not retrieved.
- HF Inference Endpoints hourly prices (A10G/L4/A100) and scale-to-zero behaviour were not retrieved in this session.
- Whether Together's fine-tuning of Qwen3.5 handles the vision tower or allows text-only training, and whether the 65,536 limit for 9B applies to LoRA only, is unconfirmed.

## Q5. Recommendation (top 2-3)

### Takeaway
1) Qwen3-8B (non-thinking), self-trained with Unsloth QLoRA/LoRA on the A100 at max_seq_length ~24-32k, no YaRN, served via vLLM. 2) Llama-3.1-8B-Instruct as the low-risk alternative that can be fine-tuned on Together for ~$4 at up to 131k context, or self-trained with Unsloth's published 40GB headroom. 3) Qwen3-14B (non-thinking) as a quality upgrade if a 14B QLoRA run fits on 40GB. Qwen3.5-9B is a watch-list option, with Together support but no QLoRA and new-architecture risk.

### Cited Findings
- Qwen3-8B non-thinking RULER 32k 91.2 vs Llama-3.1-8B 87.4 vs Mistral-7B v0.2 75.4; Qwen3-14B 96.1 — [Qwen3 Technical Report](https://arxiv.org/html/2505.09388); [RULER GitHub](https://github.com/NVIDIA/RULER)
- Llama-3.1-8B QLoRA fits 153,977 tokens on 40GB with Unsloth — [Unsloth benchmarks](https://unsloth.ai/docs/basics/unsloth-benchmarks)
- Llama-3.1-8B is on Together's fine-tune list at 131,072 LoRA context, $0.34/M tokens, $4 minimum — [Together fine-tuning models](https://docs.together.ai/docs/fine-tuning-models); [Together pricing](https://www.together.ai/pricing)
- Qwen3.5: QLoRA discouraged; bf16 LoRA 9B ≈ 22GB; vLLM ≥0.17 needed — [Unsloth Qwen3.5 fine-tune](https://unsloth.ai/docs/models/qwen3.5/fine-tune)

### Inferences
- Rationale for Qwen3-8B first: best 16k-32k retrieval at 8B, ~18k inputs fit inside its native 32k (no RoPE scaling), Apache-2.0, mature Unsloth/vLLM support. Train on direct JSON outputs with `enable_thinking=False` in the chat template, and apply the same template at inference.
- Rationale for Llama-3.1-8B second: the only small model supported both by Unsloth's published 40GB long-context table and by Together's fine-tuning, so it is the fastest no-GPU path and a clean ablation baseline. It is slightly weaker on RULER at 32k, and its license is a community license (attribution, AUP), not Apache.
- Rationale for Qwen3-14B third: the highest RULER at 32k in range. Worth it if the 8B results plateau, but it needs a feasibility test on 40GB, and with only 169 examples the larger base's prior may matter more than adapter capacity.
- Deprioritize: Gemma 3 12B (weaker 32k RULER for IT, sliding-window heavy), Gemma-2-9B (8k), Ministral-8B (non-commercial license), Phi-3.5/Phi-4-mini (3.8B, plausible but weaker extraction capacity and no Together path), Granite 4.0-H (no long-context benchmark numbers found; hybrid Mamba tooling for QLoRA less mature).
- Methodological note: with 169 examples, run a zero-shot/few-shot baseline for each candidate at 18k first, to separate base-model long-context ability from fine-tuning gains.

### Gaps
- No head-to-head evidence on scientific-paper evidence extraction for these models; the recommendation rests on RULER proxies plus tooling and deployment constraints.
- Qwen3.5-9B may outperform Qwen3-8B (newer, 262k native, LongBench v2 55.2), but no directly comparable 16k-32k numbers were found.
