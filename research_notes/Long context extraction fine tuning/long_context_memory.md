# Long-context (16k-32k) LoRA/QLoRA fine-tuning of 7-9B models on a single A100-SXM4-40GB

Scope: which memory-saving techniques make max_seq_length=16384/32768 feasible for Mistral-7B-Instruct-v0.3 on 40 GB; measured numbers; model caveats; fallbacks; stack and config; and the truncation failure mode (missing closing "]" and EOS).

Labeling convention used below: everything in "Cited Findings" has a source. Back-of-envelope arithmetic is only in "Inferences" and is marked ESTIMATE.

## Q1. What does each technique save, and how do they combine? (QLoRA, gradient checkpointing/offload, FlashAttention-2/SDPA, packing/padding-free, fused/chunked cross-entropy)

### Takeaway
At 16k-32k on a 7B model, the frozen weights are no longer the main memory cost. What dominates is (1) per-layer activations, which scale linearly with sequence length once FlashAttention-2 is used, and (2) the logits tensor (seq x vocab, often upcast to fp32). Gradient checkpointing, ideally with CPU offload, handles (1). A fused or chunked cross-entropy (Liger FLCE, Apple CCE, TRL chunked loss, Unsloth) handles (2). 4-bit QLoRA then frees about 10 GB of weight memory. Packing and padding-free mainly cut waste and raise throughput; they do not lower the peak for one 32k example.

### Cited Findings
- **FlashAttention-2 / padding-free:** HF's packing blog (DataCollatorWithFlattening) uses `flash_attn_varlen_func` with `cu_seqlens`, so concatenated examples do not attend to each other. Earlier packing in HF did cause cross-example attention. — [HF blog: Packing with FA2](https://huggingface.co/blog/packing-with-FA2)
- **Measured packing gains:** up to 2x throughput on FLAN and 1.4x on OrcaMath for llama2-7B, **mistral-7B** and granite-8B-code. Peak memory fell 20% on FLAN and 6% on OrcaMath, with identical validation loss. Setup: 8x A100-80GB. Requires `attn_implementation="flash_attention_2"` and a model that accepts `position_ids`; Mistral is listed as supported. — [HF blog: Packing with FA2](https://huggingface.co/blog/packing-with-FA2)
- **TRL padding-free:** TRL says `padding_free=True` flattens the batch into one sequence with no padding. It is "highly recommended" with FlashAttention 2/3, because other backends can contaminate the batch. Example: `SFTConfig(..., padding_free=True, model_init_kwargs={"attn_implementation": "kernels-community/flash-attn2"})`. — [TRL docs v1.9.2: Reducing memory](https://huggingface.co/docs/trl/v1.9.2/reducing_memory_usage)
- **TRL packing strategies:** `packing_strategy` can be `"bfd"` (the default; tokens past max_length are **discarded**), `"bfd_split"` (long sequences are split into chunks of at most max_length, so nothing is dropped) or `"wrapped"` (concatenate then split; this breaks sequence continuity and the docs say it hurts performance). With `"bfd"`, padding-free is switched on automatically. — [TRL SFT Trainer docs (v1.15.0)](https://huggingface.co/docs/trl/sft_trainer); [TRL reducing memory v1.9.2](https://huggingface.co/docs/trl/v1.9.2/reducing_memory_usage)
- **TRL chunked cross-entropy:** in TRL v1.9.2, `loss_type="chunked_nll"` (the default there) drops positions where labels == -100 before the lm_head matmul. It then computes cross-entropy in chunks under gradient checkpointing, so peak memory scales with chunk_size x vocab rather than seq_len x vocab. Reported savings are about 30% peak VRAM on one GPU and up to about 50% under FSDP2 (measured on Qwen3-1.7B, vocab about 151k). The v1.9.2 docs say it is "Not compatible with `use_liger_kernel=True`, PEFT, or VLM". — [TRL docs v1.9.2](https://huggingface.co/docs/trl/v1.9.2/reducing_memory_usage). **Version conflict:** the v1.15.0 SFTConfig lists `loss_type` default `"nll"` and describes `"chunked_nll"` as a "deprecated alias of `nll`, which computes the loss the same way". This suggests the chunked path became the standard nll path, but the PEFT compatibility status in v1.15 is not stated. — [TRL SFT docs v1.15.0](https://huggingface.co/docs/trl/sft_trainer)
- **TRL activation offloading:** `SFTConfig(activation_offloading=True)` moves activations to CPU RAM through PyTorch `saved_tensors_hooks`, overlapping the transfers on CUDA streams. Gradient checkpointing is **on by default** in TRL SFTConfig (`gradient_checkpointing=True`). — [TRL docs v1.9.2](https://huggingface.co/docs/trl/v1.9.2/reducing_memory_usage); [TRL SFT docs v1.15.0](https://huggingface.co/docs/trl/sft_trainer)
- **Unsloth offloaded gradient checkpointing (Apr 2024):** activations are offloaded asynchronously to system RAM. Unsloth reports a further 30% memory reduction for about 1.9% extra time, and context of 228K vs 48K (HF+FA2) on H100-80GB and 56K vs 14K on RTX 4090-24GB. For **Mistral 7B on an L4 (24 GB)**: native HF about 5K, HF+FA2 about 14K, Unsloth about 56K. — [Unsloth blog: long-context](https://unsloth.ai/blog/long-context). A later post says the overhead is now at most 0.1% thanks to CUDA streams, and credits offloading with about 10x longer context. — [Unsloth 500K context post](https://unsloth.ai/docs/blog/500k-context-length-fine-tuning). (Vendor numbers, not independently verified.)
- **Liger Kernel:** Triton kernels for RoPE, RMSNorm, SwiGLU, CrossEntropy and FusedLinearCrossEntropy (FLCE). FLCE fuses lm_head and cross-entropy and computes them in chunks. Claimed: +20% multi-GPU throughput, -60% memory. On LLaMA-3-8B (bs 8, bf16, gradient checkpointing, FSDP1, 8x A100), HF OOMs at about 4K context while HF+Liger reaches about 16K. Mistral is supported through `apply_liger_kernel_to_mistral()`; HF/TRL enable it with `use_liger_kernel=True`. The README does not discuss PEFT or bitsandbytes compatibility. — [Liger-Kernel GitHub](https://github.com/linkedin/Liger-Kernel); [TRL docs](https://huggingface.co/docs/trl/v1.9.2/reducing_memory_usage)
- **Apple Cut Cross-Entropy (CCE), arXiv 2411.09009:** on Gemma 2 (2B), the loss-computation memory drops from 24 GB to 1 MB, and the classifier head's total training memory from 28 GB to 1 GB. On an A100 SXM4, forward+backward takes 134.8 ms / 1,164 MB for CCE vs 208.8 ms / 28,000 MB for the baseline and 144.0 ms / 16,000 MB for torch.compile. Supports Llama, Phi3, **Mistral** and Gemma2 through `cce_patch("mistral")` or `cce_patch(model)`. Requires PyTorch 2.4+, Triton 3.0+ and an Ampere-or-newer GPU (A100 qualifies). When patched, the model no longer returns logits. — [arXiv 2411.09009](https://arxiv.org/abs/2411.09009); [apple/ml-cross-entropy](https://github.com/apple/ml-cross-entropy)
- **Unsloth** credits its low VRAM to its own gradient checkpointing combined with Apple's CCE. — [Unsloth benchmarks](https://unsloth.ai/docs/basics/unsloth-benchmarks)
- **Axolotl** exposes the same tools as config flags: `flash_attention`, `sample_packing`, `gradient_checkpointing`, `activation_offloading: true` (or `disk` to offload to disk for "much larger context lengths"), the CCE plugin, the Liger plugin (`liger_fused_linear_cross_entropy`), and layer offloading of frozen decoder layers ("especially effective for LoRA/QLoRA"). Sequence parallelism (`sequence_parallel_degree`) splits a sequence across GPUs, so it does not apply to a single GPU. — [Axolotl optimizations guide](https://docs.axolotl.ai/docs/optimizations.html); [Axolotl gradient checkpointing docs](https://docs.axolotl.ai/docs/gradient_checkpointing.html); [Axolotl SP blog](https://www.huggingface.co/blog/axolotl-ai-co/long-context-with-sequence-parallelism-in-axolotl)

### Inferences
- ESTIMATE: memory breakdown for Mistral-7B at **32,768 tokens, batch 1**. Architecture from config: 32 layers, hidden 4096, 8 KV heads, intermediate 14336, vocab 32768.
  - Frozen weights: about 14.5 GB in bf16, or about 4-4.5 GB as 4-bit NF4 plus quant constants.
  - LoRA r=16 on all 7 linear modules: about 42M params. With fp32 weights, grads and two Adam states (about 16 B/param) that is about 0.7 GB, so negligible.
  - Gradient-checkpoint boundaries (one bf16 hidden state per layer): 32768 x 4096 x 2 B x 32 layers is about 8.6 GB. Offloading to CPU (Unsloth/TRL/Axolotl) moves almost all of this off the GPU.
  - Recomputing one layer during backward: the MLP intermediates alone are 32768 x 14336 x 2 B, about 0.94 GB per tensor, with several live at once. Attention with FA2 is linear in sequence length (q/k/v about 0.4 GB). Total about 4-6 GB transient.
  - Logits: 32768 x 32768 is about 1.07B elements, so about 4.3 GB per fp32 copy. The unfused HF path typically holds 2-4 copies (logits, fp32 upcast, softmax grad), about 8-17 GB. A fused/chunked CE brings this to under 1 GB.
- ESTIMATE totals at 32k (excluding CUDA context, about 1 GB, and fragmentation):
  - QLoRA + FA2 + gradient checkpointing, no fused CE: about 25-35 GB. Fits in 40 GB but tight.
  - Same + fused CE (CCE/Liger/Unsloth): about 18-22 GB.
  - Same + offloaded checkpointing (Unsloth): about 10-14 GB.
  - bf16 LoRA (no quantization): add about 10 GB to each figure. With fused CE and offload, about 22-26 GB, so feasible. Without fused CE it is at the edge.
- Mistral's 32,768-token vocabulary is 4x smaller than Llama-3's 128,256, so its logits tensor is 4x smaller at equal length. The Llama-3.1-8B context numbers in Q2 therefore underestimate the context Mistral-7B can reach on the same GPU, especially for the HF+FA2 baseline, where logits dominate.
- With completion-only loss, chunked/fused CE paths that drop -100 positions before the lm_head (the TRL v1.9.2 description) only compute logits for the JSON output tokens. For an 18k-token prompt plus a few-thousand-token JSON, logit memory becomes tiny.
- How the techniques combine: they are mostly complementary because they target different tensors:
  - QLoRA: weights.
  - Gradient checkpointing/offload: activations.
  - FA2: attention memory goes from O(N^2) to O(N).
  - Fused CE: logits.
  - Packing/padding-free: padding waste.
  - Do not stack two loss replacements (Liger FLCE + CCE + TRL chunked); pick one. TRL v1.9.2 explicitly says chunked_nll is incompatible with Liger and PEFT.

### Gaps
- No primary source reports a measured peak-VRAM breakdown for Mistral-7B at exactly 32k on a 40 GB card. The figures above are arithmetic estimates and should be confirmed with a 1-step dry run that logs `torch.cuda.max_memory_allocated()` at 8k, 16k and 32k.
- TRL PR #5575, which has the long-context benchmarks for chunked loss, was not fetched.
- Liger's README does not state whether FLCE works with bitsandbytes 4-bit + PEFT. This is widely used in Axolotl QLoRA configs (see the community example cited in Q5), but no official compatibility statement was found.

## Q2. Measured/reported VRAM vs context for 7-9B models on 40-48 GB; is 32k on 40 GB realistic?

### Takeaway
Yes. Unsloth's own table puts Llama-3.1-8B QLoRA (r=32, all linear layers, bs=1) at about 154k tokens on 40 GB and about 192k on 48 GB, versus about 12k and 15.5k for plain HF+FA2. 32k on 40 GB is therefore well inside the Unsloth envelope. With plain HF+PEFT it needs at least gradient checkpointing + FA2 + a fused CE (and preferably offloading). bf16 LoRA at 32k is plausible but tight, and is an estimate only.

### Cited Findings
- **Unsloth max context for Llama 3.1 (8B) Instruct, 4-bit QLoRA, r=32, all linear layers, bs=1, padded to max length:**

  | GPU VRAM | Unsloth | HF + FA2 |
  |---|---|---|
  | 8 GB | 2,972 | OOM |
  | 12 GB | 21,848 | 932 |
  | 16 GB | 40,724 | 2,551 |
  | 24 GB | 78,475 | 5,789 |
  | 40 GB | 153,977 | 12,264 |
  | 48 GB | 191,728 | 15,502 |
  | 80 GB | 342,733 | 28,454 |

  The page also claims about 12x longer context, more than 70% VRAM reduction and 2x speed vs HF+FA2. — [Unsloth benchmarks](https://unsloth.ai/docs/basics/unsloth-benchmarks); same 8B numbers in the [Unsloth Llama 3.3 blog](https://unsloth.ai/blog/llama3-3)
- The GPU used for the 8B context test is not stated; the 70B test used an 80GB A100. — [Unsloth benchmarks](https://unsloth.ai/docs/basics/unsloth-benchmarks)
- Mistral 7B on a 24 GB L4: HF about 5K, HF+FA2 about 14K, Unsloth about 56K (April 2024, before CCE was integrated). — [Unsloth long-context blog](https://unsloth.ai/blog/long-context)
- Liger: LLaMA-3-8B HF baseline OOMs at about 4K and Liger reaches about 16K. That was full fine-tuning with FSDP, bs=8, on 8x A100, so it is not comparable to single-GPU LoRA. — [Liger-Kernel GitHub](https://github.com/linkedin/Liger-Kernel)
- Unsloth reports a short-context QLoRA run of Mistral 7B on 1x A100 at 12.4 GB peak VRAM (62% less than HF). Sequence length is not stated. — [Unsloth Mistral benchmark blog](https://unsloth.ai/blog/mistral-benchmark)

### Inferences
- The Unsloth 40 GB figure (about 154k for Llama-8B with a 128k vocab) leaves more than 4x headroom for a 32k Mistral-7B run with QLoRA. 32k on 40 GB with Unsloth is realistic even at r=16-64.
- The plain HF+FA2 baseline in Unsloth's table (12k on 40 GB for Llama-8B) is mostly limited by the 128k-vocab logits. For Mistral (32k vocab), plain HF+FA2+gradient checkpointing would plausibly reach about 20-30k, but 32k is not safe without a fused CE. ESTIMATE, not measured.
- bf16 LoRA at 32k on 40 GB is plausible only with fused CE + offloaded checkpointing (about 22-26 GB, ESTIMATE). QLoRA is the safer default on 40 GB.

### Gaps
- No torchtune or LLaMA-Factory single-GPU 32k measurements for 7-8B were found in this pass.
- No GitHub issue was found with an exact Mistral-7B 32k on 40 GB OOM report and numbers.

## Q3. Mistral-7B-v0.3 native context and sliding-window caveats

### Takeaway
v0.3 has `max_position_embeddings=32768`, `rope_theta=1e6` and `sliding_window=null` (full attention). There is no sliding-window caveat for v0.3. The 4096-token sliding window belonged to the original Mistral-7B (v0.1). Because the vocabulary is 32768, a 32k-token training sequence uses the model's full native window: the prompt plus the JSON together must stay at or below 32768 tokens.

### Cited Findings
- Mistral-7B-Instruct-v0.3 config: 32 layers, hidden 4096, 32 heads / 8 KV heads, intermediate 14336, `max_position_embeddings` 32768, no sliding window, vocab 32768, `rope_theta` 1,000,000, EOS id 2, BOS id 1. — [HF config.json](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3/raw/main/config.json)
- Derivative v0.3 configs (including Unsloth-based LoRA repos) also show `sliding_window: null` and 32768. — [dolphin-2.9.3-mistral-7B-32k config](https://huggingface.co/cognitivecomputations/dolphin-2.9.3-mistral-7B-32k/blob/d246ad570fc0d1a1285704f5d409eba4bad4f425/config.json); [LoRA from Unsloth base](https://huggingface.co/Beats119/lora_model/blob/7d13e45237b72ee3f6faebcba0f9581f4ca296f3/config.json)
- Some third-party pages still describe a 4096-token sliding window for "Mistral 7B". This conflicts with the v0.3 config and appears to describe v0.1. — [mintlify OminiX page](https://www.mintlify.com/OminiX-ai/OminiX-MLX/llm/mistral) (conflicting, lower-quality source)

### Inferences
- No RoPE scaling is needed for at most 32k. Anything beyond 32,768 tokens (for example an 18k excerpt plus a very long JSON plus the chat template) would leave the native window. Such examples should be shortened, not extended with RoPE scaling.
- When merging or saving, check that `sliding_window` stays null.

### Gaps
- No source was found evaluating Mistral-7B-v0.3's actual long-context quality (for example RULER scores at 16k-32k). Native support does not guarantee good recall near 32k.

## Q4. Practical alternatives when the full sequence does not fit; GPU rental costs (2026)

### Takeaway
On one GPU, sequence parallelism is not an option. The realistic levers are:
1. Make the input shorter: retrieve or select only the relevant paper sections so each example is about 8-16k tokens.
2. Chunk the paper into segments, each with its own target JSON (the per-chunk output must be a complete, closed JSON array).
3. Always use completion-only loss.
4. Rent an 80 GB A100 for about $1.2-2.8/h.

Option 2 also removes the truncation problem, because every training example ends with "]" + EOS.

### Cited Findings
- Sequence parallelism splits each sequence across GPUs (`sequence_parallel_degree`), so it needs more than one GPU. — [Axolotl SP blog](https://www.huggingface.co/blog/axolotl-ai-co/long-context-with-sequence-parallelism-in-axolotl)
- TRL `packing_strategy="bfd_split"` splits over-long sequences into chunks of at most max_length instead of dropping tokens, following "Fewer Truncations Improve Language Modeling" (arXiv 2404.10830). This keeps the tokens but **breaks the prompt-to-JSON dependency**: a later chunk loses the paper context, and an earlier chunk loses the closing "]". — [TRL reducing memory](https://huggingface.co/docs/trl/v1.9.2/reducing_memory_usage)
- Completion-only loss is TRL's default for prompt-completion datasets (`completion_only_loss=None` behaves as True for them). — [TRL SFT docs v1.15.0](https://huggingface.co/docs/trl/sft_trainer)
- **A100 80GB rental snapshots (2026; mostly from aggregator blogs, so verify on provider pages):**
  - RunPod: about $1.19 (Community PCIe), $1.39-1.49 (SXM) and $1.59 (Secure), billed per second.
  - Lambda: $2.06-2.79/h for SXM 80GB; one source says SXM is sold only in 8-GPU nodes.
  - Jarvislabs: $1.49/h.
  - Hyperstack: $1.35/h (PCIe).
  - Vast.ai marketplace: about $0.76/h, with a wide range.
  - AWS p4de: about $3.43 per GPU-hour, with an 8-GPU minimum.
  - Sources: [Jarvislabs A100 price 2026](https://jarvislabs.ai/blog/a100-price); [deploybase A100 price](https://deploybase.ai/articles/a100-price); [RunPod pricing review](https://www.hackceleration.com/labs/runpod-pricing)
- Modal: A100-80GB about $2.50/h and H100 about $3.95/h ($0.001097/s), per-second billing. These are comparison-site snapshots from Sep-Oct 2026. — [computeprices Modal compare](https://computeprices.com/compare/gpu-vs-modal); [Spheron on Modal pricing](https://www.spheron.network/blog/modal-gpu-pricing-2026-per-second-billing/)

### Inferences
- ESTIMATE of run cost: about 250 gold rows (gold v1.3 has 246) x 3 epochs at about 30k tokens is about 22M tokens. Assuming a QLoRA 7B throughput of a few thousand tokens/s on an A100 with Unsloth (an unmeasured assumption), that is roughly 2-6 GPU-hours, so under $10-20 on an 80 GB rental. Renting is cheap enough that it should not be ruled out if 40 GB proves fragile.
- For extraction, retrieval or section filtering (methods, results, tables) is usually the strongest lever. It cuts memory, speeds training and inference, and reduces long-context recall failures. It must be applied identically at inference time.
- Per-chunk targets need a defined merge or de-duplication step at inference. They also change the task, because the model must learn to output `[]` for chunks with no relevant content.

### Gaps
- Colab Pro/Pro+ A100 availability and compute-unit pricing in 2026 were not verified.
- Lambda's single-GPU A100 80GB on-demand availability in 2026 is unclear; sources conflict.

## Q5. Recommended stack (versions) and a 32k config block

### Takeaway
There are two routes, with QLoRA as the 40 GB default:
- **Route A (simplest, most headroom): Unsloth.** Load the 4-bit Mistral-v0.3, set `use_gradient_checkpointing="unsloth"` (offloaded), and keep its built-in CCE. Then use TRL SFTTrainer with `max_length=32768`, `completion_only_loss=True`, `packing=False`, bs=1 plus gradient accumulation.
- **Route B (pure HF):** transformers + peft + bitsandbytes + flash-attn 2 + TRL with `gradient_checkpointing=True`, `activation_offloading=True`, and Liger FLCE **or** CCE (not both).

### Cited Findings
- The current TRL docs are **v1.15.0** and link to **transformers v5.19.0** docs. In SFTConfig, defaults include `max_length=1024`, `truncation_mode="keep_start"` (the only supported value; `keep_end` is deprecated), `packing=False`, `packing_strategy="bfd"`, `padding_free=False`, `completion_only_loss=None`, `assistant_only_loss=False`, `gradient_checkpointing=True`, `bf16=True` and `use_liger_kernel=False`. If `model_init_kwargs` has no dtype, the model loads in **float32**, unlike transformers v5, which infers it. `quantization_config` plus `peft_config` give QLoRA. — [TRL SFT docs v1.15.0](https://huggingface.co/docs/trl/sft_trainer)
- TRL suggests about 1e-4 learning rate for adapters. — [TRL SFT docs](https://huggingface.co/docs/trl/sft_trainer)
- CCE needs PyTorch 2.4+, Triton 3.0+ and Ampere+. Install with `pip install "cut-cross-entropy[transformers] @ git+https://github.com/apple/ml-cross-entropy.git"`. — [apple/ml-cross-entropy](https://github.com/apple/ml-cross-entropy)
- Liger: `pip install liger-kernel`; the latest release named on the README news list is v0.5.0, and the newest news item is from Dec 2025. — [Liger-Kernel GitHub](https://github.com/linkedin/Liger-Kernel)
- A community Axolotl QLoRA + Liger example (Axolotl 0.4.1, Mistral-Nemo) uses `load_in_4bit: true`, `adapter: qlora`, the LigerPlugin and `liger_fused_linear_cross_entropy: true` at `sequence_len: 8192`. — [HF repo config](https://huggingface.co/PygTesting/sum_qlora_4pochs)

### Inferences
- Version pinning: only TRL 1.15.0 / transformers 5.19.0 were confirmed from docs in this pass. Pin whatever installs together, then record `pip freeze` in the run log. Reasonable floors: torch >= 2.4, flash-attn 2.x built for the installed torch/CUDA, bitsandbytes recent, peft recent, and Unsloth's latest release. These floors are not verified against release pages.
- Route A sketch (Unsloth), UNTESTED. The Unsloth API names come from common usage and should be checked against current Unsloth docs:

```python
from unsloth import FastLanguageModel
model, tok = FastLanguageModel.from_pretrained(
    "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
    max_seq_length=32768, load_in_4bit=True, dtype=None)
model = FastLanguageModel.get_peft_model(
    model, r=16, lora_alpha=32, lora_dropout=0.0,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    use_gradient_checkpointing="unsloth")   # offloaded checkpointing
from trl import SFTConfig, SFTTrainer
args = SFTConfig(
    max_length=32768,            # must be >= longest example; NEVER let it truncate the completion
    packing=False,               # one example per step at 32k anyway
    completion_only_loss=True,   # prompt-completion dataset
    per_device_train_batch_size=1, gradient_accumulation_steps=8,
    learning_rate=1e-4, lr_scheduler_type="cosine", warmup_steps=10,
    num_train_epochs=3, bf16=True, optim="adamw_8bit",
    logging_steps=1, save_strategy="epoch", seed=42)
```

- Route B (pure HF/TRL), UNTESTED:

```python
quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                           bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
args = SFTConfig(max_length=32768, packing=False, completion_only_loss=True,
                 gradient_checkpointing=True,
                 gradient_checkpointing_kwargs={"use_reentrant": False},
                 activation_offloading=True,
                 use_liger_kernel=True,   # OR cce_patch(model); not both
                 model_init_kwargs={"dtype": torch.bfloat16,
                                    "attn_implementation": "flash_attention_2"},
                 per_device_train_batch_size=1, gradient_accumulation_steps=8,
                 learning_rate=1e-4, bf16=True)
trainer = SFTTrainer(model="mistralai/Mistral-7B-Instruct-v0.3", args=args,
                     train_dataset=ds, peft_config=LoraConfig(r=16, lora_alpha=32,
                     target_modules="all-linear", task_type="CAUSAL_LM"),
                     quantization_config=quant)
```

- Pre-flight procedure:
  1. Run 2 steps on the **longest** example at max_length 8k, 16k and 32k, logging `torch.cuda.max_memory_allocated()`.
  2. Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to reduce fragmentation OOMs. This is a common practice; no source was fetched for it.
  3. Sort or bucket by length. Since bs=1, padding waste is already zero.

### Gaps
- The current Unsloth release number and its exact transformers/TRL pins were not fetched.
- The current PyPI versions of peft, bitsandbytes and flash-attn were not fetched.
- Liger FLCE + bitsandbytes QLoRA + TRL 1.15 compatibility was not confirmed from an official source.

## Q6. Failure mode: truncated training examples lose the closing "]" and EOS; how SFTTrainer settings interact

### Takeaway
TRL truncates at `max_length` (default **1024**) by keeping the **start** (`truncation_mode="keep_start"` is the only supported value). For prompt+completion examples longer than max_length, the end of the completion is cut, and that is exactly where the closing "]" and EOS sit. The model then never sees a loss target for closing the array or stopping, which matches the observed behaviour. With `completion_only_loss=True`, any example whose prompt alone exceeds max_length contributes **zero** loss tokens. Packing with `"bfd"` also discards overflow tokens. The fix is to never truncate: set max_length to at least the longest tokenized example (or drop and shorten over-long examples up front), and assert that every example's last supervised tokens are "]" followed by EOS.

### Cited Findings
- `max_length`: "Sequences longer than `max_length` are truncated from the left or right depending on `truncation_mode`". Default 1024; `None` disables truncation. `truncation_mode` default `"keep_start"` is "the only supported value". — [TRL SFT docs v1.15.0](https://huggingface.co/docs/trl/sft_trainer)
- TRL lists "Loss of information: Important tokens at the end of sequences may be discarded" as a drawback of truncation. It also states that `"bfd"` packing discards overflow tokens, while `"bfd_split"` keeps them by splitting. — [TRL reducing memory](https://huggingface.co/docs/trl/v1.9.2/reducing_memory_usage)
- For pre-tokenized datasets, "Sequences are truncated to `max_length` during preparation". An explicit `labels` column (with -100 masking) is used as-is, so custom masking is possible. — [TRL SFT docs v1.15.0](https://huggingface.co/docs/trl/sft_trainer)
- `eos_token`: "Token used to indicate the end of a turn or sequence. If `None`, it defaults to `processing_class.eos_token`". The docs warn that the EOS must be aligned with the chat template "to ensure the model's responses terminate correctly". — [TRL SFT docs v1.15.0](https://huggingface.co/docs/trl/sft_trainer)
- If the tokenizer has no pad token, TRL uses EOS as pad. Padding positions are masked with -100. — [TRL SFT docs v1.15.0](https://huggingface.co/docs/trl/sft_trainer)
- Mistral-7B-Instruct-v0.3 EOS id is 2 (`</s>`). — [HF config.json](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3/raw/main/config.json)

### Inferences
- Risk: if EOS is used as pad and a custom collator masks all pad-id positions, the **real** EOS also gets masked to -100 and the model never learns to stop. This is a classic pitfall. Check that the final EOS label is not -100.
- Recommended data checks before training:
  1. Tokenize every example with the exact chat template.
  2. Assert `len(ids) <= max_length`.
  3. Assert that the decoded last non-masked labels end with `]` + `</s>`.
  4. Assert that the count of supervised tokens is > 0.
  5. Log the length distribution: p50, p95 and max.
- Examples above about 32k (Mistral's native window) should be shortened through section retrieval or split into chunks with complete per-chunk JSON. Never silently truncate them.
- At inference, set `max_new_tokens` large enough for the longest gold JSON plus a margin, and keep the same template. A model trained on complete examples should then emit "]" and EOS.

### Gaps
- The docs do not show whether TRL v1.15 appends EOS automatically for prompt-completion data when the chat template already adds `</s>` (possible double EOS), or whether it appends nothing for non-conversational text. This should be checked empirically by inspecting `trainer.train_dataset[0]["input_ids"][-3:]`.
