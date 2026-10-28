# Reliable variable-length JSON array generation (truncation and repetition loops)

Context: Mistral-7B-Instruct-v0.3 + LoRA for evidence extraction. Failure modes observed: (1) over-generation of 51 identical objects with no closing `]` (repetition loop until max_new_tokens), (2) empty or useless output. Target schema: array of objects with `predictor, canonical_construct, outcome, effect_type, direction, plausible_values_handling, student_weights_used, evidence_text`. Deployment: HF (TGI / Inference Endpoints / Inference Providers) or Together AI, minimal budget. Research date: 2026-10-09.

Library versions (PyPI latest, checked 2026-10-09 via pypi.org JSON API): outlines 1.3.3 (2026-08-06), lm-format-enforcer 0.11.3 (2025-08-24, slowest release cadence), xgrammar 0.2.8 (2026-09-24), llguidance 1.9.1 (2026-09-30), instructor 1.17.0 (2026-09-09), vllm 0.31.0 (2026-10-05), sglang 0.5.21 (2026-10-01), trl 1.15.0 (2026-10-08), peft 0.21.2, transformers 5.19.0, huggingface_hub 2.2.0, together 2.40.0, mistral-common 1.12.0 — [PyPI](https://pypi.org/)

## (a) Constrained decoding: outlines, lm-format-enforcer, xgrammar, llguidance

### Takeaway
Grammar-constrained decoding (xgrammar or llguidance via vLLM/SGLang) guarantees syntactically valid, schema-conforming JSON and, with `maxItems`, a guaranteed closing bracket, but it does not by itself stop semantic repetition: a model can loop "within schema" (identical objects, or unbounded strings/numbers) until the item cap or token cap is hit. On quality, the 2024 "Let Me Speak Freely?" degradation claim was contested, and JSONSchemaBench (2025) found constrained decoding slightly improves accuracy; but backend schema-feature coverage varies a lot, so `maxItems`/`enum` support must be verified per backend.

### Cited Findings
- vLLM structured outputs: backends `xgrammar` and `guidance` (llguidance) are named; default backend `auto` picks per request; set via `--structured-outputs-config.backend`. Online API: `extra_body={"structured_outputs": {"json": schema}}` (also `choice`, `regex`, `grammar`, `structural_tag`), or OpenAI `response_format={"type":"json_schema", ...}`. Offline: `SamplingParams(structured_outputs=StructuredOutputsParams(json=schema))`. The old `guided_*` fields and `guided_decoding_backend` were removed in v0.12.0 — [vLLM docs: Structured Outputs](https://docs.vllm.ai/en/latest/features/structured_outputs.html)
- Regex dialect differs: xgrammar, guidance, outlines use Rust-style regex; lm-format-enforcer uses Python `re` — [vLLM docs](https://docs.vllm.ai/en/latest/features/structured_outputs.html)
- llguidance: enforces arbitrary CFGs, a "large subset" of JSON Schema, regex, Lark-like grammars; integrated in vLLM since v0.8.2 (March 2025), SGLang since v0.4.4 (`--grammar-backend llguidance`), llama.cpp b4613 (`-DLLAMA_LLGUIDANCE=ON`); powers OpenAI Structured Outputs; claims ~50 µs CPU per token mask on a 128k vocab, negligible compile time — [guidance-ai/llguidance README](https://github.com/guidance-ai/llguidance)
- JSONSchemaBench (Geng et al., Microsoft Research/EPFL, arXiv 2501.10868, ICML 2025 workshop): ~10K real-world schemas; compares Guidance, Outlines, llama.cpp, XGrammar, OpenAI, Gemini on efficiency, coverage, quality — [arXiv 2501.10868](https://arxiv.org/abs/2501.10868)
- JSONSchemaBench empirical coverage (Llama-3.2-1B for open engines): GlaiveAI Guidance 0.96 / llama.cpp 0.95 / Outlines 0.95 / XGrammar 0.93; GitHub-Hard Guidance 0.41 / llama.cpp 0.39 / Outlines 0.03 / XGrammar 0.28 — [JSONSchemaBench HTML v2](https://arxiv.org/html/2501.10868v2)
- JSONSchemaBench: Outlines' low compliance is mostly timeouts; features such as `minItems`, `maxItems`, `enum`, arrays can take 40 s to 10 min to compile in Outlines. Outlines grammar compile time 3.48 s on GlaiveAI vs ~0 for Guidance. In the JSON Schema Test Suite, XGrammar had 38 "under-constrained" categories (accepts outputs that violate the schema) vs 1 for Guidance — [JSONSchemaBench HTML v2](https://arxiv.org/html/2501.10868v2)
- JSONSchemaBench quality (Llama-3.1-8B-Instruct): GSM8K LM-only 80.1% vs XGrammar 83.7%, Guidance 83.8%, Outlines 81.6%, llama.cpp 82.4%; Last Letters 50.7% vs 51.2–54.0%; i.e., constrained decoding did not hurt and slightly helped — [JSONSchemaBench HTML v2](https://arxiv.org/html/2501.10868v2)
- "Let Me Speak Freely?" (Tam et al., arXiv 2408.02442, Aug 2024, v3 Oct 2024) reports "a significant decline in LLMs reasoning abilities under format restrictions", with stricter formats hurting more — [arXiv 2408.02442](https://arxiv.org/abs/2408.02442)
- Rebuttal "Say What You Mean" (Will Kurt, dottxt): re-ran evaluations with proper prompting; structured vs unstructured GSM8K 0.78 vs 0.77, Last Letter 0.77 vs 0.73, Shuffle Object 0.44 vs 0.41. Note: dottxt sells structured generation (commercial interest) — [dottxt blog](https://blog.dottxt.ai/say-what-you-mean.html)
- Repetition loops do occur inside schema-constrained output: Gemini schema-constrained output with an array + min/maxItems degenerated into repeated digits until MAX_TOKENS; temperature 0 did not prevent it — [Google AI forum](https://discuss.ai.google.dev/t/gemini-3-7-flash-schema-constrained-json-output-degenerates-into-repeated-0-until-maxoutputtokens-regression-vs-gemini-3-flash-preview/178681); a second report: valid JSON for ~14 KB then a loop inside a number field ("a JSON number's digit sequence is grammatically unbounded") — [Google AI forum](https://discuss.ai.google.dev/t/structured-output-repetition-loop-inside-a-json-number-literal-runs-to-max-tokens-flash-vertex/175138)
- llama.cpp mitigated unbounded-number loops in its JSON-schema-to-grammar by capping integer/fraction digits at 15 — [llama.cpp json_schema_to_grammar commit history](https://cdn04132025.gitlink.org.cn/replica/llama.cpp/commits/branch/master/examples/json_schema_to_grammar.py)
- "Good-Enough Structured Generation" (NeurIPS 2025) reports grammar-constrained implementations can cover as little as ~50% of real-world schemas and argues validation-with-feedback can beat complex decoding — [NeurIPS 2025 poster page](https://neurips.cc/virtual/2025/131662)
- dottxt later argued poorly implemented masking (it named XGrammar and llguidance) can degrade accuracy — secondary source only (LinkedIn repost surfaced via search), not verified — [dottxt LinkedIn](https://www.linkedin.com/company/dottxt)

### Inferences
- For this task, constrained decoding converts failure mode (1) from "unparseable, unterminated" into "parseable but possibly padded with duplicates up to maxItems". That is a strict improvement (post-hoc dedupe becomes trivial), but it is not a fix for the underlying loop.
- Schema design that limits loop surface: `maxItems` set to a realistic cap (e.g., 95th percentile of gold counts per input + margin; gold v1.3 has 246 rows, so per-paper distribution can be computed); `enum` for `effect_type`, `direction`, `plausible_values_handling`, `student_weights_used` (categorical fields, shrinks the space and removes free-text drift); `maxLength` on `evidence_text` and `predictor` strings (prevents in-string loops analogous to the digit-loop reports); `additionalProperties: false`; all fields `required`. Prefer booleans/enums over numbers where possible.
- `uniqueItems: true` should not be relied on: I found no confirmation that xgrammar or llguidance enforce it (enforcing it requires remembering previously emitted objects, which is not a finite grammar property). Dedupe in post-processing instead.
- Backend choice: in vLLM use `guidance` (llguidance) or `xgrammar`; avoid Outlines for schemas with `maxItems`/`enum` on arrays due to compile blowup reported by JSONSchemaBench; lm-format-enforcer is maintained but slower-moving (last release Aug 2025).
- Constrained decoding is useless if training already fails to emit EOS: the grammar will allow EOS after `]` but the model may assign it low probability; with a grammar, after `]` the only legal token is EOS (or whitespace if allowed), which forces termination. This is a strong reason to use it here.

### Gaps
- Explicit per-backend support matrix for `minItems`/`maxItems`/`uniqueItems`/`maxLength` in xgrammar 0.2.x and llguidance 1.9.x was not retrieved; check `llguidance/docs/json_schema.md` and the xgrammar docs before relying on them.
- No independent (non-vendor) replication settling the "Let Me Speak Freely" dispute was found beyond JSONSchemaBench.
- No study found that measures repetition-loop rates of fine-tuned 7B models with vs. without grammar constraints.

## (b) instructor library with Pydantic

### Takeaway
instructor (1.17.0, Sept 2026) wraps provider clients so a Pydantic model is the response type; it validates output and re-asks the model on validation failure (`max_retries`). It does not constrain decoding itself; each retry is a full extra generation, so for a looping 7B model retries multiply cost and latency rather than fix the root cause.

### Cited Findings
- instructor extracts structured data via Pydantic models, validates responses, and automatically re-asks on validation failure (`max_retries=3` in examples); available in Python plus TS, Go, Ruby, Elixir, Rust — [instructor docs](https://python.useinstructor.com/)
- Unified client: `instructor.from_provider("openai/gpt-4o-mini")`, `("ollama/llama3.2")`, sync/async; `client.create(response_model=..., messages=...)`; `create_iterable(...)` streams multiple objects one by one; `create_partial(...)` streams partial objects — [instructor docs](https://python.useinstructor.com/)
- Listed providers include OpenAI, Anthropic, Google, Vertex, Mistral (and Mixtral via Together integration), Ollama, llama-cpp-python, Cohere, LiteLLM, 15+ total; vLLM mentioned for local open-source models; HuggingFace not listed on the landing page — [instructor docs](https://python.useinstructor.com/); Together integration post — [instructor JS blog: Together](https://js.useinstructor.com/blog/2024/02/01/together/)
- Latest version 1.17.0 released 2026-09-09 — [PyPI instructor](https://pypi.org/project/instructor/)

### Inferences
- Any OpenAI-compatible server (vLLM on HF Inference Endpoints, TGI Messages API, Together) can be used via `instructor.from_openai(OpenAI(base_url=...))`; the mode (TOOLS vs JSON_SCHEMA vs MD_JSON) decides whether the server's own constrained decoding is engaged. Pairing instructor in TOOLS/JSON_SCHEMA mode with a server that enforces the schema gives syntax guarantees from the server and semantic validators (e.g., `evidence_text` must be a substring of the source chunk, enum normalization, dedupe) from Pydantic.
- Cost of retries: worst case (1 + max_retries) full generations per input, and for a looping model each failed attempt likely runs to max_tokens. Keep `max_retries` at 1–2 and cap `max_tokens`.
- `create_iterable` (yielding `Iterable[Evidence]`) is a natural fit for variable-length arrays: each element is validated individually, so a late failure does not discard earlier items.

### Gaps
- The landing page did not document the exact reask prompt format or the mode list; the "Mode Comparison" page was not fetched. Exact retry token overhead is undocumented.

## (c) Function calling / provider JSON modes (Mistral tools, Together, TGI, HF Inference Providers)

### Takeaway
Mistral-7B-Instruct-v0.3 is a confirmed tool-calling model in vLLM (`--tool-call-parser mistral`), and named/required tool choice there is enforced by structured outputs. TGI supports a `grammar` parameter (outlines-based) but is now in maintenance mode; vLLM on HF Inference Endpoints is the better self-host target. Together's serverless LoRA inference has been discontinued, so a custom LoRA on Together means a paid dedicated endpoint.

### Cited Findings
- vLLM: `mistralai/Mistral-7B-Instruct-v0.3` confirmed with `--tool-call-parser mistral`; Mistral 7B struggles with parallel tool calls; with the HF tokenizer backend the template expects 9-digit tool-call IDs, fixed by `examples/tool_chat_template_mistral.jinja` or `tool_chat_template_mistral_parallel.jinja`; Mistral-format flags `--tokenizer_mode mistral --config_format mistral --load_format mistral`; HF-format flags `--tokenizer_mode hf --config_format hf --load_format hf --tool-call-parser mistral --chat-template examples/tool_chat_template_mistral_parallel.jinja` — [vLLM docs: Tool Calling](https://docs.vllm.ai/en/latest/features/tool_calling.html)
- vLLM: named function `tool_choice` and `tool_choice="required"` use structured outputs, so the arguments are guaranteed to parse against the tool schema ("not necessarily high quality"); first request may incur FSM compile latency — [vLLM docs: Tool Calling](https://docs.vllm.ai/en/latest/features/tool_calling.html)
- TGI: JSON/regex grammars since v1.4.3; `/generate` takes `parameters.grammar = {"type":"json","value": schema}`; `/v1/chat/completions` supports `tools` and `tool_choice` (`auto`, `none`, `required`, or a named function); implemented with outlines; grammar compile can take seconds on first request then cached — [TGI docs: Guidance](https://huggingface.co/docs/text-generation-inference/basic_tutorials/using_guidance.md)
- TGI README: "text-generation-inference is now in maintenance mode"; only minor fixes accepted; recommends vLLM and SGLang; latest referenced image 3.3.5 — [TGI GitHub](https://github.com/huggingface/text-generation-inference)
- A Feb 2025 blog found TGI via routed providers worked only with `json_object` with the schema given as `value` inside `response_format`; Groq required schema in system prompt — [Structured output quirks blog](https://barisdeniz.is-a.dev/posts/llm-providers-structured-output/index.html)
- HF Inference Endpoints offers vLLM (and TGI, SGLang); any vLLM engine arg can be passed as container arg, e.g. `enable_lora=true`; docs say "TGI, vLLM, and SGLang will be equivalently good options" for most cases — [HF Inference Endpoints: vLLM](https://huggingface.co/docs/inference-endpoints/engines/vllm)
- HF Inference Providers: OpenAI-compatible router `https://router.huggingface.co/v1`, structured output via `response_format` (Responses API with `.parse`); behavior varies per provider (forum reports Nebius uses a different parameter for schema-constrained output) — [HF Inference Providers: Responses API guide](https://huggingface.co/docs/inference-providers/en/guides/responses-api); [HF forum](https://discuss.huggingface.co/t/standardised-rest-api-properties-for-structured-outputs/165733)
- Together structured outputs: `response_format={"type":"json_schema","json_schema":{"name":..., "schema":...}}` and `{"type":"regex","pattern":...}`; every JSON-mode model also supports regex; supported models listed in serverless/dedicated catalogs (examples: Qwen/Qwen3.5-9B, Llama-3.3-70B-Instruct-Turbo); warns of truncation when `max_tokens` too low (`finish_reason: "length"`) and that models copy errors from malformed example JSON in prompts; recommends also stating the schema in the system prompt — [Together docs: Structured Outputs](https://docs.together.ai/docs/inference/chat/structured-outputs)
- Older Together JSON-mode page listed `mistralai/Mistral-7B-Instruct-v0.1` among supported models (dated list) — [Together docs: JSON mode](https://docs.together.ai/docs/json-mode)
- Together: "Serverless LoRA inference (including Serverless Multi-LoRA) has been discontinued"; adapters are served on dedicated endpoints — [Together docs: LoRA inference](https://docs.together.ai/docs/lora-inference)
- Together dedicated pricing reported by third parties ranges widely ($2.50–4/h per H100 vs $7/h H100/H200 on-demand); unverified against together.ai/pricing — [CloudZero](https://www.cloudzero.com/together-ai-pricing/); [eesel.ai](https://eesel.ai/blog/together-ai-pricing)

### Inferences
- Cheapest controllable path: merge LoRA (or `--enable-lora`) and serve on an HF Inference Endpoint with the vLLM engine on a single small GPU (7B bf16 ~15 GB; an L4/A10G 24 GB class instance), scale-to-zero between batch runs. This gives full control of `structured_outputs`, `max_tokens`, `repetition_penalty`, `stop`, and LoRA hot-swap. Inference Providers (serverless) cannot host a custom LoRA, so they are only usable for base-model baselines.
- Tool calling vs plain JSON schema: for a LoRA fine-tuned on raw JSON arrays, switching to `[TOOL_CALLS]` format would require re-formatting SFT data into Mistral tool-call syntax. Plain `response_format: json_schema` / `structured_outputs.json` with a wrapper object `{"items": [...]}` (top-level object, array inside with `maxItems`) achieves the same enforcement without retraining. A single tool with `tool_choice` named is equivalent under vLLM because both route through structured outputs.
- TGI should not be the primary target for new work given maintenance mode; if used, the outlines-based grammar has the Outlines `maxItems` compile-time risk noted in (a).

### Gaps
- Whether Together dedicated endpoints apply `response_format` json_schema to user LoRA adapters was not documented.
- Current Together catalog status of Mistral-7B-Instruct-v0.3 for JSON mode was not confirmed.
- Exact HF Inference Endpoints hourly prices for L4/A10G were not retrieved.

## (d) Training-side fixes: EOS after closing bracket, pad/EOS pitfalls, completion-only loss, truncation

### Takeaway
The "51 identical objects, no `]`" pattern is the classic signature of a model that never learned to emit EOS (and/or saw truncated targets). The two most likely causes are (1) EOS masked out of the loss because `pad_token == eos_token` with an ID-based collator, and (2) targets truncated by `max_length` (TRL default 1024, `keep_start`), which removes the closing `]` and `</s>` from long examples. Both must be checked directly on a collated batch.

### Cited Findings
- HF forum (Mar 2024, Mistral-7B-Instruct): setting `pad_token_id = eos_token_id` makes the collator set EOS labels to -100, so "the fine-tuned model never sees the eos_token and keeps generating undefinitely"; fixes: `tokenizer.pad_token = tokenizer.unk_token`, or add a dedicated `<pad>` and `resize_token_embeddings`, or unmask the first EOS — [HF forum](https://discuss.huggingface.co/t/mistral-trouble-when-fine-tuning-dont-set-pad-token-id-eos-token-id/77928)
- Same mechanism documented for GPT-2 with `DataCollatorForLanguageModeling` — [HF forum GPT-2](https://discuss.huggingface.co/t/gpt2-finetuned-with-eos-token-will-never-yield-eos-token-during-generation/15437); Mistral SFTTrainer write-up recommends `pad_token = unk_token` — [clay-atlas](https://clay-atlas.com/us/blog/2024/01/01/mistral-sft-trainer-cannot-generate-eos-token/)
- A Mistral-7B paper explicitly avoids EOS as pad "because the loss is masked out for padding tokens", preserves loss on `<eos>` at the end of each sequence — [arXiv 2509.00921](https://arxiv.org/pdf/2509.00921)
- Conflicting advice: one guide recommends `pad_token = eos_token`; only safe if the collator masks padding by attention mask/position rather than token ID — [Sabr Research cookbook](https://sabrresearch.com/cookbooks/fix-eos-token-infinite-generation)
- TRL 1.15.0 SFTTrainer: if no pad token is set, `tokenizer.eos_token` is used as default pad; `SFTConfig.pad_token` is deprecated (set `tokenizer.pad_token` directly); `eos_token` parameter defaults to tokenizer EOS and must align with the chat template's end token — [TRL SFTTrainer docs v1.15.0](https://huggingface.co/docs/trl/sft_trainer)
- TRL 1.15.0: `max_length` defaults to 1024; longer sequences truncated with `truncation_mode="keep_start"` (only supported mode), i.e., the end of the completion is cut; `max_length=None` disables truncation — [TRL SFTTrainer docs](https://huggingface.co/docs/trl/sft_trainer)
- TRL 1.15.0: prompt-completion datasets compute loss on completion only by default (`completion_only_loss=None` -> True for prompt-completion); `assistant_only_loss=True` requires `{% generation %}`/`{% endgeneration %}` in the chat template (auto-patched only for known families such as Qwen3); packing strategies `bfd` (truncates overflow), `bfd_split`, `wrapped` (cuts mid-sequence) — [TRL SFTTrainer docs](https://huggingface.co/docs/trl/sft_trainer)
- Pre-tokenized datasets: an explicit `labels` column (-100 for excluded tokens) is used as is — [TRL SFTTrainer docs](https://huggingface.co/docs/trl/sft_trainer)

### Inferences
- Concrete checks for this project's training script:
  1. Collate one batch and assert the last non-pad label of every example equals `tokenizer.eos_token_id` (2 for Mistral v0.3) and is not -100.
  2. Assert the decoded target ends with `]</s>` (no trailing text, no missing bracket).
  3. Compute token length of prompt+completion for every training example; set `max_length` above the max (or `None`) and drop/split examples that cannot fit, instead of letting `keep_start` silently cut the `]` and EOS. Long papers make this likely: a long prompt plus a long array easily exceeds 1024 tokens.
  4. Do not use `packing_strategy="wrapped"`; with `bfd`, overflow is truncated (same risk); prefer no packing or `bfd_split` only if examples fit.
  5. Use prompt-completion format (completion = JSON array string) so completion-only loss is default and EOS is appended to the completion.
  6. Set `tokenizer.pad_token = tokenizer.unk_token` (or a dedicated pad) for Mistral as defense in depth even if the current TRL collator pads labels with -100 by position.
- Train on empty-array examples (`[]` + EOS) for inputs with no evidence, so the model has a learned short exit path; without them, a model forced to produce at least one object may loop or hallucinate. This also addresses failure mode (2) "returns nothing useful" by giving a valid explicit empty output.
- Serialize gold arrays deterministically (fixed key order, compact separators, consistent ordering of items e.g. by appearance in text). Ordered-by-position lists give the model a monotone "pointer" through the text, which discourages re-emitting the same item.
- Inference-time consistency: the generation config `eos_token_id` must match training; with vLLM `--tokenizer_mode mistral` vs `hf` the prompt formatting can differ from the one used in SFT. Use the exact same chat template at train and inference.

### Gaps
- Whether TRL 1.15.0's default `DataCollatorForLanguageModeling` masks pad by token ID or by padding position was not confirmed from source; verify empirically (check 1 above).
- Mistral v0.3 HF chat template specifics (whether `apply_chat_template` appends `</s>` after the final assistant turn in the current tokenizer_config) were not re-verified.

## (e) Output decomposition: per-object calls, JSONL, two-stage, chunk-and-merge

### Takeaway
Shorter outputs per call are the most robust structural defense against both loops and truncation for a 7B model: chunk the paper (sections/paragraphs), extract a small array per chunk with a low `maxItems`, then merge and dedupe. Two-stage (enumerate predictors, then fill slots) improves recall control but doubles calls. No source quantified recall/cost tradeoffs for this exact setting.

### Cited Findings
- Reports of constrained-output loops involved prompts with many similar items; recommended mitigation was to reduce near-identical items per prompt and split large arrays into chunks, plus post-validation and retry — [Google AI forum](https://discuss.ai.google.dev/t/structured-output-repetition-loop-inside-a-json-number-literal-runs-to-max-tokens-flash-vertex/175138)
- Together docs warn outputs truncate when `max_tokens` is too low or schemas too large (`finish_reason: "length"`) — [Together docs](https://docs.together.ai/docs/inference/chat/structured-outputs)
- instructor `create_iterable` streams and validates list elements individually — [instructor docs](https://python.useinstructor.com/)

### Inferences
- Options compared:
  - Single call, whole paper, full array: cheapest (one prompt) but longest output; highest loop/truncation risk; long-context precision drops for 7B.
  - Chunk-wise (per section/paragraph window) with `maxItems` ~5–8 per chunk: more prompt tokens overall (some overlap), short outputs, easy recovery (a bad chunk fails alone), natural parallelism; requires merge/dedupe on (`predictor`, `outcome`, `canonical_construct`) with fuzzy matching and keep-longest `evidence_text`. Best robustness/cost balance for a 7B model; also lets you skip irrelevant sections (references, appendix tables) to save cost.
  - JSONL (one object per line, stop on a sentinel or `stop=["\n\n"]`): degrades gracefully; a truncated last line is dropped, the rest survive. Can be enforced via a regex/grammar. Requires reformatting SFT targets.
  - One object per call (iterative "next item not in this list"): best per-item quality, cost scales with item count, prompt grows; prone to the model re-proposing listed items.
  - Two-stage (stage 1: list predictor-outcome pairs as short enum/strings; stage 2: per pair fill slots + evidence): stage 1 output is tiny and easy to constrain; stage 2 can be batched; ~2x calls; recall bounded by stage 1.
- Given the gold set (246 rows, frozen v1.3) is small, chunk-wise extraction matches the SFT signal best if gold rows carry location info; otherwise one must derive chunk-level targets by locating `evidence_text` in the chunk (string match), which also yields a free hallucination check.

### Gaps
- No published recall/cost benchmark comparing these decomposition strategies for scientific evidence extraction with 7B models was found.

## Decoding parameters for repetition

### Takeaway
Use greedy or low-temperature decoding with a hard `max_tokens` cap sized from the gold distribution, a mild `repetition_penalty` (~1.05–1.15) or `frequency_penalty`, and rely on grammar `maxItems` for termination. Avoid `no_repeat_ngram_size` on JSON (keys repeat legitimately). DRY is available in llama.cpp-family engines; I found no confirmation of DRY in vLLM.

### Cited Findings
- TGI's own guidance example combines `repetition_penalty: 1.3` with a JSON grammar — [TGI docs: Guidance](https://huggingface.co/docs/text-generation-inference/basic_tutorials/using_guidance.md)
- Temperature 0 did not prevent a schema-constrained repetition loop in Gemini reports — [Google AI forum](https://discuss.ai.google.dev/t/gemini-3-7-flash-schema-constrained-json-output-degenerates-into-repeated-0-until-maxoutputtokens-regression-vs-gemini-3-flash-preview/178681)
- DRY ("Don't Repeat Yourself") penalizes tokens that would extend a verbatim repeat of earlier text: penalty = `multiplier * base^(repeat_len - allowed_length)`; params `dry_multiplier` (0 = off), `dry_base` (default 1.75), `dry_allowed_length` (default 2), `dry_penalty_last_n`, `dry_sequence_breakers`; ~0.8 multiplier commonly suggested — [Aspose LLM docs: DRY](https://docs.aspose.com/llm/net/developer-reference/parameters/sampler/dry-multiplier/); [llama.cpp-style option docs](https://jsr.io/@oke/client/doc/~/LlamaCompletionOptions)
- A 2026 preprint "Don't Repeat Yourself: Stopping Verbatim Loops at ..." (arXiv 2608.22761, submitted 24 Aug 2026) exists; contents not reviewed — [alphaXiv](https://www.alphaxiv.org/abs/2608.22761)

### Inferences
- `repetition_penalty` (multiplicative on logits of all previously seen tokens) is risky for JSON with repeated keys and for `evidence_text` copied verbatim from the source (the source tokens are in the prompt, and vLLM/HF apply the penalty over prompt tokens as well); keep it mild (<=1.1) and evaluate on the dev set. `frequency_penalty`/`presence_penalty` in vLLM apply to generated tokens only, so they penalize repeated objects without penalizing copying from the prompt; a small `frequency_penalty` (0.1–0.3) is the better-targeted knob.
- `no_repeat_ngram_size` (HF transformers generate) will break JSON because key strings like `"canonical_construct": "` recur per object; do not use.
- DRY's `dry_sequence_breakers` could be set to JSON punctuation to avoid penalizing key repetition, but DRY is only available in llama.cpp/KoboldCpp-type backends per the sources found, which are not the deployment target.
- `max_tokens`: set to roughly (per-object token p95) x (maxItems) + ~20; with chunking this becomes small (e.g., 8 x 120 = ~1000), so runaway cost is bounded. Detect `finish_reason == "length"` and treat as failure (retry once or drop).
- Greedy decoding maximizes loop risk in principle for degenerate models; but for extraction, sampling adds variance. Recommended: greedy plus the grammar plus penalty; if loops persist, compare temperature 0.2–0.3 with top_p 0.9 on dev.

### Gaps
- vLLM 0.31 sampling params were not checked for DRY or other anti-loop samplers; search did not confirm either way.
- No quantitative study found on repetition_penalty effects on JSON extraction accuracy.

## Recommended layered strategy for this task

### Takeaway
Fix the root cause in training (EOS + no truncation + empty-array examples), then enforce structure at serving (vLLM on HF Inference Endpoints with llguidance/xgrammar `json` schema including `maxItems`, enums, `maxLength`), reduce output length per call (chunk-wise extraction + merge/dedupe), and use light validation/retry (Pydantic, optionally instructor) as the last layer. Each layer addresses a different failure mode; constrained decoding alone does not stop semantic loops.

### Cited Findings
- EOS masking with pad=eos is a documented cause of non-stopping Mistral fine-tunes — [HF forum](https://discuss.huggingface.co/t/mistral-trouble-when-fine-tuning-dont-set-pad-token-id-eos-token-id/77928)
- TRL default `max_length=1024` with `keep_start` truncation — [TRL docs](https://huggingface.co/docs/trl/sft_trainer)
- vLLM structured outputs + Mistral-7B-Instruct-v0.3 tool/structured support; TGI maintenance mode — [vLLM structured outputs](https://docs.vllm.ai/en/latest/features/structured_outputs.html); [vLLM tool calling](https://docs.vllm.ai/en/latest/features/tool_calling.html); [TGI GitHub](https://github.com/huggingface/text-generation-inference)
- Together serverless LoRA discontinued — [Together docs](https://docs.together.ai/docs/lora-inference)
- Constrained decoding does not hurt (and slightly helps) accuracy in JSONSchemaBench — [arXiv 2501.10868](https://arxiv.org/html/2501.10868v2)

### Inferences
Layer 0 — Diagnose (1 hour): collate a batch from the current training script; check EOS label not -100, decoded target ends with `]</s>`, and count examples whose prompt+completion exceed `max_length`. If either fails, the 51-object loop is explained.

Layer 1 — Training data/config (retrain LoRA):
```python
tok = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
tok.pad_token = tok.unk_token            # never pad with </s>
cfg = SFTConfig(
    max_length=None,                     # or > longest example; never truncate targets
    completion_only_loss=True,           # prompt-completion dataset
    packing=False,
    eos_token=tok.eos_token,             # "</s>", matches chat template end
)
# dataset rows: {"prompt": [{"role":"user","content": instr + chunk}],
#                "completion": [{"role":"assistant","content": json.dumps({"items": items}, ensure_ascii=False, separators=(",",":"))}]}
# include rows with {"items": []}; order items by position in text; fixed key order
```

Layer 2 — Serving with grammar (vLLM >= 0.12, current 0.31, on HF Inference Endpoint, single 24 GB GPU, scale-to-zero):
```python
schema = {
  "type":"object","additionalProperties":False,"required":["items"],
  "properties":{"items":{"type":"array","maxItems":8,
    "items":{"type":"object","additionalProperties":False,
      "required":["predictor","canonical_construct","outcome","effect_type","direction",
                  "plausible_values_handling","student_weights_used","evidence_text"],
      "properties":{
        "predictor":{"type":"string","maxLength":120},
        "canonical_construct":{"type":"string","maxLength":80},   # or enum of canonical list
        "outcome":{"type":"string","maxLength":80},
        "effect_type":{"enum":[...]},
        "direction":{"enum":["positive","negative","null","mixed"]},
        "plausible_values_handling":{"enum":[...]},
        "student_weights_used":{"enum":["yes","no","unclear"]},
        "evidence_text":{"type":"string","maxLength":600}}}}}}
resp = client.chat.completions.create(
    model="<merged-or-lora-name>", messages=msgs, temperature=0, max_tokens=1200,
    frequency_penalty=0.2,
    extra_body={"structured_outputs": {"json": schema}})
# server flag: --structured-outputs-config.backend guidance   (or xgrammar)
```
(Enum value lists to be filled from the project's frozen codebook; enum values above are placeholders.)

Layer 3 — Decomposition: chunk paper into sections/windows (~1–2k tokens, small overlap), run Layer 2 per chunk in parallel, merge, dedupe on normalized (predictor, outcome, canonical_construct), verify `evidence_text` appears in the chunk (fuzzy), drop otherwise.

Layer 4 — Validation and bounded retry: Pydantic model mirroring the schema plus semantic validators; on `finish_reason=="length"`, JSON error, or all items identical, retry once with temperature 0.3; else log as failure. instructor (`from_openai` against the vLLM endpoint, `max_retries=1`) is optional sugar for this.

Layer 5 — Evaluation hooks: track per-run rates of (a) length-truncation, (b) duplicate-item ratio, (c) empty outputs vs gold-empty, alongside the frozen evaluation protocol metrics, so each layer's effect is measurable.

Budget note: Together is only attractive if using a base model via serverless (no custom LoRA); for the fine-tuned adapter, an HF Inference Endpoint with vLLM and scale-to-zero, or a local/Colab vLLM run for batch extraction, is the minimal-cost path.

### Gaps
- No empirical numbers on how much each layer reduces loop rate for this specific model; must be measured on the project's dev split.
