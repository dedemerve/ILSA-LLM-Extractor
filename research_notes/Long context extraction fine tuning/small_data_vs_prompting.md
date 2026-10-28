# Small-data PEFT vs strong prompting for ILSA meta-analysis evidence extraction (N ~ 30 papers / 169 examples)

## LoRA hyperparameters for tiny data (rank, alpha, dropout, LR, epochs, target modules)

### Takeaway
At 169 examples, adapter capacity is not the bottleneck: even low-rank LoRA has far more parameters than the information in the data. The prior run's rising validation loss (1.42 to 1.50) is better explained by too many epochs, too high an effective LR, or truncation than by the choice of r=16. Recommended starting point: target all linear layers (MLP included), r=8 to 16 with alpha fixed (e.g. 16 to 32), an LR about 10x a full fine-tuning LR (~1e-4 to 2e-4), small batches, 1 to 3 epochs, and early stopping on a study-level held-out split.

### Cited Findings
- "LoRA Without Regret" (Schulman et al., Thinking Machines, Sep 2025): on small-to-medium SFT datasets, high-rank LoRA matches the learning curves and final loss of full fine-tuning (FullFT), with "the same sample efficiency." Low-rank adapters only fall behind once the dataset exceeds their capacity — [Thinking Machines](https://thinkingmachines.ai/blog/lora/)
- Same source: the optimal LoRA LR is about 10x the FullFT LR (fitted multiplier 9.8 across 14 Llama/Qwen models). For short runs (under ~100 steps) ~15x looked better, though the authors call this anecdotal. The optimal LR shifts by less than 2x between rank 4 and rank 512 — [Thinking Machines](https://thinkingmachines.ai/blog/lora/)
- Same source: with the standard alpha/r parametrization (alpha=32), the optimal LR is roughly independent of rank and early learning curves are nearly identical across ranks. The authors could not beat the standard PEFT setup (alpha=32, zero-init B, same LR for A and B) — [Thinking Machines](https://thinkingmachines.ai/blog/lora/)
- Same source: attention-only LoRA underperforms MLP-only LoRA, and adding attention on top of MLP adds nothing. On Llama-3.1-8B, attention-only rank 256 (0.25B params) lost to MLP-only rank 128 (0.24B). LoRA also tolerates large batches worse than FullFT, so small batches are preferred — [Thinking Machines](https://thinkingmachines.ai/blog/lora/)
- Same source: in RL, rank-1 LoRA matches FullFT because RL absorbs about 1 bit per episode, while rank-1 on Llama-3.1-8B already has ~3M parameters. This is the capacity-versus-information argument, extended here by analogy to tiny SFT — [Thinking Machines](https://thinkingmachines.ai/blog/lora/)
- "LoRA Learns Less and Forgets Less" (Biderman et al., 2024; ~100K instruction pairs and 20B continued-pretraining tokens, code and math): standard low-rank LoRA performs substantially worse than FullFT at that data scale, but preserves out-of-domain ability and generation diversity better. It mitigates forgetting more than weight decay or dropout do. FullFT learns perturbations of 10 to 100x higher rank than typical LoRA configs — [arXiv 2405.09673](https://arxiv.org/abs/2405.09673)
- HF PEFT: `target_modules="all-linear"` applies LoRA to every linear layer except the output head (the QLoRA-style setup said to match full fine-tuning). Otherwise the defaults are architecture-dependent and are often only the query/value projections. `lora_dropout` defaults to 0.0, and the docs' example uses 0.1 — [HF PEFT LoRA reference](https://huggingface.co/docs/peft/package_reference/lora)

### Inferences
- With 169 examples, the knobs that matter are the number of optimizer steps (epochs × examples / batch) and the LR, not the rank. Going from r=16 to r=4 or r=8 is unlikely to fix overfitting by itself. Reducing epochs and selecting the checkpoint by validation metric (not loss) is the higher-leverage change.
- Validation loss is a weak signal here. With ~20 to 30 validation examples, a move from 1.42 to 1.50 is within noise, and loss can rise while JSON field accuracy still improves. That happens because the model becomes overconfident on a few tokens while the extraction gets better. Select checkpoints by task metrics such as field-level F1 or exact match on the canonical construct.
- If the prior run used only q/v projections (a common default), switching to all-linear is the change best supported by the Thinking Machines evidence.
- Truncation matters more than any LoRA setting. If inputs of ~18k tokens were cut to a shorter `max_seq_length`, the model was trained to emit evidence pairs whose supporting text it never saw. That teaches hallucination and inflates the loss.

### Gaps
- None of the cited LoRA studies trains on fewer than a few thousand examples. No controlled source on rank versus overfitting at N < 200 was found, so the recommendations above are extrapolated.
- The QLoRA paper (Dettmers 2023) was not fetched in this pass, so its specific hyperparameters (r=64, alpha=16, dropout 0.1 for ≤13B models, LR 2e-4) are recalled from training data and should be verified before citing.

## RSLoRA and DoRA: what they change, small-data evidence, flags, costs

### Takeaway
Both are one-flag changes in HF PEFT, but neither is a regularizer. rsLoRA only matters when you push the rank high, which you should not do at N=169. DoRA may help at low ranks, but it costs training speed, is mostly validated on large benchmarks, and may not be supported for serverless adapter serving. Neither is likely to change the conclusion at this data size, so treat them as optional ablations rather than fixes.

### Cited Findings
- rsLoRA (Kalajdzievski, arXiv 2312.03732): standard LoRA scales adapters by a rank-dependent factor that slows learning at higher ranks. The paper proves adapters should be scaled by the square root of the rank, which lets larger ranks improve performance without extra inference cost. The abstract contains no experimental numbers — [arXiv 2312.03732](https://arxiv.org/abs/2312.03732)
- PEFT `use_rslora=True` changes the scale from `lora_alpha/r` to `lora_alpha/sqrt(r)`. The docs say this "stabilizes the adapters and increases the performance potential from using a higher r". It changes only a scalar, so it adds no compute or memory — [HF PEFT](https://huggingface.co/docs/peft/package_reference/lora)
- DoRA (Liu et al., ICML 2024 Oral, arXiv 2402.09353): splits each weight into magnitude and direction and applies LoRA to the direction. It "consistently outperforms LoRA" on LLaMA, LLaVA and VL-BART, across commonsense reasoning and vision-language tasks, with no added inference cost after merging. The abstract gives no small-data results — [arXiv 2402.09353](https://arxiv.org/abs/2402.09353)
- PEFT `use_dora=True`: "can improve the performance of LoRA especially at low ranks". It has "a bigger overhead than pure LoRA, so it is recommended to merge weights for inference". It supports only linear and Conv2D layers and does not work with multi-adapter batching — [HF PEFT](https://huggingface.co/docs/peft/package_reference/lora)

### Inferences
- Under rsLoRA, alpha=32 at r=16 gives a scale of 8 instead of 2, which is effectively a 4x higher LR. Switching rsLoRA on without re-tuning the LR could make the overfitting worse.
- For deployment, multi-LoRA serving such as Together's typically expects a plain LoRA adapter. If you use DoRA you may have to merge the weights and deploy a full model on a dedicated endpoint, which defeats the minimal-budget constraint. Check provider support before using DoRA.

### Gaps
- The DoRA paper's per-rank and per-data-size numbers were not retrieved, only the abstract. No evidence was found that rsLoRA or DoRA reduce overfitting at N<1000.

## Regularization and protocol for N≈30 documents: early stopping, NEFTune, weight decay, augmentation, leave-studies-out CV

### Takeaway
The most defensible protocol is K-fold or leave-studies-out cross-validation, because a single train/val split of ~30 papers gives very noisy estimates. Combine it with early stopping on a task metric and with chunk-level training examples, which also solve truncation. NEFTune is a cheap knob, but its evidence comes from open-ended chat quality, not structured extraction.

### Cited Findings
- NEFTune (Jain et al., arXiv 2310.05914) adds noise to the embedding vectors during training. On Alpaca, LLaMA-2-7B went from 29.79% to 64.69% on AlpacaEval, with roughly 8 to 10% gains on Evol-Instruct, ShareGPT and OpenPlatypus. These are judge-based chat-quality metrics — [arXiv 2310.05914](https://arxiv.org/abs/2310.05914)
- LoRA itself reduces forgetting more than weight decay or dropout do. This suggests that extra weight decay or dropout is a second-order lever in LoRA regimes — [arXiv 2405.09673](https://arxiv.org/abs/2405.09673)
- Dagdelen et al. (Nature Comms 2024): for the doping task, output was not properly structured below about 10 training samples (scores of zero). Structured output appeared sharply around 20 samples, gains then diminished, and above about 50 samples intermediate predictions needed little correction. Note that the inputs were short (sentences or abstracts), not full papers — [Dagdelen et al., PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC10869356/)

### Inferences
- Chunk-level examples (section or window, ~2 to 4k tokens, paired with the evidence pairs whose `evidence_text` falls in that chunk, plus empty-array negatives) multiply the example count. They also avoid 18k-token truncation, and they match how the 169 examples were presumably derived. The cost is that cross-section reasoning, such as linking a methods-section PV/weights statement to a results table, must be handled by a document-level field pass.
- Splits must be by study (paper), never by example. Examples from the same paper share wording, so a random split leaks information and inflates scores.
- Report the mean ± SD across folds, or bootstrap 95% CIs over papers. With ~30 papers, the CI on field-level F1 will plausibly be ±5 to 10 points. This is an inference that should be verified on the actual data.

### Gaps
- No study was found that tests NEFTune on JSON or structured extraction. No published evidence was found on leave-studies-out CV for LLM fine-tuning in systematic-review extraction specifically. It is standard ML practice, but uncited here.

## Fine-tuned small models vs few-shot/zero-shot large models for scientific / systematic-review extraction; how many examples before fine-tuning wins

### Takeaway
The evidence is thin and heterogeneous, and no controlled study gives a crossover point. The pattern is:
- **Narrow, short-input tasks:** fine-tuned 7 to 8B models can match GPT-4-class prompting with ~100 to several hundred examples (clinical report extraction, materials NER/RE).
- **Broad, inference-heavy, long-document extraction** (closest to this task): frontier models with good prompts dominate, and even they perform poorly on effect-size-level meta-analytic extraction.

With 169 examples from ~30 long papers, fine-tuning a 7B model beating a frontier few-shot baseline would be a surprising result, not the expected one.

### Cited Findings
- Dagdelen et al. (Nature Comms 2024): fine-tuned GPT-3 (davinci) and Llama-2-70B (8-bit + LoRA) on 413 sentences (doping), 507 abstracts (MOFs) and 634 abstracts (general materials). Doping host-dopant F1 was 0.726 (GPT-3) and 0.821 (Llama-2-70B). General-materials exact-match F1 was 0.33 to 0.54, and manual scoring was much higher (~0.82 to 0.83), showing that exact match understates performance. Human-in-the-loop annotation with an n=300 intermediate model cut annotation time per abstract by 57% — [PMC10869356](https://pmc.ncbi.nlm.nih.gov/articles/PMC10869356/)
- ChatExtract (Polak & Morgan, Nature Comms 2024; arXiv 2303.05352): a prompt-only, conversational pipeline (find data sentences, extract, then verify with follow-up questions) reached precision and recall both "close to 90%" with GPT-4 on materials data. It needs no training data, which makes it a strong no-fine-tuning baseline design — [arXiv 2303.05352](https://arxiv.org/abs/2303.05352)
- Clinical report extraction (Scientific Reports 2025, retrieved via search summary, not fetched): LoRA-tuned Llama-3.1-8B, trained on ≤100 reports per task on one desktop GPU, reached 90.0 ± 1.7% average exact-match accuracy. It was non-inferior to a second human annotator on all four datasets. GPT-4 (zero-shot) was non-inferior on all except kidney pathology — [PubMed 41286063](https://pubmed.ncbi.nlm.nih.gov/41286063/) (PubMed page blocked fetch; details from search snippet, verify before citing)
- LoRA Land (Predibase, 2024): 4-bit LoRA fine-tuned models beat their base models by ~34 points and GPT-4 by ~10 points on average across 31 tasks. The tasks are mostly narrow classification and generation, and the gains are smallest on broad, complex tasks. This is vendor-produced evidence — [arXiv 2405.00732](https://arxiv.org/abs/2405.00732v1)
- Rapid feasibility study (Evidence Synthesis Hackathon, 2024): GPT-4 prompt-only extraction for systematic reviews reached ~80% accuracy overall (82% human clinical, 80% animal, 72% human social sciences). Social-science studies, the closest domain to ILSA, were the hardest — [arXiv 2405.14445](https://arxiv.org/pdf/2405.14445)
- Network psychometrics metascience benchmark (Jyväskylä): Claude 3.5 Sonnet, Claude 3 Opus, GPT-4o and others reached 76 to 87% extraction accuracy, and 79.6 to 91.3% with newer models. Accuracy was higher for explicit, verbatim variables and lower for variables needing inference — [JYX](https://jyx.jyu.fi/jyx/Record/jyx_123456789_111061)
- Effect-size recovery against 28 published meta-analyses (UvA master's thesis presentation, 2026; not peer reviewed; model names as reported): the best condition reached precision .329, recall .116 and F1 .172 under a two-decimal agreement criterion. Numerical agreement was better when effects could be aligned, so the main failure was locating and aligning information — [UvA](https://student.uva.nl/en/events/2026-presentation-master-s-thesis-paulo-zirlis-psychological-methods)
- TU Delft (meta-analytic data from social robotics papers: means, SDs, Ns): Claude Opus 4.6 with Markdown input performed best among GPT-5.2, Claude and Gemini models — [TU Delft repository](https://repository.tudelft.nl/record/uuid:a9e011a8-c58c-4bf7-818f-c15de9811c78)
- Dental meta-analysis numeric extraction: Claude 3.5 Sonnet and DeepSeek R1 had higher accuracy and fewer omissions than Gemini 2.0 Flash and GPT-4o. Errors rose at higher aggregation levels — [UCM Docta](https://docta.ucm.es/entities/publication/aab87bf3-d044-4322-90e6-ac074d470128/full)
- Chemical text mining (Zhang et al., Chem. Sci. 2024, via search summary because the RSC page returned 403): fine-tuned GPT-3.5 excelled on all tasks, fine-tuned Mistral and Llama-3 were "competitive", and they beat prompted GPT-4 — [RSC D4SC00924J](https://pubs.rsc.org/en/Content/ArticleLanding/2024/SC/D4SC00924J)
- A materials-science study: GPT-4 relation extraction with only a couple of examples surpassed the baseline, but zero- and few-shot NER showed limited gains — [arXiv 2401.11052](https://arxiv.org/abs/2401.11052v1)

### Inferences
- The "fine-tuning wins" results share three features: short inputs (sentences, abstracts, single reports), a fixed schema with mostly verbatim spans, and 100 to 600+ examples. This task differs on all three. Inputs are ~18k-token full papers, `canonical_construct` mapping requires inference, and `plausible_values_handling` and `student_weights_used` require reading methods sections that sit far from results.
- In this literature, prompting typically wins on understanding long context and mapping to canonical concepts, while fine-tuning mainly buys format consistency. Format consistency can now be obtained via JSON-schema or structured-output modes on API models. Note that Dagdelen's "~20 samples to learn structure" finding predates structured-output APIs.
- A rough crossover heuristic from these studies is a few hundred examples on short-input narrow tasks. For long-document, inference-heavy extraction no crossover has been demonstrated. This is an inference, not a measured value.

### Gaps
- No head-to-head study was found of a fine-tuned 7 to 8B model versus few-shot frontier models on social-science or education full-text extraction.
- No learning-curve study was found that sweeps N against a fixed few-shot frontier baseline for scientific information extraction. SciER was not retrieved in this pass.

## Long-context few-shot / RAG prompting with API models: cost per paper (~18k input tokens)

### Takeaway
Prompting frontier or open-70B models costs cents per paper, so even a few hundred papers with multi-shot demonstrations stays well under $100. Fine-tuning a Llama-3.1-8B on Together also costs only about $4 to 10, so the real cost of fine-tuning is researcher time and evaluation burden, not dollars.

### Cited Findings
- Together serverless (per 1M tokens, input/output): Llama 3.3 70B $1.04/$1.04; Qwen3 235B A22B Instruct 2507 FP8 $0.20/$0.60; DeepSeek V4 Pro $1.32/$3.96; Qwen2.5 7B Turbo $0.30/$0.30 — [Together pricing](https://www.together.ai/pricing)
- Together fine-tuning (SFT, per 1M training tokens): Llama 3.1 8B $0.34 to $0.38, Llama 3.3 70B $2.03 to $2.24, minimum charge $4.00. Billing is training tokens × epochs plus evaluation tokens. Dedicated H100 endpoints cost $5.49/GPU-hour — [Together pricing](https://www.together.ai/pricing)
- Together serverless multi-LoRA (Dec 2024 announcement): uploaded adapters run on a compatible serverless base model (e.g. Llama 3.1, Qwen 2.5) at the base-model price. Current model support needs verification — [Together blog via plushcap](https://plushcap.com/content/together-ai/blog/together-ai-serverless-multi-lora-fine-tune-and-deploy-hundreds-of-adapters-for-model-customization-at-scale)
- Frontier API list prices (aggregators, inconsistent; verify on official pages): GPT-5 $1.25/$10; Gemini 2.5 Pro $1.25/$10 (≤200k tokens); Gemini 2.5 Flash $0.30/$2.50 (one source says $0.15 input); Claude Sonnet $3/$15 — [requesty](https://requesty.ai/models/compare/openai--gpt-5/google--gemini-2-5-flash), [trevorfox](https://trevorfox.com/tools/calculators/llm-cost/google-gemini/gemini-2.5-pro/), [anotherwrapper](https://anotherwrapper.com/tools/llm-pricing/gemini-25-flash/claude-3-5-sonnet)

### Inferences (own arithmetic from the prices above; assumes 18k input and ~2k output tokens per paper)
- Zero-shot cost per paper:

  | Model | Cost per paper |
  |---|---|
  | Gemini 2.5 Flash | ~$0.01 |
  | Llama 3.3 70B (Together) | ~$0.02 |
  | GPT-5 | ~$0.04 |
  | Claude Sonnet | ~$0.08 |

- Few-shot with 2 full-paper demonstrations (~54k input) costs about $0.06 (Llama 70B), $0.09 (GPT-5) or $0.19 (Sonnet) per paper. Prompt caching of the fixed demonstration prefix would cut this further. For 300 papers, the total is about $3 to $60.
- RAG-augmented prompting is the cheaper and stronger middle path. Retrieve the sections relevant to each field (methods for PV and weights, results tables for effects), and retrieve 1 to 3 gold-annotated examples from the 169-example set as demonstrations, chosen by embedding similarity. This uses the gold data without training on it, and the evaluation can still be run leave-studies-out.
- Fine-tuning Llama-3.1-8B on Together: 169 examples × ~20k tokens × 3 epochs ≈ 10M tokens ≈ $3.5 to $4, which hits the $4 minimum. A 5-fold CV run costs about $20.

### Gaps
- Official OpenAI, Anthropic and Google pricing pages were not fetched, and aggregator figures conflict. Model lineups changed in 2026 (GPT-5.x, Claude Opus 4.6 / Sonnet 5, Gemini 3.x), so re-check prices before writing the thesis.
- HF Inference Providers pricing for LoRA adapters was not retrieved.

## Teacher-distillation route (silver labels from a large model, then fine-tune a small model)

### Takeaway
This route is viable and well precedented, but its ceiling is roughly the teacher's quality. Students occasionally slightly beat the teacher on narrow entity types, but usually trail by a little. Its value is cheaper, reproducible open-weight inference at scale, not higher accuracy. For a thesis corpus of hundreds (not millions) of papers, the cost savings are negligible.

### Cited Findings
- Clinical NER distillation (arXiv 2501.00031): GPT-4o, o1-mini and Gemini teachers produced silver labels for BERT students. Students beat the teacher on disease (0.84 vs 0.82) and medication (0.87 vs 0.84) but trailed on symptoms (0.68 vs 0.73), while being up to 101x cheaper and 12x faster — [arXiv 2501.00031](https://arxiv.org/abs/2501.00031v1)
- Distilling step-by-step (Google, ACL 2023): a 770M T5 trained with LLM rationales outperformed 540B PaLM on a benchmark using 80% of the data — [Google Research](https://research.google/pubs/distilling-step-by-step-outperforming-larger-language-models-with-less-training-data-and-smaller-model-sizes/)
- Docusign production case: the fine-tuned student trailed its teacher by ~1 F1 point at ~1/100 the cost and 1/4 the latency (industry blog, teacher not named) — [Docusign](https://www.docusign.com/blog/how-a-small-model-learned-to-do-a-large-models-job-at-docusigns-scale)
- SHIELD (2026 de-identification preprint): a DeBERTa student trailed its Gemini 2.5 Flash teacher on span-level recall in nearly every category — [arXiv 2605.03301](https://arxiv.org/pdf/2605.03301)
- Dagdelen's human-in-the-loop variant (model pre-annotates, human corrects) cut annotation time by 57% at n=300. This is a practical way to grow the gold set beyond 30 papers — [PMC10869356](https://pmc.ncbi.nlm.nih.gov/articles/PMC10869356/)

### Inferences
- If distillation is pursued, silver labels must be evaluated against the frozen gold set first. Teacher errors on `canonical_construct` would be learned systematically by the student.
- A thesis-friendly use of distillation is to use the teacher to pre-annotate new papers, have a human verify them (growing gold or silver data cheaply), and treat any fine-tuned student as a reproducibility and open-weights contribution rather than an accuracy contribution.

### Gaps
- No distillation study was found for long-document, multi-field scientific evidence extraction or for social-science corpora.

## Honest bottom line: is fine-tuning with N=30 papers defensible in a thesis, and how to frame it

### Takeaway
Fine-tuning with N=30 papers is defensible only as a pre-registered comparison arm, not as the main method. It should be evaluated with leave-studies-out CV against strong prompting baselines, with bootstrap CIs over papers. The expected and publishable result is that few-shot or RAG prompting of a frontier or 70B model matches or beats the 7B LoRA model, with the fine-tuned model possibly better on format adherence and cost. The ROI of spending further effort on hyperparameter tuning (rsLoRA, DoRA, NEFTune) is low. The ROI of a strong prompting pipeline plus a growing, human-verified gold set is high.

### Cited Findings
- Fine-tuning wins are concentrated in narrow, verbatim, short-input tasks with 100 to 600+ examples — [PMC10869356](https://pmc.ncbi.nlm.nih.gov/articles/PMC10869356/); [arXiv 2405.00732](https://arxiv.org/abs/2405.00732v1); [PubMed 41286063](https://pubmed.ncbi.nlm.nih.gov/41286063/)
- Frontier-model prompting already reaches ~76 to 91% accuracy on metascience variable extraction, but drops sharply on inference-heavy variables and on effect-size alignment — [JYX](https://jyx.jyu.fi/jyx/Record/jyx_123456789_111061); [UvA](https://student.uva.nl/en/events/2026-presentation-master-s-thesis-paulo-zirlis-psychological-methods)
- Social-science studies were the hardest domain for GPT-4 extraction (72% vs 80 to 82%) — [arXiv 2405.14445](https://arxiv.org/pdf/2405.14445)
- Reproducibility threats in LLM-based reviews include inconsistent model versions and under-reported prompts (Felizardo et al. 2025, via search summary) — [arXiv 2602.10147](https://arxiv.org/html/2602.10147v1)

### Inferences (recommended framing)
- Run 4 to 5 arms on identical leave-studies-out folds:
  1. Zero-shot with JSON schema (frontier API)
  2. Few-shot with retrieved gold demonstrations (frontier API, plus Llama-3.3-70B or Qwen3 on Together)
  3. Zero- or few-shot base 7 to 8B (to isolate the effect of fine-tuning)
  4. LoRA 7 to 8B on chunk-level examples
  5. Optional: a ChatExtract-style verify pass on the best arm
- Report field-level precision, recall and F1, plus exact match for `direction`, `effect_type` and `canonical_construct`, with paper-level bootstrap 95% CIs and paired tests between arms. Report the cost per paper and an inter-annotator reliability ceiling, which aligns with the planned 2026-10-12 reliability step.
- Arm 3 vs arm 4 is the scientifically clean question ("does LoRA on 169 examples help a 7B model?"), and the answer is likely yes for format and canonical labels. Arm 4 vs arm 2 is the practical question ("is it worth it vs prompting?"), and the answer is likely no at N=30. Both are legitimate thesis findings, and a negative or neutral result framed this way is defensible.
- Pre-specify that the gold set is frozen, that test folds are untouched during prompt or hyperparameter iteration, and that model versions and prompts are logged. This answers the reproducibility criticism.
- Pragmatic deployment recommendation: the primary pipeline should be RAG plus few-shot prompting of a frontier model (or Llama-3.3-70B or Qwen3 on Together for an open-weight option), with human verification. If the LoRA 7B model is competitive, it serves as a cheap open-weight secondary arm, deployable via Together serverless multi-LoRA at base-model price.

### Gaps
- No published study gives a sample-size threshold for this exact task type. The claim that the result is "likely no at N=30" is a reasoned extrapolation and should be stated as a hypothesis to test, not as an established fact.
