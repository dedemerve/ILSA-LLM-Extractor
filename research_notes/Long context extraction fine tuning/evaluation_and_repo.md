# Evaluation methodology for LLM evidence extraction feeding a forecasting model, and the state of the ILSA_LLMs repo

Scope: notes for a defensible PhD evaluation design. Repo inspected read-only on 2026-10-09: main, the `experiment/ilsa-domain-finetuning` and `fix/stage5-dual-source-build-xy` branches via `git show`, plus the Colab notebooks and stage outputs in `~/Downloads`. No repo file was modified.

Notation: "REPO" marks a fact read directly from local files. Paths are given instead of URLs. Web facts carry URLs.

## Q0. What the repository actually does now (grounding facts)

### Takeaway
The evaluation code, the W_j construction and the LoRA training setup all exist, but they differ from the assignment brief in several load-bearing ways:
- The gold standard is v1.4.1 with 42 studies, not v1.3 with about 30.
- "LOCO" in the code means leave-one-**cycle**-out with an expanding window, not leave-one-country-out.
- The LoRA run used `max_seq_length=32768`, `pad_token = eos_token` and full-sequence loss.
- The existing Stage 5 outputs already show that literature weights barely move MAE. On main the effect is exactly zero.

### Cited Findings (REPO)
- **Gold standard version.** v1.3 is superseded. The current frozen gold is `outputs/finetuning/data/gold_standard_full_corpus_v1.4.csv`, then v1.4.1 after the targeted adjudication of the G0186/G0188 direction (commit f75ff34). It has 246 rows from 42 studies. The split is 30 train / 6 val / 6 test studies (seed 2026), with 187 / 24 / 35 rows. The training JSONL holds 29 train examples, because one train study was dropped (see memory note and `split_manifest_v4.json`). Source: user memory `project_finetuning_protocol.md`; branch `experiment/ilsa-domain-finetuning`, file `outputs/finetuning/EVALUATION_PROTOCOL_v1.md`.
- **Reliability.** This is a second pass by the same annotator (intra-rater, not inter-annotator). Macro kappa is 0.871 against a threshold of 0.61, recorded in `outputs/finetuning/audit/intra_rater_reliability_v4.json`.
- **Evaluation protocol** (`outputs/finetuning/EVALUATION_PROTOCOL_v1.md`, frozen 2026-10-06):
  - Field-level P/R/F1 and exact match for program, cycle, canonical_construct, direction, effect_type and plausible_values_handling.
  - Construct confusion matrix, UNMATCHED rate and hallucination rate.
  - Evidence attribution. Verbatim or paraphrase accuracy is to be scored by a human on a subset.
  - Per-stratum F1 by construct frequency (HIGH / MID / LOW / SINGLE).
  - Results at both row level and study level.
  - Downstream: ΔW_j, ΔMAE, ΔPISA coverage and construct alignment between the BASE and FINETUNED conditions, with Stage 5 W_j and LOCO folds held frozen.
  - It states: "If ΔMAE ≈ 0 despite improved extraction F1, the fine-tuning effect is annotation-quality only".
  - No significance test, no confidence interval and no LLM judge are specified.
- **The baseline matcher is construct-only and greedy.** In `scripts/baseline_inference.py` (branch experiment), `match_pairs()` loops over gold rows and takes the first unused prediction with the same `canonical_construct`. Matched pairs count as TP. Fields are then scored only on matched pairs: direction, effect_type, PV handling, student weights, and whether evidence_text is present.
  - Pair F1 ignores outcome, direction and effect_type entirely.
  - A wrong-direction pair still counts as a TP.
  - Gold order decides which duplicate gets matched.
- **The Stage 9 LoRA validation notebook uses a different matcher.** `~/Downloads/stage9_lora_val_inference.ipynb` matches on an exact composite key `(canonical_construct, direction, effect_type)`.
  - On the same 6 validation studies (24 gold pairs, 34 predicted) it gives TP=0, F1=0.0 (`stage9_outputs*/lora_results_val.json`).
  - Re-scored with BASE construct-only matching it gives TP=11, P=0.324, R=0.458, F1=0.381 (`lora_results_val_final.json`).
  - Direction accuracy on matched pairs is 0.091, effect_type 0.545 and PV handling 0.364.
  - So **the same predictions score F1 = 0.00 or 0.38 depending only on the matching rule**. This is the strongest local argument for fixing and pre-registering the matching rule.
- **BASE and LoRA were compared on different splits.** `lora_results_val_final.json` says "Different splits — directional only": BASE F1 0.3667 on **test**, LoRA F1 0.381 on **val**. Direction match is 0.364 for BASE and 0.091 for LoRA.
  - The BASE run therefore already consumed the test set, consistent with protocol §6.2.
  - A valid LoRA-vs-BASE comparison needs both on the same split. For example, run BASE on val, or LoRA on test exactly once.
- **Stage 9 parse failure.** One validation study (`5.pdf`) went into a repetition loop: 8192 output tokens, a repetition rate of 0.98, and the JSON parse failed. It was scored as zero predictions.
  - The notebook records "model never outputs mixed or not_reported — 2048 truncation artifact" because `max_new_tokens` dropped from 8192 to 2048 or 4096 between cells.
  - Generation settings were not constant across runs: 8192 in `run_inference`, 2048 in `run_inference_fixed`, and 4096 in `stage9_execution_audit.json`.

### Cited Findings: fine-tuning configuration actually used (REPO, `~/Downloads/stage7_colab_training*.ipynb`, `~/Downloads/stage7_reproducibility_manifest.json`)
- **Model and adapter.**
  - Base: `mistralai/Mistral-7B-Instruct-v0.3`, QLoRA 4-bit nf4 with double quantization, bf16 compute.
  - LoRA r=16, alpha=32, dropout 0.05, targets q/k/v/o_proj.
  - This differs from protocol §7, which provisionally said r=8 on q/v only.
- **Optimisation.** 100 steps, batch 1 × grad accumulation 4, learning rate 1e-4 cosine, warmup 0.05, seed 42.
  - That is about 12.6 epochs over 29 examples (manifest `epoch: 12.55`).
  - `load_best_model_at_end=True` on eval_loss, with `eval_steps=20` and `save_steps=50`.
- **Libraries.** transformers 4.47.0, peft 0.14.0, trl 0.13.0, on an A100 40GB.
- **`max_seq_length=32768`**, set through `SFTConfig`. The manifest records truncation as "controlled_32k — gold JSON fully preserved".
  - The only pre-check counts characters divided by 4 (`est = sum(len(content))//4`), not real token counts.
  - Training inputs have a median of about 100.9k characters and a maximum of 111.6k (my count of `outputs/finetuning/data/train_finetune.jsonl`).
  - At a realistic 3–3.5 chars/token for academic English, the longest examples could exceed 32,768 tokens. Right-side truncation by SFTTrainer would then cut off the assistant JSON and EOS. This is unverified, and real tokenizer counts are needed.
- **Padding and EOS.** `tokenizer.pad_token = tokenizer.eos_token`, with `padding_side='right'` for training and `'left'` for generation.
  - No completion-only or assistant-only collator is configured. `SFTTrainer` uses its default collator with `dataset_text_field=None` and conversational `messages`.
  - So the loss covers the whole sequence, including the roughly 100k-character source text.
  - Assistant (gold JSON) characters are only **6.5%** of training characters (3.5% in val). The reported train loss of about 0.88 and val loss of about 1.50 therefore mostly measure language modelling of the source papers, not extraction.
- **Overfitting signal.** Eval loss rose monotonically: 1.419 at step 20, 1.433 at step 40, 1.473 at step 60, 1.504 at the end (`stage7_training_log.json`). The best checkpoint by eval loss was step 20.
  - But `save_steps=50` is not a multiple of `eval_steps=20`, so it is unclear whether the step-20 state was ever saved or restored.
  - transformers normally rejects this combination when `load_best_model_at_end=True`. Verify which notebook version actually produced the adapter.
- **Train/inference mismatch.**
  - Stage 9 inference truncates the source to 75,000 characters (`SOURCE_TEXT_MAX_CHARS`), but training saw inputs of about 100k characters.
  - Inference merges the system prompt into the user turn by hand, while training used a `system` role through the chat template.
  - Inference loads the base model in bf16 without 4-bit quantization, while training used QLoRA nf4.
- **Possible silent adapter-loading failure (high priority to verify).** Stage 9 builds a fresh `get_peft_model(base_model, lora_config)` and then calls `model.load_state_dict(load_file('adapter_model.safetensors'), strict=False)`.
  - PEFT-saved keys normally omit the adapter name (for example `...q_proj.lora_A.weight`), while the live model's keys include `.default.`.
  - With `strict=False`, mismatched keys are skipped silently. The B matrices would stay at their zero initialisation, so the "LoRA" model would behave as the base model.
  - The notebook never prints the missing or unexpected keys. Use `PeftModel.from_pretrained(base, ADAPTER_PATH)` instead, or assert that the key list returned by `load_state_dict` is empty.
- **Earlier template.** `outputs/finetuning/audit/training_configuration.json` is an older "TEMPLATE — not yet finalized" (max_tokens 4096, seed 42). It does not reflect the run that was executed.

### Cited Findings: W_j derivation and forecasting (REPO)
- **W_j formula.** `scripts/compute_predictor_weights_v2.py` computes w_j = F_j × D_j × C_j × E_j, where:
  - F_j is the quality-weighted study count (pv_correct=False gives ×0.5, weight_correct=False gives ×0.7, plus a quality_flag factor);
  - D_j is the number of distinct (program, domain) pairs;
  - C_j is the share of the dominant direction (direction consistency);
  - E_j is the min-max-normalised median |effect| (√R² for R²), set to 0.5 when no effect sizes exist.
  - The result is then **min-max normalised**, so the lowest predictor always gets w=0.
  - Only rows with achievement outcomes are used.
- **Direction and PV fields feed W_j directly.** Direction enters through C_j, and PV/weight handling enters through F_j. Errors in these slots therefore propagate to W_j. Construct errors propagate through the grouping key; the matcher currently ignores outcome.
- **Vocabulary mismatch.** W_j keys come from an older vocabulary (BELONGING, SCHOOL_CLIMATE, ICT_ACCESS, SELF_EFFICACY, HOME_RESOURCES, …). The extraction codebook v1.3 uses SCHOOL_BELONGING, DISCIPLINARY_CLIMATE, ICT_HOME_ACCESS, SELF_CONCEPT, …. A crosswalk is needed before extracted pairs can feed W_j.
- **Current weights** (`outputs/stage5/predictor_weights_v2.csv`, main) are extremely skewed: ESCS 1.00, HOMEPOS 0.15, SELF_EFFICACY 0.11, GENDER 0.089, … DISCLIMA 0.0017, TEACHER_QUALITY 0.
- **How W_j enters the forecast.** In `scripts/run_loco_forecasting.py`, M1 standardises the features, multiplies each `lag_` covariate column by √w_j, and fits `RidgeCV` with alphas [0.01 … 1000] chosen by CV.
  - Score-backbone (lag score) features get weight 1.0.
  - Covariates with no weight key default to 0.5 (`_feat_weight`).
  - M0 is the same model with all weights equal to 1.
- **"LOCO" is leave-one-cycle-out.** The docstring reads "LOCO (Leave-One-Cycle-Out): her cycle c için, c'den önceki tüm cycle'lar train seti" ("for each cycle c, all cycles before c form the training set"). `validation_design_summary.csv` (branch fix/stage5-dual-source-build-xy) gives PISA as "expanding_window_LOCO, 1 held-out fold (2022 as test)", n_test = 90 countries.
- **Null propagation is already visible.**
  - On main, `outputs/stage5/loco_results.csv` and `loco_predictions.csv` have **y_M0 == y_M1 exactly** (max |diff| = 0.0 over all 24 folds; n_features = 8 per fold). In that run no weighted `lag_` covariate survived the 50% fill-rate filter, or none affected the fit, so literature weights had zero effect.
  - On branch fix/stage5-dual-source-build-xy, `loco_m0_m1_benchmark.csv` gives M1 − M0 MAE of +0.003 (PISA math), −0.30 (PISA reading), +0.10 (PISA science), +0.19 (TIMSS math), −0.10 (PIRLS), and so on. That is |ΔMAE| ≤ 0.30 score points against MAEs of 10–28.
  - **Switching literature weighting fully on or off moves MAE by at most about 0.3 points.** The BASE-vs-FINETUNED difference in W_j will be a much smaller perturbation.
- **The current DM test is mis-specified for this design.** `diebold_mariano()` in `run_loco_forecasting.py` uses squared-error differences with a lag-1 Newey-West autocovariance across rows that are **countries** (cross-sectional, with no time order), and a t(n−1) reference.
  - The lag-1 term depends on arbitrary row order. For cross-sectional paired errors a plain paired t / Wilcoxon / bootstrap is the appropriate analogue.
  - The MAE headline uses absolute loss while DM uses squared loss, so the two can disagree.
- **Sensitivity infrastructure exists.** `scripts/compute_lit_weight_sensitivity.py` and `outputs/stage5/perm_lit_weight.csv` (branch fix/...) already compare three weight specifications (w_F, w_FD, w_FDC) and run 1,000-permutation tests of ρ_lit. With only 2–3 weighted features per domain, the permutation p-values are 0.3–1.0, and n=2 is flagged "trivial".

### Inferences
- In Ridge, scaling standardised feature j by s_j = √w_j is algebraically the same as a generalised ridge penalty α/w_j on the original coefficient. Two consequences follow:
  - A uniform rescaling of all W is fully absorbed when RidgeCV re-selects α. Only **relative** weights among the few surviving covariates matter.
  - Score-lag backbone features (weight 1.0) dominate PISA predictions, so covariate weights act only at the margin.
- Both points predict ΔMAE ≈ 0 between BASE and FINETUNED extraction. The thesis must be designed so that a null downstream result is informative, not fatal.
- The 6.5% target-token share, the full-sequence loss and pad=EOS together plausibly explain the observed repetition loop (no learned stop) and the weak direction accuracy. Any extraction comparison is confounded by these training defects until they are fixed. Not verified by rerun.

### Gaps
- I could not confirm which notebook version (32 KB vs 162 KB) produced `lora_adapter_final`. The 162 KB copy shows a `triton.ops` import error at the 4-bit load cell, yet the manifest says COMPLETED.
- I could not confirm the actual Mistral token lengths of the training examples (no tokenizer run).
- I did not check whether key names in `adapter_model.safetensors` match the live model's state dict. Inspect with `safetensors.safe_open(...).keys()`.
- The PISA 2022 fold for the W-sensitive model is a single cross-section of about 90 countries. No ΔMAE confidence interval exists in the repo.

## Q1. Pair-level F1 vs slot-level F1 vs LLM-as-judge vs downstream MAE

### Takeaway
Use a hierarchy of complementary metrics:
1. **Tuple detection** (pair P/R/F1 with a fixed composite-key matching rule), reported strict and lenient.
2. **Slot accuracy conditional on a match**, per field and per stratum.
3. **Fuzzy grounding of evidence_text.**
4. **Downstream ΔW_j and ΔMAE.**

No single number is defensible. Document-level n-ary IE benchmarks (SciREX), document-processing metrics (ANLS*) and systematic-review extraction studies all report several granularities.

### Cited Findings
- SciREX is a document-level IE benchmark on full scientific papers that combines salient-entity identification with document-level N-ary relation identification. Annotation needs whole-document understanding because relations span sections. The authors report a large gap between human performance and baselines — [SciREX, ACL 2020](https://aclanthology.org/2020.acl-main.670); [arXiv 2005.00512](https://arxiv.org/pdf/2005.00512).
- ANLS* is a metric for generative LLM document extraction:
  - It scores dictionaries key by key.
  - It scores **lists with Hungarian matching** on pairwise similarity, so order does not matter.
  - It penalises both missing and hallucinated elements.
  - Strings use normalised Levenshtein similarity with threshold τ=0.5. Type mismatches score 0.
  - It is available as the `anls_star` package — [ANLS*, arXiv 2402.03848](https://arxiv.org/html/2402.03848v10); [PyPI anls-star](https://pypi.org/project/anls-star).
- The VAREX structured-extraction benchmark uses order-invariant Hungarian matching of predicted array elements to gold elements by maximum field overlap. Exact match is its primary metric and ANLS gives partial credit — [VAREX, arXiv 2603.15118](https://arxiv.org/pdf/2603.15118).
- Systematic-review extraction studies report concordance and accuracy per data item, not a single tuple F1:
  - A two-LLM living-review workflow (GPT-4-turbo + Claude-3-Opus) found 87% concordant responses with accuracy 0.94. Discordant responses had accuracy 0.41–0.50, and cross-critique raised it to 0.76 — [medRxiv 2024.09.20.24314108](https://www.medrxiv.org/content/10.1101/2024.09.20.24314108.full.pdf).
  - A study within six reviews found 77.2% human–AI concordance, AI-assisted accuracy of 91.0%, recall 89.4% and precision 98.9%. Missed items were the most common error, and error classification "required subjective judgment" — [medRxiv 2025.03.20.25324350](https://www.medrxiv.org/content/10.1101/2025.03.20.25324350.full.pdf).
- Locally (REPO): construct-only greedy matching gives LoRA val F1 0.381, while an exact (construct, direction, effect_type) key gives 0.000 on the same outputs (see Q0).

### Inferences
- **Pair F1, construct-matched.**
  - Strength: simple, and it tracks the unit that feeds W_j.
  - Weaknesses: it is blind to direction and outcome, and greedy order-dependence lets a pair with the right construct but wrong direction count as correct, even though direction is what C_j consumes.
- **Slot-level accuracy on matched pairs.**
  - Strength: diagnostic. It shows the direction failure (0.09).
  - Weakness: it is conditional on matching, so a model that matches fewer pairs can look better. Always pair it with recall, or report "joint" slot F1, where a slot counts as TP only if both the pair matched and the slot is correct.
- **LLM-as-judge.** Useful only for free-text slots: `predictor`, `outcome`, `evidence_text`, and paraphrase equivalence of the outcome description. It should not judge the controlled-vocabulary slots, which can be checked exactly.
- **Downstream MAE.** This is the thesis-relevant extrinsic test, but it is low-powered (Q4). It should be framed as "does extraction error matter for forecasting", tested by sensitivity analysis, not as the sole success criterion.
- **Recommended primary metric set (pre-register before touching the test set):**
  1. Strict pair F1: the key is (construct, normalised outcome class, direction), matched by Hungarian assignment.
  2. Lenient pair F1: construct only. Kept for continuity with the BASE result already recorded.
  3. Joint slot F1 per field (direction, effect_type, PV handling, student weights).
  4. Evidence grounding rate: fuzzy match of evidence_text against the source text.
  5. Schema validity and parse-failure rate.
  6. Per-stratum (HIGH/MID/LOW/SINGLE) and per-study macro averages.

### Gaps
- I found no published IE benchmark that uses this exact ILSA schema. The SciREX scoring details (cluster-matching thresholds) were not fetched in full.

## Q2. Matching strategy: collisions, composite keys, Hungarian matching

### Takeaway
Construct-only greedy matching is order-dependent and collides whenever a study reports the same construct for several outcomes, directions or subgroups. That is common in ILSA papers: math, reading and science, or effects for several countries.

Replace it with optimal (Hungarian) assignment over a similarity matrix built from a composite key, report strict and lenient variants, and break ties deterministically.

### Cited Findings
- ANLS* and VAREX both use Hungarian matching so that list order does not matter and duplicates and hallucinations are penalised correctly — [ANLS*](https://arxiv.org/html/2402.03848v10); [VAREX](https://arxiv.org/pdf/2603.15118).
- REPO: `match_pairs()` takes the "first unused predicted pair with matching construct", so with k gold rows sharing a construct the pairing depends on list order. Field accuracies, especially direction, then depend on which duplicate was paired.
- REPO: the Stage 9 exact composite-key matcher is too strict: TP=0 when direction accuracy is about 9%. Exact-key matching conflates detection with slot accuracy.

### Inferences
- **Recommended algorithm (per study):**
  1. Build a cost matrix between gold and predicted pairs, setting cost = 1 − sim.
     - sim = 0 unless canonical_construct matches. Construct is a hard gate because it is the W_j grouping key.
     - Otherwise sim = 0.5·[outcome match] + 0.3·[direction match] + 0.2·[effect_type match].
     - Outcome match should use a normalised outcome class (math / reading / science / other), optionally plus string similarity on the raw outcome.
  2. Run `scipy.optimize.linear_sum_assignment`.
  3. Accept assignments with sim ≥ a threshold.
- **Strict vs lenient.** "Strict" requires construct, outcome and direction all to agree. "Lenient" requires construct only, solved by Hungarian with outcome as a tie-breaker.
- Then score slots on the matched pairs and report how many matches were forced versus exact.
- Pre-register the weights and threshold, and report a sensitivity table across 2–3 settings. The 0.00 vs 0.38 example shows how much this matters.
- Evaluate UNMATCHED rows separately (detection-only), since they do not feed W_j.

### Gaps
- No literature-validated weighting for a composite similarity exists for this schema. The weights above are a design choice to be justified and sensitivity-tested.

## Q3. LLM-as-judge reliability and how to validate a judge

### Takeaway
LLM judges have documented biases and variable agreement with humans across tasks. Use a judge only for free-text slots, and validate it against human labels on a held-out sample before trusting it. Report chance-corrected agreement and score-level divergence, not raw percent agreement.

### Cited Findings
- JUDGE-BENCH (20 NLP datasets with human annotations, 11 LLMs) found substantial variance across models and datasets. Reliability depends on the property evaluated, annotator expertise, and whether the text is human- or model-generated. The authors recommend careful validation against human judgments, and raise reproducibility concerns for proprietary judges — [Bavaresco et al., ACL 2025](https://aclanthology.org/2025.acl-short.20); [arXiv 2406.18403](https://arxiv.org/abs/2406.18403v2).
- Thakur et al. tested 13 judges on TriviaQA:
  - Only the largest models aligned reasonably with humans, and even they differed by up to 5 points from human scores.
  - High percent agreement can hide very different scores, so alignment metrics beyond percent agreement are needed.
  - Judges were sensitive to prompt complexity and biased toward leniency.
  - Source: [Judging the Judges, arXiv 2406.12624](https://arxiv.org/html/2406.12624).
- A systematic study of position bias across 15 judges and more than 150,000 instances introduced repetition stability, position consistency and preference fairness metrics — [Shi et al., arXiv 2406.07791](https://arxiv.org/pdf/2406.07791v8.pdf).
- In systematic-review extraction, agreement between two LLMs predicted accuracy: concordant answers were 0.94 accurate and discordant ones 0.41–0.50 — [medRxiv 2024](https://www.medrxiv.org/content/10.1101/2024.09.20.24314108.full.pdf).

### Inferences
- **Judge validation protocol:**
  - Sample about 100–150 (gold, predicted) pairs from train and val only, never test.
  - The annotator labels each one (correct / partially correct / wrong) for `outcome` equivalence and for `evidence_text` support (supports the extracted direction and construct?).
  - Report Cohen's κ (or Gwet's AC1 when labels are skewed) between judge and human, with a bootstrap CI, alongside the human intra-rater κ (0.871) as a reference.
  - Run position and order swaps and a repeat run to measure stability.
  - Use a judge from a different model family than the extractor (Mistral) to avoid self-preference.
  - Freeze the judge prompt and model version.
- **Deterministic evidence grounding first.** Make the grounding check deterministic: normalised fuzzy substring match of evidence_text against the source text (for example a token-level partial ratio ≥ 0.9). Use the judge only for the entailment question: "does this sentence support this direction?"

### Gaps
- I found no study that validates LLM judges specifically for education or social-science effect-direction extraction.

## Q4. Downstream metric: power, statistical tests, sensitivity analysis, null propagation

### Takeaway
ΔMAE alone will almost certainly not detect the BASE-vs-FINETUNED difference:
- PISA has one held-out cycle (2022) of about 90 countries.
- Literature weights enter only through √w_j scaling of a few covariates inside RidgeCV.
- Switching the weights entirely on or off already changes MAE by at most 0.3 points.

Design the extrinsic evaluation as:
1. ΔW_j against gold-derived W_j (the "oracle extraction" condition).
2. Paired per-country loss differences with a bootstrap CI and an equivalence (TOST) margin.
3. A perturbation and sensitivity curve showing how much W_j error MAE can absorb.

### Cited Findings
- The HLN-modified Diebold-Mariano test (Harvey, Leybourne & Newbold 1997, IJF 13(2):281–291) is the standard small-sample correction for comparing forecast accuracy. It is implemented in R `forecast::dm.test` — [R docs](https://search.r-project.org/CRAN/refmans/forecast/html/dm.test.html).
- Harvey, Leybourne & Zu (2024) show that standard DM can have asymptotic size zero and reduced power when the mean loss differential varies over time — [Nottingham repository](https://nottingham-repository.worktribe.com/output/40565499/testing-for-equal-average-forecast-accuracy-in-possibly-unstable-environments).
- Dror et al. (ACL 2018) give a protocol for choosing significance tests in NLP and found that tests were often missing or misapplied in 2017 ACL/TACL papers — [ACL P18-1128](https://aclanthology.org/P18-1128/).
- Berg-Kirkpatrick, Burkett & Klein (EMNLP 2012) study paired tests, including the paired bootstrap, and how significance depends on gain size, test-set size and system similarity — [ACL D12-1091](https://aclanthology.org/D12-1091/).
- REPO evidence of a small effect:
  - M1 − M0 MAE ranges from −0.30 to +0.25 across program-domains (`loco_m0_m1_benchmark.csv`), with exactly zero on main's current outputs.
  - Permutation tests of literature-weight alignment have p = 0.3–1.0 with 2–3 features (`perm_lit_weight.csv`).

### Inferences
- **Unit of analysis and resampling.**
  - Extraction metrics: bootstrap over **studies** (6 test studies, clustered), not over rows. Rows within a study are dependent, as protocol §4.5 already notes. With 6 studies, also report per-study values and an exact sign/permutation test over the 6 paired study differences. Note that with 6 pairs the smallest two-sided sign-flip p-value is 2/64 ≈ 0.031.
  - Forecasting: per-country absolute error differences d_c = |e_FT,c| − |e_BASE,c|. Use a paired bootstrap CI of the mean d and a Wilcoxon signed-rank test. Do not use lag-1 HAC across countries (the current `diebold_mariano()` implementation), because country order is arbitrary. Keep the HLN-DM test for the TIMSS multi-cycle folds if those are ordered in time.
  - Because the extraction sample is small, the uncertainty that matters comes from the studies. A two-level bootstrap is the honest version: resample studies, rebuild W_j, rerun the forecast, and record ΔMAE. It propagates extraction uncertainty into the MAE interval.
- **Equivalence framing.** Pre-register a smallest effect size of interest, for example 1 PISA point of MAE, or a share of the M2 − M1 gap. Test with TOST. A result of "ΔMAE within ±δ with 90% CI" is a publishable, defensible conclusion when the difference is null.
- **Sensitivity and dose-response analysis.** This is the key experiment for the propagation claim.
  - (a) Compute W_j from gold extractions ("oracle"), from BASE, and from FINETUNED.
  - (b) Inject controlled noise into the gold evidence: flip a share p of directions, drop a share q of pairs, swap constructs at rate r. Recompute W_j and MAE, with about 200 replicates per level.
  - (c) Plot ΔMAE against perturbation level and ΔW_j (L1 or Spearman of W) against perturbation.
  - This gives the minimum extraction error that produces a detectable ΔMAE. Report where BASE and FINETUNED fall on that curve.
  - If even 50% direction flips leave MAE unchanged, the forecasting model is insensitive to W_j. The extrinsic null then reflects the downstream architecture, not the extraction quality.
- **Mechanisms that cause null propagation:**
  - Ridge's CV α absorbs global W rescaling.
  - The min-max normalisation of W makes one predictor (ESCS) dominate, so W_j for the other predictors is near zero whatever the extraction.
  - Unknown keys default to 0.5.
  - The fill-rate filter drops covariates.
  - Lag-score features at weight 1.0 dominate.
  - The vocabulary crosswalk gap means codebook v1.3 constructs may not map to W_j keys at all.
  - Report these as reasons, and include a "W-sensitive" variant (for example fixed α, or weights used as feature selection) only as a pre-registered secondary analysis. Never choose it after seeing the results.
- **Intermediate metrics to report:**
  - ΔW_j: L1 distance and Spearman rank correlation to the oracle W.
  - Coverage of the forecasting features.
  - Sign agreement of C_j.
  - These are far more sensitive than MAE and form the causal chain from extraction to weights to forecast.

### Gaps
- I found no published work that measures the propagation of LLM extraction error into a downstream forecasting or meta-analytic model with a power analysis. The design above is a synthesis, not an established protocol.
- I did not run the perturbation experiment, so the slope is unknown.

## Q5. Inter-annotator agreement as a ceiling for extraction F1

### Takeaway
Report human agreement as context, not as a hard ceiling. The "IAA is an upper bound" assumption is contested empirically. The repo has only **intra**-rater reliability (one annotator, two passes, κ=0.871), which overstates true inter-annotator agreement.

### Cited Findings
- Boguslav & Cohen (MEDINFO 2017) trace the IAA-as-upper-bound idea to its logical-positivist roots and present biomedical NLP data suggesting IAA is not an upper bound on system performance — [IOS Press PDF](https://ebooks.iospress.nl/pdf/doi/10.3233/978-1-61499-830-3-298).
- Richie, Grover & Tsui (BioNLP 2022) report that no evidence supports the upper-bound claim. Their simulations show a model can exceed IAA when annotators are noisy, provided the model is well specified — [ACL Anthology 2022.bionlp-1.26](https://preview.aclanthology.org/setup/2022.bionlp-1.26).
- In systematic reviews, human extraction is itself error-prone. Reported error rates reach up to 63% of included studies having at least one extraction error, as cited in the Gartlehner-group work — [medRxiv 2025](https://www.medrxiv.org/content/10.1101/2025.03.20.25324350.full.pdf) (secondary citation; the original figure was not checked).
- REPO: the reliability measure is intra-rater κ (`intra_rater_reliability_v4.json`). No second human annotator exists.

### Inferences
- **Second annotator.** Add a second annotator on a stratified subset (for example 8–10 studies, about 60 rows, from train/val) and compute:
  - inter-annotator pair F1 under the **same** Hungarian matcher;
  - per-slot κ (direction, effect_type, PV handling, construct).
  - This gives a like-for-like human reference for the model's F1. Without it, the thesis can only say "the model reaches X; the same annotator reproduces themselves at κ=0.87".
- Report model F1 as a fraction of human-human pair F1 per slot, and discuss slots where disagreement is inherent, such as "mixed" vs "not_reported" direction.

### Gaps
- No inter-annotator data exists in the repo. I did not read the full Boguslav & Cohen text, only the abstract and citing papers.

## Q6. Recommended defensible evaluation design (synthesis)

### Takeaway
Pre-register a two-tier design and report both tiers:
- **Intrinsic tier:** strict and lenient Hungarian pair F1, joint slot F1, grounding and schema validity, with study-cluster bootstrap CIs and per-stratum and per-study tables.
- **Extrinsic tier:** ΔW_j against the oracle W, ΔMAE with a paired per-country bootstrap CI and TOST equivalence, and a noise-injection dose-response curve.

Fix the training and inference confounds before making any BASE-vs-LoRA claim.

### Cited Findings
- Multi-granularity reporting is standard in document IE and structured extraction — [SciREX](https://aclanthology.org/2020.acl-main.670); [ANLS*](https://arxiv.org/html/2402.03848v10); [VAREX](https://arxiv.org/pdf/2603.15118).
- Judges must be validated against humans — [JUDGE-BENCH](https://aclanthology.org/2025.acl-short.20).
- Use appropriate paired significance tests — [Dror et al. 2018](https://aclanthology.org/P18-1128/); [HLN-DM in R](https://search.r-project.org/CRAN/refmans/forecast/html/dm.test.html).
- Pad=EOS with a collator that masks pad positions removes EOS from the loss, a common cause of fine-tuned models that never stop generating (community reports; verify against the installed TRL version) — [HF forum](https://discuss.huggingface.co/t/gpt2-finetuned-with-eos-token-will-never-yield-eos-token-during-generation/15437); [Mistral SFT EOS report](https://clay-atlas.com/us/blog/2024/01/01/mistral-sft-trainer-cannot-generate-eos-token/). The TRL docs state that padding positions are set to the ignore index −100 — [TRL SFTTrainer docs](https://huggingface.co/docs/trl/main/en/sft_trainer).

### Inferences
Concrete checklist for the thesis, in priority order:
1. **Fix comparability first.**
   - Evaluate BASE and LoRA on the same split with identical generation settings: the same `max_new_tokens` (8192 for both), the same truncation (75k characters or real-token budgeting for both), and the same prompt formatting (system role).
   - Load the adapter with `PeftModel.from_pretrained` and assert that no keys are missing.
2. **Fix training defects before drawing conclusions.**
   - Use a distinct pad token, not EOS.
   - Use completion-only loss (assistant tokens only).
   - Count real tokens and confirm no example exceeds `max_seq_length`, and that the gold JSON plus EOS survive.
   - Select the checkpoint by val extraction F1 or val loss with aligned `save_steps` and `eval_steps`.
   - Consider fewer epochs, since val loss rose from step 20.
   - These defects can explain both the repetition loop and the 9% direction accuracy.
3. **Pre-register the matcher** (Hungarian, composite key, strict and lenient) and re-score the existing BASE test predictions with it. This is legitimate because the matcher does not use the model.
4. **Run the intrinsic tier** with a study-level bootstrap of 10,000 resamples over the 6 test studies, plus the exact sign-flip test and per-study tables.
5. **Build a vocabulary crosswalk** from codebook v1.3 to the W_j feature keys, so extracted evidence can feed W_j.
6. **Run the extrinsic tier:**
   - oracle (gold) W vs BASE W vs FT W;
   - ΔW (L1, Spearman);
   - ΔMAE on the PISA 2022 fold with a paired per-country bootstrap and TOST;
   - a two-level bootstrap (studies → W → forecast);
   - the noise-injection dose-response curve.
   - Fix the DM implementation for cross-sectional errors.
7. **Report the extrinsic null honestly** if it occurs, together with the sensitivity curve showing the downstream model's insensitivity. Keep the claim narrower: "fine-tuning improves extraction; forecasting is robust to extraction errors of magnitude X".
8. **Human reference:** add second-annotator agreement on a subset, and validate the judge (κ against human) before using it for outcome and evidence_text.

### Gaps
- The notes cannot determine the true size of the BASE-vs-FT extraction gap until items 1–2 are fixed and both models are rerun on the same split.
