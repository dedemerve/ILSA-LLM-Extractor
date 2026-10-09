# Evaluation Protocol v2: LLM Extraction for ILSA Evidence

**Version:** 2.0
**Date:** 2026-10-09
**Supersedes:** v1.0 sections 4 (metrics), 5.2 (downstream metrics) and 7 (LoRA design). Sections 2, 6.1 and 9 of v1.0 remain in force, including the rule that the test set is never used for prompt engineering, example selection or hyperparameter tuning.
**Status:** pre-registration. Freeze before any v2 arm is run on the test split.
**Author:** Merve Dede

## 1. Why a v2 protocol

The Stage 7 LoRA run and its Stage 9 evaluation were not a valid test of fine-tuning. The following were verified on 2026-10-09; the audit files are in `outputs/finetuning/v2/audit/`.

| Defect | Evidence | Consequence |
|---|---|---|
| The Stage 9 notebook loaded the adapter with `load_state_dict(..., strict=False)` into a fresh `get_peft_model` | Saved keys lack the `.default.` adapter name: 0 of 256 tensors matched (`preflight_stage7_train.json`, reproduced on a Mistral-architecture model) | The "LoRA" validation outputs are base-model outputs. F1 0.381 is not a LoRA result. |
| The Mistral v0.3 chat template renders the system turn only when the user turn is the last message | In all 29 training conversations the 35k-character system message (rules, codebook, demonstrations) was dropped | The model was trained without the codebook it was evaluated with. |
| `pad_token = eos_token` with the default LM collator | The collator sets the EOS label to -100 | The model never received a loss signal for stopping. This explains the 8,192-token repetition loop on `5.pdf`. |
| No completion-only loss | The gold JSON is 10.4% of training tokens | Validation loss (1.42 to 1.50) mostly measured how well the model predicted the source papers. |
| Different splits and matchers | BASE was scored on test (0.3667) and Stage 9 on validation. On the same validation outputs the construct-only matcher gives F1 0.379 and the exact-key matcher 0.00 | No valid BASE vs LoRA comparison exists. |

Corrections to the research report (`reports/Long context extraction fine tuning.md`):
- Truncation did not occur. The real token count of the training examples is at most 30,292 against `max_seq_length=32768`; the chars/4 estimate was wrong, but conservative.
- The best checkpoint was restored. `lora_adapter_final` is identical to `checkpoint-20`, the step with the lowest validation loss.
- The reported F1 of 0.381 is 0.379: 2*11/(34+24).
- The version pins in the report are not co-installable. vLLM 0.31.0 requires `transformers<5.18`, `xgrammar==0.2.7`, `llguidance<1.8` and `torch==2.13.0`. Unsloth 2026.10.3 requires `transformers<=5.17.0`. See `requirements-finetuning-v2.txt`.

## 2. Data

| | Studies | Gold rows | Deduplicated gold pairs |
|---|---|---|---|
| train | 30 | 187 | 142 |
| val | 6 | 24 | 22 |
| test | 6 | 35 | 29 |
| all | 42 | 246 | 193 |

- Gold: `gold_standard_full_corpus_v1.4.1.csv` (after the G0186/G0188 adjudication). Split: `gold_standard_split_v4.csv` (seed 2026, unchanged).
- **Deduplicated gold pairs.** 53 rows are identical to another row of the same study on (predictor, construct, outcome, direction, effect type) and differ only in record_id. No extraction can distinguish them, so the primary metrics use the deduplicated pairs. Row-level scores are reported as secondary (`--gold-rows`).
- **Training population.** 29 of the 30 train studies have source text. The Tang (2025) study has none, so training uses 29 studies, not 42. Validation and test studies are never trained on.
- **Label normalization.** Gold labels are mapped onto the codebook vocabularies by `scripts/finetuning/ilsa_schema.py`:
  - direction: 1/+ to positive, -1/- to negative, 0/empty to not_reported
  - effect type: predictive to E_predictive, associative/unspecified to E_assoc
  - PV handling: free-text suffixes removed
  - constructs outside codebook v1.3 (`UNMATCHED_TEACHER_ICT`, `Curriculum_Instruction`, `Assessment_Methodology`): UNMATCHED

## 3. Pre-registered arms

All arms use the same chunked pipeline (`extract_v2.py --mode chunked`):
- 6,000-character passages with 600-character overlap, reference list removed;
- one JSON-schema-constrained call per passage, plus one methods call per study for PV handling and student weights;
- merge with deduplication;
- items dropped unless their `evidence_text` is grounded in the passage (≥ 0.9 of word 5-gram shingles present).

Decoding: temperature 0, `frequency_penalty` 0.2, at most 8 items and 1,200 tokens per passage.

| Arm | Model | Training | Demonstrations |
|---|---|---|---|
| A0 `base_qwen3_chunked` | Qwen3-8B, non-thinking | none | 0 |
| A1 `qwen3_lora_selected` | Qwen3-8B, non-thinking | LoRA v2 (section 5) | 0 |
| A2 `base_mistral_chunked` | Mistral-7B-Instruct-v0.3 | none | 0 |
| A3 `mistral_lora_selected` | Mistral-7B-Instruct-v0.3 | LoRA v2 | 0 |
| P1 `zeroshot_gptoss120b` | openai/gpt-oss-120b (Groq, reasoning_effort low) | none | 0 |
| P2 `rag1_gptoss120b` | openai/gpt-oss-120b (Groq, reasoning_effort low) | none | 1 retrieved train passage |
| P3 `zeroshot_qwen27b` | qwen/qwen3.8-27b (Groq, reasoning off) | none | 0 |
| P4 `rag1_qwen27b` | qwen/qwen3.8-27b (Groq, reasoning off) | none | 1 retrieved train passage |
| P5 `rag2_mistral_large` (optional) | mistralai/mistral-large-4-0 (OpenRouter) | none | 2 retrieved train passages |

Demonstrations are retrieved by TF-IDF cosine similarity from positive train-split passages only.

P1 to P4 use open-weight models on Groq's free tier. That tier allows 8,000 tokens per minute, which caps the prompt at one demonstration and requires the `--min-interval` throttle. P5 needs OpenRouter credits: the account is on the free tier, and every call was refused with HTTP 402 on 2026-10-09. The Together and Gemini keys in `.env` were rejected with HTTP 401, and the Mistral API tier does not include Mistral Large.

Legacy arms, validation only: `base_mistral_doc` and `stage7_lora_doc`. These use the frozen v1 prompt over the whole document (75k characters), with the Stage 7 adapter loaded through `PeftModel.from_pretrained`.

## 4. Metrics

**Primary metric: strict pair F1.** A predicted pair matches a gold pair only if all of these agree:
- canonical construct,
- coarse outcome class (mathematics, reading, science or other),
- direction.

Matching is one-to-one and order-invariant, by Hungarian assignment, with predictor and outcome similarity as tie-breakers. For UNMATCHED pairs the predictor strings must additionally reach a similarity of at least 0.6, because UNMATCHED is a residual class (about 45% of gold rows).

Secondary metrics:
- **Lenient pair F1:** construct only, with the same UNMATCHED rule.
- **Joint slot F1** for direction, effect type, PV handling, student weights and outcome class. A slot counts only if its pair matched (lenient) and the slot is correct.
- **Per-stratum lenient P/R/F1** for HIGH, MID, LOW, SINGLE and UNMATCHED, using the strata of v1 section 4.4.
- **Macro strict F1** over studies.
- **Grounding rate:** the share of predicted `evidence_text` with at least 0.9 shingle coverage in the source.
- **Duplicate rate** and **validity**: ok, repaired, truncated or empty outputs. Truncated outputs count every recovered object as a prediction.
- **Legacy matchers** (construct-only greedy; exact construct, direction and effect type), for continuity with v1 numbers only.

**Uncertainty.**
- Bootstrap 95% CIs resample studies, not rows, with 10,000 draws and seed 2026.
- Paired comparisons report the per-study mean difference in F1 with a bootstrap CI and an exact sign-flip test. With 6 studies the smallest attainable two-sided p is 0.031, so the conclusions rest on effect sizes and CIs.

**Hypotheses.**
- H1: A1 > A0 on strict F1, i.e. LoRA helps an 8B model.
- H2: A1 compared with the best P arm, i.e. whether fine-tuning is worth it against prompting.
- Both outcomes are reportable, and neither is a failure of the study.

**Ceiling and context.**
- The oracle run (`extract_v2.py --backend oracle`) feeds gold-derived passage targets through the same merge. It gives a pipeline ceiling of strict F1 0.83 on train and 0.78 on validation. Recall is limited by gold pairs whose evidence the builder could not locate (100 of 142 located on train, 14 of 22 on validation).
- Human agreement is intra-rater only (κ = 0.871). It is reported as context, not as a ceiling. A second annotator on 8 to 10 studies is planned.

## 5. LoRA v2 design (replaces v1 section 7)

**Data.** `build_sft_v2.py` builds 107 training records from 29 studies:
- 55 passages with gold pairs;
- 23 passages without gold pairs (empty-output examples), drawn only from passages that mention no gold predictor;
- 29 methods examples, one per study.

Labels come from gold v1.4.1. `evidence_text` is re-cut to one verbatim source sentence per pair. Pairs that could not be located (42 on train: 38 listed as UNLOCATED, plus the 4 pairs of the study without source text) are excluded from training and listed in `evidence_review_train.csv` for human correction; corrections go into `evidence_overrides.csv`.

**Tokenization** is built explicitly in `train_lora_v2.py`:
- prompt tokens get label -100;
- every completion token, including the final EOS, is supervised;
- padding is never EOS and padding labels are -100;
- instructions travel in the user turn.

The gate asserts all of this before training.

**Hyperparameters:**
- r = 16, α = 32, dropout 0.05, all linear projections;
- 4-bit NF4 (QLoRA), LR in {1e-4, 2e-4} with cosine schedule and 5% warmup, 3 epochs;
- batch size 1, gradient accumulation 4, `max_length` 8192 (the longest example has 4,527 Mistral tokens).

rsLoRA, DoRA and NEFTune are off.

**Checkpoint selection:**
- One adapter is saved per epoch. Each is reloaded through `PeftModel.from_pretrained` and compared tensor by tensor.
- `select_checkpoint.py` picks the checkpoint by validation strict F1, then lenient F1, then the earlier checkpoint. Validation loss is not used.

## 6. Downstream evaluation (replaces v1 section 5.2)

`scripts/finetuning/downstream_eval.py`. The Stage 5 code and outputs are not modified; forecasts are re-run into a scratch directory.

**1. Crosswalk.** Codebook v1.3 constructs are mapped to Stage 5 W_j keys. 11 mappings are exact and 3 approximate (SELF_CONCEPT, TEACHING_QUALITY and SCHOOL_ICT_INFRASTRUCTURE). The other constructs have no W_j.

**2. Weights.** W is computed with the unchanged Stage 5 formula for:
- oracle (gold pairs of the split),
- each arm,
- the frozen Stage 5 weights.

In the primary mode (`augment`) the split's evidence is added to the Stage 5 literature corpus.

**3. ΔW against the oracle.** Reported as L1 distance, Spearman ρ and dominant-direction agreement.

**4. ΔMAE.** M1 forecasts for the PISA 2022 fold (66 countries × 3 domains) are compared with:
- a paired bootstrap CI of per-country absolute-error differences,
- a Wilcoxon signed-rank test,
- a TOST equivalence test with margin 1.0 PISA point.

The Stage 5 cross-sectional Diebold-Mariano p-values (`DM_M1vM2`) are not cited: they apply a lag-1 Newey-West term across countries, which have no time order.

**5. Noise dose-response.** Direction flips, pair drops and construct swaps are injected at rates of 0, 10, 25, 50 and 100%, with 100 replicates per level, into the evidence of all 42 gold studies (`standalone` mode).

**Allowed claims.**
- If ΔMAE is equivalent at the margin, the thesis states that fine-tuning (or prompting) changes extraction quality by X, and that the forecasting design absorbs extraction error of magnitude Y (from the noise curve).
- A null ΔMAE is not evidence against better extraction.

**Pre-existing observation (validation, `augment` mode).** The validation evidence maps to 6 Stage 5 effect rows. The oracle and the effective-base Stage 9 outputs differ by 0.004 MAE points. The propagation test therefore needs the noise curve and ΔW to be informative.

## 7. Test set

**v1 BASE run.** The v1 BASE run (Mistral-7B, frozen prompt, document mode) has already consumed the test set, with a legacy construct-only F1 of 0.3667. Its predictions file (`baseline_predictions_test.json`) is not in the repository. If it is recovered, it is re-scored with the v2 evaluator, which is legitimate because the matcher does not use the model. It is not re-run.

**v2 arms.** Each v2 arm is run on test exactly once, after this protocol is committed:
- `extract_v2.py --final-test-run` records the arm in `outputs/finetuning/v2/test_runs.json` and refuses a second run of the same arm;
- the oracle backend can never be run on test.

## 8. Known limitations

- **`evidence_text` in training targets** is aligned automatically: either the Stage 7 evidence relocated in the source, or a sentence found by predictor search. 67 of the 100 located train pairs came from predictor search and need human review (`evidence_review_train.csv`).
- **Outcome class** is assigned by keywords and is coarse.
- **UNMATCHED pairs** are matched by predictor string similarity. The 0.6 threshold is a design choice and is not validated.
- **Few evidence rows reach W_j.** Only constructs in the crosswalk affect W_j: 66 of the 193 gold pairs map to a Stage 5 predictor.
