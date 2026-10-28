# Fine-tuning v2 pipeline

Implements the action plan in `reports/Long context extraction fine tuning.md`. The pre-registered design is in `outputs/finetuning/EVALUATION_PROTOCOL_v2.md`.

| Script | Step | Runs on |
|---|---|---|
| `ilsa_schema.py` | shared vocabularies, label normalization, Pydantic models, JSON schemas | imported |
| `chunking.py` | shared source cleaning, passage chunking, prompt rendering (training = inference) | imported |
| `preflight_checks.py` | 1: real token lengths, dropped system turns, EOS label masking, adapter key names | CPU |
| `eval_extraction.py` | 3: strict/lenient Hungarian pair F1, joint slot F1, grounding, study bootstrap, sign-flip test | CPU |
| `extract_v2.py` | 2, 4, 7: chunked constrained extraction (vLLM or OpenAI-compatible API), merge, test lock; `--backend oracle` gives the pipeline ceiling | GPU / API |
| `build_sft_v2.py` | 5: chunk-level SFT data from gold v1.4.1 (train and val only), evidence review CSV | CPU |
| `train_lora_v2.py` | 6: QLoRA with explicit masks, assertion gate, per-epoch adapters, reload verification | GPU |
| `select_checkpoint.py` | 6: checkpoint selection by validation strict F1 | CPU |
| `downstream_eval.py` | 8: crosswalk, W_j, ΔW, paired per-country tests with TOST, noise dose-response | CPU |

The GPU steps run from `notebooks/stage10_finetuning_v2_colab.ipynb`. Version pins are in `requirements-finetuning-v2.txt`: vLLM and the training stack need separate runtimes.

Tests: `python -m pytest tests/finetuning`.

The training set is built from the 29 train studies that have source text, out of the 42 gold v1.4.1 studies. Validation (6) and test (6) studies are never used for training. The test split runs once per arm, only with `--final-test-run`.
