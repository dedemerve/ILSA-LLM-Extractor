# Colab A100 Instructions — Fine-tuning v2 Steps 2–5

**Prerequisites:** Google Colab with A100-40GB runtime.  
Mount your Drive or upload the repo zip. All paths below assume the repo root is the working directory.

---

## 0. Setup

```bash
pip install -r requirements-finetuning-v2.txt
```

Set `REPO_ROOT` in your shell or notebook to the repo root path.

---

## Step 2 — Memory probe (run this first)

```bash
python3 scripts/finetuning/train_lora_v2.py \
    --model Qwen/Qwen3-8B \
    --out outputs/finetuning/v2/runs/qwen3_8b_r16 \
    --epochs 3 --r 16 --batch-size 1 \
    --memory-probe
```

Expected output: `{"max_tokens": <N>, "peak_gb": <X>}` with peak_gb < 38.

**If peak_gb > 38:** retry with `--r 8` or add `--grad-accum 8`.

---

## Step 2 — LoRA training

```bash
python3 scripts/finetuning/train_lora_v2.py \
    --model Qwen/Qwen3-8B \
    --out outputs/finetuning/v2/runs/qwen3_8b_r16 \
    --epochs 3 --r 16 --batch-size 1 --grad-accum 4
```

The gate assertions run automatically before training. If any assertion fails, **stop and report to Merve**.  
Expected: 3 checkpoint directories saved under `outputs/finetuning/v2/runs/qwen3_8b_r16/`.  
`run_manifest.json` is written at the end. `adapter_reload_verified: true` must be present.

---

## Step 3 — Select best checkpoint by val extraction F1

First, run extract_v2.py for **each saved epoch checkpoint** so select_checkpoint.py has scores to compare. For a 3-epoch run there will be checkpoint-N, checkpoint-2N, checkpoint-3N directories. Run once per checkpoint (replace `<STEP>` with the actual step number):

```bash
python3 scripts/finetuning/extract_v2.py \
    --mode chunked --backend vllm \
    --model Qwen/Qwen3-8B \
    --adapter outputs/finetuning/v2/runs/qwen3_8b_r16/checkpoint-<STEP> \
    --split val --arm qwen3_8b_r16_checkpoint-<STEP>
```

Then select:

```bash
python3 scripts/finetuning/select_checkpoint.py --split val
```

Writes `outputs/finetuning/v2/eval/checkpoint_selection_val.json`.  
The selected adapter path is in `selected.adapter` in that file.  
**Do NOT use the checkpoint with the lowest validation loss.**  
**Do NOT use the checkpoint with the lowest validation loss.**

---

## Step 4 — Extract with A0 (base) and A1 (LoRA)

```bash
# A0: base Qwen3-8B, no adapter
python3 scripts/finetuning/extract_v2.py \
    --mode chunked --backend vllm \
    --model Qwen/Qwen3-8B \
    --split val --arm A0_base_qwen3_chunked

# A1: LoRA-adapted Qwen3-8B (best checkpoint)
python3 scripts/finetuning/extract_v2.py \
    --mode chunked --backend vllm \
    --model Qwen/Qwen3-8B \
    --adapter outputs/finetuning/v2/runs/qwen3_8b_r16/best_checkpoint \
    --split val --arm A1_qwen3_lora_selected
```

Both write prediction files to `outputs/finetuning/v2/predictions/`.

---

## Step 5 — Evaluate A0 and A1

```bash
python3 scripts/finetuning/eval_extraction.py \
    --pred outputs/finetuning/v2/predictions/A0_base_qwen3_chunked_val.json \
    --split val --label A0 \
    --out outputs/finetuning/v2/eval/A0_base_qwen3_val.json

python3 scripts/finetuning/eval_extraction.py \
    --pred outputs/finetuning/v2/predictions/A1_qwen3_lora_selected_val.json \
    --split val --label A1 \
    --out outputs/finetuning/v2/eval/A1_qwen3_lora_val.json
```

---

## Step 6 — Update results_summary.md

After Steps 4–5 complete, add A0 and A1 rows to  
`outputs/finetuning/v2/results_summary.md` with their strict/lenient/legacy_construct F1 values.

---

## Hard constraints (never violate)

1. **Never run `--split test`** unless using `--final-test-run` AND the arm is not in `test_runs.json`.
2. **Never modify** `gold_standard_full_corpus_v1.4.1.csv` or the split file.
3. **Never commit** without explicit instruction from Merve Dede.
4. The Stage 7 adapter at `outputs/finetuning/lora_adapter_final/` is **INVALID** — do not load it.
5. **Do not run** `scripts/run_unified_backtest.py` — that is a separate step.

---

## Fallback: VRAM insufficient

If Qwen3-8B still exceeds VRAM at `--r 8`: switch to `mistralai/Mistral-7B-Instruct-v0.3` (arm A2/A3).  
The Mistral system-turn workaround is already built into `train_lora_v2.py`.
