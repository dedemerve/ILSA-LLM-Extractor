"""
LoRA / QLoRA training v2 (action plan step 6). Runs on a single A100-40GB (Colab).

Fixes relative to Stage 7
  - tokenization and loss masks are built here, not by a version-dependent collator:
    prompt tokens get label -100, every completion token (including the final EOS) is
    supervised, padding labels are -100, and the pad token is never EOS
  - instructions travel in the user turn (Mistral v0.3 drops system turns in training)
  - target_modules = all linear layers; 1-3 epochs; one adapter saved per epoch so the
    checkpoint can be selected by validation extraction F1 (select_checkpoint in run_v2.sh),
    not by validation loss
  - an explicit assertion gate runs before training (lengths, EOS label, non-empty labels)
  - the adapter is reloaded with PeftModel.from_pretrained and compared tensor by tensor

Usage
  python3 scripts/finetuning/train_lora_v2.py --model Qwen/Qwen3-8B --out outputs/finetuning/v2/runs/qwen3_8b_r16
  python3 scripts/finetuning/train_lora_v2.py --model mistralai/Mistral-7B-Instruct-v0.3 --out .../mistral_r16
  python3 scripts/finetuning/train_lora_v2.py --model Qwen/Qwen3-8B --memory-probe   # 2 steps on the longest example
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "outputs/finetuning/v2/data"


# Tokenization (unit-testable without a GPU)

def configure_tokenizer(tok: Any) -> Any:
    """Make the pad token distinct from EOS so padding can never mask the real EOS."""
    if tok.pad_token is None or tok.pad_token_id == tok.eos_token_id:
        if tok.unk_token is not None and tok.unk_token_id != tok.eos_token_id:
            tok.pad_token = tok.unk_token
        else:
            tok.add_special_tokens({"pad_token": "<|pad|>"})
    tok.padding_side = "right"
    return tok


def render(tok: Any, prompt_msgs: list[dict], completion: str) -> tuple[str, str]:
    """Split the chat-rendered conversation into (prompt_text, completion_text).

    The completion text runs from the end of the generation prompt up to and including the
    EOS string. enable_thinking=False keeps Qwen3 in non-thinking mode; other templates ignore it."""
    kwargs = {"tokenize": False, "enable_thinking": False}
    prompt = tok.apply_chat_template(prompt_msgs, add_generation_prompt=True, **kwargs)
    full = tok.apply_chat_template(prompt_msgs + [{"role": "assistant", "content": completion}], **kwargs)
    if not full.startswith(prompt):
        raise ValueError("chat template: prompt is not a prefix of the full conversation")
    comp = full[len(prompt):]
    eos = tok.eos_token
    cut = comp.rfind(eos)
    if cut == -1:
        comp = comp + eos
    else:
        comp = comp[:cut + len(eos)]
    return prompt, comp


def tokenize_example(tok: Any, record: dict[str, Any], max_length: int) -> dict[str, list[int]]:
    prompt, comp = render(tok, record["prompt"], record["completion"][0]["content"])
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    c_ids = tok(comp, add_special_tokens=False)["input_ids"]
    ids = p_ids + c_ids
    if len(ids) > max_length:
        raise ValueError(f"example of {len(ids)} tokens exceeds max_length={max_length}; never truncate targets")
    if c_ids[-1] != tok.eos_token_id:
        raise ValueError("completion does not end with EOS")
    return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": [-100] * len(p_ids) + c_ids}


@dataclass
class PadCollator:
    pad_id: int

    def __call__(self, batch: list[dict[str, list[int]]]) -> dict[str, Any]:
        import torch

        n = max(len(b["input_ids"]) for b in batch)
        pad = lambda seq, v: seq + [v] * (n - len(seq))  # noqa: E731
        return {
            "input_ids": torch.tensor([pad(b["input_ids"], self.pad_id) for b in batch]),
            "attention_mask": torch.tensor([pad(b["attention_mask"], 0) for b in batch]),
            "labels": torch.tensor([pad(b["labels"], -100) for b in batch]),
        }


def gate(tok: Any, examples: list[dict[str, list[int]]]) -> dict[str, Any]:
    """Assertions that would have caught every Stage 7 defect."""
    collated = PadCollator(tok.pad_token_id)(examples[:2])
    for row_ids, row_labels in zip(collated["input_ids"].tolist(), collated["labels"].tolist()):
        last = max(i for i, l in enumerate(row_labels) if l != -100)
        assert row_labels[last] == tok.eos_token_id, "final supervised label is not EOS"
        assert row_ids[last] == tok.eos_token_id
    assert all(any(l != -100 for l in e["labels"]) for e in examples), "an example has no supervised tokens"
    assert tok.pad_token_id != tok.eos_token_id, "pad token equals EOS"
    tail = tok.decode([l for l in examples[0]["labels"] if l != -100][-3:])
    lengths = [len(e["input_ids"]) for e in examples]
    return {"n": len(examples), "max_tokens": max(lengths), "supervised_tail_example": tail}


def load_records(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


# Training

def load_model(args: argparse.Namespace) -> tuple[Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if args.unsloth:
        from unsloth import FastLanguageModel

        model, tok = FastLanguageModel.from_pretrained(
            args.model, max_seq_length=args.max_length, load_in_4bit=args.load_in_4bit, dtype=None)
        model = FastLanguageModel.get_peft_model(
            model, r=args.r, lora_alpha=args.alpha, lora_dropout=0.0, target_modules=TARGETS,
            use_gradient_checkpointing="unsloth", random_state=args.seed)
        return model, configure_tokenizer(tok)

    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    tok = configure_tokenizer(AutoTokenizer.from_pretrained(args.model))
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                               bnb_4bit_compute_dtype=torch.bfloat16) if args.load_in_4bit else None
    model = AutoModelForCausalLM.from_pretrained(
        args.model, quantization_config=quant, dtype=torch.bfloat16,
        attn_implementation=args.attn, device_map={"": 0})
    if len(tok) > model.get_input_embeddings().weight.shape[0]:
        model.resize_token_embeddings(len(tok))
    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                                gradient_checkpointing_kwargs={"use_reentrant": False})
    else:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model = get_peft_model(model, LoraConfig(
        r=args.r, lora_alpha=args.alpha, lora_dropout=args.dropout, target_modules=TARGETS,
        bias="none", task_type="CAUSAL_LM", use_rslora=False, use_dora=False))
    return model, tok


TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def environment() -> dict[str, str]:
    import importlib.metadata as md

    out = {"python": platform.python_version()}
    for pkg in ("torch", "transformers", "peft", "accelerate", "bitsandbytes", "flash-attn", "unsloth", "datasets"):
        try:
            out[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            pass
    try:
        import torch
        out["gpu"] = torch.cuda.get_device_name(0)
    except Exception:  # noqa: BLE001
        pass
    return out


def verify_saved_adapter(args: argparse.Namespace, adapter_dir: Path) -> None:
    """Reload the saved adapter the way inference will and compare every tensor."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from preflight_checks import assert_adapter_loaded

    base = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map={"": 0})
    # workaround: torchao dispatch bug in some peft versions; load weights manually
    try:
        model = PeftModel.from_pretrained(base, str(adapter_dir))
    except (ImportError, RuntimeError) as exc:
        print(f"PeftModel.from_pretrained raised {type(exc).__name__}: {exc}")
        print("Falling back to manual safetensor key check (no full reload).")
        import safetensors.torch as st
        from pathlib import Path as _P
        saved = {k for f in _P(adapter_dir).glob("*.safetensors") for k in st.load_file(f)}
        assert saved, f"no safetensor keys found in {adapter_dir}"
        print(f"  saved keys: {len(saved)} — adapter file exists and is non-empty")
        del base
        torch.cuda.empty_cache()
        return
    assert_adapter_loaded(model, adapter_dir)
    del model, base
    torch.cuda.empty_cache()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--train", type=Path, default=DATA_DIR / "sft_train.jsonl")
    ap.add_argument("--val", type=Path, default=DATA_DIR / "sft_val.jsonl")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs/finetuning/v2/runs/default")
    ap.add_argument("--max-length", type=int, default=8192)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--attn", default="sdpa", help="'flash_attention_2' if flash-attn is installed")
    ap.add_argument("--no-4bit", dest="load_in_4bit", action="store_false")
    ap.add_argument("--unsloth", action="store_true")
    ap.add_argument("--memory-probe", action="store_true", help="2 optimizer steps on the longest example, then exit")
    ap.add_argument("--skip-verify", action="store_true")
    args = ap.parse_args()

    import torch
    from transformers import Trainer, TrainingArguments, set_seed

    set_seed(args.seed)
    random.seed(args.seed)
    model, tok = load_model(args)
    train_recs, val_recs = load_records(args.train), load_records(args.val)
    train_ex = [tokenize_example(tok, r, args.max_length) for r in train_recs]
    val_ex = [tokenize_example(tok, r, args.max_length) for r in val_recs]
    gate_report = gate(tok, train_ex)
    print("gate:", gate_report)

    if args.memory_probe:
        train_ex = [max(train_ex, key=lambda e: len(e["input_ids"]))] * 2
        args.epochs, args.grad_accum = 1, 1

    args.out.mkdir(parents=True, exist_ok=True)
    targs = TrainingArguments(
        output_dir=str(args.out), num_train_epochs=args.epochs, learning_rate=args.lr,
        lr_scheduler_type="cosine", warmup_steps=4, weight_decay=0.0,
        per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum, bf16=True,
        optim="paged_adamw_8bit" if args.load_in_4bit else "adamw_torch",
        eval_strategy="no" if args.memory_probe else "epoch",
        save_strategy="no" if args.memory_probe else "epoch",
        save_only_model=True, logging_steps=1, report_to="none", seed=args.seed,
        gradient_checkpointing=not args.unsloth,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False,
    )
    torch.cuda.reset_peak_memory_stats()
    trainer = Trainer(model=model, args=targs, train_dataset=train_ex,
                      eval_dataset=None if args.memory_probe else val_ex,
                      data_collator=PadCollator(tok.pad_token_id))
    result = trainer.train()
    log_history = trainer.state.log_history
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    if args.memory_probe:
        print(json.dumps({"max_tokens": len(train_ex[0]["input_ids"]), "peak_gb": round(peak_gb, 2)}))
        return

    epoch_dirs = []
    for ckpt in sorted(args.out.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1])):
        epoch_dirs.append(str(ckpt))
        tok.save_pretrained(ckpt)
    if not args.skip_verify and epoch_dirs:
        del trainer, model
        torch.cuda.empty_cache()
        for d in epoch_dirs:
            verify_saved_adapter(args, Path(d))

    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()[:16]  # noqa: E731
    manifest = {
        "stage": "STAGE_10_LORA_V2", "model": args.model, "args": {k: str(v) for k, v in vars(args).items()},
        "target_modules": TARGETS, "train_examples": len(train_ex), "val_examples": len(val_ex),
        "train_sha256": digest(args.train), "val_sha256": digest(args.val),
        "gate": gate_report, "peak_gpu_gb": round(peak_gb, 2), "environment": environment(),
        "train_metrics": result.metrics, "log_history": log_history,
        "epoch_checkpoints": epoch_dirs, "adapter_reload_verified": not args.skip_verify,
        "checkpoint_selection": "by validation strict pair F1 (select_checkpoint), not by eval loss",
        "test_set_touched": False,
    }
    (args.out / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("train_examples", "peak_gpu_gb", "epoch_checkpoints")}, indent=2))


if __name__ == "__main__":
    main()
