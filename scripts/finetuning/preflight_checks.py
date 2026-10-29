"""
Pre-training / pre-evaluation gate for the LoRA experiments (action plan step 1).

Checks a chat-format SFT JSONL against the tokenizer that will be used, and
optionally an adapter directory, for the defects found in the Stage 7/9 run:

  1. real token length of every example vs max_length (no chars/4 estimates)
  2. the chat template actually renders the system message
     (Mistral v0.3 only renders it when the user turn is the LAST message,
     so it is silently dropped from every training conversation)
  3. pad token is distinct from EOS, and under the default LM collator the
     final EOS label survives (is not -100)
  4. the supervised completion ends with the closing bracket followed by EOS
  5. adapter safetensor keys can be loaded by PeftModel.from_pretrained, and
     would be silently dropped by load_state_dict(strict=False) on a fresh
     get_peft_model (the Stage 9 loading path)

Usage
  python3 scripts/finetuning/preflight_checks.py \
      --data outputs/finetuning/data/train_finetune.jsonl \
      --tokenizer /path/to/tokenizer_or_model --max-length 32768 \
      [--pad-token unk] [--adapter /path/to/lora_adapter] [--out audit.json]

Exit code is non-zero if any blocking check fails.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any


def _render(tok: Any, messages: list[dict[str, str]]) -> str:
    return tok.apply_chat_template(messages, tokenize=False)


def check_dataset(data: Path, tokenizer: str, max_length: int, pad_token: str | None) -> dict[str, Any]:
    from transformers import AutoTokenizer, DataCollatorForLanguageModeling

    tok = AutoTokenizer.from_pretrained(tokenizer)
    if pad_token == "unk":
        tok.pad_token = tok.unk_token
    elif pad_token == "eos":
        tok.pad_token = tok.eos_token
    elif pad_token:
        tok.pad_token = pad_token

    rows = [json.loads(line) for line in data.read_text(encoding="utf-8").splitlines() if line.strip()]
    lengths, completion_lengths, system_dropped, bad_tail = [], [], 0, []
    for k, row in enumerate(rows):
        msgs = row["messages"] if "messages" in row else (row["prompt"] + row["completion"])
        text = _render(tok, msgs)
        ids = tok(text, add_special_tokens=False)["input_ids"]
        lengths.append(len(ids))
        completion = msgs[-1]["content"]
        completion_lengths.append(len(tok(completion, add_special_tokens=False)["input_ids"]))
        system = next((m["content"] for m in msgs if m["role"] == "system"), None)
        if system and system[:200] not in text:
            system_dropped += 1
        tail = tok.decode(ids[-3:])
        if not (tail.rstrip().endswith(("]" + tok.eos_token, "}" + tok.eos_token))):
            bad_tail.append({"index": k, "tail": tail})

    probe = tok("[1]", add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
    labels = DataCollatorForLanguageModeling(tok, mlm=False)([{"input_ids": probe}])["labels"][0].tolist()
    q = lambda xs: {"min": min(xs), "median": int(statistics.median(xs)),  # noqa: E731
                    "p95": sorted(xs)[int(0.95 * (len(xs) - 1))], "max": max(xs)}
    over = [i for i, n in enumerate(lengths) if n > max_length]
    result = {
        "data": str(data), "tokenizer": tokenizer, "n_examples": len(rows),
        "max_length": max_length, "total_tokens": q(lengths), "completion_tokens": q(completion_lengths),
        "completion_token_share": round(sum(completion_lengths) / sum(lengths), 4),
        "over_max_length": over,
        "system_message_dropped_by_template": system_dropped,
        "pad_token": tok.pad_token, "pad_equals_eos": tok.pad_token_id == tok.eos_token_id,
        "eos_label_masked_by_default_collator": labels[-1] == -100,
        "completion_not_ending_in_bracket_eos": bad_tail,
    }
    result["blocking_failures"] = [name for name, failed in (
        ("examples_exceed_max_length", bool(over)),
        ("system_message_dropped", system_dropped > 0),
        ("pad_token_equals_eos", result["pad_equals_eos"]),
        ("eos_label_masked", result["eos_label_masked_by_default_collator"]),
        ("completion_tail", bool(bad_tail)),
    ) if failed]
    return result


def check_adapter(adapter: Path) -> dict[str, Any]:
    from safetensors import safe_open

    cfg = json.loads((adapter / "adapter_config.json").read_text())
    with safe_open(str(adapter / "adapter_model.safetensors"), "pt") as f:
        keys = list(f.keys())
        b_norms = [float(f.get_tensor(k).float().norm()) for k in keys if "lora_B" in k]
    adapter_name = "default"
    # Keys a fresh get_peft_model(...) exposes carry the adapter name; PEFT strips it on save.
    expected = [k.replace(".lora_A.", f".lora_A.{adapter_name}.").replace(".lora_B.", f".lora_B.{adapter_name}.")
                for k in keys]
    silently_dropped = sum(1 for k, e in zip(keys, expected) if k != e)
    return {
        "adapter": str(adapter), "base_model": cfg.get("base_model_name_or_path"),
        "r": cfg.get("r"), "lora_alpha": cfg.get("lora_alpha"), "target_modules": cfg.get("target_modules"),
        "n_tensors": len(keys),
        "lora_B_nonzero": sum(n > 0 for n in b_norms), "lora_B_total": len(b_norms),
        "keys_lacking_adapter_name": silently_dropped,
        "load_state_dict_strict_false_would_load": len(keys) - silently_dropped,
        "verdict": ("Use PeftModel.from_pretrained(base, path) and assert_adapter_loaded(); "
                    "load_state_dict(strict=False) on get_peft_model() loads 0 of these tensors.")
        if silently_dropped else "Key names already carry the adapter name.",
    }


def assert_adapter_loaded(model: Any, adapter: Path) -> None:
    """Call after PeftModel.from_pretrained: every saved tensor must equal the live parameter."""
    import torch
    from safetensors.torch import load_file

    saved = load_file(str(adapter / "adapter_model.safetensors"))
    live = {k.replace(".default", ""): v for k, v in model.state_dict().items() if "lora_" in k}
    missing = [k for k in saved if k not in live]
    if missing:
        raise RuntimeError(f"{len(missing)} adapter tensors not present in model, e.g. {missing[:2]}")
    mismatched = [k for k, v in saved.items() if not torch.equal(v.to(live[k].device, live[k].dtype), live[k])]
    if mismatched:
        raise RuntimeError(f"{len(mismatched)} adapter tensors differ from the file, e.g. {mismatched[:2]}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path)
    ap.add_argument("--tokenizer")
    ap.add_argument("--max-length", type=int, default=32768)
    ap.add_argument("--pad-token", help="'unk', 'eos', or a literal token; default keeps the tokenizer's")
    ap.add_argument("--adapter", type=Path)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    report: dict[str, Any] = {}
    if args.data:
        report["dataset"] = check_dataset(args.data, args.tokenizer, args.max_length, args.pad_token)
    if args.adapter:
        report["adapter"] = check_adapter(args.adapter)
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(text)
    failures = report.get("dataset", {}).get("blocking_failures", [])
    if failures or report.get("adapter", {}).get("keys_lacking_adapter_name"):
        sys.exit(1)


if __name__ == "__main__":
    main()
