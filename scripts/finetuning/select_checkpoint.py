"""
Select the LoRA checkpoint by validation extraction quality, not by validation loss.

Scores every predictions file of the form <run>_checkpoint-<step>_val.json produced by
extract_v2.py and ranks them by strict pair F1, then lenient pair F1, then the earlier
checkpoint (less training). Writes outputs/finetuning/v2/eval/checkpoint_selection_<split>.json.

Usage
  python3 scripts/finetuning/select_checkpoint.py --split val
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_extraction import evaluate  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
PRED_DIR = ROOT / "outputs/finetuning/v2/predictions"
RUN_DIR = ROOT / "outputs/finetuning/v2/runs"
EVAL_DIR = ROOT / "outputs/finetuning/v2/eval"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="val", choices=["val"])
    args = ap.parse_args()

    pattern = re.compile(rf"^(?P<run>.+)_checkpoint-(?P<step>\d+)_{args.split}\.json$")
    rows = []
    for path in sorted(PRED_DIR.glob(f"*_checkpoint-*_{args.split}.json")):
        m = pattern.match(path.name)
        if not m:
            continue
        summary, *_ = evaluate(path, args.split)
        rows.append({
            "run": m["run"], "step": int(m["step"]), "predictions": str(path),
            "adapter": str(RUN_DIR / m["run"] / f"checkpoint-{m['step']}"),
            "strict_f1": summary["strict"]["f1"], "strict_ci95": summary["strict"]["ci95_study_bootstrap"],
            "lenient_f1": summary["lenient"]["f1"], "grounding_rate": summary["grounding_rate"],
            "duplicate_rate": summary["duplicate_rate"], "validity": summary["validity"],
        })
    if not rows:
        raise SystemExit(f"no *_checkpoint-*_{args.split}.json files in {PRED_DIR}")
    rows.sort(key=lambda r: (-r["strict_f1"], -r["lenient_f1"], r["step"]))
    best_per_run = {}
    for r in rows:
        best_per_run.setdefault(r["run"], r)
    out = {"criterion": "strict pair F1 on validation, then lenient F1, then earlier checkpoint",
           "selected": rows[0], "best_per_run": best_per_run, "all": rows,
           "caveat": "6 validation studies: differences inside the bootstrap CI are not evidence that one "
                     "checkpoint is better; prefer the earlier checkpoint when CIs overlap."}
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    target = EVAL_DIR / f"checkpoint_selection_{args.split}.json"
    target.write_text(json.dumps(out, indent=2), encoding="utf-8")
    for r in rows:
        print(f"{r['run']:<40} step {r['step']:>4}  strict {r['strict_f1']:.3f} {r['strict_ci95']}  lenient {r['lenient_f1']:.3f}")
    print(f"selected: {rows[0]['adapter']}  -> {target}")


if __name__ == "__main__":
    main()
