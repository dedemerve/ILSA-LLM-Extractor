#!/usr/bin/env python3
"""Run a deterministic draw slice of the v4 uncertainty analysis in isolation."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--script", type=Path, default=Path(__file__).resolve().with_name("run_pisa_model_input_uncertainty_v4.py"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--draw-start", type=int, required=True)
    parser.add_argument("--draw-stop", type=int, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    start, stop = args.draw_start, args.draw_stop
    if not (0 <= start < stop <= 2000) or start % 25 or stop % 25:
        raise ValueError("Draw range must be within [0,2000] and aligned to 25-draw checkpoints")

    script = args.script.resolve()
    raw = script.read_bytes()
    source = raw.decode("utf-8")
    seeds = np.random.default_rng(20261015).integers(0, 2**31, size=2000)
    seed_literal = repr([int(x) for x in seeds[start:stop]])

    out_pattern = re.compile(r"^OUT_DIR\s*=.*$", re.MULTILINE)
    n_pattern = re.compile(r"^N_DRAWS\s*=\s*2000\s*$", re.MULTILINE)
    seed_pattern = re.compile(r"^\s*draw_seeds\s*=\s*rng_master\.integers\(0, 2\*\*31, size=N_DRAWS\)\s*$", re.MULTILINE)
    if len(out_pattern.findall(source)) != 1 or len(n_pattern.findall(source)) != 1 or len(seed_pattern.findall(source)) != 1:
        raise ValueError("Source constants differ from the audited v4 script; refusing an unreviewed shard run")
    transformed = out_pattern.sub(f"OUT_DIR = Path({str(args.output_dir.resolve())!r})", source, count=1)
    transformed = n_pattern.sub(f"N_DRAWS = {stop-start}", transformed, count=1)
    transformed = seed_pattern.sub(f"    draw_seeds = np.asarray({seed_literal}, dtype=np.int64)", transformed, count=1)
    compile(transformed, str(script), "exec")
    if str(script.parent) not in sys.path:
        sys.path.insert(0, str(script.parent))
    namespace = {"__name__": "__main__", "__file__": str(script), "__package__": None}
    exec(compile(transformed, str(script), "exec"), namespace)
    checkpoint_dir = args.output_dir / "checkpoints"
    expected = list(range(0, stop-start, 25))
    actual = sorted(int(p.stem.split("_")[2]) for p in checkpoint_dir.glob("draw_metrics_*.csv"))
    if actual != expected:
        raise ValueError(f"Shard checkpoint starts differ: expected {expected}, got {actual}")
    metadata = {
        "source_script": str(script), "source_script_sha256": sha256(raw),
        "execution_source_sha256": sha256(transformed.encode()),
        "draw_start_in_full_run": start, "draw_stop_in_full_run_exclusive": stop,
        "draw_seed_slice_sha256": sha256(np.asarray(seeds[start:stop], dtype=np.int64).tobytes()),
        "only_changes": ["OUT_DIR", "local N_DRAWS", "draw_seeds replaced by corresponding slice from original 2,000-seed sequence"],
        "model_and_metric_code_unchanged": True,
        "output_dir": str(args.output_dir.resolve()), "completed_utc": datetime.now(timezone.utc).isoformat(),
    }
    (args.output_dir / "shard_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
