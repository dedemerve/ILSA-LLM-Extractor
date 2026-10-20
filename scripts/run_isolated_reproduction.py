#!/usr/bin/env python3
"""Run an unchanged analysis script with only its output directory redirected.

The computation source is kept in memory and the original script file remains
untouched. The redirection and original/transformed source hashes are recorded
beside the isolated outputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("script", type=Path)
    p.add_argument("output_dir", type=Path)
    p.add_argument("--resume", action="store_true", help="Reuse an existing isolated output directory for checkpointed scripts")
    args = p.parse_args()
    script = args.script.resolve()
    output = args.output_dir.resolve()
    if output.exists() and not args.resume:
        raise FileExistsError(output)
    source_bytes = script.read_bytes()
    source = source_bytes.decode("utf-8")
    # Redirect exactly one OUT_DIR assignment. Fail closed on scripts with
    # computed/multiple output roots; those need a purpose-built audit.
    pattern = re.compile(r"^(OUT_DIR\s*=\s*).*$", re.MULTILINE)
    matches = list(pattern.finditer(source))
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one OUT_DIR assignment; found {len(matches)}")
    replacement = f"OUT_DIR = Path({str(output)!r})"
    transformed = pattern.sub(replacement, source, count=1)
    compile(transformed, str(script), "exec")
    if str(script.parent) not in sys.path:
        sys.path.insert(0, str(script.parent))
    ns = {"__name__": "__main__", "__file__": str(script), "__package__": None}
    try:
        exec(compile(transformed, str(script), "exec"), ns)
    except BaseException as exc:
        if output.is_dir():
            (output / "isolated_reproduction_failure.txt").write_text(
                f"{type(exc).__name__}: {exc}\n", encoding="utf-8"
            )
        raise
    if not output.is_dir():
        raise RuntimeError(f"Script completed without creating expected output directory: {output}")
    wrapper = {
        "original_script": str(script), "original_script_sha256": digest(source_bytes),
        "execution_source_sha256": digest(transformed.encode()),
        "change": "Only OUT_DIR assignment redirected in-memory; all other script text unchanged",
        "output_dir": str(output), "completed_utc": datetime.now(timezone.utc).isoformat(),
    }
    (output / "isolated_reproduction_wrapper.json").write_text(json.dumps(wrapper, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(wrapper, indent=2))


if __name__ == "__main__":
    main()
