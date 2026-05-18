#!/usr/bin/env python3
"""
Re-apply GPTExtractor._sanitize (+ _post_process_model) to on-disk article JSON.

Use after post-processing rule changes without re-running OpenAI extraction.
Default: all *.json under outputs/articles/json/.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.extractors.gpt_extractor import GPTExtractor
from src.schemas.models import validate_public_article_json

DEFAULT_JSON_DIR = PROJECT_ROOT / "outputs" / "articles" / "json"


def resanitize_file(path: Path) -> dict:
    raw = json.loads(path.read_text(encoding="utf-8"))
    sanitized = GPTExtractor._sanitize(raw)
    # On-disk shape: {"metadata": ..., "data": ...}; validate full root after sanitize.
    model = validate_public_article_json(sanitized)
    GPTExtractor._post_process_model(model)
    validate_public_article_json(model.model_dump(mode="json"))
    return model.model_dump(mode="json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json-dir",
        type=Path,
        default=DEFAULT_JSON_DIR,
        help=f"Directory with article JSON files (default: {DEFAULT_JSON_DIR})",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Process all *.json under --json-dir recursively (e.g. outputs/IEA)",
    )
    parser.add_argument(
        "files",
        nargs="*",
        help="Optional specific JSON paths; default processes all in --json-dir",
    )
    args = parser.parse_args()

    if args.files:
        paths = [Path(p) for p in args.files]
    else:
        if args.recursive:
            paths = sorted(args.json_dir.rglob("*.json"))
        else:
            paths = sorted(args.json_dir.glob("*.json"))

    if not paths:
        print(f"No JSON files found in {args.json_dir}")
        return

    ok = 0
    for path in paths:
        try:
            out = resanitize_file(path)
            path.write_text(
                json.dumps(out, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            title = (out.get("metadata") or {}).get("title") or "(null)"
            n_conf = len((out.get("data") or {}).get("confounders_identified") or [])
            n_mf = len((out.get("data") or {}).get("main_findings") or [])
            print(f"OK  {path.name}")
            print(f"    title: {title[:80]}")
            print(f"    confounders: {n_conf} | main_findings: {n_mf}")
            ok += 1
        except Exception as exc:
            print(f"FAIL {path.name}: {exc}")

    print(f"\nResanitized {ok}/{len(paths)} files.")


if __name__ == "__main__":
    main()
