"""
Upload finetuning_pairs table to dedemerve/ILSA-LLM-Extractor-Dataset on HuggingFace.

Usage:
    python scripts/upload_finetuning_pairs_to_hf.py \
        --json /path/to/A1_fullcorpus_qwen3_lora_corpus.json \
        [--token YOUR_HF_TOKEN]

If --token is omitted, uses the cached login token (huggingface-cli login).
"""

import argparse
import json
import tempfile
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi, login

REPO_ID = "dedemerve/ILSA-LLM-Extractor-Dataset"
REPO_TYPE = "dataset"
TARGET_PATH = "data/processed/finetuning_pairs.parquet"
README_PATH = "README.md"

# ── New config block to inject into README YAML ────────────────────────────
NEW_CONFIG = """- config_name: finetuning_pairs
  data_files: data/processed/finetuning_pairs.parquet"""

# ── New README section to append ────────────────────────────────────────────
NEW_README_SECTION = """
### Fine-tuning Pairs (`data/processed/finetuning_pairs.parquet`)

Structured extraction pairs produced by the A1 Qwen3-8B + LoRA full-corpus run.

| Column | Description |
|--------|-------------|
| `study_id` | Unique study identifier (matches `articles_master`) |
| `pair_index` | 0-based index of the pair within the study |
| `status` | Study-level extraction status: `ok` / `truncated` / `missing_source` |
| `predictor` | Predictor variable text |
| `ilsa_variable` | ILSA-specific variable name (if available) |
| `canonical_construct` | Standardised construct code |
| `outcome` | Outcome variable text |
| `effect_type` | Effect type code (e.g. `E_assoc`) |
| `direction` | Effect direction: `positive` / `negative` / `null` |
| `effect_size` | Effect size value (if reported) |
| `evidence_text` | Supporting evidence snippet from the source document |

**Fine-tuning v2 corpus statistics:**
- Total studies: 1,266 | ok: 1,136 | truncated: 126 | missing_source: 4 | error: 0
- Total pairs: 19,245
- Model: Qwen3-8B + LoRA r16 checkpoint-81 (A1 full-corpus run)
"""


def parse_json(json_path: str) -> pd.DataFrame:
    with open(json_path) as f:
        raw = json.load(f)

    # Top-level may be a dict with a "predictions" key
    if isinstance(raw, dict):
        records = raw.get("predictions", raw)
    else:
        records = raw

    rows = []
    for rec in records:
        study_id = rec["study_id"]
        status = rec.get("status", "ok")
        for pair_idx, pair in enumerate(rec.get("pairs", [])):
            if isinstance(pair, dict):
                row = {
                    "study_id": study_id,
                    "pair_index": pair_idx,
                    "status": status,
                }
                for col, val in pair.items():
                    row[col] = json.dumps(val, ensure_ascii=False) if isinstance(val, (dict, list)) else val
                rows.append(row)
            else:
                rows.append({
                    "study_id": study_id,
                    "pair_index": pair_idx,
                    "status": status,
                    "raw": json.dumps(pair, ensure_ascii=False),
                })

    df = pd.DataFrame(rows)
    print(f"Parsed {len(df)} pairs from {len(records)} studies")
    return df


def patch_readme(readme_text: str) -> str:
    # 1. Inject config entry into YAML front-matter configs list
    if "config_name: finetuning_pairs" not in readme_text:
        # insert before closing ---
        insert_marker = "---\n\n# ILSA LLM Extractor"
        if insert_marker in readme_text:
            readme_text = readme_text.replace(
                insert_marker,
                NEW_CONFIG + "\n" + insert_marker,
            )
        else:
            # fallback: append after last config block
            last_config_end = readme_text.rfind("  data_files:")
            if last_config_end != -1:
                end_of_line = readme_text.index("\n", last_config_end)
                readme_text = (
                    readme_text[: end_of_line + 1]
                    + NEW_CONFIG
                    + "\n"
                    + readme_text[end_of_line + 1 :]
                )

    # 2. Append new section if not already present
    if "finetuning_pairs.parquet" not in readme_text:
        readme_text = readme_text.rstrip() + "\n" + NEW_README_SECTION

    return readme_text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True, help="Path to A1_fullcorpus_qwen3_lora_corpus.json")
    parser.add_argument("--token", default=None, help="HuggingFace API token (optional if already logged in)")
    args = parser.parse_args()

    if args.token:
        login(token=args.token)

    api = HfApi()

    # Parse JSON → DataFrame → Parquet
    df = parse_json(args.json)
    with tempfile.TemporaryDirectory() as tmp:
        parquet_path = Path(tmp) / "finetuning_pairs.parquet"
        df.to_parquet(parquet_path, index=False)
        print(f"Saved parquet: {parquet_path} ({parquet_path.stat().st_size / 1024:.1f} KB)")

        # Upload parquet
        api.upload_file(
            path_or_fileobj=str(parquet_path),
            path_in_repo=TARGET_PATH,
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            commit_message="Add finetuning_pairs table (A1 Qwen3 LoRA full-corpus, 19245 pairs)",
        )
        print(f"Uploaded {TARGET_PATH}")

    # Patch README
    readme_local = api.hf_hub_download(REPO_ID, README_PATH, repo_type=REPO_TYPE)
    original = open(readme_local).read()
    patched = patch_readme(original)

    if patched != original:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write(patched)
            readme_tmp = f.name
        api.upload_file(
            path_or_fileobj=readme_tmp,
            path_in_repo=README_PATH,
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            commit_message="Update README: add finetuning_pairs config and stats",
        )
        print("Updated README.md")
    else:
        print("README already up to date, skipped")

    print(f"\nDone. Dataset: https://huggingface.co/datasets/{REPO_ID}")


if __name__ == "__main__":
    main()
