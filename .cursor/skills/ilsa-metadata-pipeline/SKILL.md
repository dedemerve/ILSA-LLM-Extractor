---
name: ilsa-metadata-pipeline
description: >-
  Runs the full ILSAArticleMetadata post-processing pipeline (resanitize JSON,
  build Excel/taxonomy/synthesis, test, git push). Use when the user asks to
  run ILSAArticleMetadata scripts, refresh outputs, sync to GitHub, or execute
  the metadata pipeline after schema or extractor changes.
---

# ILSA Article Metadata Pipeline

Orchestrates all local (no OpenAI) steps that keep on-disk JSON aligned with
`ILSAArticleMetadata` in `src/schemas/models.py`, then rebuilds derived artifacts
and pushes to GitHub.

## When to use

- User mentions **ILSAArticleMetadata**, **ILSAArticleMetaData**, metadata pipeline, or "kodları çalıştır"
- After changes to `src/schemas/models.py`, `gpt_extractor.py`, or enrichment modules
- User wants outputs refreshed and pushed to `https://github.com/dedemerve/ILSA-LLM-Extractor.git`

## Quick start

From repo root (`/Users/mrved/Desktop/ILSA_LLMs`):

```bash
# Full pipeline + git push (user must confirm push if not explicitly requested)
bash scripts/run_ilsa_metadata_pipeline.sh

# Skip git
bash scripts/run_ilsa_metadata_pipeline.sh --no-git

# Push only (after manual runs)
bash scripts/run_ilsa_metadata_pipeline.sh --git-only
```

Requires: project `.venv` or system Python with `requirements.txt` installed.
`.env` is **not** needed for this pipeline (no API calls).

## Pipeline order (do not reorder)

| Step | Command | Output |
|------|---------|--------|
| 1 | `resanitize_json_outputs.py --json-dir outputs --recursive` | Updated `outputs/**/json/*.json` only (not taxonomy JSON at repo root) |
| 2 | `find_missing_main_findings.py --json-dir outputs --recursive --fix` | Filled legacy/empty `main_findings` |
| 3 | `build_tabular_dataset.py` | `outputs/ILSA_Meta_Analysis_Dataset*.xlsx` |
| 4 | `build_structured_meta_analysis.py` | `outputs/ILSA_Structured_Meta_Analysis.xlsx` |
| 5 | `build_canonical_taxonomy.py` | `outputs/taxonomy_map.json`, `knowledge_synthesis.csv`, `Canonical_View` sheet |
| 6 | `build_semantic_knowledge_base_v2.py --version v4` | `final_knowledge_synthesis_v4.csv`, codebook |
| 7 | `build_analytical_master.py` | `outputs/ILSA_Analytical_Meta_Analysis_Master.xlsx` |
| 8 | `generate_academic_synthesis.py` | `academic_synthesis_report_tr.md` |
| 9 | `pytest` (confounders, doi, main_findings) | Must pass before git |
| 10 | `_batch_git_add_outputs.sh` + commit + push | Remote sync |

**Optional (Phase 0 — not in default pipeline):** `init_phase0_verification_sheets.py` adds Sheets 5–7 when you are ready for verification/inventory work.

## Git rules

1. Never commit `.env`, credentials, or `*.parquet` (gitignored).
2. Use `_batch_git_add_outputs.sh` for large corpus trees (IEA, OECD, Scopus, WoS, ilsa_survey_articles).
3. Stage root artifacts: `outputs/*.xlsx`, `outputs/*.csv`, `outputs/*.json`, `outputs/*.md`.
4. Commit message: English, 1–2 sentences on **why** (schema refresh, pipeline rebuild, etc.).
5. **Push only when the user explicitly asks** (this skill's default script pushes; ad-hoc runs should use `--no-git` unless asked).

## Git on Desktop / iCloud

Large repos on `~/Desktop` may hit `index.lock` write timeouts. If push fails:

1. Run `bash scripts/run_ilsa_metadata_pipeline.sh --git-only` from Terminal (not sandbox)
2. Or move the repo off iCloud-synced Desktop before bulk commits
3. Use `_batch_git_add_outputs.sh` (excludes `*.log`, `*.parquet`)

## Optional API steps (NOT in default pipeline)

Run separately when user requests new extractions:

- `scripts/extract_missing_articles.py` — numbered PDFs in `~/Desktop/articles`
- `scripts/extract_survey_papers.py` — Survey Paper folder (needs `OPENAI_API_KEY`)
- `ilsa_pipeline/scripts/run_pipeline.py` — batch PDF folders

After API extraction, always re-run the full local pipeline above.

## Failure handling

- **Resanitize FAIL**: log filename; continue others; report failures at end.
- **Schema validation FAIL**: do not commit; fix extractor/sanitize rules first.
- **pytest FAIL**: stop before git; fix tests or schema.
- **Git push rejected**: report error; do not force-push to `main`.

## Validation spot-check

```bash
python -c "
from pathlib import Path
import json
from src.schemas.models import validate_public_article_json
bad = []
for p in Path('outputs').rglob('json/*.json'):
    try:
        validate_public_article_json(json.loads(p.read_text()))
    except Exception as e:
        bad.append((p.name, str(e)[:80]))
print(f'Invalid: {len(bad)} / {sum(1 for _ in Path(\"outputs\").rglob(\"json/*.json\"))}')
"
```

## Report to user (Turkish)

After a run, summarize:

- Kaç JSON resanitize edildi / kaç hata
- Excel ve synthesis dosyaları güncellendi mi
- pytest sonucu
- Git commit hash ve push durumu
