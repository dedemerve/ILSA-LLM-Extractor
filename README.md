# ILSA Literature Extraction Pipeline

Structured metadata extraction from academic PDFs on International Large-Scale Assessments (PISA, TIMSS, etc.) and machine learning. Core stack: **PyMuPDF** for text, **OpenAI** for JSON extraction, **Pydantic** schema in `src/schemas/models.py`.

## Setup

```bash
conda activate ilsa-literature-review   # or your own environment
pip install -r ilsa_pipeline/requirements.txt
cp ilsa_pipeline/.env.example ilsa_pipeline/.env
# Add OPENAI_API_KEY
```

The root `requirements.txt` is a full conda-style lockfile. For extraction only, `ilsa_pipeline/requirements.txt` is sufficient.

## Running

Main orchestration (batch PDFs, JSON plus optional SQLite):

```bash
cd /path/to/ILSA_LLMs
python ilsa_pipeline/scripts/run_pipeline.py \
  --pdf-dir ./data/pdfs \
  --output-dir ./output \
  --workers 3 \
  --resume
```

Targeted batch: `ilsa_pipeline/scripts/extract_targeted.py`

## Outputs

- `output/json/*.json`: Per PDF, a single object with **only** top-level keys `metadata` and `data` (same shape as the Pydantic `ILSAArticleMetadata` public schema). Pipeline-only failures use a sentinel prefix inside `data.outcome_summary` so `--resume` can retry those files.
- Parquet / SQLite helpers: `build_master_parquet`, `build_sqlite_database`, and `StorageManager` in `ilsa_pipeline/utils/storage.py`.

Legacy backups and alternate schemas live under `cop_kutusu/`.

## Dataset

The structured outputs are publicly available on HuggingFace:

**[dedemerve/ILSA-LLM-Extractor-Dataset](https://huggingface.co/datasets/dedemerve/ILSA-LLM-Extractor-Dataset)**

| Table | Rows | Description |
|-------|-----:|-------------|
| `articles_master` | 1,264 | Core article metadata (deduplicated, enriched) |
| `findings` | 2,126 | Main findings per article |
| `confounders` | 8,334 | Confounders and covariates per study |
| `raw/` | 1,756 | Per-article JSON extraction outputs |

To upload or update the dataset:

```bash
python scripts/upload_to_hf.py
```
