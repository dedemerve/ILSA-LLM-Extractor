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

- `output/json/*.json`: Per PDF: `file_name`, `success`, token/cost/duration fields, and `extraction` (object matching the schema).
- Parquet / SQLite helpers: `build_master_parquet`, `build_sqlite_database`, and `StorageManager` in `ilsa_pipeline/utils/storage.py`.

Legacy backups and alternate schemas live under `cop_kutusu/`.
