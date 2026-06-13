# ILSA-LLM-Extractor

A large-scale automated metadata extraction pipeline for International Large-Scale Assessment (ILSA) documents. The pipeline processes PDF publications from IEA (TIMSS, PIRLS, ICCS) and OECD (PISA, TALIS, PIAAC) assessments using a Retrieval-Augmented Generation (RAG) architecture powered by large language models.

## Overview

International Large-Scale Assessments produce thousands of technical reports, policy briefs, and research publications annually. Manually extracting structured metadata from this volume of documents is time-consuming and error-prone. This pipeline automates the process by combining PDF text extraction, structured prompting, and LLM-based inference to produce a clean, queryable dataset.

The pipeline achieved **99.2% classification accuracy** across 1,680 documents spanning six major ILSA programs.

## Dataset

The structured output dataset is publicly available on Hugging Face:

[https://huggingface.co/datasets/dedemerve/ILSA-LLM-Extractor-Dataset](https://huggingface.co/datasets/dedemerve/ILSA-LLM-Extractor-Dataset)

The dataset includes six subsets:

| Subset | Description | Rows |
|--------|-------------|------|
| `meta_analysis` | Core metadata extracted from ILSA documents | 36 |
| `knowledge_synthesis` | Synthesized knowledge across assessments | 174 |
| `codebook` | Variable definitions and descriptions | 39 |
| `semantic_synthesis` | Method-variable-effect relations for RAG retrieval | 174 |
| `variable_registry` | Official ILSA variable registry | 1 |
| `policy_taxonomy` | Policy domain taxonomy across assessments | 31 |

## Pipeline Architecture

PDF Documents (1,680)

|

v

Text Extraction (PyMuPDF)

|

v

Chunking and Preprocessing

|

v

Structured Prompting (OpenAI GPT)

|

v

Schema Validation (Pydantic)

|

v

Storage (JSON / Parquet / SQLite)

### Core Components

- **Text Extraction:** PyMuPDF extracts raw text from PDF documents while preserving structural elements such as headings, tables, and metadata blocks.
- **Schema Definition:** Pydantic models in `src/schemas/models.py` define the `ILSAArticleMetadata` schema, enforcing type safety and validation on all extracted fields.
- **LLM Extraction:** OpenAI GPT models process chunked text and return structured JSON conforming to the schema. Prompts are versioned and stored under `prompts/`.
- **Storage Layer:** `StorageManager` in `ilsa_pipeline/utils/storage.py` handles output to JSON, Parquet, and SQLite formats with normalization across six relational tables.
- **Resume Support:** Failed extractions are flagged with a sentinel prefix in `data.outcome_summary`, enabling targeted retry via `--resume`.

## Coverage

| Organization | Assessments |
|-------------|-------------|
| IEA | TIMSS, PIRLS, ICCS |
| OECD | PISA, TALIS, PIAAC |

## Setup

```bash
conda activate ilsa-literature-review
pip install -r ilsa_pipeline/requirements.txt
cp ilsa_pipeline/.env.example ilsa_pipeline/.env
# Add your OPENAI_API_KEY to .env
```

The root `requirements.txt` is a full conda-style lockfile. For extraction only, `ilsa_pipeline/requirements.txt` is sufficient.

## Usage

### Full Pipeline

```bash
cd /path/to/ILSA-LLM-Extractor
python ilsa_pipeline/scripts/run_pipeline.py \
  --pdf-dir ./data/pdfs \
  --output-dir ./output \
  --workers 3 \
  --resume
```

### Targeted Extraction

```bash
python ilsa_pipeline/scripts/extract_targeted.py \
  --pdf-dir ./data/pdfs/IEA \
  --output-dir ./output
```

## Output Structure

output/
├── json/
│   └── *.json              # Per-document extraction (metadata + data)
├── parquet/
│   └── *.parquet           # Columnar format for analysis
└── sqlite/
└── ilsa.db             # Normalized relational database (6 tables)

Each JSON file contains a single object with top-level keys `metadata` and `data`, matching the `ILSAArticleMetadata` Pydantic schema.

## Repository Structure
ILSA-LLM-Extractor/

├── data/                   # Input PDFs (not tracked)
├── docs/                   # Documentation
├── ilsa_pipeline/
│   ├── scripts/            # Orchestration scripts
│   ├── utils/
│   │   └── storage.py      # StorageManager
│   ├── requirements.txt
│   └── .env.example
├── outputs/                # Extraction results
├── prompts/                # Versioned LLM prompts
├── src/
│   └── schemas/
│       └── models.py       # Pydantic schema definitions
└── tests/

## Performance

| Metric | Value |
|--------|-------|
| Documents processed | 1,680 |
| Classification accuracy | 99.2% |
| Output variables per document | 11 |
| Storage formats | JSON, Parquet, SQLite |

## Citation

If you use this pipeline or dataset in your research, please cite:

```bibtex
@dataset{dede_cetinkaya_2026_ilsa,
  author    = {Dede, Merve and {\c{C}}etinkaya, Ekrem},
  title     = {ILSA LLM Extractor Dataset},
  year      = {2026},
  publisher = {Hugging Face},
  url       = {https://huggingface.co/datasets/dedemerve/ILSA-LLM-Extractor-Dataset}
}
```

## License

This project is licensed under the MIT License. The associated dataset is released under CC BY 4.0.
