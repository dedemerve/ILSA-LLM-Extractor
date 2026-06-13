  # ILSA-LLM-Extractor
  
  A large-scale automated metadata extraction pipeline for International Large-Scale Assessment (ILSA) documents. The pipeline processes PDF publications from IEA (TIMSS, PIRLS, ICCS) and OECD (PISA, TALIS, PIAAC) assessments using a Retrieval-Augmented Generation (RAG) architecture powered by large language models.
  
  ## Overview
  
  International Large-Scale Assessments produce thousands of technical reports, policy briefs, and research publications annually. Manually extracting structured metadata from this volume of documents is time-consuming and error-prone. This pipeline automates the process by combining PDF text extraction, structured prompting, and LLM-based inference to produce a clean, queryable dataset.
  
  The pipeline achieved **99.2% classification accuracy** across 1,680 peer-reviewed articles using data from IEA (TIMSS, PIRLS, ICCS) and OECD (PISA, TALIS, PIAAC) assessments.
  
  ## Dataset
  
  The structured output dataset is publicly available on Hugging Face:
  
  [https://huggingface.co/datasets/dedemerve/ILSA-LLM-Extractor-Dataset](https://huggingface.co/datasets/dedemerve/ILSA-LLM-Extractor-Dataset)
  
  The dataset includes six subsets:
  
  | Subset | Description | Rows |
  |--------|-------------|------|
  | `meta_analysis` | Core metadata extracted from ILSA documents | 36 |
  | `knowledge_synthesis` | Synthesized knowledge across assessments | 174 |
  | `codebook` | Variable definitions and descriptions | 39 |
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
  
  ## Pipeline Stages
  
  ### Stage 1: Knowledge Extraction (Human-AI Collaboration)
  
  **Data Collection**
  
  | Source | Count |
  |--------|-------|
  | IEA Database | 308 studies |
  | OECD iLibrary | 591 studies |
  | Scopus | 423 studies |
  | Web of Science | 302 studies |
  | Google Scholar | 56 studies |
  | Total | 1,680 studies |
  
  **Human Expert Design**
  - Pydantic schema with domain-specific fields
  - Extraction rules (inclusion/exclusion criteria)
  - Quality validation criteria
  
  **LLM-Based Structured Extraction**
  - Model: GPT-5.4 nano (2026-03-17)
  - Schema: Pydantic v2
  - Extracted fields: metadata (title, authors, DOI, venue, year), methods (ML techniques, sample design, analytical approach), variables (predictors, outcomes, confounders), findings (research questions, effect sizes, metrics)
  - Output: 1,680 JSON files generated
  
  **Deduplication**
  - DOI matching and normalized title/author comparison
  - 1,680 JSON files -> 1,266 unique study records
  
  **Validation**
  - Automated: Pydantic validation and anti-hallucination rules
  - Manual: Inter-rater validation on a subset of records
  
  **Stage 1 Output**
  - Articles Master: 1,266 rows (metadata, methods, samples)
  - Main Findings: 1,893 rows (research questions, effects)
  - Confounders: 7,655 rows (controlled variables)
  
  ### Stage 2: Knowledge Synthesis (Human + AI Validation)
  
  **Standardization Tasks**
  - Terminology alignment (e.g. multilevel modeling = hierarchical linear modeling)
  - Method taxonomy mapping
  - Variable standardization (ESCS, HISEI, PARED -> unified SES construct)
  - Cross-study pattern aggregation (effect sizes, trends, relationships)
  
  **Human Oversight**
  - Adjudicates ambiguous cases
  - Validates semantic consistency
  - Preserves theoretical validity
  - Human experts reviewed standardization rules and resolved conflicts to ensure domain-specific accuracy
  
  **Stage 2 Output**
  - Unified Reference Knowledge Base across 1,266 studies
  - Aggregated empirical patterns across studies
  - Standardized methodological taxonomy
  - Unified variable nomenclature
  
  ### Stage 3: Knowledge Transfer (RAG-Based Agent)
  
  **Retrieval-Augmented Generation**
  - Semantic search over knowledge base (text-embedding-3-large)
  - Retrieves k most relevant studies per query
  - Combines retrieved evidence with new ILSA data
  - Citation-based explainability (agent cites study IDs)
  
  **Human-Guided Prompt Engineering**
  - Context-relevant variable recommendations
  - Evidence-based analytical strategies (sampling weight usage, plausible value treatments)
  - Country-specific testable hypotheses
  
  **Stage 3 Output**
  - Country-specific analytical recommendations
  - Variable selection grounded in 1,266 studies
  - Methodological guidance with citations
  - Evidence-based hypotheses (testable predictions)
  
  ## System-Level Metrics
  
  | Metric | Value |
  |--------|-------|
  | Total PDFs processed | 1,680 |
  | Unique study records | 1,266 |
  | Main findings rows | 1,893 |
  | Confounder rows | 7,655 |
  | LLM model | GPT-5.4 nano (2026-03-17) |
  | Extraction schema | Pydantic v2 |
  | Validation | Automated (Pydantic + anti-hallucination rules) |
  | Classification accuracy | 99.2% |
  
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
