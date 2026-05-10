# ILSA Literature Extraction Pipeline

A pipeline designed to process ILSA academic literature and extract structured metadata utilizing the GPT-5.4-nano Structured Outputs API.

## Setup
```Bash
conda activate ilsa-literature-review
pip install -r requirements.txt

cp .env.example .env
# Add your OPENAI_API_KEY to the newly created .env file
```

## Usage
Test Run (20 documents)

```Bash
python scripts/run_pipeline.py \
    --pdf-dir ../data/pdfs \
    --output-dir ./output_test \
    --limit 20 \
    --workers 3
Full Execution (1,800 documents)
```

Full Execution (1,800 documents)
```Bash
python scripts/run_pipeline.py \
    --pdf-dir ../data/pdfs \
    --output-dir ./output \
    --workers 5 \
    --resume
Knowledge Base Queries
```

Knowledge Base Queries
```Bash
python scripts/example_queries.py --db ./output/ilsa_knowledge_base.db
Outputs
output/json/*.json: Individual structured JSON files for each processed document.
```

`output/ilsa_master.parquet`: Aggregated dataset optimized for machine learning and downstream analysis.

`output/ilsa_knowledge_base.db`: SQLite database for structured SQL queries.
