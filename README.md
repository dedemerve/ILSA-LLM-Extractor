# ILSA Literature Extraction Pipeline

Structured metadata extraction from academic PDFs on International Large-Scale Assessments (PISA, TIMSS, etc.) and machine learning, with a four-stage pipeline that ends in literature-informed country-level forecasting. Core stack: **PyMuPDF** for text, **OpenAI** for JSON extraction, **Pydantic** schema validation, **Ridge regression** + **SHAP** for forecasting and explainability.

## Setup

```bash
conda activate ilsa-literature-review   # or your own environment
pip install -r requirements.txt
cp ilsa_pipeline/.env.example ilsa_pipeline/.env
# Add OPENAI_API_KEY
```

The root `requirements.txt` is the full dependency lockfile. For extraction only, `ilsa_pipeline/requirements.txt` is sufficient.

## Pipeline Overview

| Stage | Script | Input | Output |
|-------|--------|-------|--------|
| 1 — Extraction | `ilsa_pipeline/scripts/run_pipeline.py` | PDFs in `data/` | `output/json/*.json` |
| 2 — Standardization | `scripts/build_structured_meta_analysis.py` | JSON outputs | `outputs/ILSA_Meta_Analysis_Dataset.xlsx` |
| 3 — RAG Synthesis | `scripts/query_engine.py` | Standardized records | `outputs/final_knowledge_synthesis.csv` |
| 4a — Country estimates | `scripts/build_country_estimates.py` | ILSA microdata | `outputs/stage4/country_estimates.csv` |
| 4b — Predictor weights | `scripts/compute_predictor_weights.py` | Knowledge synthesis | `outputs/stage4/predictor_weights.csv` |
| 4c — LOCO Forecasting | `scripts/run_loco_forecasting.py` | Estimates + weights | `outputs/stage4/loco_*.csv`, `shap_values.csv` |

## Running

### Stage 1 — Evidence Extraction

Main orchestration (batch PDFs → JSON):

```bash
python ilsa_pipeline/scripts/run_pipeline.py \
  --pdf-dir ./data/pdfs \
  --output-dir ./output \
  --workers 3 \
  --resume
```

Targeted batch: `ilsa_pipeline/scripts/extract_targeted.py`

The extraction prompt is at `prompts/extraction_system_prompt.txt`.

### Stage 4 — Country-Level Forecasting

Run stages in order:

```bash
# 4a: compute country-level estimates from ILSA microdata
python scripts/build_country_estimates.py

# 4b: derive literature-based predictor priority weights
python scripts/compute_predictor_weights.py

# 4c: run LOCO forecasting (M0 / M1 / M2) + SHAP
python scripts/run_loco_forecasting.py

# optionally restrict to specific programs or domains:
python scripts/run_loco_forecasting.py --programs PISA TIMSS --domains mathematics
```

## Stage 4 Outputs

All files land in `outputs/stage4/`:

| File | Description |
|------|-------------|
| `country_estimates.csv` | Country-level mean and SE per program, cycle, domain |
| `predictor_weights.csv` | Literature-derived priority weights ($w_j = F_j D_j C_j$) per canonical predictor |
| `loco_results.csv` | RMSE, MAE, R², Spearman per fold × model (M0/M1/M2); Diebold–Mariano rows for programs with ≥ 3 folds |
| `loco_predictions.csv` | Per-country, per-cycle predictions for all three models |
| `shap_values.csv` | Mean absolute SHAP values for M1 per program × domain × feature (requires `shap` package) |

### Models

| Model | Description |
|-------|-------------|
| M0 | Ridge, no literature weighting ($w_j = 1\ \forall j$) |
| M1 | Ridge, features scaled by $\sqrt{w_j}$ from literature weights |
| M2 | Naive persistence ($\hat{y}_{t+1} = y_t$) |

Validation uses **Leave-One-Cycle-Out (LOCO)** temporal cross-validation. The **Diebold–Mariano** test (Harvey–Leybourne–Newbold corrected) compares M1 vs M2 for programs with at least 3 evaluation folds. **SHAP** values are computed via `shap.LinearExplainer` on the fitted M1 Ridge model.

## Dataset

The structured outputs are publicly available on HuggingFace:

**[dedemerve/ILSA-LLM-Extractor-Dataset](https://huggingface.co/datasets/dedemerve/ILSA-LLM-Extractor-Dataset)**

| Table | Rows | Description |
|-------|-----:|-------------|
| `articles_master` | 1,264 | Core article metadata (deduplicated, enriched) |
| `findings` | 2,126 | Main findings per article |
| `confounders` | 8,334 | Confounders and covariates per study |
| `raw/` | 1,756 | Per-article JSON extraction outputs |

```bash
python scripts/upload_to_hf.py
```

## Extraction Outputs

- `output/json/*.json`: Per PDF, a single object with top-level keys `metadata` and `data` (same shape as `ILSAArticleMetadata`). Pipeline failures use a sentinel prefix in `data.outcome_summary` so `--resume` can retry.
- Parquet / SQLite helpers: `build_master_parquet`, `build_sqlite_database`, `StorageManager` in `ilsa_pipeline/utils/storage.py`.

## License

Code: MIT. Dataset: CC BY 4.0. Raw ILSA microdata and source articles are not redistributed.
