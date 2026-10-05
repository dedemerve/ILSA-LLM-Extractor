# ILSA Literature Extraction & Forecasting Pipeline

Structured metadata extraction from academic and official PDFs on International Large-Scale Assessments (PISA, TIMSS, PIRLS, ICCS, ICILS, PIAAC, TALIS), matched to OECD/IEA microdata, ending in literature-informed country-level forecasting.

Core stack: **PyMuPDF** → **OpenAI** JSON extraction → **Pydantic** validation → **Ridge** (M0/M1) + **SHAP**.

## Setup

```bash
conda activate ilsa-literature-review   # or your own environment
pip install -r requirements.txt
cp ilsa_pipeline/.env.example ilsa_pipeline/.env
# Add OPENAI_API_KEY
```

The root `requirements.txt` is the full dependency lockfile. For extraction only, `ilsa_pipeline/requirements.txt` is sufficient.

## Pipeline Overview

| Stage | Role | Key scripts | Primary outputs |
|-------|------|-------------|-----------------|
| 1 — Extraction | PDF → `ILSAArticleMetadata` JSON | `ilsa_pipeline/scripts/run_pipeline.py`, `scripts/run_batch_folders.py` | `outputs/{OECD,IEA,Scopus,Web of Science,ilsa_survey_articles}/**/json/` |
| 2 — Standardization | Dedup, Excel, taxonomy | `build_tabular_dataset.py`, `build_canonical_taxonomy.py` | `outputs/ILSA_*.xlsx`, `taxonomy_map.json` |
| 3 — Synthesis | Knowledge aggregates | `build_semantic_knowledge_base_v2.py`, `query_engine.py` | `final_knowledge_synthesis*.csv` |
| 4 — Country estimates | Microdata → country means | `build_country_estimates.py`, `compute_predictor_weights.py` | `outputs/stage4/` |
| **5 — Forecasting** | Evidence × panel × LOCO/forward | See Stage 5 below | **`outputs/stage5/` (source of truth)** |

Expert task inventory and completion status: [`docs/expert_ilsa_pipeline_checklist.md`](docs/expert_ilsa_pipeline_checklist.md).

## Stage 5 — Literature-informed forecasting (current)

```
effect_sizes / evidence_matrix / W_j
        +
country estimates + lag covariates
        ↓
enriched_panel → LOCO (M0/M1) / hold-out / diagnostic
        ↓
forward_predictions + per_country_accuracy
```

| Program | Validation | Forward cycle | Per-country accuracy |
|---------|------------|---------------|----------------------|
| PISA | Expanding LOCO (1 Ridge fold: 2025) | 2028 | Yes |
| TIMSS G8 | LOCO (6 folds) | 2027 | Yes |
| TIMSS G4 | LOCO (5 folds) | 2027 | Yes |
| PIRLS | LOCO (3 folds) | 2026 | Yes |
| ICCS | LOCO (1 fold: 2022, exploratory) | 2029 | Yes (9 countries) |
| ICILS | Pooled hold-out (test=2023) | 2028 | Yes |
| PIAAC | Single-transition LOOCV (2012→2017) | Blocked (2 cycles) | Yes (see `piaac_forward_status.csv`) |

Models: **M0** Ridge (unweighted), **M1** Ridge (features × √W_j), plus persistence / AR(1) baselines in LOCO.

### Stage 5 commands (order)

```bash
# Evidence & weights
python scripts/extract_effect_sizes.py
python scripts/build_evidence_matrix.py
python scripts/compute_predictor_weights_v2.py
python scripts/build_literature_priority.py
python scripts/build_canonical_crosswalk.py

# Microdata panels
python scripts/build_country_estimates.py
python scripts/build_covariate_estimates.py
python scripts/build_enriched_panel.py

# Core LOCO + forward (writes outputs/stage5/)
python scripts/run_loco_forecasting.py

# Program branches
python scripts/build_iccs_loco.py
python scripts/run_icils_piaac_holdout.py
python scripts/generate_icils_forward_predictions.py

# Accuracy table
python scripts/build_per_country_accuracy.py

# Full corpus refresh (OECD+IEA+Scopus+WoS+survey → evidence → W_j → accuracy)
bash scripts/run_stage5_corpus_pipeline.sh
```

### Key Stage 5 outputs

| File | Description |
|------|-------------|
| `forward_predictions.csv` | Next-cycle forecasts (PISA/TIMSS/TIMSS_G4/PIRLS/ICILS) |
| `iccs_forward_predictions.csv` | ICCS → 2029 |
| `loco_results.csv` / `loco_predictions.csv` | Temporal fold metrics and predictions |
| `per_country_accuracy.csv` | Country-level MAE/RMSE (M0/M1) |
| `validation_design_summary.csv` | Per-program validation design decisions |
| `effect_sizes.csv` | 3,083 rows / 577 papers (Scopus+WoS+survey) |

`predicted_cycle` uses official intervals (PISA 3y, TIMSS 4y, PIRLS 5y, ICILS 5y) with IEA override for ICCS 2029 — not mean historical gaps.

## Running Stage 1 — Evidence Extraction

```bash
python ilsa_pipeline/scripts/run_pipeline.py \
  --pdf-dir ./data/pdfs \
  --output-dir ./output \
  --workers 3 \
  --resume
```

Targeted batch: `ilsa_pipeline/scripts/extract_targeted.py`  
Corpus batch (Desktop folder tree): `scripts/run_batch_folders.py`  
Prompt: `prompts/extraction_system_prompt.txt`

## Dataset

Public structured outputs on HuggingFace:

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

## Corpus scope (explicit)

- On-disk corpus: **1,756** JSON records (OECD, IEA, Scopus, WoS, survey) — all flow through `build_evidence_matrix.py` and `augment_effect_sizes_from_corpus.py`.
- Bibliographic queries for Scopus/WoS were AI/assessment-focused (2020–2026); expanding queries requires new PDF extraction.
- PIAAC forward predictions require a third adult cycle; status file: `outputs/stage5/piaac_forward_status.csv`.
- Construct mapping OTHER rate after unified map: **~7%** (`scripts/canonical_predictor_map.py`).

## License

Code: MIT. Dataset: CC BY 4.0. Raw ILSA microdata and source articles are not redistributed.
