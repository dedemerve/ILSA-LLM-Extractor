# An Explainable LLM-Augmented Forecasting Framework for International Large-Scale Educational Assessments

## Overview

This project develops a hybrid forecasting framework that combines classical time-series models with large language model (LLM) context integration to predict country-level performance on International Large-Scale Assessments (ILSAs), with a primary focus on PISA mathematics scores. The framework enriches quantitative forecasts with policy context retrieved from structured knowledge bases, enabling explanations that go beyond numerical extrapolation. Validation is conducted via retrospective backtesting: models are trained on data up to 2018 and evaluated against the 2022 PISA cycle.

## Methodology

The framework operates in three layers:

**Layer 1 — Classical Baseline Models**
ARIMA, Prophet, and LSTM models are trained independently on historical ILSA score trajectories. These serve as the quantitative backbone and provide baseline forecasts against which the augmented pipeline is compared.

**Layer 2 — LLM Context Integration**
For each country-cycle pair, a structured prompt is constructed from retrieved policy events, World Bank indicators, and ILSA metadata. An LLM (Claude or Gemini) produces a qualitative adjustment signal — a direction and rationale — which is combined with the baseline forecast via a weighting scheme.

**Layer 3 — Multi-LLM Ensemble**
Outputs from multiple LLMs (Claude Sonnet, Gemini Pro) are aggregated to reduce provider-specific bias. Ensemble weights are calibrated on the held-out 2018 cycle before final evaluation on 2022.

## Project Structure

```
ILSA_LLMs/
├── data/
│   ├── raw_pdfs/              # Source PDFs (WoS, Scopus, OECD, IEA) — do not modify
│   │   ├── wos/
│   │   ├── scopus/
│   │   ├── oecd/
│   │   └── iea/
│   ├── ilsa_scores/           # PISA, TIMSS, PIRLS score datasets
│   ├── country_indicators/    # World Bank, UNESCO macroeconomic indicators
│   ├── policy_events/         # Education reform timelines by country
│   ├── literature_kb/         # Curated knowledge base (processed from raw_pdfs)
│   ├── processed/             # Intermediate processed files
│   └── exports/               # Final output exports (CSV, JSON)
├── src/
│   ├── data_loaders/          # Modules for ingesting ILSA and indicator data
│   ├── baseline_models/       # ARIMA, Prophet, LSTM implementations
│   ├── llm_augmentation/      # Prompt construction and LLM API integration
│   ├── backtesting/           # Retrospective validation pipeline
│   └── interpretability/      # Explanation generation and visualization
├── notebooks/                 # Exploratory and reporting notebooks
├── prompts/                   # LLM prompt templates
├── outputs/
│   ├── forecasts/             # Model forecast outputs
│   ├── validation/            # Backtesting results and metrics
│   └── figures/               # Charts and visualizations
├── cache/                     # Cached API responses
├── environment.yml            # Conda environment specification
├── .env.example               # Environment variable template
└── .gitignore
```

## Setup Instructions

### Prerequisites

- Conda or Miniconda
- API keys for Anthropic and Google Generative AI (see `.env.example`)

### Steps

1. Create and activate the conda environment:
   ```bash
   conda env create -f environment.yml
   conda activate ilsa-literature-review
   ```
   > Note: the environment name `ilsa-literature-review` is preserved for historical reasons. The project scope is now forecasting.

2. Configure environment variables:
   ```bash
   cp .env.example .env
   # Fill in ANTHROPIC_API_KEY and GOOGLE_API_KEY in .env
   ```

3. Verify key dependencies:
   ```bash
   python -c "import statsmodels, prophet, torch, anthropic; print('Core deps OK')"
   ```

## Validation Strategy

The framework uses retrospective backtesting to evaluate forecast quality:

- **Training window:** all available ILSA cycles up to and including 2018
- **Forecast target:** the 2022 PISA mathematics score cycle
- **Evaluation metrics:** MAE, RMSE, and directional accuracy at the country level
- **Baseline comparison:** classical models alone vs. LLM-augmented pipeline vs. multi-LLM ensemble

This design mirrors a realistic deployment scenario where the 2022 data was genuinely unknown at prediction time, avoiding any form of data leakage.

## References

- Thesis proposal document (internal)
- OECD PISA: [https://www.oecd.org/pisa/](https://www.oecd.org/pisa/)
- IEA TIMSS/PIRLS: [https://www.iea.nl/](https://www.iea.nl/)
- World Bank Open Data: [https://data.worldbank.org/](https://data.worldbank.org/)
