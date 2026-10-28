# Phase 3.8 — Historical TimesFM Zero-Shot Comparison

**Date:** 2026-10-15  
**Status:** Executed and output integrity checked  
**Run:** `outputs/pisa_timesfm_historical_zeroshot_v2/run_manifest.json`

## Scope and design

TimesFM 3.0 zero-shot was evaluated on the same four rolling-origin PISA folds, three domains, and frozen common-country sets as the first tabular benchmark. The target is the country-level mean score, one PISA assessment cycle ahead. Each country has at least three prior observations. Fold sizes are 40, 51, 57, and 60 countries. PISA 2025 was excluded from all forecasts and model selection because its outcomes are already exposed in this project.

Inputs were the SHA-256-verified `pisa_forecasting_panel_v2.csv` and `pisa_historical_folds_v1.json`. TimesFM used checkpoint `google/timesfm-3.0-pytorch`, cached revision `43046b85ec22d584a13f8098c2ed39c889e129c2`, distribution `timesfm==3.0.2`, PyTorch 2.14.1, Python 3.14.5, and CPU inference. The local MPS attempt was unsupported by this machine's macOS version; CPU inference was used. The checkpoint was loaded locally (`local_files_only=True`).

PISA cycles are supplied as ordered values; TimesFM receives a one-step horizon and does not receive calendar gaps as covariates. In particular, the 2018–2022 interval is four years, whereas other adjacent cycles in these folds are generally three. This is a known design limitation for the zero-shot arm.

## Equal-fold mean MAE

Each target fold receives equal weight in this summary.

| Domain | Persistence | Linear trend | Ridge | Random Forest | XGBoost | TimesFM 3.0 zero-shot |
|---|---:|---:|---:|---:|---:|---:|
| Mathematics | **9.33** | 11.30 | 9.96 | 9.81 | 9.56 | 10.63 |
| Reading | **10.16** | 13.33 | 10.83 | 10.76 | 10.38 | 11.49 |
| Science | **8.44** | 10.80 | 9.12 | 9.17 | 8.56 | 9.70 |

Persistence has the lowest equal-fold MAE in all three domains. XGBoost is the strongest learned tabular model in the equal-fold summary. TimesFM beats the linear-trend baseline, but not persistence, Ridge, Random Forest, or XGBoost in any domain's equal-fold summary.

## Fold-specific TimesFM and persistence MAE

| Target cycle | Mathematics: Persistence / TimesFM | Reading: Persistence / TimesFM | Science: Persistence / TimesFM |
|---:|---:|---:|---:|
| 2012 | 8.29 / 10.01 | 9.60 / 10.60 | 6.95 / 8.62 |
| 2015 | 8.18 / 9.23 | 10.43 / 12.01 | 10.87 / 12.01 |
| 2018 | 6.35 / 7.44 | 8.25 / 9.92 | 7.11 / 9.13 |
| 2022 | 14.48 / 15.83 | 12.35 / 13.45 | 8.83 / 9.05 |

The 2022 mathematics TimesFM MAE of 15.83 reproduces the earlier reported result on this canonical panel. Across all folds, TimesFM does not outperform persistence in any domain-fold cell. The 2022 cycle remains a pandemic-era disruption and should not be treated as the only evidence about model behavior.

## Verification and interpretation limits

- 624 finite TimesFM predictions were written: 4 folds × 3 domains × fold-specific common countries.
- Prediction keys are unique; target values match the corresponding tabular benchmark keys.
- The combined file contains all six methods: Persistence, LinearTrend, Ridge, RandomForest, XGBoost, and TimesFM3 zero-shot.
- Historical performance is temporal validation used during development, not a pristine final test. No inferential significance claim is made.
- Results are country-level aggregate forecasts, not predictions of individual students' scores.
- The output files, model revision, software versions, CPU device, fold hashes, and panel hashes are recorded in the run manifest.

## Adaptation gate

This result does not support a claim that TimesFM 3.0 zero-shot is better than the simple baselines. It does not by itself rule out ILSA adaptation, but adaptation must be tested without selecting configurations on PISA 2025. The installed/official TimesFM 3.0 PyTorch implementation is inference-only; a weight-adaptation experiment therefore needs a trainable checkpoint such as TimesFM 2.5 and a matched TimesFM 2.5 zero-shot comparator, or a clearly labeled post-hoc calibration arm. The current fine-tuning scripts are not suitable evidence: they lack temporal validation/early stopping, use zero-padding for short contexts, do not enforce minimum evaluation context correctly, and use a different TimesFM version from the zero-shot arm.

Before fitting adapters, construct outer-fold training windows with targets strictly earlier than each outer test fold, reserve only earlier temporal origins for inner validation, audit context/horizon lengths and series identities, and compare adapted TimesFM 2.5 with its own zero-shot version plus all existing baselines on identical countries. If the earliest folds do not have sufficient training examples, document that limitation rather than manufacturing random overlapping windows as independent observations.
