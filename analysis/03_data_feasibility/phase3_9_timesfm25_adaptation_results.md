# Phase 3.9 — TimesFM 2.5 ILSA Adaptation Results

**Date:** 2026-10-15  
**Status:** Historical temporal validation executed; adapters reload-verified  
**Run manifest:** `outputs/pisa_timesfm25_ils_adapter_v1/run_manifest.json`

## Design

TimesFM 2.5 zero-shot and TimesFM 2.5 LoRA adaptation were evaluated on the same PISA country/domain/target-cycle rows for the four frozen outer targets (2012, 2015, 2018, 2022). The external training corpus pooled eligible country-level score series from PISA, TIMSS, TIMSS Grade 4, and PIRLS. PISA used the corrected, hash-locked model panel; the other ILSA series were selected by their source program and domain to avoid reusing joined values copied onto unrelated rows in `unified_panel.csv`.

Training pairs use expanding histories with at least two prior observations. For each outer target, the latest earlier ILSA cycle with eligible pairs is held out as an inner temporal validation origin. Epoch count is selected by validation MAE, then a fresh adapter is refit on all windows strictly before the outer target for the selected number of epochs. The outer PISA target scores are not included in training windows. PISA 2025 is excluded from all historical runs and model selection.

The same TimesFM 2.5 checkpoint (`google/timesfm-2.5-200m-transformers`, cached revision `5a9806b9b291fad9233b5249d88263f1846304d3`) is used for its zero-shot and adapted arms. LoRA targets `q_proj` and `v_proj` (rank 4); only 409,600 of 231,698,880 parameters are trainable. TimesFM 3.0 is reported separately because its current PyTorch implementation is inference-only. All runs used CPU inference/training, Python 3.14.5, PyTorch 2.14.1, Transformers 5.19.0, and PEFT 0.21.2.

## Equal-fold mean MAE

| Domain | Persistence | XGBoost | Random Forest | Ridge | TimesFM 2.5 zero-shot | TimesFM 2.5 ILSA LoRA | TimesFM 3.0 zero-shot | Linear trend |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Mathematics | **9.33** | 9.56 | 9.81 | 9.96 | 9.87 | 9.77 | 10.63 | 11.30 |
| Reading | **10.16** | 10.38 | 10.76 | 10.83 | 10.38 | 10.27 | 11.49 | 13.33 |
| Science | **8.44** | 8.56 | 9.17 | 9.12 | 8.85 | 8.78 | 9.70 | 10.80 |

The ILSA LoRA arm lowers the equal-fold MAE relative to its matched TimesFM 2.5 zero-shot arm by approximately 0.10 points in mathematics, 0.11 in reading, and 0.07 in science. These are small changes (all under 0.2 points) and do not change the leading method: persistence remains best in every domain. XGBoost remains the strongest learned tabular benchmark overall. The adaptation result is therefore a modest within-family improvement, not evidence that fine-tuning creates a practically meaningful advantage.

Adaptation lowers MAE in 11 of 12 domain-by-fold cells; reading 2018 is essentially tied but marginally worse (increase about 0.005). The largest reductions occur in 2015 (approximately 0.13–0.31 points by domain); differences in 2018 and 2022 are very small.

## Validation and artifact checks

- Four outer folds × three domains × their frozen common-country sets produced 1,248 TimesFM 2.5 predictions (zero-shot and adapted).
- The combined historical table contains all eight methods and 96 fold/domain/model metric rows; the equal-fold table contains 24 domain/model summaries.
- Every prediction is finite and unique on target-cycle/domain/country/model.
- Each saved LoRA adapter was reloaded from disk on top of the cached base checkpoint. Four independent prediction checks matched the in-memory adapter outputs within `0.0001` score points.
- The 2022 PISA mathematics TimesFM 3.0 zero-shot MAE remains 15.83, matching the earlier benchmark.

## Limitations and interpretation

The inner validation origin varies by outer fold: 2011 (77 examples), 2012 (167), 2016 (16), and 2021 (34). The 2016 and 2021 validation sets are small, so epoch selection for the 2018 and 2022 outer folds is noisy. The selected epoch count is the maximum tested (4) for 2012 and 2015, and one epoch for 2018 and 2022. The four-fold results are development/temporal-validation evidence, not an independent final test; repeated countries across folds prevent treating all country-fold rows as independent observations.

The pooled ILSA windows represent country-level national score series, not student microdata. Program, domain, and assessment-year identifiers are not passed to TimesFM; series are treated as ordered assessment-step sequences. Different assessment intervals and differing measurement programs therefore remain a limitation. No p-value or general superiority claim is warranted from these small improvements.

## Next evaluation gate

1. Preserve this as the locked historical adaptation result; do not tune the adapter using PISA 2025.
2. Produce a separate PISA 2025 retrospective benchmark with the adapted TimesFM 2.5, its matched zero-shot version, TimesFM 3.0, persistence, trend, RF, Ridge, and XGBoost on the same eligible countries for all three domains.
3. Label 2025 as outcome-exposed in all tables and text. The prior exposure means it cannot be represented as a pristine blind test.
4. Report the modest historical adaptation effect as inconclusive in practical terms unless the 2025 retrospective result is also favorable under the fixed historical configuration; even then, describe the result as retrospective and exposure-qualified.
