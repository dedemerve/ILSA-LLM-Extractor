# Phase 3.32 — Isolated reproduction follow-up

**Date:** 2026-10-20  
**Status:** Three model branches independently rerun; TimesFM 3.0 historical zero-shot is byte-identical. The 2,000-draw model-input rerun is in progress in eight deterministic shards with the recorded package versions. No original analysis output was overwritten.

## Safe rerun method

The existing scripts have hard-coded output paths. [`run_isolated_reproduction.py`](../../scripts/run_isolated_reproduction.py) executes each source in memory after redirecting only its single `OUT_DIR` assignment to a fresh versioned check folder. It records original and execution-source hashes. Source files and previous result folders remain unchanged. For the 2025 tabular script, execution used Python 3.13 with scikit-learn 1.6.1 and XGBoost 3.0.0, matching the versions in the source run manifest; the historical quadratic script was run in Python 3.14 with its recorded package environment.

## Results

### Historical tabular benchmark v2 — exact match

The script `run_pisa_quadratic_trend_benchmark.py` was rerun to `outputs/reproducibility_check_v2/pisa_tabular_benchmark_v2/`. All three result CSVs match the originals byte-for-byte:

| Output | Result |
|---|---|
| `historical_predictions.csv` | Exact SHA-256 match |
| `historical_metrics.csv` | Exact SHA-256 match |
| `trend_selection_diagnostics.csv` | Exact SHA-256 match |

### PISA 2025 tabular retrospective — numerical match

The script `run_pisa_2025_retrospective_tabular.py` was rerun to `outputs/reproducibility_check_v2/pisa_2025_retrospective_tabular/`. Both result tables retain the same 990 prediction rows / 15 metric rows and identical nonnumeric identifiers and keys. They are not byte-identical because of floating-point serialization differences: maximum absolute prediction difference is `3.98e-13`, maximum numeric difference in the full predictions table is `2.50e-11` (squared-error column), and maximum metric difference is `5.78e-14`. These are numerical-equivalence results, not byte equality.

### TimesFM 3.0 historical zero-shot — exact match

The original `timesfm3` evaluator was available in the Python 3.14 runtime with the cached model. The rerun to `outputs/reproducibility_check_v2/pisa_timesfm_historical_zeroshot/` produced all five result CSVs with byte-identical SHA-256 hashes to `outputs/pisa_timesfm_historical_zeroshot_v2/`: 624 TimesFM forecasts, 3,744 combined model predictions, 72 fold/domain metrics, and both 18-row summaries. The original outputs remain unchanged.

### Model-input uncertainty v4 — checkpointed sharded rerun in progress

The first attempted rerun used a Python 3.13 environment with scikit-learn 1.6.1 and XGBoost 3.0.0, which did not match the recorded scikit-learn 1.9.1 and XGBoost 3.4.1. Its partial draws were abandoned and are not used as reproduction evidence. The exact package versions were then installed into a temporary, isolated target directory; Python 3.13.2, NumPy 2.5.3, pandas 3.0.6, scikit-learn 1.9.1, and XGBoost 3.4.1 were verified. Eight deterministic shards using disjoint slices of the original 2,000-seed sequence are running under `outputs/reproducibility_check_v3/model_input_shards/`. Each shard verified nonzero input-SE coverage for all 469 mathematics, 511 reading, and 469 science observations. No final summary metrics are reported until all shards finish, checkpoint ranges are merged and validated, and the original v4 aggregation code reproduces the source metrics, paired deltas, and winner probabilities.

## Reproduction artifacts and boundaries

- All rerun outputs are isolated below `outputs/reproducibility_check_v2/`.
- `isolated_reproduction_wrapper.json` records the original script hash and the one-line output-directory override for completed runs.
- The initial wrong-version partial model-input outputs are not valid evidence and must not be merged with the exact-version shards.
- The model-input run is not complete until its final manifest and summary outputs exist and the metrics, paired deltas, and winner probabilities are checked against `outputs/pisa_model_input_uncertainty_v4/`.
- This pass does not establish reproducibility for TimesFM 2.5 LoRA adaptation or the 2025 TimesFM arm. Their model runtime/checkpoint requirements remain separate open items.
