# Phase 3.32 — Isolated reproduction follow-up

**Date:** 2026-10-20  
**Status:** Six model branches independently rerun. Historical tabular v2 and all TimesFM CSV outputs are byte-identical; historical tabular v1 and PISA 2025 tabular are numerically equivalent within floating-point tolerance. The 2,000-draw model-input rerun remains in progress. No original analysis output was overwritten.

## Safe rerun method

The existing scripts have hard-coded output paths. [`run_isolated_reproduction.py`](../../scripts/run_isolated_reproduction.py) executes each source in memory after redirecting only its single `OUT_DIR` assignment to a fresh versioned check folder. It records original and execution-source hashes. Source files and previous result folders remain unchanged. For the 2025 tabular script, execution used Python 3.13 with scikit-learn 1.6.1 and XGBoost 3.0.0, matching the versions in the source run manifest; the historical quadratic script was run in Python 3.14 with its recorded package environment.

## Results

### Historical tabular benchmark v1 — numerical match

The script `run_pisa_tabular_benchmarks.py` was rerun with the Python 3.13.2, scikit-learn 1.6.1, and XGBoost 3.0.0 versions recorded by the original run. Both outputs preserve all 3,120 prediction and 60 metric rows; the metrics CSV is byte-identical. Prediction identifiers and nonnumeric values match, while floating-point serialization differs in model predictions by at most `6.82e-13` and squared errors by at most `2.48e-11`.

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

### TimesFM 2.5 LoRA historical adaptation — exact match

The rerun to `outputs/reproducibility_check_v2/timesfm25_ils_adapter/` used the same Python 3.14.5, PyTorch 2.14.1, Transformers 5.19.0, and PEFT 0.21.2 environment recorded by the source run. The four result CSVs (1,248 adapted predictions, 4,992 all-model predictions, 96 historical metrics, and 24 equal-fold summary rows) match the originals byte-for-byte. Epoch-selection and refit losses for each fold also reproduce the values recorded in the original manifest. Adapters/checkpoints were generated only under the isolated check directory.

### Model-input uncertainty v4 — checkpointed sharded rerun in progress

The first attempted rerun used a Python 3.13 environment with scikit-learn 1.6.1 and XGBoost 3.0.0, which did not match the recorded scikit-learn 1.9.1 and XGBoost 3.4.1. Its partial draws were abandoned and are not used as reproduction evidence. The exact package versions were then installed into a temporary, isolated target directory; Python 3.13.2, NumPy 2.5.3, pandas 3.0.6, scikit-learn 1.9.1, and XGBoost 3.4.1 were verified. Eight deterministic shards using disjoint slices of the original 2,000-seed sequence are running under `outputs/reproducibility_check_v3/model_input_shards/`. Each shard verified nonzero input-SE coverage for all 469 mathematics, 511 reading, and 469 science observations. At the current revision, the first 25-draw batch has completed in every shard (200 of 2,000 draws). No final summary metrics are reported until all shards finish, checkpoint ranges are merged and validated, and the original v4 aggregation code reproduces the source metrics, paired deltas, and winner probabilities.

### PISA 2025 TimesFM retrospective — exact match

The frozen-configuration TimesFM 2.5 adaptation plus TimesFM 3.0 comparator was rerun to `outputs/reproducibility_check_v2/pisa_2025_retrospective_timesfm/`. All three result CSVs (594 TimesFM forecasts, 1,584 combined model predictions, and 24 metrics) match the originals byte-for-byte. This remains an outcome-exposed retrospective comparison.

## Reproduction artifacts and boundaries

- All rerun outputs are isolated below `outputs/reproducibility_check_v2/`.
- `isolated_reproduction_wrapper.json` records the original script hash and the one-line output-directory override for completed runs.
- The initial wrong-version partial model-input outputs are not valid evidence and must not be merged with the exact-version shards.
- The model-input run is not complete until its final manifest and summary outputs exist and the metrics, paired deltas, and winner probabilities are checked against `outputs/pisa_model_input_uncertainty_v4/`.
- The model-input uncertainty run is the remaining long-running model branch in this reproduction pass; TimesFM 2.5 LoRA and 2025 TimesFM were independently reproduced exactly.
