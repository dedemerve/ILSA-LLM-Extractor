# Phase 3.32 — Isolated reproduction follow-up

**Date:** 2026-10-20  
**Status:** Two model branches independently rerun; TimesFM execution blocked by missing runtime module; 2,000-draw model-input rerun checkpointed and in progress at this report revision. No original analysis output was overwritten.

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

### TimesFM 3.0 historical zero-shot — blocked in current environment

The rerun stops at import with `ModuleNotFoundError: No module named 'timesfm3'`. The model weights are cached locally, but the Python evaluator package used by the original run is not installed or included in the repository. No package installation or model download was attempted. The original TimesFM output remains untouched and is not independently reproduced by this check.

### Model-input uncertainty v4 — checkpointed run in progress

The exact v4 script and input hashes were accepted, and its 2,000-draw, 25-draw-batch run began in `outputs/reproducibility_check_v2/pisa_model_input_uncertainty_v4/`. It verified nonzero input-SE coverage for all 469 mathematics, 511 reading, and 469 science historical observations. The first checkpoint contains draws 0–24 (1,500 fold/model metric rows); the process has started the next batch. This full rerun is computationally intensive; no summary metrics are reported until all 2,000 draws finish and the final CSVs can be compared with the v4 source outputs.

## Reproduction artifacts and boundaries

- All rerun outputs are isolated below `outputs/reproducibility_check_v2/`.
- `isolated_reproduction_wrapper.json` records the original script hash and the one-line output-directory override for completed runs.
- The TimesFM blocker is a missing execution dependency, not evidence of a numerical mismatch.
- The model-input run is not complete until its final manifest and summary outputs exist and the metrics, paired deltas, and winner probabilities are checked against `outputs/pisa_model_input_uncertainty_v4/`.
- This pass does not establish reproducibility for TimesFM 2.5 LoRA adaptation or the 2025 TimesFM arm. Their model runtime/checkpoint requirements remain separate open items.
