# Phase 3.32 — Isolated reproduction follow-up

**Date:** 2026-10-20  
**Status:** Six model branches independently rerun. Historical tabular v2 and all TimesFM CSV outputs are byte-identical; historical tabular v1 and PISA 2025 tabular are numerically equivalent within floating-point tolerance. The full 2,000-draw model-input rerun completed, but its summaries are not an exact numerical reproduction of the source; this remains a documented mismatch. No original analysis output was overwritten.

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

### Model-input uncertainty v4 — full rerun completed; numerical mismatch

The first attempted rerun used a Python 3.13 environment with scikit-learn 1.6.1 and XGBoost 3.0.0; its partial draws were abandoned. The completed rerun used the source-recorded Python 3.13.2, NumPy 2.5.3, pandas 3.0.6, scikit-learn 1.9.1, and XGBoost 3.4.1 in an isolated temporary environment. Eight deterministic shards covered the exact 2,000-seed sequence, with 80 validated checkpoints and nonzero input-SE coverage for all 469 mathematics, 511 reading, and 469 science observations. The original v4 aggregation script generated 120,000 fold/model/draw rows. Source and rerun row counts and keys match, but the aggregate CSVs are not identical. Across the 15-row metric table, maximum absolute differences are 0.01360 MAE, 0.00537 for q2.5, 0.01077 for the median, 0.01457 for q97.5, and 0.00134 for SD. Across 12 paired comparisons, maximum differences are 0.01360 in point delta, 0.02624 in q2.5, 0.01840 in q97.5, and 0.0005 in the estimated probability of beating Persistence; winner probabilities differ by at most 0.0005. Differences are concentrated in RF/XGBoost summaries. Thus this is a complete rerun with a numerical mismatch, not a successful exact reproduction. The original Phase 3.26 outputs remain the reported source results; do not replace them with this rerun. The mismatch cause has not yet been isolated.

### PISA 2025 TimesFM retrospective — exact match

The frozen-configuration TimesFM 2.5 adaptation plus TimesFM 3.0 comparator was rerun to `outputs/reproducibility_check_v2/pisa_2025_retrospective_timesfm/`. All three result CSVs (594 TimesFM forecasts, 1,584 combined model predictions, and 24 metrics) match the originals byte-for-byte. This remains an outcome-exposed retrospective comparison.

## Reproduction artifacts and boundaries

- Standard rerun outputs are isolated below `outputs/reproducibility_check_v2/`; the completed model-input rerun and its eight shards are under `outputs/reproducibility_check_v3/`.
- `isolated_reproduction_wrapper.json` records the original script hash and the one-line output-directory override for completed runs.
- The initial wrong-version partial model-input outputs are not valid evidence and must not be merged with the exact-version shards.
- Model-input uncertainty is complete as a rerun, but the numerical mismatch needs diagnosis before claiming independent reproduction. Do not substitute these outputs for `outputs/pisa_model_input_uncertainty_v4/`.
- TimesFM 2.5 LoRA and PISA 2025 TimesFM were independently reproduced exactly.
