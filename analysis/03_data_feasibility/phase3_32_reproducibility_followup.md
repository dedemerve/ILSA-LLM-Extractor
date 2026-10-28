# Phase 3.32 — Isolated reproduction follow-up

**Date:** 2026-10-21
**Status:** Six model branches independently rerun. Historical tabular v2 and all TimesFM CSV outputs are byte-identical; historical tabular v1 and PISA 2025 tabular are numerically equivalent within floating-point tolerance. The model-input discrepancy is now traced to a mixed-environment original checkpoint series: draws 0–199 for tree models match the older-library rerun, while draws 200–1,999 match the version-matched rerun. A uniform, version-matched 2,000-draw result is available under `outputs/reproducibility_check_v3/pisa_model_input_uncertainty_v4/` and is now the corrected Phase 3.26 result. The original outputs were preserved.

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

### Model-input uncertainty v4 — mixed-environment provenance diagnosed; corrected rerun completed

The first attempted rerun used the older environment Python 3.13.2, NumPy 2.1.3, pandas 2.2.3, scikit-learn 1.6.1, and XGBoost 3.0.0. Its completed 2,000-draw check is under `outputs/reproducibility_check_v2/pisa_model_input_uncertainty_v4/`. A second full rerun used the source-manifest environment Python 3.13.2, NumPy 2.5.3, pandas 3.0.6, scikit-learn 1.9.1, and XGBoost 3.4.1 in an isolated runtime. Eight deterministic shards covered the exact 2,000-seed sequence, with 80 validated checkpoints and nonzero input-SE coverage for all 469 mathematics, 511 reading, and 469 science observations. The original v4 aggregation code generated 120,000 fold/model/draw rows. A checkpoint-by-checkpoint, key-aligned comparison established the provenance pattern: Persistence, LinearTrend, and Ridge match both complete reruns across all 2,000 draws. For RandomForest and XGBoost, the original Phase 3.26 checkpoint rows for draws 0–199 match the older-library rerun exactly; draws 200–1,999 match the version-matched rerun exactly. The original manifest records only the software environment present when the final run manifest was written; its configuration did not bind checkpoint batches to package versions. This is direct evidence that the original tree-model checkpoint series was mixed across environments, not stochastic variation. The corrected, uniform-environment estimates in the v3 output folder replace the old tree-model summaries in Phase 3.26 and the final synthesis. All source outputs remain intact.

### PISA 2025 TimesFM retrospective — exact match

The frozen-configuration TimesFM 2.5 adaptation plus TimesFM 3.0 comparator was rerun to `outputs/reproducibility_check_v2/pisa_2025_retrospective_timesfm/`. All three result CSVs (594 TimesFM forecasts, 1,584 combined model predictions, and 24 metrics) match the originals byte-for-byte. This remains an outcome-exposed retrospective comparison.

## Reproduction artifacts and boundaries

- Standard rerun outputs are isolated below `outputs/reproducibility_check_v2/`; the completed model-input rerun and its eight shards are under `outputs/reproducibility_check_v3/`.
- `isolated_reproduction_wrapper.json` records the original script hash and the one-line output-directory override for completed runs.
- The initial wrong-version partial model-input outputs are not valid evidence and must not be merged with the exact-version shards.
- Model-input uncertainty is complete and the discrepancy is diagnosed. The original mixed-environment output remains preserved for audit; use the uniform, version-matched corrected output under `outputs/reproducibility_check_v3/pisa_model_input_uncertainty_v4/` for Phase 3.26 results.
- TimesFM 2.5 LoRA and PISA 2025 TimesFM were independently reproduced exactly.
