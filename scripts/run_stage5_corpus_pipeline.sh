#!/usr/bin/env bash
# Full Stage 5 corpus → evidence → weights → accuracy refresh (no LLM API).
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== Stage 5 corpus pipeline =="

python3 scripts/build_evidence_matrix.py
python3 scripts/augment_effect_sizes_from_corpus.py
python3 scripts/compute_predictor_weights_v2.py
python3 scripts/build_literature_priority.py
python3 scripts/run_icils_piaac_holdout.py
python3 scripts/build_per_country_accuracy.py

echo "Done. Key outputs:"
wc -l outputs/stage5/effect_sizes.csv outputs/stage5/evidence_matrix.csv \
  outputs/stage5/literature_priority.csv outputs/stage5/per_country_accuracy.csv
