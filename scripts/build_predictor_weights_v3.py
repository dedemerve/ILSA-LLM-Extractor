#!/usr/bin/env python3
"""
Knowledge-base predictor weights v3 (single source, single formula).

Source: outputs/stage5/evidence_matrix.csv (LLM evidence matrix, 80 studies).
Formula: build_literature_priority.compute_weights (imported, not copied):
  w_F = log1p(n_studies) / max log1p(n_studies)
  w_P = n_programs / 8, w_C = n_cycles / 8, w_N = n_countries / 50   (clipped to [0, 1])
  w_E = 1 + 0.3 * frac_predictive
  W_geo = geomean(clip(w_F, w_P, w_C, w_N; min 0.01)) * w_E
  W_j   = round(W_geo / max W_geo, 4)
  w     = round(W_j * forecastable / max(W_j * forecastable), 4)

Replaces predictor_weights_v2.csv, whose w_norm column was hand-copied from
literature_priority.csv and did not follow from its own F/D/C/E columns.
Constructs with zero studies are dropped (weight undefined).

Outputs (refuses to overwrite):
  outputs/stage5/predictor_weights_v3.csv
  outputs/stage5/predictor_weights_v3_manifest.json
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import build_literature_priority as blp  # noqa: E402

EVIDENCE = ROOT / "outputs" / "stage5" / "evidence_matrix.csv"
OUT_CSV = ROOT / "outputs" / "stage5" / "predictor_weights_v3.csv"
OUT_MANIFEST = ROOT / "outputs" / "stage5" / "predictor_weights_v3_manifest.json"

COLUMNS = [
    "canonical_construct", "n_studies", "n_rows", "n_programs", "n_cycles",
    "n_countries", "frac_predictive", "forecastable",
    "w_F", "w_P", "w_C", "w_N", "w_E", "W_geo", "W_j", "w",
]


def recompute_w(df: pd.DataFrame) -> pd.Series:
    """Recompute the final weight from the stored component columns."""
    comps = df[["w_F", "w_P", "w_C", "w_N"]].to_numpy(dtype=float).clip(0.01, None)
    w_geo = np.exp(np.log(comps).mean(axis=1)) * df["w_E"].to_numpy(dtype=float)
    w_j = np.round(w_geo / w_geo.max(), 4)
    fc = np.where(df["forecastable"].astype(bool), w_j, 0.0)
    return pd.Series(np.round(fc / fc.max(), 4), index=df.index)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    for path in (OUT_CSV, OUT_MANIFEST):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")

    audit = blp.compute_provenance_audit(blp.load_evidence())
    weights = blp.compute_weights(audit)
    dropped = weights.loc[weights["n_studies"] == 0, "canonical_construct"].tolist()
    weights = weights[weights["n_studies"] > 0].copy()
    weights = weights.rename(columns={"W_j_forecast": "w"})[COLUMNS]
    weights = weights.sort_values("w", ascending=False).reset_index(drop=True)

    # The stored components must reproduce the stored weight exactly.
    recomputed = recompute_w(weights)
    if not np.allclose(recomputed.to_numpy(), weights["w"].to_numpy(), atol=1e-12):
        raise AssertionError("w is not reproducible from its component columns")
    if weights["w"].isna().any():
        raise AssertionError("undefined weight in output")

    weights.to_csv(OUT_CSV, index=False)
    git_head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True).stdout.strip()
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_head": git_head,
        "input": str(EVIDENCE.relative_to(ROOT)),
        "input_sha256": sha256(EVIDENCE),
        "output": str(OUT_CSV.relative_to(ROOT)),
        "output_sha256": sha256(OUT_CSV),
        "formula_source": "scripts/build_literature_priority.py::compute_weights",
        "dropped_zero_study_constructs": dropped,
        "reproducibility_check": "w recomputed from w_F,w_P,w_C,w_N,w_E,forecastable: exact match",
    }
    OUT_MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(weights[["canonical_construct", "n_studies", "forecastable", "W_j", "w"]].to_string(index=False))
    print(f"\nDropped (0 studies): {dropped}")
    print(f"Saved: {OUT_CSV.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
