#!/usr/bin/env python3
"""Run joint country-cluster + target-score SE uncertainty for all nine arms."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

import summarize_pisa_joint_score_country_uncertainty as pipeline


ROOT = Path(__file__).resolve().parents[1]
PANELS = ROOT / "outputs/pisa_all_models_quadratic_trend_v1"
OUT = ROOT / "outputs/pisa_joint_score_country_uncertainty_v3"
HIST = PANELS / "historical_all_models_predictions.csv"
P25 = PANELS / "pisa2025_all_models_predictions.csv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {OUT}")
    for path in (HIST, P25, ROOT / "scripts/summarize_pisa_joint_score_country_uncertainty.py"):
        if not path.is_file():
            raise FileNotFoundError(path)
    panel_manifest = json.loads((PANELS / "run_manifest.json").read_text(encoding="utf-8"))
    for filename, digest in panel_manifest["outputs"].items():
        if sha256(PANELS / filename) != digest:
            raise ValueError(f"Unified prediction panel hash mismatch: {filename}")

    pipeline.HIST = HIST
    pipeline.P25 = P25
    pipeline.OUT = OUT
    pipeline.main()

    manifest_path = OUT / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    outputs = [OUT / name for name in manifest["outputs"]]
    historical = pd.read_csv(OUT / "joint_metric_uncertainty.csv")
    p25 = pd.read_csv(OUT / "joint_model_winner_probability.csv")
    if historical.model.nunique() != 9 or p25.loc[p25.evaluation.eq("pisa2025_all66_joint"), "model"].nunique() != 9:
        raise ValueError("Joint uncertainty output does not include all nine methods")
    manifest["orchestration_script"] = str(Path(__file__).resolve().relative_to(ROOT))
    manifest["orchestration_script_sha256"] = sha256(Path(__file__).resolve())
    manifest["analysis_script"] = str((ROOT / "scripts/summarize_pisa_joint_score_country_uncertainty.py").relative_to(ROOT))
    manifest["analysis_script_sha256"] = sha256(ROOT / "scripts/summarize_pisa_joint_score_country_uncertainty.py")
    manifest["unified_prediction_panel_manifest_sha256"] = sha256(PANELS / "run_manifest.json")
    manifest["model_count"] = 9
    manifest["output_sha256"] = {path.name: sha256(path) for path in outputs}
    manifest["checks"] = {
        **manifest["checks"],
        "nine_methods_in_historical_and_2025_evaluations": True,
        "all_output_hashes_recorded": True,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("Verified joint country-cluster + target-score SE uncertainty for nine methods.")


if __name__ == "__main__":
    main()
