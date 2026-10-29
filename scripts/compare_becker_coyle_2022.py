#!/usr/bin/env python3
"""Reconstruct Becker–Coyle's PISA 2022 trend forecasts from their supplement."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import f as f_distribution

ROOT = Path(__file__).resolve().parents[1]
YEARS = (2000, 2003, 2006, 2009, 2012, 2015, 2018)
DOMAINS = ("mathematics", "reading", "science")
DOMAIN_CODES = {"mathematics": "MTH", "reading": "REA", "science": "SCI"}
SOURCE_URL = "https://www.frontiersin.org/api/v4/articles/1734988/file/Data_Sheet_1.zip/1734988_data-sheet_1/2"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def source_forecast(y: np.ndarray) -> tuple[float, float, float, float, str]:
    """Mirror the published R code's raw-year lm and F-test calculation."""
    years = np.asarray([y for y, v in zip(YEARS, y) if pd.notna(v)], dtype=float)
    scores = np.asarray([v for v in y if pd.notna(v)], dtype=float)
    if len(scores) != len(years) or len(scores) < 2:
        return (np.nan, np.nan, np.nan, np.nan, "insufficient")
    x = years
    lin = np.polyfit(x, scores, 1)
    pred_lin = float(np.polyval(lin, 2022))
    if len(scores) < 3:
        return pred_lin, np.nan, np.nan, np.nan, "linear_insufficient"
    quad = np.polyfit(x, scores, 2)
    pred_quad = float(np.polyval(quad, 2022))
    r2_lin = 1 - float(np.square(scores - np.polyval(lin, x)).sum()) / float(np.square(scores - scores.mean()).sum())
    r2_quad = 1 - float(np.square(scores - np.polyval(quad, x)).sum()) / float(np.square(scores - scores.mean()).sum())
    denom = 1 - r2_quad
    fstat = ((r2_quad - r2_lin) / denom) * (len(scores) - 1) if denom else float("inf")
    pval = float(f_distribution.sf(fstat, 1, len(scores) - 1))
    selected = "quadratic" if pval <= 0.05 else "linear"
    return pred_lin, pred_quad, fstat, pval, selected


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--supplement-zip", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "outputs/becker_coyle_comparison_v1")
    args = ap.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.out_dir}")
    args.out_dir.mkdir(parents=True)
    with zipfile.ZipFile(args.supplement_zip) as zf:
        source_bytes = zf.read("DATA/DATA_SAS_PIS.CSV")
        source = pd.read_csv(__import__("io").BytesIO(source_bytes))
    source = source.rename(columns={"IDX_CNT_COM": "country_iso3"})
    rows = []
    for _, row in source.iterrows():
        for domain, code in DOMAIN_CODES.items():
            history = [row.get(f"SAS_PIS_{year % 1000:03d}_{code}_SCL_MEA", np.nan) for year in YEARS]
            pred_lin, pred_quad, fstat, pval, selected = source_forecast(np.asarray(history, dtype=float))
            rows.append({"country_iso3": row["country_iso3"], "domain": domain,
                         "n_context": int(np.isfinite(history).sum()), "prediction_linear": pred_lin,
                         "prediction_quadratic": pred_quad, "quadratic_f": fstat,
                         "quadratic_p": pval, "selected_model": selected,
                         "prediction_selected": pred_quad if selected == "quadratic" else pred_lin,
                         "supplement_observed_2022": row.get(f"SAS_PIS_022_{code}_SCL_MEA", np.nan)})
    reconstructed = pd.DataFrame(rows)
    panel = pd.read_csv(ROOT / "outputs/pisa_tabular_benchmark_v1/historical_predictions.csv")
    panel = panel.query("target_cycle == 2022").copy()
    true_scores = panel[["country_iso3", "domain", "true_score"]].drop_duplicates()
    joined = reconstructed.merge(true_scores, on=["country_iso3", "domain"], how="inner", validate="one_to_one")
    joined["supplement_minus_project_2022_score"] = joined["supplement_observed_2022"] - joined["true_score"]
    eligible = joined.loc[joined["prediction_selected"].notna()].copy()
    eligible["becker_coyle_error"] = eligible["prediction_selected"] - eligible["true_score"]
    eligible["becker_coyle_absolute_error"] = eligible["becker_coyle_error"].abs()
    eligible["becker_coyle_squared_error"] = eligible["becker_coyle_error"] ** 2
    comparison = eligible.merge(panel[["country_iso3", "domain", "model", "prediction"]],
                                on=["country_iso3", "domain"], how="inner", validate="one_to_many")
    comparison["project_error"] = comparison["prediction"] - comparison["true_score"]
    comparison["project_absolute_error"] = comparison["project_error"].abs()
    comparison["project_squared_error"] = comparison["project_error"] ** 2
    rows = []
    for domain, group in comparison.groupby("domain"):
        for model, model_group in group.groupby("model"):
            rows.append({"domain": domain, "model": model, "n_common_countries": int(model_group.country_iso3.nunique()),
                         "becker_coyle_MAE": float(model_group.becker_coyle_absolute_error.mean()),
                         "becker_coyle_RMSE": float(np.sqrt(model_group.becker_coyle_squared_error.mean())),
                         "project_model_MAE": float(model_group.project_absolute_error.mean()),
                         "project_model_RMSE": float(np.sqrt(model_group.project_squared_error.mean())),
                         "paired_MAE_difference_BC_minus_model": float((model_group.becker_coyle_absolute_error-model_group.project_absolute_error).mean())})
    metrics = pd.DataFrame(rows)
    reconstructed.to_csv(args.out_dir / "becker_coyle_reconstructed_forecasts.csv", index=False)
    comparison.to_csv(args.out_dir / "common_country_comparison.csv", index=False)
    metrics.to_csv(args.out_dir / "common_country_metrics.csv", index=False)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "source_url": SOURCE_URL,
        "supplement_sha256": sha256(args.supplement_zip),
        "source_member": "DATA/DATA_SAS_PIS.CSV", "source_member_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "script_sha256": sha256(Path(__file__)), "benchmark_input_sha256": sha256(ROOT / "outputs/pisa_tabular_benchmark_v1/historical_predictions.csv"),
        "method": "Reconstruction of the supplement R code: separate country/domain raw-year linear and quadratic OLS; source F formula and p-value degrees of freedom; quadratic selected at p<=.05; 2022 prediction.",
        "country_domain_matches_before_BC_eligibility": len(joined),
        "supplement_2022_target_max_abs_difference_vs_project": float(joined["supplement_minus_project_2022_score"].abs().max()),
        "eligible_country_domain_forecasts": len(eligible),
        "common_countries_each_domain": {d: int(n) for d, n in eligible.groupby("domain").country_iso3.nunique().items()},
        "python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__,
        "outputs": {p.name: sha256(p) for p in sorted(args.out_dir.glob("*.csv"))},
    }
    (args.out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(metrics.to_string(index=False))
    print(f"Eligible common forecasts={len(eligible)}; countries/domain={manifest['common_countries_each_domain']}; output={args.out_dir}")


if __name__ == "__main__":
    main()
