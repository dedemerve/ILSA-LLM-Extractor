#!/usr/bin/env python3
"""
Merge tabular benchmark and LOCO results into unified RQ1 and RQ2 outputs.

RQ1: Compare RF, XGBoost, Ridge, Persistence, LinearTrend (tabular benchmark)
     against M0-M3 (LOCO Ridge-based models) and TimesFM where available,
     across all ILSA programs and domains.

RQ2: Quantify ΔMae(M0 - M1) per program-domain, where
     M0 = unweighted Ridge (no literature weights)
     M1 = literature-weighted Ridge (integrates structured knowledge base)

Output:
  outputs/ilsa_benchmark_merged_v2/
    rq1_comparison.csv   — MAE/RMSE per model×program×domain, unified
    rq2_m0_vs_m1.csv     — ΔMae, ΔRMSE, and rank per program-domain
    manifest.json
"""
from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT   = Path(__file__).resolve().parents[1]
OUT    = ROOT / "outputs" / "ilsa_benchmark_merged_v2"
TIMESFM_PISA_SUMM = ROOT / "outputs" / "pisa_timesfm25_ils_adapter_v1" / "equal_fold_summary.csv"
TIMESFM_MULTI_SUMM = ROOT / "outputs" / "ilsa_timesfm_all_programs_v1" / "equal_fold_summary.csv"

LOCO_CSV        = ROOT / "outputs" / "stage5" / "loco_results.csv"
PIAAC_LOCO_CSV  = ROOT / "outputs" / "stage5" / "piaac_loco_results.csv"
TAB_SUMM        = ROOT / "outputs" / "ilsa_tabular_benchmark_v1" / "summary.csv"

# TIMSS (G8) uses old label "TIMSS" in loco_results.csv
LOCO_LABEL_MAP = {"TIMSS": "TIMSS_G8"}

# Models from each source
LOCO_MODELS   = {"M0": "Ridge_M0", "M0A": "Ridge_M0A",
                 "M1": "Ridge_M1", "M2": "Persistence_M2", "M3": "AR1_M3"}
TABULAR_SKIP  = {"Ridge"}    # Ridge already in LOCO as M0; skip duplication


def _aggregate_loco(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate raw LOCO rows to program-domain-model level."""
    return (
        df.groupby(["program", "domain", "model_label"], as_index=False)
          .agg(n_folds=("test_cycle", "nunique"),
               n_preds=("n_test",      "sum"     ),
               MAE=    ("MAE",          "mean"    ),
               RMSE=   ("RMSE",         lambda v: float(np.sqrt((v**2).mean()))))
          .rename(columns={"model_label": "model"})
          .assign(source="LOCO")
    )


def load_loco() -> pd.DataFrame:
    df = pd.read_csv(LOCO_CSV)
    df["program"] = df["program"].replace(LOCO_LABEL_MAP)
    df = df[df["model"].isin(LOCO_MODELS)].copy()
    df["model_label"] = df["model"].map(LOCO_MODELS)

    # Merge PIAAC-specific LOCO results if available
    if PIAAC_LOCO_CSV.is_file():
        piaac_df = pd.read_csv(PIAAC_LOCO_CSV)
        piaac_df = piaac_df[piaac_df["model"].isin(LOCO_MODELS)].copy()
        piaac_df["model_label"] = piaac_df["model"].map(LOCO_MODELS)
        # Drop any existing PIAAC rows from main loco_results (old Persistence-only rows)
        df = df[df["program"] != "PIAAC"]
        df = pd.concat([df, piaac_df], ignore_index=True)

    return _aggregate_loco(df)


def load_tabular() -> pd.DataFrame:
    df = pd.read_csv(TAB_SUMM)
    df = df[~df["model"].isin(TABULAR_SKIP)].copy()
    df["source"] = "tabular_benchmark"
    return df[["program", "domain", "model", "n_folds", "n_preds", "MAE", "RMSE", "source"]]


def load_timesfm_pisa() -> pd.DataFrame:
    """Load PISA TimesFM comparison (zero-shot, LoRA, TimesFM 3.0)."""
    df = pd.read_csv(TIMESFM_PISA_SUMM)
    names = {
        "TimesFM2_5_zero_shot": "TimesFM_v25_zeroshot",
        "TimesFM2_5_ILSA_LoRA": "TimesFM_v25_LoRA",
        "TimesFM3_zero_shot":   "TimesFM_v30_zeroshot",
    }
    df = df[df["model"].isin(names)].copy()
    return pd.DataFrame({
        "program": "PISA",
        "domain":  df["domain"],
        "model":   df["model"].map(names),
        "n_folds": df["folds"].astype(int),
        "n_preds": np.nan,
        "MAE":     df["equal_fold_MAE"],
        "RMSE":    df["equal_fold_RMSE"],
        "source":  "timesfm_pisa",
    })


def load_timesfm_multi() -> pd.DataFrame:
    """Load multi-program TimesFM 2.5 zero-shot results (non-PISA)."""
    if not TIMESFM_MULTI_SUMM.is_file():
        return pd.DataFrame(
            columns=["program", "domain", "model", "n_folds", "n_preds", "MAE", "RMSE", "source"]
        )
    df = pd.read_csv(TIMESFM_MULTI_SUMM)
    names = {"TimesFM2_5_zero_shot": "TimesFM_v25_zeroshot"}
    df = df[df["model"].isin(names)].copy()
    return pd.DataFrame({
        "program": df["program"],
        "domain":  df["domain"],
        "model":   df["model"].map(names),
        "n_folds": df["folds"].astype(int),
        "n_preds": np.nan,
        "MAE":     df["equal_fold_MAE"],
        "RMSE":    df["equal_fold_RMSE"],
        "source":  "timesfm_multi",
    })


def build_rq1(loco: pd.DataFrame, tab: pd.DataFrame,
              timesfm: pd.DataFrame, timesfm_multi: pd.DataFrame) -> pd.DataFrame:
    combined = pd.concat([loco, tab, timesfm, timesfm_multi], ignore_index=True)
    combined = combined.sort_values(["program", "domain", "MAE"]).reset_index(drop=True)
    # Add rank within each program-domain group
    combined["rank_in_group"] = (
        combined.groupby(["program", "domain"])["MAE"]
                .rank(method="min", ascending=True)
                .astype(int)
    )
    return combined


def build_rq2(loco_raw: pd.DataFrame | None = None) -> pd.DataFrame:
    """ΔMae = MAE(M0) - MAE(M1); positive = M1 improved over M0."""
    df = pd.read_csv(LOCO_CSV)
    df["program"] = df["program"].replace(LOCO_LABEL_MAP)

    rows = []
    for (prog, dom), g in df.groupby(["program", "domain"]):
        m0 = g[g["model"] == "M0"]
        m1 = g[g["model"] == "M1"]
        if m0.empty or m1.empty:
            continue
        mae0  = m0["MAE"].mean()
        mae1  = m1["MAE"].mean()
        rmse0 = float(np.sqrt((m0["RMSE"]**2).mean()))
        rmse1 = float(np.sqrt((m1["RMSE"]**2).mean()))
        n_folds = int(m0["test_cycle"].nunique())
        rows.append({
            "program":    prog,
            "domain":     dom,
            "MAE_M0":     round(mae0,  4),
            "MAE_M1":     round(mae1,  4),
            "delta_MAE":  round(mae0 - mae1, 4),   # >0 = M1 is better
            "RMSE_M0":    round(rmse0, 4),
            "RMSE_M1":    round(rmse1, 4),
            "delta_RMSE": round(rmse0 - rmse1, 4),
            "pct_improvement_MAE": round(100 * (mae0 - mae1) / mae0, 2),
            "n_folds":    n_folds,
        })

    rq2 = pd.DataFrame(rows).sort_values("delta_MAE", ascending=False).reset_index(drop=True)
    rq2["M1_better"] = rq2["delta_MAE"] > 0
    rq2["integration_type"] = "predictor_weight_scaling"
    return rq2


def build_rq2_timesfm() -> pd.DataFrame:
    """Descriptive PISA LoRA-vs-zero-shot comparison, separate from KB-weight RQ2."""
    df = pd.read_csv(TIMESFM_PISA_SUMM)
    pivot = df[df["model"].isin(["TimesFM2_5_zero_shot", "TimesFM2_5_ILSA_LoRA"])]
    pivot = pivot.pivot(index="domain", columns="model", values="equal_fold_MAE")
    pivot = pivot.rename(columns={
        "TimesFM2_5_zero_shot": "MAE_zeroshot",
        "TimesFM2_5_ILSA_LoRA": "MAE_LoRA",
    }).reset_index()
    pivot["delta_MAE"] = pivot["MAE_zeroshot"] - pivot["MAE_LoRA"]
    pivot["pct_improvement"] = 100 * pivot["delta_MAE"] / pivot["MAE_zeroshot"]
    pivot["LoRA_better"] = pivot["delta_MAE"] > 0
    pivot["interpretation"] = "PISA-only model adaptation; not the structured predictor-weight KB intervention"
    return pivot


def build_rq1_summary(rq1: pd.DataFrame) -> pd.DataFrame:
    return (rq1.groupby("model", as_index=False)
            .agg(n_program_domains=("program", "size"),
                 n_first_rank=("rank_in_group", lambda s: int((s == 1).sum())),
                 mean_rank=("rank_in_group", "mean"),
                 mean_MAE=("MAE", "mean"),
                 sources=("source", lambda s: ";".join(sorted(set(s)))))
            .sort_values(["mean_rank", "mean_MAE"]).reset_index(drop=True))


def build_rq2_summary(rq2: pd.DataFrame) -> dict:
    better = rq2[rq2["M1_better"]]
    worst_i = rq2["pct_improvement_MAE"].idxmax()
    worst = rq2.loc[worst_i]
    return {
        "n_program_domains_total": int(len(rq2)),
        "n_M1_improves": int(rq2["M1_better"].sum()),
        "n_M1_equal_or_worse": int((~rq2["M1_better"]).sum()),
        "mean_delta_MAE_where_M1_better": round(float(better["delta_MAE"].mean()), 4) if len(better) else None,
        "mean_pct_improvement_where_M1_better": round(float(better["pct_improvement_MAE"].mean()), 2) if len(better) else None,
        "max_pct_improvement": round(float(worst["pct_improvement_MAE"]), 2),
        "max_pct_improvement_program_domain": {"program": worst["program"], "domain": worst["domain"]},
        "knowledge_base_source": "outputs/stage5/predictor_weights_v2.csv (literature-derived predictor weights)",
        "integration_mechanism": "sqrt(w_j) feature scaling in Ridge regression (M1 vs M0)",
        "scope_note": "Observational benchmark comparison; reports predictive differences and does not establish causal improvement.",
    }


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite: {OUT}")

    loco          = load_loco()
    tab           = load_tabular()
    timesfm       = load_timesfm_pisa()
    timesfm_multi = load_timesfm_multi()
    rq1      = build_rq1(loco, tab, timesfm, timesfm_multi)
    rq2      = build_rq2(loco)
    rq2_timesfm = build_rq2_timesfm()
    rq1_summary = build_rq1_summary(rq1)
    rq2_summary = build_rq2_summary(rq2)

    OUT.mkdir(parents=True, exist_ok=False)

    rq1.to_csv(OUT / "rq1_comparison.csv",  index=False, float_format="%.4f")
    rq2.to_csv(OUT / "rq2_m0_vs_m1.csv",    index=False, float_format="%.4f")
    rq1_summary.to_csv(OUT / "rq1_summary_by_model.csv", index=False, float_format="%.4f")
    rq2_timesfm.to_csv(OUT / "rq2_timesfm_lora_vs_zeroshot.csv", index=False, float_format="%.4f")
    (OUT / "rq2_summary.json").write_text(json.dumps(rq2_summary, indent=2) + "\n", encoding="utf-8")

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python":      platform.python_version(),
        "inputs": {
            "loco_results":               str(LOCO_CSV.relative_to(ROOT)),
            "tabular_benchmark_summary":  str(TAB_SUMM.relative_to(ROOT)),
            "timesfm_pisa_equal_fold_summary":  str(TIMESFM_PISA_SUMM.relative_to(ROOT)),
            "timesfm_multi_equal_fold_summary": (
                str(TIMESFM_MULTI_SUMM.relative_to(ROOT))
                if TIMESFM_MULTI_SUMM.is_file()
                else "not_available"
            ),
        },
        "outputs": {
            "rq1_comparison": "rq1_comparison.csv",
            "rq2_m0_vs_m1":   "rq2_m0_vs_m1.csv",
            "rq1_summary_by_model": "rq1_summary_by_model.csv",
            "rq2_timesfm_lora_vs_zeroshot": "rq2_timesfm_lora_vs_zeroshot.csv",
            "rq2_summary": "rq2_summary.json",
        },
        "notes": {
            "TIMSS_label": "LOCO source labels TIMSS G8 as 'TIMSS'; normalized to 'TIMSS_G8' here.",
            "RQ1_models":  "LOCO and tabular models across available program-domains, plus TimesFM 2.5/3.0 variants for PISA (LoRA+zero-shot) and TimesFM 2.5 zero-shot for TIMSS_G8/TIMSS_G4/PIRLS/ICCS/ICILS. Metrics are aggregated from different benchmark designs; compare with that limitation in mind.",
            "RQ2_delta":   "delta_MAE = MAE(M0) - MAE(M1); positive means M1 (lit-weighted Ridge) improves on M0 (unweighted Ridge).",
            "TimesFM_RQ2": "LoRA is reported as a separate PISA adaptation comparison and is not treated as the structured predictor-weight knowledge-base intervention.",
        },
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print("=== RQ1: Best model per program-domain ===")
    best = rq1[rq1["rank_in_group"] == 1][["program","domain","model","MAE","RMSE","source"]]
    print(best.to_string(index=False))

    print("\n=== RQ2: M0 vs M1 (literature-weighted Ridge) ===")
    print(rq2[["program","domain","MAE_M0","MAE_M1","delta_MAE","pct_improvement_MAE","M1_better"]].to_string(index=False))
    print(f"\nM1 improves over M0 in {rq2['M1_better'].sum()}/{len(rq2)} program-domain pairs.")

    print(f"\nSaved to: {OUT.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
