#!/usr/bin/env python3
"""
ICILS LOCO Forecasting (M0 / M1)

Mevcut run_loco_forecasting.py altyapısını ICILS enriched panel üzerinde
çalıştırır. icils_enriched_panel.csv'yi ESTIMATES_CSV olarak kullanır.

ICILS LOCO design:
  Cycles: 2013, 2018, 2023
  train pair (2013→2018) → test 2018
  train pairs (2013→2018, 2018→2023) → test 2023
  → 2 eligible folds (pair-based training)

Canonical construct eşlemeleri:
  SES_COMPOSITE        → S_NISB / S_HISEI composite
  PARENTAL_EDUCATION   → S_HISCED
  HOME_LITERACY_ACTIVITIES → S_HOMLIT
  ICILS_computer_literacy → target (not weighted; W_j=1.0)

Çıktı: outputs/stage5/icils_loco_results.csv
"""
from __future__ import annotations

import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
import run_loco_forecasting as loco

STAGE5_DIR = PROJECT_ROOT / "outputs" / "stage5"

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ICILS covariate → canonical construct (for lit weight lookup)
_ICILS_TO_CANONICAL: dict[str, str] = {
    "SES_COMPOSITE":          "SES_COMPOSITE",         # not in weights → W=1.0
    "PARENTAL_EDUCATION":     "PARENTAL_EDU",           # BSDGEDUP canonical → 0.4559
    "HOME_LITERACY_ACTIVITIES": "HOME_LITERACY_ACTIVITIES",  # S_HOMLIT → check weights
}


def icils_feat_weight(f: str, weights: dict) -> float:
    """Map ICILS feature name to literature weight."""
    # lag_ prefix
    raw = f.replace("lag_", "")
    canonical = _ICILS_TO_CANONICAL.get(raw, raw)
    return weights.get(canonical, 1.0)


def run_icils_loco() -> pd.DataFrame:
    panel = pd.read_csv(STAGE5_DIR / "icils_enriched_panel.csv")
    panel["cycle"] = panel["cycle"].astype(int)

    # Load weights
    wdf = pd.read_csv(STAGE5_DIR / "predictor_weights_v2.csv")
    key_col = "feature_name" if "feature_name" in wdf.columns else "variable"
    weights: dict[str, float] = dict(zip(wdf[key_col], wdf["w_norm"]))

    target_col = "ICILS_computer_literacy"
    all_feature_cols = [c for c in panel.columns
                        if c not in ("country_iso3", "cycle", "program", "domain")]

    cycles = sorted(panel["cycle"].unique())
    log.info("ICILS LOCO — cycles: %s, target: %s", cycles, target_col)

    results_rows: list[dict] = []
    pred_rows:    list[dict] = []

    for test_cycle in cycles:
        train_cycles = [c for c in cycles if c < test_cycle]
        if not train_cycles:
            continue

        X_tr_full, y_tr, X_te_full, y_te, countries = loco.build_xy(
            panel, all_feature_cols, target_col, train_cycles, test_cycle
        )
        # ICILS minimum: n_train >= 10 for Ridge to be meaningful (p < n constraint)
        if X_tr_full is None or len(y_tr) < 10:
            reason = "None from build_xy" if X_tr_full is None else f"n_train={len(y_tr)} < 10"
            log.warning("Fold test=%d: yetersiz train — %s", test_cycle, reason)
            # Record as insufficient_overlap diagnostic
            results_rows.append({
                "program": "ICILS", "domain": "computer_literacy",
                "test_cycle": test_cycle,
                "n_train": len(y_tr) if y_tr is not None else 0,
                "n_test": len(y_te) if y_te is not None else 0,
                "model": "diagnostic",
                "MAE": np.nan, "RMSE": np.nan, "n_features": 0,
                "note": reason,
            })
            continue

        # Drop all-NaN lag cols in training
        col_any_filled = np.array([
            True if not f.startswith("lag_")
            else np.any(~np.isnan(X_tr_full[:, i]))
            for i, f in enumerate(all_feature_cols)
        ])
        fold_feat = [f for f, ok in zip(all_feature_cols, col_any_filled) if ok]
        keep_idx  = [i for i, ok in enumerate(col_any_filled) if ok]
        X_tr = X_tr_full[:, keep_idx]
        X_te = X_te_full[:, keep_idx]

        lit_w = np.array([icils_feat_weight(f, weights) for f in fold_feat])

        log.info("  test=%d  train=%s  features=%d  n_train=%d  n_test=%d",
                 test_cycle, train_cycles, len(fold_feat), len(y_tr), len(y_te))
        log.info("  active_lit_features: %s",
                 [f for f in fold_feat if icils_feat_weight(f, weights) != 1.0])

        # M0: no literature weights
        y_m0, *_ = loco.fit_ridge(X_tr, y_tr, X_te, lit_weights=None)
        # M1: √W_j scaling
        y_m1, *_ = loco.fit_ridge(X_tr, y_tr, X_te, lit_weights=lit_w)

        for model_name, y_pred in [("M0", y_m0), ("M1", y_m1)]:
            valid = ~np.isnan(y_pred)
            if valid.sum() < 2:
                continue
            yt, yp = y_te[valid], y_pred[valid]
            results_rows.append({
                "program": "ICILS", "domain": "computer_literacy",
                "test_cycle": test_cycle,
                "n_train": len(y_tr), "n_test": int(valid.sum()),
                "model": model_name,
                "MAE":  round(float(np.mean(np.abs(yp - yt))), 4),
                "RMSE": round(float(np.sqrt(np.mean((yp - yt)**2))), 4),
                "n_features": len(fold_feat),
            })
        for j, cnt in enumerate(countries):
            pred_rows.append({
                "program": "ICILS", "domain": "computer_literacy",
                "test_cycle": test_cycle, "country_iso3": cnt,
                "y_true": round(float(y_te[j]), 4),
                "y_M0":  round(float(y_m0[j]) if not np.isnan(y_m0[j]) else float("nan"), 4),
                "y_M1":  round(float(y_m1[j]) if not np.isnan(y_m1[j]) else float("nan"), 4),
            })

    res_df  = pd.DataFrame(results_rows)
    pred_df = pd.DataFrame(pred_rows)

    res_path  = STAGE5_DIR / "icils_loco_results.csv"
    pred_path = STAGE5_DIR / "icils_loco_predictions.csv"
    res_df.to_csv(res_path,  index=False)
    pred_df.to_csv(pred_path, index=False)
    log.info("Kaydedildi: %s  (%d satır)", res_path.name,  len(res_df))
    log.info("Kaydedildi: %s  (%d satır)", pred_path.name, len(pred_df))

    if not res_df.empty:
        print("\n=== ICILS LOCO Results ===")
        pivot = res_df.pivot_table(index="test_cycle", columns="model",
                                   values="MAE", aggfunc="first")
        if "M0" in pivot.columns and "M1" in pivot.columns:
            pivot["ΔMAE(M1-M0)"] = pivot["M1"] - pivot["M0"]
        print(pivot.to_string())
        print("\nMean across folds:")
        print(res_df.groupby("model")["MAE"].mean().to_string())

    return res_df


if __name__ == "__main__":
    run_icils_loco()
