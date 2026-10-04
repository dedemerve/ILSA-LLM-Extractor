#!/usr/bin/env python3
"""
TALIS ablation for PISA Reading — three conditions:
  M0               — Ridge, no lit weights
  M1_without_TALIS — Ridge, √W_j weights, TALIS columns dropped from panel
  M1_with_TALIS    — Ridge, √W_j weights, TALIS columns present (current state)

Uses the exact same build_panel / build_xy / imputation / Ridge pipeline as
run_loco_forecasting.py; only the feature set and weighting differ.
"""
from __future__ import annotations
import sys, warnings, logging
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

# Import helpers from the main forecasting script
import run_loco_forecasting as loco

TALIS_COLS = ["lag_SECLSS", "lag_TCDISCS", "lag_TEFFPROS", "lag_TJSPROS"]


def run_one_condition(panel: pd.DataFrame,
                      weights: dict,
                      drop_talis: bool,
                      use_lit_weights: bool,
                      prog: str = "PISA",
                      dom: str = "reading") -> list[dict]:
    """Single LOCO pass identical to run_loco_forecasting internals."""
    target_col = loco.make_feature_key(prog, dom)
    if target_col not in panel.columns:
        raise ValueError(f"Target col {target_col!r} not in panel")

    work_panel = panel.copy()
    if drop_talis:
        work_panel = work_panel.drop(columns=[c for c in TALIS_COLS if c in work_panel.columns])

    all_feature_cols = [c for c in work_panel.columns
                        if c not in ("country_iso3", "cycle")]

    # Only PISA reading cycles
    prog_rows = work_panel[work_panel.get("program", pd.Series(dtype=str)).eq(prog)
                           if "program" in work_panel.columns
                           else slice(None)]

    # Derive cycles from the target column having non-NaN values
    cycles_mask = work_panel[target_col].notna()
    cycles = sorted(work_panel.loc[cycles_mask, "cycle"].unique())

    fold_results = []
    for test_cycle in cycles:
        train_cycles = [c for c in cycles if c < test_cycle]
        if not train_cycles:
            continue

        X_tr_full, y_tr, X_te_full, y_te, countries = loco.build_xy(
            work_panel, all_feature_cols, target_col, train_cycles, test_cycle
        )
        if X_tr_full is None or len(y_tr) < 3:
            continue

        # Drop completely-NaN lag cols in training (same as main script)
        col_any_filled = np.array([
            True if not f.startswith("lag_")
            else np.any(~np.isnan(X_tr_full[:, i]))
            for i, f in enumerate(all_feature_cols)
        ])
        fold_feat = [f for f, ok in zip(all_feature_cols, col_any_filled) if ok]
        keep_idx  = [i for i, ok in enumerate(col_any_filled) if ok]
        X_tr = X_tr_full[:, keep_idx]
        X_te = X_te_full[:, keep_idx]

        # Drop rows where target is NaN
        mask_tr = ~np.isnan(y_tr)
        mask_te = ~np.isnan(y_te)
        if mask_tr.sum() < 3 or mask_te.sum() < 1:
            continue

        X_tr, y_tr_ = X_tr[mask_tr], y_tr[mask_tr]
        X_te, y_te_ = X_te[mask_te], y_te[mask_te]

        lit_w = (np.array([loco._feat_weight(f, fold_feat, weights) for f in fold_feat])
                 if use_lit_weights else None)

        # fit_ridge handles NaN imputation, StandardScaler, and lit_weight scaling
        preds, _, _, _, _ = loco.fit_ridge(X_tr, y_tr_, X_te, lit_weights=lit_w)

        mae_val = float(np.mean(np.abs(preds - y_te_)))
        fold_results.append({
            "test_cycle": test_cycle,
            "n_train_countries": mask_tr.sum(),
            "n_test_countries": mask_te.sum(),
            "active_talis_features": sum(1 for f in fold_feat if f in TALIS_COLS),
            "total_active_features": len(fold_feat),
            "mae": round(mae_val, 4),
        })

    return fold_results


def main():
    est, weights = loco.load_data()
    # Restrict to PISA reading
    pisa_r = est[(est["program"] == "PISA") & (est["domain"] == "reading")]
    panel  = loco.build_panel(est)  # full panel (needed for cross-program features)

    conditions = {
        "M0":               (True,  False),
        "M1_without_TALIS": (True,  True),
        "M1_with_TALIS":    (False, True),
    }

    summary_rows = []
    all_fold_rows = []

    print(f"\n{'Condition':25s}  {'Mean MAE':>10s}  {'n_folds':>8s}  TALIS_active_folds")
    print("-" * 65)

    for name, (drop_talis, use_w) in conditions.items():
        fold_results = run_one_condition(panel, weights, drop_talis, use_w)
        if not fold_results:
            print(f"{name:25s}  NO FOLDS")
            continue
        maes = [r["mae"] for r in fold_results]
        mean_mae = np.mean(maes)
        talis_active = [r["active_talis_features"] for r in fold_results]

        print(f"{name:25s}  {mean_mae:10.4f}  {len(maes):8d}  {talis_active}")
        summary_rows.append({
            "condition": name, "mean_mae": round(mean_mae, 4),
            "n_folds": len(maes),
        })
        for r in fold_results:
            r["condition"] = name
            all_fold_rows.append(r)

    if summary_rows:
        # Δ relative to M0
        m0_mae = next(r["mean_mae"] for r in summary_rows if r["condition"] == "M0")
        for r in summary_rows:
            r["delta_vs_M0"] = round(r["mean_mae"] - m0_mae, 4)

        print("\nSummary (Δ vs M0):")
        df = pd.DataFrame(summary_rows)
        print(df.to_string(index=False))

        out = PROJECT_ROOT / "outputs/stage5/ablation_talis_pisa_reading.csv"
        fold_out = PROJECT_ROOT / "outputs/stage5/ablation_talis_pisa_reading_folds.csv"
        df.to_csv(out, index=False)
        pd.DataFrame(all_fold_rows).to_csv(fold_out, index=False)
        print(f"\nSaved: {out.name}")
        print(f"Saved: {fold_out.name}")

        print("\nFold-level MAEs:")
        folds_df = pd.DataFrame(all_fold_rows)
        pivot = folds_df.pivot_table(index="test_cycle", columns="condition",
                                     values="mae", aggfunc="first")
        pivot["Δ(M1_with_TALIS - M1_without_TALIS)"] = (
            pivot["M1_with_TALIS"] - pivot["M1_without_TALIS"]
        )
        print(pivot.to_string())


if __name__ == "__main__":
    main()
