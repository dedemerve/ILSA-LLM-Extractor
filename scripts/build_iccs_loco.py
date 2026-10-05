#!/usr/bin/env python3
"""
ICCS LOCO Forecasting pipeline (M0 / M1)

Step 1: Estimates — country-level civic knowledge means from microdata
  Cycles: 2009, 2016, 2022
  Files:  ISG*c2.sav (2009), ISG*c3.sav (2016), ISA*c4.sav (2022)
  PVs:    PV1CIV–PV5CIV
  Weights: TOTWGTS, JRR-zone (JKZONES, JKREPS)

Step 2: Covariates — SES, PARENTAL_EDUCATION from same files
  ISESCS / S_NISB / S_ECOB (SES composite)
  S_HISEI / PARED (parental education)

Step 3: Enriched panel → LOCO (M0/M1)
  LOCO design:
    test=2016 → train pairs: (2009→2016)          [1 fold]
    test=2022 → train pairs: (2009→2016, 2016→2022) [2 folds]
  Note: test=2009 skipped (no training pairs)

Outputs:
  outputs/stage5/iccs_estimates.csv
  outputs/stage5/iccs_enriched_panel.csv
  outputs/stage5/iccs_loco_results.csv
  outputs/stage5/iccs_loco_predictions.csv
"""
from __future__ import annotations

import glob
import logging
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadstat
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
import run_loco_forecasting as loco

STAGE5_DIR = PROJECT_ROOT / "outputs" / "stage5"
ICCS_BASE  = Path("/Users/mrved/Desktop/ILSA Datasets/ICCS Datasets")

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

ALPHA_GRID = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

# ISO3 country code normalization (IEA uses 3-letter but some differ from standard)
_IEA_TO_ISO3: dict[str, str] = {
    "BFL": "BEL",  # Belgium Flemish → Belgium
    "DNW": "DEU",  # Germany NRW → DEU (use first match)
    "ENG": "GBR",  # England → GBR
    "SCO": "GBR",  # Scotland → GBR (duplicate, will be dropped by dedup)
    "TWN": "TWN",  # Chinese Taipei
    "TAP": "TWN",
    "RUS": "RUS",
    "SVN": "SVN",
}


def _iso3(raw: str) -> str:
    raw = str(raw).strip().upper()
    return _IEA_TO_ISO3.get(raw, raw)


# ---------------------------------------------------------------------------
# 1. Extract country means (Rubin / JRR-zone pooling)
# ---------------------------------------------------------------------------

def weighted_mean(y: np.ndarray, w: np.ndarray) -> float:
    w = np.asarray(w, dtype=float)
    w = w / w.sum()
    return float(np.sum(w * np.asarray(y, dtype=float)))


def _variance_jrr(df: pd.DataFrame, pv: str, wgt: str, theta0: float) -> float:
    zones = df["JKZONES"].values
    reps  = df["JKREPS"].values
    w0    = df[wgt].values.astype(float)
    x     = df[pv].values.astype(float)
    sq    = 0.0
    for z in np.unique(zones):
        mask_z = zones == z
        for rv in (0, 1):
            w_rep = w0.copy()
            w_rep[mask_z & (reps == rv)]  *= 2.0
            w_rep[mask_z & (reps != rv)]   = 0.0
            if w_rep.sum() > 0:
                sq += (weighted_mean(x, w_rep) - theta0) ** 2
    return sq / 2.0


def rubin_jrr(df: pd.DataFrame, pv_vars: list[str], wgt: str = "TOTWGTS") -> dict | None:
    thetas, Us = [], []
    for pv in pv_vars:
        mask = df[pv].notna() & (df[wgt] > 0)
        sub = df[mask].copy()
        if len(sub) < 20:
            continue
        th = weighted_mean(sub[pv].values, sub[wgt].values)
        U  = _variance_jrr(sub, pv, wgt, th)
        thetas.append(th)
        Us.append(U)
    if not thetas:
        return None
    T_bar = float(np.mean(thetas))
    U_bar = float(np.mean(Us))
    B     = float(np.var(thetas, ddof=1)) if len(thetas) > 1 else 0.0
    T     = U_bar + (1 + 1 / len(thetas)) * B
    SE    = float(np.sqrt(max(T, 0)))
    return {"mean": round(T_bar, 4), "se": round(SE, 4),
            "ci_lo": round(T_bar - 1.96 * SE, 2),
            "ci_hi": round(T_bar + 1.96 * SE, 2),
            "n_pv": len(thetas)}


ICCS_CYCLES = {
    2009: (ICCS_BASE / "ICCS2009_IDB_SPSS" / "Data_G8", "ISG", "C2"),
    2016: (ICCS_BASE / "ICCS2016_IDB_SPSS" / "Data",    "ISG", "C3"),
    2022: (ICCS_BASE / "ICCS2022_IDB_SPSS" / "Data",    "ISA", "C4"),
}

PV_VARS   = ["PV1CIV", "PV2CIV", "PV3CIV", "PV4CIV", "PV5CIV"]
WGT_COL   = "TOTWGTS"

# Covariate variable name mappings per cycle
COV_VARS: dict[int, dict[str, list[str]]] = {
    # Use consistent variables across cycles (same scale)
    # SES: S_NISB is available in 2016/2022 but not 2009 → use only when available
    # PARENTAL_EDU: HISCED is 1-6 scale across all cycles (S_HISEI is 0-90, incompatible)
    2009: {
        # 2009 IDB uses NISB (unstandardized); S_NISB and ISESCS don't exist in this release
        "SES_COMPOSITE":    ["NISB", "S_NISB", "ISESCS"],
        "PARENTAL_EDUCATION": ["HISCED", "PARED"],
    },
    2016: {
        "SES_COMPOSITE":    ["S_NISB", "S_ECOB"],
        # 2016 IDB uses S_HISCED (standardized); plain HISCED doesn't exist
        "PARENTAL_EDUCATION": ["S_HISCED", "HISCED", "PARED"],
    },
    2022: {
        "SES_COMPOSITE":    ["S_NISB", "S_ECOB"],
        # 2022 IDB uses S_HISCED in ISG* background files
        "PARENTAL_EDUCATION": ["S_HISCED", "HISCED"],
    },
}


def _first_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def extract_iccs_estimates() -> pd.DataFrame:
    rows = []
    for cycle, (data_dir, prefix, suffix) in ICCS_CYCLES.items():
        if not data_dir.exists():
            files = []
        else:
            files = sorted(
                [str(data_dir / f) for f in os.listdir(str(data_dir))
                 if f.startswith(prefix) and f.endswith(f"{suffix}.sav")]
            )
        log.info("ICCS %d: %d %s* files", cycle, len(files), prefix)

        for fpath in files:
            df, _ = pyreadstat.read_sav(fpath)

            # Country
            country_raw = (df["COUNTRY"].iloc[0]
                           if "COUNTRY" in df.columns
                           else str(int(df["IDCNTRY"].iloc[0])))
            country_iso3 = _iso3(country_raw)

            # Civic knowledge
            if not all(pv in df.columns for pv in PV_VARS):
                continue
            if WGT_COL not in df.columns:
                continue

            res = rubin_jrr(df, PV_VARS, WGT_COL)
            if res:
                rows.append({
                    "program": "ICCS", "cycle": cycle, "country_iso3": country_iso3,
                    "domain": "civic_knowledge", **res,
                    "n_total": len(df), "method": "JRR_zone",
                })

            # Covariates
            for domain, candidates in COV_VARS.get(cycle, {}).items():
                col = _first_col(df, candidates)
                if col is None:
                    continue
                mask = df[col].notna() & (df[WGT_COL] > 0)
                sub = df[mask]
                if len(sub) < 20:
                    continue
                mean_val = weighted_mean(sub[col].values, sub[WGT_COL].values)
                rows.append({
                    "program": "ICCS", "cycle": cycle, "country_iso3": country_iso3,
                    "domain": domain, "mean": round(mean_val, 4),
                    "se": np.nan, "ci_lo": np.nan, "ci_hi": np.nan,
                    "n_total": len(sub), "method": "weighted_mean", "n_pv": 1,
                })

    df_out = pd.DataFrame(rows)
    # Dedup: keep first for duplicate (country, cycle, domain) — handles BFL/DNW overlaps
    df_out = df_out.drop_duplicates(subset=["cycle", "country_iso3", "domain"], keep="first")
    out_path = STAGE5_DIR / "iccs_estimates.csv"
    df_out.to_csv(out_path, index=False)
    log.info("Saved: %s  (%d rows)", out_path.name, len(df_out))

    # Summary
    for cycle in sorted(df_out["cycle"].unique()):
        sub = df_out[(df_out["cycle"] == cycle) & (df_out["domain"] == "civic_knowledge")]
        log.info("  ICCS %d: %d countries, mean CK = %.1f",
                 cycle, len(sub), sub["mean"].mean())
    return df_out


# ---------------------------------------------------------------------------
# 2. Build enriched panel (wide format)
# ---------------------------------------------------------------------------

def build_iccs_panel(est: pd.DataFrame) -> pd.DataFrame:
    outcome = est[est["domain"] == "civic_knowledge"][
        ["country_iso3", "cycle", "mean"]
    ].rename(columns={"mean": "ICCS_civic_knowledge"})

    cycles = sorted(outcome["cycle"].unique())
    panel  = outcome.copy()

    cycle_pairs = [(cycles[i], cycles[i + 1]) for i in range(len(cycles) - 1)]

    for cov_domain in ["SES_COMPOSITE", "PARENTAL_EDUCATION"]:
        cov_df = est[est["domain"] == cov_domain][
            ["country_iso3", "cycle", "mean"]
        ].rename(columns={"mean": cov_domain})

        lag_map: dict[tuple, float] = {}
        for _, row in cov_df.iterrows():
            country = row["country_iso3"]
            cyc     = int(row["cycle"])
            for prev_c, next_c in cycle_pairs:
                if cyc == prev_c:
                    lag_map[(country, next_c)] = row[cov_domain]

        lag_col = f"lag_{cov_domain}"
        panel[lag_col] = panel.apply(
            lambda r: lag_map.get((r["country_iso3"], int(r["cycle"])), np.nan),
            axis=1,
        )
        n_filled = panel[lag_col].notna().sum()
        log.info("  lag_%s: %d non-NaN", cov_domain, n_filled)

    panel["program"] = "ICCS"
    panel["domain"]  = "civic_knowledge"

    out = STAGE5_DIR / "iccs_enriched_panel.csv"
    panel.to_csv(out, index=False)
    log.info("Saved: %s  (%d rows, %d cols)", out.name, len(panel), panel.shape[1])

    print("\n=== ICCS Panel ===")
    for c in sorted(panel["cycle"].unique()):
        sub = panel[panel["cycle"] == c]
        print(f"  {c}: {len(sub)} countries, "
              f"lag_SES={sub['lag_SES_COMPOSITE'].notna().sum()} non-NaN")
    return panel


# ---------------------------------------------------------------------------
# 3. LOCO (M0 / M1)
# ---------------------------------------------------------------------------

_ICCS_TO_CANONICAL: dict[str, str] = {
    "SES_COMPOSITE":    "SES_COMPOSITE",
    "PARENTAL_EDUCATION": "PARENTAL_EDU",
}


def _iccs_feat_weight(f: str, weights: dict) -> float:
    raw = f.replace("lag_", "")
    canonical = _ICCS_TO_CANONICAL.get(raw, raw)
    return weights.get(canonical, 1.0)


def run_iccs_loco(panel: pd.DataFrame, weights: dict) -> pd.DataFrame:
    target_col = "ICCS_civic_knowledge"
    all_feat   = [c for c in panel.columns
                  if c not in ("country_iso3", "cycle", "program", "domain")]

    cycles = sorted(panel["cycle"].unique())
    log.info("ICCS LOCO cycles: %s", cycles)

    results, preds = [], []
    for test_cycle in cycles:
        train_cycles = [c for c in cycles if c < test_cycle]
        if not train_cycles:
            continue

        X_tr, y_tr, X_te, y_te, countries = loco.build_xy(
            panel, all_feat, target_col, train_cycles, test_cycle
        )
        if X_tr is None or len(y_tr) < 5:
            reason = "None" if X_tr is None else f"n_train={len(y_tr)}<5"
            log.warning("Fold %d: insufficient train — %s", test_cycle, reason)
            results.append({
                "program": "ICCS", "domain": "civic_knowledge",
                "test_cycle": test_cycle,
                "n_train": 0 if X_tr is None else len(y_tr),
                "n_test": 0,
                "model": "diagnostic", "MAE": np.nan, "RMSE": np.nan,
                "note": reason,
            })
            continue

        # Drop all-NaN lag cols
        keep = [i for i, f in enumerate(all_feat)
                if not f.startswith("lag_") or np.any(~np.isnan(X_tr[:, i]))]
        fold_feat = [all_feat[i] for i in keep]
        X_tr = X_tr[:, keep]
        X_te = X_te[:, keep]
        lit_w = np.array([_iccs_feat_weight(f, weights) for f in fold_feat])

        log.info("  test=%d train=%s n_train=%d n_test=%d n_feat=%d",
                 test_cycle, train_cycles, len(y_tr), len(y_te), len(fold_feat))

        y_m0, *_ = loco.fit_ridge(X_tr, y_tr, X_te, lit_weights=None)
        y_m1, *_ = loco.fit_ridge(X_tr, y_tr, X_te, lit_weights=lit_w)

        for mname, y_pred in [("M0", y_m0), ("M1", y_m1)]:
            valid = ~np.isnan(y_pred)
            if valid.sum() < 2:
                continue
            yt, yp = y_te[valid], y_pred[valid]
            results.append({
                "program": "ICCS", "domain": "civic_knowledge",
                "test_cycle": test_cycle,
                "n_train": len(y_tr), "n_test": int(valid.sum()),
                "model": mname,
                "MAE":  round(float(np.mean(np.abs(yp - yt))), 4),
                "RMSE": round(float(np.sqrt(np.mean((yp - yt) ** 2))), 4),
                "n_features": len(fold_feat),
            })

        for j, cnt in enumerate(countries):
            preds.append({
                "program": "ICCS", "domain": "civic_knowledge",
                "test_cycle": test_cycle, "country_iso3": cnt,
                "y_true": round(float(y_te[j]), 4),
                "y_M0":  round(float(y_m0[j]), 4) if not np.isnan(y_m0[j]) else float("nan"),
                "y_M1":  round(float(y_m1[j]), 4) if not np.isnan(y_m1[j]) else float("nan"),
            })

    res_df  = pd.DataFrame(results)
    pred_df = pd.DataFrame(preds)

    res_df.to_csv(STAGE5_DIR / "iccs_loco_results.csv", index=False)
    pred_df.to_csv(STAGE5_DIR / "iccs_loco_predictions.csv", index=False)
    log.info("Saved iccs_loco_results.csv (%d rows)", len(res_df))
    log.info("Saved iccs_loco_predictions.csv (%d rows)", len(pred_df))

    m01 = res_df[res_df["model"].isin(["M0", "M1"])]
    if not m01.empty:
        print("\n=== ICCS LOCO Results ===")
        pivot = m01.pivot_table(index="test_cycle", columns="model",
                                values="MAE", aggfunc="first")
        if "M0" in pivot and "M1" in pivot:
            pivot["ΔMAE(M1-M0)"] = pivot["M1"] - pivot["M0"]
        print(pivot.to_string())
        print("\nMean:")
        print(m01.groupby("model")["MAE"].mean().to_string())

    return res_df


# ---------------------------------------------------------------------------
# 4. Forward predictions (2022→2027)
# ---------------------------------------------------------------------------

def run_iccs_forward(panel: pd.DataFrame, weights: dict) -> pd.DataFrame:
    """Train on all available pairs, predict next cycle (2027)."""
    target_col = "ICCS_civic_knowledge"
    all_feat   = [c for c in panel.columns
                  if c not in ("country_iso3", "cycle", "program", "domain")]
    cycles = sorted(panel["cycle"].unique())

    X_tr, y_tr, X_te, y_te, countries = loco.build_xy(
        panel, all_feat, target_col, cycles, None  # None = use all as train
    )
    # Actually: for forward prediction, train on all pairs and predict last cycle countries
    # Use train=all cycles, test="future" = construct X from last available cycle
    last_cycle = cycles[-1]
    train_cycles = cycles  # use all as training

    # Build training data from all consecutive pairs
    X_train_list, y_train_list = [], []
    for i in range(len(cycles) - 1):
        prev_c, curr_c = cycles[i], cycles[i + 1]
        prev_df = panel[panel["cycle"] == prev_c].set_index("country_iso3")
        curr_df = panel[panel["cycle"] == curr_c].set_index("country_iso3")
        common  = prev_df.index.intersection(curr_df.index)
        if len(common) < 3:
            continue
        score_cols = [f for f in all_feat if not f.startswith("lag_") and f in prev_df.columns]
        cov_cols   = [f for f in all_feat if f.startswith("lag_") and f in curr_df.columns]
        X_ = loco._make_X(prev_df, curr_df, common, score_cols, cov_cols)
        y_ = curr_df.loc[common, target_col].values.astype(float)
        mask = ~np.isnan(y_)
        if mask.sum() < 3:
            continue
        X_train_list.append(X_[mask])
        y_train_list.append(y_[mask])

    if not X_train_list:
        log.warning("ICCS forward: no training data")
        return pd.DataFrame()

    X_train = np.vstack(X_train_list)
    y_train = np.concatenate(y_train_list)

    # Test = last cycle countries → predict next cycle
    last_df   = panel[panel["cycle"] == last_cycle].set_index("country_iso3")
    dummy_df  = last_df.copy()  # predict "2027" using last cycle values as lag
    countries = last_df.index.tolist()
    score_cols = [f for f in all_feat if not f.startswith("lag_") and f in last_df.columns]
    cov_cols   = [f for f in all_feat if f.startswith("lag_") and f in dummy_df.columns]
    X_te = loco._make_X(last_df, dummy_df, last_df.index, score_cols, cov_cols)

    fold_feat = score_cols + cov_cols
    lit_w = np.array([_iccs_feat_weight(f, weights) for f in fold_feat])

    # Impute + scale + predict
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler
    col_means = np.nanmean(X_train, axis=0)
    col_means = np.where(np.isnan(col_means), 0.0, col_means)
    X_tr_ = np.where(np.isnan(X_train), col_means, X_train)
    X_te_ = np.where(np.isnan(X_te),   col_means, X_te)
    sc = StandardScaler(); X_tr_ = sc.fit_transform(X_tr_); X_te_ = sc.transform(X_te_)

    scale_m1 = np.sqrt(np.clip(lit_w, 1e-6, None))
    cv = min(5, len(y_train))
    ridge = RidgeCV(alphas=ALPHA_GRID, cv=cv)

    # M0
    ridge.fit(X_tr_, y_train)
    y_m0 = ridge.predict(X_te_)

    # M1
    ridge.fit(X_tr_ * scale_m1, y_train)
    y_m1 = ridge.predict(X_te_ * scale_m1)

    rows = []
    for j, cnt in enumerate(countries):
        last_score = last_df.loc[cnt, target_col] if target_col in last_df.columns else np.nan
        rows.append({
            "program": "ICCS", "domain": "civic_knowledge",
            "country_iso3": cnt, "last_cycle": last_cycle,
            "last_score": round(float(last_score), 4) if not np.isnan(last_score) else np.nan,
            "predicted_cycle": 2027,
            "y_M0": round(float(y_m0[j]), 4),
            "y_M1": round(float(y_m1[j]), 4),
        })

    fwd_df = pd.DataFrame(rows)
    fwd_df.to_csv(STAGE5_DIR / "iccs_forward_predictions.csv", index=False)
    log.info("Saved iccs_forward_predictions.csv (%d countries)", len(fwd_df))
    return fwd_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Load weights
    wdf = pd.read_csv(STAGE5_DIR / "predictor_weights_v2.csv")
    key_col = "feature_name" if "feature_name" in wdf.columns else "variable"
    weights: dict[str, float] = dict(zip(wdf[key_col], wdf["w_norm"]))

    log.info("=== Step 1: Extract ICCS estimates ===")
    est = extract_iccs_estimates()
    print(f"\nEstimates: {len(est)} rows, "
          f"{est[est['domain']=='civic_knowledge']['country_iso3'].nunique()} unique countries")

    log.info("=== Step 2: Build enriched panel ===")
    panel = build_iccs_panel(est)

    log.info("=== Step 3: LOCO forecasting ===")
    loco_res = run_iccs_loco(panel, weights)

    log.info("=== Step 4: Forward predictions (→2027) ===")
    fwd = run_iccs_forward(panel, weights)
    if not fwd.empty:
        print(f"\nForward predictions: {len(fwd)} countries → 2027")
        print(fwd[["country_iso3", "last_score", "y_M0", "y_M1"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
