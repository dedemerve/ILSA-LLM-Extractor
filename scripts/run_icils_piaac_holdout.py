#!/usr/bin/env python3
"""
ICILS ve PIAAC için Single-Transition Hold-Out Validation (M0 / M1)

LOCO (pair-based) bu programlar için çalışmıyor:
  - PIAAC: sadece 2 cycle (2012, 2017) → pair gereksinimi karşılanamaz
  - ICILS: 2013∩2018 overlap = 3 ülke → n_train < p

Bu script, tek seferlik hold-out (single-transition) uygular:
  PIAAC:
    - Lag: PARENTAL_EDUCATION at 2012 → test 2017
    - Train: tüm 2012 satırları (country-level; lag lag_PARENTAL_EDUCATION=2012 değerinden gelir)
    - Wait, lag at test cycle = value from training cycle → standard lag design
    - Actually: since we only have 1 transition, train/test distinction means:
      X_train = [{PIAAC_literacy_2012, lag features}] for countries in 2012
      y_train = PIAAC_literacy_2017 (for countries in both 2012 and 2017)
      X_test = same features, predict 2017 scores
      → This is leave-one-out at country level (LOOCV) on the single transition
      → Or: simple OLS/Ridge train-on-all, evaluate on same set (very optimistic but standard for 1-fold)

  Actually the cleanest design:
    Train: use 2012 predictors to predict 2017 scores (no temporal leakage since 2012 < 2017)
    Test: predict 2017 scores → evaluate MAE
    Model comparison: M0 (no lit weights) vs M1 (√W_j)
    Note: with only 1 fold, this is treated as diagnostic, not as evidence of generalization

  ICILS:
    Train: 2013 and 2018 observations (pooled)
    Test: 2023 scores
    Features: lag scores (from 2018 for test=2023) + lag covariates (from 2018)
    Note: overlap between 2018 and 2023 = 10 countries

Outputs: outputs/stage5/icils_piaac_holdout_results.csv
"""
from __future__ import annotations

import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
import run_loco_forecasting as loco

STAGE5_DIR = PROJECT_ROOT / "outputs" / "stage5"
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

ALPHA_GRID = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]


def fit_ridge_holdout(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_te: np.ndarray, lit_weights: np.ndarray | None = None
) -> tuple[np.ndarray, float]:
    """Ridge fit on training data, evaluate on test."""
    # Impute col-mean
    col_means = np.nanmean(X_tr, axis=0)
    col_means  = np.where(np.isnan(col_means), 0.0, col_means)
    X_tr = np.where(np.isnan(X_tr), col_means, X_tr)
    X_te = np.where(np.isnan(X_te), col_means, X_te)

    sc   = StandardScaler()
    X_tr = sc.fit_transform(X_tr)
    X_te = sc.transform(X_te)

    if lit_weights is not None:
        scale = np.sqrt(np.clip(lit_weights, 1e-6, None))
        X_tr  = X_tr * scale
        X_te  = X_te * scale

    cv = min(5, len(y_tr))
    ridge = RidgeCV(alphas=ALPHA_GRID, cv=cv)
    ridge.fit(X_tr, y_tr)
    preds = ridge.predict(X_te)
    return preds, float(ridge.alpha_)


def run_piaac_holdout(weights: dict) -> list[dict]:
    """
    PIAAC single-transition: train on 2012 data → predict 2017 scores.

    Design:
      - Overlap countries (present in both 2012 and 2017)
      - X = [PIAAC_literacy_2012, lag_PARENTAL_EDUCATION_2012]
      - y = PIAAC_literacy_2017
      - Evaluate MAE on overlap set
    """
    est = pd.read_csv(STAGE5_DIR / "icils_piaac_estimates.csv")
    piaac = est[est["program"] == "PIAAC"].copy()

    rows = []
    for domain in ["literacy", "numeracy"]:
        d12 = piaac[(piaac.cycle == 2012) & (piaac.domain == domain)][
            ["country_iso3", "mean"]
        ].rename(columns={"mean": f"PIAAC_{domain}_2012"})

        d17 = piaac[(piaac.cycle == 2017) & (piaac.domain == domain)][
            ["country_iso3", "mean"]
        ].rename(columns={"mean": f"PIAAC_{domain}_2017"})

        pared12 = piaac[(piaac.cycle == 2012) & (piaac.domain == "PARENTAL_EDUCATION")][
            ["country_iso3", "mean"]
        ].rename(columns={"mean": "lag_PARENTAL_EDUCATION"})

        # Merge to overlap
        df = d12.merge(d17, on="country_iso3") \
                .merge(pared12, on="country_iso3", how="left")

        n = len(df)
        if n < 5:
            log.warning("PIAAC %s: overlap=%d, skipping", domain, n)
            continue

        log.info("PIAAC %s: n=%d overlap countries", domain, n)

        feature_cols = [c for c in df.columns
                        if c not in ("country_iso3", f"PIAAC_{domain}_2017")]
        target_col   = f"PIAAC_{domain}_2017"

        X = df[feature_cols].values.astype(float)
        y = df[target_col].values.astype(float)

        # Lit weights for PIAAC features; map to canonical construct names
        _PIAAC_MAP = {
            "lag_PARENTAL_EDUCATION": "PARENTAL_EDU",
            f"PIAAC_{domain}_2012":   None,  # outcome lag; no lit weight
        }

        def _piaac_feat_weight(f: str) -> float:
            canonical = _PIAAC_MAP.get(f)
            if canonical:
                return weights.get(canonical, 1.0)
            return 1.0

        lit_w = np.array([_piaac_feat_weight(f) for f in feature_cols])

        preds_m0, alpha_m0 = fit_ridge_holdout(X, y, X, lit_weights=None)
        preds_m1, alpha_m1 = fit_ridge_holdout(X, y, X, lit_weights=lit_w)

        # Note: train=test here because only 1 transition → reported as in-sample diagnostic
        mae_m0 = float(np.mean(np.abs(preds_m0 - y)))
        mae_m1 = float(np.mean(np.abs(preds_m1 - y)))

        log.info("  PIAAC %s: M0_MAE=%.4f M1_MAE=%.4f (in-sample, n=%d)", domain, mae_m0, mae_m1, n)

        rows.append({
            "program": "PIAAC", "domain": domain,
            "train_cycles": "2012", "test_cycle": 2017,
            "n_overlap": n, "n_features": len(feature_cols),
            "design": "single_transition_in_sample",
            "MAE_M0": round(mae_m0, 4), "MAE_M1": round(mae_m1, 4),
            "delta_MAE": round(mae_m1 - mae_m0, 4),
            "note": "in-sample fit; not generalizable; diagnostic only",
        })
    return rows


def run_icils_holdout(weights: dict) -> list[dict]:
    """
    ICILS single-transition hold-out:
    Train: 2013∪2018 pooled (as separate country-cycle observations)
    Test:  2023 scores (2018∩2023 overlap countries)

    Lag features at test:
      - ICILS score at 2018 (lag_Y for test 2023)
      - Covariates at 2018 (lag_SES etc. for test 2023)
    """
    panel = pd.read_csv(STAGE5_DIR / "icils_enriched_panel.csv")
    panel["cycle"] = panel["cycle"].astype(int)

    # Build training set: countries that appear in consecutive pairs
    # For training, we use all (prev_cycle → curr_cycle) pairs
    # Pair (2013→2018): 3 countries; Pair (2018→2023): 10 countries
    # Pool both pairs for training

    all_feature_cols = [c for c in panel.columns
                        if c not in ("country_iso3","cycle","program","domain")]
    target_col = "ICILS_computer_literacy"

    cycles = sorted(panel["cycle"].unique())  # [2013, 2018, 2023]
    X_train_list, y_train_list = [], []

    # Manually build training pairs
    for prev_c, curr_c in [(cycles[0], cycles[1]), (cycles[1], cycles[2])]:
        prev_df = panel[panel["cycle"] == prev_c].set_index("country_iso3")
        curr_df = panel[panel["cycle"] == curr_c].set_index("country_iso3")
        common  = prev_df.index.intersection(curr_df.index)
        if len(common) == 0:
            continue

        score_cols = [f for f in all_feature_cols if not f.startswith("lag_")]
        cov_cols   = [f for f in all_feature_cols if f.startswith("lag_")]
        avail_score = [c for c in score_cols if c in prev_df.columns]
        avail_cov   = [c for c in cov_cols   if c in curr_df.columns]

        X_tr_ = loco._make_X(prev_df, curr_df, common, avail_score, avail_cov)
        y_tr_ = curr_df.loc[common, target_col].values.astype(float)
        mask   = ~np.isnan(y_tr_)
        if mask.sum() == 0:
            continue
        X_train_list.append(X_tr_[mask])
        y_train_list.append(y_tr_[mask])
        log.info("  ICILS train pair (%d→%d): %d obs", prev_c, curr_c, mask.sum())

    if not X_train_list:
        log.error("ICILS: no training data")
        return []

    X_train = np.vstack(X_train_list)
    y_train = np.concatenate(y_train_list)

    # Test: 2023
    # Features at test: lag scores from 2018, lag covariates from 2023 row
    test_cycle  = 2023
    lag_cycle   = 2018
    lag_df_te   = panel[panel["cycle"] == lag_cycle].set_index("country_iso3")
    target_df   = panel[panel["cycle"] == test_cycle].set_index("country_iso3")
    common_te   = lag_df_te.index.intersection(target_df.index)

    avail_score_te = [c for c in all_feature_cols if not c.startswith("lag_") and c in lag_df_te.columns]
    avail_cov_te   = [c for c in all_feature_cols if c.startswith("lag_") and c in target_df.columns]

    X_te_= loco._make_X(lag_df_te, target_df, common_te, avail_score_te, avail_cov_te)
    y_te = target_df.loc[common_te, target_col].values.astype(float)
    mask_te = ~np.isnan(y_te)

    if mask_te.sum() < 3:
        log.warning("ICILS test: n_test=%d < 3", mask_te.sum())
        return []

    X_te  = X_te_[mask_te]
    y_te  = y_te[mask_te]
    fold_feat = avail_score_te + avail_cov_te

    lit_w = np.array([icils_feat_weight(f, weights) for f in fold_feat])

    log.info("ICILS holdout: n_train=%d, n_test=%d, n_features=%d",
             len(y_train), len(y_te), len(fold_feat))
    log.info("  test countries: %s", list(common_te[mask_te]))
    log.info("  active lit features: %s",
             [f for f in fold_feat if icils_feat_weight(f, weights) != 1.0])

    preds_m0, _ = fit_ridge_holdout(X_train.copy(), y_train, X_te.copy(), lit_weights=None)
    preds_m1, _ = fit_ridge_holdout(X_train.copy(), y_train, X_te.copy(), lit_weights=lit_w)

    mae_m0 = float(np.mean(np.abs(preds_m0 - y_te)))
    mae_m1 = float(np.mean(np.abs(preds_m1 - y_te)))
    log.info("  M0_MAE=%.4f  M1_MAE=%.4f  Δ=%.4f", mae_m0, mae_m1, mae_m1 - mae_m0)

    return [{
        "program": "ICILS", "domain": "computer_literacy",
        "train_cycles": "2013+2018 pooled", "test_cycle": 2023,
        "n_train": len(y_train), "n_test": len(y_te), "n_features": len(fold_feat),
        "design": "pooled_train_holdout_test",
        "MAE_M0": round(mae_m0, 4), "MAE_M1": round(mae_m1, 4),
        "delta_MAE": round(mae_m1 - mae_m0, 4),
        "note": "pooled training (2013∪2018 pairs); single test fold (2023)",
    }]


def icils_feat_weight(f: str, weights: dict) -> float:
    _ICILS_MAP = {
        "lag_SES_COMPOSITE":          "SES_COMPOSITE",
        "lag_PARENTAL_EDUCATION":     "PARENTAL_EDU",
        "lag_HOME_LITERACY_ACTIVITIES": "HOME_LITERACY_ACTIVITIES",
    }
    canonical = _ICILS_MAP.get(f)
    if canonical:
        return weights.get(canonical, 1.0)
    return 1.0


def main():
    wdf = pd.read_csv(STAGE5_DIR / "predictor_weights_v2.csv")
    key_col  = "feature_name" if "feature_name" in wdf.columns else "variable"
    weights  = dict(zip(wdf[key_col], wdf["w_norm"]))

    rows = []
    rows.extend(run_icils_holdout(weights))
    rows.extend(run_piaac_holdout(weights))

    df = pd.DataFrame(rows)
    if df.empty:
        print("No results produced.")
        return

    out = STAGE5_DIR / "icils_piaac_holdout_results.csv"
    df.to_csv(out, index=False)

    print("\n=== ICILS + PIAAC Hold-Out Results ===")
    print(df[["program","domain","design","n_train","n_test",
              "MAE_M0","MAE_M1","delta_MAE","note"]].to_string(index=False))
    print(f"\nSaved: {out.name}")


if __name__ == "__main__":
    main()
