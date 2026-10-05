#!/usr/bin/env python3
"""
ICILS 2028 forward predictions.

Design rationale:
- ICILS has 3 cycles (2013, 2018, 2023).
- Standard pair-based LOCO requires ≥2 training pairs; ICILS has only 1
  feasible pair (2013→2018 transition) before the test fold, so the holdout
  evaluation used a pooled design (train=2013+2018 pooled, test=2023).
- For forward predictions we use ALL available transition pairs
  (2013→2018 and 2018→2023) to maximise training signal, then predict 2028
  from 2023 scores.
- M0: Ridge, no literature weights.
- M1: Ridge, √W_j feature scaling (unified W_j_forecast via ilsa_common).
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STAGE5 = PROJECT_ROOT / "outputs" / "stage5"
STAGE4 = PROJECT_ROOT / "outputs" / "stage4"

try:
    from scripts.ilsa_common import load_forecast_weights
except ImportError:
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.ilsa_common import load_forecast_weights

OUTCOME_COL = "ICILS_computer_literacy"
PREDICTED_CYCLE = 2028
LAST_CYCLE = 2023


FEATURE_TO_CONSTRUCT = {
    "lag_outcome":                "PRIOR_ACHIEVEMENT",
    "lag_SES_COMPOSITE":          "SES_COMPOSITE",
    "lag_PARENTAL_EDUCATION":     "PARENTAL_EDUCATION",
    "lag_HOME_LITERACY_ACTIVITIES": "HOME_LITERACY_ACTIVITIES",
}


def load_wj_map() -> dict[str, float]:
    """Return {feature_col: sqrt(W_j)} for each ICILS feature."""
    weights = load_forecast_weights()
    result = {}
    for feat, construct in FEATURE_TO_CONSTRUCT.items():
        w = weights.get(construct, weights.get(feat, 1.0))
        if w <= 0:
            w = 1.0
        result[feat] = float(w) ** 0.5
    return result


def build_pairs(ep: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Build (lag_features, outcome) pairs from the enriched panel.

    Enriched panel columns:
      ICILS_computer_literacy  — the outcome at each cycle
      lag_SES_COMPOSITE, lag_PARENTAL_EDUCATION, lag_HOME_LITERACY_ACTIVITIES
                               — already lag-named covariate columns

    For each consecutive pair (t, t+1):
      features = ICILS_computer_literacy at t  (renamed lag_outcome)
               + lag_* covariate columns at t (kept as-is, they are already lagged)
      outcome  = ICILS_computer_literacy at t+1
    """
    cycles = sorted(ep["cycle"].unique())
    covariate_cols = [c for c in ep.columns
                      if c.startswith("lag_")]
    LAG_OUTCOME = "lag_outcome"

    rows = []
    for i in range(len(cycles) - 1):
        t_lag, t_cur = cycles[i], cycles[i + 1]
        lag_df = ep[ep["cycle"] == t_lag][
            ["country_iso3", OUTCOME_COL] + covariate_cols
        ].copy().rename(columns={OUTCOME_COL: LAG_OUTCOME})
        cur_df = ep[ep["cycle"] == t_cur][["country_iso3", OUTCOME_COL]].copy()
        merged = lag_df.merge(cur_df, on="country_iso3")
        merged["test_cycle"] = t_cur
        rows.append(merged)

    if not rows:
        return pd.DataFrame(), []

    train = pd.concat(rows, ignore_index=True)
    feature_names = [LAG_OUTCOME] + covariate_cols
    return train, feature_names


def fit_and_predict(
    train: pd.DataFrame,
    pred_features: pd.DataFrame,
    feature_cols: list[str],
    wj_map: dict[str, float],
    use_weights: bool,
) -> np.ndarray:
    """Fit Ridge on training pairs; predict on pred_features."""
    X_train = train[feature_cols].copy()
    y_train = train[OUTCOME_COL].copy()

    valid = y_train.notna() & X_train[feature_cols[0]].notna()
    X_train, y_train = X_train[valid].copy(), y_train[valid].copy()

    col_medians = {col: X_train[col].median() for col in X_train.columns}
    for col in X_train.columns:
        X_train[col] = X_train[col].fillna(col_medians[col])

    if use_weights:
        for col in X_train.columns:
            X_train[col] = X_train[col] * wj_map.get(col, 1.0)

    X_pred = pred_features[feature_cols].copy()
    for col in X_pred.columns:
        X_pred[col] = X_pred[col].fillna(col_medians.get(col, 0))
    if use_weights:
        for col in X_pred.columns:
            X_pred[col] = X_pred[col] * wj_map.get(col, 1.0)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", RidgeCV(alphas=[0.01, 0.1, 1, 10, 100, 1000])),
    ])
    pipe.fit(X_train, y_train)
    return pipe.predict(X_pred)


def main():
    ep = pd.read_csv(STAGE5 / "icils_enriched_panel.csv")
    train_df, feature_cols = build_pairs(ep)

    if train_df.empty:
        print("ERROR: no training pairs built — check enriched panel")
        return

    print(f"Training pairs: {len(train_df)} rows from cycles "
          f"{sorted(train_df['test_cycle'].unique())}")
    print(f"Feature columns: {feature_cols}")

    # Prediction base: 2023 scores → features for 2028 forecast
    covariate_cols = [c for c in feature_cols if c != "lag_outcome"]
    ep_2023 = ep[ep["cycle"] == LAST_CYCLE].copy()
    pred_base = ep_2023[["country_iso3", OUTCOME_COL] + covariate_cols].copy()
    pred_base = pred_base.rename(columns={OUTCOME_COL: "lag_outcome"})

    n_countries = len(pred_base)
    print(f"Predicting {n_countries} countries for cycle {PREDICTED_CYCLE}")

    wj_map = load_wj_map()
    print(f"W_j map (√w): { {k: round(v,4) for k,v in wj_map.items()} }")

    y_m0 = fit_and_predict(train_df, pred_base, feature_cols, wj_map, use_weights=False)
    y_m1 = fit_and_predict(train_df, pred_base, feature_cols, wj_map, use_weights=True)

    # Retrieve last_score (2023 actual) from estimates
    est_path = STAGE5 / "icils_piaac_estimates.csv"
    est = pd.read_csv(est_path) if est_path.exists() else pd.DataFrame()
    if not est.empty:
        est_2023 = (est[(est.program == "ICILS") &
                        (est.domain == "computer_literacy") &
                        (est.cycle == LAST_CYCLE)]
                    [["country_iso3", "mean"]]
                    .rename(columns={"mean": "last_score"}))
        pred_base = pred_base.merge(est_2023, on="country_iso3", how="left")
    else:
        pred_base["last_score"] = pred_base[f"lag_{OUTCOME_COL}"]

    out = pd.DataFrame({
        "program":         "ICILS",
        "domain":          "computer_literacy",
        "country_iso3":    pred_base["country_iso3"].values,
        "last_cycle":      LAST_CYCLE,
        "last_score":      pred_base["last_score"].round(4).values,
        "predicted_cycle": PREDICTED_CYCLE,
        "y_M0":            y_m0.round(4),
        "y_M1":            y_m1.round(4),
        "validation_design": "pooled_train_holdout_test",
        "n_train_pairs":   len(train_df),
        "note":            "trained on 2013→2018 + 2018→2023 transitions; forecast horizon 2028",
    })
    out = out.sort_values("country_iso3").reset_index(drop=True)

    out_path = STAGE5 / "icils_forward_predictions.csv"
    out.to_csv(out_path, index=False)
    print(f"\nSaved {len(out)} rows → {out_path}")
    print("\nSample:")
    print(out[["country_iso3", "last_score", "y_M0", "y_M1"]].head(10).to_string(index=False))

    # Also merge into the main forward_predictions.csv
    fp_path = STAGE5 / "forward_predictions.csv"
    if fp_path.exists():
        fp = pd.read_csv(fp_path)
        # Remove any stale ICILS rows
        fp = fp[fp["program"] != "ICILS"]
        merge_cols = ["program", "domain", "country_iso3", "last_cycle",
                      "last_score", "predicted_cycle", "y_M0", "y_M1"]
        icils_for_merge = out[merge_cols].copy()
        # Add y_M3 column to match forward_predictions schema
        icils_for_merge["y_M3"] = np.nan
        fp_updated = pd.concat([fp, icils_for_merge], ignore_index=True)
        fp_updated = fp_updated.sort_values(["program", "domain", "country_iso3"])
        fp_updated.to_csv(fp_path, index=False)
        print(f"\nUpdated forward_predictions.csv: {len(fp)} → {len(fp_updated)} rows (+{len(out)} ICILS)")


if __name__ == "__main__":
    main()
