#!/usr/bin/env python3
"""
Step 12.5 — M0B / M1A Fold-Level Feature Lineage Audit

Her LOCO fold için:
  - Hangi feature'lar M0B ve M1A'da kullanıldı?
  - Her feature'in tier ve literature_weight değeri neydi?
  - M1A'da fiilen sıfırlanan (weight=0) feature var mı?
  - M0B ile M1A arasında feature seti aynı mı, farklılaşma salt ağırlık üzerinden mi?

Çıktılar:
  outputs/stage5/feature_audit_fold.csv     — fold × feature detay
  outputs/stage5/feature_audit_summary.csv  — fold özeti (n_feat, delta RMSE)

Yorumsal amaç:
  M1A'daki RMSE farkının "literature weighting" mi yoksa "feature suppression" mı
  kaynaklı olduğunu ayırt etmek. SHAP analizi bu audit üzerinden yorumlanacak.
"""
from __future__ import annotations

import logging
import pathlib
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT_DIR      = PROJECT_ROOT / "outputs" / "stage5"

RIDGE_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
MIN_TRAIN_N  = 10
TIER_WEIGHTS = {"A": 1.0, "B": 0.5, "C": 0.25}

CYCLE_ORDER = {
    "PISA":     [2003, 2006, 2009, 2012, 2015, 2018, 2022, 2025],
    "TIMSS":    [1995, 1999, 2003, 2007, 2011, 2015, 2019, 2023],
    "TIMSS_G4": [1995, 2003, 2007, 2011, 2015, 2019, 2023],
    "PIRLS":    [2001, 2006, 2011, 2016, 2021],
}
DOMAINS = {
    "PISA":     ["mathematics", "reading", "science"],
    "TIMSS":    ["mathematics", "science"],
    "TIMSS_G4": ["mathematics", "science"],
    "PIRLS":    ["reading"],
}
OUTCOME_MAP = {
    ("PISA",     "mathematics"): "PISA_MATHEMATICS",
    ("PISA",     "reading"):     "PISA_READING",
    ("PISA",     "science"):     "PISA_SCIENCE",
    ("TIMSS",    "mathematics"): "TIMSS_MATHEMATICS",
    ("TIMSS",    "science"):     "TIMSS_SCIENCE",
    ("TIMSS_G4", "mathematics"): "TIMSS_G4_MATHEMATICS",
    ("TIMSS_G4", "science"):     "TIMSS_G4_SCIENCE",
    ("PIRLS",    "reading"):     "PIRLS_READING",
}


def load_and_transform():
    v2    = pd.read_csv(OUT_DIR / "enriched_panel_v2.csv")
    xwalk = pd.read_csv(OUT_DIR / "canonical_crosswalk.csv")

    drop_c = set(xwalk.loc[xwalk["transformation"] == "drop_near_zero_variance",
                            "canonical_construct"])
    v2 = v2[~v2["canonical_construct"].isin(drop_c)].copy()

    negate_c = set(xwalk.loc[xwalk["transformation"] == "negate", "canonical_construct"])
    mask = v2["canonical_construct"].isin(negate_c) & (v2["forecast_role"] != "outcome")
    v2.loc[mask, "value"] = v2.loc[mask, "value"] * -1

    tier_map = dict(zip(xwalk["canonical_construct"], xwalk["tier"]))
    tier_map["lag_Y"] = "A"   # prior achievement: literature unanimous → tier A
    return v2, xwalk, tier_map


def select_train_cols(df, feat_cols):
    avail = [c for c in feat_cols if c in df.columns]
    if not avail:
        return [], np.empty((0, 0)), np.empty(0)
    X = df[avail].values.astype(float)
    y = df["Y"].values.astype(float)
    col_valid = ~np.all(np.isnan(X), axis=0)
    X = X[:, col_valid]
    avail = [a for a, v in zip(avail, col_valid) if v]
    row_valid = ~np.any(np.isnan(X), axis=1)
    return avail, X[row_valid], y[row_valid]


def build_Xy_fixed(df, fixed_cols):
    X_parts = []
    for c in fixed_cols:
        X_parts.append(df[c].values.astype(float) if c in df.columns
                       else np.full(len(df), np.nan))
    X = np.column_stack(X_parts) if X_parts else np.empty((len(df), 0))
    y = df["Y"].values.astype(float)
    row_valid = ~np.any(np.isnan(X), axis=1)
    return X[row_valid], y[row_valid]


def run_audit(v2, xwalk, tier_map):
    fold_records   = []   # fold × feature satırları
    summary_records = []  # fold özeti

    for prog in ["PISA", "TIMSS", "TIMSS_G4", "PIRLS"]:
        cycles  = CYCLE_ORDER[prog]
        domains = DOMAINS[prog]

        for domain in domains:
            outcome_col = OUTCOME_MAP[(prog, domain)]

            outcomes = v2[
                (v2["forecast_role"] == "outcome") &
                (v2["target_program"] == prog) &
                (v2["canonical_construct"] == outcome_col)
            ][["target_country","target_cycle","value"]].rename(
                columns={"target_country":"country","target_cycle":"cycle","value":"Y"})
            if outcomes.empty:
                continue

            intra = v2[
                (v2["feature_source"] == "intra_program_lag") &
                (v2["target_program"] == prog)
            ][["target_country","target_cycle","canonical_construct","value"]].rename(
                columns={"target_country":"country","target_cycle":"cycle",
                         "canonical_construct":"construct"}
            ).pivot_table(index=["country","cycle"], columns="construct",
                          values="value", aggfunc="mean").reset_index()

            talis = v2[
                (v2["feature_source"] == "talis_cross_program") &
                (v2["target_program"] == prog)
            ][["target_country","target_cycle","canonical_construct","value"]].rename(
                columns={"target_country":"country","target_cycle":"cycle",
                         "canonical_construct":"construct"}
            ).pivot_table(index=["country","cycle"], columns="construct",
                          values="value", aggfunc="mean").reset_index()

            base = outcomes.merge(intra, on=["country","cycle"], how="left")
            full = base.merge(talis,  on=["country","cycle"], how="left",
                              suffixes=("","_talis"))
            full = full.sort_values(["country","cycle"]).copy()
            full["lag_Y"] = full.groupby("country")["Y"].shift(1)

            present_cycles = sorted(full["cycle"].dropna().unique())
            if len(present_cycles) < 3:
                continue

            intra_cols = [c for c in intra.columns if c not in ("country","cycle")]
            talis_cols = [c for c in talis.columns if c not in ("country","cycle")]
            feat_m0b = ["lag_Y"] + intra_cols
            feat_m1a = ["lag_Y"] + intra_cols   # aynı feature seti, farklı ağırlık

            for i, test_cycle in enumerate(present_cycles):
                if i < 2:
                    continue
                train_cycles = [c for c in present_cycles if c < test_cycle]
                train = full[full["cycle"].isin(train_cycles)].dropna(subset=["Y"])
                test  = full[full["cycle"] == test_cycle].dropna(subset=["Y"])

                if len(train) < MIN_TRAIN_N or len(test) < 3:
                    continue

                for model_name, feat_cols, use_tier in [
                    ("M0B", feat_m0b, False),
                    ("M1A", feat_m1a, True),
                ]:
                    used_cols, X_tr, y_tr = select_train_cols(train, feat_cols)
                    if not used_cols or len(y_tr) < MIN_TRAIN_N:
                        continue

                    X_te, y_te = build_Xy_fixed(test, used_cols)
                    if len(y_te) < 3:
                        continue

                    scaler = StandardScaler()
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        X_tr_s = scaler.fit_transform(X_tr)
                        X_te_s = scaler.transform(X_te)

                    tier_weights = np.array([
                        TIER_WEIGHTS.get(tier_map.get(c, "C"), 0.25)
                        for c in used_cols
                    ])

                    if use_tier:
                        X_tr_s = X_tr_s * tier_weights
                        X_te_s = X_te_s * tier_weights

                    ridge = RidgeCV(alphas=RIDGE_ALPHAS, cv=min(len(y_tr), 5))
                    ridge.fit(X_tr_s, y_tr)
                    y_pred = ridge.predict(X_te_s)

                    residuals = y_te - y_pred
                    rmse = float(np.sqrt(np.mean(residuals**2)))
                    r2   = float(1 - np.sum(residuals**2) / np.sum((y_te - y_te.mean())**2))
                    rho, _ = stats.spearmanr(y_te, y_pred)

                    # Feature-level audit satırları
                    for j, col in enumerate(used_cols):
                        tw = float(tier_weights[j])
                        coef = float(ridge.coef_[j])
                        fold_records.append({
                            "program":          prog,
                            "domain":           domain,
                            "test_cycle":       test_cycle,
                            "model":            model_name,
                            "canonical_feature": col,
                            "tier":             tier_map.get(col, "?"),
                            "literature_weight": tw if use_tier else 1.0,
                            "ridge_coef":       coef,
                            # Efektif katsayı: ağırlıklı standart boşlukta coef
                            "effective_coef":   coef * tw if use_tier else coef,
                            "effectively_zero": abs(coef * tw) < 1e-6 if use_tier else False,
                            "train_n":          len(y_tr),
                            "test_n":           len(y_te),
                            "alpha_selected":   float(ridge.alpha_),
                        })

                    # Fold özeti
                    # Feature suppression: M1A'da tier weight < 1 olan feature'lar
                    suppressed = [c for c, w in zip(used_cols, tier_weights) if w < 1.0] if use_tier else []

                    summary_records.append({
                        "program":     prog,
                        "domain":      domain,
                        "test_cycle":  test_cycle,
                        "model":       model_name,
                        "n_features":  len(used_cols),
                        "n_tier_A":    sum(1 for c in used_cols if tier_map.get(c,"?") == "A"),
                        "n_tier_B":    sum(1 for c in used_cols if tier_map.get(c,"?") == "B"),
                        "n_suppressed_below1": len(suppressed),
                        "suppressed_features": "|".join(sorted(suppressed)) if suppressed else "",
                        "train_n":     len(y_tr),
                        "test_n":      len(y_te),
                        "rmse":        rmse,
                        "r2":          round(r2, 4),
                        "spearman":    round(float(rho), 4),
                        "alpha":       float(ridge.alpha_),
                    })

    return pd.DataFrame(fold_records), pd.DataFrame(summary_records)


def main():
    v2, xwalk, tier_map = load_and_transform()

    fold_df, summary_df = run_audit(v2, xwalk, tier_map)

    # M0B vs M1A delta RMSE per fold
    m0b = summary_df[summary_df["model"] == "M0B"][
        ["program","domain","test_cycle","rmse","n_features"]
    ].rename(columns={"rmse":"rmse_m0b","n_features":"n_feat_m0b"})
    m1a = summary_df[summary_df["model"] == "M1A"][
        ["program","domain","test_cycle","rmse","n_features","n_suppressed_below1",
         "suppressed_features","n_tier_A","n_tier_B"]
    ].rename(columns={"rmse":"rmse_m1a","n_features":"n_feat_m1a"})

    delta = m0b.merge(m1a, on=["program","domain","test_cycle"], how="outer")
    delta["delta_rmse"]   = delta["rmse_m0b"] - delta["rmse_m1a"]  # pozitif = M1A daha iyi
    delta["same_features"] = delta["n_feat_m0b"] == delta["n_feat_m1a"]

    # Kaydet
    fold_df.to_csv(OUT_DIR / "feature_audit_fold.csv", index=False)
    delta.to_csv(OUT_DIR / "feature_audit_summary.csv", index=False)
    log.info("Kaydedildi: feature_audit_fold.csv (%d satır), feature_audit_summary.csv (%d satır)",
             len(fold_df), len(delta))

    # ── Raporla ──────────────────────────────────────────────────────────────
    print("\n=== Fold Feature Seti: M0B vs M1A Aynı mı? ===")
    print(delta[["program","domain","test_cycle","n_feat_m0b","n_feat_m1a",
                 "same_features","n_suppressed_below1"]].to_string(index=False))

    print("\n=== Delta RMSE (M0B - M1A) — pozitif: M1A daha iyi ===")
    print(delta[["program","domain","test_cycle","rmse_m0b","rmse_m1a",
                 "delta_rmse","suppressed_features"]].to_string(index=False))

    print("\n=== M1A'da tier<A ağırlığıyla 'bastırılan' feature'lar ===")
    suppressed_summary = (
        fold_df[fold_df["model"] == "M1A"]
        .groupby(["canonical_feature","tier","literature_weight"])
        ["program"].count()
        .reset_index(name="n_folds_used")
        .sort_values("literature_weight")
    )
    print(suppressed_summary.to_string(index=False))

    print("\n=== Önemli bulgu: M0B ile M1A aynı feature setini mi kullanıyor? ===")
    all_same = delta["same_features"].all()
    print(f"Tüm foldlarda feature seti aynı: {all_same}")
    if all_same:
        print("  → M1A ile M0B arasındaki RMSE farkı SALT tier weighting'den kaynaklanıyor.")
        print("  → Feature suppression (sıfır) yok; fark ağırlıklandırma etkisi.")
    else:
        diff = delta[~delta["same_features"]]
        print(f"  Feature seti farklı olan fold sayısı: {len(diff)}")
        print(diff[["program","domain","test_cycle","n_feat_m0b","n_feat_m1a"]].to_string(index=False))


if __name__ == "__main__":
    main()
