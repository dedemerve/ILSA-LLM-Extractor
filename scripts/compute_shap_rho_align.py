#!/usr/bin/env python3
"""
SHAP + dual ρ_align analizi

Adımlar:
  1. Her LOCO fold'da Ridge (M0B) eğit; LinearExplainer ile SHAP hesapla
  2. w_XAI[j] = K foldda mean(|SHAP_j|) ortalaması (fold-weighted)
  3. w_lit[j] = crosswalk tier → Tier A=1.0, B=0.5, C=0.25 (lag_Y ← A=1.0)
  4. F* = F_lit ∩ F_XAI ∩ F_available (her program×domain için ayrı)
  5. Dual alignment:
       A. Full:  F_all  (lag_Y dahil)
       B. Lit:   F_lit  (lag_Y hariç — yalnızca substantive literature constructs)
  6. ρ_align = Spearman(w_lit[F*], w_XAI[F*])
  7. Fold-level ρ_align bootstrap CI (fold dağılımından)

Çıktılar:
  outputs/stage5/shap_fold_importance.csv    — fold × feature × mean|SHAP|
  outputs/stage5/shap_aggregate.csv          — program×domain × feature özeti
  outputs/stage5/rho_align_results.csv       — ρ_align (full + lit, + CI)
"""
from __future__ import annotations

import logging
import pathlib
import warnings

import numpy as np
import pandas as pd
import shap
from scipy import stats
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT_DIR      = PROJECT_ROOT / "outputs" / "stage5"

RIDGE_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
MIN_TRAIN_N  = 10

# Literature weight: W_j_forecast — corpus-based priority
# Kaynak: build_literature_priority.py → outputs/stage5/literature_priority.csv
# W_j = geometric_mean(w_F, w_P, w_C, w_N) × w_E, forecastability-filtered
# 132 makale / 530 evidence satırı / FAZ 4 (2026-10-04)
CONSTRUCT_LIT_WEIGHT = {
    "SES_COMPOSITE":                   1.0000,  # 54 studies; highest diversity
    "TEACHER_SELF_EFFICACY_CLS_MGMT":  0.8543,  # 21 studies; 6-program diversity
    "HOME_RESOURCES":                  0.7175,  # 19 studies; 3 programs
    "SCHOOL_BELONGING":                0.6185,  # 27 studies; 2 programs (PISA-dominant)
    "ICT_HOME_ACCESS":                 0.5370,  # 15 studies
    "DISCIPLINARY_CLIMATE":            0.4910,  # 17 studies; direction inconsistency noted
    "PARENTAL_EDUCATION":              0.4559,  # 7 studies — lower than prior estimate
    "TEACHER_SELF_EFFICACY_INSTRUCTION": 0.8543,  # same corpus as CLS_MGMT
    "TEACHER_SELF_EFFICACY_ENGAGEMENT": 0.8543,
    "ICT_SCHOOL_ACCESS":               0.2818,  # 3 studies
    "EFFECTIVE_PD":                    0.2547,  # 3 studies
    "HOME_LITERACY_ACTIVITIES":        0.1029,  # 1 study (PIRLS-specific)
    "JOB_SATISFACTION_ENVIRONMENT":    0.0894,
    "JOB_SATISFACTION_PROFESSION":     0.0894,  # 1 study
    "PROF_COLLABORATION":              0.2547,  # no direct corpus entry; proxy = EFFECTIVE_PD
}
# lag_Y: temporal persistence — not a substantive literature construct
LAG_Y_LIT_WEIGHT = 1.00   # used in full alignment only; excluded from lit alignment

def get_lit_weight(feature: str, tier_map: dict) -> float:
    if feature == "lag_Y":
        return LAG_Y_LIT_WEIGHT
    if feature in CONSTRUCT_LIT_WEIGHT:
        return CONSTRUCT_LIT_WEIGHT[feature]
    tier = tier_map.get(feature, "C")
    return {"A": 0.80, "B": 0.50, "C": 0.25}.get(tier, 0.25)

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


def load_data():
    v2    = pd.read_csv(OUT_DIR / "enriched_panel_v2.csv")
    xwalk = pd.read_csv(OUT_DIR / "canonical_crosswalk.csv")

    drop_c = set(xwalk.loc[xwalk["transformation"] == "drop_near_zero_variance",
                            "canonical_construct"])
    v2 = v2[~v2["canonical_construct"].isin(drop_c)].copy()

    negate_c = set(xwalk.loc[xwalk["transformation"] == "negate", "canonical_construct"])
    mask = v2["canonical_construct"].isin(negate_c) & (v2["forecast_role"] != "outcome")
    v2.loc[mask, "value"] = v2.loc[mask, "value"] * -1

    tier_map = dict(zip(xwalk["canonical_construct"], xwalk["tier"]))
    tier_map["lag_Y"] = "A"
    return v2, tier_map


def select_train_cols(df, feat_cols):
    avail = [c for c in feat_cols if c in df.columns]
    if not avail:
        return [], np.empty((0,0)), np.empty(0)
    X = df[avail].values.astype(float)
    y = df["Y"].values.astype(float)
    col_valid = ~np.all(np.isnan(X), axis=0)
    X = X[:, col_valid]
    avail = [a for a, v in zip(avail, col_valid) if v]
    row_valid = ~np.any(np.isnan(X), axis=1)
    return avail, X[row_valid], y[row_valid]


def build_Xy_fixed(df, fixed_cols):
    X_parts = [df[c].values.astype(float) if c in df.columns
               else np.full(len(df), np.nan) for c in fixed_cols]
    X = np.column_stack(X_parts) if X_parts else np.empty((len(df), 0))
    y = df["Y"].values.astype(float)
    row_valid = ~np.any(np.isnan(X), axis=1)
    return X[row_valid], y[row_valid]


def bootstrap_spearman_ci(x, y, n_boot=500, alpha=0.05, rng_seed=42):
    """Bootstrap CI for Spearman rho."""
    rng = np.random.default_rng(rng_seed)
    n = len(x)
    if n < 3:
        return np.nan, np.nan
    boot_rhos = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        r, _ = stats.spearmanr(x[idx], y[idx])
        boot_rhos.append(r)
    lo = float(np.percentile(boot_rhos, 100 * alpha / 2))
    hi = float(np.percentile(boot_rhos, 100 * (1 - alpha / 2)))
    return lo, hi


def main():
    v2, tier_map = load_data()
    log.info("Panel v2: %d satır yüklendi", len(v2))

    shap_fold_records  = []
    rho_align_records  = []

    for prog in ["PISA", "TIMSS", "TIMSS_G4", "PIRLS"]:
        for domain in DOMAINS[prog]:
            outcome_col = OUTCOME_MAP[(prog, domain)]
            log.info("=== %s / %s ===", prog, domain)

            outcomes = v2[
                (v2["forecast_role"] == "outcome") &
                (v2["target_program"] == prog) &
                (v2["canonical_construct"] == outcome_col)
            ][["target_country","target_cycle","value"]].rename(
                columns={"target_country":"country","target_cycle":"cycle","value":"Y"})

            intra = v2[
                (v2["feature_source"] == "intra_program_lag") &
                (v2["target_program"] == prog)
            ][["target_country","target_cycle","canonical_construct","value"]].rename(
                columns={"target_country":"country","target_cycle":"cycle",
                         "canonical_construct":"construct"}
            ).pivot_table(index=["country","cycle"], columns="construct",
                          values="value", aggfunc="mean").reset_index()

            full = outcomes.merge(intra, on=["country","cycle"], how="left")
            full = full.sort_values(["country","cycle"]).copy()
            full["lag_Y"] = full.groupby("country")["Y"].shift(1)

            present_cycles = sorted(full["cycle"].dropna().unique())
            if len(present_cycles) < 3:
                continue

            intra_cols = [c for c in intra.columns if c not in ("country","cycle")]
            feat_m0b   = ["lag_Y"] + intra_cols

            # Fold bazında SHAP biriktiricisi: {feature: [mean_abs_shap per fold]}
            fold_shap: dict[str, list[float]] = {}
            fold_rho_full = []
            fold_rho_lit  = []

            for i, test_cycle in enumerate(present_cycles):
                if i < 2:
                    continue
                train_cycles = [c for c in present_cycles if c < test_cycle]
                train = full[full["cycle"].isin(train_cycles)].dropna(subset=["Y"])
                test  = full[full["cycle"] == test_cycle].dropna(subset=["Y"])

                used_cols, X_tr, y_tr = select_train_cols(train, feat_m0b)
                if not used_cols or len(y_tr) < MIN_TRAIN_N:
                    continue

                X_te, y_te = build_Xy_fixed(test, used_cols)
                if len(y_te) < 3:
                    continue

                scaler  = StandardScaler()
                X_tr_s  = scaler.fit_transform(X_tr)
                X_te_s  = scaler.transform(X_te)

                ridge = RidgeCV(alphas=RIDGE_ALPHAS, cv=min(len(y_tr), 5))
                ridge.fit(X_tr_s, y_tr)

                # SHAP — LinearExplainer (tam hesaplama, sampling yok)
                explainer  = shap.LinearExplainer(ridge, X_tr_s, feature_names=used_cols)
                shap_vals  = explainer.shap_values(X_te_s)   # (n_test, n_features)
                mean_abs   = np.abs(shap_vals).mean(axis=0)   # (n_features,)

                # Fold kaydı
                for j, col in enumerate(used_cols):
                    shap_fold_records.append({
                        "program":    prog,
                        "domain":     domain,
                        "test_cycle": test_cycle,
                        "feature":    col,
                        "tier":       tier_map.get(col, "?"),
                        "lit_weight": get_lit_weight(col, tier_map),
                        "mean_abs_shap": float(mean_abs[j]),
                        "train_n":    len(y_tr),
                        "test_n":     len(y_te),
                        "n_features": len(used_cols),
                    })
                    if col not in fold_shap:
                        fold_shap[col] = []
                    fold_shap[col].append(float(mean_abs[j]))

                # Fold-level ρ_align — sadece bu fold'daki feature'lar
                fold_lw  = np.array([get_lit_weight(c, tier_map) for c in used_cols])
                fold_lw_nolagy = np.array([
                    get_lit_weight(c, tier_map) for c in used_cols if c != "lag_Y"
                ])
                shap_nolagy = mean_abs[[j for j, c in enumerate(used_cols) if c != "lag_Y"]]

                if len(used_cols) >= 2:
                    r, _ = stats.spearmanr(fold_lw, mean_abs)
                    fold_rho_full.append(r)
                if len(shap_nolagy) >= 2:
                    r2, _ = stats.spearmanr(fold_lw_nolagy, shap_nolagy)
                    fold_rho_lit.append(r2)

            if not fold_shap:
                continue

            # Aggregate w_XAI: fold ortalaması (her feature kendi n_fold sayısı ile)
            w_xai = {feat: float(np.mean(vals)) for feat, vals in fold_shap.items()}
            all_feats = list(w_xai.keys())

            # ── Dual alignment ──────────────────────────────────────────────
            for mode, exclude_lagy in [("full", False), ("lit", True)]:
                feats = [f for f in all_feats if not (exclude_lagy and f == "lag_Y")]
                if len(feats) < 2:
                    continue

                xai_vec = np.array([w_xai[f] for f in feats])
                lit_vec = np.array([get_lit_weight(f, tier_map) for f in feats])

                rho, pval = stats.spearmanr(lit_vec, xai_vec)

                # Bootstrap CI
                ci_lo, ci_hi = bootstrap_spearman_ci(lit_vec, xai_vec)

                # Fold-level dağılım
                fold_rhos = fold_rho_lit if exclude_lagy else fold_rho_full
                fold_rho_arr = np.array([r for r in fold_rhos if np.isfinite(r)])

                rho_align_records.append({
                    "program":       prog,
                    "domain":        domain,
                    "mode":          mode,
                    "n_features":    len(feats),
                    "features":      "|".join(feats),
                    "rho_align":     round(float(rho), 4),
                    "p_value":       round(float(pval), 4),
                    "ci_lo_95":      round(ci_lo, 4),
                    "ci_hi_95":      round(ci_hi, 4),
                    "n_folds":       len(fold_rho_arr),
                    "fold_rho_mean": round(float(fold_rho_arr.mean()), 4) if len(fold_rho_arr) else np.nan,
                    "fold_rho_sd":   round(float(fold_rho_arr.std()),  4) if len(fold_rho_arr) else np.nan,
                    "fold_rho_min":  round(float(fold_rho_arr.min()),  4) if len(fold_rho_arr) else np.nan,
                    "fold_rho_max":  round(float(fold_rho_arr.max()),  4) if len(fold_rho_arr) else np.nan,
                })

    # ── Kaydet ───────────────────────────────────────────────────────────────
    shap_df = pd.DataFrame(shap_fold_records)
    rho_df  = pd.DataFrame(rho_align_records)

    shap_df.to_csv(OUT_DIR / "shap_fold_importance.csv", index=False)
    rho_df.to_csv(OUT_DIR / "rho_align_results.csv", index=False)
    log.info("Kaydedildi: shap_fold_importance.csv (%d satır), rho_align_results.csv (%d satır)",
             len(shap_df), len(rho_df))

    # ── Özet tabloları ────────────────────────────────────────────────────────
    print("\n=== Aggregate SHAP importance — program×domain ===")
    agg = (
        shap_df.groupby(["program","domain","feature","tier","lit_weight"])
        ["mean_abs_shap"].mean().round(3).reset_index()
    )
    for (prog, dom), grp in agg.groupby(["program","domain"]):
        print(f"\n{prog}/{dom}:")
        top = grp.sort_values("mean_abs_shap", ascending=False)
        print(top[["feature","tier","lit_weight","mean_abs_shap"]].to_string(index=False))

    print("\n=== ρ_align (Spearman: lit_weight vs mean|SHAP|) ===")
    print(rho_df[["program","domain","mode","n_features","rho_align","p_value",
                  "ci_lo_95","ci_hi_95","fold_rho_mean","fold_rho_sd"]].to_string(index=False))

    # Önemli bulgular
    print("\n=== Full vs Lit alignment karşılaştırması ===")
    full_r = rho_df[rho_df["mode"]=="full"][["program","domain","rho_align"]].rename(
        columns={"rho_align":"rho_full"})
    lit_r  = rho_df[rho_df["mode"]=="lit"][["program","domain","rho_align"]].rename(
        columns={"rho_align":"rho_lit"})
    comp = full_r.merge(lit_r, on=["program","domain"])
    comp["delta"] = comp["rho_lit"] - comp["rho_full"]
    print(comp.to_string(index=False))
    print("\nNotlar:")
    print("  rho_full: lag_Y dahil alignment (temporal persistence + literature features)")
    print("  rho_lit:  lag_Y hariç alignment (yalnızca substantive literature constructs)")
    print("  Pozitif delta: lag_Y hariç alignment daha yüksek (lag_Y literature rank ile uyumsuz)")


if __name__ == "__main__":
    main()
