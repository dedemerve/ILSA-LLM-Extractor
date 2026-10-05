#!/usr/bin/env python3
"""
Permutation / robustness analizi — dört test

1. Literature-weight permutation
   Null: w_lit rastgele permute edildiğinde ρ_lit nasıl dağılır?
   → Gözlenen PISA math ρ_lit < 0 şans eseri mi?

2. Label permutation (country shuffle)
   Null: Y etiketleri ülkeler arası rastgele karıştırıldığında RMSE ne olur?
   → M0B RMSE < permuted RMSE ise model gerçek sinyali yakalıyor.

3. OECD vs non-OECD subgroup
   M0B RMSE ve lag_Y SHAP: iki alt grupta farklı mı?

4. Feature-availability sensitivity
   Her feature sırayla kaldırıldığında ρ_lit nasıl değişiyor?
   (PISA math için — en kırılgan durum)

Çıktılar:
  outputs/stage5/perm_lit_weight.csv    — Test 1
  outputs/stage5/perm_label.csv         — Test 2
  outputs/stage5/subgroup_oecd.csv      — Test 3
  outputs/stage5/feature_dropout_rho.csv — Test 4
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
warnings.filterwarnings("ignore")

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT_DIR      = PROJECT_ROOT / "outputs" / "stage5"

RIDGE_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
MIN_TRAIN_N  = 10
N_PERM       = 1000
RNG_SEED     = 42

# OECD üye ülkeler (ISO-3)
OECD = {
    "AUS","AUT","BEL","CAN","CHL","COL","CRI","CZE","DNK","EST","FIN","FRA",
    "DEU","GRC","HUN","ISL","IRL","ISR","ITA","JPN","KOR","LVA","LTU","LUX",
    "MEX","NLD","NZL","NOR","POL","PRT","SVK","SVN","ESP","SWE","CHE","TUR",
    "GBR","USA",
}

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

# w_FDC (composite) — sensitivity analizinin birincil speci
W_LIT = {
    "SES_COMPOSITE":    1.00, "PARENTAL_EDUCATION": 0.95,
    "HOME_RESOURCES":   0.85, "SCHOOL_BELONGING":   0.80,
    "HOME_LITERACY_ACTIVITIES": 0.55,
    "ICT_HOME_ACCESS":  0.40, "ICT_SCHOOL_ACCESS":  0.40,
    "TEACHER_SELF_EFFICACY_CLS_MGMT": 0.70,
    "TEACHER_SELF_EFFICACY_INSTRUCTION": 0.70,
    "TEACHER_SELF_EFFICACY_ENGAGEMENT": 0.65,
    "DISCIPLINARY_CLIMATE": 0.75, "PROF_COLLABORATION": 0.55,
    "JOB_SATISFACTION_ENVIRONMENT": 0.50, "JOB_SATISFACTION_PROFESSION": 0.50,
    "EFFECTIVE_PD": 0.60,
}


# ── Yardımcı fonksiyonlar ─────────────────────────────────────────────────────

def load_data():
    v2    = pd.read_csv(OUT_DIR / "enriched_panel_v2.csv")
    xwalk = pd.read_csv(OUT_DIR / "canonical_crosswalk.csv")
    drop_c   = set(xwalk.loc[xwalk["transformation"] == "drop_near_zero_variance", "canonical_construct"])
    negate_c = set(xwalk.loc[xwalk["transformation"] == "negate", "canonical_construct"])
    v2 = v2[~v2["canonical_construct"].isin(drop_c)].copy()
    mask = v2["canonical_construct"].isin(negate_c) & (v2["forecast_role"] != "outcome")
    v2.loc[mask, "value"] = v2.loc[mask, "value"] * -1
    tier_map = dict(zip(xwalk["canonical_construct"], xwalk["tier"]))
    tier_map["lag_Y"] = "A"
    return v2, tier_map


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


def build_Xy_fixed(df, fixed_cols, col_means=None):
    X_parts = [df[c].values.astype(float) if c in df.columns
               else np.full(len(df), np.nan) for c in fixed_cols]
    X = np.column_stack(X_parts) if X_parts else np.empty((len(df), 0))
    y = df["Y"].values.astype(float)
    row_valid = ~np.any(np.isnan(X), axis=1)
    return X[row_valid], y[row_valid], np.where(row_valid)[0]


def build_panel(v2, prog, domain):
    """Tek program×domain için pivot panel döndürür."""
    outcome_col = OUTCOME_MAP[(prog, domain)]
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
        columns={"target_country":"country","target_cycle":"cycle","canonical_construct":"construct"}
    ).pivot_table(index=["country","cycle"], columns="construct",
                  values="value", aggfunc="mean").reset_index()
    full = outcomes.merge(intra, on=["country","cycle"], how="left")
    full = full.sort_values(["country","cycle"]).copy()
    full["lag_Y"] = full.groupby("country")["Y"].shift(1)
    intra_cols = [c for c in intra.columns if c not in ("country","cycle")]
    return full, intra_cols


def loco_rmse(full, feat_cols):
    """LOCO expanding-window RMSE; returns list of fold RMSEs."""
    present_cycles = sorted(full["cycle"].dropna().unique())
    rmses = []
    for i, tc in enumerate(present_cycles):
        if i < 2:
            continue
        train = full[full["cycle"] < tc].dropna(subset=["Y"])
        test  = full[full["cycle"] == tc].dropna(subset=["Y"])
        used, X_tr, y_tr = select_train_cols(train, feat_cols)
        if not used or len(y_tr) < MIN_TRAIN_N:
            continue
        X_te, y_te, _ = build_Xy_fixed(test, used)
        if len(y_te) < 3:
            continue
        sc = StandardScaler()
        X_tr_s = sc.fit_transform(X_tr)
        X_te_s = sc.transform(X_te)
        ridge = RidgeCV(alphas=RIDGE_ALPHAS, cv=min(len(y_tr), 5))
        ridge.fit(X_tr_s, y_tr)
        resid = y_te - ridge.predict(X_te_s)
        rmses.append(float(np.sqrt(np.mean(resid**2))))
    return rmses


# ── TEST 1: Literature-weight permutation ─────────────────────────────────────

def test1_lit_weight_permutation(shap_df: pd.DataFrame) -> pd.DataFrame:
    """
    Null: w_lit rastgele permute → null ρ_lit dağılımı.
    Gözlenen ρ_lit'in permutation p-değerini hesaplar.
    """
    rng = np.random.default_rng(RNG_SEED)
    records = []

    for (prog, dom), grp in shap_df.groupby(["program","domain"]):
        agg = grp.groupby("feature")["mean_abs_shap"].mean()
        feats = [f for f in agg.index if f != "lag_Y"]
        if len(feats) < 2:
            continue

        xai_vec = np.array([agg[f] for f in feats])
        lit_vec = np.array([W_LIT.get(f, 0.25) for f in feats])

        obs_rho, _ = stats.spearmanr(lit_vec, xai_vec)

        perm_rhos = []
        for _ in range(N_PERM):
            perm_lit = rng.permutation(lit_vec)
            r, _ = stats.spearmanr(perm_lit, xai_vec)
            perm_rhos.append(r)
        perm_arr = np.array(perm_rhos)

        # Tek kuyruklu: obs_rho bu kadar küçük çıkma olasılığı (negatif için)
        p_left  = float(np.mean(perm_arr <= obs_rho))
        p_right = float(np.mean(perm_arr >= obs_rho))
        p_two   = float(np.mean(np.abs(perm_arr) >= abs(obs_rho)))

        records.append({
            "program":    prog,
            "domain":     dom,
            "n_features": len(feats),
            "obs_rho":    round(float(obs_rho), 4),
            "perm_mean":  round(float(perm_arr.mean()), 4),
            "perm_sd":    round(float(perm_arr.std()), 4),
            "perm_p_left":  round(p_left, 4),   # obs <= perm
            "perm_p_right": round(p_right, 4),
            "perm_p_two":   round(p_two, 4),
            "n_perm":     N_PERM,
        })

    return pd.DataFrame(records)


# ── TEST 2: Label permutation (country shuffle) ───────────────────────────────

def test2_label_permutation(v2) -> pd.DataFrame:
    """
    Null: Y etiketleri test fold'da rastgele karıştırılır.
    Gözlenen RMSE < null RMSE ise model sinyali yakalıyor.
    """
    rng = np.random.default_rng(RNG_SEED)
    records = []

    for prog in ["PISA", "TIMSS", "TIMSS_G4", "PIRLS"]:
        for domain in DOMAINS[prog]:
            full, intra_cols = build_panel(v2, prog, domain)
            feat_cols = ["lag_Y"] + intra_cols
            present_cycles = sorted(full["cycle"].dropna().unique())

            for i, tc in enumerate(present_cycles):
                if i < 2:
                    continue
                train = full[full["cycle"] < tc].dropna(subset=["Y"])
                test  = full[full["cycle"] == tc].dropna(subset=["Y"])
                used, X_tr, y_tr = select_train_cols(train, feat_cols)
                if not used or len(y_tr) < MIN_TRAIN_N:
                    continue
                X_te, y_te, _ = build_Xy_fixed(test, used)
                if len(y_te) < 3:
                    continue

                sc = StandardScaler()
                X_tr_s = sc.fit_transform(X_tr)
                X_te_s = sc.transform(X_te)
                ridge = RidgeCV(alphas=RIDGE_ALPHAS, cv=min(len(y_tr), 5))
                ridge.fit(X_tr_s, y_tr)
                y_pred = ridge.predict(X_te_s)
                obs_rmse = float(np.sqrt(np.mean((y_te - y_pred)**2)))

                # Null: permute y_te
                perm_rmses = []
                for _ in range(N_PERM):
                    y_perm = rng.permutation(y_te)
                    perm_rmses.append(float(np.sqrt(np.mean((y_perm - y_pred)**2))))
                perm_arr = np.array(perm_rmses)

                p_val = float(np.mean(perm_arr <= obs_rmse))  # model daha kötü oranı

                records.append({
                    "program":    prog, "domain": domain, "test_cycle": tc,
                    "obs_rmse":   round(obs_rmse, 3),
                    "perm_mean_rmse": round(float(perm_arr.mean()), 3),
                    "perm_sd_rmse":   round(float(perm_arr.std()), 3),
                    "p_better":       round(1 - p_val, 4),  # obs < perm oranı
                    "n_test":     len(y_te),
                })

    return pd.DataFrame(records)


# ── TEST 3: OECD vs non-OECD ─────────────────────────────────────────────────

def test3_oecd_subgroup(v2, shap_df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for prog in ["PISA", "TIMSS", "TIMSS_G4", "PIRLS"]:
        for domain in DOMAINS[prog]:
            full, intra_cols = build_panel(v2, prog, domain)
            full["oecd"] = full["country"].isin(OECD)
            feat_cols = ["lag_Y"] + intra_cols
            present_cycles = sorted(full["cycle"].dropna().unique())

            for group_name, mask_fn in [("OECD", lambda df: df[df["oecd"]]),
                                        ("non-OECD", lambda df: df[~df["oecd"]])]:
                group_rmses = []
                for i, tc in enumerate(present_cycles):
                    if i < 2:
                        continue
                    train_all = full[full["cycle"] < tc].dropna(subset=["Y"])
                    test_g    = mask_fn(full[full["cycle"] == tc].dropna(subset=["Y"]))
                    if len(test_g) < 3:
                        continue
                    used, X_tr, y_tr = select_train_cols(train_all, feat_cols)
                    if not used or len(y_tr) < MIN_TRAIN_N:
                        continue
                    X_te, y_te, _ = build_Xy_fixed(test_g, used)
                    if len(y_te) < 3:
                        continue
                    sc = StandardScaler()
                    X_tr_s = sc.fit_transform(X_tr)
                    X_te_s = sc.transform(X_te)
                    ridge = RidgeCV(alphas=RIDGE_ALPHAS, cv=min(len(y_tr), 5))
                    ridge.fit(X_tr_s, y_tr)
                    resid = y_te - ridge.predict(X_te_s)
                    group_rmses.append(float(np.sqrt(np.mean(resid**2))))

                if group_rmses:
                    records.append({
                        "program": prog, "domain": domain, "group": group_name,
                        "n_folds":     len(group_rmses),
                        "mean_rmse":   round(float(np.mean(group_rmses)), 3),
                        "median_rmse": round(float(np.median(group_rmses)), 3),
                        "sd_rmse":     round(float(np.std(group_rmses)), 3),
                    })

    return pd.DataFrame(records)


# ── TEST 4: Feature dropout — ρ_lit sensitivity (PISA math) ──────────────────

def test4_feature_dropout(shap_df: pd.DataFrame) -> pd.DataFrame:
    """
    PISA math (ve diğerleri) için her feature sırayla çıkarıldığında ρ_lit nasıl değişiyor?
    n=3 → n=2 olduğunda Spearman trivyal hale geliyor mu?
    """
    records = []
    for (prog, dom), grp in shap_df.groupby(["program","domain"]):
        agg = grp.groupby("feature")["mean_abs_shap"].mean()
        feats = [f for f in agg.index if f != "lag_Y"]
        if len(feats) < 3:
            continue

        xai_full = np.array([agg[f] for f in feats])
        lit_full  = np.array([W_LIT.get(f, 0.25) for f in feats])
        rho_full, _ = stats.spearmanr(lit_full, xai_full)

        records.append({
            "program": prog, "domain": dom,
            "dropped_feature": "none (baseline)",
            "n_remaining": len(feats),
            "rho_lit": round(float(rho_full), 4),
            "trivial": len(feats) == 2,
        })

        for drop_f in feats:
            remaining = [f for f in feats if f != drop_f]
            if len(remaining) < 2:
                continue
            xai_r = np.array([agg[f] for f in remaining])
            lit_r  = np.array([W_LIT.get(f, 0.25) for f in remaining])
            rho_r, _ = stats.spearmanr(lit_r, xai_r)
            records.append({
                "program": prog, "domain": dom,
                "dropped_feature": drop_f,
                "n_remaining": len(remaining),
                "rho_lit": round(float(rho_r), 4),
                "trivial": len(remaining) == 2,
            })

    return pd.DataFrame(records)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    v2, tier_map = load_data()
    shap_df = pd.read_csv(OUT_DIR / "shap_fold_importance.csv")

    log.info("Test 1: Literature-weight permutation (%d perms)...", N_PERM)
    perm_lit = test1_lit_weight_permutation(shap_df)
    perm_lit.to_csv(OUT_DIR / "perm_lit_weight.csv", index=False)

    log.info("Test 2: Label permutation...")
    perm_label = test2_label_permutation(v2)
    perm_label.to_csv(OUT_DIR / "perm_label.csv", index=False)

    log.info("Test 3: OECD vs non-OECD subgroup...")
    oecd_df = test3_oecd_subgroup(v2, shap_df)
    oecd_df.to_csv(OUT_DIR / "subgroup_oecd.csv", index=False)

    log.info("Test 4: Feature dropout ρ_lit sensitivity...")
    dropout_df = test4_feature_dropout(shap_df)
    dropout_df.to_csv(OUT_DIR / "feature_dropout_rho.csv", index=False)

    # ── Raporla ──────────────────────────────────────────────────────────────
    print("\n=== TEST 1: Lit-weight permutation (w_FDC) ===")
    print("(p_two: |ρ_obs| >= |ρ_perm| oranı — küçük = nadir)\n")
    print(perm_lit[["program","domain","n_features","obs_rho",
                     "perm_mean","perm_sd","perm_p_two"]].to_string(index=False))

    print("\n=== TEST 2: Label permutation — fold mean RMSE ===")
    fold_sum = (perm_label.groupby(["program","domain"])
                .agg(obs_rmse=("obs_rmse","mean"), perm_rmse=("perm_mean_rmse","mean"),
                     p_better=("p_better","mean"))
                .round(3).reset_index())
    print(fold_sum.to_string(index=False))

    print("\n=== TEST 3: OECD vs non-OECD mean RMSE ===")
    oecd_pivot = oecd_df.pivot_table(index=["program","domain"],
                                      columns="group", values="mean_rmse").reset_index()
    oecd_pivot.columns.name = None
    print(oecd_pivot.to_string(index=False))

    print("\n=== TEST 4: Feature dropout — PISA mathematics ρ_lit ===")
    pisa_math = dropout_df[(dropout_df["program"]=="PISA") &
                           (dropout_df["domain"]=="mathematics")]
    if not pisa_math.empty:
        print(pisa_math[["dropped_feature","n_remaining","rho_lit","trivial"]].to_string(index=False))
    else:
        print("  PISA math: n_features < 3, dropout analizi yapılamadı")

    print("\n=== Diğer programlar — feature dropout ===")
    others = dropout_df[~((dropout_df["program"]=="PISA") &
                          (dropout_df["domain"]=="mathematics"))]
    if not others.empty:
        print(others[["program","domain","dropped_feature","n_remaining","rho_lit","trivial"]]
              .to_string(index=False))


if __name__ == "__main__":
    main()
