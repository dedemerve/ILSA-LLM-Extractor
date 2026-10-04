#!/usr/bin/env python3
"""
Literature-weight sensitivity analysis for ρ_lit

Üç w_lit spesifikasyonu ile ρ_lit hesaplanır:

  A. w_F   — frequency-based: 10 ILSA ML makalesi içindeki görünme frekansı
  B. w_FD  — frequency × cross-program diversity: kaç farklı ILSA'da raporlandı
  C. w_FDC — mevcut composite priority (Sirin 2005 + OECD framework)

PISA mathematics ρ_lit = -1.0 bulgusunun weight specification'a bağımlılığını test eder.

Çıktı: outputs/stage5/rho_align_sensitivity.csv
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

# ── Üç literature weight spesifikasyonu ──────────────────────────────────────
# lag_Y: substantive literature predictor değil — tüm spec'lerde hariç (ρ_lit)

# A. Frequency-based (w_F)
# 10 ILSA ML makalesinde kaçında anlamlı predictor olarak raporlandı
# (Lee & Lee 2025, Zhu et al. 2025, Zhou et al. 2024, Zheng et al. 2023,
#  Song & Cutumisu 2024, Elouafi 2025, Alas & Tezer 2024,
#  Acisli-Celik & Yesilkanat 2023, Alkan et al. 2025, Aydogan & Tat 2025)
# Normalize: count / 10
W_FREQ = {
    "SES_COMPOSITE":                  1.00,  # 10/10 — ESCS / SES en sık raporlanan
    "PARENTAL_EDUCATION":             0.90,  # 9/10
    "HOME_RESOURCES":                 0.60,  # 6/10
    "SCHOOL_BELONGING":               0.40,  # 4/10
    "HOME_LITERACY_ACTIVITIES":       0.30,  # 3/10
    "ICT_HOME_ACCESS":                0.20,  # 2/10
    "ICT_SCHOOL_ACCESS":              0.20,  # 2/10
    "TEACHER_SELF_EFFICACY_CLS_MGMT": 0.30,
    "TEACHER_SELF_EFFICACY_INSTRUCTION": 0.30,
    "TEACHER_SELF_EFFICACY_ENGAGEMENT": 0.20,
    "DISCIPLINARY_CLIMATE":           0.40,
    "PROF_COLLABORATION":             0.20,
    "JOB_SATISFACTION_ENVIRONMENT":   0.10,
    "JOB_SATISFACTION_PROFESSION":    0.10,
    "EFFECTIVE_PD":                   0.20,
}

# B. Frequency × cross-program diversity (w_FD)
# w_FD = w_F * diversity_factor
# diversity_factor: kaç farklı ILSA programında ölçüldü / max_programs
# SES: PISA (1 prog) → div=0.25; PAR_EDU: TIMSS+TIMSS_G4+PIRLS (3) → div=0.75
# HOME_RES: PISA+PIRLS (2) → div=0.50; SCHOOL_BELONG: sadece PISA (1) → div=0.25
# HOME_LIT: sadece PIRLS (1) → div=0.25; ICT: PISA (1) → div=0.25
# Sonra 0-1 normalize
_diversity = {
    "SES_COMPOSITE":                  0.25,  # PISA only
    "PARENTAL_EDUCATION":             0.75,  # TIMSS, TIMSS_G4, PIRLS
    "HOME_RESOURCES":                 0.50,  # PISA, PIRLS
    "SCHOOL_BELONGING":               0.25,  # PISA only
    "HOME_LITERACY_ACTIVITIES":       0.25,  # PIRLS only
    "ICT_HOME_ACCESS":                0.25,  # PISA only
    "ICT_SCHOOL_ACCESS":              0.25,  # PISA only
    "TEACHER_SELF_EFFICACY_CLS_MGMT": 0.50,  # TALIS → PISA + TIMSS
    "TEACHER_SELF_EFFICACY_INSTRUCTION": 0.50,
    "TEACHER_SELF_EFFICACY_ENGAGEMENT": 0.50,
    "DISCIPLINARY_CLIMATE":           0.50,
    "PROF_COLLABORATION":             0.50,
    "JOB_SATISFACTION_ENVIRONMENT":   0.50,
    "JOB_SATISFACTION_PROFESSION":    0.50,
    "EFFECTIVE_PD":                   0.50,
}
_raw_fd = {k: W_FREQ.get(k, 0.25) * _diversity.get(k, 0.25) for k in W_FREQ}
_max_fd = max(_raw_fd.values())
W_FREQ_DIV = {k: round(v / _max_fd, 4) for k, v in _raw_fd.items()}

# C. Prior composite — Sirin 2005 + OECD framework (pre-corpus estimate)
W_COMPOSITE_PRIOR = {
    "SES_COMPOSITE":                  1.00,
    "PARENTAL_EDUCATION":             0.95,
    "HOME_RESOURCES":                 0.85,
    "SCHOOL_BELONGING":               0.80,
    "HOME_LITERACY_ACTIVITIES":       0.55,
    "ICT_HOME_ACCESS":                0.40,
    "ICT_SCHOOL_ACCESS":              0.40,
    "TEACHER_SELF_EFFICACY_CLS_MGMT": 0.70,
    "TEACHER_SELF_EFFICACY_INSTRUCTION": 0.70,
    "TEACHER_SELF_EFFICACY_ENGAGEMENT": 0.65,
    "DISCIPLINARY_CLIMATE":           0.75,
    "PROF_COLLABORATION":             0.55,
    "JOB_SATISFACTION_ENVIRONMENT":   0.50,
    "JOB_SATISFACTION_PROFESSION":    0.50,
    "EFFECTIVE_PD":                   0.60,
}

# D. W_j — corpus-based priority from build_literature_priority.py
# outputs/stage5/literature_priority.csv (FAZ 4, 132 articles, 2026-10-04)
W_CORPUS = {
    "SES_COMPOSITE":                   1.0000,
    "TEACHER_SELF_EFFICACY_CLS_MGMT":  0.8543,
    "HOME_RESOURCES":                  0.7175,
    "SCHOOL_BELONGING":                0.6185,
    "ICT_HOME_ACCESS":                 0.5370,
    "DISCIPLINARY_CLIMATE":            0.4910,
    "PARENTAL_EDUCATION":              0.4559,
    "TEACHER_SELF_EFFICACY_INSTRUCTION": 0.8543,
    "TEACHER_SELF_EFFICACY_ENGAGEMENT": 0.8543,
    "ICT_SCHOOL_ACCESS":               0.2818,
    "EFFECTIVE_PD":                    0.2547,
    "HOME_LITERACY_ACTIVITIES":        0.1029,
    "JOB_SATISFACTION_ENVIRONMENT":    0.0894,
    "JOB_SATISFACTION_PROFESSION":     0.0894,
    "PROF_COLLABORATION":              0.2547,
}

SPECS = {
    "w_F":    W_FREQ,
    "w_FD":   W_FREQ_DIV,
    "w_FDC":  W_COMPOSITE_PRIOR,
    "w_Wj":   W_CORPUS,       # corpus-based — primary spec
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

MIN_FEATURES_CI = 5  # Bootstrap CI bu eşiğin altında raporlanmaz


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


def build_Xy_fixed(df, fixed_cols):
    X_parts = [df[c].values.astype(float) if c in df.columns
               else np.full(len(df), np.nan) for c in fixed_cols]
    X = np.column_stack(X_parts) if X_parts else np.empty((len(df), 0))
    y = df["Y"].values.astype(float)
    row_valid = ~np.any(np.isnan(X), axis=1)
    return X[row_valid], y[row_valid]


def compute_shap_importance(v2, tier_map) -> dict[tuple, dict[str, float]]:
    """Returns {(prog, domain): {feature: mean_abs_shap}} — reuses SHAP from CSV if available."""
    shap_csv = OUT_DIR / "shap_fold_importance.csv"
    if shap_csv.exists():
        df = pd.read_csv(shap_csv)
        result = {}
        for (prog, dom), grp in df.groupby(["program", "domain"]):
            agg = grp.groupby("feature")["mean_abs_shap"].mean()
            result[(prog, dom)] = agg.to_dict()
        log.info("SHAP önem değerleri shap_fold_importance.csv'den yüklendi")
        return result

    # Fallback: yeniden hesapla (shap kütüphanesi gerekli)
    import shap as shap_lib
    result = {}
    for prog in ["PISA", "TIMSS", "TIMSS_G4", "PIRLS"]:
        for domain in DOMAINS[prog]:
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
            present_cycles = sorted(full["cycle"].dropna().unique())
            if len(present_cycles) < 3:
                continue
            intra_cols = [c for c in intra.columns if c not in ("country","cycle")]
            feat_cols  = ["lag_Y"] + intra_cols
            fold_shap: dict[str, list[float]] = {}
            for i, test_cycle in enumerate(present_cycles):
                if i < 2:
                    continue
                train_cycles = [c for c in present_cycles if c < test_cycle]
                train = full[full["cycle"].isin(train_cycles)].dropna(subset=["Y"])
                test  = full[full["cycle"] == test_cycle].dropna(subset=["Y"])
                used_cols, X_tr, y_tr = select_train_cols(train, feat_cols)
                if not used_cols or len(y_tr) < MIN_TRAIN_N:
                    continue
                X_te, y_te = build_Xy_fixed(test, used_cols)
                if len(y_te) < 3:
                    continue
                scaler = StandardScaler()
                X_tr_s = scaler.fit_transform(X_tr)
                X_te_s = scaler.transform(X_te)
                ridge = RidgeCV(alphas=RIDGE_ALPHAS, cv=min(len(y_tr), 5))
                ridge.fit(X_tr_s, y_tr)
                explainer = shap_lib.LinearExplainer(ridge, X_tr_s)
                shap_vals = explainer.shap_values(X_te_s)
                mean_abs  = np.abs(shap_vals).mean(axis=0)
                for j, col in enumerate(used_cols):
                    fold_shap.setdefault(col, []).append(float(mean_abs[j]))
            if fold_shap:
                result[(prog, domain)] = {f: float(np.mean(v)) for f, v in fold_shap.items()}
    return result


def compute_rho_for_spec(w_spec: dict, shap_importance: dict) -> list[dict]:
    records = []
    for (prog, domain), w_xai in shap_importance.items():
        # Sadece lag_Y hariç — ρ_lit
        feats = [f for f in w_xai if f != "lag_Y"]
        n = len(feats)
        xai_vec = np.array([w_xai[f] for f in feats])
        lit_vec = np.array([w_spec.get(f, 0.25) for f in feats])

        # n < 2: hesaplanamaz
        if n < 2:
            rho, pval = np.nan, np.nan
            interpretable = False
        elif n == 2:
            # Spearman n=2: trivially ±1 — not interpretable
            rho, pval = stats.spearmanr(lit_vec, xai_vec)
            interpretable = False
        else:
            rho, pval = stats.spearmanr(lit_vec, xai_vec)
            interpretable = (n >= 3)  # n=3 exploratory; n>=5 would be robust

        # lit_weight varyansı var mı?
        lit_var = float(np.var(lit_vec)) if n >= 2 else 0.0

        records.append({
            "program":        prog,
            "domain":         domain,
            "n_features_lit": n,
            "rho_lit":        round(float(rho), 4) if np.isfinite(rho) else np.nan,
            "p_value":        round(float(pval), 4) if np.isfinite(pval) else np.nan,
            "lit_weight_var": round(lit_var, 6),
            "trivial_n2":     (n == 2),
            "interpretable":  interpretable,
            "features":       "|".join(feats),
        })
    return records


def main():
    v2, tier_map = load_data()
    shap_importance = compute_shap_importance(v2, tier_map)

    all_records = []
    for spec_name, w_spec in SPECS.items():
        records = compute_rho_for_spec(w_spec, shap_importance)
        for r in records:
            r["spec"] = spec_name
        all_records.extend(records)

    df = pd.DataFrame(all_records)
    cols_order = ["program","domain","spec","n_features_lit","rho_lit","p_value",
                  "lit_weight_var","trivial_n2","interpretable","features"]
    df = df[cols_order]
    df.to_csv(OUT_DIR / "rho_align_sensitivity.csv", index=False)
    log.info("Kaydedildi: rho_align_sensitivity.csv (%d satır)", len(df))

    # ── Özet ─────────────────────────────────────────────────────────────────
    print("\n=== ρ_lit sensitivity — üç w_lit spesifikasyonu ===")
    print("(lag_Y hariç; trivial_n2=True → Spearman trivyal, yorumlanamaz)\n")

    pivot = df.pivot_table(
        index=["program","domain","n_features_lit","trivial_n2","interpretable"],
        columns="spec",
        values="rho_lit"
    ).reset_index()
    pivot.columns.name = None
    for col in ["w_F","w_FD","w_FDC"]:
        if col in pivot.columns:
            pivot[col] = pivot[col].round(4)

    # Interpretation flag
    pivot["stable_sign"] = pivot.apply(
        lambda r: _sign_stable(r.get("w_F"), r.get("w_FD"), r.get("w_FDC")), axis=1
    )
    print(pivot.to_string(index=False))

    print("\n=== Notlar ===")
    print("  trivial_n2=True  → n=2 feature; Spearman zorunlu olarak ±1; yorumlanamaz")
    print("  interpretable    → n>=3 (exploratory); n>=5 için robust CI raporlanabilir")
    print("  stable_sign      → rho_lit işareti üç spec'te tutarlı mı?")
    print()
    print("  PISA mathematics: rho_lit < 0 üç spec'te tutarlıysa → ranking discordance")
    print("  PISA mathematics: rho_lit işareti spec'e göre değişiyorsa → weight-dependent artefakt")


def _sign_stable(a, b, c):
    vals = [v for v in [a, b, c] if v is not None and np.isfinite(v)]
    if len(vals) < 2:
        return "insufficient"
    signs = set(np.sign(v) for v in vals)
    return "stable" if len(signs) == 1 else "unstable"


if __name__ == "__main__":
    main()
