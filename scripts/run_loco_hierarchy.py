#!/usr/bin/env python3
"""
LOCO Forecasting Hiyerarşisi — M2 / M0A / M0B / M1A / M1B

Forecasting unit: u = (country, program, cycle, domain)
Expanding-window LOCO: her cycle test fold'da, önceki tüm cycle'lar eğitim kümesi.

Model hiyerarşisi:
  M2   — Persistence (naïve baseline: Y_hat = Y_{t-1})
  M0A  — Achievement history only (Ridge, lag_Y feature)
  M0B  — Achievement + intra-program ILSA microdata predictors
  M1A  — M0B + literature-informed feature selection/weighting
  M1B  — M1A + cross-program contextual features (TALIS vb.)

Transformation kuralları (crosswalk'tan):
  "negate"                  → value * -1
  "drop_near_zero_variance" → drop (ülkeler arası varyans ~0)
  "none" / "rename_only"    → olduğu gibi

Her fold'da:
  1. Feature matrix X_train, X_test: training data içinde StandardScaler
  2. Ridge lambda: inner LOO-CV (training data üzerinde)
  3. Test RMSE, MAE, R², Spearman hesapla

Çıktılar:
  outputs/stage5/loco_hierarchy_results.csv  — fold × model metrikleri
  outputs/stage5/loco_hierarchy_summary.csv  — model başına özet

NOTLAR:
  - T3DISC (TALIS 2018): drop_near_zero_variance → otomatik düşürülür
  - TCDISCS (TALIS 2013): negate uygulanır → modele pozitif yönlü girer
  - M1A literatür ağırlıkları: tier A = 1.0, tier B = 0.5 (feature standardize edilmiş)
  - BSDGEDUP/ASDHEDUP/ASDHELA: negate sonrası yüksek = daha iyi eğitim/okuma
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

# ── Parametreler ──────────────────────────────────────────────────────────────
RIDGE_ALPHAS  = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
MIN_TRAIN_N   = 10   # eğitim fold minimum ülke sayısı

# M1A literatür ağırlıkları: tier → katsayı (standart Feature'lara uygulanır)
TIER_WEIGHTS = {"A": 1.0, "B": 0.5, "C": 0.25}

# ── Yardımcı fonksiyonlar ─────────────────────────────────────────────────────
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    n = len(y_true)
    if n < 3:
        return {"rmse": np.nan, "mae": np.nan, "r2": np.nan, "spearman": np.nan, "n": n}
    residuals = y_true - y_pred
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_true - y_true.mean())**2)
    r2 = 1 - ss_res/ss_tot if ss_tot > 0 else np.nan
    rho, _ = stats.spearmanr(y_true, y_pred)
    return {
        "rmse": float(np.sqrt(np.mean(residuals**2))),
        "mae":  float(np.mean(np.abs(residuals))),
        "r2":   float(r2),
        "spearman": float(rho),
        "n":    n,
    }


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """enriched_panel_v2 ve canonical_crosswalk yükle, transformation uygula."""
    v2    = pd.read_csv(OUT_DIR / "enriched_panel_v2.csv")
    xwalk = pd.read_csv(OUT_DIR / "canonical_crosswalk.csv")

    # Drop near-zero variance constructs
    drop_constructs = set(
        xwalk.loc[xwalk["transformation"] == "drop_near_zero_variance", "canonical_construct"]
    )
    log.info("Drop (near-zero variance): %s", drop_constructs)
    v2 = v2[~v2["canonical_construct"].isin(drop_constructs)].copy()

    # Negate işlemi: direction_original="-" → value * -1
    negate_keys = set(
        zip(xwalk.loc[xwalk["transformation"] == "negate", "program"],
            xwalk.loc[xwalk["transformation"] == "negate", "canonical_construct"])
    )
    # canonical_construct bazında negate (program'dan bağımsız; aynı canonical = aynı dönüşüm)
    negate_constructs = set(
        xwalk.loc[xwalk["transformation"] == "negate", "canonical_construct"]
    )
    mask_negate = v2["canonical_construct"].isin(negate_constructs) & (v2["forecast_role"] != "outcome")
    v2.loc[mask_negate, "value"] = v2.loc[mask_negate, "value"] * -1
    log.info("Negate uygulandı: %s (%d satır)", sorted(negate_constructs), mask_negate.sum())

    return v2, xwalk


def pivot_features(v2: pd.DataFrame, xwalk: pd.DataFrame,
                   target_programs: list[str],
                   feature_sources: list[str],
                   include_cross_program: bool = False) -> pd.DataFrame:
    """
    Her (country, program, cycle, domain) için geniş feature matrisi oluştur.
    feature_sources: ["intra_program_lag"] veya + ["talis_cross_program"] vb.
    """
    # Outcome
    outcomes = v2[v2["forecast_role"] == "outcome"][
        ["target_country","target_program","target_cycle","canonical_construct","value"]
    ].rename(columns={"target_country":"country","target_program":"program",
                      "target_cycle":"cycle","canonical_construct":"construct"})
    outcomes = outcomes[outcomes["program"].isin(target_programs)]

    # Target domain için pivot
    outcome_pivot = outcomes.pivot_table(
        index=["country","program","cycle"],
        columns="construct",
        values="value",
        aggfunc="mean"
    ).reset_index()

    # Features
    feat_mask = (
        v2["feature_source"].isin(feature_sources) &
        v2["target_program"].isin(target_programs)
    )
    feats = v2[feat_mask][
        ["target_country","target_program","target_cycle","canonical_construct","value"]
    ].rename(columns={"target_country":"country","target_program":"program",
                      "target_cycle":"cycle","canonical_construct":"construct"})

    feat_pivot = feats.pivot_table(
        index=["country","program","cycle"],
        columns="construct",
        values="value",
        aggfunc="mean"
    ).reset_index()

    merged = outcome_pivot.merge(feat_pivot, on=["country","program","cycle"], how="left",
                                  suffixes=("_out","_feat"))
    return merged


def get_outcome_col(program: str, domain: str) -> str:
    """Construct adını döndür (outcome)."""
    MAP = {
        ("PISA",     "mathematics"): "PISA_MATHEMATICS",
        ("PISA",     "reading"):     "PISA_READING",
        ("PISA",     "science"):     "PISA_SCIENCE",
        ("TIMSS",    "mathematics"): "TIMSS_MATHEMATICS",
        ("TIMSS",    "science"):     "TIMSS_SCIENCE",
        ("TIMSS_G4", "mathematics"): "TIMSS_G4_MATHEMATICS",
        ("TIMSS_G4", "science"):     "TIMSS_G4_SCIENCE",
        ("PIRLS",    "reading"):     "PIRLS_READING",
    }
    return MAP.get((program, domain), f"{program}_{domain}".upper())


# ── LOCO expanding window ─────────────────────────────────────────────────────
def run_loco(v2: pd.DataFrame, xwalk: pd.DataFrame,
             target_programs: list[str] = None) -> pd.DataFrame:
    if target_programs is None:
        target_programs = ["PISA", "TIMSS", "TIMSS_G4", "PIRLS"]

    # Tier lookup — lag_Y crosswalk'ta yok; literature unanimously supports prior achievement → A
    tier_map = dict(zip(xwalk["canonical_construct"], xwalk["tier"]))
    tier_map["lag_Y"] = "A"

    # Her program × domain için cycle sırası
    CYCLE_ORDER = {
        "PISA":     [2003, 2006, 2009, 2012, 2015, 2018, 2022, 2025],
        "TIMSS":    [1995, 1999, 2003, 2007, 2011, 2015, 2019, 2023],
        "TIMSS_G4": [1995, 2003, 2007, 2011, 2015, 2019, 2023],
        "PIRLS":    [2001, 2006, 2011, 2016, 2021],
    }
    DOMAINS = {
        "PISA":     ["mathematics","reading","science"],
        "TIMSS":    ["mathematics","science"],
        "TIMSS_G4": ["mathematics","science"],
        "PIRLS":    ["reading"],
    }

    all_records = []

    for prog in target_programs:
        cycles = CYCLE_ORDER.get(prog, [])
        domains = DOMAINS.get(prog, [])
        if len(cycles) < 3:
            log.info("Skip %s: çok az cycle (%d)", prog, len(cycles))
            continue

        for domain in domains:
            outcome_col = get_outcome_col(prog, domain)
            log.info("=== %s / %s (outcome: %s) ===", prog, domain, outcome_col)

            # Outcome verisini çek (intra-program lag feature olmadan da outcome var)
            outcomes = v2[
                (v2["forecast_role"] == "outcome") &
                (v2["target_program"] == prog) &
                (v2["canonical_construct"] == outcome_col)
            ][["target_country","target_cycle","value"]].rename(
                columns={"target_country":"country","target_cycle":"cycle","value":"Y"}
            )
            if outcomes.empty:
                log.info("  Outcome bulunamadı: %s", outcome_col)
                continue

            # Intra-program lag features
            intra = v2[
                (v2["feature_source"] == "intra_program_lag") &
                (v2["target_program"] == prog)
            ][["target_country","target_cycle","canonical_construct","value"]].rename(
                columns={"target_country":"country","target_cycle":"cycle",
                         "canonical_construct":"construct"}
            ).pivot_table(index=["country","cycle"], columns="construct",
                          values="value", aggfunc="mean").reset_index()

            # TALIS cross-program features
            talis_feats = v2[
                (v2["feature_source"] == "talis_cross_program") &
                (v2["target_program"] == prog)
            ][["target_country","target_cycle","canonical_construct","value"]].rename(
                columns={"target_country":"country","target_cycle":"cycle",
                         "canonical_construct":"construct"}
            ).pivot_table(index=["country","cycle"], columns="construct",
                          values="value", aggfunc="mean").reset_index()

            # Ana tablo: outcomes × intra × talis
            base = outcomes.merge(intra,       on=["country","cycle"], how="left")
            full = base.merge(talis_feats, on=["country","cycle"], how="left",
                              suffixes=("","_talis"))

            # Lag Y (persistence için ve M0A için)
            full = full.sort_values(["country","cycle"]).copy()
            full["lag_Y"] = full.groupby("country")["Y"].shift(1)

            # Cycle sırası içindeki her possible test fold (expanding window)
            present_cycles = sorted(full["cycle"].dropna().unique())
            if len(present_cycles) < 3:
                continue

            for i, test_cycle in enumerate(present_cycles):
                if i < 2:  # minimum 2 eğitim cycle'ı
                    continue
                train_cycles = [c for c in present_cycles if c < test_cycle]

                train = full[full["cycle"].isin(train_cycles)].dropna(subset=["Y"])
                test  = full[full["cycle"] == test_cycle].dropna(subset=["Y"])

                if len(train) < MIN_TRAIN_N or len(test) < 3:
                    continue

                y_train = train["Y"].values
                y_test  = test["Y"].values

                # ── M2: Persistence ──────────────────────────────────────────
                lag_test = test["lag_Y"].values
                valid_m2 = ~np.isnan(lag_test)
                if valid_m2.sum() >= 3:
                    m = compute_metrics(y_test[valid_m2], lag_test[valid_m2])
                    all_records.append({
                        "program": prog, "domain": domain, "test_cycle": test_cycle,
                        "model": "M2_Persistence", **m})

                # ── Feature set tanımları ─────────────────────────────────────
                def _select_train_cols(df_sub: pd.DataFrame, feat_cols: list[str]):
                    """Training fold için: var olan ve all-NaN olmayan sütunları seç."""
                    avail = [c for c in feat_cols if c in df_sub.columns]
                    if not avail:
                        return [], None, None
                    X = df_sub[avail].values.astype(float)
                    y = df_sub["Y"].values.astype(float)
                    col_valid = ~np.all(np.isnan(X), axis=0)
                    X = X[:, col_valid]
                    avail = [a for a, v in zip(avail, col_valid) if v]
                    row_valid = ~np.any(np.isnan(X), axis=1)
                    return avail, X[row_valid], y[row_valid]

                def _build_Xy_fixed(df_sub: pd.DataFrame, fixed_cols: list[str],
                                     col_means: dict | None = None):
                    """
                    Test fold için: fixed_cols sütunlarını kullan.
                    col_means verilmişse eksik değerleri imputate et (M1B cross-prog için).
                    Aksi halde NaN satırları at.
                    """
                    X_parts = []
                    for c in fixed_cols:
                        if c in df_sub.columns:
                            col_vals = df_sub[c].values.astype(float)
                        else:
                            col_vals = np.full(len(df_sub), np.nan)
                        if col_means is not None:
                            nan_mask = np.isnan(col_vals)
                            if nan_mask.any() and c in col_means:
                                col_vals = col_vals.copy()
                                col_vals[nan_mask] = col_means[c]
                        X_parts.append(col_vals)
                    X = np.column_stack(X_parts) if X_parts else np.empty((len(df_sub), 0))
                    y = df_sub["Y"].values.astype(float)
                    row_valid = ~np.any(np.isnan(X), axis=1)
                    return X[row_valid], y[row_valid]

                intra_cols = [c for c in intra.columns if c not in ("country","cycle")]
                talis_cols = [c for c in talis_feats.columns if c not in ("country","cycle")]

                feat_m0a = ["lag_Y"]
                feat_m0b = ["lag_Y"] + intra_cols
                feat_m1a = ["lag_Y"] + intra_cols   # tier A ağırlıkları ile
                feat_m1b = ["lag_Y"] + intra_cols + talis_cols

                for model_name, feat_cols, use_tier_weights in [
                    ("M0A_achievement", feat_m0a, False),
                    ("M0B_microdata",   feat_m0b, False),
                    ("M1A_literature",  feat_m1a, True),
                    ("M1B_integrated",  feat_m1b, True),
                ]:
                    used_cols, X_tr, y_tr = _select_train_cols(
                        train.dropna(subset=["Y"]), feat_cols)
                    if not used_cols or len(y_tr) < MIN_TRAIN_N:
                        continue

                    X_te, y_te = _build_Xy_fixed(test.dropna(subset=["Y"]), used_cols)
                    if len(y_te) < 3:
                        continue

                    # Scaling (training data içinde)
                    scaler = StandardScaler()
                    # NaN-safe: sütun std=0 olanları koru
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        X_tr_s = scaler.fit_transform(X_tr)
                        X_te_s = scaler.transform(X_te)

                    # Tier ağırlıkları (M1A/M1B için)
                    if use_tier_weights:
                        fw = np.array([
                            TIER_WEIGHTS.get(tier_map.get(c, "C"), 0.25)
                            for c in used_cols
                        ])
                        X_tr_s = X_tr_s * fw
                        X_te_s = X_te_s * fw

                    # Ridge CV
                    ridge = RidgeCV(alphas=RIDGE_ALPHAS, cv=min(len(y_tr), 5))
                    ridge.fit(X_tr_s, y_tr)
                    y_pred = ridge.predict(X_te_s)

                    m = compute_metrics(y_te, y_pred)
                    all_records.append({
                        "program": prog, "domain": domain, "test_cycle": test_cycle,
                        "model": model_name, "alpha": float(ridge.alpha_),
                        "n_features": len(used_cols),
                        **m,
                    })

    return pd.DataFrame(all_records)


def main():
    v2, xwalk = load_data()
    log.info("Panel v2: %d satır yüklendi", len(v2))

    results = run_loco(v2, xwalk)

    if results.empty:
        log.error("Hiç sonuç üretilemedi.")
        return

    # Kaydet
    results_csv = OUT_DIR / "loco_hierarchy_results.csv"
    results.to_csv(results_csv, index=False)
    log.info("Fold sonuçları kaydedildi: %s", results_csv)

    # Özet: model başına ortalama metrikler
    summary = (
        results.groupby(["program","domain","model"])
        [["rmse","mae","r2","spearman","n"]]
        .agg({"rmse":"mean","mae":"mean","r2":"mean","spearman":"mean","n":"sum"})
        .round(4)
        .reset_index()
    )
    summary_csv = OUT_DIR / "loco_hierarchy_summary.csv"
    summary.to_csv(summary_csv, index=False)

    print("\n=== LOCO Hiyerarşi Özeti (ortalama RMSE) ===")
    rmse_table = summary.pivot_table(
        index=["program","domain"], columns="model", values="rmse"
    )
    print(rmse_table.to_string())

    print("\n=== Delta RMSE: M0B → M1A → M1B ===")
    for (prog, dom), grp in summary.groupby(["program","domain"]):
        row = {r["model"]: r["rmse"] for _, r in grp.iterrows()}
        m0b = row.get("M0B_microdata", np.nan)
        m1a = row.get("M1A_literature", np.nan)
        m1b = row.get("M1B_integrated", np.nan)
        print(f"  {prog}/{dom}: M0B={m0b:.2f}  M1A={m1a:.2f} (Δ={m0b-m1a:+.2f})  "
              f"M1B={m1b:.2f} (Δ={m1a-m1b:+.2f})")

    print(f"\nToplam fold sayısı: {len(results)}")
    print("Kayıtlı:", results_csv.name, "&", summary_csv.name)


if __name__ == "__main__":
    main()
