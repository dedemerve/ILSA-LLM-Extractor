#!/usr/bin/env python3
"""
PIAAC LOCO Forecasting (M0 / M1 / M2 / M3)

PIAAC yalnızca 2 döngüye sahiptir: Cycle 1 = 2012, Cycle 2 = 2022.
Standart LOCO min_prior=2 ile 0 fold üretir.  Bu script min_prior=1
kullanarak 1 test foldu etkinleştirir: 2012 → tahmin 2022.

Fold tasarımı (ICILS ile aynı çapraz-kesit mantığı):
  - Eğitim: 2012 döngüsünde gözlemlenen tüm ülkeler (n≈25)
  - Test   : 2022 döngüsünde gözlemlenen tüm ülkeler (n≈21)
  - Kesişim zorunluluğu yok — katılım listeleri döngüler arasında değişir

Alan kapsamı: literacy, numeracy  (gender ve parental_education predictor)

Sınırlama: RF/XGBoost bu scriptte yer almaz (eğitim için yetersiz örnek;
tabular_benchmark zaten bu modelleri PIAAC için dışarıda bırakmış).

Ön koşul: fix_piaac_cycle_year.py çalıştırılmış olmalı
  (unified_panel.csv'de PIAAC cycle 2017 → 2022 düzeltmesi)

Çıktı: outputs/stage5/piaac_loco_results.csv
       outputs/stage5/piaac_loco_predictions.csv
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

# Outcome domains to evaluate (predictor/covariate domains excluded)
PIAAC_OUTCOME_DOMAINS = ["literacy", "numeracy"]

# PIAAC covariate → canonical construct (for lit weight lookup)
_PIAAC_TO_CANONICAL: dict[str, str] = {
    "PIAAC_parental_education": "PARENTAL_EDUCATION",
    "PIAAC_gender":             "GENDER_EQUITY",       # no KB entry → w=1.0
    # numeracy/literacy as cross-predictors — not in KB → w=1.0
}


def piaac_feat_weight(f: str, weights: dict) -> float:
    canonical = _PIAAC_TO_CANONICAL.get(f, f)
    return weights.get(canonical, 1.0)


def _build_piaac_fold(
    panel: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    train_cycle: int,
    test_cycle: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Cross-sectional fold builder for PIAAC.

    Like ICILS: no country intersection constraint.
    Training uses all countries available at train_cycle;
    test uses all countries at test_cycle.
    """
    tr = panel[panel["cycle"] == train_cycle].set_index("country_iso3")
    te = panel[panel["cycle"] == test_cycle].set_index("country_iso3")

    tr_valid = tr[tr[target_col].notna()]
    te_valid = te[te[target_col].notna()]

    if len(tr_valid) < 2 or len(te_valid) < 2:
        return None, None, None, None, []

    X_tr = tr_valid[feature_cols].values.astype(float)
    y_tr = tr_valid[target_col].values.astype(float)
    X_te = te_valid[feature_cols].values.astype(float)
    y_te = te_valid[target_col].values.astype(float)
    countries = te_valid.index.tolist()
    return X_tr, y_tr, X_te, y_te, countries


def run_piaac_loco() -> pd.DataFrame:
    unified = pd.read_csv(STAGE5_DIR / "unified_panel.csv", dtype={"country_iso3": str})
    piaac = unified[unified["program"] == "PIAAC"].copy()
    piaac["cycle"] = piaac["cycle"].astype(int)

    cycles = sorted(piaac["cycle"].unique())
    if 2017 in cycles:
        raise RuntimeError(
            "unified_panel.csv hâlâ PIAAC cycle 2017 içeriyor. "
            "Önce scripts/fix_piaac_cycle_year.py çalıştırın."
        )

    log.info("PIAAC döngüleri: %s", cycles)
    if len(cycles) < 2:
        raise RuntimeError("PIAAC için en az 2 döngü gerekiyor.")

    wdf = pd.read_csv(STAGE5_DIR / "predictor_weights_v2.csv")
    weights: dict[str, float] = dict(zip(wdf["predictor_canonical"], wdf["w_norm"]))

    results_rows: list[dict] = []
    pred_rows:    list[dict] = []

    for domain in PIAAC_OUTCOME_DOMAINS:
        score_col = f"PIAAC_{domain}"
        panel_d = piaac[piaac["domain"] == domain].copy() if "domain" in piaac.columns else piaac.copy()

        # PIAAC domain filtreleme: her satır bir domain etiketi taşıyor
        if "domain" in piaac.columns:
            panel_d = piaac[piaac["domain"] == domain].copy()
        else:
            # domain sütunu yoksa score sütununa göre filtrele
            panel_d = piaac.copy()

        if score_col not in panel_d.columns:
            log.warning("Sütun bulunamadı: %s — atlanıyor", score_col)
            continue

        # PIAAC covariates: parental_education and gender (measured in same wave).
        # lag_* columns are PISA-specific and NaN for PIAAC rows.
        # Cross-sectional design: train on 2012 covariate→outcome, apply to 2022.
        all_feature_cols = [
            c for c in panel_d.columns
            if c.startswith("PIAAC_") and c != score_col
        ]
        # Hedef sütunu feature listesinden çıkar
        all_feature_cols = [c for c in all_feature_cols if c != score_col]

        log.info("PIAAC/%s — features: %s", domain, all_feature_cols)

        # PIAAC sadece 2 döngü: tek fold
        test_cycle  = cycles[-1]   # 2022
        train_cycle = cycles[-2]   # 2012

        X_tr_full, y_tr, X_te_full, y_te, countries = _build_piaac_fold(
            panel_d, all_feature_cols, score_col, train_cycle, test_cycle
        )

        if X_tr_full is None or len(y_tr) < 2:
            log.warning("PIAAC/%s: yetersiz veri — atlanıyor", domain)
            continue

        # NaN sütunlarını düşür
        col_any_filled = np.array([
            np.any(~np.isnan(X_tr_full[:, i]))
            for i in range(X_tr_full.shape[1])
        ])
        fold_feat = [f for f, ok in zip(all_feature_cols, col_any_filled) if ok]
        keep_idx  = [i for i, ok in enumerate(col_any_filled) if ok]

        if not fold_feat:
            log.warning("PIAAC/%s test=%d: geçerli feature yok — atlanıyor", domain, test_cycle)
            # Persistence ve AR1 yine de üretebiliriz (feature gerektirmez)

        # Persistence (M2) ve AR1 (M3) — her zaman
        train_vals_by_country: dict[str, list[float]] = {}
        for _, row in panel_d[panel_d["cycle"] == train_cycle].iterrows():
            c = row["country_iso3"]
            v = row[score_col]
            if pd.notna(v):
                train_vals_by_country[c] = train_vals_by_country.get(c, []) + [float(v)]

        test_records = []
        for c in countries:
            te_row = panel_d[(panel_d["country_iso3"] == c) & (panel_d["cycle"] == test_cycle)]
            if te_row.empty or pd.isna(te_row.iloc[0][score_col]):
                continue
            y_true = float(te_row.iloc[0][score_col])
            hist = train_vals_by_country.get(c, [])
            test_records.append({"country": c, "y_true": y_true, "hist": hist})

        # M2: Persistence — fold-level aggregate
        y_true_arr  = np.array([r["y_true"] for r in test_records])
        last_scores = np.array([r["hist"][-1] if r["hist"] else np.nan for r in test_records])
        valid_m2 = ~np.isnan(last_scores)
        if valid_m2.sum() >= 2:
            err_m2 = last_scores[valid_m2] - y_true_arr[valid_m2]
            results_rows.append({
                "program": "PIAAC", "domain": domain, "test_cycle": test_cycle,
                "n_train": len(y_tr), "n_test": int(valid_m2.sum()),
                "model": "M2",
                "MAE":  round(float(np.abs(err_m2).mean()), 4),
                "RMSE": round(float(np.sqrt(np.mean(err_m2**2))), 4),
                "n_features": 0,
            })

        if not fold_feat:
            log.warning("PIAAC/%s: feature yok, yalnızca M2 üretildi", domain)
            continue

        X_tr = X_tr_full[:, keep_idx]
        X_te = X_te_full[:, keep_idx]
        lit_w = np.array([piaac_feat_weight(f, weights) for f in fold_feat])

        log.info("PIAAC/%s: train=%d, test=%d, features=%d",
                 domain, len(y_tr), len(y_te), len(fold_feat))
        log.info("  feature_weights: %s",
                 {f: round(piaac_feat_weight(f, weights), 3) for f in fold_feat})

        # M0 ve M1
        y_m0, *_ = loco.fit_ridge(X_tr, y_tr, X_te, lit_weights=None)
        y_m1, *_ = loco.fit_ridge(X_tr, y_tr, X_te, lit_weights=lit_w)

        for model_name, y_pred in [("M0", y_m0), ("M1", y_m1)]:
            valid = ~np.isnan(y_pred)
            if valid.sum() < 2:
                continue
            yt, yp = y_te[valid], y_pred[valid]
            results_rows.append({
                "program": "PIAAC", "domain": domain, "test_cycle": test_cycle,
                "n_train": len(y_tr), "n_test": int(valid.sum()),
                "model": model_name,
                "MAE":  round(float(np.mean(np.abs(yp - yt))), 4),
                "RMSE": round(float(np.sqrt(np.mean((yp - yt)**2))), 4),
                "n_features": len(fold_feat),
            })
        for j, cnt in enumerate(countries):
            pred_rows.append({
                "program": "PIAAC", "domain": domain, "test_cycle": test_cycle,
                "country_iso3": cnt,
                "y_true": round(float(y_te[j]), 4),
                "y_M0": round(float(y_m0[j]) if not np.isnan(y_m0[j]) else float("nan"), 4),
                "y_M1": round(float(y_m1[j]) if not np.isnan(y_m1[j]) else float("nan"), 4),
            })

    res_df  = pd.DataFrame(results_rows)
    pred_df = pd.DataFrame(pred_rows)

    res_path  = STAGE5_DIR / "piaac_loco_results.csv"
    pred_path = STAGE5_DIR / "piaac_loco_predictions.csv"
    res_df.to_csv(res_path,  index=False)
    pred_df.to_csv(pred_path, index=False)
    log.info("Kaydedildi: %s  (%d satır)", res_path.name,  len(res_df))
    log.info("Kaydedildi: %s  (%d satır)", pred_path.name, len(pred_df))

    if not res_df.empty:
        print("\n=== PIAAC LOCO Results ===")
        print(res_df[["domain","model","MAE","RMSE","n_train","n_test"]].to_string(index=False))

    return res_df


if __name__ == "__main__":
    run_piaac_loco()
