#!/usr/bin/env python3
"""
Stage 4 — Modül 3: LOCO Forecasting (M0 / M1 / M2)

Modeller:
  M0  — Ridge (veri güdüimlü): lag özellikleri, literatür ağırlığı yok
  M1  — Ridge (literatür bilgili): özellikler sqrt(w_j) ile ölçeklenir
  M2  — Persistence: son gözlenen cycle ortalamasını tahmin olarak kullan

Temporal doğrulama:
  LOCO (Leave-One-Cycle-Out): her cycle c için, c'den önceki tüm
  cycle'lar train seti; c test seti. En az 1 train cycle gerekir.

Özellik seti:
  Her (program, domain) için, test cycle'ından önceki en son
  gözlenen ülke ortalaması (lag-1 özelliği).
  Çapraz-program özellikleri: aynı domain'deki diğer programların
  lag-1 ortalamaları (program × domain çiftleri).

Literatür ağırlığı eşlemesi (M1):
  Math_Achievement    → program=PISA/TIMSS, domain=mathematics
  Reading_Achievement → program=PIRLS/PISA, domain=reading
  Science_Achievement → program=PISA/TIMSS, domain=science

Çıktılar:
  outputs/stage4/loco_results.csv      — fold × model metrikleri
  outputs/stage4/loco_predictions.csv  — her tahmin satırı
  outputs/stage4/shap_values.csv       — M1 SHAP global önem (program × domain × özellik)
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

try:
    import shap as _shap
    _SHAP_AVAILABLE = True
except ImportError:
    _SHAP_AVAILABLE = False

PROJECT_ROOT  = Path(__file__).resolve().parents[1]
STAGE4_DIR    = PROJECT_ROOT / "outputs" / "stage4"
ESTIMATES_CSV = STAGE4_DIR / "country_estimates.csv"
WEIGHTS_CSV   = STAGE4_DIR / "predictor_weights.csv"
OUT_RESULTS   = STAGE4_DIR / "loco_results.csv"
OUT_PREDS     = STAGE4_DIR / "loco_predictions.csv"
OUT_SHAP      = STAGE4_DIR / "shap_values.csv"

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# Ridge alpha grid (cross-validated)
ALPHA_GRID = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

# Program–domain → knowledge_synthesis canonical variable eşlemesi
_DOMAIN_TO_VAR: dict[tuple[str, str], str] = {
    ("PISA",  "mathematics"):  "Math_Achievement",
    ("PISA",  "reading"):      "Reading_Achievement",
    ("PISA",  "science"):      "Science_Achievement",
    ("TIMSS", "mathematics"):  "Math_Achievement",
    ("TIMSS", "science"):      "Science_Achievement",
    ("PIRLS", "reading"):      "Reading_Achievement",
}

# ---------------------------------------------------------------------------
# Yardımcı fonksiyonlar
# ---------------------------------------------------------------------------

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def spearman_r(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 3:
        return float("nan")
    rho, _ = stats.spearmanr(y_true, y_pred)
    return float(rho)


def diebold_mariano(e1: np.ndarray, e2: np.ndarray) -> tuple[float, float]:
    """Harvey-Leybourne-Newbold düzeltmeli Diebold-Mariano testi.

    H0: M1 ve M2 eşit tahmin gücu.  e1=M1 hataları, e2=M2 hataları.
    Döndürür: (DM istatistiği, p değeri)
    """
    d = e1 ** 2 - e2 ** 2
    n = len(d)
    if n < 3:
        return float("nan"), float("nan")
    d_bar = np.mean(d)
    # HAC varyans (lag-1 Newey-West)
    gamma0 = np.var(d, ddof=1)
    gamma1 = np.cov(d[:-1], d[1:])[0, 1] if n > 1 else 0.0
    v = (gamma0 + 2 * gamma1) / n
    if v <= 0:
        return float("nan"), float("nan")
    dm_stat = d_bar / np.sqrt(v)
    p_val   = 2 * (1 - stats.t.cdf(abs(dm_stat), df=n - 1))
    return float(dm_stat), float(p_val)


# ---------------------------------------------------------------------------
# Veri hazırlama
# ---------------------------------------------------------------------------

def load_data() -> tuple[pd.DataFrame, dict[str, float]]:
    if not ESTIMATES_CSV.exists():
        raise FileNotFoundError(
            f"country_estimates.csv bulunamadı: {ESTIMATES_CSV}\n"
            "Önce build_country_estimates.py çalıştırın."
        )
    est = pd.read_csv(ESTIMATES_CSV)
    est["country_iso3"] = est["country_iso3"].astype(str).str.strip()
    est["cycle"]        = est["cycle"].astype(int)
    est["program"]      = est["program"].str.upper()
    est["domain"]       = est["domain"].str.lower()

    wdf = pd.read_csv(WEIGHTS_CSV)
    weights: dict[str, float] = dict(zip(wdf["variable"], wdf["w_norm"]))

    return est, weights


def make_feature_key(program: str, domain: str) -> str:
    return f"{program}_{domain}"


def build_panel(est: pd.DataFrame) -> pd.DataFrame:
    """Uzun formatı geniş (pivot) panele dönüştür.

    Her satır bir (country_iso3, cycle) çiftini temsil eder.
    Sütunlar: program_domain ortalamaları.
    """
    est["feat_key"] = est.apply(
        lambda r: make_feature_key(r["program"], r["domain"]), axis=1
    )
    pivot = est.pivot_table(
        index=["country_iso3", "cycle"],
        columns="feat_key",
        values="mean",
        aggfunc="first",
    ).reset_index()
    pivot.columns.name = None
    return pivot


# ---------------------------------------------------------------------------
# Özellik matrisini LOCO fold için hazırla
# ---------------------------------------------------------------------------

def build_xy(
    panel: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    train_cycles: list[int],
    test_cycle: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.Index]:
    """Train ve test matrislerini oluşturur.

    Her satır bir ülkeyi temsil eder.
    X özelliği: train_cycles'ın sonuncusundaki değer (lag-1).
    y hedef: test_cycle'ındaki değer.
    """
    lag_cycle = max(train_cycles)

    lag_df    = panel[panel["cycle"] == lag_cycle].set_index("country_iso3")
    target_df = panel[panel["cycle"] == test_cycle].set_index("country_iso3")

    common = lag_df.index.intersection(target_df.index)
    if len(common) == 0:
        return None, None, None, None, None

    X_lag    = lag_df.loc[common, feature_cols].values.astype(float)
    y_target = target_df.loc[common, target_col].values.astype(float)

    # Train: her train cycle için (lag = önceki cycle)
    X_train_list, y_train_list = [], []
    sorted_train = sorted(train_cycles)
    for i, tc in enumerate(sorted_train):
        if i == 0:
            continue  # ilk cycle için lag yok
        prev_cycle = sorted_train[i - 1]
        prev_df = panel[panel["cycle"] == prev_cycle].set_index("country_iso3")
        curr_df = panel[panel["cycle"] == tc].set_index("country_iso3")
        common_tr = prev_df.index.intersection(curr_df.index)
        if len(common_tr) == 0:
            continue
        X_tr = prev_df.loc[common_tr, feature_cols].values.astype(float)
        y_tr = curr_df.loc[common_tr, target_col].values.astype(float)
        mask = ~np.isnan(y_tr)
        if mask.sum() == 0:
            continue
        X_train_list.append(X_tr[mask])
        y_train_list.append(y_tr[mask])

    if not X_train_list:
        return None, None, None, None, None

    X_train = np.vstack(X_train_list)
    y_train = np.concatenate(y_train_list)

    mask_test = ~np.isnan(y_target)
    X_test  = X_lag[mask_test]
    y_test  = y_target[mask_test]
    countries_test = common[mask_test]

    if len(X_train) == 0 or len(X_test) == 0:
        return None, None, None, None, None

    return X_train, y_train, X_test, y_test, countries_test


# ---------------------------------------------------------------------------
# Model eğitim / tahmin
# ---------------------------------------------------------------------------

def fit_ridge(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    lit_weights: np.ndarray | None = None,
) -> tuple[np.ndarray, RidgeCV, StandardScaler, np.ndarray, np.ndarray]:
    """Ridge (CV alpha) ile eğit; tahmin ve eğitim artifaktlarını döndür.

    lit_weights: M1 için her özelliğe uygulanacak sqrt(w_j) ölçekleme vektörü.
    None ise M0 (ölçekleme yok).

    Döndürür:
        y_pred       — test tahminleri
        model        — eğitilmiş RidgeCV
        scaler       — fit edilmiş StandardScaler
        scale        — uygulanan sqrt(w_j) vektörü (M0 için np.ones)
        col_means    — NaN imputation için train sütun ortalamaları
    """
    scaler = StandardScaler()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        col_means = np.nanmean(X_train, axis=0)
    col_means = np.where(np.isnan(col_means), 0.0, col_means)
    X_train = np.where(np.isnan(X_train), col_means, X_train)
    X_test  = np.where(np.isnan(X_test),  col_means, X_test)

    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    if lit_weights is not None:
        scale = np.sqrt(np.clip(lit_weights, 1e-6, None))
    else:
        scale = np.ones(X_tr.shape[1])

    X_tr = X_tr * scale
    X_te = X_te * scale

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = RidgeCV(alphas=ALPHA_GRID, cv=min(5, len(y_train)))
        model.fit(X_tr, y_train)

    return model.predict(X_te), model, scaler, scale, col_means


def compute_shap_values(
    model: RidgeCV,
    scaler: StandardScaler,
    scale: np.ndarray,
    col_means: np.ndarray,
    X_test_raw: np.ndarray,
) -> np.ndarray:
    """M1 Ridge modeli için SHAP değerlerini hesapla (LinearExplainer).

    Dönüş: shape (n_test, n_features) — her gözlem için özellik bazlı SHAP
    """
    if not _SHAP_AVAILABLE:
        return np.full((X_test_raw.shape[0], X_test_raw.shape[1]), np.nan)

    X_imp   = np.where(np.isnan(X_test_raw), col_means, X_test_raw)
    X_scaled = scaler.transform(X_imp) * scale

    # LinearExplainer: arka plan = sıfır vektör (standartlaştırılmış uzayda ortalama)
    background = np.zeros((1, X_scaled.shape[1]))
    explainer  = _shap.LinearExplainer(model, background, feature_perturbation="interventional")
    shap_vals  = explainer.shap_values(X_scaled)
    return np.array(shap_vals)


# ---------------------------------------------------------------------------
# Ana LOCO döngüsü
# ---------------------------------------------------------------------------

def run_loco(
    programs: list[str] | None = None,
    domains:  list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    est, weights = load_data()

    if programs:
        est = est[est["program"].isin([p.upper() for p in programs])]
    if domains:
        est = est[est["domain"].isin([d.lower() for d in domains])]

    panel = build_panel(est)
    all_feature_cols = [c for c in panel.columns
                        if c not in ("country_iso3", "cycle")]

    results_rows: list[dict] = []
    pred_rows:    list[dict] = []
    shap_rows:    list[dict] = []

    for (prog, dom), grp in est.groupby(["program", "domain"]):
        target_col = make_feature_key(prog, dom)
        if target_col not in panel.columns:
            continue

        cycles = sorted(grp["cycle"].unique())
        if len(cycles) < 2:
            log.warning("Yeterli cycle yok, atlanıyor: %s %s", prog, dom)
            continue

        log.info("LOCO → %s %s  (%d cycle)", prog, dom, len(cycles))

        var_name   = _DOMAIN_TO_VAR.get((prog, dom))
        w_domain   = weights.get(var_name, 0.5) if var_name else 0.5
        lit_w_vec  = np.array([
            weights.get(_DOMAIN_TO_VAR.get(
                (f.split("_")[0], "_".join(f.split("_")[1:])), f), 0.5)
            for f in all_feature_cols
        ])

        fold_preds: dict[str, list] = {
            "M0": [], "M1": [], "M2": [], "y_true": [], "countries": []
        }
        shap_accumulator: list[np.ndarray] = []  # fold SHAP değerleri (M1)

        for i, test_cycle in enumerate(cycles):
            train_cycles = [c for c in cycles if c < test_cycle]
            if not train_cycles:
                continue

            X_tr, y_tr, X_te, y_te, countries = build_xy(
                panel, all_feature_cols, target_col,
                train_cycles, test_cycle,
            )
            if X_tr is None or len(y_tr) < 3:
                log.warning("  Fold %d (%d): yetersiz veri", i, test_cycle)
                continue

            # M0
            y_m0, _, _, _, _ = fit_ridge(X_tr, y_tr, X_te, lit_weights=None)
            # M1
            y_m1, m1_model, m1_scaler, m1_scale, m1_means = fit_ridge(
                X_tr, y_tr, X_te, lit_weights=lit_w_vec
            )
            # SHAP (M1)
            shap_fold = compute_shap_values(m1_model, m1_scaler, m1_scale, m1_means, X_te)
            shap_accumulator.append(shap_fold)
            # M2: lag-1 (persistence)
            lag_cycle = max(train_cycles)
            lag_df    = panel[panel["cycle"] == lag_cycle].set_index("country_iso3")
            common_c  = countries
            y_m2 = lag_df.loc[common_c, target_col].values.astype(float) \
                   if target_col in lag_df.columns else np.full(len(y_te), np.nan)

            fold_preds["M0"].append(y_m0)
            fold_preds["M1"].append(y_m1)
            fold_preds["M2"].append(y_m2)
            fold_preds["y_true"].append(y_te)
            fold_preds["countries"].append(common_c)

            for model_name, y_pred in [("M0", y_m0), ("M1", y_m1), ("M2", y_m2)]:
                valid = ~np.isnan(y_pred)
                if valid.sum() < 2:
                    continue
                yt, yp = y_te[valid], y_pred[valid]
                results_rows.append({
                    "program": prog, "domain": dom,
                    "test_cycle": test_cycle,
                    "n_train": len(y_tr), "n_test": len(yt),
                    "model": model_name,
                    "RMSE": round(rmse(yt, yp), 4),
                    "MAE":  round(mae(yt, yp), 4),
                    "R2":   round(r2(yt, yp), 4),
                    "Spearman": round(spearman_r(yt, yp), 4),
                })

            for j, cnt in enumerate(common_c):
                pred_rows.append({
                    "program": prog, "domain": dom,
                    "test_cycle": test_cycle,
                    "country_iso3": cnt,
                    "y_true": round(float(y_te[j]), 4),
                    "y_M0":   round(float(y_m0[j]), 4),
                    "y_M1":   round(float(y_m1[j]), 4),
                    "y_M2":   round(float(y_m2[j]) if not np.isnan(y_m2[j]) else float("nan"), 4),
                })

        if fold_preds["y_true"]:
            all_true = np.concatenate(fold_preds["y_true"])
            all_m1   = np.concatenate(fold_preds["M1"])
            all_m2   = np.concatenate(fold_preds["M2"])
            valid = ~(np.isnan(all_m1) | np.isnan(all_m2))
            if valid.sum() >= 3:
                e1 = (all_true[valid] - all_m1[valid])
                e2 = (all_true[valid] - all_m2[valid])
                dm_stat, dm_p = diebold_mariano(e1, e2)
                log.info("  DM(M1 vs M2): stat=%.3f  p=%.3f", dm_stat, dm_p)
                results_rows.append({
                    "program": prog, "domain": dom,
                    "test_cycle": "ALL", "n_train": None, "n_test": int(valid.sum()),
                    "model": "DM_M1vM2",
                    "RMSE": None, "MAE": None, "R2": None,
                    "Spearman": None,
                    "DM_stat": round(dm_stat, 4), "DM_p": round(dm_p, 4),
                })

        # SHAP global önem (M1): fold SHAP'larını birleştir, mean |SHAP| hesapla
        if shap_accumulator and not all(np.all(np.isnan(s)) for s in shap_accumulator):
            all_shap = np.vstack(shap_accumulator)          # (toplam_gözlem, n_feat)
            mean_abs_shap = np.nanmean(np.abs(all_shap), axis=0)
            for feat_name, importance in zip(all_feature_cols, mean_abs_shap):
                shap_rows.append({
                    "program": prog,
                    "domain": dom,
                    "feature": feat_name,
                    "mean_abs_shap": round(float(importance), 6),
                })
            log.info("  SHAP hesaplandı: %d özellik", len(all_feature_cols))

    results_df = pd.DataFrame(results_rows)
    preds_df   = pd.DataFrame(pred_rows)
    shap_df    = pd.DataFrame(shap_rows)

    shap_df = pd.DataFrame(shap_rows)

    results_df.to_csv(OUT_RESULTS, index=False)
    preds_df.to_csv(OUT_PREDS,    index=False)
    if not shap_df.empty:
        shap_df.to_csv(OUT_SHAP, index=False)
        log.info("Kaydedildi: %s  (%d satır)", OUT_SHAP, len(shap_df))
    elif not _SHAP_AVAILABLE:
        log.warning("SHAP paketi yüklü değil; shap_values.csv oluşturulmadı.")

    log.info("Kaydedildi: %s  (%d satır)", OUT_RESULTS, len(results_df))
    log.info("Kaydedildi: %s  (%d satır)", OUT_PREDS,   len(preds_df))

    return results_df, preds_df


# ---------------------------------------------------------------------------
# Özet tablo
# ---------------------------------------------------------------------------

def print_summary(results_df: pd.DataFrame) -> None:
    if results_df.empty or "model" not in results_df.columns:
        print("Sonuç yok.")
        return
    metric_rows = results_df[results_df["model"].isin(["M0", "M1", "M2"])].copy()
    if metric_rows.empty:
        print("Sonuç yok.")
        return

    summary = (
        metric_rows
        .groupby(["program", "domain", "model"])[["RMSE", "MAE", "R2", "Spearman"]]
        .mean()
        .round(4)
        .reset_index()
    )
    print("\n=== LOCO Sonuç Özeti (ortalama fold metrikleri) ===")
    print(summary.to_string(index=False))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--programs", nargs="*", help="örn. PISA TIMSS")
    p.add_argument("--domains",  nargs="*", help="örn. mathematics reading")
    args = p.parse_args()

    results, preds = run_loco(programs=args.programs, domains=args.domains)
    print_summary(results)
