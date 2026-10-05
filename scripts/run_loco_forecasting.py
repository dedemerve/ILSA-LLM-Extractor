#!/usr/bin/env python3
"""
Stage 5 — LOCO Forecasting (M0 / M1 / M2 / M3) + forward predictions

When outputs/stage5/enriched_panel.csv (or predictor_weights_v2.csv) exists,
this script reads Stage 5 inputs and writes Stage 5 outputs. Otherwise it
falls back to Stage 4 paths.

Modeller:
  M0  — Ridge (veri güdümlü): lag özellikleri, literatür ağırlığı yok
  M1  — Ridge (literatür bilgili): özellikler sqrt(w_j) ile ölçeklenir
  M2  — Persistence: son gözlenen cycle skorunu tahmin olarak kullan
  M3  — AR(1) baseline

Temporal doğrulama:
  LOCO (Leave-One-Cycle-Out / expanding window): her cycle c için,
  c'den önceki çiftler train; c test.

predicted_cycle:
  Official intervals (PISA=3, TIMSS/TIMSS_G4=4, PIRLS=5, ICILS=5)
  with IEA override ICCS→2029. Mean historical gaps are not used.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.preprocessing import StandardScaler

try:
    import shap as _shap
    _SHAP_AVAILABLE = True
except ImportError:
    _SHAP_AVAILABLE = False

PROJECT_ROOT  = Path(__file__).resolve().parents[1]
STAGE4_DIR    = PROJECT_ROOT / "outputs" / "stage4"
STAGE5_DIR    = PROJECT_ROOT / "outputs" / "stage5"
# enriched_panel varsa onu kullan, yoksa eski country_estimates'e düş
_ENRICHED     = STAGE5_DIR / "enriched_panel.csv"
_ESTIMATES    = STAGE4_DIR / "country_estimates.csv"
ESTIMATES_CSV = _ENRICHED if _ENRICHED.exists() else _ESTIMATES
# v2 ağırlıklar varsa onu kullan, yoksa v1'e düş
_WEIGHTS_V2   = STAGE5_DIR / "predictor_weights_v2.csv"
_WEIGHTS_V1   = STAGE4_DIR / "predictor_weights.csv"
WEIGHTS_CSV   = _WEIGHTS_V2 if _WEIGHTS_V2.exists() else _WEIGHTS_V1
_OUT_DIR    = STAGE5_DIR if (_ENRICHED.exists() or _WEIGHTS_V2.exists()) else STAGE4_DIR
OUT_RESULTS = _OUT_DIR / "loco_results.csv"
OUT_PREDS   = _OUT_DIR / "loco_predictions.csv"
OUT_SHAP    = _OUT_DIR / "shap_values.csv"
OUT_FWD     = _OUT_DIR / "forward_predictions.csv"


logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# Ridge alpha grid (cross-validated)
ALPHA_GRID = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

# Official / announced cycle intervals (years). Mean historical gaps are
# unreliable for irregular calendars (e.g. PISA 2015→2022→2025).
PROGRAM_CYCLE_INTERVAL: dict[str, int] = {
    "PISA": 3,
    "TIMSS": 4,
    "TIMSS_G4": 4,
    "PIRLS": 5,
    "ICILS": 5,
    "ICCS": 7,  # fallback; IEA announced ICCS 2029 overrides when last=2022
}
# Absolute next-cycle overrides (IEA/OECD announcements beat interval math)
OFFICIAL_NEXT_CYCLE: dict[str, int] = {
    "ICCS": 2029,
}

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
    # v2: feature_name sütunu; v1: variable sütunu
    key_col = "feature_name" if "feature_name" in wdf.columns else "variable"
    weights: dict[str, float] = dict(zip(wdf[key_col], wdf["w_norm"]))

    return est, weights


def make_feature_key(program: str, domain: str) -> str:
    return f"{program}_{domain}"


def build_panel(est: pd.DataFrame) -> pd.DataFrame:
    """Uzun formatı geniş (pivot) panele dönüştür.

    enriched_panel: score sütunları + lag_ kovaryat sütunları zaten geniş formda gelir.
    country_estimates (eski format): mean + program + domain → pivot gerekir.
    """
    # enriched_panel: lag_ sütunu varsa zaten pivot'lu
    if any(c.startswith("lag_") for c in est.columns):
        # Skor sütunları: PISA_mathematics vb. — program_domain adlı tüm sayısal sütunlar
        non_feat = {"country_iso3", "cycle", "program", "domain"}
        pivot = est.drop(columns=[c for c in ("program", "domain") if c in est.columns],
                         errors="ignore") \
                   .groupby(["country_iso3", "cycle"]).first().reset_index()
        return pivot

    # Eski format: mean + feat_key pivot
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

def _make_X(
    score_df: pd.DataFrame,
    cov_df: pd.DataFrame,
    countries: pd.Index,
    score_cols: list[str],
    cov_cols: list[str],
) -> np.ndarray:
    """Skor özelliklerini score_df'den, kovaryat özelliklerini cov_df'den al.

    Tasarım gereği: lag_ESCS_t = ESCS_{t-1} enriched_panel'de t satırında saklanır.
    Dolayısıyla Y_t tahmini için:
      - skor özellikleri (lag-1 başarı) → t-1 satırından
      - kovaryat özellikleri (lag_ESCS vb.) → t satırından (orada t-1 değeri var)
    """
    parts: list[np.ndarray] = []
    if score_cols:
        parts.append(score_df.loc[countries, score_cols].values.astype(float))
    if cov_cols:
        parts.append(cov_df.loc[countries, cov_cols].values.astype(float))
    if not parts:
        return np.empty((len(countries), 0))
    return np.hstack(parts)


def build_xy(
    panel: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    train_cycles: list[int],
    test_cycle: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.Index]:
    """Train ve test matrislerini oluşturur.

    Özellik kaynağı:
      - Skor özellikleri (PISA_mathematics vb.): lag döngüsünden (prev_cycle)
      - Kovaryat özellikleri (lag_ESCS vb.): hedef döngüden (curr_cycle)
        → çünkü enriched_panel'de lag_ESCS_t = ESCS_{t-1}
    Bu ayrım, eğitim çiftlerinde kovaryatın "iki döngü gerisi" yerine
    "bir döngü gerisi"nden gelmesini sağlar.
    """
    score_cols = [f for f in feature_cols if not f.startswith("lag_")]
    cov_cols   = [f for f in feature_cols if f.startswith("lag_")]

    lag_cycle = max(train_cycles)
    lag_df    = panel[panel["cycle"] == lag_cycle].set_index("country_iso3")
    target_df = panel[panel["cycle"] == test_cycle].set_index("country_iso3")

    common = lag_df.index.intersection(target_df.index)
    if len(common) == 0:
        return None, None, None, None, None

    # Test X: skor özellikleri lag_cycle'dan, kovaryat özellikleri test_cycle'dan
    avail_score = [c for c in score_cols if c in lag_df.columns]
    avail_cov   = [c for c in cov_cols   if c in target_df.columns]
    X_lag    = _make_X(lag_df, target_df, common, avail_score, avail_cov)
    y_target = target_df.loc[common, target_col].values.astype(float)

    # Efektif feature sırası: avail_score + avail_cov (caller'la tutarlı olsun)
    eff_feature_cols = avail_score + avail_cov

    # Train: her eğitim çifti (prev_cycle → curr_cycle)
    X_train_list, y_train_list = [], []
    sorted_train = sorted(train_cycles)
    for i, tc in enumerate(sorted_train):
        if i == 0:
            continue  # ilk cycle için önceki yok
        prev_c  = sorted_train[i - 1]
        prev_df = panel[panel["cycle"] == prev_c].set_index("country_iso3")
        curr_df = panel[panel["cycle"] == tc].set_index("country_iso3")
        common_tr = prev_df.index.intersection(curr_df.index)
        if len(common_tr) == 0:
            continue
        # Kovaryat: curr_df'den → lag_ESCS_tc = ESCS_{prev_c} ✓
        avail_score_tr = [c for c in score_cols if c in prev_df.columns]
        avail_cov_tr   = [c for c in cov_cols   if c in curr_df.columns]
        X_tr = _make_X(prev_df, curr_df, common_tr, avail_score_tr, avail_cov_tr)
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
    X_test         = X_lag[mask_test]
    y_test         = y_target[mask_test]
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


def _fit_ar1(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    feature_cols: list[str],
    target_col: str,
) -> np.ndarray:
    """M3: Pooled AR(1) — yalnızca hedef alanın kendi lag'ı ile OLS.

    Hedef sütun özellik listesinde yoksa (çapraz-program fold) NaN döndürür.
    """
    if target_col not in feature_cols:
        return np.full(X_test.shape[0], np.nan)
    idx = feature_cols.index(target_col)
    x_tr = X_train[:, idx:idx+1]
    x_te = X_test[:,  idx:idx+1]
    # NaN impute
    col_mean = float(np.nanmean(x_tr)) if np.any(~np.isnan(x_tr)) else 0.0
    x_tr = np.where(np.isnan(x_tr), col_mean, x_tr)
    x_te = np.where(np.isnan(x_te), col_mean, x_te)
    mask = ~np.isnan(y_train)
    if mask.sum() < 2:
        return np.full(X_test.shape[0], np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = LinearRegression().fit(x_tr[mask], y_train[mask])
    return m.predict(x_te)


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
# ILSA değişkeni → canonical predictor eşleştirme tablosu
# Kaynak: predictor_weights_v2.csv'deki feature_name sütunuyla eşleşir.
# Buradaki map, ILSA mikroveri değişken adını (lag_ prefix'i olmadan)
# literatür ağırlık tablosundaki feature_name'e çevirir.
# Karşılığı olmayan değişkenler 1.0 alır (bilgi yokluğu ≠ orta kanıt).
# ---------------------------------------------------------------------------
_ILSA_TO_CANONICAL: dict[str, str] = {
    # ----------------------------------------------------------------
    # FORECASTABLE (A): yeterli lag geçmişi mevcut
    # ----------------------------------------------------------------
    # PISA — ESCS = ebeveyn meslek+eğitim+ev kaynakları composite (OECD dok.)
    "ESCS":    "ESCS",
    # PISA — HOMEPOS ev kaynakları indeksi; ESCS ile r>0.7 → Ridge ile kontrol edilir
    "HOMEPOS": "HOMEPOS",
    # PISA — BELONG okul aidiyet indeksi; düşük SD (~0.21) ama anlamlı varyasyon var
    "BELONG":  "BELONG",
    # TIMSS G8 — BSDGEDUP TERSİ ölçek (1=yüksek eğitim, 6=düşük); β < 0 beklenir
    "BSDGEDUP":"PARENTAL_EDU",
    "BSDG07":  "PARENTAL_EDU",
    "BSDG08":  "PARENTAL_EDU",
    # PIRLS — ASDHEDUP aynı ters ölçek yapısı (BSDGEDUP ile karşılaştırılabilir)
    "ASDHEDUP":"PARENTAL_EDU",
    # PIRLS — ASDHELA ev dili; ebeveyn eğitim seviyesinin zayıf proxy'i
    "ASDHELA": "PARENTAL_EDU",
    "ASDHELB": "PARENTAL_EDU",

    # ----------------------------------------------------------------
    # TALIS contextual predictors (2015/2019/2022 target cycles)
    # talis_covariate_estimates already stores TALIS_2013→target_2015 lag
    "SECLSS":   "TEACHER_QUALITY",   # TALIS self-efficacy cls mgmt → W_j=0.8543
    "TCDISCS":  "DISCLIMA",          # TALIS disciplinary climate → W_j=0.4910
    "TEFFPROS": "EFFPD",             # TALIS effective PD → W_j=0.2547
    "TJSPROS":  "JOB_SAT_PROF",      # TALIS job satisfaction profession → W_j=0.0894

    # ----------------------------------------------------------------
    # NOT FORECASTABLE (B): ILSA'da var ama lag geçmişi yetersiz
    # ICTAVHOM/ICTAVSCH: yalnızca 2025 cycle'da mevcut → lag 2022=NaN
    # → X_train her zaman NaN; post-build_xy filter tarafından düşürülür
    # Bu değişkenler coverage tablosunda "temporal" olarak işaretlenir.
    # Buraya mapping EKLEME — model davranışını etkilememeli.
    # ----------------------------------------------------------------

    # ----------------------------------------------------------------
    # NOT FORECASTABLE (C): semantik eşleşme geçersiz
    # ITSEX (1=kız, 2=erkek) → ülke ortalaması her cycle ~1.50, SD<0.03
    # Gender gap ölçmüyor; GENDER_GAP için ayrı achievement gap hesabı gerekir.
    # Buraya mapping EKLEME.
    # ----------------------------------------------------------------
}

# ---------------------------------------------------------------------------
# Literatür ağırlık eşlemesi
# ---------------------------------------------------------------------------

def _feat_weight(f: str, feature_list: list[str], weights: dict[str, float] | None = None) -> float:
    """Özellik adından M1 literatür ağırlığını döndürür.

    lag_ESCS    → _ILSA_TO_CANONICAL['ESCS'] → weights['ESCS']
    lag_ITSEX   → _ILSA_TO_CANONICAL['ITSEX'] → weights['GENDER_GAP']
    PISA_math   → 1.0  (skor omurgası)

    Bilinmeyen kovaryat → 1.0 (kanıt yokluğu, "orta kanıt" değil)
    """
    if weights is None:
        return 1.0
    if f.startswith("lag_"):
        raw_name   = f[4:]                              # lag_ önekini kaldır
        canonical  = _ILSA_TO_CANONICAL.get(raw_name)  # semantik eşleştirme
        if canonical is None:
            return 1.0   # eşleşme yok → etkisiz bırak (0.5 gibi keyfi değer değil)
        return weights.get(canonical, 1.0)
    return 1.0


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
        # Kanıt yokluğu → 1.0 (nötr); 0.5 gibi keyfi "orta kanıt" değil
        w_domain   = weights.get(var_name, 1.0) if var_name else 1.0

        fold_preds: dict[str, list] = {
            "M0A": [], "M0": [], "M1": [], "M2": [], "M3": [], "y_true": [], "countries": []
        }
        shap_accumulator: list[np.ndarray] = []

        for i, test_cycle in enumerate(cycles):
            train_cycles = [c for c in cycles if c < test_cycle]
            if not train_cycles:
                continue

            lag_cycle = max(train_cycles)
            lag_df    = panel[panel["cycle"] == lag_cycle].set_index("country_iso3")

            # Tüm özelliklerle build_xy çağır; sonra X_tr üzerinde tamamen NaN olan
            # lag sütunlarını eле — bu katmanda filtrelemek metodolojik olarak doğru:
            # doluluk oranını yalnızca bu fold'un gerçek eğitim matrisinden oku,
            # tüm program satırlarına göre değil.
            X_tr_full, y_tr, X_te_full, y_te, countries = build_xy(
                panel, all_feature_cols, target_col, train_cycles, test_cycle,
            )

            if X_tr_full is not None:
                # Eğitim matrisinde tamamen NaN olan lag_ sütunları bu fold için gerçekten yok
                col_any_filled = np.array([
                    True if not f.startswith("lag_") else np.any(~np.isnan(X_tr_full[:, i]))
                    for i, f in enumerate(all_feature_cols)
                ])
                fold_feat = [f for f, ok in zip(all_feature_cols, col_any_filled) if ok]
                keep_idx  = [i for i, ok in enumerate(col_any_filled) if ok]
                X_tr = X_tr_full[:, keep_idx]
                X_te = X_te_full[:, keep_idx]
                dropped = [f for f, ok in zip(all_feature_cols, col_any_filled) if not ok]
                if dropped:
                    log.debug("  Fold %d: tamamen NaN lag sütunlar atıldı: %s", i, dropped)
            else:
                fold_feat = all_feature_cols
                X_tr = X_tr_full
                X_te = X_te_full

            fold_lit_w = np.array([_feat_weight(f, fold_feat, weights) for f in fold_feat])

            if X_tr is None or len(y_tr) < 3:
                log.warning("  Fold %d (%d): M0/M1/M3 atlandı (yetersiz train), M2 kaydediliyor", i, test_cycle)
                target_df = panel[panel["cycle"] == test_cycle].set_index("country_iso3")
                lag_df_   = panel[panel["cycle"] == lag_cycle].set_index("country_iso3")
                common_c_ = lag_df_.index.intersection(target_df.index)
                if len(common_c_) == 0:
                    continue
                y_te_  = target_df.loc[common_c_, target_col].values.astype(float)
                mask_  = ~np.isnan(y_te_)
                if mask_.sum() < 2:
                    continue
                common_c_ = common_c_[mask_]
                y_te_  = y_te_[mask_]
                y_m2_  = lag_df_.loc[common_c_, target_col].values.astype(float) \
                         if target_col in lag_df_.columns else np.full(len(y_te_), np.nan)
                for fp in ["M0A", "M0", "M1", "M3"]:
                    fold_preds[fp].append(np.full(len(y_te_), np.nan))
                fold_preds["M2"].append(y_m2_)
                fold_preds["y_true"].append(y_te_)
                fold_preds["countries"].append(common_c_)
                valid_m2 = ~np.isnan(y_m2_)
                if valid_m2.sum() >= 2:
                    yt_, yp_ = y_te_[valid_m2], y_m2_[valid_m2]
                    results_rows.append({
                        "program": prog, "domain": dom,
                        "test_cycle": test_cycle, "n_train": 0, "n_test": int(valid_m2.sum()),
                        "model": "M2",
                        "RMSE": round(rmse(yt_, yp_), 4), "MAE": round(mae(yt_, yp_), 4),
                        "R2":   round(r2(yt_, yp_), 4),   "Spearman": round(spearman_r(yt_, yp_), 4),
                    })
                for j, cnt in enumerate(common_c_):
                    pred_rows.append({
                        "program": prog, "domain": dom, "test_cycle": test_cycle,
                        "country_iso3": cnt,
                        "y_true": round(float(y_te_[j]), 4),
                        "y_M0A": float("nan"), "y_M0": float("nan"),
                        "y_M1": float("nan"), "y_M3": float("nan"),
                        "y_M2": round(float(y_m2_[j]) if not np.isnan(y_m2_[j]) else float("nan"), 4),
                    })
                continue

            # Başarı (score) ve bağlam (lag_) özellik indeksleri
            score_idx = [i for i, f in enumerate(fold_feat) if not f.startswith("lag_")]
            X_tr_score = X_tr[:, score_idx] if score_idx else np.empty((X_tr.shape[0], 0))
            X_te_score = X_te[:, score_idx] if score_idx else np.empty((X_te.shape[0], 0))

            # M0A: yalnızca başarı özellikleri (bağlam yok)
            if score_idx:
                y_m0a, _, _, _, _ = fit_ridge(X_tr_score, y_tr, X_te_score, lit_weights=None)
            else:
                y_m0a = np.full(len(y_te), np.nan)

            # M0: literatür ağırlığı yok (başarı + bağlam, ağırlıksız)
            y_m0, _, _, _, _ = fit_ridge(X_tr, y_tr, X_te, lit_weights=None)
            # M1: literatür ağırlıklı (skor=1.0, kovaryat=v2)
            y_m1, m1_model, m1_scaler, m1_scale, m1_means = fit_ridge(
                X_tr, y_tr, X_te, lit_weights=fold_lit_w
            )
            # M2: persistence (lag-1)
            common_c = countries
            y_m2 = lag_df.loc[common_c, target_col].values.astype(float) \
                   if target_col in lag_df.columns else np.full(len(y_te), np.nan)
            # M3: AR(1) pooled — yalnızca hedef alan lag ile OLS
            y_m3 = _fit_ar1(X_tr, y_tr, X_te, fold_feat, target_col)

            # SHAP (M1)
            shap_fold = compute_shap_values(m1_model, m1_scaler, m1_scale, m1_means, X_te)
            shap_accumulator.append((shap_fold, fold_feat))

            fold_preds["M0A"].append(y_m0a)
            fold_preds["M0"].append(y_m0)
            fold_preds["M1"].append(y_m1)
            fold_preds["M2"].append(y_m2)
            fold_preds["M3"].append(y_m3)
            fold_preds["y_true"].append(y_te)
            fold_preds["countries"].append(common_c)

            for model_name, y_pred in [("M0A", y_m0a), ("M0", y_m0), ("M1", y_m1), ("M2", y_m2), ("M3", y_m3)]:
                valid = ~np.isnan(y_pred)
                if valid.sum() < 2:
                    continue
                yt, yp = y_te[valid], y_pred[valid]
                results_rows.append({
                    "program": prog, "domain": dom,
                    "test_cycle": test_cycle,
                    "n_train": len(y_tr), "n_test": len(yt),
                    "model": model_name,
                    "RMSE": round(rmse(yt, yp), 4), "MAE": round(mae(yt, yp), 4),
                    "R2":   round(r2(yt, yp), 4),   "Spearman": round(spearman_r(yt, yp), 4),
                    "n_features": len(fold_feat),
                })

            for j, cnt in enumerate(common_c):
                pred_rows.append({
                    "program": prog, "domain": dom, "test_cycle": test_cycle,
                    "country_iso3": cnt,
                    "y_true": round(float(y_te[j]), 4),
                    "y_M0A": round(float(y_m0a[j]) if not np.isnan(y_m0a[j]) else float("nan"), 4),
                    "y_M0":  round(float(y_m0[j]), 4),
                    "y_M1":  round(float(y_m1[j]), 4),
                    "y_M2":  round(float(y_m2[j]) if not np.isnan(y_m2[j]) else float("nan"), 4),
                    "y_M3":  round(float(y_m3[j]) if not np.isnan(y_m3[j]) else float("nan"), 4),
                })

        if fold_preds["y_true"]:
            all_true = np.concatenate(fold_preds["y_true"])
            all_m1   = np.concatenate(fold_preds["M1"])
            all_m2   = np.concatenate(fold_preds["M2"])
            # DM: M1 vs M2
            valid = ~(np.isnan(all_m1) | np.isnan(all_m2))
            if valid.sum() >= 3:
                dm_stat, dm_p = diebold_mariano(
                    all_true[valid] - all_m1[valid],
                    all_true[valid] - all_m2[valid],
                )
                log.info("  DM(M1 vs M2): stat=%.3f  p=%.3f", dm_stat, dm_p)
                results_rows.append({
                    "program": prog, "domain": dom,
                    "test_cycle": "ALL", "n_train": None, "n_test": int(valid.sum()),
                    "model": "DM_M1vM2", "RMSE": None, "MAE": None, "R2": None,
                    "Spearman": None, "DM_stat": round(dm_stat, 4), "DM_p": round(dm_p, 4),
                })
            # DM: M0A vs M2 (bağlam katkısı kontrolü)
            all_m0a = np.concatenate(fold_preds["M0A"])
            valid_0a = ~(np.isnan(all_m0a) | np.isnan(all_m2))
            if valid_0a.sum() >= 3:
                dm0a_stat, dm0a_p = diebold_mariano(
                    all_true[valid_0a] - all_m0a[valid_0a],
                    all_true[valid_0a] - all_m2[valid_0a],
                )
                log.info("  DM(M0A vs M2): stat=%.3f  p=%.3f", dm0a_stat, dm0a_p)
                results_rows.append({
                    "program": prog, "domain": dom,
                    "test_cycle": "ALL", "n_train": None, "n_test": int(valid_0a.sum()),
                    "model": "DM_M0AvM2", "RMSE": None, "MAE": None, "R2": None,
                    "Spearman": None, "DM_stat": round(dm0a_stat, 4), "DM_p": round(dm0a_p, 4),
                })
            # DM: M1 vs M3
            all_m3 = np.concatenate(fold_preds["M3"])
            valid3 = ~(np.isnan(all_m1) | np.isnan(all_m3))
            if valid3.sum() >= 3:
                dm3_stat, dm3_p = diebold_mariano(
                    all_true[valid3] - all_m1[valid3],
                    all_true[valid3] - all_m3[valid3],
                )
                log.info("  DM(M1 vs M3): stat=%.3f  p=%.3f", dm3_stat, dm3_p)
                results_rows.append({
                    "program": prog, "domain": dom,
                    "test_cycle": "ALL", "n_train": None, "n_test": int(valid3.sum()),
                    "model": "DM_M1vM3", "RMSE": None, "MAE": None, "R2": None,
                    "Spearman": None, "DM_stat": round(dm3_stat, 4), "DM_p": round(dm3_p, 4),
                })

        # SHAP global önem (M1): fold bazlı birleştirme
        if shap_accumulator:
            # fold'larda farklı özellik seti olabilir → ortak sütunları hizala
            all_shap_rows: list[dict] = []
            for shap_arr, feat_list in shap_accumulator:
                if np.all(np.isnan(shap_arr)):
                    continue
                for obs_idx in range(shap_arr.shape[0]):
                    for fi, fn in enumerate(feat_list):
                        all_shap_rows.append({"feature": fn, "val": abs(float(shap_arr[obs_idx, fi]))})
            if all_shap_rows:
                shap_agg = (pd.DataFrame(all_shap_rows)
                            .groupby("feature")["val"].mean().reset_index())
                for _, row in shap_agg.iterrows():
                    shap_rows.append({
                        "program": prog, "domain": dom,
                        "feature": row["feature"],
                        "mean_abs_shap": round(float(row["val"]), 6),
                    })
                log.info("  SHAP hesaplandı: %d özellik", len(shap_agg))

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
# İleri tahmin (tüm geçmiş → sonraki döngü)
# ---------------------------------------------------------------------------

def predict_forward(
    programs: list[str] | None = None,
    domains:  list[str] | None = None,
) -> pd.DataFrame:
    """Tüm tarihsel veriye fit edip bir sonraki döngüyü tahmin eder."""
    est, weights = load_data()
    if programs:
        est = est[est["program"].isin([p.upper() for p in programs])]
    if domains:
        est = est[est["domain"].isin([d.lower() for d in domains])]

    panel = build_panel(est)
    all_feature_cols = [c for c in panel.columns if c not in ("country_iso3", "cycle")]

    fwd_rows: list[dict] = []

    for (prog, dom), grp in est.groupby(["program", "domain"]):
        target_col = make_feature_key(prog, dom)
        if target_col not in panel.columns:
            continue
        cycles = sorted(grp["cycle"].unique())
        if len(cycles) < 2:
            continue

        last_cycle = cycles[-1]
        if prog in OFFICIAL_NEXT_CYCLE:
            next_cycle = OFFICIAL_NEXT_CYCLE[prog]
        elif prog in PROGRAM_CYCLE_INTERVAL:
            next_cycle = last_cycle + PROGRAM_CYCLE_INTERVAL[prog]
        else:
            gaps = [cycles[k] - cycles[k - 1] for k in range(1, len(cycles))]
            next_cycle = last_cycle + int(round(sum(gaps) / len(gaps)))
            log.warning(
                "predicted_cycle: %s has no official interval; using mean gap → %d",
                prog, next_cycle,
            )

        score_cols = [f for f in all_feature_cols if not f.startswith("lag_")]
        cov_cols   = [f for f in all_feature_cols if f.startswith("lag_")]

        # Tüm (prev→curr) çiftlerini eğitim verisi yap
        # Skor özellikleri prev_df'den, kovaryatlar curr_df'den (lag_ESCS_t = ESCS_{t-1})
        X_tr_list, y_tr_list = [], []
        for k in range(1, len(cycles)):
            prev_df = panel[panel["cycle"] == cycles[k-1]].set_index("country_iso3")
            curr_df = panel[panel["cycle"] == cycles[k]].set_index("country_iso3")
            common  = prev_df.index.intersection(curr_df.index)
            if len(common) == 0:
                continue
            av_sc = [c for c in score_cols if c in prev_df.columns]
            av_cv = [c for c in cov_cols   if c in curr_df.columns]
            X_k   = _make_X(prev_df, curr_df, common, av_sc, av_cv)
            y_k   = curr_df.loc[common, target_col].values.astype(float)
            mask  = ~np.isnan(y_k)
            if mask.sum() == 0:
                continue
            X_tr_list.append(X_k[mask])
            y_tr_list.append(y_k[mask])

        if not X_tr_list:
            continue
        X_train_full = np.vstack(X_tr_list)
        y_train      = np.concatenate(y_tr_list)

        # Test X: skor last_cycle'dan, kovaryat "next_cycle" yoksa last_cycle'dan
        # (ileri tahmin için bir sonraki döngünün kovaryatı yok; last_cycle'ın lag_'ı kullanılır)
        last_df   = panel[panel["cycle"] == last_cycle].set_index("country_iso3")
        countries = last_df.index
        av_sc_t   = [c for c in score_cols if c in last_df.columns]
        av_cv_t   = [c for c in cov_cols   if c in last_df.columns]
        X_test_full = _make_X(last_df, last_df, countries, av_sc_t, av_cv_t)
        fold_feat   = av_sc_t + av_cv_t

        if len(X_train_full) < 3 or len(X_test_full) == 0:
            continue

        # Eğitim matrisinde tamamen NaN olan lag sütunlarını at
        col_ok   = np.array([
            True if not f.startswith("lag_") else np.any(~np.isnan(X_train_full[:, i]))
            for i, f in enumerate(fold_feat)
        ])
        fwd_feat = [f for f, ok in zip(fold_feat, col_ok) if ok]
        keep_idx = [i for i, ok in enumerate(col_ok) if ok]
        X_train  = X_train_full[:, keep_idx]
        X_test   = X_test_full[:,  keep_idx]
        fwd_lit_w = np.array([_feat_weight(f, fwd_feat, weights) for f in fwd_feat])

        y_m0, _, _, _, _   = fit_ridge(X_train, y_train, X_test, lit_weights=None)
        y_m1, _, _, _, _   = fit_ridge(X_train, y_train, X_test, lit_weights=fwd_lit_w)
        y_m3               = _fit_ar1(X_train, y_train, X_test, fwd_feat, target_col)

        for j, cnt in enumerate(countries):
            last_score = float(last_df.loc[cnt, target_col]) \
                         if target_col in last_df.columns else float("nan")
            fwd_rows.append({
                "program":          prog,
                "domain":           dom,
                "country_iso3":     cnt,
                "last_cycle":       last_cycle,
                "last_score":       round(last_score, 2),
                "predicted_cycle":  next_cycle,
                "y_M0":             round(float(y_m0[j]), 2),
                "y_M1":             round(float(y_m1[j]), 2),
                "y_M3":             round(float(y_m3[j]) if not np.isnan(y_m3[j]) else float("nan"), 2),
            })
        log.info("İleri tahmin: %s %s → %d  (%d ülke)", prog, dom, next_cycle, len(countries))

    fwd_df = pd.DataFrame(fwd_rows)
    fwd_df.to_csv(OUT_FWD, index=False)
    log.info("Kaydedildi: %s  (%d satır)", OUT_FWD, len(fwd_df))
    return fwd_df


# ---------------------------------------------------------------------------
# Özet tablo
# ---------------------------------------------------------------------------

def _print_literature_coverage() -> None:
    """Literatür predictor'larının forecasting'e transfer tablosunu basar.

    Üç durum (reviewer taksonomisi):
      A — Forecastable: ILSA ölçümü var ve yeterli lag geçmişi var
      B — Temporal gap: ILSA ölçümü var ama lag geçmişi yetersiz
      C — No measure:   ILSA'da uyumlu country-level ölçüm yok
    """
    weights_path = _OUT_DIR.parent / "predictor_weights_v2.csv"
    if not weights_path.exists():
        return
    w = pd.read_csv(weights_path)

    # A — forecastable canonical construct'lar (mapping'te var)
    forecastable_features = set(_ILSA_TO_CANONICAL.values())

    # B — ILSA'da var ama lag geçmişi yetersiz (temporal)
    temporal_gap = {"ICT_INDEX"}   # ICTAVHOM/ICTAVSCH yalnızca 2025'te var

    def _status(feature: str) -> str:
        if feature in forecastable_features:
            return "A — forecastable"
        if feature in temporal_gap:
            return "B — temporal gap (lag eksik)"
        return "C — no compatible measure"

    print("\n=== Literatür → Tahmin Transferi (Coverage) ===")
    print(f"{'Canonical Predictor':<22} {'Feature':<16} {'w_norm':>7}  Durum")
    print("-" * 75)
    for _, row in w.iterrows():
        status = _status(row["feature_name"])
        print(f"  {row['predictor_canonical']:<20} {row['feature_name']:<16} {row['w_norm']:>7.4f}  {status}")

    n_fore = sum(1 for _, r in w.iterrows() if _status(r["feature_name"]).startswith("A"))
    n_temp = sum(1 for _, r in w.iterrows() if _status(r["feature_name"]).startswith("B"))
    n_none = sum(1 for _, r in w.iterrows() if _status(r["feature_name"]).startswith("C"))
    print(f"\nA (forecastable): {n_fore}  |  B (temporal): {n_temp}  |  C (no measure): {n_none}  |  Total: {len(w)}")


def print_summary(results_df: pd.DataFrame) -> None:
    if results_df.empty or "model" not in results_df.columns:
        print("Sonuç yok.")
        return
    metric_rows = results_df[results_df["model"].isin(["M0A", "M0", "M1", "M2", "M3"])].copy()
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

    # DM test özeti
    dm_rows = results_df[results_df["model"].isin(["DM_M1vM2", "DM_M1vM3"])].copy()
    if not dm_rows.empty:
        print("\n=== Diebold-Mariano Testi ===")
        print(dm_rows[["program", "domain", "model", "n_test", "DM_stat", "DM_p"]]
              .to_string(index=False))

    # Literature coverage tablosu (reviewer için)
    _print_literature_coverage()


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

    print("\n=== İleri Tahminler ===")
    fwd = predict_forward(programs=args.programs, domains=args.domains)
    if not fwd.empty:
        print(fwd.groupby(["program", "domain"])[["predicted_cycle"]].first().to_string())
        print(f"\nToplam {len(fwd)} ülke-alan tahmini → {OUT_FWD}")
