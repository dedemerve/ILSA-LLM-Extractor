"""
ADIM 2 — TimesFM Zero-Shot Test

KAVRAM:
  Zero-shot = modele hiç özel eğitim vermeden kullanmak.
  TimesFM Google'ın genel zaman serisi foundation modeli.
  Soru: Bu genel model, PISA verisi için persistence'tan iyi mi?

TEST TASARIMI:
  Program    : PISA matematik
  Train döngü: 2003, 2006, 2009, 2012, 2015  (geçmiş)
  Test döngü : 2022                           (tahmin hedefi)
  Kural      : Her ülke için, 2022'den önceki TÜM döngüleri
               bağlam (context) olarak TimesFM'e ver, 2022'yi tahmin et.
  Baseline   : Persistence (2015 puanı = 2022 tahmini)

ÇALIŞTIR:
  python3 scripts/step2_timesfm_zeroshot.py

ÇIKTILAR:
  outputs/step2/zeroshot_results.csv  — ülke bazlı tahminler
  outputs/step2/metrics_summary.csv   — MAE karşılaştırması
  outputs/step2/zeroshot_scatter.png  — gerçek vs tahmin grafiği
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT     = Path(__file__).resolve().parents[1]
PANEL    = ROOT / "outputs" / "stage5" / "unified_panel.csv"

# =============================================================================
# KONFİGÜRASYON — Bu bölümü değiştirerek farklı döngüler çalıştırılabilir
# =============================================================================
TEST_CYCLE  = 2022      # tahmin etmek istediğimiz döngü
PROGRAM     = "PISA"
DOMAIN      = "mathematics"
TARGET_COL  = "PISA_mathematics"
MIN_CONTEXT = 3         # en az bu kadar geçmiş döngü olmalı

# Çıktı dizini TEST_CYCLE'a bağlıdır:
#   2022 → outputs/step2/          (mevcut 2022 sonuçları korunur)
#   2025 → outputs/step2_2025/     (2022 sonuçları üzerine yazılmaz)
#   diğer → outputs/step2_<yıl>/
_LEGACY_2022_DIR = ROOT / "outputs" / "step2"
OUT_DIR = _LEGACY_2022_DIR if TEST_CYCLE == 2022 else ROOT / "outputs" / f"step2_{TEST_CYCLE}"
# Güvenlik kontrolü: 2022 dışındaki bir döngü legacy dizinine yazmamalı
assert not (TEST_CYCLE != 2022 and OUT_DIR == _LEGACY_2022_DIR), (
    f"Güvenlik hatası: TEST_CYCLE={TEST_CYCLE} için {_LEGACY_2022_DIR} dizinine yazılamaz; "
    "mevcut PISA 2022 sonuçları korunmalıdır."
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Fail-fast: hedef dizinde çıktı dosyaları zaten varsa dur (üzerine yazmayı önle)
_PLANNED_OUTPUTS = [
    OUT_DIR / "zeroshot_results.csv",
    OUT_DIR / "metrics_summary.csv",
    OUT_DIR / "zeroshot_scatter.png",
]
_existing_outputs = [str(p) for p in _PLANNED_OUTPUTS if p.exists()]
if _existing_outputs:
    raise FileExistsError(
        f"Çıktı dosyaları zaten mevcut — üzerine yazmak yerine duruyorum:\n"
        + "\n".join(f"  {p}" for p in _existing_outputs)
        + f"\nÇözüm: mevcut dosyaları taşıyın ya da TEST_CYCLE değerini değiştirin."
    )

# =============================================================================
# YARDIMCI FONKSİYON: Doğrusal trend tahmini (OLS)
# =============================================================================

def _linear_trend_pred(context_years, context_scores, target_year):
    """
    Geçmiş döngüler üzerinde OLS doğrusal trend fitting:
        puan = kesişim + eğim × yıl
    Hedef yıldaki tahmini (kesişim, eğim, tahmin) olarak döndürür.

    Kural: yalnızca target_year'dan ÖNCE gelen gözlemleri kullanır.
    Bağlam sayısı < 2 ise (np.nan, np.nan, np.nan) döner.
    context_years: gerçek değerlendirme yılları (indeks değil).
    """
    if len(context_years) < 2:
        return (np.nan, np.nan, np.nan)
    cy = np.array(context_years, dtype=np.float64)
    cs = np.array(context_scores, dtype=np.float64)
    slope, intercept = np.polyfit(cy, cs, 1)
    prediction = intercept + slope * float(target_year)
    return (float(intercept), float(slope), prediction)


# =============================================================================
# BÖLÜM 1: Veriyi hazırla
# =============================================================================
print("Veri hazırlanıyor...")
panel = pd.read_csv(PANEL)
sub = panel[(panel["program"] == PROGRAM) & (panel["domain"] == DOMAIN)].copy()
sub = sub.dropna(subset=[TARGET_COL]).sort_values(["country_iso3", "cycle"])

# Girdi tekilliği: aynı ülke-döngü çifti birden fazla satırda varsa sessizce seçim yapma
_dup_mask = sub.duplicated(subset=["country_iso3", "cycle"], keep=False)
if _dup_mask.any():
    _dups = sub[_dup_mask][["country_iso3", "cycle"]].drop_duplicates().values.tolist()
    raise ValueError(
        f"Ülke-döngü çifti tekrarı tespit edildi — veri kaynağını kontrol edin: {_dups}"
    )

# Test ülkeleri: TEST_CYCLE döngüsünde ölçülen ülkeler
test_countries = sub[sub["cycle"] == TEST_CYCLE]["country_iso3"].unique()
print(f"  {TEST_CYCLE} döngüsünde {len(test_countries)} ülke var.")

# Her ülke için context (TEST_CYCLE'dan önceki tüm döngüler) ve hedef puanı
# context_rows: cycle < TEST_CYCLE VE TARGET_COL non-null (dropna yukarıda uygulandı)
# Kural: gerçek değerlendirme yılları kullanılır; hedef döngü fit'e dahil edilmez.
records = []
skipped_countries = []  # dışlanan ülkeler: (iso3, sebep)
for country in test_countries:
    cdata = sub[sub["country_iso3"] == country].sort_values("cycle")
    context_rows = cdata[cdata["cycle"] < TEST_CYCLE]
    target_row   = cdata[cdata["cycle"] == TEST_CYCLE]

    if len(target_row) == 0:
        skipped_countries.append((country, "hedef döngü verisi yok"))
        continue
    if len(context_rows) < MIN_CONTEXT:
        skipped_countries.append((
            country,
            f"yetersiz geçmiş: {len(context_rows)} gözlem (gerekli: {MIN_CONTEXT})"
        ))
        continue

    records.append({
        "country":        country,
        "context_cycles": context_rows["cycle"].tolist(),   # gerçek yıllar (indeks değil)
        "context_scores": context_rows[TARGET_COL].tolist(), # non-null (dropna garantisi)
        "true_score":     float(target_row[TARGET_COL].iloc[0]),
        "last_score":     float(context_rows[TARGET_COL].iloc[-1]),  # persistence tahmini
    })

print(f"  Yeterli geçmişi olan: {len(records)} ülke (en az {MIN_CONTEXT} önceki döngü)")
if skipped_countries:
    print(f"  Dışlanan ülkeler ({len(skipped_countries)}):")
    for iso3, reason in skipped_countries:
        print(f"    {iso3}: {reason}")
print()

# =============================================================================
# BÖLÜM 2: TimesFM yükle
# =============================================================================
print("TimesFM yükleniyor... (ilk seferinde model indirilir ~500MB)")
print("Bu 1-3 dakika sürebilir, bekleyin.\n")

import torch
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"  Kullanılan cihaz: {device}")

try:
    from timesfm3 import TimesFM3Evaluator, ModelConfig
    config = ModelConfig(
        checkpoint_path="google/timesfm-3.0-pytorch",
        per_core_batch_size=16,
        device=device,
    )
    forecaster = TimesFM3Evaluator(config)
    print("  TimesFM 3.0 hazır.\n")
    USE_TIMESFM = True
except Exception as e:
    print(f"  TimesFM yüklenemedi: {e}")
    print("  Devam ediyoruz ama TimesFM sütunu boş kalacak.\n")
    USE_TIMESFM = False

# =============================================================================
# BÖLÜM 3: Tahmin yap
# =============================================================================
print("Tahminler hesaplanıyor...")

results = []
if USE_TIMESFM:
    # TimesFM batch olarak çalışır — tüm serileri birlikte gönder
    contexts = [np.array(r["context_scores"], dtype=np.float32) for r in records]

    # horizon=1: bir adım ilerisi (TEST_CYCLE)
    batch_outputs = list(forecaster.predict_batch(contexts, horizon=1, return_quantiles=False))

    # Batch çıktı sayısı ile ülke sayısı eşleşmeli — sıralama güvenliği
    if len(batch_outputs) != len(records):
        raise RuntimeError(
            f"TimesFM batch çıktı sayısı ülke sayısıyla eşleşmiyor: "
            f"beklenen={len(records)}, alınan={len(batch_outputs)}. "
            "Sonuçlar yazılmadan duruyorum."
        )

    for r, out in zip(records, batch_outputs):
        # out.forecast shape: (1,) — tek adım tahmini
        tfm_pred = float(out.forecast[0])
        lt_intercept, lt_slope, lt_pred = _linear_trend_pred(
            r["context_cycles"], r["context_scores"], TEST_CYCLE
        )

        results.append({
            "country":             r["country"],
            "n_context":           len(r["context_scores"]),
            "context_cycles_str":  ";".join(str(c) for c in r["context_cycles"]),  # gerçek yıllar
            "last_cycle":          r["context_cycles"][-1],
            "true_score":          r["true_score"],
            "persistence":         r["last_score"],
            "linear_trend":        lt_pred,
            "linear_intercept":    lt_intercept,
            "linear_slope":        lt_slope,
            "timesfm_zeroshot":    tfm_pred,
            "err_persistence":     abs(r["true_score"] - r["last_score"]),
            "err_linear":          abs(r["true_score"] - lt_pred) if not np.isnan(lt_pred) else np.nan,
            "err_timesfm":         abs(r["true_score"] - tfm_pred),
        })
else:
    for r in records:
        lt_intercept, lt_slope, lt_pred = _linear_trend_pred(
            r["context_cycles"], r["context_scores"], TEST_CYCLE
        )
        results.append({
            "country":             r["country"],
            "n_context":           len(r["context_scores"]),
            "context_cycles_str":  ";".join(str(c) for c in r["context_cycles"]),  # gerçek yıllar
            "last_cycle":          r["context_cycles"][-1],
            "true_score":          r["true_score"],
            "persistence":         r["last_score"],
            "linear_trend":        lt_pred,
            "linear_intercept":    lt_intercept,
            "linear_slope":        lt_slope,
            "timesfm_zeroshot":    np.nan,
            "err_persistence":     abs(r["true_score"] - r["last_score"]),
            "err_linear":          abs(r["true_score"] - lt_pred) if not np.isnan(lt_pred) else np.nan,
            "err_timesfm":         np.nan,
        })

df = pd.DataFrame(results)

# =============================================================================
# BÖLÜM 4: Metrikler
# =============================================================================
print("=" * 60)
print(f"SONUÇLAR: PISA Matematik {TEST_CYCLE} Tahmini")
print("=" * 60)

# --- Sayısal geçerlilik: çıktı dosyası yazılmadan önce ---
# true_score ve persistence: asla NaN/inf içermemeli (upstream dropna garantisi).
# linear_trend ve timesfm_zeroshot: NaN kabul edilir (yöntem başarısız), inf kabul edilmez.
_inf_true = (~np.isfinite(df["true_score"])).sum()
_inf_pers = (~np.isfinite(df["persistence"])).sum()
if _inf_true > 0 or _inf_pers > 0:
    raise ValueError(
        f"Geçersiz değer: true_score'da {_inf_true}, persistence'da {_inf_pers} "
        "sonlu olmayan (NaN/inf) satır var. Çıktı yazılmadan duruyorum."
    )
for _col in ["linear_trend", "timesfm_zeroshot"]:
    _inf_col = (~np.isfinite(df[_col]) & df[_col].notna()).sum()
    if _inf_col > 0:
        raise ValueError(
            f"Geçersiz değer: {_col} sütununda {_inf_col} sonlu olmayan (inf) değer var. "
            "Çıktı yazılmadan duruyorum."
        )

# --- Yöntem başına örnek sayısı ---
n_persist_valid = df["persistence"].notna().sum()
n_linear_valid  = df["linear_trend"].notna().sum()
n_tfm_valid     = df["timesfm_zeroshot"].notna().sum()
print(f"\n  Geçerli tahmin sayısı — Persistence: {n_persist_valid}, "
      f"Linear trend: {n_linear_valid}, TimesFM: {n_tfm_valid}")

# --- Ortak ülke kümesi: NaN ve inf değerler dışlanır; true_score da kontrol edilir ---
_common_cols = ["persistence", "linear_trend", "timesfm_zeroshot"]
_finite_mask = (
    df[_common_cols].apply(lambda c: np.isfinite(c)).all(axis=1)
    & np.isfinite(df["true_score"])
)
df_common = df[_finite_mask].copy()
if len(df_common) == 0:
    raise RuntimeError(
        "Ortak ülke kümesi BOŞ — üç yöntem için de geçerli tahmin yok. "
        "Adil karşılaştırma yapılamaz; değerlendirme durduruluyor."
    )
print(f"  Ortak ülke kümesi (3 yöntem için tam, sonlu): n={len(df_common)}")
_missing_from_common = sorted(set(df["country"]) - set(df_common["country"]))
if _missing_from_common:
    print(f"  Ortak kümeden çıkan ülkeler: {_missing_from_common}")

# Tüm ön kontroller geçtikten sonra tahmin sonuçlarını yaz
df.to_csv(OUT_DIR / "zeroshot_results.csv", index=False)
print(f"  {len(df)} ülke için tahmin tamamlandı; kaydedildi: {OUT_DIR / 'zeroshot_results.csv'}\n")

# --- Ortak kümede işaretli hatalar (prediction − actual; pozitif = aşırı tahmin) ---
df_common = df_common.copy()
df_common["signed_err_persistence"] = df_common["persistence"]    - df_common["true_score"]
df_common["signed_err_linear"]      = df_common["linear_trend"]   - df_common["true_score"]
df_common["signed_err_timesfm"]     = df_common["timesfm_zeroshot"] - df_common["true_score"]

mae_persist_common  = df_common["err_persistence"].mean()
mae_linear_common   = df_common["err_linear"].mean()
mae_tfm_common      = df_common["err_timesfm"].mean()
mse_persist_common  = df_common["signed_err_persistence"].mean()
mse_linear_common   = df_common["signed_err_linear"].mean()
mse_tfm_common      = df_common["signed_err_timesfm"].mean()

print(f"\n  [Ortak küme n={len(df_common)}] MAE:")
print(f"    Persistence  : {mae_persist_common:.2f}  (ortalama işaretli hata: {mse_persist_common:+.2f})")
print(f"    Linear trend : {mae_linear_common:.2f}  (ortalama işaretli hata: {mse_linear_common:+.2f})")
print(f"    TimesFM      : {mae_tfm_common:.2f}  (ortalama işaretli hata: {mse_tfm_common:+.2f})")
print(f"  [Not: işaretli hata = tahmin − gerçek; pozitif = aşırı tahmin]")

mae_persist = df["err_persistence"].mean()
print(f"\n  Persistence MAE     : {mae_persist:.2f} puan  (tam küme, n={len(df)})")

mae_linear = df["err_linear"].mean() if df["err_linear"].notna().any() else np.nan
if not np.isnan(mae_linear):
    delta_lin = mae_linear - mae_persist
    sign_lin = "↑ DAHA KÖTÜ" if delta_lin > 0 else "↓ DAHA İYİ"
    print(f"  Linear trend MAE    : {mae_linear:.2f} puan  ({sign_lin} {abs(delta_lin):.2f})")

if USE_TIMESFM and df["err_timesfm"].notna().sum() > 0:
    mae_tfm = df["err_timesfm"].mean()
    delta = mae_tfm - mae_persist
    sign = "↑ DAHA KÖTÜ" if delta > 0 else "↓ DAHA İYİ"
    print(f"  TimesFM zero-shot   : {mae_tfm:.2f} puan  ({sign} {abs(delta):.2f})")

    # Ülke bazlı: kazanım / kayıp / eşitlik — yalnızca ortak küme üzerinde
    # Kural: NaN veya inf içeren ülkeler hiçbir karşılaştırmaya dahil edilmez (df_common garantisi).
    _nc = len(df_common)
    tfm_wins  = (df_common["err_timesfm"] < df_common["err_persistence"]).sum()
    tfm_ties  = (df_common["err_timesfm"] == df_common["err_persistence"]).sum()
    tfm_loses = _nc - tfm_wins - tfm_ties
    print(f"\n  TimesFM vs Persistence [ortak küme n={_nc}]:")
    print(f"    TimesFM daha iyi: {tfm_wins},  eşit: {tfm_ties},  daha kötü: {tfm_loses}")
    lin_wins_vs_persist  = (df_common["err_linear"] < df_common["err_persistence"]).sum()
    lin_ties_vs_persist  = (df_common["err_linear"] == df_common["err_persistence"]).sum()
    lin_loses_vs_persist = _nc - lin_wins_vs_persist - lin_ties_vs_persist
    lin_wins_vs_tfm  = (df_common["err_linear"] < df_common["err_timesfm"]).sum()
    lin_ties_vs_tfm  = (df_common["err_linear"] == df_common["err_timesfm"]).sum()
    lin_loses_vs_tfm = _nc - lin_wins_vs_tfm - lin_ties_vs_tfm
    print(f"  Linear trend vs Persistence [ortak küme n={_nc}]:")
    print(f"    Linear daha iyi: {lin_wins_vs_persist},  eşit: {lin_ties_vs_persist},  daha kötü: {lin_loses_vs_persist}")
    print(f"  Linear trend vs TimesFM [ortak küme n={_nc}]:")
    print(f"    Linear daha iyi: {lin_wins_vs_tfm},  eşit: {lin_ties_vs_tfm},  daha kötü: {lin_loses_vs_tfm}")

    # En büyük iyileştirmeler ve kötüleşmeler — ortak küme üzerinde (NaN güvenli)
    df_common["improvement"] = df_common["err_persistence"] - df_common["err_timesfm"]  # pozitif = TimesFM iyi
    top5_better = df_common.nlargest(5, "improvement")[["country","true_score","persistence","timesfm_zeroshot","improvement"]]
    top5_worse  = df_common.nsmallest(5, "improvement")[["country","true_score","persistence","timesfm_zeroshot","improvement"]]

    print("\n  TimesFM en çok iyileştirdiği 5 ülke:")
    print(top5_better.to_string(index=False))
    print("\n  TimesFM en çok kötüleştirdiği 5 ülke:")
    print(top5_worse.to_string(index=False))

    # n_method: bu yöntem için geçerli tahmin sayısı (yöntem-özgü)
    # n_common: 3 yöntem için de geçerli tahmin bulunan ülke sayısı (adil karşılaştırma)
    # MAE_method: tam yöntem-özgü küme üzerinde MAE
    # MAE_common / mean_signed_error_common: yalnızca ortak küme üzerinde
    _n_common = len(df_common)
    metrics_rows = [
        {"model": "Persistence",
         "MAE_method": round(mae_persist, 2),        "n_method": int(n_persist_valid),
         "MAE_common": round(mae_persist_common, 2),  "mean_signed_error_common": round(mse_persist_common, 2),  "n_common": _n_common},
        {"model": "Linear_trend",
         "MAE_method": round(mae_linear, 2) if not np.isnan(mae_linear) else np.nan, "n_method": int(n_linear_valid),
         "MAE_common": round(mae_linear_common, 2),   "mean_signed_error_common": round(mse_linear_common, 2),   "n_common": _n_common},
        {"model": "TimesFM_zero-shot",
         "MAE_method": round(mae_tfm, 2),             "n_method": int(n_tfm_valid),
         "MAE_common": round(mae_tfm_common, 2),      "mean_signed_error_common": round(mse_tfm_common, 2),      "n_common": _n_common},
    ]
    metrics = pd.DataFrame(metrics_rows)
else:
    # USE_TIMESFM=False: bu dal yalnızca df_common boş RuntimeError'ı atlatılırsa erişilebilir
    # (pratikte erişilmez — TimesFM NaN olduğunda df_common boş kalır ve RuntimeError fırlatılır)
    _n_common = len(df_common)
    metrics_rows = [
        {"model": "Persistence",
         "MAE_method": round(mae_persist, 2),        "n_method": int(n_persist_valid),
         "MAE_common": round(mae_persist_common, 2),  "mean_signed_error_common": round(mse_persist_common, 2),  "n_common": _n_common},
    ]
    if not np.isnan(mae_linear):
        metrics_rows.append(
            {"model": "Linear_trend",
             "MAE_method": round(mae_linear, 2),          "n_method": int(n_linear_valid),
             "MAE_common": round(mae_linear_common, 2),   "mean_signed_error_common": round(mse_linear_common, 2),  "n_common": _n_common}
        )
    metrics = pd.DataFrame(metrics_rows)

metrics.to_csv(OUT_DIR / "metrics_summary.csv", index=False)
print(f"\n  → Kaydedildi: {OUT_DIR / 'metrics_summary.csv'}")

# =============================================================================
# BÖLÜM 5: Görsel — Gerçek vs Tahmin scatter
# =============================================================================
if USE_TIMESFM and df["err_timesfm"].notna().sum() > 0:
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    fig.suptitle(f"PISA Matematik {TEST_CYCLE} — Gerçek vs Tahmin\n(Her nokta bir ülke)", fontsize=13)

    lim = (300, 630)
    err_keys = {"persistence": "err_persistence", "timesfm_zeroshot": "err_timesfm"}
    for ax, (col, label, color) in zip(axes, [
        ("persistence",      "Persistence (son döngü = tahmin)", "#dc2626"),
        ("timesfm_zeroshot", "TimesFM Zero-Shot",                "#2563eb"),
    ]):
        err_col = err_keys[col]
        mae_val = df[err_col].mean()
        ax.scatter(df["true_score"], df[col], alpha=0.6, color=color, s=40)
        ax.plot(lim, lim, "k--", linewidth=1, alpha=0.4, label="Mükemmel tahmin")
        ax.set_xlabel(f"Gerçek Puan ({TEST_CYCLE})")
        ax.set_ylabel("Tahmin")
        ax.set_title(f"{label}\nMAE = {mae_val:.1f}")
        ax.set_xlim(*lim); ax.set_ylim(*lim)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "zeroshot_scatter.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → Kaydedildi: {OUT_DIR / 'zeroshot_scatter.png'}")

print("\n" + "=" * 60)
print("Adım 2 Tamamlandı")
print("=" * 60)
print("""
Ne öğrendik:
  - TimesFM zero-shot persistence'ı geçiyor mu? (yukarıdaki MAE'lere bak)
  - Hangi ülkelerde modelin daha başarılı olduğunu gördük

Sıradaki soru (Adım 3):
  TimesFM'i ILSA verisine özel fine-tune edersek daha iyi olur mu?
  Bu bizim asıl iddiamız.
""")
