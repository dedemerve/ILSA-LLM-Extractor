"""
ADIM 1 — Veriyi Tanı ve Persistence Baseline

Bu script şunu yapar:
  1. Her ILSA programı için kaç ülke, kaç döngü, puan aralığı nedir — gösterir
  2. 5 örnek ülkenin puan eğrisini çizer (zaman içinde nasıl değişiyor)
  3. Persistence baseline'ı hesaplar:
       tahmin = ülkenin bir önceki döngüdeki puanı
       bu ne kadar doğru?
  4. Sonuçları sade bir tablo olarak kaydeder

ÇALIŞTIR:
  python3 scripts/step1_understand_data.py

ÇIKTILAR:
  outputs/step1/panel_summary.csv   — program bazlı özet
  outputs/step1/persistence_mae.csv — persistence baseline MAE
  outputs/step1/sample_trends.png   — örnek ülke eğrileri
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- Yollar ---
ROOT      = Path(__file__).resolve().parents[1]
PANEL_CSV = ROOT / "outputs" / "stage5" / "unified_panel.csv"
OUT_DIR   = ROOT / "outputs" / "step1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Hangi program-domain çiftlerini inceliyoruz ---
# Sadece öğrenci başarı puanı olan programlar
TARGETS = {
    ("PISA",     "mathematics"): "PISA_mathematics",
    ("PISA",     "reading"):     "PISA_reading",
    ("PISA",     "science"):     "PISA_science",
    ("TIMSS",    "mathematics"): "TIMSS_mathematics",
    ("TIMSS",    "science"):     "TIMSS_science",
    ("PIRLS",    "reading"):     "PIRLS_reading",
    ("ICCS",     "civic_knowledge"): "ICCS_civic_knowledge",
    ("ICILS",    "computer_literacy"): "ICILS_computer_literacy",
}

# --- Veriyi yükle ---
print("Veri yükleniyor...")
panel = pd.read_csv(PANEL_CSV)
print(f"  Panel boyutu: {panel.shape[0]} satır × {panel.shape[1]} sütun\n")

# =============================================================================
# BÖLÜM 1: Her program için özet tablo
# =============================================================================
print("=" * 60)
print("BÖLÜM 1: Panel Özeti")
print("=" * 60)

summary_rows = []
for (prog, dom), col in TARGETS.items():
    sub = panel[(panel["program"] == prog) & (panel["domain"] == dom)].copy()
    if col not in sub.columns:
        continue
    valid = sub.dropna(subset=[col])
    cycles = sorted(valid["cycle"].unique())
    n_countries = valid["country_iso3"].nunique()
    score_min = valid[col].min()
    score_max = valid[col].max()
    score_mean = valid[col].mean()

    row = {
        "program":   prog,
        "domain":    dom,
        "n_cycles":  len(cycles),
        "cycles":    f"{cycles[0]}–{cycles[-1]}",
        "n_countries": n_countries,
        "score_min": round(score_min, 1),
        "score_max": round(score_max, 1),
        "score_mean": round(score_mean, 1),
    }
    summary_rows.append(row)
    print(f"  {prog:8} {dom:20} | {len(cycles)} döngü ({cycles[0]}–{cycles[-1]}) "
          f"| {n_countries:3} ülke | puan {score_min:.0f}–{score_max:.0f}")

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(OUT_DIR / "panel_summary.csv", index=False)
print(f"\n  → Kaydedildi: outputs/step1/panel_summary.csv\n")

# =============================================================================
# BÖLÜM 2: Persistence Baseline — Ne kadar doğru?
#
# Persistence: "gelecek puan = geçmiş puan"
# Bu, herhangi bir modelin geçmesi gereken en basit basamak.
# Eğer modelimiz persistence'tan kötüyse, bir şeyler yanlış demektir.
# =============================================================================
print("=" * 60)
print("BÖLÜM 2: Persistence Baseline")
print("=" * 60)
print("  Persistence = 'bir sonraki döngüdeki puan = şimdiki puan' tahmini")
print()

mae_rows = []
for (prog, dom), col in TARGETS.items():
    sub = panel[(panel["program"] == prog) & (panel["domain"] == dom)].copy()
    if col not in sub.columns:
        continue
    sub = sub.dropna(subset=[col]).sort_values(["country_iso3", "cycle"])

    # Her ülke için: lag (bir önceki döngü puanı) ve hedef (şimdiki döngü puanı)
    sub["lag_score"] = sub.groupby("country_iso3")[col].shift(1)
    pairs = sub.dropna(subset=["lag_score", col])

    if len(pairs) < 5:
        continue

    errors = np.abs(pairs[col] - pairs["lag_score"])
    mae_val = errors.mean()
    median_err = errors.median()
    pct_within_10 = (errors <= 10).mean() * 100
    pct_within_20 = (errors <= 20).mean() * 100

    row = {
        "program": prog,
        "domain": dom,
        "n_pairs": len(pairs),
        "persistence_MAE": round(mae_val, 2),
        "median_error": round(median_err, 2),
        "pct_within_10pts": round(pct_within_10, 1),
        "pct_within_20pts": round(pct_within_20, 1),
    }
    mae_rows.append(row)
    print(f"  {prog:8} {dom:20} | MAE={mae_val:5.1f} puan | "
          f"±10pt içinde: {pct_within_10:.0f}% | n={len(pairs)}")

mae_df = pd.DataFrame(mae_rows)
mae_df.to_csv(OUT_DIR / "persistence_mae.csv", index=False)
print(f"\n  → Kaydedildi: outputs/step1/persistence_mae.csv")
print()
print("  NOT: MAE = ortalama mutlak hata (puan cinsinden)")
print("  Örnek: PISA MAE=15 → ortalamada 15 puan yanılıyoruz")
print("  PISA ölçeği ~300-600 arası, yani 15 puan yaklaşık %5 hata\n")

# =============================================================================
# BÖLÜM 3: Örnek ülke eğrileri — görsel
# =============================================================================
print("=" * 60)
print("BÖLÜM 3: Örnek Ülke Eğrileri")
print("=" * 60)

# PISA matematik için 6 ülke seç (farklı puan seviyelerinden)
prog, dom, col = "PISA", "mathematics", "PISA_mathematics"
sub = panel[(panel["program"] == prog) & (panel["domain"] == dom)].dropna(subset=[col])

# En az 5 döngüde görünen ülkeler
country_counts = sub.groupby("country_iso3")["cycle"].nunique()
eligible = country_counts[country_counts >= 5].index.tolist()

# Farklı puan seviyelerinden 6 ülke seç
country_means = sub[sub["country_iso3"].isin(eligible)].groupby("country_iso3")[col].mean().sort_values()
n = len(country_means)
sample_countries = [
    country_means.index[int(n * 0.10)],  # düşük
    country_means.index[int(n * 0.25)],
    country_means.index[int(n * 0.45)],
    country_means.index[int(n * 0.60)],
    country_means.index[int(n * 0.80)],
    country_means.index[int(n * 0.95)],  # yüksek
]

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
fig.suptitle("PISA Matematik — Örnek Ülke Eğrileri\n(Persistence: kesikli çizgi = bir önceki değer)",
             fontsize=13)

for ax, country in zip(axes.flatten(), sample_countries):
    cdata = sub[sub["country_iso3"] == country].sort_values("cycle")
    cycles = cdata["cycle"].values
    scores = cdata[col].values

    ax.plot(cycles, scores, "o-", linewidth=2, markersize=6, color="#2563eb", label="Gerçek puan")

    # Persistence tahmini (t-1 → t)
    if len(cycles) > 1:
        ax.plot(cycles[1:], scores[:-1], "s--", linewidth=1.5, markersize=5,
                color="#dc2626", alpha=0.7, label="Persistence tahmini")

    ax.set_title(f"{country}", fontsize=11, fontweight="bold")
    ax.set_xlabel("Döngü")
    ax.set_ylabel("Puan")
    ax.set_ylim(280, 640)
    ax.grid(True, alpha=0.3)
    if ax == axes[0][0]:
        ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(OUT_DIR / "sample_trends.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  → Kaydedildi: outputs/step1/sample_trends.png\n")

# =============================================================================
# ÖZET
# =============================================================================
print("=" * 60)
print("ÖZET — Adım 1 Tamamlandı")
print("=" * 60)
print()
print("Öğrendiklerimiz:")
print("  1. Hangi programlar için yeterli zaman serisi var")
print("  2. Persistence baseline MAE değerleri (hedef: bunu geçmek)")
print("  3. Ülkelerin puanları zaman içinde nasıl değişiyor")
print()
print("Sıradaki soru:")
print("  TimesFM zero-shot bu baseline'ı geçebilir mi?")
print("  (Adım 2: TimesFM kurulum ve zero-shot test)")
print()
print("Çıktılar:")
for f in sorted(OUT_DIR.glob("*")):
    print(f"  {f}")
