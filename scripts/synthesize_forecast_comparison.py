#!/usr/bin/env python3
"""
Kapsamlı PISA forecasting karşılaştırma ve sentez analizi.

Yapılan işlemler:
1. Korpus yöntem frekansı özeti (1266 kayıt)
2. Tarihsel backtesting: RF, XGBoost, TimesFM, TimesFM+ILSA-LoRA (2012–2022)
3. PISA 2025 nihai karşılaştırma tablosu
4. İstatistiksel testler: Wilcoxon, Diebold-Mariano
5. Noise dose-response ve W_j → MAE zinciri
6. Kapsamlı Markdown raporu
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs/forecast_synthesis_v2"
OUT_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────
# 1. Korpus yöntem frekansı
# ─────────────────────────────────────────────────────────────
FREQ_CSV = ROOT / "analysis/03_data_feasibility/audit/ilsa_ml_method_frequency_v1.csv"

def load_method_frequency() -> pd.DataFrame:
    df = pd.read_csv(FREQ_CSV)
    df = df[df["scope"] == "predictive_studies_with_primary_method"].copy()
    df = df[["method", "study_count", "share_of_coded_scope"]].copy()
    df["share_pct"] = (df["share_of_coded_scope"] * 100).round(1)
    df = df.sort_values("study_count", ascending=False).reset_index(drop=True)
    return df

def load_country_method_frequency() -> pd.DataFrame:
    df = pd.read_csv(FREQ_CSV)
    return df[df["scope"] == "country_system_studies_with_primary_method"].copy()

# ─────────────────────────────────────────────────────────────
# 2. Tarihsel backtesting
# ─────────────────────────────────────────────────────────────
HIST_CSV = ROOT / "outputs/pisa_timesfm25_ils_adapter_v1/historical_metrics.csv"
HIST_PANEL = ROOT / "outputs/pisa_all_models_quadratic_trend_v1/historical_all_models_predictions.csv"
PISA2025_PANEL = ROOT / "outputs/pisa_all_models_quadratic_trend_v1/pisa2025_all_models_predictions.csv"

MODEL_LABELS = {
    "XGBoost": "XGBoost",
    "RandomForest": "Random Forest",
    "Ridge": "Ridge",
    "Persistence": "Persistence (baseline)",
    "TimesFM2_5_zero_shot": "TimesFM 2.5 (zero-shot)",
    "TimesFM2_5_ILSA_LoRA": "TimesFM 2.5 + ILSA LoRA",
    "TimesFM3_zero_shot": "TimesFM 3.0 (zero-shot)",
    "LinearTrend": "Linear Trend",
    "LinearQuadraticTrend": "Linear-Quadratic Trend",
}

def load_historical() -> pd.DataFrame:
    raw = pd.read_csv(HIST_PANEL)
    df = (raw.groupby(["target_cycle", "domain", "model"], as_index=False)
          .agg(MAE=("absolute_error", "mean")))
    df["model_label"] = df["model"].map(MODEL_LABELS).fillna(df["model"])
    return df

def historical_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby("model_label")["MAE"]
        .agg(mean_MAE="mean", std_MAE="std", min_MAE="min", max_MAE="max")
        .round(3)
        .sort_values("mean_MAE")
        .reset_index()
    )
    return summary

def historical_by_domain(df: pd.DataFrame) -> pd.DataFrame:
    pivot = (
        df.groupby(["model_label", "domain"])["MAE"]
        .mean()
        .round(3)
        .unstack("domain")
        .sort_values("mathematics")
        .reset_index()
    )
    return pivot

def historical_by_cycle(df: pd.DataFrame) -> pd.DataFrame:
    pivot = (
        df.groupby(["model_label", "target_cycle"])["MAE"]
        .mean()
        .round(3)
        .unstack("target_cycle")
        .sort_values(2022)
        .reset_index()
    )
    return pivot

# ─────────────────────────────────────────────────────────────
# 3. PISA 2025 nihai karşılaştırma
# ─────────────────────────────────────────────────────────────
PISA2025_CSV = ROOT / "outputs/pisa_2025_retrospective_timesfm_v1/metrics.csv"
PISA2025_PRED_CSV = PISA2025_PANEL

def load_2025_metrics() -> pd.DataFrame:
    raw = pd.read_csv(PISA2025_PANEL)
    df = (raw.groupby(["domain", "model"], as_index=False)
          .agg(MAE=("absolute_error", "mean")))
    df["model_label"] = df["model"].map(MODEL_LABELS).fillna(df["model"])
    return df

def pisa2025_summary(df: pd.DataFrame) -> pd.DataFrame:
    pivot = (
        df.pivot_table(index="model_label", columns="domain", values="MAE")
        .round(3)
        .reset_index()
    )
    pivot["mean_MAE"] = pivot[["mathematics", "reading", "science"]].mean(axis=1).round(3)
    pivot = pivot.sort_values("mean_MAE").reset_index(drop=True)
    return pivot

# ─────────────────────────────────────────────────────────────
# 4. İstatistiksel testler (ülke bazında hata farklılıkları)
# ─────────────────────────────────────────────────────────────

def wilcoxon_model_pairs(pred_df: pd.DataFrame) -> pd.DataFrame:
    """Her domain için XGBoost vs TimesFM2_5_ILSA_LoRA Wilcoxon signed-rank testi."""
    results = []
    for domain in ["mathematics", "reading", "science"]:
        sub = pred_df[pred_df["domain"] == domain].copy()
        # Pivot by country
        pivot = sub.pivot_table(index="country", columns="model", values="abs_error")
        comparisons = [
            ("XGBoost", "TimesFM2_5_ILSA_LoRA"),
            ("XGBoost", "RandomForest"),
            ("XGBoost", "Persistence"),
            ("TimesFM2_5_ILSA_LoRA", "TimesFM2_5_zero_shot"),
        ]
        for m1, m2 in comparisons:
            if m1 not in pivot.columns or m2 not in pivot.columns:
                continue
            diff = (pivot[m1] - pivot[m2]).dropna()
            if len(diff) < 5:
                continue
            stat, pval = stats.wilcoxon(diff, alternative="two-sided")
            n = len(diff)
            mean_diff = diff.mean()
            results.append({
                "domain": domain,
                "model_A": MODEL_LABELS.get(m1, m1),
                "model_B": MODEL_LABELS.get(m2, m2),
                "n_countries": n,
                "mean_diff_A_minus_B": round(mean_diff, 4),
                "interpretation": "A: lower mean AE" if mean_diff < 0 else "B: lower mean AE",
                "wilcoxon_statistic": round(stat, 2),
                "p_value": "<1e-16" if pval == 0 else f"{pval:.6g}",
                "raw_p_lt_005": pval < 0.05,
            })
    return pd.DataFrame(results)

def add_abs_error(pred_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(pred_csv)
    # actual CSV kolonları: target_cycle, domain, country_iso3, true_score, model, prediction, signed_error
    if "true_score" in df.columns and "prediction" in df.columns:
        df["abs_error"] = (df["prediction"] - df["true_score"]).abs()
        df = df.rename(columns={"country_iso3": "country"})
    elif "actual" in df.columns and "predicted" in df.columns:
        df["abs_error"] = (df["predicted"] - df["actual"]).abs()
    elif "pred" in df.columns and "obs" in df.columns:
        df["abs_error"] = (df["pred"] - df["obs"]).abs()
    else:
        print("  Uyarı: abs_error hesaplanamadı, kolonlar:", df.columns.tolist()[:10])
    return df

# ─────────────────────────────────────────────────────────────
# 5. Noise dose-response analizi
# ─────────────────────────────────────────────────────────────
NOISE_JSON = ROOT / "outputs/finetuning/v2/downstream/noise_all_standalone.json"

def load_noise_analysis() -> dict:
    with open(NOISE_JSON) as f:
        return json.load(f)

def noise_summary(data: dict) -> pd.DataFrame:
    rows = data.get("noise_dose_response", [])
    df = pd.DataFrame(rows)
    return df

def w_delta_summary(data: dict) -> dict:
    dw = data.get("delta_w", {})
    mae = data.get("mae", {})
    return {
        "oracle_MAE": mae.get("oracle"),
        "stage5_frozen_MAE": mae.get("stage5_frozen"),
        "delta_MAE": round(mae.get("oracle", 0) - mae.get("stage5_frozen", 0), 4),
        "W_L1_to_oracle": dw.get("stage5_frozen", {}).get("l1_to_oracle"),
        "W_spearman_to_oracle": dw.get("stage5_frozen", {}).get("spearman_to_oracle"),
        "W_direction_agreement": dw.get("stage5_frozen", {}).get("direction_agreement"),
        "evidence_rows_oracle": data.get("evidence_rows_mapped", {}).get("oracle"),
    }

# ─────────────────────────────────────────────────────────────
# 6. Rapor üretimi
# ─────────────────────────────────────────────────────────────

def format_table(df: pd.DataFrame, title: str) -> str:
    """Markdown table without tabulate dependency."""
    lines = [f"### {title}", ""]
    cols = list(df.columns)
    header = "| " + " | ".join(str(c) for c in cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    rows = []
    for _, row in df.iterrows():
        rows.append("| " + " | ".join(str(v) for v in row) + " |")
    lines += [header, sep] + rows + [""]
    return "\n".join(lines)

def write_report(
    method_freq: pd.DataFrame,
    country_method_freq: pd.DataFrame,
    hist_summary: pd.DataFrame,
    hist_domain: pd.DataFrame,
    hist_cycle: pd.DataFrame,
    pisa2025: pd.DataFrame,
    stats_tests: pd.DataFrame,
    noise_df: pd.DataFrame,
    w_delta: dict,
) -> str:

    # LoRA etkisi hesapla
    lora_row = pisa2025[pisa2025["model_label"] == "TimesFM 2.5 + ILSA LoRA"]
    zeroshot_row = pisa2025[pisa2025["model_label"] == "TimesFM 2.5 (zero-shot)"]
    lora_gain = ""
    if not lora_row.empty and not zeroshot_row.empty:
        delta = zeroshot_row["mean_MAE"].values[0] - lora_row["mean_MAE"].values[0]
        lora_gain = f"{delta:+.3f} PISA puanı (LoRA lehine)"

    lines = [
        "# PISA Tahmin Sistemi — Kapsamlı Sentez Raporu",
        "",
        f"**Tarih:** 2026-10-24  ",
        f"**Dalı:** `feat/finetuning-v2-pipeline`  ",
        f"**Hazırlayan:** Merve Dede  ",
        "",
        "---",
        "",
        "## Ana bulgu",
        "",
        "Persistence dört tarihsel hedef döngüsünde üç alanın tamamında en düşük eşit-ağırlıklı MAE'yi vermiştir. PISA 2025 sonuçlarına maruz kalınmış retrospektif karşılaştırmada ise alan liderleri ayrışmıştır. Bulgular ülke/ekonomi ortalama puanları içindir; öğrenci düzeyinde tahmin veya ileriye dönük bağımsız doğrulama değildir.",
        "",
        "## 1. Araştırma Sorusu ve Yaklaşım",
        "",
        "Bu çalışma, uluslararası öğrenci değerlendirmelerinden (ILSA) biri olan "
        "PISA'nın 2025 döngüsündeki matematik, okuma ve fen alanı ülke ortalamalarını "
        "tahmin etmek için tabular yöntemleri ve TimesFM'i karşılaştırmaktadır. Literatür ağırlıklı Stage 5/LOCO ve extraction fine-tuning analizleri ayrı araştırma hatlarıdır; bu dokuz-modelli PISA benchmark'ının katmanları değildir:",
        "",
        "1. **Tabular ML** — Literatürde en sık kullanılan yöntemler (RF, XGBoost, Ridge)",
        "2. **Zaman Serisi Modeli** — Google TimesFM (2.5 ve 3.0), sıfır-atış ve ILSA adaptasyonlu",
        "",
        "Tarihsel rolling-origin hedefleri 2012, 2015, 2018 ve 2022'dir; her tahminde hedef öncesindeki döngüler kullanılır. Tabular modeller havuzlanmış geçmiş geçişleri gecikme özellikleriyle kullanırken TimesFM mevcut sıralı geçmişi alır. 2025 tahminleri 2000–2022 geçmişine dayanır ve gerçekleşen hedeflerle sonuçlara maruz kalınmış retrospektif olarak karşılaştırılır.",
        "",
        "---",
        "",
        "## 2. Literatür Korpusu Yöntem Frekansı",
        "",
        "1.266 kayıtlık kodlanmış korpusta birincil yöntemi bulunan 147 tahminsel çalışma analiz edilmiştir. Özgün Scopus/WoS arama geçmişi, ham veri dışa aktarımları ve tarama kararları bulunmadığından bu sayım eksiksizliği doğrulanmış sistematik bir tarama değildir; sınırdaki çalışmalar için bağımsız ikinci kodlayıcı da beklenmektedir.",
        "",
        format_table(
            method_freq[["method", "study_count", "share_pct"]].head(12).rename(
                columns={"method": "Yöntem", "study_count": "Çalışma N", "share_pct": "147 kodlu tahminsel çalışmanın %'si"}
            ),
            "Tablo 1 — Birincil Yöntem Frekansı (tahminsel çalışmalar, N=147)"
        ),
        format_table(
            country_method_freq[country_method_freq["method"].isin(["SVM", "Random Forest", "XGBoost"])][["method", "study_count"]].rename(columns={"method": "Yöntem", "study_count": "Çalışma N"}),
            "Tablo 1b — Ülke Düzeyinde Tahmin Alt-Kümesindeki Yöntemler (N=17)"
        ),
        "Geniş tahminsel altkümede RF (42; %28,6) ve XGBoost (17; %11,6) en sık kodlanan yöntemlerdir. Dar ülke düzeyi altkümede SVM 4, RF 3 ve XGBoost 2 çalışmayla öndedir. RF/XGBoost genel karşılaştırıcı olarak gerekçelendirilebilir; doğrulanmış ülke düzeyi gelecek-döngü ILSA tahminlerinde en sık yöntemler oldukları söylenemez.",
        "",
        "---",
        "",
        "## 3. Tarihsel Backtesting (2012–2022)",
        "",
        "Dört rolling-origin hedefi (2012, 2015, 2018, 2022), üç alan ve ortak örnekler kullanılmıştır: hedef başına sırasıyla 40, 51, 57 ve 60 ülke/ekonomi. Genel ortalama, 12 alan×hedef MAE hücresinin eşit ağırlıklı ortalamasıdır; standart sapma bu 12 betimsel hücre üzerindedir ve 12 bağımsız zaman dönemi anlamına gelmez.",
        "",
        format_table(
            hist_summary.rename(columns={
                "model_label": "Model", "mean_MAE": "Ort. MAE",
                "std_MAE": "Std", "min_MAE": "Min", "max_MAE": "Max"
            }),
            "Tablo 2 — Tarihsel Backtesting Özeti (MAE, N=12 fold)"
        ),
        format_table(
            hist_domain.rename(columns={
                "model_label": "Model",
                "mathematics": "Matematik MAE",
                "reading": "Okuma MAE",
                "science": "Fen MAE"
            }),
            "Tablo 3 — Domain Bazında Tarihsel Ortalama MAE"
        ),
        format_table(
            hist_cycle.rename(columns={"model_label": "Model"}),
            "Tablo 4 — Döngü Bazında Tarihsel Ortalama MAE"
        ),
        "**Bulgular:**",
        "- Persistence üç alanın tamamında en düşük eşit-fold MAE'yi verdi; bu protokolde karmaşık yöntemler son-gözlem baseline'ını geçemedi.",
        "- XGBoost Persistence'a yakın olsa da tarihsel alan ortalamalarının hiçbirinde Persistence'ı geçmedi.",
        "- TimesFM 2.5 LoRA (ILSA adaptasyonlu), zero-shot'a göre ortalama ~0.09 MAE iyileştirme sağlar.",
        "- LoRA iyileşmesi her alanda 0.2 puanın altındadır. TimesFM 2.5 LoRA tarihselde Ridge ve RF'den daha düşük ortalama MAE verirken Persistence ve XGBoost'u geçememiştir; yöntem sıralaması modele ve alana göre değişir.",
        "- Becker–Coyle'un yayımlanmış 2022 doğrusal/kuadratik trend seçimi kodundan yeniden kurulan tahminler, ortak 60 ülke örneğinde üç alanda da Persistence ve karşılaştırılan tabular modellerden daha yüksek MAE vermiştir. Bu bir supplement-kodu rekonstrüksiyonudur; yayımlanmış hazır tahmin tablosu veya ileriye dönük doğrulama değildir.",
        "",
        "---",
        "",
        "## 4. PISA 2025 Nihai Karşılaştırma",
        "",
        "> **Dikkat — sonuçlara maruz kalınmış retrospektif değerlendirme:** Kod ve manifestlerde 2025 hedefi model girdisi/eğitim hedefi yapılmamıştır; ancak 2025 sonuçları ve önceki model sonuçları proje ekibinin erişimine açıktı. Bu nedenle bu karşılaştırma kör ve bağımsız ileriye dönük doğrulama değildir.",
        "",
        format_table(
            pisa2025.rename(columns={
                "model_label": "Model",
                "mathematics": "Matematik MAE",
                "reading": "Okuma MAE",
                "science": "Fen MAE",
                "mean_MAE": "Ortalama MAE"
            }),
            "Tablo 5 — PISA 2025 Model Karşılaştırması (66 ülke/ekonomi; alan başına)"
        ),
        "**Ana Bulgular:**",
        "- Alan liderleri farklıdır: matematikte XGBoost (9.318), okumada Ridge (11.357), fende Persistence (9.378).",
        "- Üç alan MAE'sinin basit ortalamasında Ridge (10.107), XGBoost'tan (10.660) düşüktür; bu sıralama da seçili örneklem ve retrospektif tasarıma özgüdür.",
        "- TimesFM 2.5 LoRA'nın üç-alan ortalama MAE'si Ridge'den 4.276 puan yüksektir; bu fark 2025 retrospektif karşılaştırmasına özgüdür.",
        f"- **ILSA LoRA'nın TimesFM'e katkısı:** {lora_gain}",
        "- Üç alan ve 66 ülke birlikte havuzlandığında dokuz modelin ortalama signed error'ı pozitiftir; alan bazında bu tüm modeller için geçerli değildir. Bu betimsel desen COVID-19 gibi nedenlere ilişkin nedensel açıklama sağlamaz.",
        "",
        "### Belirsizlik kapsamı",
        "Hedef puan örnekleme hatası/ülke-kümesi duyarlılığı (Phase 3.25) ve tarihsel girdi belirsizliğiyle model yeniden eğitimi (Phase 3.26) ayrı koşullu analizlerdir; birleşik aralık oluşturmaz. Phase 3.26'da 2.000 marjinal-SE çekiminde Persistence'ın en düşük eşit-fold MAE sıklığı matematikte %97,45, okumada %98,55, fende %99,75'tir; bu gelecek döngüler için olasılık değildir ve TimesFM yeniden eğitim belirsizliğini kapsamaz. Tam tarihsel ölçek-link kovaryansı doğrulanamadığı için link hatası da dahil değildir; sonuçlar tam belirsizlik yayılımı olarak sunulmamalıdır.",
        "",
        "---",
        "",
        "## 5. Keşfedici ülke-eşleştirmeli testler",
        "",
    ]

    if not stats_tests.empty:
        lines.append("> Bu Wilcoxon testleri 2025'e maruz kalınmış tek bir değerlendirmedeki ülke bazlı mutlak hataları eşleştirir. p değerleri çoklu karşılaştırma düzeltmesizdir; ülkeler arası bağımlılık, hedef puan örnekleme belirsizliği ve ölçek-link belirsizliği modellenmemiştir. Bu nedenle p değerleri doğrulayıcı üstünlük kanıtı sayılmamalı ve ana rapordaki koşullu küme-bootstrap/puan-SE duyarlılık analizlerinin yerine geçmemelidir.\n")
        lines.append(format_table(
            stats_tests.rename(columns={
                "domain": "Domain",
                "model_A": "Model A",
                "model_B": "Model B",
                "n_countries": "N Ülke",
                "mean_diff_A_minus_B": "Ortalama Δ (A−B)",
                "interpretation": "Düşük ort. mutlak hata",
                "wilcoxon_statistic": "W",
                "p_value": "p",
                "raw_p_lt_005": "Ham p<0.05?"
            }),
            "Tablo 6 — Wilcoxon Signed-Rank Testleri (ülke bazında hata farkı)"
        ))
    else:
        lines.append("*Tahmin dosyası mevcut değil; testler üretilemedi.*\n")

    lines += [
        "---",
        "",
        "## 6. Ayrı araştırma hattı: extraction → W_j → Stage 5 LOCO",
        "",
        "### 6.1 W_j Ağırlık Farkı (Downstream Evaluation)",
        "",
        "Bu bölüm ana dokuz-modelli PISA ülke ortalaması benchmark'ından farklı bir downstream deneydir; sonuçları PISA 2025 yöntem sıralamasına eklenmemelidir.",
        "",
        f"| Metrik | Değer |",
        f"|---|---|",
        f"| Oracle MAE (altın etki satırlarıyla) | {w_delta.get('oracle_MAE', '—')} |",
        f"| Stage 5 Frozen MAE (mevcut ağırlıklarla) | {w_delta.get('stage5_frozen_MAE', '—')} |",
        f"| ΔMAE (oracle − frozen) | {w_delta.get('delta_MAE', '—')} |",
        f"| L1 uzaklığı (W frozen → W oracle) | {w_delta.get('W_L1_to_oracle', '—')} |",
        f"| Spearman ρ (W frozen, W oracle) | {w_delta.get('W_spearman_to_oracle', '—')} |",
        f"| Yön uyumu (W frozen, W oracle) | {w_delta.get('W_direction_agreement', '—')} |",
        f"| W_j'ye eşlenen kanıt satırı (oracle) | {w_delta.get('evidence_rows_oracle', '—')} |",
        "",
        "**Yorum:** ΔMAE = oracle − frozen = −0.3946 olduğundan oracle MAE 0.3946 puan daha düşüktür (daha iyidir); eski yorumdaki 'oracle daha kötü' ifadesi ters işaretlidir. Oracle koşulu değerlendirilen split'in altın etiketlerini kullandığından deploy edilebilir extraction kolu veya bağımsız genelleme kanıtı değildir. Ağırlık benzerliği ölçüleri bu ayrı pipeline'a özgüdür.",
        "",
        "### 6.2 Noise Dose-Response (Yön Çevirme)",
        "",
    ]

    if not noise_df.empty:
        flip = noise_df[noise_df["perturbation"] == "flip"]
        drop = noise_df[noise_df["perturbation"] == "drop"]
        lines.append(format_table(
            flip[["level", "mae_mean", "mae_p2_5", "mae_p97_5"]].rename(columns={
                "level": "Flip oranı", "mae_mean": "MAE ort.",
                "mae_p2_5": "2.5% yüzdeliği", "mae_p97_5": "97.5% yüzdeliği"
            }),
            "Tablo 7 — Yön Flip Pertürbasyonu (0–100%; 100 tekrar/düzey)"
        ))
        lines.append(format_table(
            drop[["level", "mae_mean", "mae_p2_5", "mae_p97_5"]].rename(columns={
                "level": "Drop oranı", "mae_mean": "MAE ort.",
                "mae_p2_5": "2.5% yüzdeliği", "mae_p97_5": "97.5% yüzdeliği"
            }),
            "Tablo 8 — Çift Düşürme Pertürbasyonu (0–100%; 100 tekrar/düzey)"
        ))
        lines.append(
            "**Yorum:** Bu sentetik pertürbasyon tasarımında flip/drop ortalamaları baseline çevresindedir; %100 flip sonucunun baseline ile aynı olması pertürbasyon semantiğinin raporlanmasını gerektirir. Bu sonuç gerçek extraction hatalarının dağılımına genellenemez ve düşük duyarlılığı kanıtlamaz. "
            "Construct swap pertürbasyonu ise MAE'yi ~2 puan düşürmektedir — "
            "bu bir artifakt: yanlış construct seçimi tahmin dışı kalan "
            "bir predictor ağırlığını arttırarak bazen MAE'yi iyileştirebilir."
        )
    else:
        lines.append("*Noise analizi bulunamadı.*")

    lines += [
        "",
        "---",
        "",
        "## 7. Ayrı araştırma hattı: extraction fine-tuning",
        "",
        "| Kol | Model | Durum | Val Strict F1 |",
        "|---|---|---|---|",
        "| Oracle ceiling | — | ✅ Tamamlandı | 0.778 |",
        "| A0 base_qwen3_chunked | Qwen3-8B | ⬜ Yerel v2 değerlendirme çıktısı yok | — |",
        "| A1 qwen3_lora_selected | Qwen3-8B + LoRA | ⬜ Yerel v2 değerlendirme çıktısı yok | — |",
        "| A2 base_mistral_chunked | Mistral-7B | ⬜ Yerel v2 değerlendirme çıktısı yok | — |",
        "| A3 mistral_lora_selected | Mistral-7B + LoRA | ⬜ Yerel v2 değerlendirme çıktısı yok | — |",
        "| P3 zeroshot_qwen27b | qwen/qwen3.8-27b (sağlayıcı çıktı kaydında yok) | ✅ Tamamlandı | **0.000** |",
        "| Stage 7/9 LoRA (geçersiz) | Mistral-7B | ❌ 0/256 tensor yüklendi | — |",
        "",
        "**Yorum sınırı:** Zero-shot Qwen kolundaki strict F1=0.000 tek başına domain fine-tuning'in gerekli ya da üstün olduğunu kanıtlamaz. Aynı tahminlerin matcher tanımına göre farklı F1 alması ölçüm duyarlılığı gösterir. Geçerli base/LoRA kıyası için aynı split, aynı çıkarım/değerlendirme ayarları ve tamamlanmış eşleşik kollar gerekir. Yerel v2 değerlendirme envanterinde oracle ve zero-shot Qwen çıktıları bulunuyor; eşleşik base/LoRA kolları görünmüyor.",
        "",
        "**Sonraki adımlar:**",
        "1. Önce doğrulama bölmesinde aynı ayarlarla eşleşik base, retrieval/prompt ve LoRA kollarını tamamla.",
        "2. Eğitim kollarını GPU'da çalıştır ve her adapter'ın yüklendiğini doğrula.",
        "3. Checkpoint'i validation strict extraction metriğiyle seç; kolları aynı test ayarlarında değerlendir.",
        "4. Test bölümünü yalnızca protokol, model ve kod kilitlendikten sonra bir kez çalıştır.",
        "",
        "---",
        "",
        "## 8. Metodolojik Değerlendirme",
        "",
        "### ✅ Doğru Yapılanlar",
        "- Literatürden en sık kullanılan yöntemler (RF, XGBoost) seçilmiş ve uygulanmış",
        "- Retrospektif backtesting prensibi doğru: hedef döngüden önceki verilerle eğitim",
        "- TimesFM'e ILSA bağlamı verilmiş (unified_ilsa_panel ile LoRA adaptasyonu)",
        "- 2025 tahminleri 66 ülke için üretilmiş",
        "- Ayrı bir downstream deneyde sentetik W_j/noise duyarlılığı incelenmiş; sonuçları ana PISA benchmark'ından ayrılmıştır",
        "- Fine-tuning v2 protokolü belgelenmiş; önceden dışarıda kayıt altına alındığı doğrulanmamıştır; geçersiz Stage 7/9 sonucu ayrıca belgelenmiştir",
        "",
        "### ⚠️ Dikkat Edilmesi Gereken Sorunlar",
        "- **TimesFM LoRA katkısı marjinal (~0.09 MAE)** — ILSA context injection yeterli iyileştirme sağlamıyor",
        "- **Pooled overprediction betimseldir** — domain bazında bütün yöntemler pozitif sapma göstermiyor; bu sonuçtan neden/etki çıkarılamaz",
        "- **W_j/noise sonuçları ayrı exploratory downstream tasarıma özgüdür**; PISA benchmark'ına doğrudan aktarılmaz",
        "- **Fine-tuning kolları eksiktir** — yerel v2 envanteri eşleşik base/LoRA karşılaştırmasını göstermiyor",
        "",
        "### ❌ Geçersiz Sayılan Sonuçlar",
        "- Stage 7/9 Mistral LoRA F1 = 0.379 → base model sonucu, LoRA değil",
        "- P3 zero-shot F1 = 0.000 → yalnızca belirtilen strict matcher ve kol için sonuç; fine-tuning üstünlüğü kanıtı değildir",
        "",
        "---",
        "",
        "## 9. Özet",
        "",
        "| Soru | Cevap |",
        "|---|---|",
        "| Tahminsel literatürde en sık kodlananlar? | RF 42/147, XGBoost 17/147; ülke düzeyi altkümede SVM 4/17 |",
        "| 2025 alan liderleri? | Matematik XGBoost; okuma Ridge; fen Persistence; üç-alan ortalaması Ridge |",
        "| TimesFM ILSA LoRA işe yarıyor mu? | Tarihsel eşleşik MAE farkı <0.2; güçlü üstünlük kanıtı yok |",
        "| Extraction kalitesi forecasting'i etkiliyor mu? | Mevcut sentetik-noise deneyi bunu kesinleştirmiyor |",
        "| Fine-tuning tamamlandı mı? | Eşleşik base/LoRA kolları yerel envanterde tamamlanmış görünmüyor |",
        "| 2025 tahminleri var mı? | Evet — 66 ülke/ekonomi × 3 alan; yöntem-alan MAE aralığı 9.318–19.584 |",
        "| Bağımsız ileriye dönük doğrulama var mı? | Hayır; PISA 2029 için kod, veri, model ve tahminler sonuçlar açılmadan önce dondurulmalı |",
        "",
        "## Kaynak ve yeniden üretim",
        "- Ana tahmin panelleri: `outputs/pisa_all_models_quadratic_trend_v1/`.",
        "- Literatür sıklığı/kapsam notu: `analysis/03_data_feasibility/phase3_11_corpus_method_frequency.md`.",
        "- Araştırma amacı ve kanıt denetimi: `analysis/03_data_feasibility/phase3_33_objective_alignment_audit.md`.",
        "- Bu sentezi üreten betik: `scripts/synthesize_forecast_comparison.py`.",
    ]

    return "\n".join(lines)

# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main() -> None:
    print("Veriler yükleniyor...")

    # Yöntem frekansı
    method_freq = load_method_frequency()
    country_method_freq = load_country_method_frequency()
    method_freq.to_csv(OUT_DIR / "method_frequency.csv", index=False)
    print(f"  Yöntem frekansı: {len(method_freq)} satır")

    # Tarihsel backtesting
    hist_df = load_historical()
    hist_sum = historical_summary(hist_df)
    hist_dom = historical_by_domain(hist_df)
    hist_cyc = historical_by_cycle(hist_df)
    hist_sum.to_csv(OUT_DIR / "historical_summary.csv", index=False)
    hist_dom.to_csv(OUT_DIR / "historical_by_domain.csv", index=False)
    hist_cyc.to_csv(OUT_DIR / "historical_by_cycle.csv", index=False)
    print(f"  Tarihsel backtesting: {len(hist_df)} satır, {hist_df['target_cycle'].nunique()} döngü")

    # PISA 2025
    p25_df = load_2025_metrics()
    p25_sum = pisa2025_summary(p25_df)
    p25_sum.to_csv(OUT_DIR / "pisa2025_comparison.csv", index=False)
    print(f"  PISA 2025: {len(p25_df)} satır")

    # İstatistiksel testler
    stats_tests = pd.DataFrame()
    if PISA2025_PRED_CSV.exists():
        pred_df = add_abs_error(PISA2025_PRED_CSV)
        cols = pred_df.columns.tolist()
        if "model" in cols and "abs_error" in pred_df.columns:
            stats_tests = wilcoxon_model_pairs(pred_df)
            stats_tests.to_csv(OUT_DIR / "wilcoxon_tests.csv", index=False)
            print(f"  İstatistiksel testler: {len(stats_tests)} karşılaştırma")
        else:
            print(f"  Uyarı: tahmin CSV'de gerekli kolonlar yok: {cols[:8]}")
    else:
        print("  Uyarı: PISA 2025 tahmin CSV bulunamadı, testler atlandı")

    # Noise analizi
    noise_data = load_noise_analysis()
    noise_df = noise_summary(noise_data)
    w_delta = w_delta_summary(noise_data)
    noise_df.to_csv(OUT_DIR / "noise_dose_response.csv", index=False)
    print(f"  Noise analizi: {len(noise_df)} satır")

    # Rapor
    print("Rapor yazılıyor...")
    report_md = write_report(
        method_freq, country_method_freq, hist_sum, hist_dom, hist_cyc,
        p25_sum, stats_tests, noise_df, w_delta
    )
    report_path = OUT_DIR / "synthesis_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_md)
    print(f"  Rapor: {report_path}")

    # Konsol özeti
    print()
    print("=" * 60)
    print("PISA 2025 Model Karşılaştırması (Ortalama MAE, 66 ülke)")
    print("=" * 60)
    print(p25_sum[["model_label", "mathematics", "reading", "science", "mean_MAE"]].to_string(index=False))
    print()
    print("Tarihsel Backtesting Özeti (Ortalama MAE, 12 fold)")
    print("-" * 60)
    print(hist_sum[["model_label", "mean_MAE", "std_MAE"]].to_string(index=False))
    print()
    print(f"Çıktılar: {OUT_DIR}")


if __name__ == "__main__":
    main()
