#!/usr/bin/env python3
"""
Human validation Cohen's Kappa hesaplama.

Kullanım:
  python3 scripts/compute_human_validation_kappa.py

Önce outputs/finetuning/audit/human_validation_sample_v1.csv dosyasındaki
human_canonical_construct sütununu doldurmuş olmalısın.
"""
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

GOLD  = PROJECT_ROOT / "outputs/finetuning/data/gold_standard_full_corpus_v1.4.1.csv"
HUMAN = PROJECT_ROOT / "outputs/finetuning/audit/human_validation_sample_v1.csv"
OUT   = PROJECT_ROOT / "outputs/finetuning/audit/kappa_comparison_v1.csv"


def cohen_kappa(y1: list, y2: list) -> float:
    """sklearn bağımlılığı olmadan Cohen's Kappa."""
    from collections import Counter
    n = len(y1)
    assert n == len(y2) and n > 0

    agree = sum(a == b for a, b in zip(y1, y2))
    p_o = agree / n

    cats = list(set(y1) | set(y2))
    c1 = Counter(y1)
    c2 = Counter(y2)
    p_e = sum((c1[c] / n) * (c2[c] / n) for c in cats)

    if p_e == 1.0:
        return 1.0
    return (p_o - p_e) / (1.0 - p_e)


def main() -> None:
    if not HUMAN.is_file():
        sys.exit(
            f"Human validation dosyası bulunamadı: {HUMAN}\n"
            "Adım 1'i tamamla: örnekleme betiğini çalıştır ve dosyayı doldur."
        )

    gold_df  = pd.read_csv(GOLD)[["record_id", "canonical_construct"]]
    human_df = pd.read_csv(HUMAN)[["record_id", "human_canonical_construct"]]

    human_df = human_df[
        human_df["human_canonical_construct"].notna() &
        (human_df["human_canonical_construct"].astype(str).str.strip() != "")
    ].copy()

    if human_df.empty:
        sys.exit(
            "human_canonical_construct sütunu boş. "
            "Adım 3'ü tamamla: manuel etiketleme yap."
        )

    merged = human_df.merge(gold_df, on="record_id", how="inner")
    print(f"Karşılaştırılan kayıt: {len(merged)}")

    y_llm   = merged["canonical_construct"].tolist()
    y_human = merged["human_canonical_construct"].str.strip().tolist()

    kappa     = cohen_kappa(y_llm, y_human)
    agreement = sum(a == b for a, b in zip(y_llm, y_human)) / len(y_llm)

    print(f"\nCohen's Kappa   : {kappa:.4f}")
    print(f"Basit uyum oranı: {agreement:.4f} ({agreement * 100:.1f}%)")
    print()

    if kappa >= 0.80:
        interp = "Güçlü uyum (tez için yeterli)"
    elif kappa >= 0.60:
        interp = "Kabul edilebilir uyum"
    else:
        interp = "Zayıf uyum — etiketleme protokolünü gözden geçir"
    print(f"Yorum: κ = {kappa:.2f} → {interp}")

    disagreements = merged[
        merged["canonical_construct"].str.strip() !=
        merged["human_canonical_construct"].str.strip()
    ].copy()

    if not disagreements.empty:
        print(f"\nUyuşmazlık ({len(disagreements)} satır):")
        print(
            disagreements[
                ["record_id", "canonical_construct", "human_canonical_construct"]
            ].to_string(index=False)
        )

    merged.to_csv(OUT, index=False)
    print(f"\nKarşılaştırma tablosu kaydedildi: {OUT.relative_to(PROJECT_ROOT)}")

    # Tez paragrafı taslağı
    print("\n--- TEZ PARAGRAFı TASLAĞI ---")
    print(
        f"Inter-rater reliability of the LLM-assisted label extraction was assessed\n"
        f"by independently re-coding a stratified random sample of\n"
        f"{merged['record_id'].nunique()} records. Cohen's kappa between the\n"
        f"LLM-assigned canonical construct labels and the independent human coder\n"
        f"was κ = {kappa:.2f} (simple agreement = {agreement * 100:.0f}%),\n"
        f"indicating {interp.lower()} (Landis & Koch, 1977)."
    )


if __name__ == "__main__":
    main()
