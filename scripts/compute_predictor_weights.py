#!/usr/bin/env python3
"""
Stage 4 — Modül 2: Literatür Öncelik Ağırlıkları

Her predictor değişkeni için w_j = F_j * D_j * C_j hesaplar.

  F_j (Frequency)   : değişkene ait toplam Study_Count (tüm method + trend)
  D_j (Diversity)   : değişkeni inceleyen farklı Canonical_Method sayısı
  C_j (Consistency) : baskın trend yönünün çalışma sayısı payı
                      max(count_positive, count_negative, count_null) / total_count

Ağırlıklar min-max normalizasyonuyla [0, 1] aralığına getirilir.

Çıktı: outputs/stage4/predictor_weights.csv
  Sütunlar: variable, F_raw, D_raw, C_raw, w_raw, w_norm,
            dominant_trend, dominant_share, study_count, method_count
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_CSV    = PROJECT_ROOT / "outputs" / "final_knowledge_synthesis.csv"
OUTPUT_DIR   = PROJECT_ROOT / "outputs" / "stage4"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_CSV   = OUTPUT_DIR / "predictor_weights.csv"


def compute_weights(df: pd.DataFrame) -> pd.DataFrame:
    empirical = df[df["Metadata_Filter_Flag"] == "empirical_finding"].copy()

    records = []
    for var, grp in empirical.groupby("Canonical_Variable"):
        total_count = grp["Study_Count"].sum()

        # F_j: toplam çalışma sayısı
        F = float(total_count)

        # D_j: kaç farklı yöntem bu değişkeni inceledi
        D = float(grp["Canonical_Method"].nunique())

        # C_j: baskın trend yönünün payı
        trend_counts = grp.groupby("Aggregate_Effect_Trend")["Study_Count"].sum()
        dominant_trend = trend_counts.idxmax()
        dominant_count = trend_counts.max()
        C = float(dominant_count / total_count) if total_count > 0 else 0.0

        records.append({
            "variable":        var,
            "F_raw":           F,
            "D_raw":           D,
            "C_raw":           C,
            "w_raw":           F * D * C,
            "dominant_trend":  dominant_trend,
            "dominant_share":  round(C, 4),
            "study_count":     int(total_count),
            "method_count":    int(D),
        })

    result = pd.DataFrame(records).sort_values("w_raw", ascending=False)

    # Min-max normalizasyonu
    w = result["w_raw"].values
    w_min, w_max = w.min(), w.max()
    if w_max > w_min:
        result["w_norm"] = (w - w_min) / (w_max - w_min)
    else:
        result["w_norm"] = 1.0

    result["w_norm"] = result["w_norm"].round(6)
    result = result.reset_index(drop=True)

    return result


def main() -> pd.DataFrame:
    df = pd.read_csv(INPUT_CSV)
    weights = compute_weights(df)
    weights.to_csv(OUTPUT_CSV, index=False)

    print(f"Kaydedildi: {OUTPUT_CSV}  ({len(weights)} değişken)")
    print()
    print(weights[["variable", "w_norm", "dominant_trend", "dominant_share",
                   "study_count", "method_count"]].to_string(index=False))
    return weights


if __name__ == "__main__":
    main()
