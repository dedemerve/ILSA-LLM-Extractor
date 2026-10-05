#!/usr/bin/env python3
"""
Stage 5 — Modül 2: Etki Büyüklüğü Tabanlı Predictor Ağırlıkları (v2)

v1'den farkı:
  - Sadece frekans × çeşitlilik × tutarlılık değil;
  - Ayrıca **medyan etki büyüklüğü** bileşeni (E_j) ekleniyor
  - Sadece country-level veya school-level çalışmalar dikkate alınır
    (student-level etki büyüklükleri ülke tahminini etkilemez)
  - pv_correct veya weight_correct = false olan çalışmalar yarım ağırlık alır

Formül:
  w_j = F_j * D_j * C_j * E_j
  E_j = normalize(median |effect_value|) for studies with effect_value != null
        = 0.5 for predictors with no quantitative effect sizes

Çıktı: outputs/stage5/predictor_weights_v2.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EFFECT_CSV   = PROJECT_ROOT / "outputs" / "stage5" / "effect_sizes.csv"
OUT_DIR      = PROJECT_ROOT / "outputs" / "stage5"
OUT_CSV      = OUT_DIR / "predictor_weights_v2.csv"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Mapping: predictor_canonical → forecasting feature
# (hangi PISA/TIMSS değişkeniyle eşleşiyor)
PREDICTOR_TO_FEATURE = {
    "SES_COMPOSITE":       "ESCS",
    "HOME_RESOURCES":      "HOMEPOS",
    "PARENTAL_EDUCATION":  "PARENTAL_EDU",
    "SCHOOL_BELONGING":    "BELONG",
    "BELONGING":           "BELONG",
    "MOTIVATION":          "MOTIVATION",
    "SELF_EFFICACY":       "SELF_EFFICACY",
    "ANXIETY":             "ANXIETY",
    "TEACHER_QUALITY":     "TEACHER_QUALITY",
    "TEACHER_SELF_EFFICACY_CLS_MGMT": "TEACHER_QUALITY",
    "TEACHER_SELF_EFFICACY_INSTRUCTION": "TEACHER_QUALITY",
    "TEACHER_SELF_EFFICACY_ENGAGEMENT": "TEACHER_QUALITY",
    "DISCIPLINARY_CLIMATE": "DISCLIMA",
    "SCHOOL_CLIMATE":      "DISCLIMA",
    "SCHOOL_RESOURCES":    "SCHOOL_RES",
    "ICT_HOME_ACCESS":     "ICT_INDEX",
    "ICT_ACCESS":          "ICT_INDEX",
    "ICT_SCHOOL_ACCESS":   "ICT_INDEX",
    "GENDER":              "GENDER_GAP",
    "INSTRUCTIONAL_TIME":  "INSTR_TIME",
    "PRIOR_ACHIEVEMENT":   "LAG_SCORE",
    "GDP_EXPENDITURE":     "GDP_PC",
    "HOME_LITERACY_ACTIVITIES": "PARENTAL_EDU",
    "PROF_COLLABORATION":  "TEACHER_QUALITY",
    "EFFECTIVE_PD":        "EFFPD",
    "JOB_SATISFACTION_PROFESSION": "JOB_SAT_PROF",
    "JOB_SATISFACTION_ENVIRONMENT": "JOB_SAT_PROF",
    "CIVIC_KNOWLEDGE":     "LAG_SCORE",
    "IMMIGRANT_STATUS":    "GENDER_GAP",
    "LANGUAGE_BACKGROUND": "GENDER_GAP",
    "AGE_GRADE":           "GENDER_GAP",
    "CURRICULUM_TYPE":     "MOTIVATION",
}


def compute_weights(df: pd.DataFrame) -> pd.DataFrame:
    # Sadece başarı hedefli çalışmalar
    achievement_outcomes = ["math", "reading", "science", "mathematics",
                             "achievement", "score", "performance"]
    mask_outcome = df["outcome"].str.lower().apply(
        lambda x: any(k in str(x) for k in achievement_outcomes)
    )
    df = df[mask_outcome].copy()

    # Metodoloji kalitesi düzeltmesi
    def quality_factor(row) -> float:
        factor = 1.0
        if row.get("pv_correct") is False:
            factor *= 0.5
        if row.get("weight_correct") is False:
            factor *= 0.7
        qf = {"high": 1.0, "medium": 0.8, "low": 0.5}.get(
            str(row.get("quality_flag", "")).lower(), 0.8
        )
        return factor * qf

    df["q_factor"] = df.apply(quality_factor, axis=1)

    records = []
    for predictor, grp in df.groupby("predictor_canonical"):
        if predictor == "OTHER":
            continue

        # F_j: kalite-ağırlıklı çalışma sayısı
        F = grp["q_factor"].sum()

        # D_j: kaç farklı program/domain kombinasyonu
        D = float(grp[["program", "domain"]].drop_duplicates().shape[0])

        # C_j: yön tutarlılığı
        directions = grp["direction"].value_counts()
        total_dir = directions.sum()
        dominant_dir = directions.idxmax() if len(directions) > 0 else "null"
        C = directions.max() / total_dir if total_dir > 0 else 0.5

        # E_j: medyan etki büyüklüğü (normalize edilmemiş ham değer)
        # Yalnızca karşılaştırılabilir ölçekler: standardized_beta, r_correlation, √R²
        # ML metrikleri (AUC, accuracy, F1) farklı ölçek aralığında olduğu için dışlanır
        EXCLUDED_TYPES = {"accuracy", "auc", "f1", "f1_score"}
        notna_mask = grp["effect_value"].notna()
        type_mask = ~grp["effect_type"].str.lower().isin(EXCLUDED_TYPES)
        valid_idx = notna_mask & type_mask
        vals = grp.loc[valid_idx, "effect_value"]
        if len(vals) > 0:
            abs_vals = vals.abs()
            # R² → √R² (yaklaşık r'ye dönüştür)
            r2_mask = grp.loc[valid_idx, "effect_type"].str.lower() == "r2"
            if r2_mask.any():
                abs_vals.loc[r2_mask[r2_mask].index] = np.sqrt(
                    abs_vals.loc[r2_mask[r2_mask].index]
                )
            E_raw = float(abs_vals.median())
        else:
            E_raw = np.nan

        records.append({
            "predictor_canonical": predictor,
            "feature_name":        PREDICTOR_TO_FEATURE.get(predictor, predictor),
            "F_raw":               float(F),
            "D_raw":               float(D),
            "C_raw":               float(C),
            "E_raw":               E_raw,
            "dominant_direction":  dominant_dir,
            "study_count":         int(len(grp["paper_id"].unique())),
            "effect_count":        int(grp["effect_value"].notna().sum()),
        })

    result = pd.DataFrame(records)

    # E_j normalize (0-1)
    e_vals = result["E_raw"].dropna()
    if len(e_vals) > 0 and e_vals.max() > e_vals.min():
        result["E_norm"] = (result["E_raw"] - e_vals.min()) / (e_vals.max() - e_vals.min())
    else:
        result["E_norm"] = 0.5
    result["E_norm"] = result["E_norm"].fillna(0.5)  # etki değeri olmayanlar = orta

    # w_raw = F * D * C * E
    result["w_raw"] = result["F_raw"] * result["D_raw"] * result["C_raw"] * result["E_norm"]

    # w_norm min-max
    w = result["w_raw"].values
    w_min, w_max = w.min(), w.max()
    if w_max > w_min:
        result["w_norm"] = (w - w_min) / (w_max - w_min)
    else:
        result["w_norm"] = 1.0
    result["w_norm"] = result["w_norm"].round(6)

    return result.sort_values("w_norm", ascending=False).reset_index(drop=True)


def main():
    if not EFFECT_CSV.exists():
        raise FileNotFoundError(
            f"effect_sizes.csv bulunamadı: {EFFECT_CSV}\n"
            "Önce extract_effect_sizes.py çalıştırın."
        )

    df = pd.read_csv(EFFECT_CSV)
    print(f"Yüklendi: {len(df)} etki büyüklüğü kaydı, {df['paper_id'].nunique()} makale")

    weights = compute_weights(df)
    weights.to_csv(OUT_CSV, index=False)

    print(f"\nKaydedildi: {OUT_CSV}  ({len(weights)} predictor)")
    print()
    cols = ["predictor_canonical", "feature_name", "w_norm", "dominant_direction",
            "study_count", "effect_count", "E_raw"]
    print(weights[cols].to_string(index=False))


if __name__ == "__main__":
    main()
