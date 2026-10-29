#!/usr/bin/env python3
"""
Predictor weights v4 — A1 corpus entegrasyonu.

v3'ten fark:
  - w_F, 80 manuel çalışma yerine A1 LoRA corpus'tan (1,266 çalışma) gelen
    study_count ile güncellenir.
  - Diğer bileşenler (w_P, w_C, w_N, w_E, forecastable) v3'ten korunur:
    bu metadata sadece manuel evidence_matrix'te mevcut.
  - UNMATCHED ve bilinmeyen canonical_construct'lar dışlanır.
  - Yeni construct'lar (v3'te olmayan) w=0 ile eklenir (forecastable=False).

Güncelleme formülü:
  n_studies_v4 = alpha * n_studies_v3 + (1-alpha) * n_studies_corpus_norm * n_studies_v3_max
  alpha = 0.3  (manuel veri kalite ağırlığı)

  w_F_v4 = log1p(n_studies_v4) / max(log1p(n_studies_v4))
  W_geo_v4 = geomean(w_F_v4, w_P, w_C, w_N) * w_E
  W_j_v4 = W_geo_v4 / max(W_geo_v4)
  w_v4 = W_j_v4 * forecastable / max(W_j_v4 * forecastable)

Çıktı: outputs/stage5/predictor_weights_v4.csv
       outputs/stage5/predictor_weights_v4_manifest.json
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CORPUS_JSON = Path("/Users/mrved/Downloads/ILSA_finetuning_v2_outputs/predictions"
                   "/A1_fullcorpus_qwen3_lora_corpus.json")
V3_CSV      = ROOT / "outputs" / "stage5" / "predictor_weights_v3.csv"
OUT_CSV     = ROOT / "outputs" / "stage5" / "predictor_weights_v4.csv"
OUT_MF      = ROOT / "outputs" / "stage5" / "predictor_weights_v4_manifest.json"

# Hangi canonical_construct'lar forecastable?
# (v3'ten alınır — bu listeyi değiştirmiyoruz)
ALPHA = 0.3   # manuel veri güven ağırlığı

UNMATCHED = {"UNMATCHED", "OTHER", "", "UNKNOWN"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_corpus_study_counts() -> pd.DataFrame:
    """A1 corpus'tan her canonical_construct için kaç benzersiz çalışma var."""
    with open(CORPUS_JSON) as f:
        corpus = json.load(f)

    study_constructs: dict[str, set[str]] = {}
    direction_counter: dict[str, dict[str, int]] = {}

    for rec in corpus["predictions"]:
        sid = rec["study_id"]
        for pair in rec.get("pairs", []):
            cc = (pair.get("canonical_construct") or "").strip()
            if not cc or cc in UNMATCHED:
                continue
            study_constructs.setdefault(cc, set()).add(sid)
            d = (pair.get("direction") or "").lower().strip()
            direction_counter.setdefault(cc, {})
            direction_counter[cc][d] = direction_counter[cc].get(d, 0) + 1

    rows = []
    for cc, studies in study_constructs.items():
        dirs = direction_counter.get(cc, {})
        total = sum(dirs.values()) or 1
        dominant = max(dirs, key=dirs.get) if dirs else "null"
        rows.append({
            "canonical_construct":   cc,
            "n_studies_corpus":      len(studies),
            "dominant_dir_corpus":   dominant,
            "dir_consistency_corpus": dirs.get(dominant, 0) / total,
        })

    return pd.DataFrame(rows).sort_values("n_studies_corpus", ascending=False)


def blend_weights(v3: pd.DataFrame, corpus: pd.DataFrame) -> pd.DataFrame:
    df = v3.merge(corpus[["canonical_construct", "n_studies_corpus"]],
                  on="canonical_construct", how="left")
    df["n_studies_corpus"] = df["n_studies_corpus"].fillna(0.0)

    n_max_v3     = float(df["n_studies"].max())
    n_max_corpus = float(corpus["n_studies_corpus"].max())

    # corpus study count'u v3 ölçeğine normalize et
    df["n_studies_corpus_scaled"] = (
        df["n_studies_corpus"] / n_max_corpus * n_max_v3
        if n_max_corpus > 0 else 0.0
    )

    # Ağırlıklı birleşim: ALPHA = manuel kalite güveni
    df["n_studies_v4"] = (
        ALPHA * df["n_studies"] + (1 - ALPHA) * df["n_studies_corpus_scaled"]
    )

    # w_F yeniden hesapla
    log_f = np.log1p(df["n_studies_v4"].values)
    df["w_F_v4"] = log_f / log_f.max()

    # W_geo yeniden hesapla (w_P, w_C, w_N, w_E v3'ten gelir)
    components = df[["w_F_v4", "w_P", "w_C", "w_N"]].values.astype(float).clip(0.01)
    geo = np.exp(np.log(components).mean(axis=1)) * df["w_E"].values.astype(float)
    df["W_geo_v4"] = geo
    df["W_j_v4"]   = np.round(geo / geo.max(), 4)

    # w (forecastable normalizasyonu)
    fc = np.where(df["forecastable"].astype(bool), df["W_j_v4"], 0.0)
    fc_max = fc.max()
    df["w_v4"] = np.round(fc / fc_max if fc_max > 0 else fc, 4)

    return df


def main() -> None:
    if OUT_CSV.exists():
        print(f"UYARI: {OUT_CSV} zaten var — üzerine yazılıyor.")

    v3     = pd.read_csv(V3_CSV)
    corpus = load_corpus_study_counts()

    print(f"v3: {len(v3)} construct | corpus: {len(corpus)} construct")
    print(f"Corpus top 10:")
    print(corpus.head(10)[["canonical_construct","n_studies_corpus"]].to_string(index=False))

    result = blend_weights(v3, corpus)

    # Çıktı sütunları
    out_cols = [
        "canonical_construct", "n_studies", "n_studies_corpus", "n_studies_v4",
        "n_rows", "n_programs", "n_cycles", "n_countries",
        "frac_predictive", "forecastable",
        "w_F", "w_F_v4", "w_P", "w_C", "w_N", "w_E",
        "W_geo", "W_geo_v4", "W_j", "W_j_v4", "w", "w_v4",
    ]
    result[out_cols].to_csv(OUT_CSV, index=False)
    print(f"\nYazıldı: {OUT_CSV}")

    # Manifest
    manifest = {
        "created_utc":   datetime.now(timezone.utc).isoformat(),
        "source_v3":     str(V3_CSV),
        "source_corpus": str(CORPUS_JSON),
        "sha256_v3":     sha256(V3_CSV),
        "sha256_corpus": sha256(CORPUS_JSON),
        "alpha":         ALPHA,
        "n_constructs_v3":     int(len(v3)),
        "n_constructs_corpus": int(len(corpus)),
        "description": (
            f"w_F updated using A1 LoRA full-corpus ({len(corpus)} constructs, "
            f"alpha={ALPHA} for manual data). Other components from v3."
        ),
    }
    with open(OUT_MF, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest: {OUT_MF}")

    # Değişim raporu
    changed = result[result["forecastable"].astype(bool)].copy()
    changed["delta_w"] = changed["w_v4"] - changed["w"]
    changed = changed.sort_values("delta_w", ascending=False)
    print("\nW değişimleri (forecastable construct'lar):")
    print(changed[["canonical_construct","n_studies","n_studies_corpus",
                    "n_studies_v4","w","w_v4","delta_w"]].to_string(index=False))


if __name__ == "__main__":
    main()
