#!/usr/bin/env python3
"""
PISA 2003 ve 2009 TXT Kovaryat Parser

Sabit genişlikli ASCII dosyalarından ülke düzeyinde ağırlıklı ortalamalar hesaplar.
Çıktı: outputs/stage5/covariate_estimates_pisa_early.csv

PISA 2003 (INT_stui_2003_v2.txt, 1884 char/satır):
  CNT     : col 4-6   (ISO3 alpha, 0-indexed: 3:6)
  HOMEPOS : col 532-540, F9.4, missing >= 999
  BELONG  : col 559-567, F9.4, missing >= 999
  ESCS    : col 730-739, F10.5, missing >= 999
  W_FSTUWT: col 1100-1108, F9.4
  Source  : StQ_CodeBook_2003.pdf

PISA 2006 (veri yok — DATA GAP):
  Sadece cognitive item dosyaları mevcut (INT_Cogn06_*.txt).
  Student questionnaire (INT_Stu06_*.txt) arşivde bulunmuyor.

PISA 2009 (INT_STQ09_DEC11.txt, 1826 char/satır):
  CNT     : col 1-3   (ISO3 alpha, 0-indexed: 0:3)
  ESCS    : col 556-564, F8.2, missing >= 999
  HOMEPOS : col 583-591, F9.4, missing >= 999
  W_FSTUWT: col 905-913, F9.4
  Source  : Codebook_ERA_Stu09_June11.pdf (ERA codebook, INT_STQ09 ile aynı format)
  Not     : BELONG PISA2009'da ERA codebook kapsamı dışında; bu parse'da atlandı.
"""
from __future__ import annotations

import logging
import pathlib

import numpy as np
import pandas as pd

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA_ROOT    = pathlib.Path("/Users/mrved/Desktop/ILSA Datasets/PISA Datasets")
OUT_DIR      = PROJECT_ROOT / "outputs" / "stage5"
OUT_CSV      = OUT_DIR / "covariate_estimates_pisa_early.csv"
OUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ── Dosya ve kolon tanımları ──────────────────────────────────────────────────

PISA_2003 = {
    "cycle": 2003,
    "file":  DATA_ROOT / "PISA 2003 Data" / "INT_stui_2003_v2.txt",
    "cnt_slice":  (3, 6),      # col 4-6, 0-indexed
    "weight_slice": (1099, 1108),  # col 1100-1108, F9.4
    "missing_above": 990.0,
    "variables": {
        "ESCS":    (729, 739),    # col 730-739, F10.5
        "HOMEPOS": (531, 540),    # col 532-540, F9.4
        "BELONG":  (558, 567),    # col 559-567, F9.4
    },
}

PISA_2009 = {
    "cycle": 2009,
    "file":  DATA_ROOT / "PISA 2009 Data" / "INT_STQ09_DEC11.txt",
    "cnt_slice":  (0, 3),      # col 1-3, 0-indexed
    "weight_slice": (904, 913),    # col 905-913, F9.4
    "missing_above": 990.0,
    "variables": {
        "ESCS":    (555, 564),    # col 556-564, F8.2
        "HOMEPOS": (582, 591),    # col 583-591, F9.4
        # BELONG: ERA codebook dışında, konum doğrulanamadı
    },
}


def parse_fwf_covariates(spec: dict) -> pd.DataFrame:
    """Sabit genişlikli TXT'den ülke bazlı ağırlıklı kovaryat ortalamaları hesaplar."""
    fpath = spec["file"]
    cycle = spec["cycle"]
    if not fpath.exists():
        log.warning("Dosya bulunamadı: %s  → atlanıyor", fpath)
        return pd.DataFrame()

    log.info("Okuma: %s  (cycle=%d)", fpath.name, cycle)

    cnt_s, cnt_e = spec["cnt_slice"]
    w_s, w_e     = spec["weight_slice"]
    vars_map     = spec["variables"]
    miss_thresh  = spec["missing_above"]

    # Ülke bazında toplanacak yapı: {cnt: {var: ([val], [weight])}}
    accum: dict[str, dict[str, list]] = {}

    with open(fpath, encoding="latin-1") as fh:
        for line in fh:
            cnt = line[cnt_s:cnt_e]
            if len(cnt) != 3 or not cnt.isalpha():
                continue

            w_str = line[w_s:w_e].strip()
            try:
                w = float(w_str)
            except ValueError:
                continue
            if w <= 0 or w >= 999990:
                continue

            if cnt not in accum:
                accum[cnt] = {v: ([], []) for v in vars_map}

            for var, (vs, ve) in vars_map.items():
                s = line[vs:ve].strip()
                try:
                    val = float(s)
                except ValueError:
                    continue
                if val >= miss_thresh or val <= -miss_thresh:
                    continue
                accum[cnt][var][0].append(val)
                accum[cnt][var][1].append(w)

    records = []
    for cnt, var_data in accum.items():
        for var, (vals, wts) in var_data.items():
            if len(vals) < 10:
                continue
            wmean = float(np.average(vals, weights=wts))
            records.append({
                "program":      "PISA",
                "cycle":        cycle,
                "country_iso3": cnt,
                "variable":     var,
                "mean":         wmean,
                "n_students":   len(vals),
            })

    df = pd.DataFrame(records)
    log.info("  → %d ülke×değişken kaydı", len(df))
    return df


def main() -> None:
    dfs = []
    for spec in [PISA_2003, PISA_2009]:
        df = parse_fwf_covariates(spec)
        if not df.empty:
            dfs.append(df)

    if not dfs:
        log.error("Hiç veri üretilemedi.")
        return

    result = pd.concat(dfs, ignore_index=True)
    result.to_csv(OUT_CSV, index=False)
    log.info("Kaydedildi: %s  (%d satır)", OUT_CSV, len(result))

    # Özet
    print("\n── Özet ──────────────────────────────────")
    pivot = result.pivot_table(
        index=["cycle", "variable"], values="mean",
        aggfunc=["count", "mean", "std"]
    )
    print(pivot.to_string())

    # AUS kontrolü
    print("\n── AUS değerleri ──")
    aus = result[result["country_iso3"] == "AUS"][["cycle", "variable", "mean", "n_students"]]
    print(aus.to_string(index=False))

    # DATA GAP notu
    print("\n── VERİ EKSİKLİĞİ NOTU ──")
    print("PISA 2006: Student questionnaire (INT_Stu06_*.txt) arşivde mevcut değil.")
    print("  Mevcut: INT_Cogn06_S_Dec07.txt (cognitive items only).")
    print("  2006 kovaryatları bu run'da üretilemedi.")


if __name__ == "__main__":
    main()
