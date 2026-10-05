#!/usr/bin/env python3
"""
Stage 4 — Modül 1: Ülke Düzeyinde Tahmin Üretimi

Her ILSA programı ve cycle için, öğrenci düzeyindeki mikroveriyi (SAV) okuyarak
ülke bazlı ağırlıklı ortalama ve standart hata hesaplar.

Varyans yöntemleri (tez Tablo 3.5):
  PISA        → BRR_Fay  (k=0.5, R=80)
  TIMSS/PIRLS/TALIS/ICCS/ICILS → JRR
  PIAAC       → BRR      (R=80)

PV pooling: Rubin (1987) — T = U_bar + (1 + 1/m) * B

Çıktı: outputs/stage4/country_estimates.csv
  Sütunlar: program, cycle, country_iso3, domain, mean, se, ci_lo, ci_hi,
            n_total, n_analytic, method, n_pv, n_replicates
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import pyreadstat

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR   = PROJECT_ROOT / "outputs" / "stage4"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

ILSA_BASE = Path("/Users/mrved/Desktop/ILSA Datasets")

# ---------------------------------------------------------------------------
# Sabitler
# ---------------------------------------------------------------------------
BRR_FAY_K = 0.5
BRR_FAY_R = 80
BRR_FAY_DENOM = BRR_FAY_R * (1 - BRR_FAY_K) ** 2   # = 20.0

# ---------------------------------------------------------------------------
# Program kataloğu: her cycle için SAV yolu, değişken adları
# ---------------------------------------------------------------------------
class CycleSpec(NamedTuple):
    program:      str
    cycle:        int
    domain:       str
    sav_path:     str          # PROJECT_ROOT'a göre göreceli
    country_var:  str          # ülke değişkeni
    weight_var:   str          # ana tasarım ağırlığı
    pv_vars:      list[str]    # plausible value değişkenleri (boş = PV yok)
    rep_prefix:   str          # replicate weight ön eki (örn. "W_FSTR")
    rep_range:    tuple[int,int] | None  # (1, 80) gibi; None = katalogdan oku
    method:       str          # BRR_FAY | JRR | BRR

class MultiFileCycleSpec(NamedTuple):
    """TIMSS/PIRLS gibi ülke başına ayrı SAV olan programlar için."""
    program:     str
    cycle:       int
    domain:      str
    data_dir:    str       # BSG*.sav dosyalarını içeren dizin
    file_glob:   str       # örn. "BSG*.sav" veya "AST*.sav"
    country_var: str
    weight_var:  str
    pv_vars:     list[str]
    rep_prefix:  str
    rep_range:   tuple[int, int] | None
    method:      str


CATALOG: list[CycleSpec] = [
    # ── PISA (tek uluslararası SAV) ───────────────────────────────────────
    CycleSpec("PISA", 2015, "mathematics",
        str(ILSA_BASE / "PISA Datasets/PISA 2015 Data"
            "/PUF_SPSS_COMBINED_CMB_STU_QQQ/CY6_MS_CMB_STU_QQQ.sav"),
        "CNT", "W_FSTUWT",
        [f"PV{i}MATH" for i in range(1, 11)],
        "W_FSTURWT", (1, 80), "BRR_FAY"),
    CycleSpec("PISA", 2015, "reading",
        str(ILSA_BASE / "PISA Datasets/PISA 2015 Data"
            "/PUF_SPSS_COMBINED_CMB_STU_QQQ/CY6_MS_CMB_STU_QQQ.sav"),
        "CNT", "W_FSTUWT",
        [f"PV{i}READ" for i in range(1, 11)],
        "W_FSTURWT", (1, 80), "BRR_FAY"),
    CycleSpec("PISA", 2015, "science",
        str(ILSA_BASE / "PISA Datasets/PISA 2015 Data"
            "/PUF_SPSS_COMBINED_CMB_STU_QQQ/CY6_MS_CMB_STU_QQQ.sav"),
        "CNT", "W_FSTUWT",
        [f"PV{i}SCIE" for i in range(1, 11)],
        "W_FSTURWT", (1, 80), "BRR_FAY"),

    # PISA 2018 ana student dosyası (CY07_MSU_STU_QQQ) indirmede eksik kalmış;
    # dosya temin edildiğinde buraya tam yolu ekle ve satırın başındaki # kaldır.
    # CycleSpec("PISA", 2018, "mathematics",
    #     str(ILSA_BASE / "PISA Datasets/PISA 2018 Data/CY07_MSU_STU_QQQ.sav"),
    #     "CNT", "W_FSTUWT",
    #     [f"PV{i}MATH" for i in range(1, 11)],
    #     "W_FSTURWT", (1, 80), "BRR_FAY"),

    CycleSpec("PISA", 2022, "reading",
        str(ILSA_BASE / "PISA Datasets/PISA 2022 Data"
            "/School questionnaire data file (CY08MSP_STU_QQQ).SAV"),
        "CNT", "W_FSTUWT",
        [f"PV{i}READ" for i in range(1, 11)],
        "W_FSTURWT", (1, 80), "BRR_FAY"),
    CycleSpec("PISA", 2022, "science",
        str(ILSA_BASE / "PISA Datasets/PISA 2022 Data"
            "/School questionnaire data file (CY08MSP_STU_QQQ).SAV"),
        "CNT", "W_FSTUWT",
        [f"PV{i}SCIE" for i in range(1, 11)],
        "W_FSTURWT", (1, 80), "BRR_FAY"),
    CycleSpec("PISA", 2022, "mathematics",
        str(ILSA_BASE / "PISA Datasets/PISA 2022 Data"
            "/School questionnaire data file (CY08MSP_STU_QQQ).SAV"),  # noqa: E501 — klasör adı parantez içeriyor
        "CNT", "W_FSTUWT",
        [f"PV{i}MATH" for i in range(1, 11)],
        "W_FSTURWT", (1, 80), "BRR_FAY"),

    CycleSpec("PISA", 2025, "mathematics",
        str(ILSA_BASE / "PISA Datasets/PISA 2025 Data/CY09_MS_STU_PUF.sav"),
        "CNT", "W_FSTUWT",
        [f"PV{i}MATH" for i in range(1, 11)],
        "W_FSTURWT", (1, 80), "BRR_FAY"),
    CycleSpec("PISA", 2025, "reading",
        str(ILSA_BASE / "PISA Datasets/PISA 2025 Data/CY09_MS_STU_PUF.sav"),
        "CNT", "W_FSTUWT",
        [f"PV{i}READ" for i in range(1, 11)],
        "W_FSTURWT", (1, 80), "BRR_FAY"),
    CycleSpec("PISA", 2025, "science",
        str(ILSA_BASE / "PISA Datasets/PISA 2025 Data/CY09_MS_STU_PUF.sav"),
        "CNT", "W_FSTUWT",
        [f"PV{i}SCIE" for i in range(1, 11)],
        "W_FSTURWT", (1, 80), "BRR_FAY"),
]

MULTI_CATALOG: list[MultiFileCycleSpec] = [
    # ── TIMSS Grade 8  (IEA Zone-Jackknife: JKZONE + JKREP + TOTWGT) ──────
    # ── TIMSS G8 mathematics ──────────────────────────────────────────────────
    MultiFileCycleSpec("TIMSS", 1995, "mathematics",
        str(ILSA_BASE / "TIMSS Datasets/TIMSS Data/TIMSS1995_IDB_SPSS_G8/Data"),
        "[Bb][Ss][Gg]*.sav", "IDCNTRY", "TOTWGT",
        [f"BSMMAT0{i}" for i in range(1, 6)], "", None, "JRR_ZONES"),
    MultiFileCycleSpec("TIMSS", 1999, "mathematics",
        str(ILSA_BASE / "TIMSS Datasets/TIMSS Data/TIMSS1999_IDB_SPSS_G8/Data"),
        "[Bb][Ss][Gg]*.sav", "IDCNTRY", "TOTWGT",
        [f"BSMMAT0{i}" for i in range(1, 6)], "", None, "JRR_ZONES"),
    MultiFileCycleSpec("TIMSS", 2003, "mathematics",
        str(ILSA_BASE / "TIMSS Datasets/TIMSS Data/TIMSS2003_IDB_SPSS_G8/Data"),
        "[Bb][Ss][Gg]*.sav", "IDCNTRY", "TOTWGT",
        [f"BSMMAT0{i}" for i in range(1, 6)], "", None, "JRR_ZONES"),
    MultiFileCycleSpec("TIMSS", 2007, "mathematics",
        str(ILSA_BASE / "TIMSS Datasets/TIMSS Data/TIMSS2007_IDB_SPSS_G8/Data"),
        "[Bb][Ss][Gg]*.sav", "IDCNTRY", "TOTWGT",
        [f"BSMMAT0{i}" for i in range(1, 6)], "", None, "JRR_ZONES"),
    MultiFileCycleSpec("TIMSS", 2011, "mathematics",
        str(ILSA_BASE / "TIMSS Datasets/TIMSS Data/TIMSS2011_IDB_SPSS_G8/Data"),
        "[Bb][Ss][Gg]*.sav", "IDCNTRY", "TOTWGT",
        [f"BSMMAT0{i}" for i in range(1, 6)], "", None, "JRR_ZONES"),
    MultiFileCycleSpec("TIMSS", 2015, "mathematics",
        str(ILSA_BASE / "TIMSS Datasets/TIMSS Data/TIMSS2015_IDB_SPSS_G8/Data"),
        "[Bb][Ss][Gg]*.sav", "IDCNTRY", "TOTWGT",
        [f"BSMMAT0{i}" for i in range(1, 6)], "", None, "JRR_ZONES"),
    MultiFileCycleSpec("TIMSS", 2019, "mathematics",
        str(ILSA_BASE / "TIMSS Datasets/TIMSS Data/TIMSS2019_IDB_SPSS_G8/Data"),
        "[Bb][Ss][Gg]*.sav", "IDCNTRY", "TOTWGT",
        [f"BSMMAT0{i}" for i in range(1, 6)], "", None, "JRR_ZONES"),
    MultiFileCycleSpec("TIMSS", 2023, "mathematics",
        str(ILSA_BASE / "TIMSS Datasets/TIMSS Data/TIMSS2023_IDB_SPSS_G8/2_Data Files/SPSS Data"),
        "[Bb][Ss][Gg]*.sav", "IDCNTRY", "TOTWGT",
        [f"BSMMAT0{i}" for i in range(1, 6)], "", None, "JRR_ZONES"),

    # ── TIMSS G8 science ──────────────────────────────────────────────────────
    MultiFileCycleSpec("TIMSS", 1995, "science",
        str(ILSA_BASE / "TIMSS Datasets/TIMSS Data/TIMSS1995_IDB_SPSS_G8/Data"),
        "[Bb][Ss][Gg]*.sav", "IDCNTRY", "TOTWGT",
        [f"BSSSCI0{i}" for i in range(1, 6)], "", None, "JRR_ZONES"),
    MultiFileCycleSpec("TIMSS", 1999, "science",
        str(ILSA_BASE / "TIMSS Datasets/TIMSS Data/TIMSS1999_IDB_SPSS_G8/Data"),
        "[Bb][Ss][Gg]*.sav", "IDCNTRY", "TOTWGT",
        [f"BSSSCI0{i}" for i in range(1, 6)], "", None, "JRR_ZONES"),
    MultiFileCycleSpec("TIMSS", 2003, "science",
        str(ILSA_BASE / "TIMSS Datasets/TIMSS Data/TIMSS2003_IDB_SPSS_G8/Data"),
        "[Bb][Ss][Gg]*.sav", "IDCNTRY", "TOTWGT",
        [f"BSSSCI0{i}" for i in range(1, 6)], "", None, "JRR_ZONES"),
    MultiFileCycleSpec("TIMSS", 2007, "science",
        str(ILSA_BASE / "TIMSS Datasets/TIMSS Data/TIMSS2007_IDB_SPSS_G8/Data"),
        "[Bb][Ss][Gg]*.sav", "IDCNTRY", "TOTWGT",
        [f"BSSSCI0{i}" for i in range(1, 6)], "", None, "JRR_ZONES"),
    MultiFileCycleSpec("TIMSS", 2011, "science",
        str(ILSA_BASE / "TIMSS Datasets/TIMSS Data/TIMSS2011_IDB_SPSS_G8/Data"),
        "[Bb][Ss][Gg]*.sav", "IDCNTRY", "TOTWGT",
        [f"BSSSCI0{i}" for i in range(1, 6)], "", None, "JRR_ZONES"),
    MultiFileCycleSpec("TIMSS", 2015, "science",
        str(ILSA_BASE / "TIMSS Datasets/TIMSS Data/TIMSS2015_IDB_SPSS_G8/Data"),
        "[Bb][Ss][Gg]*.sav", "IDCNTRY", "TOTWGT",
        [f"BSSSCI0{i}" for i in range(1, 6)], "", None, "JRR_ZONES"),
    MultiFileCycleSpec("TIMSS", 2019, "science",
        str(ILSA_BASE / "TIMSS Datasets/TIMSS Data/TIMSS2019_IDB_SPSS_G8/Data"),
        "[Bb][Ss][Gg]*.sav", "IDCNTRY", "TOTWGT",
        [f"BSSSCI0{i}" for i in range(1, 6)], "", None, "JRR_ZONES"),
    MultiFileCycleSpec("TIMSS", 2023, "science",
        str(ILSA_BASE / "TIMSS Datasets/TIMSS Data/TIMSS2023_IDB_SPSS_G8/2_Data Files/SPSS Data"),
        "[Bb][Ss][Gg]*.sav", "IDCNTRY", "TOTWGT",
        [f"BSSSCI0{i}" for i in range(1, 6)], "", None, "JRR_ZONES"),

    # ── TIMSS G4 mathematics ──────────────────────────────────────────────────
    MultiFileCycleSpec("TIMSS_G4", 1995, "mathematics",
        str(ILSA_BASE / "TIMSS Datasets/TIMSS Data/TIMSS1995_IDB_SPSS_G4/Data"),
        "[Aa][Ss][Gg]*.sav", "IDCNTRY", "TOTWGT",
        [f"ASMMAT0{i}" for i in range(1, 6)], "", None, "JRR_ZONES"),
    MultiFileCycleSpec("TIMSS_G4", 2003, "mathematics",
        str(ILSA_BASE / "TIMSS Datasets/TIMSS Data/TIMSS2003_IDB_SPSS_G4/Data"),
        "[Aa][Ss][Gg]*.sav", "IDCNTRY", "TOTWGT",
        [f"ASMMAT0{i}" for i in range(1, 6)], "", None, "JRR_ZONES"),
    MultiFileCycleSpec("TIMSS_G4", 2007, "mathematics",
        str(ILSA_BASE / "TIMSS Datasets/TIMSS Data/TIMSS2007_IDB_SPSS_G4/Data"),
        "[Aa][Ss][Gg]*.sav", "IDCNTRY", "TOTWGT",
        [f"ASMMAT0{i}" for i in range(1, 6)], "", None, "JRR_ZONES"),
    MultiFileCycleSpec("TIMSS_G4", 2011, "mathematics",
        str(ILSA_BASE / "TIMSS Datasets/TIMSS Data/TIMSS2011_IDB_SPSS_G4/Data"),
        "[Aa][Ss][Gg]*.sav", "IDCNTRY", "TOTWGT",
        [f"ASMMAT0{i}" for i in range(1, 6)], "", None, "JRR_ZONES"),
    MultiFileCycleSpec("TIMSS_G4", 2015, "mathematics",
        str(ILSA_BASE / "TIMSS Datasets/TIMSS Data/TIMSS2015_IDB_SPSS_G4/Data"),
        "[Aa][Ss][Gg]*.sav", "IDCNTRY", "TOTWGT",
        [f"ASMMAT0{i}" for i in range(1, 6)], "", None, "JRR_ZONES"),
    MultiFileCycleSpec("TIMSS_G4", 2019, "mathematics",
        str(ILSA_BASE / "TIMSS Datasets/TIMSS Data/TIMSS2019_IDB_SPSS_G4/Data"),
        "[Aa][Ss][Gg]*.sav", "IDCNTRY", "TOTWGT",
        [f"ASMMAT0{i}" for i in range(1, 6)], "", None, "JRR_ZONES"),
    MultiFileCycleSpec("TIMSS_G4", 2023, "mathematics",
        str(ILSA_BASE / "TIMSS Datasets/TIMSS Data/TIMSS2023_IDB_SPSS_G4/2_Data Files/SPSS Data"),
        "[Aa][Ss][Gg]*.sav", "IDCNTRY", "TOTWGT",
        [f"ASMMAT0{i}" for i in range(1, 6)], "", None, "JRR_ZONES"),

    # ── TIMSS G4 science ──────────────────────────────────────────────────────
    MultiFileCycleSpec("TIMSS_G4", 1995, "science",
        str(ILSA_BASE / "TIMSS Datasets/TIMSS Data/TIMSS1995_IDB_SPSS_G4/Data"),
        "[Aa][Ss][Gg]*.sav", "IDCNTRY", "TOTWGT",
        [f"ASSSCI0{i}" for i in range(1, 6)], "", None, "JRR_ZONES"),
    MultiFileCycleSpec("TIMSS_G4", 2003, "science",
        str(ILSA_BASE / "TIMSS Datasets/TIMSS Data/TIMSS2003_IDB_SPSS_G4/Data"),
        "[Aa][Ss][Gg]*.sav", "IDCNTRY", "TOTWGT",
        [f"ASSSCI0{i}" for i in range(1, 6)], "", None, "JRR_ZONES"),
    MultiFileCycleSpec("TIMSS_G4", 2007, "science",
        str(ILSA_BASE / "TIMSS Datasets/TIMSS Data/TIMSS2007_IDB_SPSS_G4/Data"),
        "[Aa][Ss][Gg]*.sav", "IDCNTRY", "TOTWGT",
        [f"ASSSCI0{i}" for i in range(1, 6)], "", None, "JRR_ZONES"),
    MultiFileCycleSpec("TIMSS_G4", 2011, "science",
        str(ILSA_BASE / "TIMSS Datasets/TIMSS Data/TIMSS2011_IDB_SPSS_G4/Data"),
        "[Aa][Ss][Gg]*.sav", "IDCNTRY", "TOTWGT",
        [f"ASSSCI0{i}" for i in range(1, 6)], "", None, "JRR_ZONES"),
    MultiFileCycleSpec("TIMSS_G4", 2015, "science",
        str(ILSA_BASE / "TIMSS Datasets/TIMSS Data/TIMSS2015_IDB_SPSS_G4/Data"),
        "[Aa][Ss][Gg]*.sav", "IDCNTRY", "TOTWGT",
        [f"ASSSCI0{i}" for i in range(1, 6)], "", None, "JRR_ZONES"),
    MultiFileCycleSpec("TIMSS_G4", 2019, "science",
        str(ILSA_BASE / "TIMSS Datasets/TIMSS Data/TIMSS2019_IDB_SPSS_G4/Data"),
        "[Aa][Ss][Gg]*.sav", "IDCNTRY", "TOTWGT",
        [f"ASSSCI0{i}" for i in range(1, 6)], "", None, "JRR_ZONES"),
    MultiFileCycleSpec("TIMSS_G4", 2023, "science",
        str(ILSA_BASE / "TIMSS Datasets/TIMSS Data/TIMSS2023_IDB_SPSS_G4/2_Data Files/SPSS Data"),
        "[Aa][Ss][Gg]*.sav", "IDCNTRY", "TOTWGT",
        [f"ASSSCI0{i}" for i in range(1, 6)], "", None, "JRR_ZONES"),

    # ── PIRLS  (aynı IEA Zone-Jackknife yapısı) ───────────────────────────
    MultiFileCycleSpec("PIRLS", 2001, "reading",
        str(ILSA_BASE / "PIRLS Datasets/PIRLS2001_IDB_SPSS/Data"),
        "[Aa][Ss][Gg]*.sav",
        "IDCNTRY", "TOTWGT",
        [f"ASRREA0{i}" for i in range(1, 6)],
        "", None, "JRR_ZONES"),

    MultiFileCycleSpec("PIRLS", 2006, "reading",
        str(ILSA_BASE / "PIRLS Datasets/PIRLS2006_IDB_SPSS/Data"),
        "[Aa][Ss][Gg]*.sav",
        "IDCNTRY", "TOTWGT",
        [f"ASRREA0{i}" for i in range(1, 6)],
        "", None, "JRR_ZONES"),

    MultiFileCycleSpec("PIRLS", 2011, "reading",
        str(ILSA_BASE / "PIRLS Datasets/PIRLS2011_IDB_SPSS/Data"),
        "[Aa][Ss][Gg]*.sav",
        "IDCNTRY", "TOTWGT",
        [f"ASRREA0{i}" for i in range(1, 6)],
        "", None, "JRR_ZONES"),

    MultiFileCycleSpec("PIRLS", 2016, "reading",
        str(ILSA_BASE / "PIRLS Datasets/PIRLS2016_IDB_SPSS/Data"),
        "[Aa][Ss][Gg]*.sav",
        "IDCNTRY", "TOTWGT",
        [f"ASRREA0{i}" for i in range(1, 6)],
        "", None, "JRR_ZONES"),

    MultiFileCycleSpec("PIRLS", 2021, "reading",
        str(ILSA_BASE / "PIRLS Datasets/PIRLS2021_IDB_SPSS/3_International Database/1_SPSS Data"),
        "[Aa][Ss][Gg]*.sav",
        "IDCNTRY", "TOTWGT",
        [f"ASRREA0{i}" for i in range(1, 6)],
        "", None, "JRR_ZONES"),
]

# ---------------------------------------------------------------------------
# Çekirdek istatistik fonksiyonları
# ---------------------------------------------------------------------------

def weighted_mean(y: np.ndarray, w: np.ndarray) -> float:
    w = w / w.sum()
    return float(np.sum(w * y))


def _variance_brr_fay(
    df: pd.DataFrame, pv: str, weight: str,
    rep_cols: list[str], theta0: float,
) -> float:
    x = df[pv].values
    sq = sum(
        (weighted_mean(x, df[r].values) - theta0) ** 2
        for r in rep_cols
    )
    return sq / BRR_FAY_DENOM


def _variance_jrr(
    df: pd.DataFrame, pv: str,
    rep_cols: list[str], theta0: float,
) -> float:
    x = df[pv].values
    return sum(
        (weighted_mean(x, df[r].values) - theta0) ** 2
        for r in rep_cols
    )


def _variance_brr(
    df: pd.DataFrame, pv: str,
    rep_cols: list[str], theta0: float,
) -> float:
    x = df[pv].values
    sq = sum(
        (weighted_mean(x, df[r].values) - theta0) ** 2
        for r in rep_cols
    )
    return sq / len(rep_cols)


def _variance_jrr_zones(
    df: pd.DataFrame, pv: str, weight: str, theta0: float,
) -> float:
    """IEA Zone-Jackknife varyansı.

    TIMSS/PIRLS veri setlerinde JKZONE + JKREP + TOTWGT sütunları kullanılır.
    Her zone için iki pseudo-replicate oluşturulur:
      - JKREP=1 grubuna ait gözlemlerin ağırlığı 2× yapılır, diğerleri 0
      - JKREP=0 grubuna ait gözlemlerin ağırlığı 2× yapılır, diğerleri 0
    V = Σ_z (θ̂_{z,1} - θ̂_{z,0})²  (IEA TDB formülü)
    """
    if "JKZONE" not in df.columns or "JKREP" not in df.columns:
        return 0.0

    x   = df[pv].values
    w0  = df[weight].values
    zones = df["JKZONE"].values
    reps  = df["JKREP"].values

    sq = 0.0
    for z in np.unique(zones):
        mask_z = zones == z
        for r_val in (0, 1):
            w_rep = w0.copy().astype(float)
            w_rep[mask_z & (reps == r_val)]    *= 2.0
            w_rep[mask_z & (reps != r_val)]     = 0.0
            if w_rep.sum() > 0:
                th_r = weighted_mean(x, w_rep)
                sq  += (th_r - theta0) ** 2

    return sq / 2.0  # IEA formülünde zone başına tek fark terimi


def variance_for_method(method: str, df, pv, weight, rep_cols, theta0) -> float:
    if method == "BRR_FAY":
        return _variance_brr_fay(df, pv, weight, rep_cols, theta0)
    if method == "JRR":
        return _variance_jrr(df, pv, rep_cols, theta0)
    if method == "JRR_ZONES":
        return _variance_jrr_zones(df, pv, weight, theta0)
    if method == "BRR":
        return _variance_brr(df, pv, rep_cols, theta0)
    raise ValueError(f"Bilinmeyen method: {method}")


def rubin_pool(
    df: pd.DataFrame,
    pv_vars: list[str],
    weight: str,
    rep_cols: list[str],
    method: str,
) -> dict:
    """Rubin (1987): T = U_bar + (1 + 1/m) * B"""
    m = len(pv_vars)
    w0 = df[weight].values
    thetas, Us = [], []

    for pv in pv_vars:
        y = df[pv].values
        th = weighted_mean(y, w0)
        U  = variance_for_method(method, df, pv, weight, rep_cols, th)
        thetas.append(th)
        Us.append(U)

    theta_bar = float(np.mean(thetas))
    U_bar     = float(np.nanmean(Us))
    B         = float(np.var(thetas, ddof=1)) if m > 1 else 0.0
    T         = U_bar + (1 + 1 / m) * B
    SE        = float(np.sqrt(max(T, 0)))

    return {
        "mean":   round(theta_bar, 4),
        "se":     round(SE, 4),
        "ci_lo":  round(theta_bar - 1.96 * SE, 2),
        "ci_hi":  round(theta_bar + 1.96 * SE, 2),
        "T":      T, "U_bar": U_bar, "B": B,
        "m_pv":   m,
    }


def single_pv_estimate(
    df: pd.DataFrame,
    outcome_var: str,
    weight: str,
    rep_cols: list[str],
    method: str,
) -> dict:
    """PV olmayan programlar için (TALIS gibi)."""
    y = df[outcome_var].values
    w = df[weight].values
    th = weighted_mean(y, w)
    U  = variance_for_method(method, df, outcome_var, weight, rep_cols, th)
    SE = float(np.sqrt(max(U, 0)))
    return {
        "mean":  round(th, 4),
        "se":    round(SE, 4),
        "ci_lo": round(th - 1.96 * SE, 2),
        "ci_hi": round(th + 1.96 * SE, 2),
        "m_pv":  0,
    }

# ---------------------------------------------------------------------------
# Veri yükleyici
# ---------------------------------------------------------------------------

def load_sav(path: str, needed_cols: list[str]) -> pd.DataFrame:
    # PISA SAV'larında CNT string ISO3 kodu içeriyor; value_formats=False yeterli.
    # cp1252 önce: latin-1 bazı dosyalarda başarısız oluyor.
    for enc in ("utf-8", "cp1252", "latin-1", "iso-8859-1"):
        try:
            df, _ = pyreadstat.read_sav(
                path, usecols=needed_cols,
                apply_value_formats=False, encoding=enc,
            )
            return df
        except Exception:
            continue
    raise RuntimeError(f"SAV okunamadı: {path}")


def get_rep_cols(df: pd.DataFrame, prefix: str, rep_range: tuple | None) -> list[str]:
    if rep_range:
        lo, hi = rep_range
        cols = [f"{prefix}{i:03d}" for i in range(lo, hi + 1)]
        # Sadece gerçekten var olanları döndür
        present = [c for c in cols if c in df.columns]
        if present:
            return present
    # Prefix ile başlayan tüm sütunlar
    return sorted(c for c in df.columns if c.startswith(prefix))

# ---------------------------------------------------------------------------
# Tek cycle × tek ülke tahmini
# ---------------------------------------------------------------------------

def estimate_country_generic(
    df_country: pd.DataFrame,
    country_var: str,
    weight_var: str,
    pv_vars: list[str],
    method: str,
    rep_cols: list[str],
) -> dict:
    n_total = len(df_country)
    needed  = (pv_vars or [weight_var]) + [weight_var]
    df_a    = df_country.dropna(subset=needed)
    df_a    = df_a[df_a[weight_var] > 0]
    n_analytic = len(df_a)

    if n_analytic < 30:
        return None

    if pv_vars:
        stats = rubin_pool(df_a, pv_vars, weight_var, rep_cols, method)
    else:
        stats = single_pv_estimate(df_a, pv_vars[0], weight_var, rep_cols, method)

    return {**stats, "n_total": n_total, "n_analytic": n_analytic,
            "n_replicates": len(rep_cols)}


def estimate_country(
    df_country: pd.DataFrame,
    spec: CycleSpec,
    rep_cols: list[str],
) -> dict:
    return estimate_country_generic(
        df_country, spec.country_var, spec.weight_var,
        list(spec.pv_vars), spec.method, rep_cols,
    )

# ---------------------------------------------------------------------------
# Bir cycle'ı işle → tüm ülkeler
# ---------------------------------------------------------------------------

def process_cycle_multi(spec: MultiFileCycleSpec) -> list[dict]:
    """TIMSS/PIRLS gibi ülke başına ayrı SAV dosyası olan programlar için.

    Tüm ülke dosyalarını tek tek okur, replicate sütunlarını normalize eder
    ve birleştirilmiş DataFrame üzerinden ülke tahminleri hesaplar.
    """
    import glob as _glob

    data_dir = Path(spec.data_dir)
    if not data_dir.exists():
        log.warning("Dizin bulunamadı, atlanıyor: %s", data_dir)
        return []

    files = sorted(data_dir.glob(spec.file_glob))
    if not files:
        log.warning("Eşleşen dosya yok: %s / %s", data_dir, spec.file_glob)
        return []

    log.info("→ %s %d  (%d ülke dosyası)", spec.program, spec.cycle, len(files))

    frames = []
    country_from_file: dict[int, str] = {}  # frame index → ISO3 (dosya adından)
    for fpath in files:
        try:
            df_c, _ = pyreadstat.read_sav(
                str(fpath), apply_value_formats=False,
            )
            # Ülke kodunu dosya adından çıkar: BSGQATm6.sav → QAT
            iso3 = fpath.stem[3:6].upper()
            df_c["_country_iso3_fn"] = iso3
            frames.append(df_c)
        except Exception as exc:
            log.warning("   Okunamadı %s: %s", fpath.name, exc)
            continue

    if not frames:
        log.error("   Hiçbir dosya okunamadı: %s %d", spec.program, spec.cycle)
        return []

    df = pd.concat(frames, ignore_index=True, sort=False)
    # Sütun adlarını büyük harfe normalize et (1995 gibi eski dosyalar lowercase kullanır)
    df.columns = [c.upper() if c != "_country_iso3_fn" else c for c in df.columns]
    # Duplicate sütun adlarını kaldır (aynı adlı ilkini tut)
    df = df.loc[:, ~df.columns.duplicated()]
    log.info("   Birleştirilen satır sayısı: %d", len(df))

    # Replicate weight sütunlarını bul (JRR_ZONES için boş olabilir)
    rep_cols = get_rep_cols(df, spec.rep_prefix, spec.rep_range)
    if not rep_cols and spec.method not in ("JRR_ZONES",):
        log.error("   Replicate weight bulunamadı prefix=%s", spec.rep_prefix)
        return []

    # JRR_ZONES için JKZONE + JKREP sütunlarını dahil et
    jk_cols = [c for c in ("JKZONE", "JKREP") if c in df.columns] \
              if spec.method == "JRR_ZONES" else []

    # Sadece ihtiyaç duyulan sütunları tut
    seen: set[str] = set()
    keep_vars: list[str] = []
    for c in (["_country_iso3_fn", spec.weight_var]
              + list(spec.pv_vars) + rep_cols + jk_cols):
        if c not in seen:
            seen.add(c)
            keep_vars.append(c)
    keep = [c for c in keep_vars if c in df.columns]
    df   = df[keep].copy()

    # Eksik PV kontrolü
    pv_available = [p for p in spec.pv_vars if p in df.columns]
    if not pv_available:
        log.error("   PV değişkenleri bulunamadı: %s", spec.pv_vars)
        return []

    countries = df["_country_iso3_fn"].unique()
    rows = []
    for cnt in sorted(countries):
        sub = df[df["_country_iso3_fn"] == cnt]
        est = estimate_country_generic(
            sub, "_country_iso3_fn", spec.weight_var,
            pv_available, spec.method, rep_cols,
        )
        if est is None:
            continue
        rows.append({
            "program":      spec.program,
            "cycle":        spec.cycle,
            "domain":       spec.domain,
            "country_iso3": cnt,
            "method":       spec.method,
            **est,
        })

    log.info("   %d ülke tamamlandı", len(rows))
    return rows


def process_cycle(spec: CycleSpec) -> list[dict]:
    path = Path(spec.sav_path)
    if not path.exists():
        log.warning("Dosya bulunamadı, atlanıyor: %s", path.name)
        return []

    log.info("→ %s %d  (%s)", spec.program, spec.cycle, path.name)

    all_vars = [spec.country_var, spec.weight_var] + list(spec.pv_vars)
    # Replicate ön ek tahminini ekle — yoksa sonra filtreleyeceğiz
    df = load_sav(str(path), None)   # tüm sütunlar, sonra filtre
    rep_cols = get_rep_cols(df, spec.rep_prefix, spec.rep_range)

    if not rep_cols:
        log.error("Replicate weight bulunamadı prefix=%s", spec.rep_prefix)
        return []

    log.info("   Replicate: %d  |  PV: %d", len(rep_cols), len(spec.pv_vars))

    # Sadece ihtiyaç duyulan sütunları tut
    keep = [spec.country_var, spec.weight_var] + list(spec.pv_vars) + rep_cols
    df   = df[[c for c in keep if c in df.columns]].copy()

    # Ülke sütununu string yap
    df[spec.country_var] = df[spec.country_var].astype(str).str.strip()
    countries = df[spec.country_var].unique()

    rows = []
    for cnt in sorted(countries):
        sub = df[df[spec.country_var] == cnt]
        est = estimate_country(sub, spec, rep_cols)
        if est is None:
            continue
        rows.append({
            "program":     spec.program,
            "cycle":       spec.cycle,
            "domain":      spec.domain,
            "country_iso3": cnt,
            "method":      spec.method,
            **est,
        })

    log.info("   %d ülke tamamlandı", len(rows))
    return rows

# ---------------------------------------------------------------------------
# Ana fonksiyon
# ---------------------------------------------------------------------------

def build_estimates(programs: list[str] | None = None) -> pd.DataFrame:
    all_rows = []
    for spec in CATALOG:
        if programs and spec.program not in programs:
            continue
        rows = process_cycle(spec)
        all_rows.extend(rows)
    for spec in MULTI_CATALOG:
        if programs and spec.program not in programs:
            continue
        rows = process_cycle_multi(spec)
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    for col in ("mean", "se", "ci_lo", "ci_hi"):
        if col in df.columns:
            df[col] = df[col].round(4)
    out = OUTPUT_DIR / "country_estimates.csv"
    df.to_csv(out, index=False)
    log.info("Kaydedildi: %s  (%d satır)", out, len(df))
    return df


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--programs", nargs="*",
                   help="Sadece belirtilen programları çalıştır (örn. PISA TIMSS)")
    args = p.parse_args()
    build_estimates(programs=args.programs)
