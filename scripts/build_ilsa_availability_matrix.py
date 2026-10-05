#!/usr/bin/env python3
"""
7-ILSA Availability Matrix

Program × Cycle × Country × Construct düzeyinde kapsamlı availability tablosu.
Forecasting için: source_cycle → target_cycle → lag_years zinciri dahil.

Çıktılar:
  outputs/stage5/ilsa_availability_matrix.csv   — uzun format
  outputs/stage5/ilsa_cycle_country_matrix.csv  — program×cycle × ülke pivot

Kaynak veriler:
  - outputs/stage5/covariate_estimates.csv        (PISA / TIMSS / PIRLS)
  - outputs/stage5/talis_covariate_estimates.csv  (TALIS 2013/2018)
  - ICILS BSG dosyaları (arşiv taraması)
  - PIAAC CSV dosyaları (arşiv taraması)
"""
from __future__ import annotations

import glob
import logging
import pathlib

import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT_DIR      = PROJECT_ROOT / "outputs" / "stage5"
ILSA_BASE    = pathlib.Path("/Users/mrved/Desktop/ILSA Datasets")

# ── Sub-national / non-standard kodlar ───────────────────────────────────────
SUBNATIONAL = {
    # TIMSS/PIRLS benchmark participants (sayı içeren — zaten filtrelendi)
    # TALIS
    "AAD", "ABA", "BFL", "CAB", "CSH", "ENG", "INT",
    # ICILS
    "CNL", "COT", "HKG", "DNW", "RMO",
    # ICILS 2023 (tartışmalı ama küçük nüfus/özel bölge)
    # BIH, AZE, OMN geçerli ülkeler — filtrelenmez
}


def load_pisa_timss_pirls() -> pd.DataFrame:
    """Mevcut covariate_estimates.csv'den program×cycle×country×construct."""
    cov = pd.read_csv(OUT_DIR / "covariate_estimates.csv")
    # Yalnızca temel outcome'u çıkar (construct adı olarak variable kullan)
    # ITSEX kaldırıldığı için filtre gerekmiyor; PISA 2003/2009 dahil
    recs = []
    for _, row in cov.iterrows():
        recs.append({
            "program":  row["program"],
            "cycle":    int(row["cycle"]),
            "country_iso3": row["country_iso3"],
            "construct": row["variable"],
            "data_type": "covariate",
            "source_file": "covariate_estimates.csv",
        })
    return pd.DataFrame(recs)


def load_talis() -> pd.DataFrame:
    talis = pd.read_csv(OUT_DIR / "talis_covariate_estimates.csv")
    recs = []
    seen = set()
    for _, row in talis.iterrows():
        key = (row["cycle"], row["country_iso3"], row["variable"])
        if key in seen:
            continue
        seen.add(key)
        recs.append({
            "program":    "TALIS",
            "cycle":      int(row["cycle"]),
            "country_iso3": row["country_iso3"],
            "construct":  row["construct"],
            "data_type":  "covariate",
            "source_file": "talis_covariate_estimates.csv",
        })
    return pd.DataFrame(recs)


def load_icils() -> pd.DataFrame:
    """ICILS BSG dosyalarından ülke×cycle varlığını çıkar."""
    icils_base = ILSA_BASE / "ICILS Datasets"
    recs = []
    cycle_map = {
        "ICILS2013_IDB_SPSS": 2013,
        "ICILS2018_IDB_SPSS": 2018,
        "ICILS2023_IDB_SPSS": 2023,
    }
    # Cycle'a göre mevcut construct'lar
    constructs_by_cycle = {
        2013: ["CIL", "S_HISCED", "S_HISEI", "S_HOMLIT", "S_NISB", "S_BASEFF", "S_ADVEFF"],
        2018: ["CIL", "CT", "S_HISCED", "S_HISEI", "S_HOMLIT", "S_NISB",
               "S_GENEFF", "S_SPECEFF", "S_ICTPOS", "S_ICTNEG"],
        2023: ["CIL", "CT", "S_HISCED", "S_HISEI", "S_HOMLIT",
               "S_GENEFF", "S_SPECEFF", "S_ICTPOSG", "S_ICTNEG"],
    }

    for dir_name, year in cycle_map.items():
        d = icils_base / dir_name
        bsg_files = sorted(
            p for p in glob.glob(str(d / "**/*.sav"), recursive=True)
            if pathlib.Path(p).name.upper().startswith("BSG")
        )
        for fpath in bsg_files:
            iso3 = pathlib.Path(fpath).stem[3:6].upper()
            if iso3 in SUBNATIONAL:
                continue
            for construct in constructs_by_cycle.get(year, []):
                recs.append({
                    "program":    "ICILS",
                    "cycle":      year,
                    "country_iso3": iso3,
                    "construct":  construct,
                    "data_type":  "outcome" if construct in ("CIL", "CT") else "covariate",
                    "source_file": pathlib.Path(fpath).name,
                })
    return pd.DataFrame(recs)


def load_piaac() -> pd.DataFrame:
    """PIAAC CSV dosyalarından ülke×cycle varlığını çıkar."""
    piaac_base = ILSA_BASE / "PIAAC Datasets"
    recs = []
    cycle_map = {
        "PIAAC Cycle1": (2012, ["PVLIT", "PVNUM", "PVPSL", "YRSQUAL"]),
        "PIAAC Cycle2": (2022, ["PVLIT", "PVNUM", "PVAPS", "YRSQUAL"]),
    }
    for dir_name, (year, constructs) in cycle_map.items():
        d = piaac_base / dir_name
        csv_files = sorted(d.glob("*.csv")) if d.exists() else []
        seen_countries = set()
        for fpath in csv_files:
            iso3 = fpath.stem[3:6].upper()
            if iso3 in seen_countries:  # USA 2012+2017
                continue
            seen_countries.add(iso3)
            for construct in constructs:
                recs.append({
                    "program":    "PIAAC",
                    "cycle":      year,
                    "country_iso3": iso3,
                    "construct":  construct,
                    "data_type":  "outcome" if construct.startswith("PV") else "covariate",
                    "source_file": fpath.name,
                })
    return pd.DataFrame(recs)


def build_forecasting_pairs(matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Her program×ülke için source_cycle → target_cycle lag tablosu.
    Yalnızca aynı ülkenin ardışık cycle'ları arasında ve leakage olmadan.
    """
    # Outcome'a sahip (achievement) program×country×cycle kombinasyonları
    # LOCO hedef programlar: PISA, TIMSS, TIMSS_G4, PIRLS, ICILS
    # TALIS ve PIAAC: kaynak (predictor) olarak kullanılır
    target_progs = {"PISA", "TIMSS", "TIMSS_G4", "PIRLS", "ICILS"}

    # Her target program için döngü sırası
    cycle_order = {
        "PISA":      [2003, 2006, 2009, 2012, 2015, 2018, 2022, 2025],
        "TIMSS":     [1995, 1999, 2003, 2007, 2011, 2015, 2019, 2023],
        "TIMSS_G4":  [1995, 2003, 2007, 2011, 2015, 2019, 2023],
        "PIRLS":     [2001, 2006, 2011, 2016, 2021],
        "ICILS":     [2013, 2018, 2023],
        "TALIS":     [2008, 2013, 2018, 2024],
        "PIAAC":     [2012, 2022],
    }

    # Predictor programların hedef programa lag eşleştirmesi
    # TALIS → PISA/TIMSS temporal alignment (daha önce talis_covariate_estimates.csv'de zaten var)
    cross_prog_pairs = [
        # (source_prog, source_cycle, target_prog, target_cycle, lag_years)
        ("TALIS", 2013, "PISA",  2015, 2),
        ("TALIS", 2013, "TIMSS", 2015, 2),
        ("TALIS", 2018, "PISA",  2022, 4),
        ("TALIS", 2018, "TIMSS", 2019, 1),
        ("ICILS", 2013, "PISA",  2015, 2),
        ("ICILS", 2018, "PISA",  2022, 4),
        ("ICILS", 2018, "TIMSS", 2019, 1),
        ("ICILS", 2023, "PISA",  2025, 2),
        ("PIAAC", 2012, "PISA",  2015, 3),
        ("PIAAC", 2012, "TIMSS", 2015, 3),
        ("PIAAC", 2022, "PISA",  2025, 3),
    ]

    # Ülke bazında cross-program overlap hesapla
    records = []
    for src_prog, src_cycle, tgt_prog, tgt_cycle, lag_years in cross_prog_pairs:
        src_countries = set(
            matrix[
                (matrix["program"] == src_prog) &
                (matrix["cycle"] == src_cycle)
            ]["country_iso3"].unique()
        )
        tgt_countries = set(
            matrix[
                (matrix["program"] == tgt_prog) &
                (matrix["cycle"] == tgt_cycle)
            ]["country_iso3"].unique()
        )
        overlap = src_countries & tgt_countries
        records.append({
            "source_program": src_prog,
            "source_cycle":   src_cycle,
            "target_program": tgt_prog,
            "target_cycle":   tgt_cycle,
            "lag_years":      lag_years,
            "n_overlap":      len(overlap),
            "overlap_countries": "|".join(sorted(overlap)),
        })

    return pd.DataFrame(records)


def main():
    log.info("Program kovaryat verileri yükleniyor...")
    df_pisa_etc = load_pisa_timss_pirls()
    df_talis    = load_talis()
    df_icils    = load_icils()
    df_piaac    = load_piaac()

    matrix = pd.concat([df_pisa_etc, df_talis, df_icils, df_piaac], ignore_index=True)

    # Geçersiz ISO3 filtresi
    valid = matrix["country_iso3"].apply(lambda x: str(x).isalpha() and len(str(x)) == 3)
    n_dropped = (~valid).sum()
    if n_dropped:
        log.info("Geçersiz ISO3 filtrelendi: %d satır", n_dropped)
    matrix = matrix[valid].copy()

    log.info("Toplam availability matrix: %d satır", len(matrix))

    # Program × Cycle × Country sayısı
    summary = matrix.groupby(["program", "cycle"])["country_iso3"].nunique().unstack("cycle", fill_value=0)
    log.info("\nProgram × Cycle ülke sayısı:\n%s", summary.to_string())

    # Kaydet
    matrix.to_csv(OUT_DIR / "ilsa_availability_matrix.csv", index=False)
    log.info("Kaydedildi: ilsa_availability_matrix.csv")

    # Forecasting pair tablosu
    pairs = build_forecasting_pairs(matrix)
    pairs_out = OUT_DIR / "ilsa_forecasting_pairs.csv"
    pairs.to_csv(pairs_out, index=False)
    log.info("Kaydedildi: ilsa_forecasting_pairs.csv")

    print("\n=== Cross-Program Forecasting Pairs ===")
    print(pairs[["source_program","source_cycle","target_program","target_cycle",
                 "lag_years","n_overlap"]].to_string(index=False))

    # Pivot: her program×cycle için unique country sayısı
    pivot = matrix.groupby(["program","cycle"])["country_iso3"].nunique().reset_index()
    pivot.columns = ["program", "cycle", "n_countries"]
    print("\n=== Program × Cycle Ülke Sayısı ===")
    print(pivot.sort_values(["program","cycle"]).to_string(index=False))


if __name__ == "__main__":
    main()
