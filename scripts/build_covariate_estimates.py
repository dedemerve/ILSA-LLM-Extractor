#!/usr/bin/env python3
"""
Stage 5 — Modül 3: Kovaryat Ülke Ortalamaları

PISA, TIMSS ve PIRLS mikroverisinden arka plan değişkenlerinin
ülke bazlı ağırlıklı ortalamalarını hesaplar.

Kullanılan değişkenler:
  PISA:
    ESCS         — SES kompoziti (6 döngüde tutarlı, 2022 referans)
    HOMEPOS      — Ev kaynakları indeksi
    BELONG       — Okula aidiyet
    ICTAVHOM     — Evde ICT erişimi
    ICTAVSCH     — Okulda ICT erişimi

  TIMSS G8:
    BSDGEDUP     — Ebeveyn eğitim düzeyi (G8)
    BSDGHRL      — Ev dili (ev kitabı sayısı proxy)

  PIRLS:
    ASDHEDUP     — Ebeveyn eğitim düzeyi (G4)

Çıktı: outputs/stage5/covariate_estimates.csv
  Sütunlar: program, cycle, country_iso3, variable, mean, se, n_total
"""
from __future__ import annotations

import logging
import pathlib
import warnings
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd
import pyreadstat

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
ILSA_BASE    = pathlib.Path("/Users/mrved/Desktop/ILSA Datasets")
OUT_DIR      = PROJECT_ROOT / "outputs" / "stage5"
OUT_CSV      = OUT_DIR / "covariate_estimates.csv"
OUT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class CovSpec:
    """Tek bir kovaryat çıkarım spesifikasyonu"""
    program:      str
    cycle:        int
    variables:    List[str]          # İstenen sütun adları
    filepath:     Optional[str]      # Tek dosya (PISA)
    glob_dir:     Optional[str]      # Çoklu dosya dizini (TIMSS/PIRLS)
    glob_prefix:  str = "BSG"        # Dosya ön eki
    country_col:  str = "CNT"
    weight_col:   str = "W_FSTUWT"
    encoding_hint: str = "utf-8"

    def file_paths(self) -> list[pathlib.Path]:
        if self.filepath:
            p = pathlib.Path(self.filepath)
            return [p] if p.exists() else []
        if self.glob_dir:
            import glob as _glob
            d = pathlib.Path(self.glob_dir)
            # Tüm .sav dosyaları bul, prefix'e göre filtrele (case-insensitive)
            all_sav = _glob.glob(str(d / "**" / "*.sav"), recursive=True)
            prefix_up = self.glob_prefix.upper()
            return sorted(
                pathlib.Path(p) for p in all_sav
                if pathlib.Path(p).name.upper().startswith(prefix_up)
            )
        return []


PISA_BASE  = ILSA_BASE / "PISA Datasets"
TIMSS_BASE = ILSA_BASE / "TIMSS Datasets" / "TIMSS Data"
PIRLS_BASE = ILSA_BASE / "PIRLS Datasets"

SPECS: list[CovSpec] = [
    # PISA — tek dosya, BRR ağırlıkları
    CovSpec("PISA", 2015,
        variables=["ESCS", "HOMEPOS", "BELONG", "ICTAVHOM", "ICTAVSCH"],
        filepath=str(PISA_BASE / "PISA 2015 Data/PUF_SPSS_COMBINED_CMB_STU_QQQ/CY6_MS_CMB_STU_QQQ.sav"),
        glob_dir=None, country_col="CNT", weight_col="W_FSTUWT",
    ),
    CovSpec("PISA", 2022,
        variables=["ESCS", "HOMEPOS", "BELONG", "ICTAVHOM", "ICTAVSCH"],
        filepath=str(PISA_BASE / "PISA 2022 Data/School questionnaire data file (CY08MSP_STU_QQQ).SAV"),
        glob_dir=None, country_col="CNT", weight_col="W_FSTUWT",
    ),
    CovSpec("PISA", 2025,
        variables=["ESCS", "HOMEPOS", "BELONG", "ICTAVHOM", "ICTAVSCH"],
        filepath=str(PISA_BASE / "PISA 2025 Data/CY09_MS_STU_PUF.sav"),
        glob_dir=None, country_col="CNT", weight_col="W_FSTUWT",
        encoding_hint="iso-8859-1",
    ),
    # TIMSS G8 — çoklu dosya
    # 2023: country_col="CTY" (ISO3 doğrudan), önceki döngüler "_COUNTRY_ISO3_FN"
    CovSpec("TIMSS", 2023,
        variables=["BSDGEDUP"],          # ITSEX kaldırıldı (MUST FIX #2)
        filepath=None,
        glob_dir=str(TIMSS_BASE / "TIMSS2023_IDB_SPSS_G8"),
        glob_prefix="bsg", country_col="CTY", weight_col="TOTWGT",
    ),
    CovSpec("TIMSS", 2019,
        variables=["ITSEX", "BSDGEDUP", "BSDG07", "BSDG08"],
        filepath=None,
        glob_dir=str(TIMSS_BASE / "TIMSS2019_IDB_SPSS_G8"),
        glob_prefix="BSG", country_col="_COUNTRY_ISO3_FN", weight_col="TOTWGT",
    ),
    CovSpec("TIMSS", 2015,
        variables=["ITSEX", "BSDGEDUP", "BSDG07", "BSDG08"],
        filepath=None,
        glob_dir=str(TIMSS_BASE / "TIMSS2015_IDB_SPSS_G8"),
        glob_prefix="BSG", country_col="_COUNTRY_ISO3_FN", weight_col="TOTWGT",
    ),
    CovSpec("TIMSS", 2011,
        variables=["ITSEX", "BSDGEDUP", "BSDG07", "BSDG08"],
        filepath=None,
        glob_dir=str(TIMSS_BASE / "TIMSS2011_IDB_SPSS_G8"),
        glob_prefix="BSG", country_col="_COUNTRY_ISO3_FN", weight_col="TOTWGT",
    ),
    CovSpec("TIMSS", 2007,
        variables=["ITSEX", "BSDGEDUP", "BSDG07", "BSDG08"],
        filepath=None,
        glob_dir=str(TIMSS_BASE / "TIMSS2007_IDB_SPSS_G8"),
        glob_prefix="BSG", country_col="_COUNTRY_ISO3_FN", weight_col="TOTWGT",
    ),
    CovSpec("TIMSS", 2003,
        variables=["ITSEX", "BSDGEDUP", "BSDG07", "BSDG08"],
        filepath=None,
        glob_dir=str(TIMSS_BASE / "TIMSS2003_IDB_SPSS_G8"),
        glob_prefix="BSG", country_col="_COUNTRY_ISO3_FN", weight_col="TOTWGT",
    ),
    # TIMSS G4 — ev anketi (ASH) ebeveyn eğitimi
    # 2023: CTY (ISO3 doğrudan); öncekiler: dosya adından çıkarılıyor (stem[3:6])
    CovSpec("TIMSS_G4", 2023,
        variables=["ASDHEDUP"],
        filepath=None,
        glob_dir=str(TIMSS_BASE / "TIMSS2023_IDB_SPSS_G4"),
        glob_prefix="ASH", country_col="CTY", weight_col="TOTWGT",
    ),
    CovSpec("TIMSS_G4", 2019,
        variables=["ASDHEDUP"],
        filepath=None,
        glob_dir=str(TIMSS_BASE / "TIMSS2019_IDB_SPSS_G4"),
        glob_prefix="ASH", country_col="_COUNTRY_ISO3_FN", weight_col="TOTWGT",
    ),
    CovSpec("TIMSS_G4", 2015,
        variables=["ASDHEDUP"],
        filepath=None,
        glob_dir=str(TIMSS_BASE / "TIMSS2015_IDB_SPSS_G4"),
        glob_prefix="ASH", country_col="_COUNTRY_ISO3_FN", weight_col="TOTWGT",
    ),
    CovSpec("TIMSS_G4", 2011,
        variables=["ASDHEDUP"],
        filepath=None,
        glob_dir=str(TIMSS_BASE / "TIMSS2011_IDB_SPSS_G4"),
        glob_prefix="ASH", country_col="_COUNTRY_ISO3_FN", weight_col="TOTWGT",
    ),
    # PIRLS G4 — ev anketi (ASH) ebeveyn eğitimi için
    CovSpec("PIRLS", 2021,
        variables=["ASDHEDUP", "ASDHELA", "ASDHELB"],
        filepath=None,
        glob_dir=str(PIRLS_BASE / "PIRLS2021_IDB_SPSS"),
        glob_prefix="ASH", country_col="_COUNTRY_ISO3_FN", weight_col="TOTWGT",
    ),
    CovSpec("PIRLS", 2016,
        variables=["ASDHEDUP", "ASDHELA", "ASDHELB"],
        filepath=None,
        glob_dir=str(PIRLS_BASE / "PIRLS2016_IDB_SPSS"),
        glob_prefix="ASH", country_col="_COUNTRY_ISO3_FN", weight_col="TOTWGT",
    ),
    CovSpec("PIRLS", 2011,
        variables=["ASDHEDUP", "ASDHELA", "ASDHELB"],
        filepath=None,
        glob_dir=str(PIRLS_BASE / "PIRLS2011_IDB_SPSS"),
        glob_prefix="ASH", country_col="_COUNTRY_ISO3_FN", weight_col="TOTWGT",
    ),
]


def load_sav(fpath: pathlib.Path, usecols: list[str], encoding_hint: str = "utf-8") -> pd.DataFrame:
    for enc in (encoding_hint, "utf-8", "cp1252", "latin-1", "iso-8859-1"):
        try:
            df, _ = pyreadstat.read_sav(
                str(fpath), usecols=usecols,
                encoding=enc, apply_value_formats=False
            )
            return df
        except Exception:
            continue
    raise RuntimeError(f"Hiçbir encoding çalışmadı: {fpath.name}")


def weighted_mean_se(values: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    """Basit ağırlıklı ortalama ve SE (Taylor linearization yaklaşımı)"""
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    v, w = values[mask], weights[mask]
    if len(v) == 0:
        return np.nan, np.nan
    w_sum = w.sum()
    mu = np.dot(w, v) / w_sum
    # SE: ağırlıklı standart hata
    se = np.sqrt(np.dot(w, (v - mu) ** 2) / (w_sum ** 2) * len(v))
    return float(mu), float(se)


def process_pisa_spec(spec: CovSpec) -> list[dict]:
    """Tek dosyalı PISA spesifikasyonu"""
    fps = spec.file_paths()
    if not fps:
        log.warning("PISA %d dosyası bulunamadı", spec.cycle)
        return []

    fpath = fps[0]
    avail_cols = []
    try:
        _, meta = pyreadstat.read_sav(str(fpath), metadataonly=True)
        cols_upper = {c.upper(): c for c in meta.column_names}
        avail_cols = [cols_upper[v] for v in spec.variables if v.upper() in cols_upper]
        cnt_col = cols_upper.get(spec.country_col.upper(), spec.country_col)
        wgt_col = cols_upper.get(spec.weight_col.upper(), spec.weight_col)
        load_cols = list(set(avail_cols + [cnt_col, wgt_col]))
    except Exception as e:
        log.error("Meta okuma hatası %s: %s", fpath.name, e)
        return []

    if not avail_cols:
        log.warning("PISA %d: hiç kovaryat bulunamadı", spec.cycle)
        return []

    log.info("PISA %d: %s", spec.cycle, fpath.name)
    try:
        df = load_sav(fpath, load_cols, spec.encoding_hint)
    except Exception as e:
        log.error("Yüklenemedi: %s", e)
        return []

    df.columns = [c.upper() for c in df.columns]
    cnt_col_u = spec.country_col.upper()
    wgt_col_u = spec.weight_col.upper()

    records = []
    for country, grp in df.groupby(cnt_col_u):
        weights = grp[wgt_col_u].values.astype(float)
        for var in avail_cols:
            var_u = var.upper()
            if var_u not in grp.columns:
                continue
            vals = grp[var_u].values.astype(float)
            mu, se = weighted_mean_se(vals, weights)
            if np.isfinite(mu):
                records.append({
                    "program": spec.program,
                    "cycle": spec.cycle,
                    "country_iso3": str(country).strip(),
                    "variable": var_u,
                    "mean": round(mu, 4),
                    "se": round(se, 4),
                    "n_total": int((weights > 0).sum()),
                })
    return records


def process_multi_spec(spec: CovSpec) -> list[dict]:
    """TIMSS/PIRLS: her ülke ayrı dosya"""
    fps = spec.file_paths()
    if not fps:
        log.warning("%s %d: dosya bulunamadı (prefix=%s)", spec.program, spec.cycle, spec.glob_prefix)
        return []

    log.info("%s %d: %d dosya", spec.program, spec.cycle, len(fps))
    records = []
    for fpath in fps:
        iso3 = fpath.stem[3:6].upper()
        try:
            _, meta = pyreadstat.read_sav(str(fpath), metadataonly=True)
            cols_upper = {c.upper(): c for c in meta.column_names}
            avail = [v for v in spec.variables if v.upper() in cols_upper]
            if not avail:
                continue
            wgt_col_a = cols_upper.get(spec.weight_col.upper())
            if wgt_col_a:
                # Normal ağırlıklı ortalama (TIMSS öğrenci dosyası)
                load_cols = [cols_upper[v] for v in avail] + [wgt_col_a]
                df = load_sav(fpath, load_cols)
                df.columns = [c.upper() for c in df.columns]
                weights = df[spec.weight_col.upper()].values.astype(float)
            else:
                # Ağırlık yok (PIRLS ev anketi): eşit ağırlık kullan
                load_cols = [cols_upper[v] for v in avail]
                df = load_sav(fpath, load_cols)
                df.columns = [c.upper() for c in df.columns]
                weights = np.ones(len(df))

            for var in avail:
                vals = df[var.upper()].values.astype(float)
                mu, se = weighted_mean_se(vals, weights)
                if np.isfinite(mu):
                    records.append({
                        "program": spec.program,
                        "cycle": spec.cycle,
                        "country_iso3": iso3,
                        "variable": var.upper(),
                        "mean": round(mu, 4),
                        "se": round(se, 4),
                        "n_total": int((weights > 0).sum()),
                    })
        except Exception as e:
            log.debug("  %s atlandı: %s", fpath.name, e)
            continue
    return records


def main():
    all_records = []
    for spec in SPECS:
        if spec.filepath:
            recs = process_pisa_spec(spec)
        else:
            recs = process_multi_spec(spec)
        all_records.extend(recs)
        log.info("  → %d satır eklendi", len(recs))

    if not all_records:
        log.error("Hiç kayıt üretilemedi.")
        return

    df = pd.DataFrame(all_records)

    # PISA 2003/2009 TXT parser çıktısını birleştir (build_pisa_txt_covariates.py)
    early_csv = OUT_CSV.parent / "covariate_estimates_pisa_early.csv"
    if early_csv.exists():
        early = pd.read_csv(early_csv)
        early = early.rename(columns={"n_students": "n_total"})
        early["se"] = float("nan")
        df = pd.concat([df, early], ignore_index=True)
        log.info("PISA 2003/2009 TXT covariates eklendi: %d satır", len(early))

    df.to_csv(OUT_CSV, index=False)
    log.info("\nKaydedildi: %s (%d satır)", OUT_CSV, len(df))
    log.info("Özet:")
    log.info("\n%s", df.groupby(["program", "cycle", "variable"])["country_iso3"].count().to_string())


if __name__ == "__main__":
    main()
