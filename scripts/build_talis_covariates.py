#!/usr/bin/env python3
"""
TALIS 2013 ve 2018 öğretmen kovaryatları

Her ülke için TCHWGT ile ağırlıklı ortalama üretir.
Çıktı: outputs/stage5/talis_covariate_estimates.csv

Sütunlar:
  program, cycle, country_iso3, variable, construct,
  mean, n, weight_used, source, lag_years, target_ilsa

Metodolojik notlar:
  - Composite scale değişkenleri IDB tarafından zaten ölçeklenmiş WLE/scale scores.
    Yeniden standardize edilmiyor — ham ağırlıklı ortalama kullanılıyor.
  - Yalnızca 2013 ve 2018: scale harmonization bu iki döngüde mümkün.
  - 2008: composite scale yok → forecasting feature üretilmiyor.
  - 2024: ülke başına dosya henüz yayınlanmamış → atlandı.
  - Alt-ulusal birimler (AAD, ABA, BFL, CAB, CSH, ENG, INT) filtreleniyor.
  - lag_years: TALIS cycle → en yakın sonraki PISA/TIMSS cycle arasındaki yıl farkı.
"""
from __future__ import annotations

import logging
import pathlib
import glob

import numpy as np
import pandas as pd
import pyreadstat

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
TALIS_BASE   = pathlib.Path("/Users/mrved/Desktop/ILSA Datasets/TALIS Datasets")
OUT_DIR      = PROJECT_ROOT / "outputs" / "stage5"
OUT_CSV      = OUT_DIR / "talis_covariate_estimates.csv"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Alt-ulusal / non-standard TALIS kodları ───────────────────────────────────
SUBNATIONAL = {"AAD", "ABA", "BFL", "CAB", "CSH", "ENG", "INT"}

# ── Construct harmonization: raw variable → canonical adı ────────────────────
# 2013 ve 2018 için aynı canonical ismi kullanıyoruz; karşılaştırılabilirlik sağlanmış.
CONSTRUCT_MAP = {
    # 2013 → canonical
    "SECLSS":   ("self_efficacy_cls_mgmt",     "Self-Efficacy: Classroom Management"),
    "SEINSS":   ("self_efficacy_instruction",  "Self-Efficacy: Instruction"),
    "SEENGS":   ("self_efficacy_engagement",   "Self-Efficacy: Student Engagement"),
    "TCDISCS":  ("disciplinary_climate",       "Disciplinary Climate"),
    "TCCOLLS":  ("prof_collaboration",         "Professional Collaboration"),
    "TJSENVS":  ("job_sat_environment",        "Job Satisfaction: Work Environment"),
    "TJSPROS":  ("job_sat_profession",         "Job Satisfaction: Profession"),
    "TEFFPROS": ("effective_pd",               "Effective Professional Development"),
    # 2018 → canonical (aynı adlar)
    "T3SECLS":  ("self_efficacy_cls_mgmt",     "Self-Efficacy: Classroom Management"),
    "T3SEINS":  ("self_efficacy_instruction",  "Self-Efficacy: Instruction"),
    "T3SEENG":  ("self_efficacy_engagement",   "Self-Efficacy: Student Engagement"),
    "T3DISC":   ("disciplinary_climate",       "Disciplinary Climate"),
    "T3COLES":  ("prof_collaboration",         "Professional Collaboration"),
    "T3JSENV":  ("job_sat_environment",        "Job Satisfaction: Work Environment"),
    "T3JSPRO":  ("job_sat_profession",         "Job Satisfaction: Profession"),
    "T3EFFPD":  ("effective_pd",               "Effective Professional Development"),
}

# ── Cycle spesifikasyonları ───────────────────────────────────────────────────
CYCLE_SPECS = [
    {
        "cycle":      2013,
        "data_dir":   TALIS_BASE / "TALIS 2013 Data",
        "btg_prefix": "BTG",
        "cntry_col":  "CNTRY",         # ISO3 doğrudan
        "weight_col": "TCHWGT",
        "variables":  ["SECLSS", "SEINSS", "SEENGS", "TCDISCS",
                       "TCCOLLS", "TJSENVS", "TJSPROS", "TEFFPROS"],
        # TALIS 2013 → sonraki ILSA döngüsü (yıl farkı)
        "temporal_targets": {
            "PISA":  {"next_cycle": 2015, "lag_years": 2},
            "TIMSS": {"next_cycle": 2015, "lag_years": 2},
        },
    },
    {
        "cycle":      2018,
        "data_dir":   TALIS_BASE / "TALIS 2018 Data",
        "btg_prefix": "BTG",
        "cntry_col":  "CNTRY",         # ISO3 doğrudan
        "weight_col": "TCHWGT",
        "variables":  ["T3SECLS", "T3SEINS", "T3SEENG", "T3DISC",
                       "T3COLES", "T3JSENV", "T3JSPRO", "T3EFFPD"],
        "temporal_targets": {
            "PISA":  {"next_cycle": 2022, "lag_years": 4},
            "TIMSS": {"next_cycle": 2019, "lag_years": 1},
        },
    },
]


def weighted_mean(vals: np.ndarray, wts: np.ndarray) -> tuple[float, int]:
    mask = np.isfinite(vals) & np.isfinite(wts) & (wts > 0)
    v, w = vals[mask], wts[mask]
    if len(v) == 0:
        return np.nan, 0
    return float(np.average(v, weights=w)), int(len(v))


def process_cycle(spec: dict) -> list[dict]:
    d = spec["data_dir"]
    all_sav = glob.glob(str(d / "**/*.sav"), recursive=True)
    btg_files = sorted(
        p for p in all_sav
        if pathlib.Path(p).name.upper().startswith(spec["btg_prefix"].upper())
    )
    if not btg_files:
        log.warning("TALIS %d: BTG dosyası bulunamadı", spec["cycle"])
        return []

    log.info("TALIS %d: %d BTG dosyası bulundu", spec["cycle"], len(btg_files))
    records = []

    for fpath in btg_files:
        iso3 = pathlib.Path(fpath).stem[3:6].upper()

        # Alt-ulusal filtresi
        if iso3 in SUBNATIONAL:
            log.debug("  %s atlandı (alt-ulusal)", iso3)
            continue

        # Metadata: mevcut değişkenleri kontrol et
        try:
            _, meta = pyreadstat.read_sav(fpath, metadataonly=True)
        except Exception as e:
            log.debug("  %s metadata hatası: %s", pathlib.Path(fpath).name, e)
            continue

        cols_upper = {c.upper(): c for c in meta.column_names}
        avail_vars = [v for v in spec["variables"] if v.upper() in cols_upper]
        wgt_orig   = cols_upper.get(spec["weight_col"].upper())

        if not avail_vars:
            log.debug("  %s: hiç hedef değişken yok", iso3)
            continue

        load_cols = [cols_upper[v.upper()] for v in avail_vars]
        if wgt_orig:
            load_cols.append(wgt_orig)

        try:
            df, _ = pyreadstat.read_sav(fpath, usecols=load_cols, apply_value_formats=False)
        except Exception as e:
            log.debug("  %s yüklenemedi: %s", iso3, e)
            continue

        df.columns = [c.upper() for c in df.columns]

        if wgt_orig:
            weights = df[spec["weight_col"].upper()].values.astype(float)
            weight_used = spec["weight_col"]
        else:
            weights = np.ones(len(df))
            weight_used = "equal"

        for var in avail_vars:
            canonical, construct_label = CONSTRUCT_MAP.get(var.upper(), (var.upper(), var.upper()))
            vals = df[var.upper()].values.astype(float)
            mu, n = weighted_mean(vals, weights)
            if not np.isfinite(mu) or n < 30:
                continue

            # Her temporal target için ayrı satır
            for ilsa_prog, tgt in spec["temporal_targets"].items():
                records.append({
                    "program":      "TALIS",
                    "cycle":        spec["cycle"],
                    "country_iso3": iso3,
                    "variable":     var.upper(),
                    "construct":    canonical,
                    "construct_label": construct_label,
                    "mean":         round(mu, 4),
                    "n":            n,
                    "weight_used":  weight_used,
                    "source":       pathlib.Path(fpath).name,
                    "target_program": ilsa_prog,
                    "target_cycle": tgt["next_cycle"],
                    "lag_years":    tgt["lag_years"],
                })

    return records


def main():
    all_records = []
    for spec in CYCLE_SPECS:
        recs = process_cycle(spec)
        all_records.extend(recs)
        log.info("  → %d kayıt (cycle=%d)", len(recs), spec["cycle"])

    if not all_records:
        log.error("Hiç kayıt üretilemedi.")
        return

    df = pd.DataFrame(all_records)

    # Duplicate kontrolü: (country_iso3, cycle, variable, target_program) benzersiz olmalı
    dup_key = ["country_iso3", "cycle", "variable", "target_program"]
    n_dup = df.duplicated(subset=dup_key).sum()
    if n_dup:
        log.error("DUPLICATE HATASI: %d yinelenen (country, cycle, variable, target) satırı!", n_dup)
    else:
        log.info("Duplicate kontrolü: 0 yineleme ✓")

    df.to_csv(OUT_CSV, index=False)
    log.info("Kaydedildi: %s (%d satır)", OUT_CSV, len(df))

    # Özet: cycle × construct × ülke sayısı
    log.info("\nÖzet (cycle × construct, ülke sayısı):")
    summary = (
        df[df["target_program"] == "PISA"]
        .groupby(["cycle", "construct"])["country_iso3"]
        .nunique()
        .unstack("construct", fill_value=0)
    )
    log.info("\n%s", summary.to_string())

    # Cycle availability tablosu
    print("\n── Availability: cycle × ülke (PISA target) ──")
    avail = (
        df[df["target_program"] == "PISA"]
        .groupby(["cycle", "construct"])["country_iso3"]
        .nunique()
    )
    print(avail.to_string())

    print("\n── Örnek satırlar ──")
    print(df[df["country_iso3"] == "TUR"][
        ["cycle", "country_iso3", "variable", "construct", "mean", "n", "target_program", "lag_years"]
    ].to_string(index=False))


if __name__ == "__main__":
    main()
