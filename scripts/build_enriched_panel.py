#!/usr/bin/env python3
"""
Stage 5 - Modul 4: Zenginlestirilmis Panel Olusturma

country_estimates.csv + covariate_estimates.csv → enriched_panel.csv

Her (country_iso3, cycle) için:
  - Başarı puanları (mevcut lag-1 özellikler)
  - Kovaryatlar: ESCS, HOMEPOS, BELONG, ICT, BSDGEDUP, ASDHEDUP
    → lag-1 olarak (bir önceki döngüdeki kovaryat değeri)

Döngüler arası eşleştirme:
  PISA  : 2015 → 2022 → 2025  (3-7 yıl aralık)
  TIMSS : 2003 → 2007 → 2011 → 2015 → 2019  (4 yıl aralık)
  PIRLS : 2011 → 2016 → 2021  (5 yıl aralık)

Çıktı: outputs/stage5/enriched_panel.csv
"""
from __future__ import annotations

import logging
import pathlib

import numpy as np
import pandas as pd

PROJECT_ROOT  = pathlib.Path(__file__).resolve().parents[1]
ESTIMATES_CSV = PROJECT_ROOT / "outputs" / "stage4" / "country_estimates.csv"
COVARIATES_CSV = PROJECT_ROOT / "outputs" / "stage5" / "covariate_estimates.csv"
OUT_DIR       = PROJECT_ROOT / "outputs" / "stage5"
OUT_CSV       = OUT_DIR / "enriched_panel.csv"
OUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# PISA ESCS 2006-referans döngüye göre normalize değil — ham ortalama kullanılır
# TIMSS BSDGEDUP: 1=üniversite üstü ... 6=ilkokul altı (ters skala) — negatif yön normal
# PIRLS ASDHEDUP: benzer ters skala

COVARIATE_VARS = {
    "PISA":    ["ESCS", "HOMEPOS", "BELONG", "ICTAVHOM", "ICTAVSCH"],
    "TIMSS":   ["BSDGEDUP"],
    "TIMSS_G4":["ASDHEDUP"],
    "PIRLS":   ["ASDHEDUP", "ASDHELA"],
}


def pivot_covariates(cov_df: pd.DataFrame, program: str) -> pd.DataFrame:
    """Kovaryat uzun tabloyu geniş forma çevir: (country, cycle) × variable"""
    vars_wanted = COVARIATE_VARS.get(program, [])
    sub = cov_df[
        (cov_df["program"] == program) &
        (cov_df["variable"].isin(vars_wanted))
    ].copy()
    if sub.empty:
        return pd.DataFrame()

    pivot = sub.pivot_table(
        index=["country_iso3", "cycle"],
        columns="variable",
        values="mean",
        aggfunc="first",
    ).reset_index()
    pivot.columns.name = None
    return pivot


def add_lag_covariates(panel: pd.DataFrame, cov_wide: pd.DataFrame,
                       cov_cols: list[str]) -> pd.DataFrame:
    """Her (country, cycle) için bir önceki döngüdeki kovaryatı ekle"""
    if cov_wide.empty or not cov_cols:
        return panel

    cycles = sorted(cov_wide["cycle"].unique())
    cycle_prev = {c: cycles[i-1] for i, c in enumerate(cycles) if i > 0}

    merged = panel.copy()
    for col in cov_cols:
        if col not in cov_wide.columns:
            continue
        lag_map = {}
        for country, grp in cov_wide.groupby("country_iso3"):
            grp_sorted = grp.sort_values("cycle")
            for _, row in grp_sorted.iterrows():
                prev = cycle_prev.get(row["cycle"])
                if prev is not None:
                    prev_val = grp_sorted.loc[grp_sorted["cycle"] == prev, col]
                    if not prev_val.empty and np.isfinite(prev_val.values[0]):
                        lag_map[(country, row["cycle"])] = prev_val.values[0]

        lag_col = f"lag_{col}"
        merged[lag_col] = merged.apply(
            lambda r: lag_map.get((r["country_iso3"], r["cycle"]), np.nan), axis=1
        )
    return merged


def main():
    est = pd.read_csv(ESTIMATES_CSV)
    cov = pd.read_csv(COVARIATES_CSV)

    est["cycle"]       = est["cycle"].astype(int)
    est["program"]     = est["program"].str.upper()
    est["domain"]      = est["domain"].str.lower()
    est["country_iso3"] = est["country_iso3"].str.strip()
    cov["cycle"]       = cov["cycle"].astype(int)
    cov["program"]     = cov["program"].str.upper()
    cov["country_iso3"] = cov["country_iso3"].str.strip()

    # Benchmark/alt-ulusal katılımcıları filtrele (sayı içeren veya 3 harf olmayan ISO3)
    valid_mask = est["country_iso3"].apply(lambda x: str(x).isalpha() and len(str(x)) == 3)
    n_bench = (~valid_mask).sum()
    if n_bench:
        bench = est.loc[~valid_mask, "country_iso3"].unique().tolist()
        log.info("Benchmark katılımcılar filtrelendi (%d satır): %s", n_bench, sorted(bench))
    est = est[valid_mask].copy()

    programs = est["program"].unique()
    log.info("Programlar: %s", list(programs))

    all_panels = []
    for prog in programs:
        prog_est = est[est["program"] == prog].copy()
        cov_prog = prog  # her program kendi kovaryatını kullanır
        cov_wide = pivot_covariates(cov, cov_prog)
        cov_cols = [c for c in COVARIATE_VARS.get(cov_prog, []) if c in (cov_wide.columns if not cov_wide.empty else [])]

        # Her domain için panel oluştur
        for domain in prog_est["domain"].unique():
            sub = prog_est[prog_est["domain"] == domain][["country_iso3", "cycle", "mean"]].copy()
            sub = sub.rename(columns={"mean": f"{prog}_{domain}"})
            # Kovaryatları ekle
            if not cov_wide.empty and cov_cols:
                sub = add_lag_covariates(sub, cov_wide, cov_cols)
            sub["program"] = prog
            sub["domain"]  = domain
            all_panels.append(sub)

    if not all_panels:
        log.error("Panel oluşturulamadı")
        return

    combined = pd.concat(all_panels, ignore_index=True)
    combined.to_csv(OUT_CSV, index=False)

    log.info("Kaydedildi: %s (%d satır)", OUT_CSV, len(combined))
    # Kovaryat doluluk oranı
    lag_cols = [c for c in combined.columns if c.startswith("lag_")]
    if lag_cols:
        log.info("Lag kovaryat sütunları: %s", lag_cols)
        for col in lag_cols:
            pct = combined[col].notna().mean() * 100
            log.info("  %s: %.1f%% dolu", col, pct)


if __name__ == "__main__":
    main()
