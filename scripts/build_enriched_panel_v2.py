#!/usr/bin/env python3
"""
enriched_panel_v2.csv — Cross-Program Feature Space

Her satır: (target_country, target_program, target_cycle,
             source_program, source_cycle, lag_years,
             canonical_construct, value, direction, level, tier)

Kaynaklar:
  1. outputs/stage5/enriched_panel.csv        → PISA/TIMSS/PIRLS intra-program lags
  2. outputs/stage5/talis_covariate_estimates.csv → TALIS cross-program features
  3. outputs/stage5/canonical_crosswalk.csv   → direction/level/tier metadata

LOCO-uyumlu kural: source_cycle < target_cycle (leakage yok)
"""
from __future__ import annotations

import logging
import pathlib

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT_DIR      = PROJECT_ROOT / "outputs" / "stage5"

# ── Canonical crosswalk yükle ─────────────────────────────────────────────────
def load_crosswalk() -> pd.DataFrame:
    xwalk = pd.read_csv(OUT_DIR / "canonical_crosswalk.csv")
    # (program, raw_variable) → (canonical_construct, direction, level, tier)
    return xwalk


def crosswalk_lookup(xwalk: pd.DataFrame, program: str, raw_var: str) -> dict:
    row = xwalk[(xwalk["program"] == program) & (xwalk["raw_variable"] == raw_var)]
    if row.empty:
        return {}
    r = row.iloc[0]
    return {
        "canonical_construct": r["canonical_construct"],
        "direction": r["direction"],
        "level":     r["level"],
        "tier":      r["tier"],
        "forecast_role": r["forecast_role"],
    }


# ── 1. Intra-program panel'den uzun format oluştur ────────────────────────────
def build_intra_program_features(xwalk: pd.DataFrame) -> pd.DataFrame:
    """
    enriched_panel.csv'deki lag_* sütunlarını uzun formata çevir.
    Her lag_X sütunu → (source=same program, source_cycle=target_cycle-lag, canonical_construct=X)
    """
    panel = pd.read_csv(OUT_DIR / "enriched_panel.csv")

    # lag_* sütunlarını bul
    lag_cols = [c for c in panel.columns if c.startswith("lag_")]

    # Her program için hangi raw değişkenlerin lag'ini tuttuğunu biliyoruz:
    # lag_ESCS, lag_HOMEPOS, lag_BELONG, lag_ICTAVHOM, lag_ICTAVSCH → PISA
    # lag_BSDGEDUP → TIMSS
    # lag_ASDHEDUP → TIMSS_G4, PIRLS (ayrı program)
    # lag_ASDHELA  → PIRLS
    INTRA_MAP = {
        "lag_ESCS":      ("PISA",     "ESCS"),
        "lag_HOMEPOS":   ("PISA",     "HOMEPOS"),
        "lag_BELONG":    ("PISA",     "BELONG"),
        "lag_ICTAVHOM":  ("PISA",     "ICTAVHOM"),
        "lag_ICTAVSCH":  ("PISA",     "ICTAVSCH"),
        "lag_BSDGEDUP":  ("TIMSS",    "BSDGEDUP"),
        "lag_ASDHEDUP":  ("TIMSS_G4", "ASDHEDUP"),  # PIRLS için ayrı mantık gerekir
        "lag_ASDHELA":   ("PIRLS",    "ASDHELA"),
    }

    # Domain bilgisi olup olmadığını kontrol et
    has_domain = "domain" in panel.columns

    records = []
    for _, row in panel.iterrows():
        tgt_prog  = row["program"]
        tgt_cycle = int(row["cycle"])
        country   = row["country_iso3"]
        domain    = row["domain"] if has_domain else None

        for lag_col, (src_prog_hint, raw_var) in INTRA_MAP.items():
            val = row.get(lag_col, np.nan)
            if not np.isfinite(float(val) if pd.notna(val) else np.nan):
                continue

            # Kaynak program: eğer target==PIRLS ve lag_ASDHEDUP ise src=PIRLS
            if lag_col == "lag_ASDHEDUP" and tgt_prog == "PIRLS":
                src_prog = "PIRLS"
            elif lag_col == "lag_ASDHEDUP" and tgt_prog == "TIMSS_G4":
                src_prog = "TIMSS_G4"
            else:
                src_prog = src_prog_hint

            # Crosswalk'tan metadata
            meta = crosswalk_lookup(xwalk, src_prog, raw_var)
            if not meta:
                meta = {"canonical_construct": raw_var, "direction": "?",
                        "level": "?", "tier": "?", "forecast_role": "?"}

            # Tahmini kaynak cycle: target - 1 cycle
            # (panel'de tam lag yılı bilinmiyor; source=prior cycle notasyonu)
            records.append({
                "target_country":  country,
                "target_program":  tgt_prog,
                "target_cycle":    tgt_cycle,
                "target_domain":   domain,
                "source_program":  src_prog,
                "source_cycle":    None,    # intra-program; panel'den tam değer yok
                "lag_years":       None,    # aynı
                "canonical_construct": meta["canonical_construct"],
                "raw_variable":    raw_var,
                "value":           float(val),
                "direction":       meta["direction"],
                "level":           meta["level"],
                "tier":            meta["tier"],
                "forecast_role":   meta["forecast_role"],
                "feature_source":  "intra_program_lag",
            })

    df = pd.DataFrame(records)
    # Domain'den bağımsız covariate satırlarını deduplicate et
    # (aynı country×program×cycle×source×construct değeri birden fazla domain için tekrar eder)
    dup_key = ["target_country","target_program","target_cycle",
               "source_program","canonical_construct","raw_variable"]
    df = df.drop_duplicates(subset=dup_key).copy()
    df = df.drop(columns=["target_domain"], errors="ignore")
    log.info("Intra-program features (deduplicated): %d satır", len(df))
    return df


# ── 2. TALIS cross-program features ──────────────────────────────────────────
def build_talis_features(xwalk: pd.DataFrame) -> pd.DataFrame:
    talis = pd.read_csv(OUT_DIR / "talis_covariate_estimates.csv")

    # TALIS construct → canonical: construct_map (build_talis_covariates'ten)
    CONSTRUCT_TO_RAWVAR = {
        "self_efficacy_cls_mgmt":     {2013: "SECLSS",   2018: "T3SECLS"},
        "self_efficacy_instruction":  {2013: "SEINSS",   2018: "T3SEINS"},
        "self_efficacy_engagement":   {2013: "SEENGS",   2018: "T3SEENG"},
        "disciplinary_climate":       {2013: "TCDISCS",  2018: "T3DISC"},
        "prof_collaboration":         {2013: "TCCOLLS",  2018: "T3COLES"},
        "job_sat_environment":        {2013: "TJSENVS",  2018: "T3JSENV"},
        "job_sat_profession":         {2013: "TJSPROS",  2018: "T3JSPRO"},
        "effective_pd":               {2013: "TEFFPROS", 2018: "T3EFFPD"},
    }

    records = []
    for _, row in talis.iterrows():
        cycle     = int(row["cycle"])
        construct = row["construct"]
        raw_map   = CONSTRUCT_TO_RAWVAR.get(construct, {})
        raw_var   = raw_map.get(cycle, construct)

        meta = crosswalk_lookup(xwalk, "TALIS", raw_var)
        if not meta:
            meta = {"canonical_construct": construct.upper(), "direction": "?",
                    "level": "teacher", "tier": "B", "forecast_role": "contextual_predictor"}

        records.append({
            "target_country":  row["country_iso3"],
            "target_program":  row["target_program"],
            "target_cycle":    int(row["target_cycle"]),
            "source_program":  "TALIS",
            "source_cycle":    cycle,
            "lag_years":       int(row["lag_years"]),
            "canonical_construct": meta["canonical_construct"],
            "raw_variable":    raw_var,
            "value":           float(row["mean"]),
            "direction":       meta["direction"],
            "level":           meta["level"],
            "tier":            meta["tier"],
            "forecast_role":   meta["forecast_role"],
            "feature_source":  "talis_cross_program",
        })

    df = pd.DataFrame(records)
    log.info("TALIS cross-program features: %d satır", len(df))
    return df


# ── 3. Panel'deki primary outcome (LOCO hedefi) ──────────────────────────────
def build_outcome_rows() -> pd.DataFrame:
    panel = pd.read_csv(OUT_DIR / "enriched_panel.csv")
    outcome_cols = [c for c in panel.columns
                    if c not in ("program","cycle","country_iso3","domain") and
                    not c.startswith("lag_")]

    records = []
    for _, row in panel.iterrows():
        for col in outcome_cols:
            val = row.get(col, np.nan)
            if pd.isna(val):
                continue
            records.append({
                "target_country":  row["country_iso3"],
                "target_program":  row["program"],
                "target_cycle":    int(row["cycle"]),
                "source_program":  row["program"],
                "source_cycle":    int(row["cycle"]),
                "lag_years":       0,
                "canonical_construct": col.upper(),
                "raw_variable":    col,
                "value":           float(val),
                "direction":       "+",
                "level":           "country",
                "tier":            "A",
                "forecast_role":   "outcome",
                "feature_source":  "primary_outcome",
            })

    df = pd.DataFrame(records)
    log.info("Outcome rows: %d satır", len(df))
    return df


def main():
    xwalk = load_crosswalk()
    log.info("Crosswalk yüklendi: %d satır", len(xwalk))

    intra  = build_intra_program_features(xwalk)
    talis  = build_talis_features(xwalk)
    outcome = build_outcome_rows()

    panel_v2 = pd.concat([outcome, intra, talis], ignore_index=True)

    # Leakage kontrolü: source_cycle >= target_cycle olanları işaretle
    leakage = panel_v2[
        panel_v2["source_cycle"].notna() &
        (panel_v2["source_cycle"] >= panel_v2["target_cycle"]) &
        (panel_v2["lag_years"] != 0)
    ]
    if len(leakage):
        log.warning("LEAKAGE uyarısı: %d satır (source_cycle >= target_cycle)", len(leakage))
        log.warning(leakage[["target_program","target_cycle","source_program","source_cycle"]].drop_duplicates().to_string())
    else:
        log.info("Leakage kontrolü: temiz ✓")

    # Duplicate feature kontrolü (outcome hariç)
    dup_key = ["target_country","target_program","target_cycle",
               "source_program","canonical_construct"]
    non_outcome = panel_v2[panel_v2["forecast_role"] != "outcome"]
    n_dup = non_outcome.duplicated(subset=dup_key).sum()
    if n_dup:
        log.warning("Duplicate feature satırları: %d", n_dup)

    out_csv = OUT_DIR / "enriched_panel_v2.csv"
    panel_v2.to_csv(out_csv, index=False)
    log.info("Kaydedildi: %s (%d satır)", out_csv, len(panel_v2))

    # Özet
    print("\n=== Feature source × forecast_role ===")
    print(panel_v2.groupby(["feature_source","forecast_role"])["target_country"].count().to_string())

    print("\n=== TALIS features: target_program × target_cycle × construct ===")
    talis_summary = (
        talis.groupby(["target_program","target_cycle","canonical_construct"])
        ["target_country"].count()
        .reset_index()
        .rename(columns={"target_country": "n_countries"})
    )
    print(talis_summary.to_string(index=False))

    print("\n=== Direction dağılımı ===")
    print(panel_v2[panel_v2["forecast_role"] != "outcome"]
          .groupby("direction")["canonical_construct"].nunique().to_string())

    print(f"\nToplam: {len(panel_v2)} satır")
    print(f"  - Outcome:   {(panel_v2['forecast_role']=='outcome').sum()}")
    print(f"  - Features:  {(panel_v2['forecast_role']!='outcome').sum()}")
    print(f"  - Countries: {panel_v2['target_country'].nunique()}")
    print(f"  - Constructs: {panel_v2['canonical_construct'].nunique()}")


if __name__ == "__main__":
    main()
