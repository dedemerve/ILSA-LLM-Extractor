#!/usr/bin/env python3
"""
Unified panel oluşturucu: tüm 7 ILSA programını ve lag_ kovaryatları birleştirir.

Strateji:
  1. country_estimates.csv → uzun format → geniş pivot (tüm 7 program skoru)
  2. enriched_panel.csv → lag_ kovaryat sütunları çıkar
  3. 2. adımdaki kovaryatları (country_iso3, cycle) üzerinden birleştir
  4. PISA legacy cycle'ları (2003-2012) için ESCS ülke ortalamalarını ayrıca merge et
  5. Çıktı: outputs/stage5/unified_panel.csv  (eski enriched_panel'in tam yedeği)

Panel formatı (geniş):
  country_iso3 | cycle | program | domain |
  <prog_domain score cols>... | lag_ESCS | lag_HOMEPOS | ... |
  <prog-specific lag cols>...
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT  = Path(__file__).resolve().parents[1]
STAGE4_DIR    = PROJECT_ROOT / "outputs" / "stage4"
STAGE5_DIR    = PROJECT_ROOT / "outputs" / "stage5"
ILSA_BASE     = Path("/Users/mrved/Desktop/ILSA Datasets/PISA Datasets")

ESTIMATES_CSV  = STAGE4_DIR / "country_estimates.csv"
ENRICHED_CSV   = STAGE5_DIR / "enriched_panel.csv"
OUT_UNIFIED    = STAGE5_DIR / "unified_panel.csv"


# ── 1. Skor pivot ────────────────────────────────────────────────────────────

def make_score_pivot(est: pd.DataFrame) -> pd.DataFrame:
    """country_estimates → geniş skor pivot.

    Her (country_iso3, cycle, program, domain) → <program>_<domain> sütunu.
    Program/domain bilgisi korunur — LOCO döngüsünün hedef sütunu seçmesi için.
    """
    est = est.copy()
    est["feat_key"] = est["program"].str.upper() + "_" + est["domain"].str.lower()
    # Pivot: her (country, cycle) için tüm program_domain skorları
    wide = est.pivot_table(
        index=["country_iso3", "cycle"],
        columns="feat_key",
        values="mean",
        aggfunc="first",
    ).reset_index()
    wide.columns.name = None

    # program + domain bilgisini geri ekle (LOCO target seçimi için)
    prog_dom = est[["country_iso3", "cycle", "program", "domain"]].drop_duplicates()
    wide = prog_dom.merge(wide, on=["country_iso3", "cycle"], how="left")
    return wide


# ── 2. Kovaryat çıkarma ───────────────────────────────────────────────────────

def extract_covariates(enriched: pd.DataFrame) -> pd.DataFrame:
    """enriched_panel'den (country_iso3, cycle) → lag_ kovaryat tablosu."""
    lag_cols = [c for c in enriched.columns if c.startswith("lag_")]
    cov = (
        enriched[["country_iso3", "cycle"] + lag_cols]
        .groupby(["country_iso3", "cycle"])
        .first()
        .reset_index()
    )
    return cov


# ── 3. PISA legacy ESCS ülke ortalamaları ────────────────────────────────────

def _parse_spss_colmap(syntax_path: Path, encoding: str = "latin-1") -> dict[str, tuple[int, int]]:
    import re
    text = syntax_path.read_text(encoding=encoding, errors="replace")
    in_dl, col_map = False, {}
    for line in text.splitlines():
        s = line.strip()
        if re.search(r"DATA\s+LIST", s, re.IGNORECASE):
            in_dl = True; continue
        if in_dl:
            if re.match(r"(VARIABLE\s+LABELS|VALUE\s+LABELS|FORMATS|MISSING|EXECUTE|SAVE|GET)", s, re.IGNORECASE):
                break
            m = re.match(r"([A-Z_][A-Z0-9_]*)\s+(\d+)\s*[-–]\s*(\d+)", s, re.IGNORECASE)
            if m:
                col_map[m.group(1).upper()] = (int(m.group(2)) - 1, int(m.group(3)))
    return col_map


PISA_LEGACY_ESCS = [
    {
        "cycle": 2003,
        "syntax": ILSA_BASE / "PISA 2003 Data/PISA2003_SPSS_student.txt",
        "data":   ILSA_BASE / "PISA 2003 Data/INT_stui_2003_v2.txt",
        "cnt_col": "CNT", "escs_col": "ESCS", "w_col": "W_FSTUWT",
    },
    {
        "cycle": 2006,
        "syntax": ILSA_BASE / "PISA 2006 Data/PISA2006_SPSS_student.txt",
        "data":   ILSA_BASE / "PISA 2006 Data/INT_Stu06_Dec07.txt",
        "cnt_col": "CNT", "escs_col": "ESCS", "w_col": "W_FSTUWT",
    },
    {
        "cycle": 2009,
        "syntax": ILSA_BASE / "PISA 2009 Data/PISA2009_SPSS_student.txt",
        "data":   ILSA_BASE / "PISA 2009 Data/INT_STQ09_DEC11.txt",
        "cnt_col": "CNT", "escs_col": "ESCS", "w_col": "W_FSTUWT",
    },
    {
        "cycle": 2012,
        "syntax": ILSA_BASE / "PISA 2012 Data/SPSS syntax to read in student questionnaire data file.txt",
        "data":   ILSA_BASE / "PISA 2012 Data/INT_STU12_DEC03.txt",
        "cnt_col": "CNT", "escs_col": "ESCS", "w_col": "W_FSTUWT",
    },
]

_CNT_TO_ISO3 = {"TAP": "TWN", "SHA": "CHN", "QCN": "CHN"}

def _cnt_iso3(code: str) -> str:
    c = str(code).strip().upper()
    return _CNT_TO_ISO3.get(c, c)


def build_pisa_legacy_escs() -> pd.DataFrame:
    """PISA 2003-2012 cycle'ları için ülke bazlı ağırlıklı ESCS ortalamaları.

    Dönüş: DataFrame(cycle, country_iso3, lag_ESCS)
    lag_ESCS_t = bir sonraki cycle için önceki cycle'ın ESCS'i
    (ör. 2003 ESCS → 2006 modeli için lag_ESCS)
    """
    rows = []
    for spec in PISA_LEGACY_ESCS:
        if not spec["syntax"].exists() or not spec["data"].exists():
            log.warning("  ESCS için dosya bulunamadı: %d", spec["cycle"])
            continue
        log.info("PISA %d ESCS okunuyor...", spec["cycle"])
        col_map = _parse_spss_colmap(spec["syntax"])

        needed = [spec["cnt_col"], spec["escs_col"], spec["w_col"]]
        available = [v for v in needed if v in col_map]
        if spec["escs_col"] not in col_map:
            log.warning("  %d: ESCS sütunu syntax'ta yok", spec["cycle"])
            continue

        colspecs = [col_map[v] for v in available]
        df = pd.read_fwf(
            spec["data"],
            colspecs=colspecs,
            names=available,
            encoding="latin-1",
            dtype=str,
        )
        df[spec["escs_col"]] = pd.to_numeric(df[spec["escs_col"]], errors="coerce")
        df[spec["w_col"]]    = pd.to_numeric(df[spec["w_col"]],    errors="coerce")
        # ESCS missing value kodlarını temizle (9997, 9998, 9999 → NaN)
        df.loc[df[spec["escs_col"]] > 90, spec["escs_col"]] = np.nan

        for cnt, grp in df.groupby(spec["cnt_col"]):
            iso3 = _cnt_iso3(str(cnt))
            if len(iso3) != 3:
                continue
            mask = grp[spec["escs_col"]].notna() & grp[spec["w_col"]].notna()
            if mask.sum() < 10:
                continue
            w = grp.loc[mask, spec["w_col"]]
            y = grp.loc[mask, spec["escs_col"]]
            wmean = float((w * y).sum() / w.sum())
            rows.append({"source_cycle": spec["cycle"], "country_iso3": iso3, "escs_mean": wmean})

    if not rows:
        return pd.DataFrame(columns=["cycle", "country_iso3", "lag_ESCS"])

    escs_df = pd.DataFrame(rows)
    log.info("Legacy ESCS: %d ülke-cycle", len(escs_df))

    # lag_ESCS_t = ESCS_{t-1}: sonraki PISA cycle'ına ata
    # PISA cycle gap: 3 yıl (2003→2006→2009→2012→2015)
    cycle_next = {2003: 2006, 2006: 2009, 2009: 2012, 2012: 2015}
    escs_df["cycle"] = escs_df["source_cycle"].map(cycle_next)
    escs_df = escs_df.dropna(subset=["cycle"])
    escs_df["cycle"] = escs_df["cycle"].astype(int)
    escs_df = escs_df.rename(columns={"escs_mean": "lag_ESCS"})[["cycle", "country_iso3", "lag_ESCS"]]
    return escs_df


# ── 4. Ana birleştirme ────────────────────────────────────────────────────────

def build_unified_panel() -> pd.DataFrame:
    log.info("country_estimates.csv okunuyor...")
    est = pd.read_csv(ESTIMATES_CSV)
    est["program"]      = est["program"].str.upper()
    est["domain"]       = est["domain"].str.lower()
    est["country_iso3"] = est["country_iso3"].astype(str).str.strip()
    est["cycle"]        = est["cycle"].astype(int)

    log.info("Skor pivot oluşturuluyor...")
    panel = make_score_pivot(est)

    log.info("enriched_panel kovaryatları merge ediliyor...")
    enriched = pd.read_csv(ENRICHED_CSV)
    cov = extract_covariates(enriched)
    panel = panel.merge(cov, on=["country_iso3", "cycle"], how="left")

    # PISA legacy ESCS — enriched_panel'de 2015 öncesi yok
    log.info("PISA legacy ESCS ekleniyor...")
    legacy_escs = build_pisa_legacy_escs()
    if not legacy_escs.empty:
        panel = panel.merge(legacy_escs, on=["cycle", "country_iso3"], how="left", suffixes=("", "_legacy"))
        # Mevcut lag_ESCS NaN ise legacy değerini kullan
        if "lag_ESCS_legacy" in panel.columns:
            panel["lag_ESCS"] = panel["lag_ESCS"].fillna(panel["lag_ESCS_legacy"])
            panel.drop(columns=["lag_ESCS_legacy"], inplace=True)

    # PISA 2000 cycle için ESCS ayrı SAV'dan — şimdilik NaN bırak (2000 lag olan fold yok)

    log.info("Unified panel: %d satır, %d sütun", len(panel), len(panel.columns))
    log.info("Programlar: %s", sorted(panel["program"].unique()))
    log.info("lag_ sütunları: %s", [c for c in panel.columns if c.startswith("lag_")])

    return panel


def main() -> None:
    STAGE5_DIR.mkdir(parents=True, exist_ok=True)
    panel = build_unified_panel()
    panel.to_csv(OUT_UNIFIED, index=False)
    log.info("Kaydedildi: %s  (%d satır)", OUT_UNIFIED, len(panel))

    # Özet
    print("\n=== Unified Panel Özeti ===")
    print(panel.groupby("program").size().rename("n_rows"))
    lag_cols = [c for c in panel.columns if c.startswith("lag_")]
    print(f"\nlag_ sütun sayısı: {len(lag_cols)}")
    for lc in lag_cols:
        filled = panel[lc].notna().sum()
        print(f"  {lc}: {filled}/{len(panel)} dolu ({100*filled/len(panel):.0f}%)")


if __name__ == "__main__":
    main()
