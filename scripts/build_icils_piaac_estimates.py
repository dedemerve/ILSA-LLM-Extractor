#!/usr/bin/env python3
"""
ICILS (2013, 2018, 2023) ve PIAAC (Cycle1=2012, Cycle2=2017) için
ülke düzeyinde ağırlıklı ortalamalar ve standart hatalar.

ICILS : IEA Zone-JRR (JKZONE+JKREP+TOTWGTS) → BSG*.sav ülke dosyaları
PIAAC : BRR (SPFWT1..SPFWT80, temel=SPFWT0) → prg[ISO3]p[c].csv dosyaları

Çıktı: outputs/stage5/icils_piaac_estimates.csv
Sütunlar: program, cycle, country_iso3, domain, mean, se, ci_lo, ci_hi,
          n_total, n_analytic, n_replicates, method, n_pv
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadstat

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STAGE5_DIR   = PROJECT_ROOT / "outputs" / "stage5"
STAGE5_DIR.mkdir(parents=True, exist_ok=True)

try:
    from scripts.ilsa_common import microdata_root
except ImportError:
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.ilsa_common import microdata_root

ILSA_BASE = microdata_root()

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Temel istatistik yardımcıları (build_country_estimates.py ile aynı)
# ---------------------------------------------------------------------------

def weighted_mean(y: np.ndarray, w: np.ndarray) -> float:
    w = w / w.sum()
    return float(np.sum(w * y))


def _variance_jrr_zones(df: pd.DataFrame, pv: str, weight: str, theta0: float) -> float:
    if "JKZONE" not in df.columns or "JKREP" not in df.columns:
        return 0.0
    x    = df[pv].values
    w0   = df[weight].values.astype(float)
    zones = df["JKZONE"].values
    reps  = df["JKREP"].values
    sq   = 0.0
    for z in np.unique(zones):
        mask_z = zones == z
        for r_val in (0, 1):
            w_rep = w0.copy()
            w_rep[mask_z & (reps == r_val)]  *= 2.0
            w_rep[mask_z & (reps != r_val)]   = 0.0
            if w_rep.sum() > 0:
                sq += (weighted_mean(x, w_rep) - theta0) ** 2
    return sq / 2.0


def _variance_brr(df: pd.DataFrame, pv: str, rep_cols: list[str], theta0: float) -> float:
    x  = df[pv].values.astype(float)
    sq = sum((weighted_mean(x, df[r].values.astype(float)) - theta0) ** 2 for r in rep_cols)
    return sq / len(rep_cols)


def rubin_pool_jrr_zones(
    df: pd.DataFrame, pv_vars: list[str], weight: str
) -> dict:
    """Rubin (1987) pooling over IEA Zone-JRR replicates."""
    m      = len(pv_vars)
    w0     = df[weight].values
    thetas, Us = [], []
    for pv in pv_vars:
        mask = df[pv].notna() & (w0 > 0)
        sub  = df[mask]
        if len(sub) < 30:
            continue
        th = weighted_mean(sub[pv].values, sub[weight].values)
        U  = _variance_jrr_zones(sub, pv, weight, th)
        thetas.append(th)
        Us.append(U)
    if not thetas:
        return None
    T_bar = float(np.mean(thetas))
    U_bar = float(np.mean(Us))
    B     = float(np.var(thetas, ddof=1)) if len(thetas) > 1 else 0.0
    T     = U_bar + (1 + 1 / len(thetas)) * B
    SE    = float(np.sqrt(max(T, 0)))
    return {"mean": round(T_bar, 4), "se": round(SE, 4),
            "ci_lo": round(T_bar - 1.96 * SE, 2),
            "ci_hi": round(T_bar + 1.96 * SE, 2),
            "n_pv":  len(thetas)}


def rubin_pool_brr(
    df: pd.DataFrame, pv_vars: list[str], weight: str, rep_cols: list[str]
) -> dict:
    """Rubin (1987) pooling over BRR replicates."""
    thetas, Us = [], []
    for pv in pv_vars:
        mask = df[pv].notna() & (df[weight] > 0)
        sub  = df[mask]
        if len(sub) < 30:
            continue
        th = weighted_mean(sub[pv].values.astype(float), sub[weight].values.astype(float))
        U  = _variance_brr(sub, pv, rep_cols, th)
        thetas.append(th)
        Us.append(U)
    if not thetas:
        return None
    T_bar = float(np.mean(thetas))
    U_bar = float(np.mean(Us))
    B     = float(np.var(thetas, ddof=1)) if len(thetas) > 1 else 0.0
    T     = U_bar + (1 + 1 / len(thetas)) * B
    SE    = float(np.sqrt(max(T, 0)))
    return {"mean": round(T_bar, 4), "se": round(SE, 4),
            "ci_lo": round(T_bar - 1.96 * SE, 2),
            "ci_hi": round(T_bar + 1.96 * SE, 2),
            "n_pv":  len(thetas)}


# ---------------------------------------------------------------------------
# ICILS — IEA Zone-JRR, BSG*.sav ülke dosyaları
# ---------------------------------------------------------------------------

ICILS_CATALOG = [
    # (cycle, data_dir, pv_vars, file_suffix)
    (2013, ILSA_BASE / "ICILS Datasets/ICILS2013_IDB_SPSS/Data",
     [f"PV{i}CIL" for i in range(1, 6)], "I1"),
    (2018, ILSA_BASE / "ICILS Datasets/ICILS2018_IDB_SPSS/Data",
     [f"PV{i}CIL" for i in range(1, 6)], "I2"),
    (2023, ILSA_BASE / "ICILS Datasets/ICILS2023_IDB_SPSS/Data",
     [f"PV{i}CIL" for i in range(1, 6)], "I3"),
]

# Non-standard or ambiguous 3-letter codes in ICILS filenames → correct ISO3
ICILS_CODE_FIX = {
    "ABA": "ARE",   # Abu Dhabi (UAE benchmark participant)
    "CNL": "TWN",   # Chinese Taipei (IEA code)
    "COT": "COL",   # Colombia
    "RMO": "MKD",   # Republic of Macedonia / North Macedonia
    "DNW": "DNK",   # Denmark (alternate IEA suffix; verify per cycle)
    "BFL": "BEL",   # Belgium (Flemish Community)
}


def single_weighted_mean_brr(
    df: pd.DataFrame, covar: str, weight: str, rep_cols: list[str]
) -> dict | None:
    """Country-level mean for a single continuous covariate via BRR."""
    try:
        df = df.copy()
        df[covar]  = pd.to_numeric(df[covar],  errors="coerce")
        df[weight] = pd.to_numeric(df[weight], errors="coerce")
    except Exception:
        return None
    sub = df[df[covar].notna() & (df[weight] > 0)]
    if len(sub) < 30:
        return None
    th = weighted_mean(sub[covar].values.astype(float), sub[weight].values.astype(float))
    U  = _variance_brr(sub, covar, rep_cols, th)
    SE = float(np.sqrt(max(U, 0)))
    return {"mean": round(th, 4), "se": round(SE, 4),
            "ci_lo": round(th - 1.96 * SE, 2),
            "ci_hi": round(th + 1.96 * SE, 2),
            "n_pv": 0}


def single_weighted_mean_jrr(
    df: pd.DataFrame, covar: str, weight: str
) -> dict | None:
    """Country-level mean for a single continuous covariate via JRR_ZONES."""
    sub = df[df[covar].notna() & (df[weight] > 0)]
    if len(sub) < 30:
        return None
    th = weighted_mean(sub[covar].values.astype(float), sub[weight].values.astype(float))
    U  = _variance_jrr_zones(sub, covar, weight, th)
    SE = float(np.sqrt(max(U, 0)))
    return {"mean": round(th, 4), "se": round(SE, 4),
            "ci_lo": round(th - 1.96 * SE, 2),
            "ci_hi": round(th + 1.96 * SE, 2),
            "n_pv": 0}


def process_icils_cycle(cycle: int, data_dir: Path, pv_vars: list[str]) -> list[dict]:
    if not data_dir.exists():
        log.warning("ICILS %d dizin yok: %s", cycle, data_dir)
        return []

    sav_files = sorted(data_dir.glob("BSG*.sav"))
    if not sav_files:
        log.warning("ICILS %d BSG*.sav bulunamadı", cycle)
        return []

    log.info("ICILS %d → %d ülke dosyası", cycle, len(sav_files))

    rows = []
    for fpath in sav_files:
        raw_code = fpath.stem[3:6].upper()
        iso3     = ICILS_CODE_FIX.get(raw_code, raw_code)

        try:
            df, _ = pyreadstat.read_sav(str(fpath), apply_value_formats=False)
        except Exception as exc:
            log.warning("  Okunamadı %s: %s", fpath.name, exc)
            continue

        df.columns = [c.upper() for c in df.columns]

        # Verify PVs present
        avail_pv = [p for p in pv_vars if p in df.columns]
        if not avail_pv:
            log.debug("  PV bulunamadı %s, atlanıyor", fpath.name)
            continue

        # Weight
        weight = "TOTWGTS" if "TOTWGTS" in df.columns else "TOTWGT"
        if weight not in df.columns:
            log.warning("  Ağırlık sütunu yok: %s", fpath.name)
            continue

        n_total    = len(df)
        mask       = df[avail_pv[0]].notna() & (df[weight] > 0)
        n_analytic = mask.sum()
        if n_analytic < 30:
            log.debug("  n_analytic=%d < 30, atlanıyor: %s", n_analytic, iso3)
            continue

        stats = rubin_pool_jrr_zones(df[mask], avail_pv, weight)
        if stats is None:
            continue

        rows.append({
            "program":      "ICILS",
            "cycle":        cycle,
            "domain":       "computer_literacy",
            "country_iso3": iso3,
            "method":       "JRR_ZONES",
            "n_total":      n_total,
            "n_analytic":   n_analytic,
            "n_replicates": 0,
            **stats,
        })

        # Covariates
        for covar_domain, covar_col in ICILS_COVARIATES.items():
            ucol = covar_col.upper()
            # Fallback: S_NISB → S_HISEI for ICILS 2023
            if ucol not in df.columns and covar_domain == "SES_COMPOSITE":
                ucol = ICILS_SES_FALLBACK.upper()
            if ucol not in df.columns:
                continue
            weight = "TOTWGTS" if "TOTWGTS" in df.columns else "TOTWGT"
            cov_stats = single_weighted_mean_jrr(df, ucol, weight)
            if cov_stats is None:
                continue
            rows.append({
                "program":      "ICILS",
                "cycle":        cycle,
                "domain":       covar_domain,
                "country_iso3": iso3,
                "method":       "JRR_ZONES",
                "n_total":      n_total,
                "n_analytic":   df[ucol].notna().sum(),
                "n_replicates": 0,
                **cov_stats,
            })

    log.info("  ICILS %d → %d satır (outcome+covariates)", cycle, len(rows))
    return rows


# ---------------------------------------------------------------------------
# PIAAC — BRR (SPFWT0=base, SPFWT1..SPFWT80=replicates)
# ---------------------------------------------------------------------------

PIAAC_CATALOG = [
    # (cycle_label, cycle_year, data_dir, sep, file_glob)
    ("C1", 2012,
     ILSA_BASE / "PIAAC Datasets/PIAAC Cycle1",
     ",", "prg???p1.csv"),
    ("C2", 2017,
     ILSA_BASE / "PIAAC Datasets/PIAAC Cycle2",
     ";", "prg???p2.csv"),
]

PIAAC_PV_VARS = {
    "literacy":  [f"PVLIT{i}"  for i in range(1, 11)],
    "numeracy":  [f"PVNUM{i}"  for i in range(1, 11)],
}

# Covariates: single continuous variable (no PV pooling needed)
# domain label → column name per cycle
ICILS_COVARIATES = {
    # S_NISB: National Index of SES (~ESCS composite) — available 2013 & 2018
    # S_HISEI: Highest ISEI of parents — available all cycles (fallback)
    "SES_COMPOSITE":          "S_NISB",   # preferred; falls back to S_HISEI if absent
    # S_HISCED: highest ISCED of parents
    "PARENTAL_EDUCATION":     "S_HISCED",
    # S_HOMLIT: home literacy environment
    "HOME_LITERACY_ACTIVITIES": "S_HOMLIT",
}

ICILS_SES_FALLBACK = "S_HISEI"   # used when S_NISB absent (ICILS 2023)

# PIAAC covariate columns per cycle
PIAAC_COVARIATES = {
    2012: {
        "PARENTAL_EDUCATION": "PARED",    # parents' highest ISCED (0–4 or 0–3)
        "GENDER":             "GENDER_R", # 1=male, 2=female
    },
    2017: {
        "PARENTAL_EDUCATION": "PAREDC2",
        "GENDER":             "GENDER_R",
    },
}


def process_piaac_cycle(cycle_label: str, cycle_year: int,
                         data_dir: Path, sep: str, file_glob: str) -> list[dict]:
    if not data_dir.exists():
        log.warning("PIAAC %s dizin yok: %s", cycle_label, data_dir)
        return []

    csv_files = sorted(data_dir.glob(file_glob))
    if not csv_files:
        log.warning("PIAAC %s dosya bulunamadı", cycle_label)
        return []

    log.info("PIAAC %s (%d) → %d ülke dosyası", cycle_label, cycle_year, len(csv_files))

    rows = []
    for fpath in csv_files:
        iso3 = fpath.stem[3:6].upper()

        try:
            df = pd.read_csv(str(fpath), sep=sep, low_memory=False)
        except Exception as exc:
            log.warning("  Okunamadı %s: %s", fpath.name, exc)
            continue

        # Deduplicate columns (some files have repeated country cols)
        df = df.loc[:, ~df.columns.duplicated()]

        # Base weight
        if "SPFWT0" not in df.columns:
            log.warning("  SPFWT0 yok: %s", fpath.name)
            continue

        rep_cols = sorted(c for c in df.columns if c.startswith("SPFWT") and c != "SPFWT0")
        if not rep_cols:
            log.warning("  BRR replicate ağırlık yok: %s", fpath.name)
            continue

        n_total = len(df)

        for domain, pv_vars in PIAAC_PV_VARS.items():
            avail_pv = [p for p in pv_vars if p in df.columns]
            if not avail_pv:
                continue

            mask       = df[avail_pv[0]].notna() & (df["SPFWT0"] > 0)
            n_analytic = mask.sum()
            if n_analytic < 30:
                continue

            stats = rubin_pool_brr(df[mask], avail_pv, "SPFWT0", rep_cols)
            if stats is None:
                continue

            rows.append({
                "program":      "PIAAC",
                "cycle":        cycle_year,
                "domain":       domain,
                "country_iso3": iso3,
                "method":       "BRR",
                "n_total":      n_total,
                "n_analytic":   n_analytic,
                "n_replicates": len(rep_cols),
                **stats,
            })

        # Covariates
        for covar_domain, covar_col in PIAAC_COVARIATES.get(cycle_year, {}).items():
            if covar_col not in df.columns:
                continue
            cov_stats = single_weighted_mean_brr(df, covar_col, "SPFWT0", rep_cols)
            if cov_stats is None:
                continue
            rows.append({
                "program":      "PIAAC",
                "cycle":        cycle_year,
                "domain":       covar_domain,
                "country_iso3": iso3,
                "method":       "BRR",
                "n_total":      n_total,
                "n_analytic":   df[covar_col].notna().sum(),
                "n_replicates": len(rep_cols),
                **cov_stats,
            })

    log.info("  PIAAC %s → %d satır (outcome+covariates)", cycle_label, len(rows))
    return rows


# ---------------------------------------------------------------------------
# Ana fonksiyon
# ---------------------------------------------------------------------------

def build_icils_piaac_estimates() -> pd.DataFrame:
    all_rows: list[dict] = []

    # ICILS
    for cycle, data_dir, pv_vars, _ in ICILS_CATALOG:
        all_rows.extend(process_icils_cycle(cycle, data_dir, pv_vars))

    # PIAAC
    for cycle_label, cycle_year, data_dir, sep, file_glob in PIAAC_CATALOG:
        all_rows.extend(process_piaac_cycle(
            cycle_label, cycle_year, data_dir, sep, file_glob
        ))

    df = pd.DataFrame(all_rows)
    if df.empty:
        log.error("Hiç satır üretilemedi!")
        return df

    for col in ("mean", "se", "ci_lo", "ci_hi"):
        if col in df.columns:
            df[col] = df[col].round(4)

    out = STAGE5_DIR / "icils_piaac_estimates.csv"
    df.to_csv(out, index=False)
    log.info("Kaydedildi: %s  (%d satır)", out, len(df))

    # Özet
    print("\n=== Özet ===")
    summary = df.groupby(["program","cycle","domain"])["country_iso3"].count().reset_index()
    summary.columns = ["program","cycle","domain","n_countries"]
    print(summary.to_string(index=False))
    return df


if __name__ == "__main__":
    build_icils_piaac_estimates()
