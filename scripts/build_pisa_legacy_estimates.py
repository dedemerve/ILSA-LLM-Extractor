#!/usr/bin/env python3
"""
PISA 2000–2012 legacy (fixed-width ASCII) veri okuyucu.

Her cycle için SPSS syntax dosyasından sütun pozisyonlarını çıkarır,
ASCII veri dosyasını pandas read_fwf ile okur, ülke bazlı PV ortalaması
ve BRR Fay (k=0.5) standart hatasını hesaplar.

Çıktı: outputs/stage4/pisa_legacy_estimates.csv
       (country_estimates.csv ile aynı format)

Ardından bu dosya country_estimates.csv'ye merge edilir.
"""

from __future__ import annotations

import re
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ILSA_BASE    = Path("/Users/mrved/Desktop/ILSA Datasets/PISA Datasets")
OUT_DIR      = PROJECT_ROOT / "outputs" / "stage4"
OUT_LEGACY   = OUT_DIR / "pisa_legacy_estimates.csv"
ESTIMATES_CSV = OUT_DIR / "country_estimates.csv"

# BRR Fay k=0.5
FAY_K = 0.5

# ── Cycle kataloğu ──────────────────────────────────────────────────────────
CYCLE_CATALOG = [
    {
        "cycle": 2000,
        "syntax": ILSA_BASE / "PISA 2000 Data/PISA2000_SPSS_student_reading.txt",
        "data":   ILSA_BASE / "PISA 2000 Data/intstud_read_v3.txt",
        "cnt_var": "COUNTRY",   # 2000'de CNT yok, COUNTRY var (3-char)
        "domains": {
            "reading": ("pv1read","pv2read","pv3read","pv4read","pv5read"),
        },
        "escs_var": None,       # 2000'de ESCS ana dosyada yok (ayrı SAV var)
        "weight_var": "w_fstuwt",
        "n_repl": 80,
        "repl_prefix": "w_fstr",
        "encoding": "latin-1",
    },
    {
        "cycle": 2003,
        "syntax": ILSA_BASE / "PISA 2003 Data/PISA2003_SPSS_student.txt",
        "data":   ILSA_BASE / "PISA 2003 Data/INT_stui_2003_v2.txt",
        "cnt_var": "CNT",
        "domains": {
            "mathematics": ("PV1MATH","PV2MATH","PV3MATH","PV4MATH","PV5MATH"),
            "reading":     ("PV1READ","PV2READ","PV3READ","PV4READ","PV5READ"),
            "science":     ("PV1SCIE","PV2SCIE","PV3SCIE","PV4SCIE","PV5SCIE"),
        },
        "escs_var": "ESCS",
        "weight_var": "W_FSTUWT",
        "n_repl": 80,
        "repl_prefix": "W_FSTR",
        "encoding": "latin-1",
    },
    {
        "cycle": 2006,
        "syntax": ILSA_BASE / "PISA 2006 Data/PISA2006_SPSS_student.txt",
        "data":   ILSA_BASE / "PISA 2006 Data/INT_Stu06_Dec07.txt",
        "cnt_var": "CNT",
        "domains": {
            "mathematics": ("PV1MATH","PV2MATH","PV3MATH","PV4MATH","PV5MATH"),
            "reading":     ("PV1READ","PV2READ","PV3READ","PV4READ","PV5READ"),
            "science":     ("PV1SCIE","PV2SCIE","PV3SCIE","PV4SCIE","PV5SCIE"),
        },
        "escs_var": "ESCS",
        "weight_var": "W_FSTUWT",
        "n_repl": 80,
        "repl_prefix": "W_FSTR",
        "encoding": "latin-1",
    },
    {
        "cycle": 2009,
        "syntax": ILSA_BASE / "PISA 2009 Data/PISA2009_SPSS_student.txt",
        "data":   ILSA_BASE / "PISA 2009 Data/INT_STQ09_DEC11.txt",
        "cnt_var": "CNT",
        "domains": {
            "mathematics": ("PV1MATH","PV2MATH","PV3MATH","PV4MATH","PV5MATH"),
            "reading":     ("PV1READ","PV2READ","PV3READ","PV4READ","PV5READ"),
            "science":     ("PV1SCIE","PV2SCIE","PV3SCIE","PV4SCIE","PV5SCIE"),
        },
        "escs_var": "ESCS",
        "weight_var": "W_FSTUWT",
        "n_repl": 80,
        "repl_prefix": "W_FSTR",
        "encoding": "latin-1",
    },
    {
        "cycle": 2012,
        "syntax": ILSA_BASE / "PISA 2012 Data/SPSS syntax to read in student questionnaire data file.txt",
        "data":   ILSA_BASE / "PISA 2012 Data/INT_STU12_DEC03.txt",
        "cnt_var": "CNT",
        "domains": {
            "mathematics": ("PV1MATH","PV2MATH","PV3MATH","PV4MATH","PV5MATH"),
            "reading":     ("PV1READ","PV2READ","PV3READ","PV4READ","PV5READ"),
            "science":     ("PV1SCIE","PV2SCIE","PV3SCIE","PV4SCIE","PV5SCIE"),
        },
        "escs_var": "ESCS",
        "weight_var": "W_FSTUWT",
        "n_repl": 80,
        "repl_prefix": "W_FSTR",
        "encoding": "latin-1",
    },
]

# OECD PISA country code → ISO3 haritası (PISA'da kullanılan 3-harf kodlar zaten ISO3)
# Bazı exception'lar:
_CNT_TO_ISO3: dict[str, str] = {
    "TAP": "TWN",  # Chinese Taipei
    "QAT": "QAT",
    "MAC": "MAC",  # Macao
    "HKG": "HKG",
    "SHA": "CHN",  # Shanghai-China → CHN
    "QCN": "CHN",  # (bazı cycle'larda)
}


def cnt_to_iso3(code: str) -> str:
    code = str(code).strip().upper()
    return _CNT_TO_ISO3.get(code, code)


# ── SPSS Syntax parser ──────────────────────────────────────────────────────

def parse_spss_syntax(syntax_path: Path, encoding: str = "latin-1") -> dict[str, tuple[int, int]]:
    """DATA LIST bloğundaki 'VARNAME  start - end' satırlarını parse eder.

    Dönüş: {VARNAME_upper: (start_0idx, end_0idx_excl)}
    SPSS 1-indexed, dahil; Python 0-indexed, hariç.
    """
    text = syntax_path.read_text(encoding=encoding, errors="replace")

    # DATA LIST bloğunu bul (ilk / ile başlar, END DATA veya VARIABLE LABELS kadar)
    in_datalist = False
    col_map: dict[str, tuple[int, int]] = {}

    for line in text.splitlines():
        stripped = line.strip()

        # DATA LIST satırı
        if re.search(r"DATA\s+LIST", stripped, re.IGNORECASE):
            in_datalist = True
            continue

        if in_datalist:
            # Blok sonu
            if re.match(r"(VARIABLE\s+LABELS|VALUE\s+LABELS|FORMATS|MISSING|EXECUTE|SAVE|GET|^\s*\.$)",
                        stripped, re.IGNORECASE):
                break

            # VARNAME   start - end  (isteğe bağlı type)
            m = re.match(
                r"([A-Z_][A-Z0-9_]*)\s+(\d+)\s*[-–]\s*(\d+)",
                stripped, re.IGNORECASE
            )
            if m:
                varname = m.group(1).upper()
                start   = int(m.group(2)) - 1   # 0-indexed
                end     = int(m.group(3))        # hariç (Python slice)
                col_map[varname] = (start, end)

    return col_map


def _needed_vars(spec: dict) -> list[str]:
    """Bu cycle için okunması gereken değişken listesi."""
    needed = [spec["cnt_var"].upper()]
    for pvs in spec["domains"].values():
        needed.extend(v.upper() for v in pvs)
    if spec["escs_var"]:
        needed.append(spec["escs_var"].upper())
    needed.append(spec["weight_var"].upper())
    for i in range(1, spec["n_repl"] + 1):
        needed.append(f"{spec['repl_prefix'].upper()}{i}")
    return needed


def read_cycle(spec: dict) -> pd.DataFrame:
    """SPSS syntax + ASCII veri dosyasını oku, gerekli sütunları döndür."""
    log.info("Parsing syntax: %s", spec["syntax"].name)
    col_map = parse_spss_syntax(spec["syntax"], spec["encoding"])

    needed = _needed_vars(spec)
    missing_vars = [v for v in needed if v not in col_map]
    if missing_vars:
        log.warning("  Syntax'ta bulunamayan değişkenler: %s", missing_vars)
        # Gerçekten kritik olanları kontrol et
        critical = [v for v in missing_vars
                    if v not in (spec["repl_prefix"].upper() + str(i) for i in range(1, spec["n_repl"]+1))]
        if any(v == spec["cnt_var"].upper() or
               v == spec["weight_var"].upper() or
               any(v in pvs for pvs in spec["domains"].values())
               for v in critical):
            raise ValueError(f"Kritik değişken eksik: {critical}")

    available = [v for v in needed if v in col_map]

    # colspecs: list of (start, end) for read_fwf
    colspecs = [col_map[v] for v in available]
    names    = [v for v in available]

    log.info("  Veri okunuyor: %s  (%d sütun)", spec["data"].name, len(names))
    df = pd.read_fwf(
        spec["data"],
        colspecs=colspecs,
        names=names,
        encoding=spec["encoding"],
        na_values=["", " ", "9", "99", "999", "9999", "99999", "999999"],
        dtype=str,
    )
    log.info("  Okunan: %d satır", len(df))

    # Sayısal dönüşüm
    for col in names:
        if col != spec["cnt_var"].upper():
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# ── BRR Fay SE hesabı ───────────────────────────────────────────────────────

def brr_fay_se(
    df: pd.DataFrame,
    pv_cols: list[str],
    weight_col: str,
    repl_cols: list[str],
    fay_k: float = 0.5,
) -> tuple[float, float]:
    """Ülke alt-kümesi için PV ortalaması ve BRR Fay SE hesapla."""
    # PV sampling variance (R=len(pv_cols))
    pv_means = []
    for pv in pv_cols:
        mask = df[pv].notna() & df[weight_col].notna()
        if mask.sum() == 0:
            return np.nan, np.nan
        w = df.loc[mask, weight_col]
        y = df.loc[mask, pv]
        pv_means.append(float((w * y).sum() / w.sum()))

    theta_hat = float(np.mean(pv_means))

    # BRR measurement variance
    brr_vars = []
    for rc in repl_cols:
        if rc not in df.columns:
            continue
        rep_means = []
        for pv in pv_cols:
            mask = df[pv].notna() & df[rc].notna()
            if mask.sum() == 0:
                rep_means.append(np.nan)
                continue
            w_r = df.loc[mask, rc]
            y_r = df.loc[mask, pv]
            rep_means.append(float((w_r * y_r).sum() / w_r.sum()))
        rep_mean = float(np.nanmean(rep_means))
        brr_vars.append((rep_mean - theta_hat) ** 2)

    if not brr_vars:
        return theta_hat, np.nan

    R = len(brr_vars)
    var_brr  = (1 / (R * (1 - fay_k) ** 2)) * sum(brr_vars)
    # Imputation variance
    var_imp  = (1 + 1/len(pv_cols)) * float(np.var(pv_means, ddof=1)) if len(pv_means) > 1 else 0.0
    var_total = var_brr + var_imp
    se = float(np.sqrt(max(var_total, 0)))

    return theta_hat, se


# ── Ana işlev ────────────────────────────────────────────────────────────────

def process_cycle(spec: dict) -> pd.DataFrame:
    df = read_cycle(spec)

    cnt_col  = spec["cnt_var"].upper()
    w_col    = spec["weight_var"].upper()
    repl_cols = [f"{spec['repl_prefix'].upper()}{i}"
                 for i in range(1, spec["n_repl"] + 1)
                 if f"{spec['repl_prefix'].upper()}{i}" in df.columns]

    rows = []
    for domain, pv_names in spec["domains"].items():
        pv_cols = [v.upper() for v in pv_names if v.upper() in df.columns]
        if not pv_cols:
            log.warning("  %d %s: PV sütunları bulunamadı, atlanıyor", spec["cycle"], domain)
            continue

        for cnt, grp in df.groupby(cnt_col):
            cnt_str = cnt_to_iso3(str(cnt))
            if len(cnt_str) != 3:
                continue

            mean_val, se_val = brr_fay_se(grp, pv_cols, w_col, repl_cols, FAY_K)
            if np.isnan(mean_val):
                continue

            n_analytic = int(grp[pv_cols[0]].notna().sum())
            n_total    = int(grp[w_col].notna().sum())

            rows.append({
                "program":      "PISA",
                "cycle":        spec["cycle"],
                "country_iso3": cnt_str,
                "domain":       domain,
                "mean":         round(mean_val, 4),
                "se":           round(se_val, 4) if not np.isnan(se_val) else None,
                "ci_lo":        round(mean_val - 1.96 * se_val, 4) if not np.isnan(se_val) else None,
                "ci_hi":        round(mean_val + 1.96 * se_val, 4) if not np.isnan(se_val) else None,
                "method":       "BRR_Fay",
                "n_total":      n_total,
                "n_analytic":   n_analytic,
                "n_replicates": len(repl_cols),
                "n_pv":         len(pv_cols),
            })

        log.info("  %d %s: %d ülke", spec["cycle"], domain, sum(1 for r in rows if r["cycle"] == spec["cycle"] and r["domain"] == domain))

    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_dfs = []

    for spec in CYCLE_CATALOG:
        if not spec["syntax"].exists():
            log.error("Syntax dosyası yok: %s", spec["syntax"])
            continue
        if not spec["data"].exists():
            log.error("Veri dosyası yok: %s", spec["data"])
            continue

        log.info("=== PISA %d ===", spec["cycle"])
        try:
            cdf = process_cycle(spec)
            log.info("  → %d satır", len(cdf))
            all_dfs.append(cdf)
        except Exception as e:
            log.error("  PISA %d başarısız: %s", spec["cycle"], e)

    if not all_dfs:
        log.error("Hiç veri üretilemedi.")
        return

    legacy = pd.concat(all_dfs, ignore_index=True)
    legacy.to_csv(OUT_LEGACY, index=False)
    log.info("Kaydedildi: %s  (%d satır)", OUT_LEGACY, len(legacy))

    # country_estimates.csv'ye ekle (duplikat yoksa)
    if ESTIMATES_CSV.exists():
        existing = pd.read_csv(ESTIMATES_CSV)
        existing["program"] = existing["program"].str.upper()
        existing["domain"]  = existing["domain"].str.lower()

        # Zaten PISA 2000-2012 varsa çıkar (idempotent)
        existing = existing[~(
            (existing["program"] == "PISA") &
            (existing["cycle"].isin([2000, 2003, 2006, 2009, 2012]))
        )]
        combined = pd.concat([existing, legacy], ignore_index=True)
        combined = combined.sort_values(["program", "cycle", "country_iso3", "domain"]).reset_index(drop=True)
        combined.to_csv(ESTIMATES_CSV, index=False)
        log.info("country_estimates.csv güncellendi: %d satır (önceki: %d)",
                 len(combined), len(existing))
    else:
        legacy.to_csv(ESTIMATES_CSV, index=False)
        log.info("Yeni country_estimates.csv oluşturuldu.")


if __name__ == "__main__":
    main()
