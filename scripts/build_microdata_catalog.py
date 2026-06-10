#!/usr/bin/env python3
"""
ILSA Microdata Catalog Builder — Phase 1

Tarama kapsamı  : ~/Desktop/ILSA Datasets/ altındaki tüm veri dosyaları
Desteklenen formatlar:
  - .sav / .SAV     SPSS binary  → pyreadstat (metadata-only)
  - .sas7bdat       SAS binary   → pyreadstat (metadata-only)
  - .csv            CSV          → başlık satırı + satır sayısı
  - .txt (sabit genişlikli) → sütun bilgisi yok; companion .SPS/.SAS aranır
  - .SPS / .sps / .SAS / .sas / .do → SYNTAX dosyası, veri değil
  - diğer (.pdf, .xlsx, .docx …) → DOCUMENTATION, atlanır

Çıktı:
  outputs/ilsa_microdata_catalog/parquet/  → catalog_files.parquet
                                             catalog_variables.parquet
                                             catalog_countries.parquet
  outputs/ilsa_microdata_catalog/duckdb/   → ilsa_microdata_catalog.duckdb
  outputs/ilsa_microdata_catalog/schema_manifest.json  → güncellenir

Kullanım:
  python scripts/build_microdata_catalog.py
  python scripts/build_microdata_catalog.py --root "~/Desktop/ILSA Datasets"
  python scripts/build_microdata_catalog.py --resume   # önceki taramayı genişlet
  python scripts/build_microdata_catalog.py --isa PISA TIMSS   # filtreli
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import re
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import duckdb
import pandas as pd
import pyreadstat
from tqdm import tqdm

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT  = Path.home() / "Desktop" / "ILSA Datasets"
CATALOG_DIR   = PROJECT_ROOT / "outputs" / "ilsa_microdata_catalog"
PARQUET_DIR   = CATALOG_DIR / "parquet"
DUCKDB_DIR    = CATALOG_DIR / "duckdb"
MANIFEST_PATH = CATALOG_DIR / "schema_manifest.json"
ENGINE_VERSION = "1.0.0-phase1"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(PROJECT_ROOT / "outputs" / "build_microdata_catalog.log", mode="a"),
    ],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ISA tespiti: dizin yolundan ISA adını çıkar
# ---------------------------------------------------------------------------
_ISA_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bPISA\b",  re.I), "PISA"),
    (re.compile(r"\bTIMSS\b", re.I), "TIMSS"),
    (re.compile(r"\bPIRLS\b", re.I), "PIRLS"),
    (re.compile(r"\bICCS\b",  re.I), "ICCS"),
    (re.compile(r"\bICILS\b", re.I), "ICILS"),
    (re.compile(r"\bPIAAC\b", re.I), "PIAAC"),
    (re.compile(r"\bTALIS\b", re.I), "TALIS"),
]

def _detect_isa(path: Path) -> str:
    parts = " ".join(path.parts)
    for pat, name in _ISA_PATTERNS:
        if pat.search(parts):
            return name
    return "UNKNOWN"

def _detect_cycle(path: Path) -> int | None:
    for part in reversed(path.parts):
        m = re.search(r"\b(19[89]\d|20[012]\d)\b", part)
        if m:
            return int(m.group(1))
    return None

def _detect_file_role(name: str) -> str:
    n = name.upper()
    if any(k in n for k in ("STU", "STUDENT")):  return "student"
    if any(k in n for k in ("SCH", "SCHOOL")):   return "school"
    if any(k in n for k in ("TCH", "TEACHER")):  return "teacher"
    if any(k in n for k in ("PAR", "PARENT")):   return "parent"
    if any(k in n for k in ("COG", "COGNITIVE")): return "cognitive"
    if any(k in n for k in ("PRIN", "PRINCIPAL")): return "principal"
    return "other"

def _detect_grade(path: Path) -> str | None:
    parts = " ".join(path.parts).upper()
    if "GRADE 4" in parts or "G4" in parts:  return "4"
    if "GRADE 8" in parts or "G8" in parts:  return "8"
    if "15-YEAR" in parts or "AGE 15" in parts: return "15"
    if "ADULT" in parts:                     return "Adult"
    return None

def _file_id(path: Path) -> str:
    digest = hashlib.sha256(str(path).encode()).hexdigest()[:12]
    isa = _detect_isa(path)
    return f"{isa}_{digest}"

# ---------------------------------------------------------------------------
# ISA-spesifik psikometrik kurallar
# ---------------------------------------------------------------------------

# Plausible Value kalıpları — her ISA için
_PV_RULES: dict[str, list[tuple[re.Pattern, str, str]]] = {
    # (derleme deseni, domain, draw_group)
    "PISA": [
        (re.compile(r"^PV(\d+)MATH",  re.I), "math",    "PV_MATH"),
        (re.compile(r"^PV(\d+)READ",  re.I), "reading", "PV_READ"),
        (re.compile(r"^PV(\d+)SCIE",  re.I), "science", "PV_SCIE"),
        (re.compile(r"^PV(\d+)CPS",   re.I), "cps",     "PV_CPS"),
        (re.compile(r"^PV(\d+)CT",    re.I), "ct",      "PV_CT"),
        (re.compile(r"^PV(\d+)FIN",   re.I), "fin",     "PV_FIN"),
        (re.compile(r"^PV(\d+)GLO",   re.I), "global",  "PV_GLO"),
        (re.compile(r"^PV(\d+)CREA",  re.I), "creative","PV_CREA"),
        (re.compile(r"^PVMATH(\d+)",  re.I), "math",    "PV_MATH"),  # 2022 format
        (re.compile(r"^PVREAD(\d+)",  re.I), "reading", "PV_READ"),
        (re.compile(r"^PVSCIE(\d+)",  re.I), "science", "PV_SCIE"),
    ],
    "TIMSS": [
        (re.compile(r"^BSMMAT0[1-5]$", re.I), "math",    "PV_MATH_G8"),
        (re.compile(r"^BSSSCI0[1-5]$", re.I), "science", "PV_SCIE_G8"),
        (re.compile(r"^ASMMAT0[1-5]$", re.I), "math",    "PV_MATH_G4"),
        (re.compile(r"^ASSSCI0[1-5]$", re.I), "science", "PV_SCIE_G4"),
    ],
    "PIRLS": [
        (re.compile(r"^ASRREA0[1-5]$", re.I), "reading", "PV_READ_G4"),
    ],
    "ICCS": [
        (re.compile(r"^PV[1-5]CIV$",  re.I), "civic",   "PV_CIV"),
    ],
    "ICILS": [
        (re.compile(r"^PV[1-5]CIL$",  re.I), "cil",     "PV_CIL"),
        (re.compile(r"^PV[1-5]CT$",   re.I), "ct",      "PV_CT"),
    ],
    "PIAAC": [
        (re.compile(r"^PVLIT(\d+)$",  re.I), "literacy", "PV_LIT"),
        (re.compile(r"^PVNUM(\d+)$",  re.I), "numeracy", "PV_NUM"),
        (re.compile(r"^PVPSL(\d+)$",  re.I), "psl",      "PV_PSL"),
    ],
    "TALIS": [],  # TALIS PV kullanmaz
}

# Ağırlık değişken desenleri
_WEIGHT_RULES: dict[str, dict[str, re.Pattern]] = {
    "PISA":  {
        "primary":   re.compile(r"^W_FSTUWT$", re.I),
        # PISA 2000-2012: W_FSTR1-80  | PISA 2015+: W_FSTURWT1-80
        "replicate": re.compile(r"^W_F(STR|STURWT)\d+$", re.I),
    },
    "TIMSS": {
        "primary":   re.compile(r"^TOTWGT$", re.I),
        "replicate": re.compile(r"^(JKZONE|JKREP\d*)$", re.I),
    },
    "PIRLS": {
        "primary":   re.compile(r"^TOTWGT$", re.I),
        "replicate": re.compile(r"^(JKZONE|JKREP\d*)$", re.I),
    },
    "ICCS":  {
        "primary":   re.compile(r"^TOTWGT$", re.I),
        "replicate": re.compile(r"^(JKZONE|JKREP\d*)$", re.I),
    },
    "ICILS": {
        "primary":   re.compile(r"^TOTWGT$", re.I),
        "replicate": re.compile(r"^(JKZONE|JKREP\d*)$", re.I),
    },
    "PIAAC": {
        "primary":   re.compile(r"^SPFWT0$", re.I),
        "replicate": re.compile(r"^SPFWT\d+$", re.I),
    },
    "TALIS": {
        "primary":   re.compile(r"^(TCHWGT|SCHWGT|NEWWGT)$", re.I),
        "replicate": re.compile(r"^(JKZONE|JKREP\d*|TRWGT\d+)$", re.I),
    },
}

# Ülke kodu değişkenleri
_COUNTRY_VAR: dict[str, str] = {
    "PISA": "CNT",
    "TIMSS": "IDCNTRY",
    "PIRLS": "IDCNTRY",
    "ICCS": "IDCNTRY",
    "ICILS": "IDCNTRY",
    "PIAAC": "CNTRYID",
    "TALIS": "IDCNTRY",
}

def _classify_variable(
    var_name: str,
    var_label: str,
    isa: str,
) -> dict:
    """Bir değişkenin psikometrik rolünü belirle."""
    result = {
        "is_weight":           False,
        "is_replicate_weight": False,
        "is_pv":               False,
        "is_country_code":     False,
        "is_achievement_scale": False,
        "pv_domain":           None,
        "pv_draw_index":       None,
        "detection_rule_id":   "RULE_NONE",
    }

    # Ülke kodu
    expected_cnt = _COUNTRY_VAR.get(isa, "")
    if var_name.upper() in {"CNT", "IDCNTRY", "CNTRYID", "COUNTRY", expected_cnt.upper()}:
        result["is_country_code"] = True
        result["detection_rule_id"] = "RULE_GEO_01"
        return result

    # Ağırlık (primary)
    wgt = _WEIGHT_RULES.get(isa, _WEIGHT_RULES.get("TIMSS", {}))
    if wgt.get("primary") and wgt["primary"].match(var_name):
        result["is_weight"] = True
        result["detection_rule_id"] = "RULE_WGT_01"
        return result

    # Replicate weight
    if wgt.get("replicate") and wgt["replicate"].match(var_name):
        result["is_weight"] = True
        result["is_replicate_weight"] = True
        result["detection_rule_id"] = "RULE_WGT_02"
        return result

    # Plausible Value
    for pat, domain, rule_group in _PV_RULES.get(isa, []):
        m = pat.match(var_name)
        if m:
            result["is_pv"] = True
            result["pv_domain"] = domain
            try:
                result["pv_draw_index"] = int(m.group(1))
            except (IndexError, ValueError):
                result["pv_draw_index"] = None
            result["detection_rule_id"] = f"RULE_PV_{rule_group}"
            return result

    return result

# ---------------------------------------------------------------------------
# Format-spesifik okuyucular (metadata-only)
# ---------------------------------------------------------------------------

def _read_sav_meta(path: Path, isa: str) -> dict | None:
    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            _, meta = pyreadstat.read_sav(
                str(path), metadataonly=True,
                apply_value_formats=False, encoding=enc
            )
            break
        except Exception as exc:
            last_exc = exc
    else:
        return {"error": str(last_exc)}
    return {
        "n_rows":         meta.number_rows or 0,
        "n_vars":         meta.number_columns or 0,
        "var_names":      list(meta.column_names),
        "var_labels":     dict(zip(meta.column_names, meta.column_labels)),
        "var_types":      {n: "numeric" for n in meta.column_names},
        "spss_creation_date": None,
        "format":         "sav",
        "notes":          None,
    }

def _read_sas7bdat_meta(path: Path, isa: str) -> dict | None:
    try:
        _, meta = pyreadstat.read_sas7bdat(
            str(path), metadataonly=True
        )
        return {
            "n_rows":   meta.number_rows or 0,
            "n_vars":   meta.number_columns or 0,
            "var_names": list(meta.column_names),
            "var_labels": dict(zip(meta.column_names, meta.column_labels)),
            "var_types": {n: "numeric" for n in meta.column_names},
            "spss_creation_date": None,
            "format":   "sas7bdat",
            "notes":    None,
        }
    except Exception as exc:
        return {"error": str(exc)}

def _read_csv_meta(path: Path, isa: str) -> dict | None:
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f)
            header = next(reader, [])
            n_rows = sum(1 for _ in reader)
        return {
            "n_rows":    n_rows,
            "n_vars":    len(header),
            "var_names": header,
            "var_labels": {h: "" for h in header},
            "var_types": {h: "unknown" for h in header},
            "spss_creation_date": None,
            "format":    "csv",
            "notes":     None,
        }
    except Exception as exc:
        return {"error": str(exc)}

def _read_txt_meta(path: Path, isa: str) -> dict | None:
    """
    Sabit genişlikli TXT — sütun tanımları yok.
    Companion .SPS/.SAS dosyası aranır, varsa kaydedilir.
    """
    parent = path.parent
    companion = None
    for ext in (".sps", ".SPS", ".sas", ".SAS"):
        stem = path.stem
        cand = parent / (stem + ext)
        if cand.exists():
            companion = str(cand)
            break
    if companion is None:
        for f in parent.iterdir():
            if f.suffix.upper() in (".SPS", ".SAS"):
                companion = str(f)
                break
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            first = f.readline()
        n_cols_est = len(first.rstrip("\n")) if first else 0
        with open(path, encoding="utf-8", errors="replace") as f:
            n_rows = sum(1 for _ in f)
    except Exception as exc:
        return {"error": str(exc)}
    return {
        "n_rows":    n_rows,
        "n_vars":    0,
        "var_names": [],
        "var_labels": {},
        "var_types": {},
        "spss_creation_date": None,
        "format":    "txt_fixed_width",
        "notes":     (
            f"Fixed-width; sütun tanımı yok. "
            f"Companion syntax: {companion or 'bulunamadı'}. "
            f"Tahmini satır genişliği: {n_cols_est}."
        ),
    }

_FORMAT_READERS = {
    ".sav":      _read_sav_meta,
    ".SAV":      _read_sav_meta,
    ".sas7bdat": _read_sas7bdat_meta,
    ".csv":      _read_csv_meta,
    ".txt":      _read_txt_meta,
    ".TXT":      _read_txt_meta,
}

_SYNTAX_EXTENSIONS = {".sps", ".SPS", ".sas", ".SAS", ".do", ".DO", ".ado", ".ADO"}
_DOC_EXTENSIONS    = {".pdf", ".PDF", ".xlsx", ".xls", ".docx", ".doc",
                      ".DOC", ".lst", ".LST", ".url", ".md", ".zip", ".CDF", ".CDT",
                      ".cdt", ".cdf", ".sdb", ".SDB"}

def _classify_extension(path: Path) -> str:
    ext = path.suffix
    if ext in _FORMAT_READERS:      return "data"
    if ext in _SYNTAX_EXTENSIONS:   return "syntax"
    if ext in _DOC_EXTENSIONS:      return "documentation"
    return "unknown"

# ---------------------------------------------------------------------------
# Ana tarayıcı
# ---------------------------------------------------------------------------

def _iter_data_files(root: Path, filter_isa: list[str] | None) -> Iterator[Path]:
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if ".DS_Store" in str(p):
            continue
        if _classify_extension(p) != "data":
            continue
        if filter_isa:
            isa = _detect_isa(p)
            if isa not in filter_isa:
                continue
        yield p

def _build_rows(path: Path, existing_ids: set[str]) -> tuple[dict, list[dict], list[dict]] | None:
    fid = _file_id(path)
    if fid in existing_ids:
        return None

    isa   = _detect_isa(path)
    cycle = _detect_cycle(path)
    role  = _detect_file_role(path.stem)
    grade = _detect_grade(path)
    size  = path.stat().st_size
    ext   = path.suffix.lower()

    reader = _FORMAT_READERS.get(path.suffix)
    if reader is None:
        return None

    meta = reader(path, isa)
    if meta is None:
        return None

    if "error" in meta:
        log.warning("Okuma hatası %s: %s", path.name, meta["error"])
        file_row = {
            "file_id":              fid,
            "isa":                  isa,
            "cycle":                cycle,
            "cycle_label":          f"{isa} {cycle}" if cycle else None,
            "grade":                grade,
            "file_role":            role,
            "file_format":          path.suffix,
            "file_path":            str(path),
            "file_name":            path.name,
            "file_mask":            _make_mask(path.name),
            "file_size_bytes":      size,
            "n_rows":               0,
            "n_vars":               0,
            "spss_creation_date":   None,
            "catalog_build_version": ENGINE_VERSION,
            "catalog_built_at":     datetime.now(timezone.utc).isoformat(),
            "metadata_only_scan":   True,
            "has_replicate_weights": False,
            "primary_weight_var":   None,
            "country_var_candidate": _COUNTRY_VAR.get(isa),
            "n_pv_draws":           None,
            "weight_method":        None,
            "measurement_note":     None,
            "parse_error":          meta["error"],
            "notes":                f"PARSE_ERROR: {meta['error']}",
        }
        return file_row, [], []

    var_names   = meta["var_names"]
    var_labels  = meta["var_labels"]
    var_types   = meta["var_types"]

    # Değişken sınıflandırma
    var_rows: list[dict] = []
    primary_weight = None
    has_replicate  = False
    pv_counts: dict[str, set] = {}
    country_var    = None

    for vname in var_names:
        clf = _classify_variable(vname, var_labels.get(vname, ""), isa)
        vrow = {
            "variable_id":        f"{fid}::{vname}",
            "file_id":            fid,
            "variable_name":      vname,
            "variable_label":     (var_labels.get(vname) or "")[:512],
            "var_type":           var_types.get(vname, "unknown"),
            "measurement_level":  "unknown",
            "is_weight":          clf["is_weight"],
            "is_replicate_weight": clf["is_replicate_weight"],
            "is_pv":              clf["is_pv"],
            "is_country_code":    clf["is_country_code"],
            "is_achievement_scale": clf["is_achievement_scale"],
            "pv_domain":          clf["pv_domain"],
            "pv_draw_index":      clf["pv_draw_index"],
            "detection_rule_id":  clf["detection_rule_id"],
        }
        var_rows.append(vrow)

        if clf["is_weight"] and not clf["is_replicate_weight"]:
            primary_weight = vname
        if clf["is_replicate_weight"]:
            has_replicate = True
        if clf["is_pv"] and clf["pv_domain"]:
            pv_counts.setdefault(clf["pv_domain"], set()).add(clf["pv_draw_index"])
        if clf["is_country_code"]:
            country_var = vname

    # Ülke satırları (CSV'den sadece CNT kolonundan)
    country_rows: list[dict] = []
    if country_var and ext in (".csv",):
        try:
            df_cnt = pd.read_csv(
                path, usecols=[country_var],
                encoding="utf-8", on_bad_lines="skip", low_memory=False
            )
            for code, grp in df_cnt.groupby(country_var):
                country_rows.append({
                    "country_row_id":    f"{fid}::{code}",
                    "file_id":           fid,
                    "country_code":      str(code),
                    "country_code_system": "ISO3" if len(str(code)) == 3 else "OTHER",
                    "country_label":     str(code),
                    "n_unweighted":      int(len(grp)),
                    "is_analyzable":     len(grp) >= 30,
                })
        except Exception as exc:
            log.debug("Ülke satırları okunamadı %s: %s", path.name, exc)

    # Measurement invariance notu
    measure_note = None
    if isa == "PISA" and cycle and cycle < 2015:
        measure_note = (
            "PISA 2012 ve öncesi PV ölçekleri yeni cycle'larla doğrudan karşılaştırılamaz; "
            "OECD trend linking belgesi gereklidir."
        )
    elif isa == "TIMSS":
        measure_note = (
            "TIMSS Grade 4 ve Grade 8 ölçekleri ayrı latent boyutlardır; havuzlamayın."
        )

    n_pv_total = sum(len(v) for v in pv_counts.values())
    weight_method = None
    if has_replicate:
        if isa == "PISA":      weight_method = "BRR_Fay"
        elif isa in ("TIMSS", "PIRLS", "ICCS", "ICILS"): weight_method = "JRR"
        elif isa == "PIAAC":   weight_method = "BRR"
        elif isa == "TALIS":   weight_method = "JRR"

    file_row = {
        "file_id":              fid,
        "isa":                  isa,
        "cycle":                cycle,
        "cycle_label":          f"{isa} {cycle}" if cycle else None,
        "grade":                grade,
        "file_role":            role,
        "file_format":          path.suffix,
        "file_path":            str(path),
        "file_name":            path.name,
        "file_mask":            _make_mask(path.name),
        "file_size_bytes":      size,
        "n_rows":               meta["n_rows"],
        "n_vars":               meta["n_vars"],
        "spss_creation_date":   meta.get("spss_creation_date"),
        "catalog_build_version": ENGINE_VERSION,
        "catalog_built_at":     datetime.now(timezone.utc).isoformat(),
        "metadata_only_scan":   True,
        "has_replicate_weights": has_replicate,
        "primary_weight_var":   primary_weight,
        "country_var_candidate": country_var or _COUNTRY_VAR.get(isa),
        "n_pv_draws":           n_pv_total if n_pv_total > 0 else None,
        "weight_method":        weight_method,
        "measurement_note":     measure_note,
        "parse_error":          None,
        "notes":                meta.get("notes"),
    }

    return file_row, var_rows, country_rows


def _make_mask(name: str) -> str:
    return re.sub(r"\d{4}", "*", name)

# ---------------------------------------------------------------------------
# Parquet + DuckDB yazma
# ---------------------------------------------------------------------------

def _save_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)

def _write_duckdb(
    files_df: pd.DataFrame,
    vars_df:  pd.DataFrame,
    cnts_df:  pd.DataFrame,
) -> None:
    db_path = DUCKDB_DIR / "ilsa_microdata_catalog.duckdb"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))
    for table, df in [
        ("catalog_files",     files_df),
        ("catalog_variables", vars_df),
        ("catalog_countries", cnts_df),
    ]:
        con.execute(f"DROP TABLE IF EXISTS {table}")
        con.execute(f"CREATE TABLE {table} AS SELECT * FROM df")
    # Özet görünüm
    con.execute("""
        CREATE OR REPLACE VIEW v_catalog_file_summary AS
        SELECT
            f.file_id, f.isa, f.cycle, f.grade, f.file_role, f.file_format,
            f.file_path, f.n_rows, f.n_vars,
            f.primary_weight_var, f.weight_method,
            f.country_var_candidate, f.n_pv_draws,
            f.measurement_note, f.parse_error,
            COUNT(DISTINCT c.country_code) AS n_countries_available,
            SUM(CASE WHEN v.is_pv    THEN 1 ELSE 0 END) AS n_pv_vars,
            SUM(CASE WHEN v.is_weight THEN 1 ELSE 0 END) AS n_weight_vars
        FROM catalog_files f
        LEFT JOIN catalog_variables v ON f.file_id = v.file_id
        LEFT JOIN catalog_countries c ON f.file_id = c.file_id
        GROUP BY ALL
    """)
    con.close()
    log.info("DuckDB yazıldı: %s", db_path)

def _update_manifest(n_files: int, n_vars: int, n_countries: int) -> None:
    manifest = {
        "schema_version":   ENGINE_VERSION,
        "status":           "PHASE1_POPULATED",
        "built_at":         datetime.now(timezone.utc).isoformat(),
        "n_files_cataloged": n_files,
        "n_variables_cataloged": n_vars,
        "n_country_rows":   n_countries,
        "storage_layout": {
            "parquet_dir":  str(PARQUET_DIR.relative_to(PROJECT_ROOT)),
            "duckdb_path":  str((DUCKDB_DIR / "ilsa_microdata_catalog.duckdb").relative_to(PROJECT_ROOT)),
            "tables":       ["catalog_files", "catalog_variables", "catalog_countries"],
        },
        "build_policy": {
            "metadata_only_scan":    True,
            "load_full_data_into_ram": False,
            "raw_data_root":         str(DEFAULT_ROOT),
        },
        "formats_supported": list(_FORMAT_READERS.keys()),
        "specification_document": "docs/phase0_methodological_specification.md",
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    log.info("Manifest güncellendi: %s", MANIFEST_PATH)

# ---------------------------------------------------------------------------
# Giriş noktası
# ---------------------------------------------------------------------------

def build_catalog(
    root: Path,
    filter_isa: list[str] | None = None,
    resume: bool = False,
) -> None:
    log.info("=== ILSA Microdata Catalog Builder %s ===", ENGINE_VERSION)
    log.info("Kök dizin: %s", root)

    existing_ids: set[str] = set()
    old_files: list[dict] = []
    old_vars:  list[dict] = []
    old_cnts:  list[dict] = []

    if resume:
        fp = PARQUET_DIR / "catalog_files.parquet"
        vp = PARQUET_DIR / "catalog_variables.parquet"
        cp = PARQUET_DIR / "catalog_countries.parquet"
        if fp.exists():
            df_old = pd.read_parquet(fp)
            existing_ids = set(df_old["file_id"].tolist())
            old_files = df_old.to_dict("records")
            log.info("Resume: %d mevcut dosya yüklendi.", len(existing_ids))
        if vp.exists():
            old_vars = pd.read_parquet(vp).to_dict("records")
        if cp.exists():
            old_cnts = pd.read_parquet(cp).to_dict("records")

    all_paths = list(_iter_data_files(root, filter_isa))
    log.info("Taranacak dosya sayısı: %d", len(all_paths))

    new_files: list[dict] = []
    new_vars:  list[dict] = []
    new_cnts:  list[dict] = []

    errors = 0
    for path in tqdm(all_paths, desc="Taranıyor", unit="dosya"):
        result = _build_rows(path, existing_ids)
        if result is None:
            continue
        frow, vrows, crows = result
        new_files.append(frow)
        new_vars.extend(vrows)
        new_cnts.extend(crows)
        if frow.get("parse_error"):
            errors += 1

    all_files = old_files + new_files
    all_vars  = old_vars  + new_vars
    all_cnts  = old_cnts  + new_cnts

    files_df = pd.DataFrame(all_files)
    vars_df  = pd.DataFrame(all_vars)  if all_vars  else pd.DataFrame()
    cnts_df  = pd.DataFrame(all_cnts)  if all_cnts  else pd.DataFrame()

    _save_parquet(files_df, PARQUET_DIR / "catalog_files.parquet")
    _save_parquet(vars_df,  PARQUET_DIR / "catalog_variables.parquet")
    _save_parquet(cnts_df,  PARQUET_DIR / "catalog_countries.parquet")

    _write_duckdb(files_df, vars_df, cnts_df)
    _update_manifest(len(all_files), len(all_vars), len(all_cnts))

    log.info("Tamamlandı: %d dosya, %d değişken, %d ülke satırı, %d hata",
             len(all_files), len(all_vars), len(all_cnts), errors)

    # Özet rapor
    if not files_df.empty and "isa" in files_df.columns:
        print("\n--- ISA × Format Özeti ---")
        print(files_df.groupby(["isa", "file_format"])["file_id"].count().to_string())
        print(f"\nToplam: {len(all_files)} dosya | {len(all_vars)} değişken | {errors} hata")

    # Pilot claim kontrolü
    if not files_df.empty:
        pisa15 = files_df[
            (files_df["isa"] == "PISA") &
            (files_df["cycle"] == 2015) &
            (files_df["file_role"] == "student")
        ]
        if not pisa15.empty:
            fid = pisa15.iloc[0]["file_id"]
            log.info("Pilot claim için PISA 2015 student dosyası bulundu: %s", fid)
            print(f"\nPilot claim hazır → catalog_file_id: {fid}")
            print("Sheet 5'teki CLM_PISA_2012_000001 satırını bu file_id ile güncelleyiniz.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",   type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--isa",    nargs="+",
                        choices=["PISA","TIMSS","PIRLS","ICCS","ICILS","PIAAC","TALIS"])
    parser.add_argument("--resume", action="store_true",
                        help="Mevcut kataloga yeni dosyaları ekle, eskini koru.")
    args = parser.parse_args()
    build_catalog(args.root.expanduser(), filter_isa=args.isa, resume=args.resume)


if __name__ == "__main__":
    main()
