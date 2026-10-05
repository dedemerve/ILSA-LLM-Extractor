#!/usr/bin/env python3
"""
Audit presence of ILSA microdata required by the forecasting pipeline.

Run on the Mac that holds the data:

  export ILSA_MICRODATA_ROOT="/Users/mrved/Desktop/ILSA Datasets"
  python scripts/audit_ilsa_microdata_presence.py

Prints MISSING / PRESENT / PARTIAL for each required path.
Also writes outputs/stage5/microdata_presence_audit.csv when run from repo root.
"""
from __future__ import annotations

import csv
import os
from pathlib import Path

ROOT = Path(os.environ.get("ILSA_MICRODATA_ROOT", "")).expanduser()
if not ROOT.as_posix():
    ROOT = Path.home() / "Desktop" / "ILSA Datasets"

# (program, cycle, role, relative_path, check_type)
# check_type: file | dir | glob
CHECKS: list[tuple[str, str, str, str, str]] = [
    # PISA achievement (pipeline wired)
    ("PISA", "2015", "student_QQQ", "PISA Datasets/PISA 2015 Data/PUF_SPSS_COMBINED_CMB_STU_QQQ/CY6_MS_CMB_STU_QQQ.sav", "file"),
    ("PISA", "2018", "student_QQQ", "PISA Datasets/PISA 2018 Data/CY07_MSU_STU_QQQ.sav", "file"),
    ("PISA", "2018", "student_QQQ_alt_glob", "PISA Datasets/PISA 2018 Data", "glob:**/*STU_QQQ*.[Ss][Aa][Vv]"),
    ("PISA", "2022", "student_QQQ", "PISA Datasets/PISA 2022 Data/School questionnaire data file (CY08MSP_STU_QQQ).SAV", "file"),
    ("PISA", "2025", "student_PUF", "PISA Datasets/PISA 2025 Data/CY09_MS_STU_PUF.sav", "file"),
    # PISA early (covariates only / not in achievement panel)
    ("PISA", "2000", "folder", "PISA Datasets/PISA 2000 Data", "dir"),
    ("PISA", "2003", "TXT_stu", "PISA Datasets/PISA 2003 Data/INT_stui_2003_v2.txt", "file"),
    ("PISA", "2006", "folder", "PISA Datasets/PISA 2006 Data", "dir"),
    ("PISA", "2009", "TXT_stu", "PISA Datasets/PISA 2009 Data/INT_STQ09_DEC11.txt", "file"),
    ("PISA", "2012", "folder", "PISA Datasets/PISA 2012 Data", "dir"),
    # TIMSS G8
    ("TIMSS", "1995", "G8_Data", "TIMSS Datasets/TIMSS Data/TIMSS1995_IDB_SPSS_G8/Data", "dir"),
    ("TIMSS", "1999", "G8_Data", "TIMSS Datasets/TIMSS Data/TIMSS1999_IDB_SPSS_G8/Data", "dir"),
    ("TIMSS", "2003", "G8_Data", "TIMSS Datasets/TIMSS Data/TIMSS2003_IDB_SPSS_G8/Data", "dir"),
    ("TIMSS", "2007", "G8_Data", "TIMSS Datasets/TIMSS Data/TIMSS2007_IDB_SPSS_G8/Data", "dir"),
    ("TIMSS", "2011", "G8_Data", "TIMSS Datasets/TIMSS Data/TIMSS2011_IDB_SPSS_G8/Data", "dir"),
    ("TIMSS", "2015", "G8_Data", "TIMSS Datasets/TIMSS Data/TIMSS2015_IDB_SPSS_G8/Data", "dir"),
    ("TIMSS", "2019", "G8_Data", "TIMSS Datasets/TIMSS Data/TIMSS2019_IDB_SPSS_G8/Data", "dir"),
    ("TIMSS", "2023", "G8_Data", "TIMSS Datasets/TIMSS Data/TIMSS2023_IDB_SPSS_G8/2_Data Files/SPSS Data", "dir"),
    # TIMSS G4
    ("TIMSS_G4", "1995", "G4_Data", "TIMSS Datasets/TIMSS Data/TIMSS1995_IDB_SPSS_G4/Data", "dir"),
    ("TIMSS_G4", "2003", "G4_Data", "TIMSS Datasets/TIMSS Data/TIMSS2003_IDB_SPSS_G4/Data", "dir"),
    ("TIMSS_G4", "2007", "G4_Data", "TIMSS Datasets/TIMSS Data/TIMSS2007_IDB_SPSS_G4/Data", "dir"),
    ("TIMSS_G4", "2011", "G4_Data", "TIMSS Datasets/TIMSS Data/TIMSS2011_IDB_SPSS_G4/Data", "dir"),
    ("TIMSS_G4", "2015", "G4_Data", "TIMSS Datasets/TIMSS Data/TIMSS2015_IDB_SPSS_G4/Data", "dir"),
    ("TIMSS_G4", "2019", "G4_Data", "TIMSS Datasets/TIMSS Data/TIMSS2019_IDB_SPSS_G4/Data", "dir"),
    ("TIMSS_G4", "2023", "G4_Data", "TIMSS Datasets/TIMSS Data/TIMSS2023_IDB_SPSS_G4/2_Data Files/SPSS Data", "dir"),
    # PIRLS
    ("PIRLS", "2001", "Data", "PIRLS Datasets/PIRLS2001_IDB_SPSS/Data", "dir"),
    ("PIRLS", "2006", "Data", "PIRLS Datasets/PIRLS2006_IDB_SPSS/Data", "dir"),
    ("PIRLS", "2011", "Data", "PIRLS Datasets/PIRLS2011_IDB_SPSS/Data", "dir"),
    ("PIRLS", "2016", "Data", "PIRLS Datasets/PIRLS2016_IDB_SPSS/Data", "dir"),
    ("PIRLS", "2021", "Data", "PIRLS Datasets/PIRLS2021_IDB_SPSS/3_International Database/1_SPSS Data", "dir"),
    # ICCS
    ("ICCS", "2009", "Data_G8", "ICCS Datasets/ICCS2009_IDB_SPSS/Data_G8", "dir"),
    ("ICCS", "2016", "Data", "ICCS Datasets/ICCS2016_IDB_SPSS/Data", "dir"),
    ("ICCS", "2022", "Data", "ICCS Datasets/ICCS2022_IDB_SPSS/Data", "dir"),
    # ICILS
    ("ICILS", "2013", "Data", "ICILS Datasets/ICILS2013_IDB_SPSS/Data", "dir"),
    ("ICILS", "2018", "Data", "ICILS Datasets/ICILS2018_IDB_SPSS/Data", "dir"),
    ("ICILS", "2023", "Data", "ICILS Datasets/ICILS2023_IDB_SPSS/Data", "dir"),
    # PIAAC
    ("PIAAC", "2012", "Cycle1", "PIAAC Datasets/PIAAC Cycle1", "dir"),
    ("PIAAC", "2017", "Cycle2", "PIAAC Datasets/PIAAC Cycle2", "dir"),
    # TALIS
    ("TALIS", "2013", "Data", "TALIS Datasets/TALIS 2013 Data", "dir"),
    ("TALIS", "2018", "Data", "TALIS Datasets/TALIS 2018 Data", "dir"),
]


def check(rel: str, kind: str) -> tuple[str, str]:
    path = ROOT / rel
    if kind == "file":
        ok = path.is_file()
        return ("PRESENT" if ok else "MISSING", str(path))
    if kind == "dir":
        ok = path.is_dir()
        detail = ""
        if ok:
            n = sum(1 for _ in path.rglob("*") if _.is_file())
            detail = f" ({n} files under dir)"
        return ("PRESENT" if ok else "MISSING", str(path) + detail)
    if kind.startswith("glob:"):
        pattern = kind.split(":", 1)[1]
        base = ROOT / rel
        hits = list(base.glob(pattern)) if base.exists() else []
        # exclude Moscow-only QMC if international STU_QQQ sought
        intl = [h for h in hits if "QMC" not in h.name.upper() and "VNM" not in h.name.upper()]
        if intl:
            return ("PRESENT", "; ".join(str(h) for h in intl[:5]))
        if hits:
            return ("PARTIAL_ONLY_SPECIAL", "; ".join(str(h) for h in hits[:5]))
        return ("MISSING", str(base / pattern))
    return ("UNKNOWN", str(path))


def main() -> None:
    print(f"ILSA_MICRODATA_ROOT = {ROOT}")
    print(f"exists = {ROOT.exists()}\n")
    if not ROOT.exists():
        print("ROOT missing — set ILSA_MICRODATA_ROOT and re-run.")
        return

    rows = []
    for program, cycle, role, rel, kind in CHECKS:
        status, detail = check(rel, kind)
        rows.append({
            "program": program,
            "cycle": cycle,
            "role": role,
            "relative_path": rel,
            "status": status,
            "detail": detail,
        })
        mark = {"PRESENT": "OK", "MISSING": "XX", "PARTIAL_ONLY_SPECIAL": "!!"}.get(status, "??")
        print(f"[{mark}] {program:8} {cycle:4} {role:22} {status}")
        if status != "PRESENT":
            print(f"       → {detail}")

    missing = [r for r in rows if r["status"] != "PRESENT"]
    print(f"\n=== SUMMARY: {len(rows)-len(missing)} present / {len(missing)} missing-or-partial / {len(rows)} checks ===")

    out = Path("outputs/stage5/microdata_presence_audit.csv")
    if Path("outputs/stage5").exists():
        with out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
