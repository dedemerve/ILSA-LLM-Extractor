#!/usr/bin/env python3
"""
Extract PISA 2018 country achievement + covariates from local CY07_MSU_STU_QQQ.sav
and merge into stage4/stage5 outputs, replacing any PUBLISHED_WB rows for 2018.

Requires:
  export ILSA_MICRODATA_ROOT=...
  .../PISA Datasets/PISA 2018 Data/CY07_MSU_STU_QQQ.sav
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts import build_country_estimates as bce
from scripts import build_covariate_estimates as bcove
from scripts.ilsa_common import microdata_root

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

EST_PATH = PROJECT_ROOT / "outputs" / "stage4" / "country_estimates.csv"
COV_PATH = PROJECT_ROOT / "outputs" / "stage5" / "covariate_estimates.csv"


def _replace_cycle(existing: pd.DataFrame, new_rows: pd.DataFrame, cycle: int) -> pd.DataFrame:
    keep = existing[~((existing["program"] == "PISA") & (existing["cycle"] == cycle))]
    return pd.concat([keep, new_rows], ignore_index=True)


def main() -> None:
    root = microdata_root()
    sav = root / "PISA Datasets" / "PISA 2018 Data" / "CY07_MSU_STU_QQQ.sav"
    if not sav.is_file():
        raise SystemExit(f"Missing SAV: {sav}")

    specs = [s for s in bce.CATALOG if s.program == "PISA" and s.cycle == 2018]
    if not specs:
        raise SystemExit("No PISA 2018 CycleSpec in catalog")

    rows: list[dict] = []
    for spec in specs:
        log.info("Processing %s %s %s", spec.program, spec.cycle, spec.domain)
        rows.extend(bce.process_cycle(spec))
    if not rows:
        raise SystemExit("No 2018 achievement rows produced")

    new_est = pd.DataFrame(rows)
    est = pd.read_csv(EST_PATH)
    merged = _replace_cycle(est, new_est, 2018)
    merged.to_csv(EST_PATH, index=False)
    log.info("Updated %s with %d PISA 2018 achievement rows", EST_PATH, len(new_est))

    cov_specs = [s for s in bcove.SPECS if s.program == "PISA" and s.cycle == 2018]
    cov_rows: list[dict] = []
    for spec in cov_specs:
        cov_rows.extend(bcove.process_pisa_spec(spec))
    if cov_rows:
        new_cov = pd.DataFrame(cov_rows)
        if COV_PATH.exists():
            cov = pd.read_csv(COV_PATH)
            cov = _replace_cycle(cov, new_cov, 2018)
        else:
            cov = new_cov
        cov.to_csv(COV_PATH, index=False)
        log.info("Updated %s with %d PISA 2018 covariate rows", COV_PATH, len(new_cov))
    else:
        log.warning("No covariate rows for PISA 2018")


if __name__ == "__main__":
    main()
