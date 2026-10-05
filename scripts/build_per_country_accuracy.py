#!/usr/bin/env python3
"""Rebuild outputs/stage5/per_country_accuracy.csv from fold-level predictions.

Sources (in priority order):
  - loco_predictions.csv          → PISA / TIMSS / TIMSS_G4 / PIRLS
  - iccs_loco_predictions.csv     → ICCS (if present)
  - icils_holdout_predictions.csv → ICILS hold-out (if present)

PIAAC is intentionally omitted: in-sample diagnostic only.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STAGE5 = PROJECT_ROOT / "outputs" / "stage5"
OUT = STAGE5 / "per_country_accuracy.csv"

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def _agg_country(df: pd.DataFrame, y_true: str, y_m0: str, y_m1: str) -> pd.DataFrame:
    rows = []
    for (prog, dom, cnt), g in df.groupby(["program", "domain", "country_iso3"]):
        yt = g[y_true].astype(float)
        e0 = yt - g[y_m0].astype(float)
        e1 = yt - g[y_m1].astype(float)
        rows.append({
            "program": prog,
            "domain": dom,
            "country_iso3": cnt,
            "n_folds": int(len(g)),
            "MAE_M0": round(float(np.mean(np.abs(e0))), 2),
            "MAE_M1": round(float(np.mean(np.abs(e1))), 2),
            "RMSE_M0": round(float(np.sqrt(np.mean(e0 ** 2))), 2),
            "RMSE_M1": round(float(np.sqrt(np.mean(e1 ** 2))), 2),
        })
    return pd.DataFrame(rows)


def main() -> None:
    parts: list[pd.DataFrame] = []

    loco_path = STAGE5 / "loco_predictions.csv"
    if loco_path.exists():
        loco = pd.read_csv(loco_path)
        # Ridge folds only: require non-null y_M0
        loco = loco.dropna(subset=["y_M0", "y_M1", "y_true"])
        parts.append(_agg_country(loco, "y_true", "y_M0", "y_M1"))
        log.info("LOCO countries: %d", len(parts[-1]))

    iccs_path = STAGE5 / "iccs_loco_predictions.csv"
    if iccs_path.exists() and iccs_path.stat().st_size > 1:
        iccs = pd.read_csv(iccs_path)
        # Accept either y_M0/y_M1 or MAE-ready columns
        cols = set(iccs.columns)
        if {"y_true", "y_M0", "y_M1"}.issubset(cols):
            iccs = iccs.dropna(subset=["y_M0", "y_M1", "y_true"])
            parts.append(_agg_country(iccs, "y_true", "y_M0", "y_M1"))
            log.info("ICCS countries: %d", len(parts[-1]))
        else:
            log.warning("ICCS predictions present but missing y_true/y_M0/y_M1 columns")

    icils_path = STAGE5 / "icils_holdout_predictions.csv"
    if icils_path.exists() and icils_path.stat().st_size > 1:
        icils = pd.read_csv(icils_path)
        icils = icils.dropna(subset=["y_M0", "y_M1", "y_true"])
        parts.append(_agg_country(icils, "y_true", "y_M0", "y_M1"))
        log.info("ICILS countries: %d", len(parts[-1]))
    else:
        log.warning(
            "ICILS hold-out predictions missing — re-run "
            "scripts/run_icils_piaac_holdout.py to generate them"
        )

    for domain in ("literacy", "numeracy"):
        ppath = STAGE5 / f"piaac_holdout_predictions_{domain}.csv"
        if ppath.exists() and ppath.stat().st_size > 1:
            piaac = pd.read_csv(ppath)
            piaac = piaac.dropna(subset=["y_M0", "y_M1", "y_true"])
            parts.append(_agg_country(piaac, "y_true", "y_M0", "y_M1"))
            log.info("PIAAC %s countries: %d", domain, len(parts[-1]))

    if not parts:
        raise SystemExit("No prediction sources found under outputs/stage5/")

    out = pd.concat(parts, ignore_index=True)
    out = out.sort_values(["program", "domain", "country_iso3"]).reset_index(drop=True)
    out.to_csv(OUT, index=False)
    log.info("Wrote %s (%d rows)", OUT, len(out))
    print(out.groupby("program").size().to_string())


if __name__ == "__main__":
    main()
