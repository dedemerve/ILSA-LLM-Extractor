#!/usr/bin/env python3
"""
ICILS enriched panel builder.

icils_piaac_estimates.csv → icils_enriched_panel.csv

Wide format: (country_iso3, cycle) × {ICILS_computer_literacy, lag_SES_COMPOSITE,
                                        lag_PARENTAL_EDUCATION, lag_HOME_LITERACY_ACTIVITIES}

LOCO design for ICILS:
  Cycles: 2013, 2018, 2023
  test=2018 → train pairs: (2013→2018)
  test=2023 → train pairs: (2013→2018, 2018→2023)
  → 2 eligible LOCO folds

Outputs: outputs/stage5/icils_enriched_panel.csv
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STAGE5_DIR   = PROJECT_ROOT / "outputs" / "stage5"

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

ICILS_COVARIATES = ["SES_COMPOSITE", "PARENTAL_EDUCATION", "HOME_LITERACY_ACTIVITIES"]


def build_icils_panel() -> pd.DataFrame:
    est = pd.read_csv(STAGE5_DIR / "icils_piaac_estimates.csv")
    icils = est[est["program"] == "ICILS"].copy()
    icils["cycle"] = icils["cycle"].astype(int)

    # --- Outcome: ICILS_computer_literacy ---
    outcome = icils[icils["domain"] == "computer_literacy"][
        ["country_iso3", "cycle", "mean"]
    ].rename(columns={"mean": "ICILS_computer_literacy"})

    cycles = sorted(icils["cycle"].unique())
    log.info("ICILS cycles: %s", cycles)

    # --- Build wide panel (one row per country×cycle) ---
    panel = outcome.copy()

    # --- Lag covariates ---
    # lag_X at (country, cycle=c) = X at (country, cycle=prev_c)
    cycle_pairs = [(cycles[i], cycles[i+1]) for i in range(len(cycles) - 1)]

    for cov_domain in ICILS_COVARIATES:
        cov_df = icils[icils["domain"] == cov_domain][
            ["country_iso3", "cycle", "mean"]
        ].rename(columns={"mean": cov_domain})

        lag_map: dict[tuple, float] = {}
        for _, row in cov_df.iterrows():
            country = row["country_iso3"]
            cyc     = int(row["cycle"])
            # Find next cycle where this country appears
            for prev_c, next_c in cycle_pairs:
                if cyc == prev_c:
                    lag_map[(country, next_c)] = row[cov_domain]

        lag_col = f"lag_{cov_domain}"
        panel[lag_col] = panel.apply(
            lambda r: lag_map.get((r["country_iso3"], int(r["cycle"])), np.nan),
            axis=1,
        )
        n_filled = panel[lag_col].notna().sum()
        log.info("  lag_%s: %d non-NaN values", cov_domain, n_filled)

    panel["program"] = "ICILS"
    panel["domain"]  = "computer_literacy"

    out = STAGE5_DIR / "icils_enriched_panel.csv"
    panel.to_csv(out, index=False)
    log.info("Kaydedildi: %s  (%d satır, %d columns)", out, len(panel), panel.shape[1])

    # Summary
    print("\n=== ICILS Panel Summary ===")
    print(f"Rows: {len(panel)}")
    print(f"Cycles: {sorted(panel['cycle'].unique())}")
    print(f"Countries per cycle:")
    for c in sorted(panel["cycle"].unique()):
        n = panel[panel["cycle"]==c]["country_iso3"].nunique()
        print(f"  {c}: {n} countries")
    print(f"\nNon-NaN by column:")
    for col in panel.columns:
        if col not in ("country_iso3","cycle","program","domain"):
            print(f"  {col}: {panel[col].notna().sum()}")
    return panel


if __name__ == "__main__":
    build_icils_panel()
