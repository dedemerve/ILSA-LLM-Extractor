#!/usr/bin/env python3
"""
TIMSS 2023 (G8 + G4) country-level estimates extraction.

Mevcut enriched_panel.csv ve forward_predictions.csv dosyalarını 2023 verileriyle günceller:
  1. TIMSS/TIMSS_G4 2023 country estimates → outputs/stage4/country_estimates.csv'ye eklenir
  2. outputs/stage5/enriched_panel.csv'de TIMSS 2023 satırları eklenir
  3. LOCO yeniden çalışır (TIMSS G8: +1 fold → toplam 6; G4: +1 fold → toplam 5)
  4. forward_predictions.csv güncellenir (predicted_cycle: 2023→2027, 2024→2028)
"""
from __future__ import annotations

import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import build_country_estimates as bce
import build_enriched_panel_v2 as bep
import run_loco_forecasting as loco

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

STAGE4_DIR = PROJECT_ROOT / "outputs" / "stage4"
STAGE5_DIR = PROJECT_ROOT / "outputs" / "stage5"


# ---------------------------------------------------------------------------
# Step 1: Extract TIMSS 2023 estimates from microdata
# ---------------------------------------------------------------------------
def extract_timss2023_estimates() -> pd.DataFrame:
    log.info("=== Step 1: TIMSS 2023 microdata extraction ===")
    timss2023_specs = [
        s for s in bce.MULTI_CATALOG
        if s.program in ("TIMSS", "TIMSS_G4") and s.cycle == 2023
    ]
    log.info("Processing %d catalog entries for TIMSS/TIMSS_G4 2023", len(timss2023_specs))
    rows = []
    for spec in timss2023_specs:
        log.info("  %s %d %s — %s", spec.program, spec.cycle, spec.domain, spec.data_dir)
        try:
            r = bce.process_cycle_multi(spec)
            log.info("    → %d countries extracted", len(r))
            rows.extend(r)
        except Exception as e:
            log.error("  FAILED: %s", e)
    if not rows:
        log.error("No rows extracted from TIMSS 2023 — check data paths")
        return pd.DataFrame()
    df_new = pd.DataFrame(rows)
    for col in ("mean", "se", "ci_lo", "ci_hi"):
        if col in df_new.columns:
            df_new[col] = df_new[col].round(4)
    log.info("Extracted %d rows for TIMSS 2023", len(df_new))
    return df_new


# ---------------------------------------------------------------------------
# Step 2: Merge into existing country_estimates.csv
# ---------------------------------------------------------------------------
def update_country_estimates(df_new: pd.DataFrame) -> pd.DataFrame:
    log.info("=== Step 2: Merge into country_estimates.csv ===")
    est_path = STAGE4_DIR / "country_estimates.csv"
    df_existing = pd.read_csv(est_path)
    log.info("Existing rows: %d", len(df_existing))

    # Drop any stale 2023 rows
    mask_old = (df_existing["program"].isin(["TIMSS", "TIMSS_G4"])) & (df_existing["cycle"] == 2023)
    df_existing = df_existing[~mask_old]
    log.info("After removing stale 2023: %d rows", len(df_existing))

    df_merged = pd.concat([df_existing, df_new], ignore_index=True)
    df_merged = df_merged.sort_values(["program", "domain", "cycle", "country_iso3"])
    df_merged.to_csv(est_path, index=False)
    log.info("Saved %d rows → %s", len(df_merged), est_path.name)
    return df_merged


# ---------------------------------------------------------------------------
# Step 3: Rebuild enriched_panel.csv for TIMSS and TIMSS_G4
# ---------------------------------------------------------------------------
def update_enriched_panel() -> pd.DataFrame:
    log.info("=== Step 3: Rebuild TIMSS enriched panel ===")
    panel_path = STAGE5_DIR / "enriched_panel.csv"
    df_panel = pd.read_csv(panel_path)
    log.info("Existing panel rows: %d", len(df_panel))

    # Remove stale TIMSS 2023 rows (if any)
    mask_stale = (df_panel["program"].isin(["TIMSS", "TIMSS_G4"])) & (df_panel["cycle"] == 2023)
    df_panel = df_panel[~mask_stale]
    log.info("After removing stale TIMSS 2023 from panel: %d rows", len(df_panel))

    # Call the build_enriched_panel_v2 rebuild for TIMSS/TIMSS_G4
    try:
        df_timss_new = bep.build_timss_panel()
        log.info("New TIMSS panel rows (all cycles): %d", len(df_timss_new))

        # Replace all TIMSS/TIMSS_G4 rows in panel with freshly rebuilt ones
        df_panel_non_timss = df_panel[~df_panel["program"].isin(["TIMSS", "TIMSS_G4"])]
        df_panel_updated = pd.concat([df_panel_non_timss, df_timss_new], ignore_index=True)
        df_panel_updated = df_panel_updated.sort_values(["program", "domain", "country_iso3", "cycle"])
        df_panel_updated.to_csv(panel_path, index=False)
        log.info("Saved updated panel: %d rows → %s", len(df_panel_updated), panel_path.name)
        return df_panel_updated
    except Exception as e:
        log.warning("build_enriched_panel_v2.build_timss_panel() not available: %s", e)
        log.info("Falling back: appending 2023 rows directly from country_estimates")
        return _append_timss2023_to_panel(df_panel)


def _append_timss2023_to_panel(df_panel: pd.DataFrame) -> pd.DataFrame:
    """Fallback: derive 2023 panel rows from country_estimates.csv directly."""
    panel_path = STAGE5_DIR / "enriched_panel.csv"
    est_path   = STAGE4_DIR / "country_estimates.csv"
    df_est = pd.read_csv(est_path)

    # Get 2019 covariate rows for lag features at 2023
    timss_covs = df_est[
        (df_est["program"].isin(["TIMSS", "TIMSS_G4"])) &
        (df_est["domain"].isin(["PARENTAL_EDU", "SES_COMPOSITE", "PARENTAL_EDUCATION"]))
    ]

    new_rows = []
    for prog in ["TIMSS", "TIMSS_G4"]:
        for dom in ["mathematics", "science"]:
            df_2023 = df_est[
                (df_est["program"] == prog) & (df_est["cycle"] == 2023) & (df_est["domain"] == dom)
            ][["country_iso3", "mean"]].rename(columns={"mean": f"{prog}_{dom}"})

            df_2019 = df_est[
                (df_est["program"] == prog) & (df_est["cycle"] == 2019) & (df_est["domain"] == dom)
            ][["country_iso3", "mean"]].rename(columns={"mean": f"lag_{prog}_{dom}"})

            df_cov = timss_covs[
                (timss_covs["program"] == prog) & (timss_covs["cycle"] == 2019)
            ].pivot_table(index="country_iso3", columns="domain", values="mean").reset_index()

            df_row = df_2023.merge(df_2019, on="country_iso3", how="left")
            df_row = df_row.merge(df_cov, on="country_iso3", how="left")
            df_row["cycle"]   = 2023
            df_row["program"] = prog
            df_row["domain"]  = dom

            # Align to existing panel columns
            for col in df_panel.columns:
                if col not in df_row.columns:
                    df_row[col] = np.nan
            new_rows.append(df_row[df_panel.columns])

    if new_rows:
        df_new = pd.concat(new_rows, ignore_index=True)
        df_updated = pd.concat([df_panel, df_new], ignore_index=True)
        df_updated = df_updated.sort_values(["program", "domain", "country_iso3", "cycle"])
        df_updated.to_csv(panel_path, index=False)
        log.info("Appended %d rows → panel now %d rows", len(df_new), len(df_updated))
        return df_updated
    return df_panel


# ---------------------------------------------------------------------------
# Step 4: Re-run LOCO for TIMSS and TIMSS_G4
# ---------------------------------------------------------------------------
def rerun_timss_loco():
    log.info("=== Step 4: Re-run TIMSS LOCO ===")
    try:
        # run_loco_forecasting has a main() or run_loco() we can call
        from run_loco_forecasting import run_loco_for_program
        for prog in ["TIMSS", "TIMSS_G4"]:
            for dom in ["mathematics", "science"]:
                log.info("  LOCO %s %s", prog, dom)
                run_loco_for_program(prog, dom)
    except (ImportError, AttributeError):
        log.info("Calling run_loco_forecasting as subprocess")
        import subprocess
        result = subprocess.run(
            [sys.executable,
             str(PROJECT_ROOT / "scripts" / "run_loco_forecasting.py"),
             "--programs", "TIMSS", "TIMSS_G4"],
            capture_output=True, text=True, cwd=str(PROJECT_ROOT)
        )
        print(result.stdout[-3000:] if result.stdout else "")
        if result.returncode != 0:
            log.error("LOCO subprocess failed: %s", result.stderr[-1000:])
        else:
            log.info("LOCO subprocess completed successfully")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    df_new = extract_timss2023_estimates()
    if df_new.empty:
        log.error("Extraction failed — aborting")
        return

    print("\n=== TIMSS 2023 Extraction Summary ===")
    summary = df_new.groupby(["program", "domain"])["country_iso3"].count().reset_index()
    summary.columns = ["program", "domain", "n_countries"]
    print(summary.to_string(index=False))
    print("\nSample (first 10 rows):")
    print(df_new[["program", "cycle", "domain", "country_iso3", "mean", "se"]].head(10).to_string(index=False))

    update_country_estimates(df_new)
    update_enriched_panel()
    rerun_timss_loco()

    log.info("=== TIMSS 2023 pipeline complete ===")


if __name__ == "__main__":
    main()
