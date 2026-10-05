#!/usr/bin/env python3
"""
Ingest published PISA country means (World Bank EdStats) for cycles missing
from stage4 country_estimates — used to unlock expanding-window Ridge when
local microdata SAV files are not yet available.

Source: World Bank indicators
  LO.PISA.MAT / LO.PISA.REA / LO.PISA.SCI

Does NOT overwrite existing BRR/JRR microdata estimates for the same
(program, cycle, domain, country). method label: PUBLISHED_WB.
"""
from __future__ import annotations

import json
import logging
import urllib.request
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EST_PATH = PROJECT_ROOT / "outputs" / "stage4" / "country_estimates.csv"
OUT_PUB = PROJECT_ROOT / "outputs" / "stage5" / "pisa_published_means_wb.csv"

INDICATORS = {
    "LO.PISA.MAT": "mathematics",
    "LO.PISA.REA": "reading",
    "LO.PISA.SCI": "science",
}

# Cycles to fill when microdata estimates are absent
TARGET_CYCLES = {2000, 2003, 2006, 2009, 2012, 2018}

# Non-country aggregates to drop
DROP_ISO3 = {
    "",
    "OED",
    "WLD",
    "HIC",
    "UMC",
    "LMC",
    "LIC",
    "EAS",
    "ECS",
    "LCN",
    "MEA",
    "NAC",
    "SAS",
    "SSF",
    "EUU",
    "TEA",
    "TEC",
    "TLA",
    "TMN",
    "TSA",
    "TSS",
    "CEB",
    "EAR",
    "EMU",
    "PSS",
    "PST",
}


def fetch_indicator(code: str) -> list[dict]:
    url = (
        f"https://api.worldbank.org/v2/country/all/indicator/{code}"
        f"?format=json&per_page=20000&date=2000:2018"
    )
    log.info("GET %s", code)
    with urllib.request.urlopen(url, timeout=120) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    if not isinstance(payload, list) or len(payload) < 2:
        raise RuntimeError(f"Unexpected WB response for {code}")
    return payload[1]


def build_published_frame() -> pd.DataFrame:
    rows: list[dict] = []
    for code, domain in INDICATORS.items():
        for r in fetch_indicator(code):
            if r.get("value") is None:
                continue
            iso3 = (r.get("countryiso3code") or "").strip().upper()
            if not iso3 or iso3 in DROP_ISO3 or len(iso3) != 3:
                continue
            cycle = int(r["date"])
            if cycle not in TARGET_CYCLES:
                continue
            rows.append(
                {
                    "program": "PISA",
                    "cycle": cycle,
                    "domain": domain,
                    "country_iso3": iso3,
                    "method": "PUBLISHED_WB",
                    "mean": round(float(r["value"]), 4),
                    "se": None,
                    "ci_lo": None,
                    "ci_hi": None,
                    "T": None,
                    "U_bar": None,
                    "B": None,
                    "m_pv": None,
                    "n_total": None,
                    "n_analytic": None,
                    "n_replicates": None,
                    "source_indicator": code,
                }
            )
    df = pd.DataFrame(rows)
    df = df.drop_duplicates(
        subset=["program", "cycle", "domain", "country_iso3"], keep="last"
    )
    return df.sort_values(["cycle", "domain", "country_iso3"]).reset_index(drop=True)


def merge_into_estimates(pub: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    est = pd.read_csv(EST_PATH)
    key = ["program", "cycle", "domain", "country_iso3"]
    est["_k"] = list(zip(est["program"], est["cycle"], est["domain"], est["country_iso3"]))
    existing = set(est["_k"])
    add = pub.copy()
    add["_k"] = list(
        zip(add["program"], add["cycle"], add["domain"], add["country_iso3"])
    )
    add = add[~add["_k"].isin(existing)].drop(columns=["_k", "source_indicator"], errors="ignore")
    est = est.drop(columns=["_k"])
    merged = pd.concat([est, add], ignore_index=True)
    merged = merged.sort_values(["program", "cycle", "domain", "country_iso3"]).reset_index(
        drop=True
    )
    return merged, len(add)


def main() -> None:
    pub = build_published_frame()
    OUT_PUB.parent.mkdir(parents=True, exist_ok=True)
    pub.to_csv(OUT_PUB, index=False)
    log.info("Wrote %s (%d rows)", OUT_PUB, len(pub))

    merged, n_added = merge_into_estimates(pub)
    merged.to_csv(EST_PATH, index=False)
    log.info(
        "Merged into %s (+%d new rows; total=%d)",
        EST_PATH,
        n_added,
        len(merged),
    )
    pisa = merged[merged["program"] == "PISA"]
    print(pisa.groupby(["cycle", "domain"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    main()
