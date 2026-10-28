#!/usr/bin/env python3
"""Build a versioned PISA country-cycle panel for forecasting experiments.

The source unified panel is read-only. PISA 2000 reading country identifiers
are canonicalized with the source-verified numeric-to-alpha3 crosswalk. Output
files are versioned and fail fast if they already exist.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "outputs" / "stage5" / "unified_panel.csv"
CROSSWALK = (
    ROOT
    / "analysis"
    / "03_data_feasibility"
    / "audit"
    / "pisa_2000_reading_country_crosswalk.csv"
)
OUTPUT = ROOT / "outputs" / "stage5" / "pisa_forecasting_panel_v2.csv"
MANIFEST = ROOT / "outputs" / "stage5" / "pisa_forecasting_panel_v2_manifest.json"
DOMAINS = ("mathematics", "reading", "science")
MIN_CONTEXT = 3
TARGET_CYCLES = (2009, 2012, 2015, 2018, 2022, 2025)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    for path in (INPUT, CROSSWALK):
        if not path.is_file():
            raise FileNotFoundError(path)
    for path in (OUTPUT, MANIFEST):
        if path.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing artifact: {path}. "
                "Choose a new versioned output path."
            )

    panel = pd.read_csv(INPUT, dtype={"country_iso3": str})
    required = {"program", "domain", "country_iso3", "cycle"}
    missing = sorted(required - set(panel.columns))
    if missing:
        raise ValueError(f"Input panel lacks required columns: {missing}")

    panel["program"] = panel["program"].astype(str).str.strip().str.upper()
    panel["domain"] = panel["domain"].astype(str).str.strip().str.lower()
    panel["country_iso3"] = panel["country_iso3"].astype(str).str.strip().str.upper()
    panel["cycle"] = pd.to_numeric(panel["cycle"], errors="raise").astype(int)

    panel = panel[
        (panel["program"] == "PISA") & panel["domain"].isin(DOMAINS)
    ].copy()
    if panel.empty:
        raise ValueError("No PISA math/reading/science rows found in input panel")

    source_key = ["program", "domain", "country_iso3", "cycle"]
    if panel.duplicated(source_key).any():
        duplicates = panel.loc[panel.duplicated(source_key, keep=False), source_key]
        raise ValueError(
            "Input contains duplicate source country-domain-cycle keys: "
            f"{duplicates.drop_duplicates().to_dict(orient='records')[:10]}"
        )

    crosswalk = pd.read_csv(CROSSWALK, dtype=str).fillna("")
    needed_crosswalk = {
        "source_cycle",
        "source_country_id",
        "canonical_alpha3",
        "verification_status",
    }
    missing = sorted(needed_crosswalk - set(crosswalk.columns))
    if missing:
        raise ValueError(f"Crosswalk lacks required columns: {missing}")
    if not crosswalk["verification_status"].eq("VERIFIED").all():
        raise ValueError("Crosswalk contains rows not marked VERIFIED")
    if not crosswalk["source_cycle"].eq("2000").all():
        raise ValueError("This crosswalk must contain PISA 2000 rows only")
    if crosswalk["source_country_id"].duplicated().any():
        raise ValueError("Crosswalk source identifiers are not unique")
    if crosswalk["canonical_alpha3"].duplicated().any():
        raise ValueError("Crosswalk canonical identifiers are not unique")
    if not crosswalk["canonical_alpha3"].map(
        lambda value: bool(re.fullmatch(r"[A-Z]{3}", value))
    ).all():
        raise ValueError("Crosswalk contains invalid alpha-3 identifiers")

    panel["country_iso3_source"] = panel["country_iso3"]
    is_2000_reading = (panel["cycle"] == 2000) & (panel["domain"] == "reading")
    source_ids = set(panel.loc[is_2000_reading, "country_iso3"])
    crosswalk_ids = set(crosswalk["source_country_id"])
    if source_ids != crosswalk_ids:
        raise ValueError(
            "PISA 2000 reading identifiers do not exactly match the verified "
            f"crosswalk (input-only={sorted(source_ids - crosswalk_ids)}, "
            f"crosswalk-only={sorted(crosswalk_ids - source_ids)})"
        )
    mapping = dict(
        zip(crosswalk["source_country_id"], crosswalk["canonical_alpha3"])
    )
    panel.loc[is_2000_reading, "country_iso3"] = panel.loc[
        is_2000_reading, "country_iso3"
    ].map(mapping)

    valid_ids = panel["country_iso3"].map(
        lambda value: bool(re.fullmatch(r"[A-Z]{3}", value))
    )
    if not valid_ids.all():
        invalid = sorted(panel.loc[~valid_ids, "country_iso3"].unique().tolist())
        raise ValueError(f"Non-alpha3 country keys remain: {invalid}")

    key = ["program", "domain", "country_iso3", "cycle"]
    if panel.duplicated(key).any():
        duplicates = panel.loc[panel.duplicated(key, keep=False), key]
        raise ValueError(
            "Canonicalization created duplicate country-domain-cycle keys: "
            f"{duplicates.drop_duplicates().to_dict(orient='records')[:10]}"
        )

    coverage: dict[str, dict[str, dict[str, int]]] = {}
    eligible_sets: dict[int, dict[str, set[str]]] = {}
    for domain in DOMAINS:
        score_col = f"PISA_{domain}"
        if score_col not in panel.columns:
            raise ValueError(f"Missing score column: {score_col}")
        series = panel.loc[
            (panel["domain"] == domain) & panel[score_col].notna(),
            ["country_iso3", "cycle", score_col],
        ].copy()
        if series.duplicated(["country_iso3", "cycle"]).any():
            raise ValueError(f"Duplicate score series key in {domain}")

        domain_coverage: dict[str, dict[str, int]] = {}
        for target in TARGET_CYCLES:
            target_countries = set(
                series.loc[series["cycle"] == target, "country_iso3"]
            )
            history = series.loc[series["cycle"] < target].groupby(
                "country_iso3"
            )["cycle"].nunique()
            eligible = target_countries & set(
                history.loc[history >= MIN_CONTEXT].index
            )
            eligible_sets.setdefault(target, {})[domain] = eligible
            domain_coverage[str(target)] = {
                "countries_with_target": len(target_countries),
                "eligible_min_3_prior_cycles": len(eligible),
            }
        coverage[domain] = domain_coverage

    common_counts = {
        str(target): len(
            set.intersection(*(eligible_sets[target][domain] for domain in DOMAINS))
        )
        for target in TARGET_CYCLES
    }

    panel = panel.sort_values(
        ["program", "domain", "cycle", "country_iso3"]
    ).reset_index(drop=True)
    panel.to_csv(OUTPUT, index=False)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input": str(INPUT.relative_to(ROOT)),
        "input_sha256": sha256(INPUT),
        "crosswalk": str(CROSSWALK.relative_to(ROOT)),
        "crosswalk_sha256": sha256(CROSSWALK),
        "output": str(OUTPUT.relative_to(ROOT)),
        "output_sha256": sha256(OUTPUT),
        "rows": len(panel),
        "domains": list(DOMAINS),
        "key": key,
        "canonicalized_rows": int(is_2000_reading.sum()),
        "minimum_prior_cycles": MIN_CONTEXT,
        "eligible_country_counts_by_domain_and_target": coverage,
        "common_country_count_by_target_across_all_domains": common_counts,
        "pisa_2025_status": "outcome_exposed_retrospective_benchmark_only",
        "note": "Source panel was not modified; no modeling or scoring was run.",
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT.relative_to(ROOT)} ({len(panel)} rows)")
    print(f"Wrote {MANIFEST.relative_to(ROOT)}")
    print(f"Canonicalized PISA 2000 reading rows: {int(is_2000_reading.sum())}")
    print(f"Common eligible countries by target: {common_counts}")


if __name__ == "__main__":
    main()
