#!/usr/bin/env python3
"""Freeze country sets and cutoffs for PISA historical rolling-origin folds."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "outputs" / "stage5" / "pisa_forecasting_panel_v2.csv"
OUT = ROOT / "analysis" / "03_data_feasibility" / "audit" / "pisa_historical_folds_v1.json"
DOMAINS = ("mathematics", "reading", "science")
TARGET_CYCLES = (2012, 2015, 2018, 2022)
MIN_CONTEXT = 3
EXPECTED_DOMAIN_COUNTS = {
    2012: {"mathematics": 40, "reading": 45, "science": 40},
    2015: {"mathematics": 51, "reading": 53, "science": 51},
    2018: {"mathematics": 57, "reading": 57, "science": 57},
    2022: {"mathematics": 60, "reading": 61, "science": 60},
}
EXPECTED_COMMON_COUNTS = {2012: 40, 2015: 51, 2018: 57, 2022: 60}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if not PANEL.is_file():
        raise FileNotFoundError(PANEL)
    if OUT.exists():
        raise FileExistsError(
            f"Refusing to overwrite {OUT}; choose a new manifest version."
        )

    panel = pd.read_csv(PANEL, dtype={"country_iso3": str})
    panel = panel.loc[panel["program"].eq("PISA")].copy()
    panel["cycle"] = pd.to_numeric(panel["cycle"], errors="raise").astype(int)
    manifest_folds = []

    for target in TARGET_CYCLES:
        domain_sets: dict[str, set[str]] = {}
        domain_details = {}
        for domain in DOMAINS:
            score = f"PISA_{domain}"
            series = panel.loc[
                (panel["domain"] == domain) & panel[score].notna(),
                ["country_iso3", "cycle", score],
            ]
            target_countries = set(
                series.loc[series["cycle"] == target, "country_iso3"]
            )
            history = series.loc[series["cycle"] < target].groupby(
                "country_iso3"
            )["cycle"].nunique()
            eligible = target_countries & set(
                history.loc[history >= MIN_CONTEXT].index
            )
            domain_sets[domain] = eligible
            expected_n = EXPECTED_DOMAIN_COUNTS[target][domain]
            if len(eligible) != expected_n:
                raise ValueError(
                    f"Unexpected {domain} eligible count for {target}: "
                    f"{len(eligible)} != {expected_n}"
                )
            domain_details[domain] = {
                "target_score_countries": len(target_countries),
                "eligible_country_count": len(eligible),
                "eligible_countries": sorted(eligible),
                "training_cycles": sorted(
                    int(year)
                    for year in series.loc[series["cycle"] < target, "cycle"].unique()
                ),
                "cutoff_rule": f"score cycle < {target}",
            }

        common = set.intersection(*(domain_sets[d] for d in DOMAINS))
        expected_common = EXPECTED_COMMON_COUNTS[target]
        if len(common) != expected_common:
            raise ValueError(
                f"Unexpected common eligible count for {target}: "
                f"{len(common)} != {expected_common}"
            )
        manifest_folds.append(
            {
                "target_cycle": target,
                "minimum_prior_observations": MIN_CONTEXT,
                "domain_specific": domain_details,
                "common_three_domain_country_count": len(common),
                "common_three_domain_countries": sorted(common),
            }
        )

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "panel": str(PANEL.relative_to(ROOT)),
        "panel_sha256": sha256(PANEL),
        "domains": list(DOMAINS),
        "fold_role": "historical_development_and_temporal_validation",
        "folds": manifest_folds,
        "pisa_2025": {
            "role": "outcome_exposed_retrospective_benchmark_only",
            "excluded_from_historical_fold_manifest": True,
        },
        "leakage_rule": (
            "For target cycle t, training may use only score observations "
            "with cycle < t; preprocessing and tuning must also be fit within "
            "the training portion of each fold."
        ),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {OUT.relative_to(ROOT)}")
    print(
        "Common three-domain eligible counts:",
        {fold["target_cycle"]: fold["common_three_domain_country_count"] for fold in manifest_folds},
    )


if __name__ == "__main__":
    main()
