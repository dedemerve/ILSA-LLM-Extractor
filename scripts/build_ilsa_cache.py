#!/usr/bin/env python3
"""
Build ILSA local data catalog and precompute trend/microstats caches.

Usage:
  python scripts/build_ilsa_cache.py
  python scripts/build_ilsa_cache.py --countries Turkey Germany Finland
  python scripts/build_ilsa_cache.py --rebuild-catalog
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from ilsa_local.store import LocalILSAStore  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

DEFAULT_COUNTRIES = [
    "Turkey", "Germany", "Finland", "Korea", "Japan", "United States",
    "United Kingdom", "Poland", "France", "Australia", "Canada", "Singapore",
    "Brazil", "Indonesia", "Chile", "Mexico", "Spain", "Italy", "Sweden",
    "Norway", "Netherlands", "New Zealand", "Portugal", "Chinese Taipei",
]

# Multi-word names must be quoted on CLI; this merges bare tokens when omitted.
_MULTI_WORD_COUNTRIES = tuple(
    sorted(
        {c for c in DEFAULT_COUNTRIES if " " in c},
        key=len,
        reverse=True,
    )
)


def _join_multi_word_countries(tokens: list[str]) -> list[str]:
    if not tokens:
        return list(DEFAULT_COUNTRIES)
    out: list[str] = []
    i = 0
    while i < len(tokens):
        matched = False
        for name in _MULTI_WORD_COUNTRIES:
            parts = name.split()
            if tokens[i : i + len(parts)] == parts:
                out.append(name)
                i += len(parts)
                matched = True
                break
        if not matched:
            out.append(tokens[i])
            i += 1
    return out
PROGRAMS = [
    ("PISA", "mathematics"), ("PISA", "reading"), ("PISA", "science"),
    ("TIMSS", "mathematics"), ("TIMSS", "science"),
    ("PIRLS", "reading"),
    ("PIAAC", "literacy"),
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Build ILSA local data cache")
    parser.add_argument("--countries", nargs="+", default=None,
                        help="Country list (quote multi-word names). Default: DEFAULT_COUNTRIES")
    parser.add_argument("--all-countries", action="store_true",
                        help="All countries with local SPSS/CSV files in catalog")
    parser.add_argument("--all-kb-countries", action="store_true",
                        help="All 184 KB-indexed countries (uses embedded trends where local missing)")
    parser.add_argument("--coverage-matrix", action="store_true",
                        help="Build country_coverage_matrix.json and exit")
    parser.add_argument("--rebuild-catalog", action="store_true")
    parser.add_argument("--skip-trends", action="store_true")
    args = parser.parse_args()

    if args.coverage_matrix:
        from country_registry import save_coverage_matrix
        p = save_coverage_matrix()
        log.info("Coverage matrix: %d countries → outputs/ilsa_cache/country_coverage_matrix.json", p["n_countries"])
        return 0

    if args.all_kb_countries:
        from country_registry import kb_countries_for_cache
        countries = kb_countries_for_cache()
        log.info("All KB countries: %d", len(countries))
    elif args.all_countries:
        from country_registry import cache_build_countries
        countries = cache_build_countries()
        log.info("All catalog countries: %d", len(countries))
    elif args.countries:
        countries = _join_multi_word_countries(args.countries)
    else:
        countries = list(DEFAULT_COUNTRIES)

    store = LocalILSAStore()
    if not store.available:
        log.error("ILSA Datasets folder not found: %s", store.root)
        return 1

    if args.rebuild_catalog or not store.catalog:
        n = store.refresh_catalog()
        log.info("Catalog rebuilt: %d entries", n)
    else:
        log.info("Catalog loaded: %d entries", len(store.catalog))
        log.info("Summary: %s", json.dumps(store.summary(), indent=2))

    if args.skip_trends:
        return 0

    trends: dict = store.load_trends_cache()
    if "_meta" in trends:
        del trends["_meta"]

    for program, domain in PROGRAMS:
        prog = program.upper()
        if prog not in trends:
            trends[prog] = {}
        for country in countries:
            if country not in trends[prog]:
                trends[prog][country] = {}
            log.info("Computing trends: %s / %s / %s", prog, country, domain)
            computed = store.build_trends_for_country(prog, country, domain)
            if computed:
                trends[prog][country][domain] = {str(k): v for k, v in computed.items()}
                log.info("  → %s", computed)
            store.get_real_stats(prog, country, domain)

    store.save_trends_cache(trends)
    log.info("Done. Cache at %s", store.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
