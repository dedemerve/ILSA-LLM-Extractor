#!/usr/bin/env python3
"""
Deterministic verification engine — Phase 0 specification stub.

Full implementation gated on Phase 1 microdata catalog build.
See: docs/phase0_methodological_specification.md (Section 3)

Usage (Phase 0 — schema validation only):
  python scripts/verify_claim.py --claim-id CLM_PISA_2012_000001 --dry-run

Usage (Phase 1+ — after catalog build):
  python scripts/verify_claim.py --claim-id CLM_PISA_2012_000001 \\
      --checks correlation_direction
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_XLSX = PROJECT_ROOT / "outputs" / "ILSA_Meta_Analysis_Dataset_CLEAN.xlsx"
CATALOG_MANIFEST = PROJECT_ROOT / "outputs" / "ilsa_microdata_catalog" / "schema_manifest.json"
ENGINE_VERSION = "0.0.0-phase0"


def _load_claim(xlsx: Path, claim_id: str) -> dict:
    import pandas as pd

    df = pd.read_excel(xlsx, sheet_name="5_Verification_Claims")
    match = df[df["claim_id"] == claim_id]
    if match.empty:
        raise ValueError(f"claim_id not found in Sheet 5: {claim_id}")
    return match.iloc[0].to_dict()


def verify_claim(
    claim_id: str,
    *,
    checks: list[str],
    xlsx: Path = DEFAULT_XLSX,
    dry_run: bool = True,
) -> dict:
    """
    Phase 0: validate claim row exists and catalog is not yet built.
    Phase 1+: resolve .sav via DuckDB, apply Rubin pooling, write Sheet 6.
    """
    claim = _load_claim(xlsx, claim_id)
    catalog_ready = (
        CATALOG_MANIFEST.exists()
        and json.loads(CATALOG_MANIFEST.read_text()).get("status", "").startswith("PHASE0")
        is False
    )
    if not catalog_ready:
        return {
            "claim_id": claim_id,
            "engine_version": ENGINE_VERSION,
            "match_status": "Not_Runnable",
            "failure_code": "CATALOG_NOT_BUILT",
            "checks_requested": checks,
            "dry_run": dry_run,
            "message": (
                "Phase 0 stub: claim loaded successfully; microdata catalog not populated. "
                "Run Phase 1 build_microdata_catalog.py before deterministic replication."
            ),
        }
    raise NotImplementedError("Phase 1 verification engine not yet implemented.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--claim-id", required=True)
    parser.add_argument(
        "--checks",
        nargs="+",
        default=["correlation_direction"],
        choices=[
            "descriptive",
            "weighted_mean",
            "correlation_direction",
            "correlation_magnitude",
            "regression_coefficient_sign",
            "metric_replication",
            "sample_size",
        ],
    )
    parser.add_argument("--xlsx", type=Path, default=DEFAULT_XLSX)
    parser.add_argument("--dry-run", action="store_true", default=True)
    args = parser.parse_args()

    result = verify_claim(
        args.claim_id,
        checks=args.checks,
        xlsx=args.xlsx.resolve(),
        dry_run=args.dry_run,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if result.get("failure_code") == "CATALOG_NOT_BUILT":
        sys.exit(0)
    sys.exit(0 if result.get("match_status") == "Full_Match" else 1)


if __name__ == "__main__":
    main()
