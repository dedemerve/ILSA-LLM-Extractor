#!/usr/bin/env python3
"""
Initialize Phase 0 verification sheets (5–7) on ILSA_Meta_Analysis_Dataset_CLEAN.xlsx.

Optional / manual only — not run by the default metadata pipeline.
Run when you are ready to start claim verification or prediction inventory work.

Append-only: does not modify Sheets 0–4 or Canonical_View data rows.
Writes pilot triad sample rows (PISA 2012, Turkey, ESCS → Math PV) and applies
Excel Table + dropdown formatting.

Usage:
  python scripts/init_phase0_verification_sheets.py
  python scripts/init_phase0_verification_sheets.py --xlsx outputs/ILSA_Meta_Analysis_Dataset_CLEAN.xlsx
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.enrichment.excel_workbook_format import (
    PHASE0_SHEET_NAMES,
    SHEET5_COLUMNS,
    SHEET6_COLUMNS,
    SHEET7_COLUMNS,
    format_clean_workbook,
)

DEFAULT_XLSX = PROJECT_ROOT / "outputs" / "ILSA_Meta_Analysis_Dataset_CLEAN.xlsx"

PILOT_SHEET5: dict[str, object] = {
    "claim_id": "CLM_PISA_2012_000001",
    "file_name": "pilot_pisa2012_tur_escs_math.pdf",
    "doi": "10.0000/pilot.pisa2012.escs",
    "finding_row_hash": "sha256:pilot_finding_row_001",
    "claim_text": (
        "ESCS is positively associated with mathematics achievement among "
        "15-year-olds in Turkey (PISA 2012)."
    ),
    "normalized_isa": "PISA",
    "normalized_cycle": 2012,
    "normalized_grade": "15",
    "normalized_domain": "Mathematics",
    "countries_iso3": "TUR",
    "analysis_level": "student",
    "literature_outcome_label": "Mathematics achievement (plausible values)",
    "literature_predictor_labels": "ESCS; socio-economic status",
    "official_outcome_vars": "PISA_PV_MATH_C6",
    "official_predictor_vars": "ESCS",
    "weight_var": "W_FSTUWT",
    "weight_profile_key": "PISA_STUDENT_WGT_C6",
    "pv_rule": "rubin_rules",
    "pv_draw_used_by_paper": None,
    "catalog_file_id": "PISA_pending_phase1_scan",
    "catalog_match_confidence": "Medium",
    "verification_eligible": "TRUE",
    "verification_priority": "P1_pilot",
    "bridge_provenance": "ALIAS_ESCS;PISA_PV_MATH_C6;RULE_PV_01",
    "created_at": "2026-05-31T12:00:00Z",
}

PILOT_SHEET6: dict[str, object] = {
    "result_id": "RES_CLM_PISA_2012_000001_correlation_direction_001",
    "claim_id": "CLM_PISA_2012_000001",
    "check_type": "correlation_direction",
    "check_subtype": "weighted_pearson_rubin_pooled",
    "paper_reported_value": "Positive association (direction only reported)",
    "paper_reported_direction": "Positive",
    "paper_significance": "Significant",
    "replicated_value": None,
    "replicated_se": None,
    "replicated_p_value": None,
    "replicated_direction": None,
    "match_delta": None,
    "match_status": "Not_Runnable",
    "alignment_narrative": (
        "Phase 0 schema row; deterministic replication pending microdata catalog build."
    ),
    "n_analytic": None,
    "pv_rule_applied": "rubin_rules",
    "weight_var_applied": "W_FSTUWT",
    "country_filter_applied": "TUR",
    "evidence_path": "outputs/verification/runs/CLM_PISA_2012_000001/",
    "evidence_artifact_list": "manifest.json;claim_spec.json",
    "engine_version": "0.0.0-phase0",
    "run_timestamp": "2026-05-31T12:00:00Z",
    "run_duration_sec": 0,
    "failure_code": "CATALOG_NOT_BUILT",
}

PILOT_SHEET7: dict[str, object] = {
    "inventory_id": "INV_Math_Achievement_Mathematics",
    "target_domain": "Mathematics",
    "canonical_variable": "Math_Achievement",
    "registry_outcome_key": "PISA_MATH_ACHIEVEMENT_PV",
    "isa_coverage": "PISA;TIMSS",
    "cycle_coverage": "2012;2015;2018;2022",
    "predictor_category_mix": "Student: SES;Student: Demographic",
    "top_predictors_canonical": "SES;Gender_Demographics;Prior_Achievement",
    "n_supporting_studies": 245,
    "n_verified_claims": 0,
    "dominant_algorithms_in_literature": "Traditional_Stats;Ensemble_Learning;Deep_Learning",
    "dominant_canonical_method": "Traditional_Stats",
    "effect_direction_consensus": "Positive",
    "microdata_ready": "FALSE",
    "feature_store_status": "Planned",
    "multi_output_pipeline_priority": "Critical",
    "recommended_model_class": "weighted_mixed_effects_rubin_pv",
    "measurement_invariance_note": (
        "PISA PV scales are cycle-specific; do not pool 2012 and 2022 without "
        "OECD linking documentation."
    ),
    "last_inventory_update": "2026-05-31T12:00:00Z",
}


def _write_or_replace_sheet(
    xlsx_path: Path,
    sheet_name: str,
    columns: tuple[str, ...],
    rows: list[dict[str, object]],
) -> None:
    df = pd.DataFrame(rows, columns=list(columns))
    with pd.ExcelWriter(
        xlsx_path,
        engine="openpyxl",
        mode="a",
        if_sheet_exists="replace",
    ) as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)


def _phase0_sheets_present(xlsx_path: Path) -> bool:
    xl = pd.ExcelFile(xlsx_path)
    return all(name in xl.sheet_names for name in PHASE0_SHEET_NAMES)


def init_phase0_sheets(
    xlsx_path: Path,
    *,
    include_pilot: bool = True,
    force: bool = False,
) -> str:
    """Ensure Phase 0 sheets exist on CLEAN workbook.

    Returns action taken: ``created``, ``preserved``, or ``replaced``.
    """
    if not xlsx_path.exists():
        raise FileNotFoundError(f"CLEAN workbook not found: {xlsx_path}")

    if _phase0_sheets_present(xlsx_path) and not force:
        format_clean_workbook(xlsx_path)
        return "preserved"

    sheet5_rows = [PILOT_SHEET5] if include_pilot else []
    sheet6_rows = [PILOT_SHEET6] if include_pilot else []
    sheet7_rows = [PILOT_SHEET7] if include_pilot else []

    if not sheet5_rows:
        sheet5_rows = [{c: None for c in SHEET5_COLUMNS}]
    if not sheet6_rows:
        sheet6_rows = [{c: None for c in SHEET6_COLUMNS}]
    if not sheet7_rows:
        sheet7_rows = [{c: None for c in SHEET7_COLUMNS}]

    _write_or_replace_sheet(xlsx_path, "5_Verification_Claims", SHEET5_COLUMNS, sheet5_rows)
    _write_or_replace_sheet(xlsx_path, "6_Verification_Results", SHEET6_COLUMNS, sheet6_rows)
    _write_or_replace_sheet(xlsx_path, "7_Prediction_Inventory", SHEET7_COLUMNS, sheet7_rows)

    format_clean_workbook(xlsx_path)
    return "replaced" if force else "created"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xlsx", type=Path, default=DEFAULT_XLSX)
    parser.add_argument("--no-pilot-row", action="store_true")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace existing Phase 0 sheets with pilot rows (default: preserve existing data).",
    )
    args = parser.parse_args()

    xlsx = args.xlsx.resolve()
    action = init_phase0_sheets(
        xlsx,
        include_pilot=not args.no_pilot_row,
        force=args.force,
    )
    print(f"Phase 0 sheets on {xlsx}: {action}")
    print("  5_Verification_Claims")
    print("  6_Verification_Results")
    print("  7_Prediction_Inventory")
    print("Tab order: literature (0–4, Canonical_View) → Phase 0 (5–7) → _FilterLists (hidden).")


if __name__ == "__main__":
    main()
