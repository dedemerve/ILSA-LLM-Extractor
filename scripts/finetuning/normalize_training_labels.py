#!/usr/bin/env python3
"""
normalize_training_labels.py

Maps schema-drift values in existing LLM extractions to valid
schema Literal values, preserving original values and recording
the transformation.

Produces:
    outputs/finetuning/data/normalized_labels.csv

Normalization rules are deterministic and versioned here.
Ambiguous transformations are flagged for manual review.

IMPORTANT:
- Does NOT overwrite original JSON files.
- Does NOT modify outputs/stage5/ artifacts.
- Operates only on copies/records under outputs/finetuning/.
"""

from __future__ import annotations
import csv, json, re
from pathlib import Path

# ── Schema Literal definitions (frozen from src/schemas/models.py) ─────────
VALID_PV = {"single_pv", "all_pvs", "not_applicable", "not_reported"}
VALID_MISSING = {
    "listwise_deletion", "multiple_imputation", "mean_imputation",
    "not_reported", "not_applicable",
}
VALID_DESIGN = {"cross_sectional", "longitudinal", "experimental", "not_reported"}

# ── Normalization maps ──────────────────────────────────────────────────────
# Format: original_value → (normalized_value, rule_name, confidence, manual_review)
PV_NORMALIZATION: dict[str, tuple[str, str, str, bool]] = {
    "rubin_rules":  ("all_pvs",       "rubin_rules→all_pvs:multiple PV aggregation", "high", False),
    "average_pv":   ("all_pvs",       "average_pv→all_pvs:averaging is PV aggregation", "high", False),
    "irt_theta":    ("not_applicable", "irt_theta→not_applicable:latent score, not ILSA PV", "medium", True),
    "wle":          ("not_applicable", "wle→not_applicable:Warm likelihood, not standard PV", "medium", True),
    "all_pv":       ("all_pvs",        "all_pv→all_pvs:canonical spelling", "high", False),
}

MISSING_NORMALIZATION: dict[str, tuple[str, str, str, bool]] = {
    "knn_imputation":    ("multiple_imputation", "knn→multiple_imputation:KNN is an imputation method", "high", False),
    "single_imputation": ("mean_imputation",     "single→mean:single-value imputation coded as mean", "medium", True),
}

DESIGN_NORMALIZATION: dict[str, tuple[str, str, str, bool]] = {
    "predictive":          ("cross_sectional", "predictive→cross_sectional:predictive ML typically cross-sectional", "medium", True),
    "exploratory":         ("cross_sectional", "exploratory→cross_sectional:exploratory ML typically cross-sectional", "medium", True),
    "causal_observational":("cross_sectional", "causal_obs→cross_sectional:observational design", "low", True),
}


def normalize_field(
    field_name: str,
    original_value: str,
    norm_map: dict,
    valid_set: set,
) -> dict:
    v = (original_value or "").strip().lower()
    if v in valid_set:
        return {
            "field": field_name,
            "original_value": original_value,
            "normalized_value": original_value,
            "normalization_rule": "already_valid",
            "confidence": "exact",
            "manual_review_required": False,
        }
    if v in norm_map:
        nv, rule, conf, review = norm_map[v]
        return {
            "field": field_name,
            "original_value": original_value,
            "normalized_value": nv,
            "normalization_rule": rule,
            "confidence": conf,
            "manual_review_required": review,
        }
    # Unknown value
    return {
        "field": field_name,
        "original_value": original_value,
        "normalized_value": "not_reported",
        "normalization_rule": f"unknown_value→not_reported:value '{v}' not in schema or map",
        "confidence": "low",
        "manual_review_required": True,
    }


def run():
    json_dir = Path(__file__).resolve().parents[2] / "outputs" / "ilsa_survey_articles" / "json"
    out_dir = Path(__file__).resolve().parents[2] / "outputs" / "finetuning" / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted([f for f in json_dir.glob("*.json") if "(1)" not in f.name])

    rows = []
    for fp in files:
        try:
            with open(fp) as f:
                d = json.load(f)
            data = d.get("data", {})
            study_id = fp.stem[:80]

            for field_name, original, norm_map, valid_set in [
                ("plausible_values_handling",
                 data.get("plausible_values_handling", ""),
                 PV_NORMALIZATION, VALID_PV),
                ("missing_data_handling",
                 data.get("missing_data_handling", ""),
                 MISSING_NORMALIZATION, VALID_MISSING),
                ("research_design_type",
                 data.get("research_design_type", ""),
                 DESIGN_NORMALIZATION, VALID_DESIGN),
            ]:
                result = normalize_field(field_name, original, norm_map, valid_set)
                result["study_id"] = study_id
                rows.append(result)

        except Exception as e:
            rows.append({
                "study_id": fp.stem[:80],
                "field": "parse_error",
                "original_value": str(e),
                "normalized_value": "",
                "normalization_rule": "parse_error",
                "confidence": "none",
                "manual_review_required": True,
            })

    # Reorder columns
    fieldnames = [
        "study_id", "field", "original_value", "normalized_value",
        "normalization_rule", "confidence", "manual_review_required",
    ]
    with open(out_dir / "normalized_labels.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Summary
    from collections import Counter
    drift = [r for r in rows if r["normalization_rule"] != "already_valid" and r["field"] != "parse_error"]
    print(f"Total normalization records: {len(rows)}")
    print(f"Schema drift instances: {len(drift)}")
    print(f"Manual review required: {sum(1 for r in rows if r['manual_review_required'])}")
    print(f"Written: {out_dir / 'normalized_labels.csv'}")


if __name__ == "__main__":
    run()
