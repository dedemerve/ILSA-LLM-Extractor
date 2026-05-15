"""
Micro-feature confounder filter test (no PDF / no API).

Feeds _sanitize with adversarial confounders including raw log codes,
TF-IDF columns, and slider triples. Expects only conceptual variables
(ESCS, gender, VOTAT) in the output list.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.extractors.gpt_extractor import GPTExtractor

ADVERSARIAL_PAYLOAD = {
    "metadata": {
        "file_name": "micro_confounder_trap.pdf",
        "title": "Micro-Feature Confounder Trap Test",
    },
    "data": {
        "survey_design": {
            "student_weights_used": False,
            "weight_fields_interpretation": "Test payload.",
        },
        "plausible_values_handling": "average_pv",
        "missing_data_handling": "listwise_deletion",
        "sample_details": {"total_students": 10000, "countries": ["TUR"]},
        "ml_techniques": {"primary": "XGBoost", "all_techniques": ["XGBoost"]},
        "confounders_identified": [
            {"variable_code": "ESCS", "variable_name": "Socioeconomic status (ESCS)", "category": "socioeconomic"},
            {"variable_code": "ST004Q01TA", "variable_name": "Gender", "category": "demographic"},
            {"variable_code": "VOTAT", "variable_name": "VOTAT navigation behavior", "category": "process_data"},
            {"variable_code": "1_0_0", "variable_name": "Action triple (1_0_0)", "category": "process_data"},
            {"variable_code": "1_2_0", "variable_name": "Slider state (1_2_0)", "category": "process_data"},
            {"variable_code": "start", "variable_name": "Start action frequency", "category": "process_data"},
            {"variable_code": "reset", "variable_name": "Reset button clicks", "category": "process_data"},
            {"variable_code": "tfidf_start", "variable_name": "TF-IDF start behavior weight", "category": "process_data"},
            {"variable_code": "word2vec_cosine", "variable_name": "Word2vec cosine similarity feature", "category": "process_data"},
            {"variable_code": "0_0_0", "variable_name": "All sliders at zero (0_0_0)", "category": "process_data"},
        ],
        "main_findings": [],
        "outcome_summary": "Synthetic test outcome summary.",
        "research_design_type": "predictive",
    },
}

FORBIDDEN_CODES = frozenset({
    "1_0_0", "1_2_0", "0_0_0", "start", "reset",
    "tfidf_start", "word2vec_cosine",
})
FORBIDDEN_NAME_FRAGMENTS = ("tf-idf", "tfidf", "word2vec", "slider", "action triple", "reset")
REQUIRED_CODES = frozenset({"ESCS", "ST004Q01TA", "VOTAT"})


def main() -> int:
    print("=" * 70)
    print("CONFOUNDER MICRO-FEATURE FILTER TEST (_sanitize)")
    print("=" * 70)

    raw_count = len(ADVERSARIAL_PAYLOAD["data"]["confounders_identified"])
    print(f"\nInput confounders: {raw_count} (includes micro-feature noise)")

    sanitized = GPTExtractor._sanitize(ADVERSARIAL_PAYLOAD)
    conf = sanitized["data"]["confounders_identified"]
    codes = {c["variable_code"] for c in conf}
    names_blob = " ".join(c["variable_name"].lower() for c in conf)

    print(f"Output confounders: {len(conf)}\n")
    for c in conf:
        print(f"  [{c['category']:18s}] {c['variable_code']:16s} → {c['variable_name']}")

    leaked_codes = codes & FORBIDDEN_CODES
    leaked_names = [frag for frag in FORBIDDEN_NAME_FRAGMENTS if frag in names_blob]
    missing_required = REQUIRED_CODES - {c.upper() for c in codes}

    print("\n" + "=" * 70)
    print("ASSERTIONS")
    print("=" * 70)

    ok = True
    if leaked_codes:
        ok = False
        print(f"  FAIL — forbidden codes still present: {sorted(leaked_codes)}")
    else:
        print("  PASS — no raw log / TF-IDF / slider micro-codes in output")

    if leaked_names:
        ok = False
        print(f"  FAIL — forbidden name fragments: {leaked_names}")
    else:
        print("  PASS — no TF-IDF / slider / reset labels in output")

    if missing_required:
        ok = False
        print(f"  FAIL — missing conceptual controls: {sorted(missing_required)}")
    else:
        print("  PASS — ESCS, gender, VOTAT retained")

    print("\n" + "=" * 70)
    if ok:
        print("VERDICT: MICRO-FEATURE FILTER OK — _sanitize dropped all noise.")
    else:
        print("VERDICT: MICRO-FEATURE FILTER FAILED — see assertions above.")
    print("=" * 70)

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
