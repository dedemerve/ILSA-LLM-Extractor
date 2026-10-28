"""
Phase 2: Study-Level Prediction-Task Taxonomy Classification
Input:  outputs/ILSA_Meta_Analysis_Dataset_CLEAN.xlsx  (authoritative, read-only)
        outputs/ILSA_Analytical_Meta_Analysis_Master.xlsx  (supplementary, read-only)
Output: analysis/02_prediction_taxonomy/study_classification.csv
        analysis/02_prediction_taxonomy/program_normalization.csv
        analysis/02_prediction_taxonomy/ambiguous_studies.csv
        analysis/02_prediction_taxonomy/phase2_audit.json
"""

import pandas as pd
import numpy as np
import json
import re
import os
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent  # ILSA_LLMs/
OUTPUT_DIR = Path(__file__).parent

# ── 1. Load data (read-only) ──────────────────────────────────────────────────
xl_clean = pd.ExcelFile(ROOT / "outputs/ILSA_Meta_Analysis_Dataset_CLEAN.xlsx")
xl_anal  = pd.ExcelFile(ROOT / "outputs/ILSA_Analytical_Meta_Analysis_Master.xlsx")

art  = xl_clean.parse("1_Articles_Master")
find = xl_clean.parse("2_Main_Findings")
study_anal = xl_anal.parse("Study_Details")
find_anal  = xl_anal.parse("Empirical_Findings")

print(f"Articles: {len(art)}, Findings: {len(find)}")

# ── 2. Taxonomy vocabularies ──────────────────────────────────────────────────
TASK_LABELS = {
    "A": "student_within_cycle_prediction",
    "B": "student_cross_cycle_prediction",
    "C": "school_teacher_classroom_prediction",
    "D": "country_system_prediction",
    "E": "future_cycle_forecasting",
    "F": "other_predictive",
    "G": "non_predictive_analytical",
    "H": "uncertain_insufficient_evidence",
}

TEMPORAL_STATUS_VOCAB = {
    "confirmed_future_cycle_forecasting",
    "cross_cycle_prediction_not_fully_verified",
    "within_cycle_prediction",
    "no_temporal_prediction",
    "not_applicable",
    "insufficient_evidence",
}

VALIDATION_VOCAB = {
    "random_train_test_split",
    "k_fold_cross_validation",
    "leave_one_out",
    "country_holdout",
    "school_holdout",
    "temporal_holdout_cross_cycle",
    "rolling_origin",
    "external_validation",
    "no_split_reported",
    "not_applicable",
    "mixed_methods",
    "insufficient_evidence",
}

# ── 3. ILSA Program Normalization ─────────────────────────────────────────────
PROGRAM_MAP = {
    # Pattern → (canonical_name, notes)
    r"\bPISA\b":   ("PISA",   "Programme for International Student Assessment (OECD)"),
    r"\bTIMSS\b":  ("TIMSS",  "Trends in International Mathematics and Science Study (IEA)"),
    r"\bPIRLS\b":  ("PIRLS",  "Progress in International Reading Literacy Study (IEA)"),
    r"\bTALIS\b":  ("TALIS",  "Teaching and Learning International Survey (OECD)"),
    r"\bPIAAC\b":  ("PIAAC",  "Programme for the International Assessment of Adult Competencies (OECD)"),
    r"\bICCS\b":   ("ICCS",   "International Civic and Citizenship Education Study (IEA)"),
    r"\bICILS\b":  ("ICILS",  "International Computer and Information Literacy Study (IEA)"),
    r"\bTIMSS Advanced\b": ("TIMSS_Advanced", "TIMSS Advanced (IEA)"),
    r"\bPISA-D\b": ("PISA_D", "PISA for Development (OECD)"),
}

CYCLE_PATTERN = re.compile(r"\b(19|20)\d{2}\b")

def extract_programs(text: str) -> list[str]:
    if pd.isna(text):
        return []
    programs = []
    for pattern, (name, _) in PROGRAM_MAP.items():
        if re.search(pattern, text, re.IGNORECASE):
            programs.append(name)
    return list(dict.fromkeys(programs))  # deduplicate, preserve order

def extract_cycles(text: str, programs: list[str]) -> list[str]:
    if pd.isna(text):
        return []
    years = sorted(set(CYCLE_PATTERN.findall(text)))
    return years

# ── 4. Merge article + findings info ─────────────────────────────────────────
# Aggregate findings per file_name
def agg_findings(group):
    datasets = " | ".join(group["dataset_used"].dropna().astype(str).unique())
    domains = " | ".join(group["target_domain"].dropna().astype(str).unique())
    dimensions = " | ".join(group["target_dimension"].dropna().astype(str).unique())
    target_vars = " | ".join(group["target_variable"].dropna().astype(str).unique())
    has_metrics = any(
        v not in ("not reported", "n/a: technical report", "nan", "")
        for v in group["performance_metrics"].fillna("").str.lower()
    )
    return pd.Series({
        "all_datasets_used": datasets,
        "all_target_domains": domains,
        "all_target_dimensions": dimensions,
        "all_target_variables": target_vars,
        "has_performance_metrics": has_metrics,
        "n_findings": len(group),
    })

find_agg = find.groupby("file_name").apply(agg_findings).reset_index()

art_merged = art.merge(find_agg, on="file_name", how="left")

# ── 5. ILSA program normalization ─────────────────────────────────────────────
# Use dataset_used + outcome_summary for program extraction
def get_programs_for_study(row):
    text = " ".join([
        str(row.get("all_datasets_used", "") or ""),
        str(row.get("outcome_summary", "") or ""),
    ])
    return extract_programs(text)

def get_cycles_for_study(row, programs):
    text = " ".join([
        str(row.get("all_datasets_used", "") or ""),
    ])
    return extract_cycles(text, programs)

art_merged["normalized_programs"] = art_merged.apply(
    lambda r: get_programs_for_study(r), axis=1
)
art_merged["normalized_cycles"] = art_merged.apply(
    lambda r: get_cycles_for_study(r, r["normalized_programs"]), axis=1
)
art_merged["n_programs"] = art_merged["normalized_programs"].apply(len)
art_merged["n_cycles_extracted"] = art_merged["normalized_cycles"].apply(len)

# ── 6. Temporal forecasting signal detection ──────────────────────────────────
FC_KEYWORDS = re.compile(
    r"forecast|predict.*next|next.*cycle|future.*cycle|future.*assessment|"
    r"trend.*predict|time.series.*predict|cross.cycle.*predict|"
    r"transfer.*pisa|pisa.*transfer|train.*earlier.*test.*later",
    re.IGNORECASE
)

MULTI_CYCLE_ML_ARROW = re.compile(r"\d{4}\s*[→\-–]\s*\d{4}", re.IGNORECASE)

def temporal_forecasting_status(row):
    sft = str(row.get("study_filter_type", ""))
    datasets = str(row.get("all_datasets_used", "") or "")
    outcome = str(row.get("outcome_summary", "") or "")
    text = datasets + " " + outcome

    if "Technical/Assessment Framework" in sft or "Descriptive National Report" in sft:
        return "not_applicable"

    if "Machine Learning" in sft:
        # Strongest signal: arrow notation between cycles
        if MULTI_CYCLE_ML_ARROW.search(datasets):
            return "cross_cycle_prediction_not_fully_verified"
        if FC_KEYWORDS.search(text):
            return "cross_cycle_prediction_not_fully_verified"
        n_cycles = row.get("n_cycles_extracted", 0)
        if isinstance(n_cycles, (int, float)) and n_cycles >= 2:
            return "cross_cycle_prediction_not_fully_verified"
        return "within_cycle_prediction"

    if "Traditional Statistics" in sft:
        if FC_KEYWORDS.search(text):
            return "cross_cycle_prediction_not_fully_verified"
        return "no_temporal_prediction"

    return "insufficient_evidence"

art_merged["temporal_forecasting_status"] = art_merged.apply(
    temporal_forecasting_status, axis=1
)

# ── 7. Prediction target level ────────────────────────────────────────────────
def prediction_target_level(row):
    dims = str(row.get("all_target_dimensions", "") or "").lower()
    domains = str(row.get("all_target_domains", "") or "").lower()
    outcome = str(row.get("outcome_summary", "") or "").lower()

    if "school" in dims or "teacher" in dims or "classroom" in dims:
        return "school_teacher_classroom"
    if "country" in dims or "system" in dims or "national" in outcome:
        return "country_system"
    if "cognitive" in dims:
        return "student"
    if "n/a" in dims or "technical report" in dims:
        return "not_applicable"
    if "attitudinal" in dims or "affective" in dims:
        return "student"
    return "unclear"

art_merged["prediction_target_level"] = art_merged.apply(
    prediction_target_level, axis=1
)

# ── 8. Primary research task classification ───────────────────────────────────
def classify_task(row):
    sft  = str(row.get("study_filter_type", "") or "")
    rdt  = str(row.get("research_design_type", "") or "")
    tfs  = str(row.get("temporal_forecasting_status", "") or "")
    tl   = str(row.get("prediction_target_level", "") or "")
    dims = str(row.get("all_target_dimensions", "") or "").lower()

    if "Technical/Assessment Framework" in sft or "Descriptive National Report" in sft:
        return "G", "Low"

    if "Traditional Statistics" in sft:
        if "causal" in rdt or "predictive" in rdt:
            return "G", "Medium"   # traditional stats with predictive label but not ML
        return "G", "Low"

    if "Machine Learning" not in sft:
        return "G", "Medium"

    # ML study branch
    if "confirmed_future_cycle_forecasting" in tfs:
        if "country" in tl:
            return "E", "High"
        return "E", "Medium"

    if "cross_cycle_prediction_not_fully_verified" in tfs:
        if "country" in tl:
            return "D", "Medium"
        return "B", "Medium"

    if "within_cycle_prediction" in tfs:
        if tl == "country_system":
            return "D", "High"
        if tl == "school_teacher_classroom":
            return "C", "High"
        if tl == "student":
            return "A", "High"
        if "cognitive" in dims:
            return "A", "Medium"
        return "F", "Medium"

    return "H", "Low"

results = art_merged.apply(classify_task, axis=1)
art_merged["primary_task_code"] = results.apply(lambda x: x[0])
art_merged["primary_task_label"] = art_merged["primary_task_code"].map(TASK_LABELS)
art_merged["classification_confidence"] = results.apply(lambda x: x[1])

# ── 9. Validation strategy classification ────────────────────────────────────
def classify_validation(row):
    sft = str(row.get("study_filter_type", "") or "")
    if "Technical" in sft or "Descriptive" in sft:
        return "not_applicable"
    if "Machine Learning" not in sft:
        return "no_split_reported"

    datasets = str(row.get("all_datasets_used", "") or "").lower()
    outcome  = str(row.get("outcome_summary", "") or "").lower()
    text     = datasets + " " + outcome

    if re.search(r"rolling.origin|rolling.window|expanding.window", text, re.IGNORECASE):
        return "rolling_origin"
    if MULTI_CYCLE_ML_ARROW.search(datasets):
        return "temporal_holdout_cross_cycle"
    if re.search(r"cross.valid|k.fold|\bkfold\b|cv\b|leave.one.out", text, re.IGNORECASE):
        return "k_fold_cross_validation"
    if re.search(r"hold.?out|train.*test|test.*split|80.*20|70.*30", text, re.IGNORECASE):
        return "random_train_test_split"
    return "insufficient_evidence"

art_merged["validation_strategy"] = art_merged.apply(classify_validation, axis=1)

# ── 10. Build final output table ──────────────────────────────────────────────
KEEP_ORIGINAL = [
    "file_name", "doi", "title", "authors", "year",
    "publication_type", "source_category", "research_design_type",
    "outcome_summary", "ml_primary", "ml_all_techniques", "ml_family",
    "student_weights_used", "plausible_values_handling",
    "study_filter_type", "document_class",
]
NEW_FIELDS = [
    "all_datasets_used", "all_target_domains", "all_target_dimensions",
    "all_target_variables", "has_performance_metrics", "n_findings",
    "normalized_programs", "normalized_cycles", "n_programs", "n_cycles_extracted",
    "temporal_forecasting_status", "prediction_target_level",
    "primary_task_code", "primary_task_label", "classification_confidence",
    "validation_strategy",
]

out = art_merged[KEEP_ORIGINAL + NEW_FIELDS].copy()
out["normalized_programs"] = out["normalized_programs"].apply(
    lambda x: "|".join(x) if isinstance(x, list) else ""
)
out["normalized_cycles"] = out["normalized_cycles"].apply(
    lambda x: "|".join(x) if isinstance(x, list) else ""
)

# Mark potential duplicates
no_doi_titles = art[art["doi"].isna()]["title"].dropna()
dup_titles = set(no_doi_titles[no_doi_titles.duplicated(keep=False)])
out["potential_duplicate"] = (
    art["doi"].isna() & art["title"].isin(dup_titles)
)
out["manual_review_required"] = (
    (out["classification_confidence"] != "High") |
    (out["temporal_forecasting_status"] == "cross_cycle_prediction_not_fully_verified") |
    out["potential_duplicate"]
)

out.to_csv(OUTPUT_DIR / "study_classification.csv", index=False)
print(f"Saved study_classification.csv: {len(out)} rows")

# ── 11. Program normalization dictionary ──────────────────────────────────────
prog_records = []
for _, row in art_merged.iterrows():
    raw_datasets = str(row.get("all_datasets_used", "") or "")
    progs = row["normalized_programs"]
    if not isinstance(progs, list):
        progs = []
    cycles = row["normalized_cycles"]
    if not isinstance(cycles, list):
        cycles = []
    for prog in progs:
        prog_records.append({
            "file_name": row["file_name"],
            "raw_text_fragment": raw_datasets[:200],
            "canonical_program": prog,
            "extracted_cycle_years": "|".join(cycles),
            "confidence": "extraction_based",
        })

prog_df = pd.DataFrame(prog_records)
# Summary per program
prog_summary = prog_df.groupby("canonical_program")["file_name"].nunique().reset_index()
prog_summary.columns = ["canonical_program", "n_studies"]
prog_summary.to_csv(OUTPUT_DIR / "program_normalization.csv", index=False)
print(f"Saved program_normalization.csv: {len(prog_summary)} programs")
print(prog_summary.to_string())

# ── 12. Ambiguous studies ─────────────────────────────────────────────────────
ambig = out[out["manual_review_required"]].copy()
ambig.to_csv(OUTPUT_DIR / "ambiguous_studies.csv", index=False)
print(f"\nAmbiguous/manual-review studies: {len(ambig)}")

# ── 13. Summary statistics ────────────────────────────────────────────────────
print("\n=== PRIMARY TASK DISTRIBUTION ===")
task_counts = out["primary_task_label"].value_counts()
for label, count in task_counts.items():
    pct = 100 * count / len(out)
    print(f"  {label:45s}  {count:4d}  ({pct:.1f}%)")

print("\n=== TEMPORAL FORECASTING STATUS ===")
print(out["temporal_forecasting_status"].value_counts().to_string())

print("\n=== VALIDATION STRATEGY (ML studies only) ===")
ml_only = out[out["study_filter_type"] == "Empirical Study - Machine Learning"]
print(ml_only["validation_strategy"].value_counts().to_string())

print("\n=== CLASSIFICATION CONFIDENCE ===")
print(out["classification_confidence"].value_counts().to_string())

print("\n=== ML STUDIES — TASK DISTRIBUTION ===")
print(ml_only["primary_task_label"].value_counts().to_string())

print("\n=== ML STUDIES — PREDICTION TARGET LEVEL ===")
print(ml_only["prediction_target_level"].value_counts().to_string())

# ── 14. Machine-readable audit ────────────────────────────────────────────────
audit = {
    "phase": 2,
    "date": "2026-10-14",
    "input_file": "outputs/ILSA_Meta_Analysis_Dataset_CLEAN.xlsx",
    "total_studies": int(len(out)),
    "task_distribution": {k: int(v) for k, v in out["primary_task_code"].value_counts().items()},
    "task_labels": TASK_LABELS,
    "ml_study_count_confirmed": int(len(ml_only)),
    "temporal_forecasting_status_counts": {
        k: int(v) for k, v in out["temporal_forecasting_status"].value_counts().items()
    },
    "confirmed_future_cycle_forecasting": int(
        (out["temporal_forecasting_status"] == "confirmed_future_cycle_forecasting").sum()
    ),
    "cross_cycle_prediction_unverified": int(
        (out["temporal_forecasting_status"] == "cross_cycle_prediction_not_fully_verified").sum()
    ),
    "manual_review_required_count": int(out["manual_review_required"].sum()),
    "potential_duplicates": int(out["potential_duplicate"].sum()),
    "program_coverage": {k: int(v) for k, v in zip(
        prog_summary["canonical_program"], prog_summary["n_studies"]
    )},
    "classification_method": "rule_based_extraction_not_verified_against_source_articles",
    "limitations": [
        "temporal_forecasting_status requires manual verification against source articles",
        "validation_strategy inferred from free-text fields — may be incomplete",
        "271 DOI-less records use file_name as primary key",
        "10 potential duplicate records flagged for manual review",
        "cross_cycle classification requires article-level inspection to confirm",
    ]
}

with open(OUTPUT_DIR / "phase2_audit.json", "w") as f:
    json.dump(audit, f, indent=2)
print("\nSaved phase2_audit.json")
