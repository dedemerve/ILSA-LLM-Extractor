"""
Post-processing: tüm JSON çıktılarını tarar, sabit şemalı bir master tablo üretir.
Çıktı: outputs/master_structured_table.xlsx  (3 sheet: Papers, Findings, Confounders)
"""

import json
import os
import glob
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent.parent
JSON_DIRS = [
    ROOT / "outputs/ilsa_survey_articles/json",
    ROOT / "outputs/IEA",
    ROOT / "outputs/OECD",
    ROOT / "outputs/Scopus",
    ROOT / "outputs/Web of Science",
]
OUT = ROOT / "outputs/master_structured_table.xlsx"


def collect_json_paths():
    paths = []
    for d in JSON_DIRS:
        if d.exists():
            paths.extend(d.rglob("*.json"))
    return sorted(set(paths))


def safe(val, default=""):
    if val is None:
        return default
    import re
    _ILLEGAL = re.compile(r"[\x00-\x08\x0b-\x0c\x0e-\x1f]")
    clean = lambda s: _ILLEGAL.sub("", str(s)).strip()
    if isinstance(val, list):
        return "; ".join(clean(v) for v in val if v)
    return clean(val)


def parse_paper(path: Path, corpus_source: str):
    with open(path, encoding="utf-8") as f:
        d = json.load(f)

    meta = d.get("metadata", {})
    data = d.get("data", {})
    survey = data.get("survey_design", {})
    sample = data.get("sample_details", {})
    ml = data.get("ml_techniques", {})

    paper = {
        "file_name": safe(meta.get("file_name"), path.stem),
        "corpus_source": corpus_source,
        "title": safe(meta.get("title")),
        "authors": safe(meta.get("authors")),
        "year": meta.get("year"),
        "doi": safe(meta.get("doi")),
        "venue": safe(meta.get("venue")),
        "publication_type": safe(meta.get("publication_type")),
        "source_category": safe(meta.get("source_category")),
        "open_access": meta.get("open_access"),
        # survey design
        "student_weights_used": survey.get("student_weights_used"),
        "replicate_weights_used": survey.get("replicate_weights_used"),
        "weight_variable_name": safe(survey.get("weight_variable_name")),
        "weight_interpretation": safe(survey.get("weight_fields_interpretation")),
        "plausible_values_handling": safe(data.get("plausible_values_handling")),
        "missing_data_handling": safe(data.get("missing_data_handling")),
        "handling_not_reported_explanation": safe(data.get("handling_not_reported_explanation")),
        # sample
        "total_students": sample.get("total_students"),
        "countries": safe(sample.get("countries")),
        "sample_filtering_criteria": safe(sample.get("sample_filtering_criteria")),
        # methods
        "ml_primary": safe(ml.get("primary")),
        "ml_all_techniques": safe(ml.get("all_techniques")),
        "research_design_type": safe(data.get("research_design_type")),
        "outcome_summary": safe(data.get("outcome_summary")),
    }

    findings = []
    for i, fn in enumerate(data.get("main_findings", []), 1):
        findings.append({
            "file_name": paper["file_name"],
            "corpus_source": corpus_source,
            "year": paper["year"],
            "countries": paper["countries"],
            "finding_index": i,
            "dataset_used": safe(fn.get("dataset_used")),
            "target_variable": safe(fn.get("target_variable")),
            "top_predictors": safe(fn.get("top_predictors")),
            "performance_metrics": safe(fn.get("performance_metrics")),
            "standardized_conclusion": safe(fn.get("standardized_conclusion")),
            "ml_primary": paper["ml_primary"],
            "publication_type": paper["publication_type"],
            "source_category": paper["source_category"],
        })

    confounders = []
    for cv in data.get("confounders_identified", []):
        confounders.append({
            "file_name": paper["file_name"],
            "corpus_source": corpus_source,
            "year": paper["year"],
            "variable_code": safe(cv.get("variable_code")),
            "variable_name": safe(cv.get("variable_name")),
            "category": safe(cv.get("category")),
            "description": safe(cv.get("description")),
            "predictor_level": safe(cv.get("predictor_level")),
            "predictor_category": safe(cv.get("predictor_category")),
        })

    return paper, findings, confounders


def main():
    paths = collect_json_paths()
    print(f"Bulunan JSON sayısı: {len(paths)}")

    papers, all_findings, all_confounders = [], [], []
    errors = []

    for path in paths:
        corpus = path.parent.name
        try:
            p, f, c = parse_paper(path, corpus)
            papers.append(p)
            all_findings.extend(f)
            all_confounders.extend(c)
        except Exception as e:
            errors.append((str(path), str(e)))

    df_papers = pd.DataFrame(papers)
    df_findings = pd.DataFrame(all_findings)
    df_confounders = pd.DataFrame(all_confounders)

    print(f"Papers     : {len(df_papers)} satır")
    print(f"Findings   : {len(df_findings)} satır")
    print(f"Confounders: {len(df_confounders)} satır")
    if errors:
        print(f"Hatalı dosya: {len(errors)}")
        for p, e in errors[:5]:
            print(f"  {p}: {e}")

    with pd.ExcelWriter(OUT, engine="openpyxl") as writer:
        df_papers.to_excel(writer, sheet_name="Papers", index=False)
        df_findings.to_excel(writer, sheet_name="Findings", index=False)
        df_confounders.to_excel(writer, sheet_name="Confounders", index=False)

        # her sheet'te autofilter + sütun genişliği
        for sheet_name, df in [("Papers", df_papers), ("Findings", df_findings), ("Confounders", df_confounders)]:
            ws = writer.sheets[sheet_name]
            ws.auto_filter.ref = ws.dimensions
            for col in ws.columns:
                max_len = max((len(str(c.value)) if c.value else 0) for c in col)
                ws.column_dimensions[col[0].column_letter].width = min(max_len + 2, 50)

    print(f"\nKaydedildi: {OUT}")


if __name__ == "__main__":
    main()
