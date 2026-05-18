"""Offline backfill of countries/total_students from extracted text only."""

from src.extractors.gpt_extractor import (
    GPTExtractor,
    _backfill_countries_from_extracted_text,
    _backfill_total_students_from_extracted_text,
)


def test_backfill_countries_from_outcome_text():
    data = {
        "sample_details": {"countries": [], "sample_filtering_criteria": "test"},
        "outcome_summary": (
            "The study analyzes PISA 2018 data for Germany, Finland, and Norway."
        ),
        "ml_techniques": {"primary": "Random Forest", "all_techniques": ["Random Forest"]},
    }
    meta = {"source_category": "peer_reviewed_research", "publication_type": "journal"}
    _backfill_countries_from_extracted_text(data, meta)
    codes = {c["country_code"] for c in data["sample_details"]["countries"]}
    assert codes >= {"DEU", "FIN", "NOR"}


def test_backfill_total_students_from_n_equals():
    data = {
        "sample_details": {"countries": [], "total_students": None},
        "outcome_summary": "After exclusions, the analytic sample was N = 5,211 students.",
    }
    _backfill_total_students_from_extracted_text(data, {})
    assert data["sample_details"]["total_students"] == 5211


def test_sanitize_strips_catalog_folder_path():
    payload = {
        "metadata": {
            "file_name": "x.pdf",
            "catalog_folder_path": "Scopus/TIMSS 2019",
            "source_category": "peer_reviewed_research",
        },
        "data": {
            "survey_design": {
                "student_weights_used": False,
                "weight_fields_interpretation": "Weights not discussed in this test fixture.",
            },
            "plausible_values_handling": "not_reported",
            "missing_data_handling": "not_reported",
            "handling_not_reported_explanation": "Test.",
            "sample_details": {
                "countries": [],
                "sample_filtering_criteria": "Full sample.",
            },
            "ml_techniques": {"primary": None, "all_techniques": []},
            "confounders_identified": [],
            "main_findings": [],
            "outcome_summary": "Descriptive review without student-level modeling.",
            "research_design_type": "exploratory",
        },
    }
    out = GPTExtractor._sanitize(payload)
    assert "catalog_folder_path" not in (out.get("metadata") or {})
