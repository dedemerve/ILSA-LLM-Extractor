"""OECD resanitize backfills: title, DOI, research_design_type (no API)."""

from src.extractors.gpt_extractor import GPTExtractor, _title_from_file_name
from src.schemas.models import validate_public_article_json


def _minimal_report_payload(*, title=None, doi=None, file_name="report.pdf", outcome=""):
    return {
        "metadata": {
            "file_name": file_name,
            "title": title,
            "authors": [],
            "year": 2009,
            "doi": doi,
            "venue": None,
            "publication_type": "report",
            "source_category": "technical_report",
        },
        "data": {
            "survey_design": {
                "student_weights_used": False,
                "weight_fields_interpretation": "PISA uses complex sampling weights.",
            },
            "plausible_values_handling": "not_applicable",
            "missing_data_handling": "not_reported",
            "handling_not_reported_explanation": "Framework document.",
            "sample_details": {
                "total_students": None,
                "countries": [],
                "sample_filtering_criteria": "Not applicable.",
            },
            "ml_techniques": {"primary": None, "all_techniques": []},
            "confounders_identified": [],
            "main_findings": [],
            "outcome_summary": outcome or (
                "PISA 2009 international report on student performance. "
                "https://doi.org/10.1787/9789264112995-en"
            ),
            "research_design_type": None,
            "null_fields_interpretation": "Official OECD report.",
        },
    }


def test_title_from_oecd_hash_filename():
    assert _title_from_file_name("e68ee91a-en.pdf") == "OECD document e68ee91a-en"


def test_sanitize_backfills_doi_from_outcome_summary():
    payload = _minimal_report_payload(doi=None)
    sanitized = GPTExtractor._sanitize(payload)
    assert sanitized["metadata"]["doi"] == "10.1787/9789264112995-en"
    validate_public_article_json(sanitized)


def test_sanitize_coerces_null_research_design_for_reports():
    payload = _minimal_report_payload()
    sanitized = GPTExtractor._sanitize(payload)
    assert sanitized["data"]["research_design_type"] == "exploratory"
    validate_public_article_json(sanitized)
