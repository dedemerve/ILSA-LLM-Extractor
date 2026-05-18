"""Official-report literal coercion (invalid LLM strings → schema literals)."""

import pytest
from pydantic import ValidationError

from src.extractors.gpt_extractor import GPTExtractor
from src.schemas.findings_validation import (
    _coerce_report_literals,
    should_apply_report_literal_coercion,
)
from src.schemas.models import validate_public_article_json


def _broken_report_payload():
    return {
        "metadata": {
            "file_name": "TIMSS-2023-Assessment-Framework.pdf",
            "title": "TIMSS 2023 Assessment Framework",
            "publication_type": "Official Report",
            "source_category": "Official technical report",
        },
        "data": {
            "survey_design": {
                "student_weights_used": "N/A",
                "replicate_weights_used": "unknown",
                "weight_fields_interpretation": (
                    "TIMSS 2023 uses a two-stage stratified cluster sample design. "
                    "Student sampling weights such as W_FSTUWT should be applied when "
                    "estimating population parameters from the international database."
                ),
            },
            "plausible_values_handling": "Official Report",
            "missing_data_handling": "Not applicable to framework",
            "handling_not_reported_explanation": "",
            "sample_details": {
                "total_students": None,
                "countries": [],
                "sample_filtering_criteria": "Not applicable to framework document.",
            },
            "ml_techniques": {"primary": None, "all_techniques": []},
            "confounders_identified": [
                {
                    "variable_code": "N/A",
                    "variable_name": "ESCS",
                    "category": "socioeconomic",
                },
            ],
            "main_findings": [
                {
                    "dataset_used": "TIMSS 2023",
                    "target_variable": "Mathematics",
                    "top_predictors": [],
                    "performance_metrics": "n/a",
                    "standardized_conclusion": "Invalid row for framework.",
                },
            ],
            "outcome_summary": (
                "This TIMSS 2023 assessment framework describes the two-stage stratified "
                "cluster sampling design, IRT scaling for mathematics and science, and "
                "guidance for analysts using the international database and plausible values."
            ),
            "research_design_type": "predictive",
            "null_fields_interpretation": "",
        },
    }


def test_should_apply_report_literal_coercion_on_invalid_enums():
    payload = _broken_report_payload()
    assert should_apply_report_literal_coercion(payload["data"], payload["metadata"])


def test_coerce_report_literals_maps_official_report_to_not_applicable():
    payload = _broken_report_payload()
    data = payload["data"]
    meta = payload["metadata"]
    _coerce_report_literals(data, meta)
    assert meta["publication_type"] == "report"
    assert meta["source_category"] == "technical_report"
    assert data["plausible_values_handling"] == "not_applicable"
    assert data["missing_data_handling"] == "not_reported"
    assert data["research_design_type"] == "exploratory"
    assert data["main_findings"] == []
    assert data["confounders_identified"] == []
    assert data["handling_not_reported_explanation"]
    assert data["null_fields_interpretation"]


def test_sanitize_report_passes_pydantic_validation():
    sanitized = GPTExtractor._sanitize(_broken_report_payload())
    record = validate_public_article_json(sanitized)
    assert record.metadata.publication_type == "report"
    assert record.data.plausible_values_handling == "not_applicable"
    assert record.data.main_findings == []


def test_sanitize_rejects_before_coercion_without_fix():
    payload = _broken_report_payload()
    with pytest.raises(ValidationError):
        validate_public_article_json(payload)


def test_coerce_report_literals_sets_null_research_design_to_exploratory():
    payload = _broken_report_payload()
    data = payload["data"]
    data["research_design_type"] = None
    _coerce_report_literals(data, payload["metadata"])
    assert data["research_design_type"] == "exploratory"


def test_sanitize_null_research_design_on_official_report():
    payload = _broken_report_payload()
    payload["data"]["research_design_type"] = None
    sanitized = GPTExtractor._sanitize(payload)
    assert sanitized["data"]["research_design_type"] == "exploratory"


def test_failed_extraction_payload_passes_validation():
    from ilsa_pipeline.utils.storage import failed_extraction_payload_for_disk
    from src.extractors.gpt_extractor import ExtractionResult

    result = ExtractionResult(
        file_name="9789264112995-en.pdf",
        success=False,
        extraction=None,
        error="API error: connection reset",
        input_tokens=0,
        output_tokens=0,
        cost_usd=0.0,
        duration_seconds=0.0,
    )
    payload = failed_extraction_payload_for_disk(result)
    record = validate_public_article_json(payload)
    assert record.data.research_design_type == "exploratory"
    assert record.data.outcome_summary.startswith("__EXTRACTION_FAILED__:")
