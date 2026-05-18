"""Tests for report sanitize, conclusion meaning clause, bool coercion."""

import pytest

from src.extractors.gpt_extractor import (
    GPTExtractor,
    _build_standardized_conclusion,
    _ensure_conclusion_meaning_clause,
    _is_generic_weight_interpretation,
    _weight_fields_interpretation_from_outcome,
)
from src.schemas.findings_validation import (
    article_requires_main_findings,
    is_official_report_document,
)
from src.schemas.models import SurveyDesign, coerce_optional_bool


def _report_payload(**data_overrides):
    data = {
        "survey_design": {
            "student_weights_used": "N/A",
            "replicate_weights_used": "unknown",
            "weight_fields_interpretation": "",
        },
        "plausible_values_handling": "not_applicable",
        "missing_data_handling": "not_reported",
        "sample_details": {
            "countries": [],
            "sample_filtering_criteria": "Not applicable to framework document.",
        },
        "ml_techniques": {"primary": None, "all_techniques": []},
        "confounders_identified": [{"variable_code": "ESCS", "variable_name": "ESCS", "category": "socioeconomic"}],
        "main_findings": [{"dataset_used": "TIMSS", "target_variable": "x", "top_predictors": [], "performance_metrics": "n/a", "standardized_conclusion": "bad"}],
        "outcome_summary": (
            "This TIMSS 2023 assessment framework describes a two-stage stratified cluster "
            "sample design with school and classroom sampling. Plausible values are produced "
            "via IRT scaling for mathematics and science achievement. Analysts should apply "
            "student sampling weights when estimating population parameters."
        ),
        "research_design_type": "exploratory",
        **data_overrides,
    }
    return {
        "metadata": {
            "file_name": "TIMSS-2023-Assessment-Framework.pdf",
            "publication_type": "report",
            "source_category": "technical_report",
            "title": "TIMSS 2023 Assessment Framework",
        },
        "data": data,
    }


def test_is_official_report_document_detects_framework():
    payload = _report_payload()
    assert is_official_report_document(payload["data"], payload["metadata"])


def test_article_requires_main_findings_false_for_official_report():
    payload = _report_payload()
    assert not article_requires_main_findings(payload["data"], payload["metadata"])


def test_is_official_report_document_detects_national_report_title():
    payload = _report_payload(
        outcome_summary=(
            "This TIMSS 2023 national report presents achievement results for "
            "grade 4 and grade 8 students using the international sampling design."
        ),
    )
    payload["metadata"]["title"] = "TIMSS 2023 National Report for Norway"
    payload["metadata"]["publication_type"] = "report"
    assert is_official_report_document(payload["data"], payload["metadata"])


def test_sanitize_report_clears_findings_and_fills_weight_interpretation():
    sanitized = GPTExtractor._sanitize(_report_payload())
    data = sanitized["data"]
    assert data["main_findings"] == []
    assert data["confounders_identified"] == []
    wfi = data["survey_design"]["weight_fields_interpretation"]
    assert wfi and len(wfi) > 80
    assert "stratified" in wfi.lower() or "sampling" in wfi.lower()
    assert data["survey_design"]["student_weights_used"] is None
    assert data["survey_design"]["replicate_weights_used"] is None
    SurveyDesign.model_validate(data["survey_design"])


def test_ensure_conclusion_meaning_clause_appends_implication():
    base = (
        "Using PISA 2022 data, the study leveraged ESCS to predict mathematics "
        "achievement, finding that higher ESCS was associated with higher scores."
    )
    out = _ensure_conclusion_meaning_clause(base)
    assert "this indicates that" in out.lower()


def test_build_standardized_conclusion_includes_meaning():
    out = _build_standardized_conclusion(
        "PISA 2022",
        ["ESCS"],
        "Mathematics achievement",
        effect_hint="higher ESCS predicted higher mathematics scores",
    )
    assert "this indicates that" in out.lower()


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("N/A", None),
        ("unknown", None),
        ("false", False),
        (1, True),
    ],
)
def test_coerce_optional_bool_extended(raw, expected):
    assert coerce_optional_bool(raw) is expected


def test_weight_interpretation_from_outcome_not_generic():
    text = _weight_fields_interpretation_from_outcome(
        "TIMSS uses a two-stage cluster sample and W_FSTUWT student weights."
    )
    assert not _is_generic_weight_interpretation(text)
