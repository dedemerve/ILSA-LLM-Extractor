"""SurveyDesign bool coercion for LLM placeholder strings (no API)."""

import pytest
from pydantic import ValidationError

from src.extractors.gpt_extractor import GPTExtractor
from src.schemas.models import SurveyDesign, coerce_optional_bool


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("N/A", None),
        ("n/a", None),
        ("na", None),
        ("not applicable", None),
        ("not_applicable", None),
        ("", None),
        ("none", None),
        ("null", None),
        ("true", True),
        ("yes", True),
        ("1", True),
        ("false", False),
        ("no", False),
        ("0", False),
        ("maybe", None),
        ("unknown", None),
        (True, True),
        (False, False),
        (None, None),
    ],
)
def test_coerce_optional_bool(raw, expected):
    assert coerce_optional_bool(raw) is expected


def test_survey_design_accepts_na_strings():
    sd = SurveyDesign.model_validate({
        "student_weights_used": "N/A",
        "replicate_weights_used": "na",
        "weight_fields_interpretation": "IEA framework; weighting not discussed.",
    })
    assert sd.student_weights_used is None
    assert sd.replicate_weights_used is None


def test_sanitize_coerces_survey_design_bools():
    payload = {
        "metadata": {"file_name": "report.pdf"},
        "data": {
            "survey_design": {
                "student_weights_used": "N/A",
                "replicate_weights_used": "not applicable",
                "weight_fields_interpretation": "Technical report on TIMSS design.",
            },
            "plausible_values_handling": "not_reported",
            "missing_data_handling": "not_reported",
            "sample_details": {
                "countries": [],
                "sample_filtering_criteria": "Full international sample as reported.",
            },
            "ml_techniques": {"primary": None, "all_techniques": []},
            "confounders_identified": [],
            "main_findings": [],
            "outcome_summary": "Framework document without student-level ML.",
            "research_design_type": "exploratory",
        },
    }
    sanitized = GPTExtractor._sanitize(payload)
    sd = sanitized["data"]["survey_design"]
    assert sd["student_weights_used"] is None
    assert sd["replicate_weights_used"] is None
    SurveyDesign.model_validate(sd)


def test_survey_design_unknown_type_coerces_to_none():
    sd = SurveyDesign.model_validate({
        "student_weights_used": {"bad": "dict"},
        "weight_fields_interpretation": "x",
    })
    assert sd.student_weights_used is None
