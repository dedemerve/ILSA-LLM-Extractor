"""Tests for main_findings migration and schema requirements (no API)."""

import pytest
from pydantic import ValidationError

from src.extractors.gpt_extractor import (
    GPTExtractor,
    _normalize_findings_fields,
    _upgrade_legacy_and_empty_findings,
)
from src.schemas.findings_validation import (
    article_requires_main_findings,
    is_official_report_document,
)
from src.schemas.models import (
    ILSAArticleMetadata,
    StructuredFinding,
    validate_public_article_json,
)


def _minimal_data(**overrides):
    base = {
        "survey_design": {
            "student_weights_used": False,
            "weight_fields_interpretation": "Weights not used in this illustrative test record.",
        },
        "plausible_values_handling": "not_reported",
        "missing_data_handling": "not_reported",
        "handling_not_reported_explanation": "Test fixture.",
        "sample_details": {
            "countries": [],
            "sample_filtering_criteria": "PISA 2022 micro-data for selected countries.",
        },
        "ml_techniques": {"primary": "XGBoost", "all_techniques": ["XGBoost"]},
        "confounders_identified": [
            {
                "variable_code": "ESCS",
                "variable_name": "Socioeconomic status",
                "category": "socioeconomic",
            }
        ],
        "main_findings": [],
        "outcome_summary": (
            "Using PISA 2022 data, XGBoost achieved R²=0.51 predicting mathematics "
            "achievement with ESCS as the strongest predictor."
        ),
        "research_design_type": "predictive",
    }
    base.update(overrides)
    return base


def test_article_requires_main_findings_with_outcome():
    data = _minimal_data()
    assert article_requires_main_findings(data, {"source_category": "peer_reviewed_research"})


def test_migration_skips_skeleton_for_official_national_report():
    data = _minimal_data(
        ml_techniques={"primary": None, "all_techniques": []},
        main_findings=[],
        outcome_summary=(
            "This TIMSS 2023 national report summarizes mathematics and science "
            "achievement for participating countries using complex survey weights."
        ),
    )
    meta = {
        "publication_type": "report",
        "source_category": "technical_report",
        "title": "TIMSS 2023 National Report",
        "file_name": "TIMSS_2023_National_Report.pdf",
    }
    assert is_official_report_document(data, meta)
    _normalize_findings_fields(data, meta)
    assert data["main_findings"] == []


def test_sanitize_fills_descriptive_findings_for_official_report():
    payload = {
        "metadata": {
            "file_name": "005c9979-en.pdf",
            "title": "PISA 2022 Creative Thinking Factsheet – Slovenia",
            "publication_type": "report",
            "source_category": "technical_report",
        },
        "data": _minimal_data(
            ml_techniques={"primary": None, "all_techniques": []},
            confounders_identified=[],
            main_findings=[],
            outcome_summary=(
                "This OECD PISA 2022 factsheet summarises Slovenia's performance on "
                "the PISA Creative Thinking cognitive test. The mean creative thinking "
                "score is 30/60, significantly below the OECD average (33). "
                "Correlations are 0.6 (creative thinking–math) and 0.59 "
                "(creative thinking–reading)."
            ),
            research_design_type="exploratory",
        ),
    }
    sanitized = GPTExtractor._sanitize(payload)
    mf = sanitized["data"]["main_findings"]
    assert len(mf) >= 1
    assert any("creative thinking" in row["target_variable"].lower() for row in mf)
    row = mf[0]
    assert "PISA" in row["dataset_used"]
    assert "0.6" in row["performance_metrics"] or "30/60" in row["performance_metrics"]
    assert "This indicates that" in row["standardized_conclusion"]


def test_migration_fills_empty_from_outcome_summary():
    data = _minimal_data()
    meta = {"source_category": "peer_reviewed_research", "title": "PISA 2022 Math Study"}
    _normalize_findings_fields(data, meta)
    assert len(data["main_findings"]) == 1
    row = data["main_findings"][0]
    assert "PISA" in row["dataset_used"]
    assert row["target_variable"]
    assert "legacy migration" not in row["dataset_used"].lower()
    assert "ESCS" in row["top_predictors"][0] or row["top_predictors"]


def test_upgrade_legacy_placeholder():
    data = _minimal_data(
        main_findings=[{
            "dataset_used": "ILSA dataset not specified (legacy migration)",
            "target_variable": "Primary outcome (legacy migration)",
            "top_predictors": [],
            "performance_metrics": "Not reported",
            "standardized_conclusion": "Legacy text only.",
        }],
    )
    meta = {"source_category": "review_article", "title": "PISA 2022 process data review"}
    _upgrade_legacy_and_empty_findings(data, meta)
    _normalize_findings_fields(data, meta)
    assert "legacy migration" not in data["main_findings"][0]["dataset_used"].lower()
    assert "PISA" in data["main_findings"][0]["dataset_used"]


def test_public_json_root_not_metadata_subdict():
    """On-disk files are {metadata, data}; root must validate as ILSAArticleMetadata."""
    payload = {
        "metadata": {
            "file_name": "x.pdf",
            "source_category": "review_article",
        },
        "data": _minimal_data(
            main_findings=[{
                "dataset_used": "PISA 2022",
                "target_variable": "Math",
                "top_predictors": ["ESCS"],
                "performance_metrics": "R²=0.4",
                "standardized_conclusion": (
                    "Using PISA 2022 data, ESCS predicted mathematics achievement."
                ),
            }],
            outcome_summary="Review with no empirical predictive results reported.",
        ),
    }
    validate_public_article_json(payload)
    with pytest.raises(ValidationError):
        ILSAArticleMetadata.model_validate(payload["metadata"])


def test_sanitize_then_validates_schema():
    payload = {
        "metadata": {
            "file_name": "test.pdf",
            "title": "PISA 2022 resilience",
            "source_category": "peer_reviewed_research",
        },
        "data": _minimal_data(),
    }
    sanitized = GPTExtractor._sanitize(payload)
    model = ILSAArticleMetadata.model_validate(sanitized)
    assert len(model.data.main_findings) >= 1


def test_schema_rejects_required_empty_without_migration():
    finding = StructuredFinding(
        dataset_used="TIMSS 2019",
        target_variable="Science score",
        top_predictors=["ESCS"],
        performance_metrics="R²: 0.4",
        standardized_conclusion=(
            "Using TIMSS 2019 data, the study leveraged ESCS to predict science score, "
            "finding moderate association."
        ),
    )
    payload = {
        "metadata": {
            "file_name": "x.pdf",
            "source_category": "peer_reviewed_research",
        },
        "data": _minimal_data(
            main_findings=[],
            outcome_summary=(
                "XGBoost on PISA 2022 predicted math achievement with strong fit."
            ),
        ),
    }
    with pytest.raises(ValidationError):
        ILSAArticleMetadata.model_validate(payload)

    payload["data"]["main_findings"] = [finding.model_dump()]
    ILSAArticleMetadata.model_validate(payload)
