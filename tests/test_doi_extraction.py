"""DOI regex extraction and post-process backfill tests (no API)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.extractors.gpt_extractor import GPTExtractor
from src.extractors.pdf_processor import extract_doi_from_document, extract_dois_from_text, process_pdf
from src.schemas.models import ILSAArticleMetadata, MetadataBlock, DataBlock, SurveyDesign, SampleDetails, MLTechniques


ARTICLES_DIR = Path.home() / "Desktop" / "articles"


def test_extract_dois_from_text_handles_urls_and_labels():
    text = (
        "Published online: https://doi.org/10.1007/s10639-023-11908-0\n"
        "DOI: 10.1080/09500693.2024.2359099\n"
    )
    dois = extract_dois_from_text(text)
    assert "10.1007/s10639-023-11908-0" in dois
    assert "10.1080/09500693.2024.2359099" in dois


@pytest.mark.skipif(
    not ARTICLES_DIR.exists(),
    reason="~/Desktop/articles not available",
)
@pytest.mark.parametrize(
    "prefix,expected_doi",
    [
        ("4. Zheng", "10.1007/s10639-023-11908-0"),
        ("10. Aydog", "10.21031/epod.1525454"),
        ("3. Zhou", "10.1016/j.heliyon.2024.e35945"),
    ],
)
def test_process_pdf_extracts_doi_from_real_articles(prefix: str, expected_doi: str):
    pdf_path = next(ARTICLES_DIR.glob(f"{prefix}*.pdf"))
    processed = process_pdf(pdf_path, source_database="articles")
    assert processed.metadata.get("extracted_doi") == expected_doi
    assert expected_doi in (processed.metadata.get("doi_candidates") or [])


def test_post_process_fills_null_doi_from_pdf_hint():
    class MockPDF:
        metadata = {
            "extracted_doi": "10.1007/s10639-023-11908-0",
            "doi_candidates": ["10.1007/s10639-023-11908-0"],
        }

    extraction = ILSAArticleMetadata(
        metadata=MetadataBlock(
            file_name="test.pdf",
            title="Test",
            authors=[],
            year=2023,
            doi=None,
            venue="Education and Information Technologies",
            publication_type="journal",
            open_access=None,
            source_category="peer_reviewed_research",
        ),
        data=DataBlock(
            survey_design=SurveyDesign(
                student_weights_used=False,
                weight_fields_interpretation="Test.",
            ),
            plausible_values_handling="rubin_rules",
            missing_data_handling="listwise_deletion",
            sample_details=SampleDetails(
                countries=[],
                sample_filtering_criteria="Test sample — no filtering reported.",
            ),
            ml_techniques=MLTechniques(primary="SVM", all_techniques=["SVM"]),
            confounders_identified=[],
            main_findings=[],
            outcome_summary="Test outcome narrative summary.",
            research_design_type="predictive",
        ),
    )
    GPTExtractor._post_process_model(extraction, MockPDF())  # type: ignore[arg-type]
    assert extraction.metadata.doi == "10.1007/s10639-023-11908-0"


def test_build_user_message_includes_doi_hint(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-for-unit-tests")
    class MockPDF:
        file_name = "4. Zheng.pdf"
        source_database = "articles"
        sections = {}
        extraction_text = "Abstract only body."
        metadata = {
            "extracted_doi": "10.1007/s10639-023-11908-0",
            "doi_candidates": ["10.1007/s10639-023-11908-0"],
        }

    extractor = GPTExtractor()
    parts = extractor._build_user_message(MockPDF())  # type: ignore[arg-type]
    doc_text = parts[0]["text"]
    assert "EXTRACTED_DOI_HINT" in doc_text
    assert "10.1007/s10639-023-11908-0" in doc_text
