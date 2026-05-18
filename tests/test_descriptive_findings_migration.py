"""Descriptive main_findings from outcome_summary (no API, grounded text only)."""

from src.extractors.gpt_extractor import (
    _build_descriptive_findings_from_outcome,
    _build_descriptive_standardized_conclusion,
    _infer_descriptive_target_from_outcome,
)


def test_infer_multiple_targets_from_performed_in_domains():
    from src.extractors.gpt_extractor import _infer_descriptive_targets_from_outcome

    outcome = (
        "This OECD PISA 2022 factsheet summarizes how students performed in "
        "mathematics, reading, and science."
    )
    targets = _infer_descriptive_targets_from_outcome(outcome)
    assert len(targets) == 3
    assert any("mathematics" in t.lower() for t in targets)


def test_infer_descriptive_target_from_mean_score():
    outcome = (
        "The mean mathematics achievement score is 482 points, "
        "below the OECD average."
    )
    target = _infer_descriptive_target_from_outcome(outcome)
    assert target is not None
    assert "mathematics" in target.lower()


def test_descriptive_finding_skips_when_target_not_in_text():
    outcome = "Short text without measurable outcome."
    assert _build_descriptive_findings_from_outcome(
        {"outcome_summary": outcome},
        {"publication_type": "report"},
    ) == []


def test_descriptive_conclusion_uses_outcome_not_generic_migration():
    outcome = (
        "PISA 2022 results show reading performance at Level 3 for most students. "
        "Scores are significantly below the OECD average in rural schools."
    )
    text = _build_descriptive_standardized_conclusion(
        "PISA 2022",
        ["ESCS"],
        "reading performance",
        outcome,
    )
    assert "significantly below the OECD average" in text
    assert "heuristic migration" not in text.lower()
    assert "This indicates that" in text
