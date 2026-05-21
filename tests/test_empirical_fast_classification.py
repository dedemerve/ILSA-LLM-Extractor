"""Fast classification boxes for residual empirical Uncategorized rows."""

from __future__ import annotations

from src.enrichment.canonical_taxonomy import (
    ASSESSMENT_METHODOLOGY,
    CIVIC_ENGAGEMENT,
    LABOR_MARKET_OUTCOMES,
    SCHOOL_EFFICIENCY_CLIMATE,
    THEORETICAL_META_SYNTHESIS,
    UNCATEGORIZED,
    WELLBEING_PSYCHOLOGICAL,
    map_variable_to_canonical,
)


def test_civic_engagement_voting() -> None:
    assert (
        map_variable_to_canonical("Expected electoral participation (ELECPART)")
        == CIVIC_ENGAGEMENT
    )


def test_school_efficiency_segregation() -> None:
    assert (
        map_variable_to_canonical(
            "Social segregation across schools (Dissimilarity Index D)"
        )
        == SCHOOL_EFFICIENCY_CLIMATE
    )


def test_wellbeing_boredom() -> None:
    assert map_variable_to_canonical("Boredom (ATM scale)") == WELLBEING_PSYCHOLOGICAL


def test_assessment_qmatrix() -> None:
    assert map_variable_to_canonical("Q-matrix recovery rate (QRR)") == ASSESSMENT_METHODOLOGY


def test_labor_market_wages() -> None:
    assert map_variable_to_canonical("Log hourly wages") == LABOR_MARKET_OUTCOMES
    assert (
        map_variable_to_canonical("Automation risk (individual-level, 0–1)")
        == LABOR_MARKET_OUTCOMES
    )


def test_heuristic_not_theoretical_when_empirical_context() -> None:
    """Migration label still maps to meta; civic label does not."""
    assert (
        map_variable_to_canonical("Primary study outcome (heuristic migration)")
        == THEORETICAL_META_SYNTHESIS
    )
    assert (
        map_variable_to_canonical("Political knowledge")
        == CIVIC_ENGAGEMENT
    )
    assert map_variable_to_canonical("Log hourly wages") == LABOR_MARKET_OUTCOMES
