"""Tests for Theoretical_and_Meta_Synthesis canonical isolation."""

from __future__ import annotations

from src.enrichment.canonical_taxonomy import (
    METADATA_FILTER_META,
    THEORETICAL_META_SYNTHESIS,
    UNCATEGORIZED,
    is_theoretical_meta_synthesis_label,
    map_variable_to_canonical,
    metadata_filter_flag_for,
)


def test_heuristic_migration_maps_to_meta_synthesis() -> None:
    label = "Primary study outcome (heuristic migration)"
    assert map_variable_to_canonical(label) == THEORETICAL_META_SYNTHESIS
    assert metadata_filter_flag_for(THEORETICAL_META_SYNTHESIS) == METADATA_FILTER_META


def test_literature_synthesis_outcome_maps_to_meta() -> None:
    label = "Literature synthesis outcome (not student-level prediction)"
    assert map_variable_to_canonical(label) == THEORETICAL_META_SYNTHESIS


def test_framework_study_unmapped_prose_is_meta() -> None:
    label = "how TIMSS 2019 sampling weights are applied in national reports"
    assert (
        map_variable_to_canonical(
            label,
            study_filter="Technical/Assessment Framework",
        )
        == THEORETICAL_META_SYNTHESIS
    )


def test_empirical_math_achievement_stays_empirical() -> None:
    label = "Mathematics achievement (PV1MATH)"
    assert map_variable_to_canonical(
        label,
        study_filter="Empirical Study - Traditional Statistics",
    ) not in (THEORETICAL_META_SYNTHESIS, UNCATEGORIZED)


def test_is_meta_helper_detects_prose_fragment() -> None:
    assert is_theoretical_meta_synthesis_label(
        "83% of between-school variance and 33% of between-student variance achievement"
    )
