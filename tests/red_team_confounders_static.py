"""
Static red-team for confounder schema + normalization (NO PDF/API).

Verifies:
  - Changes are in extraction layer only (not hardcoded article paths).
  - Post-process fixes lazy N/A / invalid category / legacy string confounders.
  - Pydantic rejects removed 'other' category.
  - Normalization is text-source agnostic (ProcessedPDF interface, not PDF paths).
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.extractors.gpt_extractor import (
    CONFOUNDER_CATEGORIES,
    GPTExtractor,
    _coerce_confounder_category,
    _expand_confounder_dict,
    _is_micro_process_confounder,
    _is_survey_weight_confounder,
    _normalize_findings_fields,
    _normalize_confounder_code,
    _normalize_confounder_dict,
    _normalize_confounders_list,
    _dedupe_main_findings,
    _normalize_main_findings_list,
    _split_confounder_label,
)
from src.schemas.models import Confounder, DataBlock, ILSAArticleMetadata, StructuredFinding


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"


# ── Scope: no article/desktop/json_v3 hardcoding in production src ──────────

FORBIDDEN_SRC_SNIPPETS = (
    "Desktop/articles",
    "json_v3",
    "deneme2",
    "~/Desktop",
    "run_articles",
)


def test_src_has_no_article_specific_paths():
    for py_file in SRC_ROOT.rglob("*.py"):
        text = py_file.read_text(encoding="utf-8")
        for snippet in FORBIDDEN_SRC_SNIPPETS:
            assert snippet not in text, (
                f"{py_file.relative_to(PROJECT_ROOT)} contains article-specific "
                f"path '{snippet}' — extraction logic must stay source-agnostic."
            )


def test_gpt_extractor_extract_accepts_processed_pdf_interface_only():
    source = (SRC_ROOT / "extractors" / "gpt_extractor.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    gpt_class = next(
        n for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "GPTExtractor"
    )
    extract_fn = next(
        item for item in gpt_class.body
        if isinstance(item, ast.FunctionDef) and item.name == "extract"
    )
    arg_names = [a.arg for a in extract_fn.args.args]
    assert arg_names == ["self", "processed"]
    assert "ProcessedPDF" in source  # type hint only — not a filesystem path


# ── Category enum: 13 literals, no 'other' ───────────────────────────────────

def test_confounder_category_enum_has_13_values_no_other():
    schema = Confounder.model_json_schema()
    enum_vals = schema["properties"]["category"]["enum"]
    assert len(enum_vals) == 13
    assert "other" not in enum_vals
    assert set(enum_vals) == set(CONFOUNDER_CATEGORIES)


def test_pydantic_coerces_na_to_slug_for_lee_style_indicator():
    c = Confounder(
        variable_code="N/A",
        variable_name="Economic and education indicators",
        category="system_level",
    )
    assert c.variable_code == "economic_and_education_indicators"
    assert c.variable_code.upper() != "N/A"


def test_pydantic_rejects_other_category():
    with pytest.raises(ValidationError):
        Confounder(
            variable_code="ESCS",
            variable_name="Socioeconomic status",
            category="other",
        )


# ── Anti-lazy N/A: normalization helpers ─────────────────────────────────────

@pytest.mark.parametrize(
    "lazy_code, name, expected",
    [
        ("N/A", "TF-IDF start behavior weight", "tf_idf_start_behavior_weight"),
        ("n/a", "VOTAT navigation behavior", "votat_navigation_behavior"),
        ("", "Reset action frequency", "reset_action_frequency"),
        (None, "Gender", "gender"),
        ("ESCS", "Socioeconomic status (ESCS)", "ESCS"),
        ("N/A", "Socioeconomic status (ESCS)", "ESCS"),  # paren extraction
    ],
)
def test_normalize_confounder_code_fixes_lazy_na(lazy_code, name, expected):
    assert _normalize_confounder_code(lazy_code, name) == expected


def test_coerce_other_maps_to_valid_category():
    cat = _coerce_confounder_category(
        "other",
        "TF-IDF VOTAT reset sequence",
        "N/A",
    )
    assert cat == "process_data"
    assert cat in CONFOUNDER_CATEGORIES


def test_split_confounder_label_separates_and_and_commas():
    assert _split_confounder_label("Gender and Age combined") == ["Gender", "Age"]
    assert _split_confounder_label("ESCS, HOMEPOS, and WEALTH") == [
        "ESCS", "HOMEPOS", "WEALTH",
    ]
    assert _split_confounder_label("Math self-efficacy") == ["Math self-efficacy"]
    assert _split_confounder_label("Economic and education indicators") == [
        "Economic and education indicators",
    ]


def test_expand_confounder_dict_splits_grouped_entry():
    rows = _expand_confounder_dict({
        "variable_code": "N/A",
        "variable_name": "Gender and Age combined",
        "category": "other",
    })
    assert len(rows) == 2
    assert {r["variable_code"] for r in rows} == {"gender", "age"}
    assert {r["variable_name"] for r in rows} == {"Gender", "Age"}
    assert all(r["category"] in CONFOUNDER_CATEGORIES for r in rows)


def test_expand_confounder_dict_splits_comma_separated_ilsa_codes():
    rows = _expand_confounder_dict({
        "variable_code": None,
        "variable_name": "ESCS, HOMEPOS, and WEALTH",
        "category": None,
    })
    assert len(rows) == 3
    assert {r["variable_code"] for r in rows} == {"ESCS", "HOMEPOS", "WEALTH"}
    assert all(r["category"] == "socioeconomic" for r in rows)


def test_coerce_invalid_category_never_returns_other():
    cat = _coerce_confounder_category(
        "not_a_real_category",
        "Something vague",
        "xyz",
    )
    assert cat in CONFOUNDER_CATEGORIES
    assert cat != "other"


# ── Adversarial lazy LLM payloads → _sanitize repair ───────────────────────

LAZY_CONFOUNDER_PAYLOAD = {
    "metadata": {
        "file_name": "red_team_lazy_confounders.pdf",
        "title": "Red Team Lazy Confounders",
    },
    "data": {
        "survey_design": {
            "student_weights_used": False,
            "weight_fields_interpretation": "Test weight summary for red team.",
        },
        "plausible_values_handling": "rubin_rules",
        "missing_data_handling": "listwise_deletion",
        "sample_details": {
            "total_students": 1000,
            "countries": [],
            "sample_filtering_criteria": (
                "Excluded students with any missing values (listwise deletion)."
            ),
        },
        "ml_techniques": {"primary": "XGBoost", "all_techniques": ["XGBoost"]},
        "confounders_identified": [
            {
                "variable_code": "N/A",
                "variable_name": "Gender and Age combined",
                "category": "other",
            },
            {
                "variable_code": "N/A",
                "variable_name": "TF-IDF reset behavior",
                "category": "other",
            },
            "ESCS, HOMEPOS, and WEALTH",  # legacy string — must split into 3
            {
                "variable_code": "FAKE_ILSA_CODE_XYZ",
                "variable_name": "Math self-efficacy",
                "category": "student_attitude",
            },
        ],
        "outcome_summary": (
            "The study applied ML to PISA 2022 mathematics data and reported moderate "
            "predictive performance with gender and ESCS as key predictors."
        ),
        "main_findings": [
            {
                "dataset_used": "PISA 2022 Mathematics",
                "target_variable": "Math achievement",
                "top_predictors": ["Gender", "Socioeconomic status (ESCS)"],
                "performance_metrics": "R²: 0.45",
                "standardized_conclusion": (
                    "Using PISA 2022 Mathematics data, the study leveraged gender and "
                    "ESCS to predict math achievement, finding positive associations."
                ),
            },
        ],
        "research_design_type": "predictive",
    },
}


def test_sanitize_repairs_lazy_confounders_without_pdf():
    sanitized = GPTExtractor._sanitize(LAZY_CONFOUNDER_PAYLOAD)
    conf = sanitized["data"]["confounders_identified"]

    assert all(c["variable_code"].upper() not in ("N/A", "NA", "") for c in conf)
    assert all(c["category"] in CONFOUNDER_CATEGORIES for c in conf)
    assert all("other" != c["category"] for c in conf)

    # Grouped entry must split into Gender + Age
    assert any("gender" in c["variable_name"].lower() for c in conf)
    assert any(c["variable_name"].lower() == "age" or c["variable_code"].lower() == "age" for c in conf)

    # Micro ML/log features must be dropped (not tabulation-ready confounders)
    codes_lower = {c["variable_code"].lower() for c in conf}
    names_lower = " ".join(c["variable_name"].lower() for c in conf)
    assert "tfidf" not in names_lower and "reset" not in codes_lower

    # Legacy comma list must become three ILSA-coded rows
    codes = {c["variable_code"].upper() for c in conf}
    assert {"ESCS", "HOMEPOS", "WEALTH"}.issubset(codes)
    assert len(conf) >= 6


def test_sanitize_does_not_depend_on_pdf_filename():
    payload = json.loads(json.dumps(LAZY_CONFOUNDER_PAYLOAD))
    payload["metadata"]["file_name"] = "totally_different_report_from_iea.pdf"
    conf = GPTExtractor._sanitize(payload)["data"]["confounders_identified"]
    assert len(conf) >= 3
    assert all(c["category"] in CONFOUNDER_CATEGORIES for c in conf)


# ── Prompt/schema alignment (static string checks) ─────────────────────────

def test_is_micro_process_confounder_detects_zhou_style_noise():
    assert _is_micro_process_confounder("Start action frequency", "start")
    assert _is_micro_process_confounder("All sliders at zero (0_0_0)", "0_0_0")
    assert _is_micro_process_confounder("0_0_0 behavior usage", "zero_state_behavior")
    assert _is_micro_process_confounder("TF-IDF start behavior weight", "tfidf_start")
    assert _is_micro_process_confounder(
        "Word2vec action embedding vector", "word2vec_behavior_vector"
    )
    assert not _is_micro_process_confounder("VOTAT strategy behavior set", "VOTAT")
    assert not _is_micro_process_confounder("Sequence length", "sequence_length")


def test_is_survey_weight_confounder_detects_wgt_codes():
    assert _is_survey_weight_confounder("Science teacher weight", "SCIWGT")
    assert _is_survey_weight_confounder("Student final weight", "W_FSTUWT")
    assert _is_survey_weight_confounder("Total weight", "TOTWGT")
    assert not _is_survey_weight_confounder("Socioeconomic status (ESCS)", "ESCS")
    assert not _is_survey_weight_confounder("Gender", "ST004D01")


def test_normalize_confounders_list_strips_survey_weights():
    raw = [
        {"variable_code": "SCIWGT", "variable_name": "Science teacher weight", "category": "system_level"},
        {"variable_code": "ESCS", "variable_name": "Socioeconomic status", "category": "socioeconomic"},
    ]
    out = _normalize_confounders_list(raw)
    codes = {c["variable_code"] for c in out}
    assert codes == {"ESCS"}


def test_normalize_confounders_list_strips_zhou_micro_features():
    raw = [
        {"variable_code": "start", "variable_name": "Start action frequency", "category": "process_data"},
        {"variable_code": "0_0_0", "variable_name": "All sliders at zero (0_0_0)", "category": "process_data"},
        {"variable_code": "zero_state_behavior", "variable_name": "0_0_0 behavior usage", "category": "process_data"},
        {"variable_code": "tfidf_start", "variable_name": "TF-IDF start behavior weight", "category": "process_data"},
        {"variable_code": "VOTAT", "variable_name": "VOTAT strategy behavior set", "category": "process_data"},
        {"variable_code": "sequence_length", "variable_name": "Sequence length", "category": "process_data"},
    ]
    out = _normalize_confounders_list(raw)
    codes = {c["variable_code"] for c in out}
    assert codes == {"VOTAT", "sequence_length"}


def test_sanitize_backfills_title_from_file_name():
    payload = {
        "metadata": {
            "file_name": (
                "10. Aydogan & Tat. (2025). Investigating the Performance of "
                "Artificial Neural Networks in Predicting Affective Responses .pdf"
            ),
            "title": None,
            "authors": ["Aydogan, I."],
            "year": 2025,
            "publication_type": "journal",
            "source_category": "peer_reviewed_research",
        },
        "data": {
            "survey_design": {
                "student_weights_used": False,
                "weight_fields_interpretation": "No weights reported.",
            },
            "plausible_values_handling": "not_applicable",
            "missing_data_handling": "listwise_deletion",
            "handling_not_reported_explanation": "Affective DV; PVs not applicable.",
            "sample_details": {
                "total_students": 100,
                "countries": [],
                "sample_filtering_criteria": "Lebanon PISA 2018 subsample.",
            },
            "ml_techniques": {"primary": "Neural Network", "all_techniques": ["Neural Network"]},
            "confounders_identified": [],
            "main_findings": [],
            "research_design_type": "predictive",
        },
    }
    meta = GPTExtractor._sanitize(payload)["metadata"]
    assert meta["title"] == (
        "Investigating the Performance of Artificial Neural Networks in "
        "Predicting Affective Responses"
    )


def test_sanitize_fills_missing_sample_filtering_criteria():
    payload = {
        "metadata": {"file_name": "filter_test.pdf", "title": "Filter Test"},
        "data": {
            "survey_design": {
                "student_weights_used": False,
                "weight_fields_interpretation": "Weights not used.",
            },
            "plausible_values_handling": "rubin_rules",
            "missing_data_handling": "listwise_deletion",
            "sample_details": {"total_students": 500, "countries": []},
            "ml_techniques": {"primary": "RF", "all_techniques": ["RF"]},
            "confounders_identified": [],
            "main_findings": [],
            "research_design_type": "predictive",
        },
    }
    out = GPTExtractor._sanitize(payload)
    sfc = out["data"]["sample_details"]["sample_filtering_criteria"]
    assert sfc and "full available sample" in sfc.lower()


def test_normalize_findings_keeps_outcome_summary_and_migrates_main_findings():
    data = {
        "outcome_summary": "XGBoost achieved 98% accuracy predicting resilience.",
        "confounders_identified": [],
    }
    _normalize_findings_fields(data)
    assert "outcome_summary" in data
    assert "98%" in data["outcome_summary"]
    assert len(data["main_findings"]) == 1
    assert "98%" in data["main_findings"][0]["standardized_conclusion"]


def test_normalize_findings_preserves_both_when_present():
    data = {
        "outcome_summary": "Narrative summary with R²=0.51.",
        "main_findings": [{
            "dataset_used": "TIMSS 2019 Grade 8 Science",
            "target_variable": "Science achievement",
            "top_predictors": ["SES"],
            "performance_metrics": "R²: 0.51",
            "standardized_conclusion": (
                "Using TIMSS 2019 Grade 8 Science data, the study leveraged SES to "
                "predict science achievement, finding a moderate effect."
            ),
        }],
    }
    _normalize_findings_fields(data)
    assert "Narrative summary" in data["outcome_summary"]
    assert data["main_findings"][0]["dataset_used"].startswith("TIMSS 2019")


def test_normalize_main_findings_list_requires_target():
    out = _normalize_main_findings_list([
        {"target_variable": "", "top_predictors": ["ESCS"], "performance_metrics": "R²: 0.5",
         "standardized_conclusion": "Test."},
        {"target_variable": "Science score", "top_predictors": ["ESCS", "Gender"],
         "performance_metrics": "", "standardized_conclusion": "Science findings."},
    ])
    assert len(out) == 1
    assert out[0]["target_variable"] == "Science score"
    assert out[0]["performance_metrics"] == "Not reported"
    assert out[0]["dataset_used"] == "ILSA dataset not specified in manuscript"
    assert "TIMSS" not in out[0]["standardized_conclusion"] or "Using" in out[0]["standardized_conclusion"]


def test_dedupe_main_findings_merges_clustering_auxiliary_row():
    raw = [
        {
            "dataset_used": "PISA 2012 Problem Solving process data (CP02501)",
            "target_variable": "Correct answer (binary 0/1)",
            "top_predictors": ["VOTAT navigation behavior", "Sequence length"],
            "performance_metrics": "RF F1=0.84",
            "standardized_conclusion": "Primary supervised result.",
        },
        {
            "dataset_used": "PISA 2012 Problem Solving process data (CP02501)",
            "target_variable": "Behavior-strategy clusters (k-means groups)",
            "top_predictors": ["VOTAT navigation behavior", "Sequence length"],
            "performance_metrics": "k-means K=3 cluster correct rates 81/42/36%",
            "standardized_conclusion": "Clustering auxiliary result.",
        },
    ]
    out = _dedupe_main_findings(_normalize_main_findings_list(raw))
    assert len(out) == 1
    assert "k-means" in out[0]["performance_metrics"].lower()
    assert out[0]["target_variable"].startswith("Correct answer")


def test_build_standardized_conclusion_avoids_double_data():
    from src.extractors.gpt_extractor import _build_standardized_conclusion

    text = _build_standardized_conclusion(
        "PISA 2012 Problem Solving process data (CP02501)",
        ["VOTAT"],
        "Correct answer (binary 0/1)",
        effect_hint="Random Forest performed best.",
    )
    assert "process data (CP02501) data," not in text.lower()
    assert text.startswith("Using PISA 2012")


def test_normalize_main_findings_injects_dataset_into_conclusion():
    out = _normalize_main_findings_list([{
        "dataset_used": "TIMSS 2019 Grade 8 Science",
        "target_variable": "Science achievement (PVs)",
        "top_predictors": ["Socioeconomic background (SES)"],
        "performance_metrics": "R²: 0.51",
        "standardized_conclusion": "SES was the strongest predictor.",
    }])
    assert len(out) == 1
    assert "TIMSS 2019 Grade 8 Science" in out[0]["standardized_conclusion"]
    assert out[0]["standardized_conclusion"].startswith("Using TIMSS 2019")


def test_data_block_accepts_structured_main_findings():
    finding = StructuredFinding(
        dataset_used="PISA 2022 Creative Thinking",
        target_variable="Resilience (top 2%)",
        top_predictors=["Books at home", "ESCS"],
        performance_metrics="Accuracy: 98%",
        standardized_conclusion=(
            "Using PISA 2022 Creative Thinking data, the study leveraged books at home "
            "and ESCS to predict resilience, finding strong predictive performance."
        ),
    )
    block = DataBlock(
        survey_design={"student_weights_used": False, "weight_fields_interpretation": "x"},
        plausible_values_handling="average_pv",
        missing_data_handling="listwise_deletion",
        sample_details={
            "countries": [],
            "sample_filtering_criteria": "Full sample for specified countries.",
        },
        ml_techniques={"primary": "XGBoost", "all_techniques": ["XGBoost"]},
        confounders_identified=[],
        main_findings=[finding],
        outcome_summary=(
            "Using PISA 2022 Creative Thinking data, the model achieved 98% accuracy "
            "predicting resilience with books at home and ESCS as top predictors."
        ),
    )
    assert block.main_findings[0].target_variable.startswith("Resilience")


def test_system_prompt_forbids_other_and_lazy_na():
    from src.extractors.gpt_extractor import SYSTEM_PROMPT

    assert 'no "other"' in SYSTEM_PROMPT or "NO 'other'" in SYSTEM_PROMPT
    assert "13 categor" in SYSTEM_PROMPT or "13 literals" in SYSTEM_PROMPT
    assert "N/A" in SYSTEM_PROMPT  # mentioned as forbidden/lazy, not encouraged
    assert "Tier 1" in SYSTEM_PROMPT or "Tier 2" in SYSTEM_PROMPT
    assert "CONCEPTUAL ONLY" in SYSTEM_PROMPT or "NOT ML FEATURE" in SYSTEM_PROMPT
    assert "tfidf_start" not in SYSTEM_PROMPT
    assert "main_findings" in SYSTEM_PROMPT or "MAIN FINDINGS" in SYSTEM_PROMPT


def test_post_process_model_normalizes_confounders_in_place():
    """Structured-output path: repair lazy N/A via _post_process_model."""
    extraction = ILSAArticleMetadata.model_validate(
        GPTExtractor._sanitize(LAZY_CONFOUNDER_PAYLOAD)
    )
    # Re-inject one lazy row after sanitize (simulates model returning N/A before post-process)
    extraction.data.confounders_identified.append(
        Confounder.model_construct(
            variable_code="N/A",
            variable_name="Word2vec cosine similarity",
            category="process_data",
        )
    )

    GPTExtractor._post_process_model(extraction)
    conf = extraction.data.confounders_identified
    assert len(conf) >= 1
    for c in conf:
        assert c.variable_code.upper() not in ("N/A", "NA", "")
        assert c.category in CONFOUNDER_CATEGORIES


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
