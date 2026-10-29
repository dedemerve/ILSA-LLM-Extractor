import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "finetuning"))

from eval_extraction import (  # noqa: E402
    hungarian_match, legacy_exact_key_match, legacy_greedy_construct_match,
    grounding_score, _shingles, score_study, sign_flip_test, prf,
)
from ilsa_schema import (  # noqa: E402
    EvidencePair, json_schema, norm_direction, norm_effect_type, norm_pv,
    norm_weights, normalize_gold_row, normalize_pred, outcome_class, parse_output,
    serialize_target,
)


def _gold(construct, direction="positive", outcome="Mathematics achievement", predictor="ESCS"):
    return normalize_gold_row({
        "record_id": "G1", "study_id": "s.pdf", "predictor": predictor,
        "canonical_construct": construct, "target_variable": outcome,
        "effect_type": "E_assoc", "direction_canonical": direction,
        "plausible_values_handling": "single_pv (first PV)", "student_weights_used": "TRUE",
    })


def _pred(construct, direction="positive", outcome="math score", predictor="ESCS"):
    return normalize_pred({
        "predictor": predictor, "canonical_construct": construct, "outcome": outcome,
        "effect_type": "E_assoc", "direction": direction,
        "plausible_values_handling": "single_pv", "student_weights_used": True,
        "evidence_text": "ESCS was positively associated with mathematics achievement.",
    })


def test_gold_label_normalization():
    assert norm_direction("1") == "positive"
    assert norm_direction("-") == "negative"
    assert norm_direction("0") == "not_reported"
    assert norm_direction("") == "not_reported"
    assert norm_effect_type("predictive") == "E_predictive"
    assert norm_effect_type("unspecified") == "E_assoc"
    assert norm_pv("single_pv (first PV for numeracy)") == "single_pv"
    assert norm_pv("rubin_rules (5 PVs via Rubin)") == "rubin_rules"
    assert norm_weights("TRUE") is True and norm_weights("not_reported") is None
    assert _gold("UNMATCHED_TEACHER_ICT")["canonical_construct"] == "UNMATCHED"
    assert outcome_class("PV1MATH") == "mathematics"
    assert outcome_class("Natural log of hourly wages") == "other"


def test_hungarian_is_order_invariant_and_strict_checks_direction():
    gold = [_gold("SES_COMPOSITE"), _gold("GENDER", direction="mixed", predictor="Gender")]
    pred = [_pred("GENDER", direction="positive", predictor="Gender"), _pred("SES_COMPOSITE")]
    assert len(hungarian_match(gold, pred, strict=False)) == 2
    assert len(hungarian_match(gold, list(reversed(pred)), strict=False)) == 2
    strict = hungarian_match(gold, pred, strict=True)
    assert strict == [(0, 1)]  # GENDER pair has the wrong direction


def test_unmatched_requires_predictor_similarity():
    gold = [_gold("UNMATCHED", predictor="Teacher collaboration frequency")]
    assert hungarian_match(gold, [_pred("UNMATCHED", predictor="Country GDP")], strict=False) == []
    assert len(hungarian_match(gold, [_pred("UNMATCHED", predictor="teacher collaboration")], strict=False)) == 1


def test_legacy_matchers_reproduce_v1_behaviour():
    gold = [_gold("SES_COMPOSITE", direction="mixed")]
    pred = [_pred("SES_COMPOSITE", direction="positive")]
    assert legacy_greedy_construct_match(gold, pred) == 1   # wrong direction still a TP
    assert legacy_exact_key_match(gold, pred) == 0


def test_joint_slot_counts_only_matched_pairs():
    gold = [_gold("SES_COMPOSITE")]
    res = score_study(gold, [_pred("GENDER", predictor="Gender")], source=None)
    assert res["lenient_tp"] == 0 and res["slot_direction_tp"] == 0


def test_grounding():
    src = _shingles("In the results, ESCS was positively associated with mathematics achievement across countries.")
    assert grounding_score("ESCS was positively associated with mathematics achievement", src) == 1.0
    assert grounding_score("Teachers reported high job satisfaction in all participating schools", src) == 0.0


def test_sign_flip_and_prf():
    assert sign_flip_test([0.1] * 6) == pytest.approx(2 / 64)
    assert prf(0, 0, 5)["f1"] == 0.0


def test_parse_output_wrapper_bare_and_truncated():
    pairs, status = parse_output('{"items": [{"predictor": "a"}]}')
    assert status == "ok" and len(pairs) == 1
    pairs, status = parse_output('[{"predictor": "a"}, {"predictor": "b"},]')
    assert status == "repaired" and len(pairs) == 2
    pairs, status = parse_output('[{"predictor": "a"}, {"predictor": "b"}, {"predic')
    assert status == "truncated" and len(pairs) == 2


def test_schema_and_serialization_round_trip():
    pair = {
        "predictor": "ESCS", "ilsa_variable": None, "canonical_construct": "SES_COMPOSITE",
        "outcome": "Math", "effect_type": "E_assoc", "direction": "positive", "effect_size": 0.3,
        "plausible_values_handling": "all_pvs", "student_weights_used": True,
        "evidence_text": "ESCS predicts math.",
    }
    EvidencePair(**pair)
    text = serialize_target([pair])
    assert text.startswith('{"items":[{"predictor":"ESCS"')
    assert parse_output(text)[1] == "ok"
    schema = json_schema(max_items=5)
    assert schema["properties"]["items"]["maxItems"] == 5
    assert "uniqueItems" not in schema["properties"]["items"]
