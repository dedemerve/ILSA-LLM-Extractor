import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "finetuning"))

from chunking import chunk_text, clean_source, render_chunk_prompt  # noqa: E402
from extract_v2 import check_test_lock, merge_study  # noqa: E402

SFT_TRAIN = ROOT / "outputs/finetuning/v2/data/sft_train.jsonl"


def test_clean_source_joins_wrapped_lines_and_drops_references():
    body = "Results show that ESCS is associ-\nated with achievement.\nNext line.\n\n" * 30
    text = clean_source(body + "\nReferences\nSmith, J. (2020). A paper.\n")
    assert "associated with achievement. Next line." in text
    assert "Smith, J." not in text


def test_chunks_cover_text_with_overlap():
    text = ("Sentence number one is here. " * 40 + "\n\n") * 10
    chunks = chunk_text(text, size=1000, overlap=100)
    assert chunks[0][0] == 0
    # a short tail (< MIN_CHUNK_CHARS) is merged into the previous chunk
    assert all(len(c) <= 1000 + 400 for _, c in chunks)
    assert chunks[-1][0] + len(chunks[-1][1]) >= len(text.strip()) - 1


def test_prompt_rendering_is_deterministic():
    a = render_chunk_prompt("T", "PISA", "PISA 2018", 1, 3, "passage")
    assert a == render_chunk_prompt("T", "PISA", "PISA 2018", 1, 3, "passage")
    assert a.endswith("PASSAGE 1 of 3:\npassage\n")


def _item(pred, evidence, direction="positive", effect_type="E_assoc"):
    return {"predictor": pred, "ilsa_variable": None, "canonical_construct": "SES_COMPOSITE",
            "outcome": "math", "effect_type": effect_type, "direction": direction,
            "effect_size": None, "evidence_text": evidence}


def test_merge_dedupes_drops_ungrounded_and_attaches_methods():
    passage = "We found that ESCS was positively associated with mathematics achievement in all countries."
    grounded = "ESCS was positively associated with mathematics achievement in all countries."
    results = [
        {"passage": passage, "items": [_item("ESCS", grounded), _item("ESCS", grounded),
                                       _item("Gender", "Girls outperformed boys in every reading domain.")]},
    ]
    pairs, stats = merge_study(results, {"plausible_values_handling": "rubin_rules", "student_weights_used": True})
    assert stats == {"raw_items": 3, "ungrounded_dropped": 1, "duplicates_dropped": 1}
    assert len(pairs) == 1
    assert pairs[0]["plausible_values_handling"] == "rubin_rules" and pairs[0]["student_weights_used"] is True


def test_merge_keeps_pairs_that_differ_in_effect_type():
    passage = "ESCS was positively associated with mathematics achievement in all countries."
    results = [{"passage": passage, "items": [_item("ESCS", passage), _item("ESCS", passage, effect_type="E_predictive")]}]
    pairs, _ = merge_study(results, {})
    assert len(pairs) == 2


def test_test_split_is_locked_without_flag():
    with pytest.raises(SystemExit):
        check_test_lock("test", "any_arm", final=False)
    check_test_lock("val", "any_arm", final=False)


@pytest.mark.skipif(not SFT_TRAIN.exists(), reason="run build_sft_v2.py first")
def test_sft_targets_are_valid_and_grounded():
    for line in SFT_TRAIN.read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        assert [m["role"] for m in rec["prompt"]] == ["user"]
        target = json.loads(rec["completion"][0]["content"])
        for item in target.get("items", []):
            assert item["evidence_text"][:60] in rec["prompt"][0]["content"]


def _tokenizer(name):
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained(name)
    except Exception:  # noqa: BLE001  offline or gated
        pytest.skip(f"tokenizer {name} unavailable")


@pytest.mark.skipif(not SFT_TRAIN.exists(), reason="run build_sft_v2.py first")
@pytest.mark.parametrize("name", ["Qwen/Qwen3-8B", "mistralai/Mistral-7B-Instruct-v0.3"])
def test_training_masks_supervise_eos_and_never_pad_with_eos(name):
    from train_lora_v2 import configure_tokenizer, gate, tokenize_example

    tok = configure_tokenizer(_tokenizer(name))
    recs = [json.loads(l) for l in SFT_TRAIN.read_text(encoding="utf-8").splitlines()[:5]]
    examples = [tokenize_example(tok, r, 8192) for r in recs]
    report = gate(tok, examples)
    assert report["supervised_tail_example"].endswith(tok.eos_token)
    assert tok.pad_token_id != tok.eos_token_id
