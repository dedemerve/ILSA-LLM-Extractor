"""
Build the v2 SFT dataset: chunk-level extraction examples plus one methods example per study
(action plan step 5).

Design
  - Labels come from gold v1.4.1 (adjudicated), normalized with ilsa_schema. The Stage 7
    JSONL is used only to recover evidence_text / effect_size / ilsa_variable per pair.
  - Each gold pair is assigned to the passage that contains its evidence. evidence_text is
    re-cut to ONE verbatim sentence of that passage (<= 600 chars), so every target is
    grounded by construction. Pairs that cannot be located are reported, not invented.
  - Passages without any gold pair become {"items": []} negatives, but only "clean" ones
    (no gold predictor phrase in them), capped at --neg-ratio x positives per study.
  - Study-level fields (PV handling, student weights) are learned by a separate methods
    example per study and attached at merge time (they are constant within every study).
  - Targets are serialized deterministically; instructions travel in the user turn.
  - Only train and val studies are read. The test split is never touched.

Output (outputs/finetuning/v2/data/)
  sft_{split}.jsonl      TRL prompt-completion records
  manifest_{split}.json  per-study counts, unlocated pairs, evidence provenance
  evidence_review_{split}.csv  one row per gold pair for human verification; corrections
                         copied into evidence_overrides.csv are picked up on the next build
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from chunking import (  # noqa: E402
    chunk_text, clean_source, methods_passages, render_chunk_prompt, render_methods_prompt,
    user_messages,
)
from eval_extraction import _shingles, find_source_text, gold_by_study, hungarian_match, load_gold  # noqa: E402
from ilsa_schema import (  # noqa: E402
    CHUNK_FIELDS, MAX_LEN_EFFECT_SIZE, MAX_LEN_EVIDENCE, MAX_LEN_ILSA_VARIABLE, normalize_pred,
    serialize_target,
)

ROOT = Path(__file__).resolve().parents[2]
STAGE7_JSONL = {
    "train": ROOT / "outputs/finetuning/data/train_finetune.jsonl",
    "val": ROOT / "outputs/finetuning/data/val_finetune.jsonl",
}
OUT_DIR = ROOT / "outputs/finetuning/v2/data"
# Human-verified evidence (optional). Columns: record_id, corrected_evidence_text.
# A correction is used only if it occurs verbatim in the study's cleaned source text.
OVERRIDES_CSV = OUT_DIR / "evidence_overrides.csv"
LOCATE_MIN_COVERAGE = 0.6
SEED = 2026

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[])|\n+")
_RESULT_CUE = re.compile(r"signific|associat|predict|coefficient|\bbeta\b|β|odds ratio|correlat|"
                         r"effect|higher|lower|positive|negative|importan|\bp\s*[<=]|\d\.\d", re.I)
_STOP = {"with", "from", "that", "this", "were", "have", "their", "between", "level", "score",
         "scores", "index", "students", "student", "school", "schools", "variable", "effect"}


_BOILERPLATE = re.compile(r"©|\bdoi\b|issn|contents lists|journal homepage|creativecommons|"
                          r"\bconference\b|all rights reserved|corresponding author|e-mail|https?://|"
                          r"\(\d{4}\)\s*\d{5,}", re.I)


def sentences(text: str) -> list[str]:
    """Verbatim sentence substrings of `text`. A piece cut at a paragraph break inside a
    sentence (two-column PDF layouts) is re-joined by widening the span, never by rewriting it."""
    spans: list[list[int]] = []
    start = 0
    for m in list(_SENT_SPLIT.finditer(text)) + [None]:
        end = m.start() if m else len(text)
        piece = text[start:end].strip()
        if piece:
            if spans and not re.search(r"[.!?:;)\]]$", text[spans[-1][0]:spans[-1][1]].rstrip()) \
                    and piece[:1].islower():
                spans[-1][1] = end
            else:
                spans.append([start, end])
        if m:
            start = m.end()
    out = []
    for a, b in spans:
        s = text[a:b].strip()
        letters = sum(ch.isalpha() for ch in s)
        if len(s) >= 40 and letters >= 0.6 * len(s) and not _BOILERPLATE.search(s):
            out.append(s)
    return out


def anchored(sentence: str, predictor: str) -> bool:
    """The sentence must name the predictor: all content words of a one- or two-word
    predictor, at least half of them otherwise."""
    words = content_words(predictor)
    need = len(words) if len(words) <= 2 else (len(words) + 1) // 2
    return bool(words) and len(content_words(sentence) & words) >= need


def content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]{4,}", (text or "").lower()) if w not in _STOP}


def stage7_pairs(split: str) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    path = STAGE7_JSONL[split]
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        msgs = json.loads(line)["messages"]
        sid = msgs[1]["content"].split("SOURCE: ", 1)[1].split("\n", 1)[0].strip()
        out[sid] = json.loads(msgs[-1]["content"])
    return out


def best_sentence(chunk: str, target: set[tuple[str, ...]] | None, words: set[str]) -> tuple[str, float]:
    """Pick the chunk sentence that best overlaps the evidence shingles (or, failing that, the predictor words)."""
    best, best_score = "", 0.0
    for s in sentences(chunk):
        if target:
            sh = _shingles(s)
            score = len(sh & target) / max(1, min(len(sh), len(target)))
        else:
            sw = content_words(s)
            score = len(sw & words) / max(1, len(words))
            if score > 0 and _RESULT_CUE.search(s):
                score += 0.25  # prefer reported findings over literature-review mentions
        if score > best_score:
            best, best_score = s, score
    return best[:MAX_LEN_EVIDENCE], best_score


def load_overrides() -> dict[str, str]:
    if not OVERRIDES_CSV.exists():
        return {}
    with OVERRIDES_CSV.open(encoding="utf-8") as f:
        return {r["record_id"]: r["corrected_evidence_text"].strip() for r in csv.DictReader(f)
                if r.get("corrected_evidence_text", "").strip()}


def locate(pair_gold: dict, pair_s7: dict | None, chunks: list[tuple[int, str]],
           override: str | None = None) -> dict[str, Any] | None:
    if override:
        norm = re.sub(r"\s+", " ", override)
        for k, (start, c) in enumerate(chunks):
            pos = re.sub(r"\s+", " ", c).find(norm)
            if pos != -1:
                return {"chunk": k, "evidence_text": norm[:MAX_LEN_EVIDENCE],
                        "evidence_source": "human_override", "offset": start + pos}
    evidence = (pair_s7 or {}).get("evidence_text") or ""
    ev_sh = _shingles(evidence) if evidence else set()
    if ev_sh:
        cover = [len(ev_sh & _shingles(c)) / len(ev_sh) for _, c in chunks]
        k = max(range(len(chunks)), key=cover.__getitem__)
        if cover[k] >= LOCATE_MIN_COVERAGE:
            sent, _ = best_sentence(chunks[k][1], ev_sh, set())
            # Some Stage 7 evidence strings start at the article header; accept the sentence
            # only if it names the predictor.
            if sent and anchored(sent, pair_gold["predictor"]):
                return {"chunk": k, "evidence_text": sent, "evidence_source": "stage7_evidence",
                        "offset": chunks[k][0] + chunks[k][1].find(sent[:40])}
    words = content_words(pair_gold["predictor"])
    if not words:
        return None
    scored = []
    for k, (_, c) in enumerate(chunks):
        sent, score = best_sentence(c, None, words)
        scored.append((score, -k, k, sent))
    score, _, k, sent = max(scored)
    if score >= 0.75 and sent and anchored(sent, pair_gold["predictor"]):
        return {"chunk": k, "evidence_text": sent, "evidence_source": "predictor_search",
                "offset": chunks[k][0] + chunks[k][1].find(sent[:40])}
    return None


def _short(value: Any, limit: int) -> str | None:
    if value is None or str(value).strip() in ("", "null", "None"):
        return None
    return str(value).strip()[:limit]


def build_study(sid: str, gold: list[dict], s7: list[dict], meta: dict[str, str], neg_ratio: float,
                rng: random.Random, overrides: dict[str, str]) -> tuple[list[dict], dict[str, Any]]:
    source = find_source_text(sid)
    report: dict[str, Any] = {"study_id": sid, "n_gold": len(gold)}
    if source is None:
        report["skipped"] = "no source text"
        return [], report
    chunks = chunk_text(clean_source(source))
    s7_norm = [normalize_pred(p) for p in s7]
    links = dict(hungarian_match(gold, s7_norm, strict=False))

    assigned: dict[int, list[dict[str, Any]]] = {}
    unlocated, provenance = [], {"human_override": 0, "stage7_evidence": 0, "predictor_search": 0}
    review = []
    for i, g in enumerate(gold):
        raw7 = s7[links[i]] if i in links else None
        loc = locate(g, raw7, chunks, overrides.get(g["record_id"]))
        review.append({"study_id": sid, "record_id": g["record_id"], "predictor": g["predictor"],
                       "canonical_construct": g["canonical_construct"], "outcome": g["outcome"],
                       "direction": g["direction"], "chunk": loc["chunk"] + 1 if loc else "",
                       "evidence_source": loc["evidence_source"] if loc else "UNLOCATED",
                       "evidence_text": loc["evidence_text"] if loc else "",
                       "reviewer_ok": "", "corrected_evidence_text": ""})
        if loc is None:
            unlocated.append({"record_id": g["record_id"], "predictor": g["predictor"],
                              "canonical_construct": g["canonical_construct"]})
            continue
        provenance[loc["evidence_source"]] += 1
        assigned.setdefault(loc["chunk"], []).append({
            "predictor": g["predictor"][:200],
            "ilsa_variable": _short((raw7 or {}).get("ilsa_variable"), MAX_LEN_ILSA_VARIABLE),
            "canonical_construct": g["canonical_construct"],
            "outcome": g["outcome"][:200],
            "effect_type": g["effect_type"],
            "direction": g["direction"],
            "effect_size": _short((raw7 or {}).get("effect_size"), MAX_LEN_EFFECT_SIZE),
            "evidence_text": loc["evidence_text"],
            "_offset": loc["offset"],
            "_record_id": g["record_id"],
        })

    gold_words = [content_words(g["predictor"]) for g in gold]
    clean_negatives = [k for k in range(len(chunks)) if k not in assigned and not any(
        w and len(w & content_words(chunks[k][1])) >= max(1, len(w) // 2 + 1) for w in gold_words)]
    n_neg = min(len(clean_negatives), int(round(neg_ratio * len(assigned))))
    negatives = sorted(rng.sample(clean_negatives, n_neg))

    records = []
    for k in sorted(list(assigned) + negatives):
        items = sorted(assigned.get(k, []), key=lambda p: p["_offset"])
        prompt = render_chunk_prompt(meta["title"], meta["program"], meta["cycle"], k + 1, len(chunks), chunks[k][1])
        records.append({
            "prompt": user_messages(prompt),
            "completion": [{"role": "assistant", "content": serialize_target(items, CHUNK_FIELDS)}],
            "_meta": {"study_id": sid, "kind": "chunk", "chunk": k, "n_items": len(items),
                      "record_ids": [p["_record_id"] for p in items]},
        })

    passage = methods_passages(chunks)
    ev, _ = best_sentence(passage, None, {"plausible", "weight", "weights", "weighted", "rubin"})
    methods_target = {
        "plausible_values_handling": gold[0]["plausible_values_handling"],
        "student_weights_used": gold[0]["student_weights_used"],
        "evidence_text": ev or None,
    }
    records.append({
        "prompt": user_messages(render_methods_prompt(meta["title"], meta["program"], meta["cycle"], passage)),
        "completion": [{"role": "assistant",
                        "content": json.dumps(methods_target, ensure_ascii=False, separators=(",", ":"))}],
        "_meta": {"study_id": sid, "kind": "methods"},
    })
    report.update({
        "n_chunks": len(chunks), "positive_chunks": len(assigned), "negative_chunks": len(negatives),
        "located": len(gold) - len(unlocated), "unlocated": unlocated, "evidence_provenance": provenance,
        "_review": review,
    })
    return records, report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits", nargs="+", default=["train", "val"], choices=["train", "val"])
    ap.add_argument("--neg-ratio", type=float, default=1.0)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    gold_df = load_gold("all")
    overrides = load_overrides()
    meta_by_study = {sid: {"title": str(g["title"].iloc[0]), "program": str(g["program"].iloc[0]),
                           "cycle": str(g["cycle"].iloc[0])} for sid, g in gold_df.groupby("study_id")}
    for split in args.splits:
        rng = random.Random(SEED)
        gold_map = gold_by_study(load_gold(split))
        s7 = stage7_pairs(split)
        records, reports = [], []
        for sid in sorted(gold_map):
            recs, rep = build_study(sid, gold_map[sid], s7.get(sid, []), meta_by_study[sid], args.neg_ratio, rng, overrides)
            records.extend(recs)
            reports.append(rep)
        with (args.out_dir / f"sft_{split}.jsonl").open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps({"prompt": r["prompt"], "completion": r["completion"]}, ensure_ascii=False) + "\n")
        review_rows = [row for r in reports for row in r.pop("_review", [])]
        if review_rows:
            with (args.out_dir / f"evidence_review_{split}.csv").open("w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(review_rows[0]))
                w.writeheader()
                w.writerows(review_rows)
        n_gold = sum(r["n_gold"] for r in reports)
        located = sum(r.get("located", 0) for r in reports)
        manifest = {
            "split": split, "gold": "gold_standard_full_corpus_v1.4.1.csv",
            "studies_in_split": len(gold_map),
            "studies_with_source": sum("skipped" not in r for r in reports),
            "gold_pairs": n_gold, "pairs_located": located,
            "records": len(records),
            "chunk_records": sum(r["_meta"]["kind"] == "chunk" for r in records),
            "empty_chunk_records": sum(r["_meta"]["kind"] == "chunk" and r["_meta"]["n_items"] == 0 for r in records),
            "methods_records": sum(r["_meta"]["kind"] == "methods" for r in records),
            "neg_ratio": args.neg_ratio, "seed": SEED,
            "record_meta": [r["_meta"] for r in records],
            "studies": reports,
        }
        (args.out_dir / f"manifest_{split}.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"{split}: studies {manifest['studies_with_source']}/{manifest['studies_in_split']} with source, "
              f"pairs located {located}/{n_gold}, records {len(records)} "
              f"(chunk {manifest['chunk_records']}, empty {manifest['empty_chunk_records']}, methods {manifest['methods_records']})")


if __name__ == "__main__":
    main()
