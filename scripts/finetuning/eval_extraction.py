"""
Extraction evaluator v2 (pre-registered in EVALUATION_PROTOCOL_v2.md).

Replaces the construct-only greedy matcher in scripts/baseline_inference.py and
the exact composite-key matcher in the Stage 9 notebook. Both are kept
reproducible here as `legacy_*` metrics so earlier numbers can be re-derived.

Metrics
  strict pair F1   construct + outcome class + direction must agree (Hungarian matching)
  lenient pair F1  construct only (Hungarian matching, outcome/predictor as tie-breaker)
  joint slot F1    per field: a slot counts only if its pair matched (lenient) AND the slot is correct
  grounding rate   share of predicted evidence_text whose word 5-gram shingles are >= 90% present in the source
  validity         parse status, duplicate rate, empty-output rate
  per-stratum and per-study (macro) scores
  uncertainty      bootstrap over studies; paired comparison: bootstrap of the difference + exact sign-flip test

UNMATCHED is a residual class (about 45% of gold rows), so construct equality alone
is not evidence of a match there: an UNMATCHED pair additionally needs predictor
similarity >= UNMATCHED_PREDICTOR_SIM.

Usage
  python3 scripts/finetuning/eval_extraction.py --pred PRED.json --split val --label lora_v2
  python3 scripts/finetuning/eval_extraction.py --pred A.json --pred-b B.json --split val \
      --label base --label-b lora_v2 --out outputs/finetuning/v2/eval/base_vs_lora_val.json
"""

from __future__ import annotations

import argparse
import ast
import itertools
import json
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ilsa_schema import (  # noqa: E402
    STRATA, normalize_gold_row, normalize_pred, normalize_text, parse_output,
)

ROOT = Path(__file__).resolve().parents[2]
GOLD_CSV = ROOT / "outputs/finetuning/data/gold_standard_full_corpus_v1.4.1.csv"
SPLIT_CSV = ROOT / "outputs/finetuning/data/gold_standard_split_v4.csv"
SOURCE_DIR = ROOT / "outputs/finetuning/source_recovery/texts"

UNMATCHED_PREDICTOR_SIM = 0.6
GROUNDING_THRESHOLD = 0.9
SHINGLE = 5
SLOT_FIELDS = ("direction", "effect_type", "plausible_values_handling",
               "student_weights_used", "outcome_class")
N_BOOT = 10_000
SEED = 2026


# Loading

def load_gold(split: str | None) -> pd.DataFrame:
    gold = pd.read_csv(GOLD_CSV, dtype=str, keep_default_na=False)
    split_df = pd.read_csv(SPLIT_CSV, dtype=str, keep_default_na=False)[["record_id", "split"]]
    gold = gold.merge(split_df, on="record_id", how="left")
    if split and split != "all":
        gold = gold[gold["split"] == split]
    return gold


def gold_key(g: dict[str, Any]) -> tuple[str, ...]:
    return (normalize_text(g["predictor"]), g["canonical_construct"], normalize_text(g["outcome"]),
            g["direction"], g["effect_type"])


def gold_by_study(gold: pd.DataFrame, dedup: bool = True) -> dict[str, list[dict[str, Any]]]:
    """Gold pairs per study. With dedup, rows identical on (predictor, construct, outcome,
    direction, effect_type) collapse to one: 79 of 246 v1.4.1 rows differ from another row of
    the same study only in record_id, so no extraction could tell them apart."""
    out: dict[str, list[dict[str, Any]]] = {}
    seen: set[tuple[str, ...]] = set()
    for _, row in gold.iterrows():
        g = normalize_gold_row(row.to_dict())
        key = (g["study_id"],) + gold_key(g)
        if dedup and key in seen:
            continue
        seen.add(key)
        out.setdefault(g["study_id"], []).append(g)
    return out


def _as_pairs(entry: dict[str, Any]) -> tuple[list[dict[str, Any]], str]:
    raw = entry.get("raw_output")
    if isinstance(raw, str) and raw.strip():
        return parse_output(raw)
    for key in ("pairs", "predicted", "items"):
        val = entry.get(key)
        if isinstance(val, list):
            return [p for p in val if isinstance(p, dict)], "ok"
        if isinstance(val, str) and val.strip():
            try:
                parsed = ast.literal_eval(val)
                return [p for p in parsed if isinstance(p, dict)], "ok"
            except (ValueError, SyntaxError):
                return parse_output(val)
    return [], "empty"


def load_predictions(path: Path, truncated_policy: str = "salvage") -> dict[str, dict[str, Any]]:
    """Accepts baseline_inference.py output, the Stage 9 list format and the v2 format.

    truncated_policy: "salvage" counts every complete object recovered from a truncated
    output as a prediction (penalizes runaway generation; v2 default); "zero" treats a
    truncated output as no prediction (the v1 BASE protocol rule, kept for reproduction)."""
    data = json.loads(path.read_text(encoding="utf-8"))
    entries = data.get("predictions", data) if isinstance(data, dict) else data
    out: dict[str, dict[str, Any]] = {}
    for e in entries:
        sid = e.get("study_id") or e.get("source")
        pairs, status = _as_pairs(e)
        if e.get("finish_reason") == "length":
            status = "truncated"
        if status == "truncated" and truncated_policy == "zero":
            pairs = []
        out[sid] = {"pairs": [normalize_pred(p) for p in pairs], "status": status}
    return out


def find_source_text(study_id: str) -> str | None:
    stem = Path(study_id).stem
    candidates = sorted(SOURCE_DIR.glob("*.txt"))
    for p in candidates:
        if p.stem == stem:
            return p.read_text(encoding="utf-8", errors="replace")
    for p in candidates:
        if p.stem.startswith(stem[:20]) or stem[:15].lower() in p.stem.lower():
            return p.read_text(encoding="utf-8", errors="replace")
    return None


# Matching

def _sim(a: str, b: str) -> float:
    a, b = normalize_text(a), normalize_text(b)
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def _compatible(g: dict[str, Any], p: dict[str, Any], strict: bool) -> float | None:
    """Return a match score in (0, 3] if g and p may be matched, else None."""
    if g["canonical_construct"] != p["canonical_construct"]:
        return None
    pred_sim = _sim(g["predictor"], p["predictor"])
    if g["canonical_construct"] == "UNMATCHED" and pred_sim < UNMATCHED_PREDICTOR_SIM:
        return None
    if strict and (g["outcome_class"] != p["outcome_class"] or g["direction"] != p["direction"]):
        return None
    return 1.0 + pred_sim + _sim(g["outcome"], p["outcome"])


def hungarian_match(gold: list[dict], pred: list[dict], strict: bool) -> list[tuple[int, int]]:
    if not gold or not pred:
        return []
    score = np.zeros((len(gold), len(pred)))
    for i, g in enumerate(gold):
        for j, p in enumerate(pred):
            s = _compatible(g, p, strict)
            score[i, j] = s if s is not None else 0.0
    rows, cols = linear_sum_assignment(-score)
    return [(i, j) for i, j in zip(rows, cols) if score[i, j] > 0]


def legacy_greedy_construct_match(gold: list[dict], pred: list[dict]) -> int:
    """Reproduces match_pairs() in scripts/baseline_inference.py (first unused same-construct prediction)."""
    used: set[int] = set()
    tp = 0
    for g in gold:
        gc = g["canonical_construct_raw"].lower()
        for j, p in enumerate(pred):
            if j not in used and p["canonical_construct_raw"].lower() == gc:
                used.add(j)
                tp += 1
                break
    return tp


def legacy_exact_key_match(gold: list[dict], pred: list[dict]) -> int:
    """Reproduces the Stage 9 exact (construct, direction, effect_type) key matcher."""
    key = lambda d: (d["canonical_construct"], d["direction"], d["effect_type"])  # noqa: E731
    remaining = [key(p) for p in pred]
    tp = 0
    for g in gold:
        k = key(g)
        if k in remaining:
            remaining.remove(k)
            tp += 1
    return tp


# Grounding and validity

def _shingles(text: str) -> set[tuple[str, ...]]:
    toks = re.findall(r"\w+", text.lower())
    if len(toks) < SHINGLE:
        return {tuple(toks)} if toks else set()
    return {tuple(toks[i:i + SHINGLE]) for i in range(len(toks) - SHINGLE + 1)}


def grounding_score(evidence: str, source_shingles: set[tuple[str, ...]]) -> float:
    ev = _shingles(evidence)
    if not ev:
        return 0.0
    return len(ev & source_shingles) / len(ev)


def duplicate_count(pairs: list[dict]) -> int:
    keys = [(normalize_text(p["predictor"]), p["canonical_construct"], normalize_text(p["outcome"]))
            for p in pairs]
    return len(keys) - len(set(keys))


# Scoring

def score_study(gold: list[dict], pred: list[dict], source: str | None) -> dict[str, Any]:
    res: dict[str, Any] = {"n_gold": len(gold), "n_pred": len(pred)}
    for mode in ("strict", "lenient"):
        m = hungarian_match(gold, pred, strict=(mode == "strict"))
        res[f"{mode}_tp"] = len(m)
        if mode == "lenient":
            for f in SLOT_FIELDS:
                res[f"slot_{f}_tp"] = sum(1 for i, j in m if gold[i][f] == pred[j][f])
            res["lenient_pairs"] = m
    res["legacy_greedy_tp"] = legacy_greedy_construct_match(gold, pred)
    res["legacy_exact_tp"] = legacy_exact_key_match(gold, pred)
    res["duplicates"] = duplicate_count(pred)
    if source is not None:
        sh = _shingles(source)
        scores = [grounding_score(p["evidence_text"], sh) for p in pred if p["evidence_text"]]
        res["grounded"] = sum(s >= GROUNDING_THRESHOLD for s in scores)
        res["with_evidence"] = len(scores)
    return res


def prf(tp: float, n_pred: float, n_gold: float) -> dict[str, float]:
    p = tp / n_pred if n_pred else 0.0
    r = tp / n_gold if n_gold else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4)}


def aggregate(per_study: dict[str, dict[str, Any]], studies: list[str]) -> dict[str, Any]:
    tot = lambda k: sum(per_study[s].get(k, 0) for s in studies)  # noqa: E731
    n_pred, n_gold = tot("n_pred"), tot("n_gold")
    out: dict[str, Any] = {"n_studies": len(studies), "n_gold": n_gold, "n_pred": n_pred}
    for name, key in (("strict", "strict_tp"), ("lenient", "lenient_tp"),
                      ("legacy_greedy_construct", "legacy_greedy_tp"),
                      ("legacy_exact_key", "legacy_exact_tp")):
        out[name] = {"tp": tot(key), **prf(tot(key), n_pred, n_gold)}
    out["joint_slot"] = {f: prf(tot(f"slot_{f}_tp"), n_pred, n_gold) for f in SLOT_FIELDS}
    out["macro_strict_f1"] = round(float(np.mean(
        [prf(per_study[s]["strict_tp"], per_study[s]["n_pred"], per_study[s]["n_gold"])["f1"]
         for s in studies])), 4) if studies else 0.0
    with_ev = tot("with_evidence")
    out["grounding_rate"] = round(tot("grounded") / with_ev, 4) if with_ev else None
    out["duplicate_rate"] = round(tot("duplicates") / n_pred, 4) if n_pred else 0.0
    return out


def stratum_scores(gold_map: dict, pred_map: dict, per_study: dict) -> dict[str, dict[str, float]]:
    counts: dict[str, dict[str, int]] = {}
    for sid, res in per_study.items():
        gold, pred = gold_map[sid], pred_map.get(sid, {"pairs": []})["pairs"]
        matched_g = {i for i, _ in res["lenient_pairs"]}
        matched_p = {j for _, j in res["lenient_pairs"]}
        for i, g in enumerate(gold):
            c = counts.setdefault(STRATA.get(g["canonical_construct"], "UNMATCHED"), {"tp": 0, "fn": 0, "fp": 0})
            c["tp" if i in matched_g else "fn"] += 1
        for j, p in enumerate(pred):
            if j not in matched_p:
                counts.setdefault(STRATA.get(p["canonical_construct"], "UNMATCHED"), {"tp": 0, "fn": 0, "fp": 0})["fp"] += 1
    return {s: {**c, **prf(c["tp"], c["tp"] + c["fp"], c["tp"] + c["fn"])} for s, c in sorted(counts.items())}


def bootstrap_f1(per_study: dict, studies: list[str], key: str, rng: np.random.Generator) -> list[float]:
    arr = np.array([[per_study[s][key], per_study[s]["n_pred"], per_study[s]["n_gold"]] for s in studies], float)
    idx = rng.integers(0, len(studies), size=(N_BOOT, len(studies)))
    tp, npred, ngold = (arr[idx, k].sum(axis=1) for k in range(3))
    p = np.divide(tp, npred, out=np.zeros_like(tp), where=npred > 0)
    r = np.divide(tp, ngold, out=np.zeros_like(tp), where=ngold > 0)
    f = np.divide(2 * p * r, p + r, out=np.zeros_like(tp), where=(p + r) > 0)
    return [round(float(np.percentile(f, 2.5)), 4), round(float(np.percentile(f, 97.5)), 4)]


def sign_flip_test(diffs: list[float]) -> float:
    """Exact two-sided sign-flip (permutation) test on per-study differences."""
    d = np.asarray(diffs, float)
    obs = abs(d.mean())
    flips = np.array(list(itertools.product((1, -1), repeat=len(d))))
    null = np.abs((flips * d).mean(axis=1))
    return float((null >= obs - 1e-12).mean())


def evaluate(pred_path: Path, split: str, truncated_policy: str = "salvage",
             gold_dedup: bool = True) -> tuple[dict[str, Any], dict, dict, dict]:
    gold_map = gold_by_study(load_gold(split), dedup=gold_dedup)
    pred_map = load_predictions(pred_path, truncated_policy)
    studies = sorted(gold_map)
    per_study: dict[str, dict[str, Any]] = {}
    for sid in studies:
        pred = pred_map.get(sid, {"pairs": [], "status": "missing"})
        per_study[sid] = score_study(gold_map[sid], pred["pairs"], find_source_text(sid))
        per_study[sid]["status"] = pred["status"]
    rng = np.random.default_rng(SEED)
    summary = aggregate(per_study, studies)
    summary["strict"]["ci95_study_bootstrap"] = bootstrap_f1(per_study, studies, "strict_tp", rng)
    summary["lenient"]["ci95_study_bootstrap"] = bootstrap_f1(per_study, studies, "lenient_tp", rng)
    summary["per_stratum_lenient"] = stratum_scores(gold_map, pred_map, per_study)
    statuses = [per_study[s]["status"] for s in studies]
    summary["validity"] = {k: statuses.count(k) for k in sorted(set(statuses))}
    summary["extra_predicted_studies"] = sorted(set(pred_map) - set(studies))
    return summary, per_study, gold_map, pred_map


def per_study_table(per_study: dict) -> dict[str, dict[str, Any]]:
    return {s: {"n_gold": r["n_gold"], "n_pred": r["n_pred"], "status": r["status"],
                "strict_f1": prf(r["strict_tp"], r["n_pred"], r["n_gold"])["f1"],
                "lenient_f1": prf(r["lenient_tp"], r["n_pred"], r["n_gold"])["f1"]}
            for s, r in per_study.items()}


def compare(a: dict, b: dict, studies: list[str]) -> dict[str, Any]:
    """Paired comparison B - A over the same studies."""
    out: dict[str, Any] = {}
    rng = np.random.default_rng(SEED)
    for key in ("strict", "lenient"):
        fa = [prf(a[s][f"{key}_tp"], a[s]["n_pred"], a[s]["n_gold"])["f1"] for s in studies]
        fb = [prf(b[s][f"{key}_tp"], b[s]["n_pred"], b[s]["n_gold"])["f1"] for s in studies]
        diffs = [y - x for x, y in zip(fa, fb)]
        idx = rng.integers(0, len(studies), size=(N_BOOT, len(studies)))
        boot = np.asarray(diffs)[idx].mean(axis=1)
        out[key] = {
            "mean_per_study_delta_f1": round(float(np.mean(diffs)), 4),
            "ci95_study_bootstrap": [round(float(np.percentile(boot, 2.5)), 4),
                                     round(float(np.percentile(boot, 97.5)), 4)],
            "sign_flip_p": round(sign_flip_test(diffs), 4),
            "min_attainable_p": round(2 / 2 ** len(studies), 4),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", type=Path, required=True)
    ap.add_argument("--pred-b", type=Path)
    ap.add_argument("--split", default="val", choices=["train", "val", "test", "all"])
    ap.add_argument("--label", default="A")
    ap.add_argument("--label-b", default="B")
    ap.add_argument("--truncated-policy", default="salvage", choices=["salvage", "zero"])
    ap.add_argument("--gold-rows", action="store_true",
                    help="score against raw gold rows instead of deduplicated gold pairs (v1 behaviour)")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    dedup = not args.gold_rows
    summary_a, per_a, _, _ = evaluate(args.pred, args.split, args.truncated_policy, dedup)
    report: dict[str, Any] = {
        "_evaluator": "eval_extraction.py v2 (EVALUATION_PROTOCOL_v2.md)",
        "_split": args.split, "_gold": GOLD_CSV.name, "_truncated_policy": args.truncated_policy,
        "_gold_dedup": dedup,
        args.label: {"file": str(args.pred), "summary": summary_a, "per_study": per_study_table(per_a)},
    }
    if args.pred_b:
        summary_b, per_b, _, _ = evaluate(args.pred_b, args.split, args.truncated_policy, dedup)
        report[args.label_b] = {"file": str(args.pred_b), "summary": summary_b,
                                "per_study": per_study_table(per_b)}
        report[f"delta_{args.label_b}_minus_{args.label}"] = compare(per_a, per_b, sorted(per_a))

    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
