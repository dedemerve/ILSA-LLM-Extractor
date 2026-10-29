"""
Baseline Inference Script — Mistral 7B Few-Shot Extraction
===========================================================
Protocol source: EVALUATION_PROTOCOL_v1.md (frozen)
Prompt:          outputs/finetuning/baseline/baseline_prompt_frozen.txt (FROZEN)
Gold source:     outputs/finetuning/data/gold_standard_full_corpus_v1.4.1.csv
Model:           mistralai/Mistral-7B-Instruct-v0.3
Temperature:     0.0
Max tokens:      8192

Usage:
    python3 scripts/baseline_inference.py --split test
    python3 scripts/baseline_inference.py --split all
    python3 scripts/baseline_inference.py --split val

Requires:
    HF_TOKEN in .env  (HuggingFace Inference API for Mistral-7B-Instruct-v0.3)
    OR --backend together with TOGETHER_API_KEY in .env

DO NOT run LoRA or modify Stage 5 outputs after this script produces results.
The output baseline_results_frozen.json is the frozen BASE condition.
"""

import os
import sys
import json
import time
import re
import argparse
import hashlib
import pandas as pd
from pathlib import Path
from datetime import datetime

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ─── paths ────────────────────────────────────────────────────────────────────
ROOT          = Path(__file__).parent.parent
GOLD_CSV      = ROOT / "outputs/finetuning/data/gold_standard_full_corpus_v1.4.1.csv"
SPLIT_CSV     = ROOT / "outputs/finetuning/data/gold_standard_split_v4.csv"
PROMPT_FILE   = ROOT / "outputs/finetuning/baseline/baseline_prompt_frozen.txt"
SOURCE_DIR    = ROOT / "outputs/finetuning/source_recovery/texts"
OUT_DIR       = ROOT / "outputs/finetuning/baseline"
AUDIT_DIR     = ROOT / "outputs/finetuning/audit"

# ─── frozen inference parameters ──────────────────────────────────────────────
MODEL_ID      = "mistralai/Mistral-7B-Instruct-v0.3"
TEMPERATURE   = 0.0
MAX_TOKENS    = 8192
CONTEXT_WIN   = 32768
MAX_EXCERPT_CHARS = 75_000   # ~18,750 tokens; leaves room for static prompt (~4,400 tok)

# ─── controlled vocabularies (for validation) ─────────────────────────────────
VALID_CONSTRUCTS = {
    "AGE_GROUP","ANXIETY","DISCIPLINARY_CLIMATE","EDUCATIONAL_ATTAINMENT",
    "GENDER","HOMEWORK_TIME","HOME_LITERACY_ACTIVITIES","ICT_HOME_ACCESS",
    "IMMIGRANT_STATUS","INSTRUCTION_TIME","JOB_TRAINING","LITERACY_OUTCOME",
    "MOTIVATION","NUMERACY_OUTCOME","PARENTAL_EDUCATION","PRIVATE_TUTORING",
    "PROCESS_DATA_BEHAVIOR","SCHOOL_BELONGING","SCHOOL_ICT_INFRASTRUCTURE",
    "SCHOOL_LEADERSHIP","SCHOOL_LOCATION","SELF_CONCEPT","SES_COMPOSITE",
    "TEACHER_EXPERIENCE","TEACHER_ICT_USE","TEACHER_PD",
    "TEACHER_SELF_EFFICACY_CLS_MGMT","TEACHING_QUALITY","WORKPLACE_AUTONOMY",
    "UNMATCHED",
}
VALID_DIRECTIONS    = {"positive","negative","mixed","not_reported"}
VALID_EFFECT_TYPES  = {"E_assoc","E_predictive","E_cross_cycle","E_cross_country"}
VALID_PV_HANDLING   = {"rubin_rules","average_pv","single_pv","all_pvs","not_applicable","not_reported"}
SCHEMA_FIELDS       = [
    "predictor","ilsa_variable","canonical_construct","outcome",
    "effect_type","direction","effect_size","plausible_values_handling",
    "student_weights_used","evidence_text",
]


# ─── helpers ──────────────────────────────────────────────────────────────────

def load_prompt_template() -> str:
    """Load the frozen system message from baseline_prompt_frozen.txt."""
    text = PROMPT_FILE.read_text(encoding="utf-8")
    # Extract just the SYSTEM MESSAGE block
    start = text.find("SYSTEM MESSAGE\n" + "="*100)
    end   = text.find("FEW-SHOT DEMONSTRATIONS\n" + "="*100)
    if start == -1 or end == -1:
        raise ValueError("Could not locate SYSTEM MESSAGE block in prompt file.")
    system_block = text[start:end].strip()
    # Strip the header line
    lines = system_block.splitlines()
    system_msg = "\n".join(lines[2:]).strip()  # skip "SYSTEM MESSAGE\n====..."
    return system_msg


def load_few_shot_examples() -> str:
    """Load the four demonstration blocks verbatim from the frozen prompt."""
    text = PROMPT_FILE.read_text(encoding="utf-8")
    start = text.find("FEW-SHOT DEMONSTRATIONS\n" + "="*100)
    end   = text.find("USER MESSAGE TEMPLATE")
    if start == -1 or end == -1:
        raise ValueError("Could not locate FEW-SHOT DEMONSTRATIONS block.")
    demos_block = text[start:end].strip()
    lines = demos_block.splitlines()
    return "\n".join(lines[2:]).strip()  # skip header


def find_source_text(study_id: str) -> str | None:
    """Find the source .txt file for a given study PDF ID."""
    stem = Path(study_id).stem
    for p in SOURCE_DIR.glob("*.txt"):
        if p.stem == stem or p.stem.startswith(stem[:20]):
            return p.read_text(encoding="utf-8", errors="replace")
    # fuzzy fallback: compare stems
    for p in SOURCE_DIR.glob("*.txt"):
        if stem[:15].lower() in p.stem.lower():
            return p.read_text(encoding="utf-8", errors="replace")
    return None


def truncate_to_budget(text: str, max_chars: int = MAX_EXCERPT_CHARS) -> str:
    """Truncate source text to fit context budget; prefer abstract + results."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n\n[... truncated to fit context window ...]"


def build_user_message(program: str, cycle: str, study_id: str, source_text: str) -> str:
    return (
        f"ILSA ANNOTATION TASK\n\n"
        f"PROGRAM: {program}\n"
        f"CYCLE: {cycle}\n"
        f"SOURCE: {study_id}\n\n"
        f"SOURCE TEXT (abstract + methods + results excerpt):\n{source_text}\n\n"
        f"TASK: Extract all predictor–outcome evidence pairs from this text.\n"
        f"For each pair, provide the fields below.\n\n"
        f"Return a JSON array. Each element of the array must contain exactly these fields:\n\n"
        f'{{\n'
        f'  "predictor": "raw predictor string as it appears in the source text",\n'
        f'  "ilsa_variable": "ILSA survey variable code if identifiable, else null",\n'
        f'  "canonical_construct": "canonical construct label from codebook v1.3",\n'
        f'  "outcome": "target variable as described in the source text",\n'
        f'  "effect_type": "E_assoc | E_predictive | E_cross_cycle | E_cross_country",\n'
        f'  "direction": "positive | negative | mixed | not_reported",\n'
        f'  "effect_size": "reported effect size or coefficient, else null",\n'
        f'  "plausible_values_handling": "rubin_rules | average_pv | single_pv | all_pvs | not_applicable | not_reported",\n'
        f'  "student_weights_used": true | false | null,\n'
        f'  "evidence_text": "verbatim sentence(s) from source text supporting this extraction"\n'
        f'}}\n\n'
        f"Return ONLY the JSON array. No prose before or after."
    )


def parse_json_output(raw: str) -> list[dict]:
    """Parse the model's raw string output into a list of evidence pair dicts."""
    # Strip markdown code fences if present
    raw = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.MULTILINE)
    raw = re.sub(r"```\s*$", "", raw.strip(), flags=re.MULTILINE)
    raw = raw.strip()
    # Find JSON array
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if not match:
        return []
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        # Try to fix common issues (trailing commas)
        cleaned = re.sub(r",\s*([}\]])", r"\1", match.group(0))
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return []


def validate_pair(pair: dict) -> dict:
    """Return a validation summary for one extracted pair."""
    issues = []
    for f in SCHEMA_FIELDS:
        if f not in pair:
            issues.append(f"missing_field:{f}")
    if pair.get("canonical_construct") not in VALID_CONSTRUCTS:
        issues.append(f"invalid_construct:{pair.get('canonical_construct')}")
    if pair.get("direction") not in VALID_DIRECTIONS:
        issues.append(f"invalid_direction:{pair.get('direction')}")
    if pair.get("effect_type") not in VALID_EFFECT_TYPES:
        issues.append(f"invalid_effect_type:{pair.get('effect_type')}")
    if pair.get("plausible_values_handling") not in VALID_PV_HANDLING:
        issues.append(f"invalid_pv:{pair.get('plausible_values_handling')}")
    if not pair.get("evidence_text"):
        issues.append("null_evidence_text")
    return {"valid": len(issues) == 0, "issues": issues}


# ─── inference backends ────────────────────────────────────────────────────────

def call_hf_inference(system_msg: str, few_shot: str, user_msg: str) -> str:
    """Call Mistral-7B-Instruct-v0.3 via HuggingFace Inference API."""
    from huggingface_hub import InferenceClient

    token = os.environ.get("HF_TOKEN", "")
    if not token:
        raise EnvironmentError(
            "HF_TOKEN not set. Add HF_TOKEN=hf_xxx to your .env file.\n"
            "Get a token at https://huggingface.co/settings/tokens"
        )

    client = InferenceClient(model=MODEL_ID, token=token)

    # Mistral instruct format: system + few-shot context + user turn
    full_system = system_msg + "\n\n" + few_shot
    messages = [
        {"role": "system", "content": full_system},
        {"role": "user",   "content": user_msg},
    ]

    response = client.chat_completion(
        messages=messages,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )
    return response.choices[0].message.content


def call_together_inference(system_msg: str, few_shot: str, user_msg: str) -> str:
    """Call Mistral-7B-Instruct-v0.3 via Together AI API."""
    import httpx

    key = os.environ.get("TOGETHER_API_KEY", "")
    if not key:
        raise EnvironmentError("TOGETHER_API_KEY not set in .env")

    full_system = system_msg + "\n\n" + few_shot
    payload = {
        "model": "mistralai/Mistral-7B-Instruct-v0.3",
        "messages": [
            {"role": "system", "content": full_system},
            {"role": "user",   "content": user_msg},
        ],
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
    }
    resp = httpx.post(
        "https://api.together.xyz/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json=payload,
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


# ─── matching ──────────────────────────────────────────────────────────────────

def normalize(s) -> str:
    if s is None:
        return ""
    return str(s).strip().lower()


def match_pairs(predicted: list[dict], gold_rows: pd.DataFrame) -> dict:
    """
    Soft matching: match predicted pairs to gold rows by canonical_construct.
    For each unique canonical_construct in gold, check if any predicted pair
    has the same construct.

    Returns per-field TP/FP/FN counts and matched pairs for downstream analysis.
    """
    gold_constructs = list(gold_rows["canonical_construct"].str.lower())
    pred_constructs = [normalize(p.get("canonical_construct")) for p in predicted]

    results = {
        "n_gold":      len(gold_rows),
        "n_predicted": len(predicted),
        "matched":     [],
        "unmatched_gold": [],
        "extra_predicted": [],
        "field_accuracy": {},
    }

    used_pred = set()
    for _, grow in gold_rows.iterrows():
        gc = normalize(grow["canonical_construct"])
        # Find first unused predicted pair with matching construct
        match_idx = None
        for i, pc in enumerate(pred_constructs):
            if i not in used_pred and pc == gc:
                match_idx = i
                break

        if match_idx is not None:
            used_pred.add(match_idx)
            pred = predicted[match_idx]
            matched = {
                "gold_record_id":          grow["record_id"],
                "gold_construct":          grow["canonical_construct"],
                "pred_construct":          pred.get("canonical_construct"),
                "construct_match":         normalize(pred.get("canonical_construct")) == gc,
            }
            # Compare each Tier-1 field
            for field, gold_col in [
                ("direction",              "direction_canonical"),
                ("effect_type",            "effect_type"),
                ("plausible_values_handling", "plausible_values_handling"),
                ("student_weights_used",   "student_weights_used"),
            ]:
                gold_val = normalize(grow.get(gold_col, ""))
                pred_val = normalize(pred.get(field))
                matched[f"gold_{field}"] = str(grow.get(gold_col, ""))
                matched[f"pred_{field}"] = str(pred.get(field, ""))
                matched[f"{field}_match"] = gold_val == pred_val

            matched["evidence_text_present"] = bool(pred.get("evidence_text"))
            results["matched"].append(matched)
        else:
            results["unmatched_gold"].append({
                "gold_record_id": grow["record_id"],
                "gold_construct": grow["canonical_construct"],
            })

    for i, pred in enumerate(predicted):
        if i not in used_pred:
            results["extra_predicted"].append({
                "pred_construct": pred.get("canonical_construct"),
                "pred_predictor": pred.get("predictor"),
            })

    # Aggregate field accuracy over matched pairs
    fields = ["construct_match","direction_match","effect_type_match",
              "plausible_values_handling_match","student_weights_used_match",
              "evidence_text_present"]
    for f in fields:
        vals = [m[f] for m in results["matched"]]
        results["field_accuracy"][f] = sum(vals) / len(vals) if vals else None

    return results


# ─── evaluation metrics ────────────────────────────────────────────────────────

def compute_metrics(all_study_results: list[dict]) -> dict:
    """
    Aggregate per-study match results into corpus-level metrics.
    Implements EVALUATION_PROTOCOL_v1.md §4.1–4.4.
    """
    total_gold = sum(r["n_gold"]      for r in all_study_results)
    total_pred = sum(r["n_predicted"] for r in all_study_results)
    total_tp   = sum(len(r["matched"]) for r in all_study_results)  # matched pairs
    total_fp   = sum(len(r["extra_predicted"]) for r in all_study_results)
    total_fn   = sum(len(r["unmatched_gold"])   for r in all_study_results)

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall    = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # Field-level accuracy over all matched pairs
    all_matched = [m for r in all_study_results for m in r["matched"]]
    fields = ["construct_match","direction_match","effect_type_match",
              "plausible_values_handling_match","student_weights_used_match",
              "evidence_text_present"]
    field_acc = {}
    for f in fields:
        vals = [m[f] for m in all_matched]
        field_acc[f] = round(sum(vals) / len(vals), 4) if vals else None

    # Construct-level confusion
    construct_tp: dict[str, int] = {}
    construct_fn: dict[str, int] = {}
    for r in all_study_results:
        for m in r["matched"]:
            gc = m["gold_construct"]
            construct_tp[gc] = construct_tp.get(gc, 0) + (1 if m["construct_match"] else 0)
        for ug in r["unmatched_gold"]:
            gc = ug["gold_construct"]
            construct_fn[gc] = construct_fn.get(gc, 0) + 1

    # Hallucination rate: predicted constructs not in valid vocab
    all_pred_constructs = [
        m["pred_construct"]
        for r in all_study_results
        for m in r["matched"]
    ] + [
        ep["pred_construct"]
        for r in all_study_results
        for ep in r["extra_predicted"]
    ]
    hallucinated = [c for c in all_pred_constructs
                    if c and c.upper() not in VALID_CONSTRUCTS]
    hallucination_rate = len(hallucinated) / len(all_pred_constructs) if all_pred_constructs else 0

    return {
        "pair_level": {
            "total_gold":      total_gold,
            "total_predicted": total_pred,
            "TP":  total_tp,
            "FP":  total_fp,
            "FN":  total_fn,
            "precision": round(precision, 4),
            "recall":    round(recall, 4),
            "F1":        round(f1, 4),
        },
        "field_accuracy": field_acc,
        "hallucination_rate": round(hallucination_rate, 4),
        "construct_tp_counts": construct_tp,
        "construct_fn_counts": construct_fn,
    }


# ─── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Baseline inference: Mistral 7B few-shot extraction")
    parser.add_argument("--split",   default="test", choices=["test","val","train","all"],
                        help="Which split to run on (default: test)")
    parser.add_argument("--backend", default="hf", choices=["hf","together"],
                        help="Inference backend (default: hf = HuggingFace Inference API)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print prompts but do not call the API")
    args = parser.parse_args()

    # ── load frozen components ──────────────────────────────────────────────
    print(f"[{datetime.now():%H:%M:%S}] Loading frozen prompt from {PROMPT_FILE.name} ...")
    system_msg = load_prompt_template()
    few_shot   = load_few_shot_examples()

    # ── load gold ───────────────────────────────────────────────────────────
    gold = pd.read_csv(GOLD_CSV)
    # Merge split assignments from split_v4 (v1.4.1 doesn't carry split column)
    split_df = pd.read_csv(SPLIT_CSV)[["record_id", "split"]]
    gold = gold.merge(split_df, on="record_id", how="left")
    missing_split = gold["split"].isna().sum()
    if missing_split > 0:
        print(f"  WARNING: {missing_split} gold rows have no split assignment (dropped).")
        gold = gold.dropna(subset=["split"])

    # Normalize direction_canonical: map '1'/'-1' numeric to string
    gold["direction_canonical"] = gold["direction_canonical"].apply(lambda x:
        "positive"     if str(x).strip() in ("1", "1.0")  else
        "negative"     if str(x).strip() in ("-1", "-1.0") else
        str(x).strip()
    )

    if args.split == "all":
        target = gold
    else:
        target = gold[gold["split"] == args.split]

    study_ids = target["study_id"].unique().tolist()
    print(f"[{datetime.now():%H:%M:%S}] Split='{args.split}': {len(study_ids)} studies, {len(target)} gold rows")

    # ── per-study inference ─────────────────────────────────────────────────
    all_predictions: list[dict] = []
    all_study_results: list[dict] = []
    failed_studies: list[str] = []

    for i, study_id in enumerate(study_ids, 1):
        gold_rows = target[target["study_id"] == study_id]
        program   = gold_rows["program"].iloc[0]
        cycle     = gold_rows["cycle"].iloc[0]

        print(f"[{datetime.now():%H:%M:%S}] [{i}/{len(study_ids)}] {study_id} "
              f"({program}, {len(gold_rows)} gold rows)")

        # Load source text
        source_raw = find_source_text(study_id)
        if source_raw is None:
            print(f"  WARNING: source text not found for {study_id}. Skipping.")
            failed_studies.append(study_id)
            continue
        source_text = truncate_to_budget(source_raw)
        if len(source_raw) > MAX_EXCERPT_CHARS:
            print(f"  INFO: source text truncated from {len(source_raw):,} to {MAX_EXCERPT_CHARS:,} chars.")

        user_msg = build_user_message(program, cycle, study_id, source_text)

        if args.dry_run:
            print(f"  [DRY RUN] Would call {MODEL_ID}, user msg length={len(user_msg)} chars")
            continue

        # Call inference backend
        try:
            call_fn = call_hf_inference if args.backend == "hf" else call_together_inference
            raw_output = call_fn(system_msg, few_shot, user_msg)
            time.sleep(1.0)  # rate limit courtesy
        except Exception as e:
            print(f"  ERROR calling API for {study_id}: {e}")
            failed_studies.append(study_id)
            continue

        # Parse output
        pairs = parse_json_output(raw_output)
        print(f"  Extracted {len(pairs)} pairs (gold={len(gold_rows)})")

        # Validate each pair
        for p in pairs:
            v = validate_pair(p)
            p["_validation"] = v

        # Match against gold
        match_result = match_pairs(pairs, gold_rows)
        match_result["study_id"] = study_id
        all_study_results.append(match_result)

        all_predictions.append({
            "study_id":   study_id,
            "program":    program,
            "cycle":      cycle,
            "n_gold":     int(len(gold_rows)),
            "n_predicted": len(pairs),
            "raw_output": raw_output,
            "pairs":      pairs,
        })

    if args.dry_run:
        print(f"\n[DRY RUN complete] Would process {len(study_ids)} studies.")
        return

    if not all_predictions:
        print("No predictions generated. Check API key and source text availability.")
        sys.exit(1)

    # ── compute metrics ─────────────────────────────────────────────────────
    metrics = compute_metrics(all_study_results)

    # ── save outputs ────────────────────────────────────────────────────────
    run_ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_hash = hashlib.sha256(json.dumps(all_predictions, sort_keys=True).encode()).hexdigest()[:12]

    pred_file = OUT_DIR / f"baseline_predictions_{args.split}.json"
    results_file = OUT_DIR / "baseline_results_frozen.json" if args.split == "test" else \
                   OUT_DIR / f"baseline_results_{args.split}_{run_ts}.json"

    pred_out = {
        "_generated":     run_ts,
        "_model":         MODEL_ID,
        "_temperature":   TEMPERATURE,
        "_max_tokens":    MAX_TOKENS,
        "_split":         args.split,
        "_gold_source":   "gold_standard_full_corpus_v1.4.1.csv",
        "_prompt_file":   "baseline_prompt_frozen.txt",
        "_run_hash":      run_hash,
        "_failed_studies": failed_studies,
        "predictions":    all_predictions,
    }
    results_out = {
        "_generated":    run_ts,
        "_model":        MODEL_ID,
        "_split":        args.split,
        "_n_studies":    len(all_predictions),
        "_gold_source":  "gold_standard_full_corpus_v1.4.1.csv",
        "_failed":       failed_studies,
        "metrics":       metrics,
        "per_study":     all_study_results,
    }

    pred_file.write_text(json.dumps(pred_out, indent=2, ensure_ascii=False), encoding="utf-8")
    results_file.write_text(json.dumps(results_out, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n{'='*60}")
    print(f"BASELINE INFERENCE COMPLETE — split={args.split}")
    print(f"{'='*60}")
    print(f"Studies processed:  {len(all_predictions)}")
    print(f"Studies failed:     {len(failed_studies)}")
    print(f"\nPair-level metrics:")
    pl = metrics["pair_level"]
    print(f"  Gold pairs:       {pl['total_gold']}")
    print(f"  Predicted pairs:  {pl['total_predicted']}")
    print(f"  TP / FP / FN:     {pl['TP']} / {pl['FP']} / {pl['FN']}")
    print(f"  Precision:        {pl['precision']:.4f}")
    print(f"  Recall:           {pl['recall']:.4f}")
    print(f"  F1:               {pl['F1']:.4f}")
    print(f"\nField accuracy (matched pairs):")
    for k, v in metrics["field_accuracy"].items():
        print(f"  {k:<40} {v:.4f}" if v is not None else f"  {k:<40} N/A")
    print(f"\nHallucination rate: {metrics['hallucination_rate']:.4f}")
    print(f"\nOutputs saved:")
    print(f"  {pred_file}")
    print(f"  {results_file}")

    if args.split == "test":
        print(f"\nNOTE: baseline_results_frozen.json is the frozen BASE condition.")
        print(f"Do NOT re-run this script on the test set to improve metrics.")
        print(f"Do NOT modify the baseline prompt after this run.")


if __name__ == "__main__":
    main()
