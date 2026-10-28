"""
Extraction runner v2 (action plan steps 2, 4 and 7).

Modes
  chunked   v2 pipeline: one constrained call per passage + one methods call per study,
            then merge (dedupe, grounding filter, attach study-level fields)
  document  frozen v1 prompt over the whole (75k-char truncated) source, used only to
            re-run BASE and the Stage 7 adapter on validation under identical settings

Backends
  vllm      local GPU (Colab). Optional LoRA adapter. JSON-schema constrained decoding.
  openai    any OpenAI-compatible endpoint (Together, OpenRouter, Gemini, an HF Inference
            Endpoint running vLLM). response_format=json_schema when --schema is on,
            Pydantic validation with one re-ask on failure.

Prompting arm (step 4): --demos K adds K retrieved train-split demonstrations per passage
(TF-IDF cosine over passages; train split only, so validation/test text never leaks in).

Output: a predictions JSON readable by eval_extraction.py.

The test split is locked: each --arm label may be run on test once (recorded in
outputs/finetuning/v2/test_runs.json), and only with --final-test-run.

Examples
  # step 2: BASE vs Stage 7 adapter on validation, identical settings
  python3 scripts/finetuning/extract_v2.py --mode document --backend vllm \
      --model mistralai/Mistral-7B-Instruct-v0.3 --split val --arm base_mistral_doc
  python3 scripts/finetuning/extract_v2.py --mode document --backend vllm \
      --model mistralai/Mistral-7B-Instruct-v0.3 --adapter /content/lora_adapter_final --split val --arm stage7_lora_doc
  # step 4: RAG few-shot prompting arm via Together
  python3 scripts/finetuning/extract_v2.py --mode chunked --backend openai --base-url https://api.together.xyz/v1 \
      --api-key-env TOGETHER_API_KEY --model meta-llama/Llama-3.3-70B-Instruct-Turbo --demos 2 --split val --arm rag2_llama70b
  # step 7: fine-tuned Qwen3-8B, constrained
  python3 scripts/finetuning/extract_v2.py --mode chunked --backend vllm --model Qwen/Qwen3-8B \
      --adapter outputs/finetuning/v2/runs/qwen3_8b_r16/checkpoint-81 --split val --arm qwen3_lora_e3
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from chunking import (  # noqa: E402
    chunk_text, clean_source, methods_passages, render_chunk_prompt, render_methods_prompt,
    user_messages,
)
from eval_extraction import _shingles, find_source_text, grounding_score, load_gold  # noqa: E402


def load_corpus():
    """Return DataFrame(study_id, title, program, cycle) for 577 text-extracted corpus studies.

    Metadata for studies present in gold comes from gold; non-gold studies default to
    program='PISA' and cycle='multiple', which is correct for the vast majority of the corpus.
    """
    import pandas as pd

    inv = pd.read_csv(INVENTORY_CSV, dtype=str)
    inv = inv[inv["text_extracted"].str.strip().str.lower() == "true"].copy()
    gold = pd.read_csv(GOLD_CSV, dtype=str)
    gold_meta = gold.groupby("study_id")[["title", "program", "cycle"]].first().reset_index()
    merged = inv.merge(gold_meta, left_on="file_name", right_on="study_id", how="left",
                       suffixes=("_inv", "_gold"))
    result = merged.copy()
    result["study_id"] = merged["file_name"]
    result["title"] = merged["title_gold"].fillna(merged["title_inv"])
    result["program"] = merged["program"].fillna("PISA")
    result["cycle"] = merged["cycle"].fillna("multiple")
    return result[["study_id", "title", "program", "cycle"]]
from ilsa_schema import (  # noqa: E402
    CHUNK_FIELDS, MethodsOutput, json_schema, methods_json_schema, normalize_text, parse_output,
)
from pydantic import ValidationError  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "outputs/finetuning/v2/predictions"
TEST_LOCK = ROOT / "outputs/finetuning/v2/test_runs.json"
FROZEN_PROMPT = ROOT / "outputs/finetuning/baseline/baseline_prompt_frozen.txt"
SFT_TRAIN = ROOT / "outputs/finetuning/v2/data/sft_train.jsonl"
INVENTORY_CSV = ROOT / "outputs/finetuning/source_recovery/full_source_inventory.csv"
GOLD_CSV = ROOT / "outputs/finetuning/data/gold_standard_full_corpus_v1.4.1.csv"

MAX_ITEMS_PER_CHUNK = 8          # train gold maximum per passage is 7
CHUNK_MAX_TOKENS = 1200
METHODS_MAX_TOKENS = 300
DOC_MAX_TOKENS = 8192
DOC_MAX_CHARS = 75_000
FREQUENCY_PENALTY = 0.2
GROUNDING_MIN = 0.9


# Prompts

def frozen_document_messages(program: str, cycle: str, study_id: str, source: str) -> list[dict[str, str]]:
    """Reproduces the Stage 9 prompt: frozen system message + demonstrations merged into the user turn."""
    sys.path.insert(0, str(ROOT / "scripts"))
    from baseline_inference import build_user_message, load_few_shot_examples, load_prompt_template

    system = load_prompt_template() + "\n\n" + load_few_shot_examples()
    user = build_user_message(program, cycle, study_id, source[:DOC_MAX_CHARS])
    return [{"role": "user", "content": system + "\n\n" + user}]


class DemoRetriever:
    """TF-IDF retrieval of positive train-split passages as few-shot demonstrations."""

    def __init__(self, k: int):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.k = k
        recs = [json.loads(l) for l in SFT_TRAIN.read_text(encoding="utf-8").splitlines() if l.strip()]
        self.demos = [(r["prompt"][0]["content"], r["completion"][0]["content"]) for r in recs
                      if r["completion"][0]["content"].startswith('{"items":[{')]
        self.passages = [p.split("PASSAGE ", 1)[-1] for p, _ in self.demos]
        self.vec = TfidfVectorizer(sublinear_tf=True, stop_words="english", min_df=1)
        self.matrix = self.vec.fit_transform(self.passages)

    def messages(self, passage: str, exclude_study_title: str) -> list[dict[str, str]]:
        sims = (self.matrix @ self.vec.transform([passage]).T).toarray().ravel()
        out: list[dict[str, str]] = []
        for i in sims.argsort()[::-1]:
            prompt, target = self.demos[i]
            if exclude_study_title and f"ARTICLE: {exclude_study_title}\n" in prompt:
                continue
            out += [{"role": "user", "content": prompt}, {"role": "assistant", "content": target}]
            if len(out) == 2 * self.k:
                break
        return out


# Backends

class VLLMBackend:
    def __init__(self, model: str, adapter: str | None, max_model_len: int, schema: bool):
        from vllm import LLM

        self.schema = schema
        self.adapter = adapter
        self.llm = LLM(model=model, dtype="bfloat16", max_model_len=max_model_len,
                       enable_lora=adapter is not None, max_lora_rank=64, gpu_memory_utilization=0.9,
                       seed=0)
        self.tok = self.llm.get_tokenizer()

    def _params(self, max_tokens: int, schema: dict | None) -> Any:
        from vllm import SamplingParams

        kwargs: dict[str, Any] = {"temperature": 0.0, "max_tokens": max_tokens,
                                  "frequency_penalty": FREQUENCY_PENALTY}
        if schema is not None and self.schema:
            try:
                from vllm.sampling_params import StructuredOutputsParams
                kwargs["structured_outputs"] = StructuredOutputsParams(json=schema)
            except ImportError:  # vLLM < 0.11
                from vllm.sampling_params import GuidedDecodingParams
                kwargs["guided_decoding"] = GuidedDecodingParams(json=schema)
        return SamplingParams(**kwargs)

    def generate(self, batch: list[tuple[list[dict], int, dict | None]]) -> list[tuple[str, str]]:
        from vllm.lora.request import LoRARequest

        prompts = [self.tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True,
                                                enable_thinking=False) for m, _, _ in batch]
        params = [self._params(mt, sc) for _, mt, sc in batch]
        lora = LoRARequest("adapter", 1, self.adapter) if self.adapter else None
        outs = self.llm.generate(prompts, params, lora_request=lora)
        return [(o.outputs[0].text, o.outputs[0].finish_reason) for o in outs]


class OpenAIBackend:
    def __init__(self, model: str, base_url: str, api_key_env: str, schema: bool,
                 min_interval: float = 0.0, extra_body: dict | None = None):
        from openai import OpenAI

        key = os.environ.get(api_key_env)
        if not key:
            raise SystemExit(f"{api_key_env} is not set")
        self.client = OpenAI(base_url=base_url, api_key=key)
        self.model = model
        self.schema = schema
        self.min_interval = min_interval  # client-side throttle for tokens-per-minute limits
        self.extra_body = extra_body or {}
        self._last = 0.0

    def _call(self, messages: list[dict], max_tokens: int, schema: dict | None) -> tuple[str, str]:
        kwargs: dict[str, Any] = {"model": self.model, "messages": messages, "temperature": 0.0,
                                  "max_tokens": max_tokens}
        if self.extra_body:
            kwargs["extra_body"] = self.extra_body
        if schema is not None and self.schema:
            kwargs["response_format"] = {"type": "json_schema",
                                         "json_schema": {"name": "extraction", "schema": schema, "strict": True}}
        for attempt in range(6):
            wait = self.min_interval - (time.time() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.time()
            try:
                r = self.client.chat.completions.create(**kwargs)
                return r.choices[0].message.content or "", r.choices[0].finish_reason or ""
            except Exception as e:  # noqa: BLE001  rate limits / transient errors
                status = getattr(e, "status_code", None)
                if status == 400 and "max completion tokens reached" in str(e):
                    # Groq rejects schema outputs cut by max_tokens; this is a truncation, not a failure
                    body = getattr(e, "body", None) or {}
                    partial = body.get("failed_generation", "") if isinstance(body, dict) else ""
                    return partial or "", "length"
                if attempt == 5 or (status is not None and 400 <= status < 500 and status != 429):
                    raise
                time.sleep(min(2 ** attempt * 5, 90))
                print(f"  retry after error: {e}", file=sys.stderr)
        raise RuntimeError("unreachable")

    def generate(self, batch: list[tuple[list[dict], int, dict | None]]) -> list[tuple[str, str]]:
        out = []
        for messages, max_tokens, schema in batch:
            try:
                text, finish = self._call(messages, max_tokens, schema)
                err = validation_error(text, schema)
                if err and schema is not None:  # one re-ask with the validation error (instructor-style)
                    note = {"role": "user", "content": f"The previous output failed validation: {err[:500]}. "
                                                       "Return only corrected JSON."}
                    retry = messages + ([{"role": "assistant", "content": text}] if text.strip() else []) + [note]
                    text, finish = self._call(retry, max_tokens, schema)
            except Exception as e:  # noqa: BLE001  one failed call must not abort the study
                print(f"  call failed: {str(e)[:300]}", file=sys.stderr)
                text, finish = "", "error"
            out.append((text, finish))
        return out


class OracleBackend:
    """Answers each prompt with its SFT target (gold-derived) and {"items": []} otherwise.
    Measures the ceiling of the chunked pipeline (pairs the builder could not locate are lost)
    and checks that training and inference render byte-identical prompts."""

    def __init__(self, split: str):
        path = ROOT / f"outputs/finetuning/v2/data/sft_{split}.jsonl"
        self.answers = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            self.answers[r["prompt"][-1]["content"]] = r["completion"][0]["content"]

    def generate(self, batch: list[tuple[list[dict], int, dict | None]]) -> list[tuple[str, str]]:
        return [(self.answers.get(m[-1]["content"], '{"items":[]}'), "stop") for m, _, _ in batch]


def validation_error(text: str, schema: dict | None) -> str | None:
    if schema is None:
        return None
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as e:
        return f"invalid JSON: {e}"
    try:
        if "items" in schema.get("properties", {}):
            from ilsa_schema import EvidencePair
            for item in obj["items"]:
                EvidencePair(**{**item, "plausible_values_handling": "not_reported", "student_weights_used": None})
        else:
            MethodsOutput(**obj)
    except (ValidationError, KeyError, TypeError) as e:
        return str(e)
    return None


# Merge

def merge_study(chunk_results: list[dict[str, Any]], methods: dict[str, Any]) -> tuple[list[dict], dict[str, int]]:
    stats = {"raw_items": 0, "ungrounded_dropped": 0, "duplicates_dropped": 0}
    kept: dict[tuple, dict] = {}
    for cr in chunk_results:
        sh = _shingles(cr["passage"])
        for item in cr["items"]:
            stats["raw_items"] += 1
            if grounding_score(item.get("evidence_text") or "", sh) < GROUNDING_MIN:
                stats["ungrounded_dropped"] += 1
                continue
            # same identity as the evaluator's gold dedup key
            key = (normalize_text(item.get("predictor")), item.get("canonical_construct"),
                   normalize_text(item.get("outcome")), item.get("direction"), item.get("effect_type"))
            if key in kept:
                stats["duplicates_dropped"] += 1
                if len(item.get("evidence_text") or "") <= len(kept[key].get("evidence_text") or ""):
                    continue
            kept[key] = item
    pairs = [{**{k: p.get(k) for k in CHUNK_FIELDS},
              "plausible_values_handling": methods.get("plausible_values_handling", "not_reported"),
              "student_weights_used": methods.get("student_weights_used")} for p in kept.values()]
    return pairs, stats


# Runner

def check_test_lock(split: str, arm: str, final: bool) -> None:
    if split != "test":
        return
    if not final:
        raise SystemExit("The test split is locked. Pass --final-test-run only for the single pre-registered run.")
    runs = json.loads(TEST_LOCK.read_text()) if TEST_LOCK.exists() else {}
    if arm in runs:
        raise SystemExit(f"arm '{arm}' was already run on test at {runs[arm]}; no second run is allowed.")
    runs[arm] = datetime.now(timezone.utc).isoformat()
    TEST_LOCK.parent.mkdir(parents=True, exist_ok=True)
    TEST_LOCK.write_text(json.dumps(runs, indent=2))


def run(args: argparse.Namespace) -> Path:
    if args.split == "corpus":
        import pandas as pd
        meta_df = load_corpus()
        study_meta = {row["study_id"]: (row["title"], row["program"], row["cycle"])
                      for _, row in meta_df.iterrows()}
        studies = sorted(study_meta.keys())
    else:
        gold = load_gold(args.split)
        study_meta = {sid: (str(g["title"].iloc[0]), str(g["program"].iloc[0]), str(g["cycle"].iloc[0]))
                      for sid in gold["study_id"].unique()
                      for g in [gold[gold["study_id"] == sid]]}
        studies = sorted(study_meta.keys())
    if args.limit:
        studies = studies[:args.limit]
    if args.backend == "vllm":
        backend: Any = VLLMBackend(args.model, args.adapter, args.max_model_len, args.schema)
    elif args.backend == "oracle":
        backend = OracleBackend(args.split)
    else:
        backend = OpenAIBackend(args.model, args.base_url, args.api_key_env, args.schema,
                                args.min_interval, json.loads(args.extra_body) if args.extra_body else None)
    retriever = DemoRetriever(args.demos) if args.demos else None

    # --checkpoint: append each finished study to a JSONL file so a dropped session can resume.
    # Later lines win; studies whose calls errored are retried on resume.
    done: dict[str, dict] = {}
    if args.checkpoint and args.checkpoint.exists():
        for line in args.checkpoint.read_text(encoding="utf-8").splitlines():
            try:
                p = json.loads(line)
            except json.JSONDecodeError:
                continue  # a line cut off by the disconnect
            done[p["study_id"]] = p
        done = {sid: p for sid, p in done.items() if p.get("status") != "error"}
        print(f"resuming from {args.checkpoint}: {len(done)} studies already done")

    predictions = []

    def record(pred: dict) -> None:
        predictions.append(pred)
        if args.checkpoint:
            args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
            with args.checkpoint.open("a", encoding="utf-8") as f:
                f.write(json.dumps(pred, ensure_ascii=False) + "\n")
                f.flush()
                os.fsync(f.fileno())

    for n, sid in enumerate(studies, 1):
        if sid in done:
            predictions.append(done[sid])
            continue
        title, program, cycle = study_meta[sid]
        source = find_source_text(sid)
        if source is None:
            record({"study_id": sid, "pairs": [], "status": "missing_source"})
            continue
        t0 = time.time()
        if args.mode == "document":
            [(text, finish)] = backend.generate([(frozen_document_messages(program, cycle, sid, source), DOC_MAX_TOKENS, None)])
            pairs, status = parse_output(text)
            record({"study_id": sid, "pairs": pairs, "raw_output": text, "finish_reason": finish,
                                "status": "truncated" if finish == "length" else status,
                                "seconds": round(time.time() - t0, 1)})
            print(f"[{n}/{len(studies)}] {sid[:60]}: {len(pairs)} pairs, finish={finish}")
            continue

        chunks = chunk_text(clean_source(source))
        schema = json_schema(MAX_ITEMS_PER_CHUNK, CHUNK_FIELDS)
        batch = []
        for k, (_, passage) in enumerate(chunks):
            msgs = user_messages(render_chunk_prompt(title, program, cycle, k + 1, len(chunks), passage))
            if retriever:
                msgs = retriever.messages(passage, title) + msgs
            batch.append((msgs, CHUNK_MAX_TOKENS, schema))
        batch.append((user_messages(render_methods_prompt(title, program, cycle, methods_passages(chunks))),
                      METHODS_MAX_TOKENS, methods_json_schema()))
        outputs = backend.generate(batch)

        chunk_results, finishes = [], []
        for (text, finish), (_, passage) in zip(outputs[:-1], chunks):
            items, status = parse_output(text)
            chunk_results.append({"passage": passage, "items": items, "status": status, "finish": finish,
                                  "raw": text})
            finishes.append(finish)
        m_text, m_finish = outputs[-1]
        try:
            methods = json.loads(m_text)
        except json.JSONDecodeError:
            methods = {}
        pairs, stats = merge_study(chunk_results, methods)
        truncated = any(f == "length" for f in finishes) or any(c["status"] == "truncated" for c in chunk_results)
        failed = any(f == "error" for f in finishes + [m_finish])
        record({
            "study_id": sid, "pairs": pairs, "status": "error" if failed else ("truncated" if truncated else "ok"),
            "methods": methods, "merge_stats": stats, "n_chunks": len(chunks),
            "finish_reasons": {f: finishes.count(f) for f in set(finishes)},
            "chunks": [{"items": c["items"], "status": c["status"], "finish": c["finish"]} for c in chunk_results],
            "seconds": round(time.time() - t0, 1),
        })
        print(f"[{n}/{len(studies)}] {sid[:60]}: {len(chunks)} chunks, {stats['raw_items']} raw -> {len(pairs)} pairs")

    n_failed = sum(p.get("status") == "error" for p in predictions)
    out = args.out or OUT_DIR / f"{args.arm}_{args.split}.json"
    if n_failed:
        # never let a run with failed API calls pass as a result
        out = out.with_suffix(".incomplete.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "_arm": args.arm, "_split": args.split, "_mode": args.mode, "_backend": args.backend,
        "_model": args.model, "_adapter": args.adapter, "_demos": args.demos, "_schema": args.schema,
        "_extra_body": args.extra_body,
        "_decoding": {"temperature": 0.0, "frequency_penalty": FREQUENCY_PENALTY,
                      "chunk_max_tokens": CHUNK_MAX_TOKENS, "doc_max_tokens": DOC_MAX_TOKENS,
                      "max_items_per_chunk": MAX_ITEMS_PER_CHUNK, "grounding_min": GROUNDING_MIN},
        "_generated": datetime.now(timezone.utc).isoformat(),
        "predictions": predictions,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"saved {out}")
    if n_failed:
        raise SystemExit(f"{n_failed} studies had failed calls; saved as {out.name} and not usable as a result")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["chunked", "document"], default="chunked")
    ap.add_argument("--backend", choices=["vllm", "openai", "oracle"], default="vllm")
    ap.add_argument("--model", default="")
    ap.add_argument("--adapter")
    ap.add_argument("--base-url", default="https://api.together.xyz/v1")
    ap.add_argument("--api-key-env", default="TOGETHER_API_KEY")
    ap.add_argument("--split", default="val", choices=["train", "val", "test", "corpus"])
    ap.add_argument("--arm", required=True, help="label of this experimental arm")
    ap.add_argument("--demos", type=int, default=0)
    ap.add_argument("--min-interval", type=float, default=0.0, help="seconds between API calls (TPM limits)")
    ap.add_argument("--extra-body", help='provider-specific JSON, e.g. \'{"reasoning_effort": "low"}\'')
    ap.add_argument("--no-schema", dest="schema", action="store_false")
    ap.add_argument("--max-model-len", type=int, default=32768)
    ap.add_argument("--limit", type=int, default=0, help="first N studies only (smoke test)")
    ap.add_argument("--final-test-run", action="store_true")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--checkpoint", type=Path,
                    help="JSONL file (e.g. on Google Drive) written after every study; rerun with it to resume")
    args = ap.parse_args()
    if args.mode == "document" and args.demos:
        ap.error("--demos applies to chunked mode only")
    if args.backend == "oracle" and args.split == "test":
        ap.error("the oracle backend reads gold-derived targets and is never run on test")
    if args.backend != "oracle" and not args.model:
        ap.error("--model is required")
    check_test_lock(args.split, args.arm, args.final_test_run)
    run(args)


if __name__ == "__main__":
    main()
