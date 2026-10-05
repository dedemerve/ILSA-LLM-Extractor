#!/usr/bin/env python3
"""
Deterministic effect/evidence rows from ALL on-disk JSON (1,756 corpus).

Adds OECD + IEA + full Scopus/WoS/survey coverage without new LLM calls.
Merges into outputs/stage5/effect_sizes.csv and remaps predictor_canonical.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import pandas as pd

try:
    from scripts.canonical_predictor_map import canonical_map, normalize_canonical
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.canonical_predictor_map import canonical_map, normalize_canonical

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = PROJECT_ROOT / "outputs"
OUT_CSV = PROJECT_ROOT / "outputs" / "stage5" / "effect_sizes.csv"

JSON_ROOTS = [
    OUTPUTS / "OECD",
    OUTPUTS / "IEA",
    OUTPUTS / "Scopus",
    OUTPUTS / "Web of Science",
    OUTPUTS / "ilsa_survey_articles" / "json",
]

PROGRAM_KEYWORDS = {
    "PISA": ["pisa"],
    "TIMSS_G4": ["timss grade 4", "timss g4", "timss fourth"],
    "TIMSS": ["timss"],
    "PIRLS": ["pirls"],
    "TALIS": ["talis"],
    "ICILS": ["icils"],
    "ICCS": ["iccs"],
    "PIAAC": ["piaac"],
}


def _detect_program(text: str) -> str:
    tl = (text or "").lower()
    for prog in ["TIMSS_G4", "PISA", "TIMSS", "PIRLS", "TALIS", "ICILS", "ICCS", "PIAAC"]:
        if any(kw in tl for kw in PROGRAM_KEYWORDS[prog]):
            return prog
    return ""


def _detect_domain(text: str) -> str:
    tl = (text or "").lower()
    for dom in ["mathematics", "reading", "science", "literacy", "numeracy", "civic"]:
        if dom in tl:
            return dom
    return "other"


def _parse_effect_from_metrics(metrics) -> tuple[str | None, float | None]:
    if not metrics:
        return None, None
    if isinstance(metrics, dict):
        for k, v in metrics.items():
            kl = str(k).lower()
            if v is None:
                continue
            try:
                val = float(v)
            except (TypeError, ValueError):
                continue
            if "r2" in kl or "r²" in kl:
                return "R2", val / 100.0 if val > 1 else val
            if kl in ("r", "correlation", "pearson"):
                return "r_correlation", val
            if "beta" in kl or "coef" in kl:
                return "standardized_beta", val
            if "auc" in kl:
                return "AUC", val
            if "accuracy" in kl:
                return "accuracy", val
    return None, None


def _paper_key(meta: dict, path: Path) -> tuple[str, str, int]:
    title = str(meta.get("title") or path.stem)
    doi = str(meta.get("doi") or "")[:40]
    pid = doi or re.sub(r"\W+", "_", title.lower())[:40]
    try:
        year = int(meta.get("year") or 0)
    except (TypeError, ValueError):
        year = 0
    return pid, title, year


def iter_corpus_json() -> list[Path]:
    seen: set[str] = set()
    files: list[Path] = []
    for root in JSON_ROOTS:
        if not root.exists():
            continue
        pattern = "*.json" if root.name == "json" else "**/json/*.json"
        for fpath in sorted(root.glob(pattern) if "**" in pattern else root.glob("*.json")):
            if "(1)" in fpath.name:
                continue
            try:
                meta = json.loads(fpath.read_text(errors="ignore")).get("metadata", {})
                key = (meta.get("doi") or meta.get("title") or fpath.stem)[:120]
            except Exception:
                key = fpath.stem
            if key in seen:
                continue
            seen.add(key)
            files.append(fpath)
    return files


def rows_from_json(fpath: Path) -> list[dict]:
    try:
        doc = json.loads(fpath.read_text(encoding="utf-8"))
    except Exception:
        return []
    meta = doc.get("metadata", {})
    data = doc.get("data", {})
    pid, title, year = _paper_key(meta, fpath)

    rel = str(fpath.relative_to(OUTPUTS))
    if rel.startswith("OECD"):
        corpus_source = "OECD"
    elif rel.startswith("IEA"):
        corpus_source = "IEA"
    elif rel.startswith("Scopus"):
        corpus_source = "Scopus"
    elif rel.startswith("Web of Science"):
        corpus_source = "WoS"
    else:
        corpus_source = "survey"

    rows: list[dict] = []
    base = {
        "paper_id": pid,
        "title": title[:200],
        "year": year,
        "corpus_source": corpus_source,
        "extraction_method": "corpus_deterministic",
    }

    # Confounders
    for conf in data.get("confounders_identified") or []:
        if not isinstance(conf, dict):
            continue
        vname = conf.get("variable_name") or conf.get("variable_code") or ""
        vcode = conf.get("variable_code") or ""
        cat = conf.get("category") or ""
        canon = canonical_map(vname, variable_code=vcode, category=cat)
        if canon == "OTHER":
            continue
        ctx = f"{title} {vname} {cat}"
        rows.append({
            **base,
            "program": _detect_program(ctx),
            "domain": _detect_domain(ctx),
            "predictor_raw": str(vname)[:120],
            "predictor_canonical": normalize_canonical(canon),
            "predictor_category": cat,
            "effect_type": "associative",
            "effect_value": None,
            "direction": "positive",
            "outcome": "achievement",
            "level": "student",
            "n_countries": None,
            "n_students": None,
            "pv_correct": True,
            "weight_correct": True,
            "quality_flag": "medium",
        })

    # Main findings
    for mf in data.get("main_findings") or []:
        if not isinstance(mf, dict):
            continue
        dataset = mf.get("dataset_used") or ""
        outcome = mf.get("target_variable") or "achievement"
        program = _detect_program(f"{dataset} {title}") or _detect_program(title)
        domain = _detect_domain(f"{outcome} {dataset}")
        eff_type, eff_val = _parse_effect_from_metrics(mf.get("performance_metrics"))
        conclusion = mf.get("standardized_conclusion") or ""
        direction = "positive"
        if any(w in conclusion.lower() for w in ("negative", "inverse", "lower", "decrease")):
            direction = "negative"

        preds = mf.get("top_predictors") or []
        if not isinstance(preds, list):
            preds = []
        for pred in preds:
            pred_s = str(pred)
            canon = canonical_map(pred_s)
            if canon == "OTHER":
                continue
            rows.append({
                **base,
                "program": program,
                "domain": domain,
                "predictor_raw": pred_s[:120],
                "predictor_canonical": normalize_canonical(canon),
                "predictor_category": "",
                "effect_type": eff_type or "associative",
                "effect_value": eff_val,
                "direction": direction,
                "outcome": str(outcome)[:120],
                "level": "country" if "country" in conclusion.lower() else "student",
                "n_countries": None,
                "n_students": None,
                "pv_correct": True,
                "weight_correct": True,
                "quality_flag": "high" if eff_val is not None else "medium",
            })

    return rows


def rematch_existing(df: pd.DataFrame) -> pd.DataFrame:
    def _remap(row):
        return normalize_canonical(
            canonical_map(
                str(row.get("predictor_raw") or ""),
                category=str(row.get("predictor_category") or "") or None,
            )
        )

    df = df.copy()
    df["predictor_canonical"] = df.apply(_remap, axis=1)
    return df


def main() -> None:
    files = iter_corpus_json()
    log.info("Corpus JSON files: %d", len(files))

    new_rows: list[dict] = []
    for fpath in files:
        new_rows.extend(rows_from_json(fpath))
    log.info("Deterministic rows extracted: %d", len(new_rows))
    new_df = pd.DataFrame(new_rows)

    if OUT_CSV.exists():
        old = pd.read_csv(OUT_CSV)
        if "corpus_source" not in old.columns:
            old["corpus_source"] = "legacy_llm"
        if "extraction_method" not in old.columns:
            old["extraction_method"] = "llm"
        # Drop prior deterministic pass so re-runs are idempotent
        old = old[old["extraction_method"] != "corpus_deterministic"].copy()
        old = rematch_existing(old)
    else:
        old = pd.DataFrame()

    if not new_df.empty:
        key_cols = ["paper_id", "predictor_raw", "program", "outcome"]
        combined = pd.concat([old, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=key_cols, keep="first")
    else:
        combined = old

    combined = rematch_existing(combined)
    combined.to_csv(OUT_CSV, index=False)

    other_rate = (combined["predictor_canonical"] == "OTHER").mean()
    log.info(
        "Saved %s — %d rows, %d papers, OTHER=%.1f%%",
        OUT_CSV.name,
        len(combined),
        combined["paper_id"].nunique(),
        100 * other_rate,
    )
    print(combined["predictor_canonical"].value_counts().head(15).to_string())
    print("\nCorpus source counts:")
    if "corpus_source" in combined.columns:
        print(combined["corpus_source"].value_counts().to_string())


if __name__ == "__main__":
    main()
