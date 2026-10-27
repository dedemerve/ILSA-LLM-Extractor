#!/usr/bin/env python3
"""
Agreement between LLM-extracted evidence-matrix labels and blind expert coding.

Inputs:
  outputs/kb_integrity_v1/evidence_matrix_validation_sample_v1.csv        (llm_* labels)
  outputs/kb_integrity_v1/evidence_matrix_validation_sample_v1_blind.csv  (human_* codes;
      coded blind by Merve_Dede from the PDFs, single coder)

Outputs (refuses to overwrite), outputs/kb_integrity_v1/:
  evidence_validation_agreement_v1.json   kappa / agreement per field, study-cluster bootstrap CI
  evidence_validation_per_construct_v1.csv  precision of LLM construct labels
  evidence_validation_confusion_v1.csv
  predictor_weights_v3_sample_corrected.csv  weight sensitivity: sampled pairs relabelled
      to the expert construct (UNMATCHED pairs removed), same formula as v3
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import build_literature_priority as blp  # noqa: E402

DIR = ROOT / "outputs" / "kb_integrity_v1"
LLM = DIR / "evidence_matrix_validation_sample_v1.csv"
HUMAN = DIR / "evidence_matrix_validation_sample_v1_blind.csv"
WEIGHTS_V3 = ROOT / "outputs" / "stage5" / "predictor_weights_v3.csv"
OUTS = {
    "summary": DIR / "evidence_validation_agreement_v1.json",
    "per_construct": DIR / "evidence_validation_per_construct_v1.csv",
    "confusion": DIR / "evidence_validation_confusion_v1.csv",
    "weights": DIR / "predictor_weights_v3_sample_corrected.csv",
}
FIELDS = ["canonical_construct", "direction", "effect_type", "analysis_level"]
N_BOOT = 5000
SEED = 20261027


def kappa(a: list[str], b: list[str]) -> tuple[float, float]:
    n = len(a)
    p_o = sum(x == y for x, y in zip(a, b)) / n
    c1, c2 = Counter(a), Counter(b)
    p_e = sum(c1[c] / n * c2[c] / n for c in set(a) | set(b))
    return ((p_o - p_e) / (1 - p_e) if p_e < 1 else 1.0), p_o


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    for p in OUTS.values():
        if p.exists():
            raise FileExistsError(f"Refusing to overwrite {p}")

    llm = pd.read_csv(LLM)
    hum = pd.read_csv(HUMAN)
    d = llm[["sample_id", "study_id"] + [f"llm_{f}" for f in FIELDS]].merge(
        hum[["sample_id", "human_programs_cycles_ok"] + [f"human_{f}" for f in FIELDS]], on="sample_id")
    if len(d) != 51 or d[[f"human_{f}" for f in FIELDS]].isna().any().any():
        raise ValueError("expected 51 fully coded records")
    # LLM left most directions empty; treat empty as its own category.
    d["llm_direction"] = d["llm_direction"].fillna("missing")

    rng = np.random.default_rng(SEED)
    studies = d["study_id"].unique()
    groups = {s: g for s, g in d.groupby("study_id")}
    fields = {}
    for f in FIELDS:
        a, b = d[f"llm_{f}"].astype(str).tolist(), d[f"human_{f}"].astype(str).tolist()
        k, po = kappa(a, b)
        boots = []
        for _ in range(N_BOOT):
            g = pd.concat([groups[s] for s in rng.choice(studies, len(studies), replace=True)])
            boots.append(kappa(g[f"llm_{f}"].astype(str).tolist(), g[f"human_{f}"].astype(str).tolist())[0])
        lo, hi = np.nanpercentile(boots, [2.5, 97.5])
        fields[f] = {"kappa": round(k, 4), "agreement": round(po, 4),
                     "kappa_study_bootstrap_95ci": [round(lo, 4), round(hi, 4)],
                     "llm_distribution": dict(Counter(a)), "human_distribution": dict(Counter(b))}

    per = (d.assign(correct=d["llm_canonical_construct"] == d["human_canonical_construct"])
             .groupby("llm_canonical_construct")["correct"].agg(n="size", n_correct="sum"))
    per["precision"] = (per["n_correct"] / per["n"]).round(4)
    per = per.sort_values(["precision", "n"])
    per.to_csv(OUTS["per_construct"])
    pd.crosstab(d["llm_canonical_construct"], d["human_canonical_construct"],
                rownames=["llm"], colnames=["expert"]).to_csv(OUTS["confusion"])

    # Weight sensitivity: relabel sampled (study, construct) pairs to the expert construct.
    ev = blp.load_evidence()
    relabel = {(r.study_id, r.llm_canonical_construct): r.human_canonical_construct for r in d.itertuples()}
    key = list(zip(ev["study_id"], ev["canonical_construct"]))
    new = [relabel.get(k, k[1]) for k in key]
    n_changed = int(sum(a != b for a, b in zip(new, ev["canonical_construct"])))
    ev = ev.assign(canonical_construct=new)
    ev = ev[ev["canonical_construct"] != "UNMATCHED"]
    w_new = blp.compute_weights(blp.compute_provenance_audit(ev))[["canonical_construct", "n_studies", "W_j_forecast"]]
    w_old = pd.read_csv(WEIGHTS_V3)[["canonical_construct", "n_studies", "w"]]
    comp = w_old.merge(w_new.rename(columns={"n_studies": "n_studies_corrected", "W_j_forecast": "w_corrected"}),
                       on="canonical_construct", how="outer").sort_values("w", ascending=False)
    comp.to_csv(OUTS["weights"], index=False, float_format="%.4f")

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "n_records": int(len(d)), "n_studies": int(len(studies)),
        "coder": "Merve_Dede (single expert coder, blind to LLM labels, coded from PDFs; self-reported)",
        "inputs": {str(p.relative_to(ROOT)): sha256(p) for p in (LLM, HUMAN, WEIGHTS_V3)},
        "fields": fields,
        "programs_cycles_ok": dict(Counter(d["human_programs_cycles_ok"])),
        "evidence_rows_relabelled_for_weight_sensitivity": n_changed,
        "bootstrap": {"unit": "study", "replicates": N_BOOT, "seed": SEED},
        "design_note": "LLM-vs-expert agreement with one human coder; not inter-rater reliability between humans.",
    }
    OUTS["summary"].write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({f: {k: v for k, v in x.items() if "distribution" not in k} for f, x in fields.items()}, indent=1))
    print(per.to_string())
    print(comp.to_string(index=False))


if __name__ == "__main__":
    main()
