#!/usr/bin/env python3
"""
Agreement statistics for canonical_construct labels (v2).

Reference labels are the PRE-adjudication gold labels read from git
(commit ad7b100^), because v1.4.1 at ad7b100 changed five labels to the
human codes (G0113, G0172, G0182, G0232, G0233). Using adjudicated labels as
the reference would count those as agreements by construction.

Note: the gold labels were verified by the same person who produced the
re-codes, so this is intra-rater (re-coding) agreement, not independent
inter-rater agreement.

Outputs (refuses to overwrite): outputs/kb_integrity_v1/
  kappa_summary.json, per_construct_agreement.csv, confusion_matrix.csv
"""
from __future__ import annotations

import io
import json
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
GOLD_PATH = "outputs/finetuning/data/gold_standard_full_corpus_v1.4.1.csv"
REF_COMMIT = "ad7b100^"
HUMAN = ROOT / "outputs" / "finetuning" / "audit" / "human_validation_sample_v1.csv"
OUT = ROOT / "outputs" / "kb_integrity_v1"
N_BOOT = 5000
SEED = 20261027


def cohen_kappa(a: list[str], b: list[str]) -> tuple[float, float]:
    n = len(a)
    p_o = sum(x == y for x, y in zip(a, b)) / n
    c1, c2 = Counter(a), Counter(b)
    p_e = sum(c1[c] / n * c2[c] / n for c in set(a) | set(b))
    return ((p_o - p_e) / (1 - p_e) if p_e < 1 else 1.0), p_o


def git_csv(rev: str, path: str) -> pd.DataFrame:
    text = subprocess.run(["git", "show", f"{rev}:{path}"], cwd=ROOT,
                          capture_output=True, text=True, check=True).stdout
    return pd.read_csv(io.StringIO(text))


def main() -> None:
    if OUT.exists() and any((OUT / f).exists() for f in
                            ("kappa_summary.json", "per_construct_agreement.csv", "confusion_matrix.csv")):
        raise FileExistsError(f"Refusing to overwrite outputs in {OUT}")
    OUT.mkdir(parents=True, exist_ok=True)

    ref_sha = subprocess.run(["git", "rev-parse", REF_COMMIT], cwd=ROOT,
                             capture_output=True, text=True, check=True).stdout.strip()
    pre = git_csv(REF_COMMIT, GOLD_PATH)[["record_id", "study_id", "canonical_construct"]]
    post = pd.read_csv(ROOT / GOLD_PATH)[["record_id", "canonical_construct"]]
    human = pd.read_csv(HUMAN)[["record_id", "human_canonical_construct"]].dropna()

    df = human.merge(pre, on="record_id").merge(post, on="record_id", suffixes=("_pre", "_post"))
    for c in ("human_canonical_construct", "canonical_construct_pre", "canonical_construct_post"):
        df[c] = df[c].astype(str).str.strip()

    k_pre, a_pre = cohen_kappa(df["canonical_construct_pre"].tolist(), df["human_canonical_construct"].tolist())
    k_post, a_post = cohen_kappa(df["canonical_construct_post"].tolist(), df["human_canonical_construct"].tolist())

    # Cluster bootstrap over studies
    rng = np.random.default_rng(SEED)
    studies = df["study_id"].unique()
    groups = {s: g for s, g in df.groupby("study_id")}
    boots = []
    for _ in range(N_BOOT):
        g = pd.concat([groups[s] for s in rng.choice(studies, len(studies), replace=True)])
        boots.append(cohen_kappa(g["canonical_construct_pre"].tolist(), g["human_canonical_construct"].tolist())[0])
    lo, hi = np.percentile(boots, [2.5, 97.5])

    per = (df.assign(agree=df["canonical_construct_pre"] == df["human_canonical_construct"])
             .groupby("human_canonical_construct")
             .agg(n=("agree", "size"), n_agree=("agree", "sum")))
    per["agreement"] = (per["n_agree"] / per["n"]).round(4)
    per = per.sort_values("n", ascending=False)
    per.to_csv(OUT / "per_construct_agreement.csv")

    conf = pd.crosstab(df["canonical_construct_pre"], df["human_canonical_construct"],
                       rownames=["llm_pre_adjudication"], colnames=["human"])
    conf.to_csv(OUT / "confusion_matrix.csv")

    summary = {
        "n_records": int(len(df)),
        "n_studies": int(len(studies)),
        "n_categories_human": int(df["human_canonical_construct"].nunique()),
        "reference_labels": f"{GOLD_PATH} at {REF_COMMIT} ({ref_sha})",
        "kappa_pre_adjudication": round(k_pre, 4),
        "agreement_pre_adjudication": round(a_pre, 4),
        "kappa_pre_adjudication_study_bootstrap_95ci": [round(lo, 4), round(hi, 4)],
        "bootstrap_replicates": N_BOOT,
        "kappa_post_adjudication_NOT_VALID_AS_RELIABILITY": round(k_post, 4),
        "agreement_post_adjudication": round(a_post, 4),
        "design_note": "Single coder (Merve_Dede) verified gold labels and produced re-codes: intra-rater agreement, not inter-rater.",
        "fields_validated": ["canonical_construct"],
        "fields_not_validated": ["direction", "effect_type", "effect_value", "analysis_level"],
    }
    (OUT / "kappa_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("\nLowest-agreement constructs (n>=3):")
    print(per[per["n"] >= 3].sort_values("agreement").head(8).to_string())


if __name__ == "__main__":
    main()
