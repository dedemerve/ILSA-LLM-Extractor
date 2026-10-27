#!/usr/bin/env python3
"""
Draw a human-validation sample from the evidence matrix that feeds the KB
weights (predictor_weights_v3.csv), not from the gold corpus.

Unit: one (study, canonical_construct) pair, because the weight formula counts
studies per construct and their programs/cycles/countries. Strata: every
construct with w > 0; up to PER_CONSTRUCT pairs per construct (all pairs for
smaller constructs). The coder fills the human_* columns blind to the llm_*
columns where possible.

Output (refuses to overwrite):
  outputs/kb_integrity_v1/evidence_matrix_validation_sample_v1.csv
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "outputs" / "stage5" / "evidence_matrix.csv"
WEIGHTS = ROOT / "outputs" / "stage5" / "predictor_weights_v3.csv"
OUT = ROOT / "outputs" / "kb_integrity_v1" / "evidence_matrix_validation_sample_v1.csv"
PER_CONSTRUCT = 6
SEED = 20261027

HUMAN_COLUMNS = [
    "human_construct_correct",   # yes / no / unclear
    "human_canonical_construct", # fill when not correct
    "human_direction",           # + / - / mixed / null
    "human_effect_type",         # predictive / associative / cross_cycle / cross_country
    "human_analysis_level",      # student / school / country
    "human_programs_cycles_ok",  # yes / no
    "human_notes",
]


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUT}")
    ev = pd.read_csv(EVIDENCE)
    w = pd.read_csv(WEIGHTS)
    used = set(w.loc[w["w"] > 0, "canonical_construct"])
    ev = ev[ev["canonical_construct"].isin(used)]

    # Collapse to (study, construct) pairs; keep the first evidence row as the example.
    pairs = (ev.sort_values(["study_id", "rank_in_study"])
               .groupby(["study_id", "canonical_construct"], as_index=False)
               .agg(title=("title", "first"), year=("year", "first"), doi=("doi", "first"),
                    programs=("programs", "first"), cycles=("cycles", "first"),
                    predictor_raw=("predictor_raw", lambda s: " | ".join(s.astype(str).unique()[:3])),
                    outcome_raw=("outcome_raw", "first"),
                    llm_direction=("direction", "first"), llm_effect_type=("effect_type", "first"),
                    llm_analysis_level=("analysis_level", "first"), n_rows=("study_id", "size")))

    sample = (pairs.groupby("canonical_construct", group_keys=False)
                   .apply(lambda g: g.sample(min(len(g), PER_CONSTRUCT), random_state=SEED)))
    sample = sample.rename(columns={"canonical_construct": "llm_canonical_construct"})
    sample = sample.sample(frac=1, random_state=SEED).reset_index(drop=True)  # shuffle for blind coding
    sample.insert(0, "sample_id", [f"EV{i + 1:03d}" for i in range(len(sample))])
    for c in HUMAN_COLUMNS:
        sample[c] = ""

    OUT.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(OUT, index=False)
    print(f"{len(sample)} pairs, {sample['study_id'].nunique()} studies")
    print(sample["llm_canonical_construct"].value_counts().to_string())
    print(f"Saved: {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
