#!/usr/bin/env python3
"""Log independently discovered forecast candidates against the locked corpus.

This supplements, but does not mutate, the 1,266-record source corpus. The
review is deliberately a candidate log, not a claim of exhaustive searching.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "analysis/02_prediction_taxonomy/study_classification.csv"
OUT = ROOT / "analysis/03_data_feasibility/audit/external_forecast_candidate_review_v2.csv"
MANIFEST = ROOT / "analysis/03_data_feasibility/audit/external_forecast_candidate_review_v2_manifest.json"

DOI = "10.3389/feduc.2026.1734988"
REVIEW = {
    "doi": DOI,
    "title": "Education during the COVID-19 pandemic: Does national culture predict school closures and changes in student performances?",
    "publication_year": 2026,
    "source_corpus_status": "not_present_in_1266_records",
    "target_program": "PISA",
    "target_cycles": "2022",
    "target_unit": "country-level PISA domain means (also percentiles and derived cross-domain factors)",
    "method": "country-specific linear and quadratic trend regressions; quadratic retained only when R-squared improvement over linear is significant",
    "training_information": "PISA cycles through 2018 only; forecasts 2022 from each country's historical time trend",
    "target_outcome_used_in_model_selection": "No indication in inspected Methods section; linear-vs-quadratic comparison is based on model R-squared and pre-target historical trajectory, then the 2022 forecast deviation is calculated against observed 2022 score.",
    "forecast_evaluation": "forecast deviations computed as observed PISA 2022 minus forecast; then analyzed as outcomes in cross-national regressions",
    "review_class": "qualifying_historical_country_level_future_cycle_forecast",
    "evidence_basis": "publisher full text inspected: Data (2000-2022), Trends and forecasts, analysis; exact cycle cutoff/method/evaluation stated",
    "source_url": "https://doi.org/10.3389/feduc.2026.1734988",
    "source_method_url": "https://www.frontiersin.org/journals/education/articles/10.3389/feduc.2026.1734988/full",
    "source_evidence_note": "Methods state 2022 forecasts use PISA waves 2000-2018; country-specific linear/quadratic trend regression; model choice by R-squared F-test; forecast deviations use observed-vs-forecast 2022. This qualifies as a historical held-out cycle forecast. It is not ML, and the paper was published in 2026 after the target results existed; classification concerns the temporal forecast design, not prospective publication timing.",
    "corpus_implication": "The 1,266-record corpus has at least one relevant false negative or scope omission relative to a broad claim about published PISA forecasting. Do not add it silently to locked counts; establish inclusion criteria and update corpus in a separately versioned release if eligible.",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    if OUT.exists() or MANIFEST.exists():
        raise SystemExit(f"Refusing to overwrite existing candidate audit: {OUT}")
    corpus = pd.read_csv(CORPUS, dtype=str).fillna("")
    if DOI in set(corpus["doi"].str.strip()):
        raise SystemExit(f"Candidate DOI {DOI} is already in the corpus; resolve duplicate/version before writing")
    pd.DataFrame([REVIEW]).to_csv(OUT, index=False)
    manifest = {
        "review_version": "v2",
        "corpus_file": str(CORPUS.relative_to(ROOT)),
        "corpus_sha256": sha256(CORPUS),
        "script_file": str(Path(__file__).resolve().relative_to(ROOT)),
        "script_sha256": sha256(Path(__file__).resolve()),
        "output_file": str(OUT.relative_to(ROOT)),
        "output_sha256": sha256(OUT),
        "candidate_count": 1,
        "corpus_match_count_for_candidate_doi": 0,
        "exhaustive_search_claim": False,
        "important_interpretation": "At least one external qualifying candidate exists. This demonstrates a false-negative/scope-coverage risk, but this single discovery does not establish how many additional studies are omitted.",
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(f"Wrote {OUT.relative_to(ROOT)}")
    print(f"Candidate DOI absent from 1,266-row classification corpus: {DOI}")


if __name__ == "__main__":
    main()
