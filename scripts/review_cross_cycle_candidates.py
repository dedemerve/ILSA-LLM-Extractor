#!/usr/bin/env python3
"""Create a source-aware review of the 28 heuristic cross-cycle records.

This is a second-pass review artifact. It never edits the source taxonomy or
any model/evaluation data. Output is versioned and refuses to overwrite.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "analysis/02_prediction_taxonomy/study_classification.csv"
OUT = ROOT / "analysis/03_data_feasibility/audit/cross_cycle_candidate_review_v2.csv"
MANIFEST = ROOT / "analysis/03_data_feasibility/audit/cross_cycle_candidate_review_v2_manifest.json"

# Manual dispositions follow the strict target definition: a country/system
# score for a later ILSA cycle, forecast using only information available from
# earlier cycles, with the later-cycle outcome held out through model selection.
REVIEW = {
    "10.1016/j.procs.2024.09.259": ("out_of_scope_non_empirical_review", "No ILSA microdata analysis or ILSA score target; narrative employment literature review.", "publisher abstract", "https://doi.org/10.1016/j.procs.2024.09.259"),
    "10.1016/j.econedurev.2025.102654": ("out_of_scope_non_ilsa_target", "STEP household survey and non-cognitive outcomes; not an ILSA score forecast.", "corpus extraction; dataset/outcome fields", "https://doi.org/10.1016/j.econedurev.2025.102654"),
    "10.1016/j.jmoneco.2025.103884": ("out_of_scope_non_forecasting_target", "PIAAC/readability evidence is descriptive/theoretical; no future ILSA country score target.", "corpus extraction; dataset/outcome fields", "https://doi.org/10.1016/j.jmoneco.2025.103884"),
    "10.1016/j.jebo.2021.12.027": ("out_of_scope_non_ilsa_target", "Original worker survey and retraining intention outcome; no ILSA assessment score forecast.", "corpus extraction; dataset/outcome fields", "https://doi.org/10.1016/j.jebo.2021.12.027"),
    "10.1016/j.ijer.2023.102274": ("out_of_scope_non_score_target", "PISA 2000/2018 is used for trends in student teaching-career expectations; target is not achievement score and no future-cycle score forecast is evaluated.", "corpus extraction; publisher/index record", "https://doi.org/10.1016/j.ijer.2023.102274"),
    "10.1016/j.lindif.2025.102729": ("out_of_scope_conceptual", "Conceptual curriculum synthesis; TIMSS is contextual evidence, not an analyzed forecast target.", "corpus extraction; dataset/outcome fields", "https://doi.org/10.1016/j.lindif.2025.102729"),
    "10.1016/j.ijedro.2025.100445": ("related_trend_analysis_not_forecast", "Models historical PISA equity indicators (R-squared/ESCS relationship), not country mean achievement. Historical trend slopes are not a held-out future-cycle score forecast.", "full text; author-hosted publisher PDF", "https://doi.org/10.1016/j.ijedro.2025.100445"),
    "10.1016/j.ijedro.2025.100489": ("out_of_scope_non_score_target", "Weighted regression examines pooled non-cognitive outcomes across PISA cycles; target is not future-cycle achievement score.", "corpus extraction; dataset/outcome fields", "https://doi.org/10.1016/j.ijedro.2025.100489"),
    "10.1016/j.caeai.2026.100566": ("out_of_scope_non_ilsa", "Uses Netherlands OML Grade 6/exit-test data, not an ILSA cycle or country-level ILSA mean. The model target described in the record is student digital-tool use.", "publisher abstract and corpus full-text extraction", "https://doi.org/10.1016/j.caeai.2026.100566"),
    "10.1016/j.stueduc.2024.101330": ("out_of_scope_descriptive_unsupervised", "TIMSS multi-cycle data are used for variance decomposition/correlation/Gaussian-mixture exploration of school inequality, not supervised forecasting of a future country mean.", "publisher abstract", "https://doi.org/10.1016/j.stueduc.2024.101330"),
    "10.1016/j.compedu.2025.105245": ("out_of_scope_review", "Systematic review of process-data publications; no original future-cycle country-score model.", "corpus extraction; dataset/outcome fields", "https://doi.org/10.1016/j.compedu.2025.105245"),
    "10.1016/j.stueduc.2025.101529": ("out_of_scope_linking_scaling", "IEA science studies are linked to create a comparable historical scale and illustrate trends; this is measurement/linking, not held-out forecasting of a future ILSA cycle.", "publisher full-text page", "https://doi.org/10.1016/j.stueduc.2025.101529"),
    "10.1111/bmsp.12290": ("out_of_scope_within_assessment_process", "PIAAC process logs are used to predict next-step response correctness within a single assessment; no future cycle or country-mean target.", "corpus extraction; dataset/outcome fields", "https://doi.org/10.1111/bmsp.12290"),
    "10.1186/s42409-022-00039-w": ("out_of_scope_measurement_method", "Discusses scaling-model analytical choices using one PISA cycle; not a future-cycle achievement forecast.", "corpus extraction; dataset/outcome fields", "https://doi.org/10.1186/s42409-022-00039-w"),
    "10.5281/zenodo.16778072": ("related_dataset_imputation_not_cycle_forecast", "Builds a harmonized annual score panel and imputes missing country-year cells using interpolation/LASSO. Missing-cell estimation and extrapolation are not evaluated as held-out future ILSA-cycle forecasts.", "local full-text extraction; Zenodo/publisher record", "https://doi.org/10.5281/zenodo.16778072"),
    "10.3390/data7110147": ("out_of_scope_non_ilsa_topic_model", "Indonesian physics-education conference-paper corpus is analyzed with topic modeling; no ILSA outcome.", "publisher abstract", "https://doi.org/10.3390/data7110147"),
    "10.1177/0004944120957477": ("out_of_scope_student_longitudinal_non_score", "Links Australian PISA student measures to later LSAY science-course enrolment; the later target is student course choice, not a future country ILSA achievement mean.", "local full-text extraction/corpus record", "https://doi.org/10.1177/0004944120957477"),
    "10.1007/s00521-023-08901-6": ("cross_cycle_student_transfer_with_aggregated_country_metrics", "Trains/tunes on PISA 2015 student records and applies the learned relationship to PISA 2018 student records; the paper then aggregates individual predictions into country means. This is meaningful cross-cycle transfer evidence, but the modeled unit is student-level and country means are derived summaries rather than direct forecasts of country-level score time series. The paper does not establish that target-cycle predictors would be available before the assessment, so it is not a verified pre-cycle country-mean forecast under this review definition.", "local full text inspected (Methods 2.2, 3.4, Conclusion)", "https://doi.org/10.1007/s00521-023-08901-6"),
    "10.1016/j.seps.2019.06.009": ("out_of_scope_single_cycle_school_target", "Uses PISA 2012 to estimate school-level DEA/DDF inefficiency; the outcome is school efficiency within one cycle, not a future country mean.", "local full-text extraction/corpus record", "https://doi.org/10.1016/j.seps.2019.06.009"),
    "10.1007/s40797-025-00354-1": ("out_of_scope_review_descriptive", "Review/secondary descriptive evidence; no fitted ILSA future-score forecasting model.", "corpus extraction; dataset/outcome fields", "https://doi.org/10.1007/s40797-025-00354-1"),
    "10.1037/edu0000999": ("out_of_scope_talis_video_target", "TALIS video computer-vision model predicts teacher nonverbal-immediacy/interaction constructs, not country-level ILSA achievement.", "corpus extraction; outcome and dataset fields", "https://doi.org/10.1037/edu0000999"),
    "10.1016/j.paid.2023.112110": ("related_derived_index_extrapolation", "Extrapolates a derived societal IQ/Flynn-effect indicator to 2100 from PISA/TIMSS/PIRLS history; it does not forecast future ILSA country mean scores or validate against a subsequently observed assessment cycle.", "publisher/index abstract and corpus extraction", "https://doi.org/10.1016/j.paid.2023.112110"),
    "10.1186/s40536-024-00197-9": ("out_of_scope_methodological_review", "Methodological discussion of causal inference in ILSA; no empirical forecast model.", "corpus extraction; dataset/outcome fields", "https://doi.org/10.1186/s40536-024-00197-9"),
    "10.1186/s40536-025-00262-x": ("retrospective_target_contaminated_qc", "TIMSS 2011/2015/2019 country mean scores are predicted for quality-screening residuals, but the article states its NN was initially trained jointly using observed target-cycle student scores (e.g. 2011 Grade 4 and 2015 Grade 8); it also explicitly notes no leave-one-out validation. The target is QC triage, and target outcomes enter fitting, so this is not a temporally held-out future-cycle forecast.", "local full text inspected (Methods and limitations)", "https://doi.org/10.1186/s40536-025-00262-x"),
    "10.1186/s40536-025-00264-9": ("out_of_scope_methodological_pv", "NAEP simulation and PV/context-variable dimensionality-reduction method; not ILSA country-cycle forecasting.", "corpus extraction; dataset/outcome fields", "https://doi.org/10.1186/s40536-025-00264-9"),
    "10.3390/su13084285": ("out_of_scope_non_ilsa_single_study", "Local Mexican elementary-school reading study, not PISA or another ILSA; single-period classification target.", "corpus extraction; dataset/outcome fields", "https://doi.org/10.3390/su13084285"),
    "NO_DOI_REVIEW_ITEM_DIFFICULTY": ("out_of_scope_review_item_level", "Review of item-difficulty modeling literature; no original future-cycle country-score forecast.", "corpus extraction; DOI unavailable", ""),
    "10.1007/s10639-023-12389-x": ("out_of_scope_within_assessment_process", "PSLC tutoring-log model predicts next-step response correctness within a single learning process; not ILSA future-cycle or country-level score forecasting.", "corpus extraction; dataset/outcome fields", "https://doi.org/10.1007/s10639-023-12389-x"),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    if OUT.exists() or MANIFEST.exists():
        raise SystemExit(f"Refusing to overwrite existing review artifact: {OUT} or {MANIFEST}")
    source = pd.read_csv(SOURCE, dtype=str).fillna("")
    flagged = source[source["temporal_forecasting_status"].eq("cross_cycle_prediction_not_fully_verified")].copy()
    if len(flagged) != 28:
        raise SystemExit(f"Expected 28 flagged records, found {len(flagged)}; re-audit mappings before proceeding")

    rows = []
    for _, r in flagged.iterrows():
        doi = r["doi"].strip()
        key = doi if doi and doi.lower() != "nan" else "NO_DOI_REVIEW_ITEM_DIFFICULTY"
        if key not in REVIEW:
            raise SystemExit(f"No manual disposition for {key}: {r['title']}")
        decision, rationale, evidence, source_url = REVIEW[key]
        rows.append({
            "file_name": r["file_name"],
            "doi": doi,
            "title": r["title"],
            "publication_year": r["year"],
            "original_task_code": r["primary_task_code"],
            "original_task_label": r["primary_task_label"],
            "original_target_level": r["prediction_target_level"],
            "original_temporal_status": r["temporal_forecasting_status"],
            "datasets_as_coded": r["all_datasets_used"],
            "target_variables_as_coded": r["all_target_variables"],
            "review_disposition": decision,
            "qualifies_as_genuine_country_level_future_cycle_score_forecast": "no_confirmed",
            "review_rationale": rationale,
            "evidence_basis": evidence,
            "source_url": source_url,
        })
    out = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    manifest = {
        "review_version": "v2",
        "source_file": str(SOURCE.relative_to(ROOT)),
        "source_sha256": sha256(SOURCE),
        "script_file": str(Path(__file__).resolve().relative_to(ROOT)),
        "script_sha256": sha256(Path(__file__).resolve()),
        "output_file": str(OUT.relative_to(ROOT)),
        "output_sha256": sha256(OUT),
        "n_flagged_reviewed": len(out),
        "n_confirmed_genuine_country_level_future_cycle_forecasts": 0,
        "n_cross_cycle_transfer_candidates_not_meeting_strict_definition": int(out["review_disposition"].str.startswith(("cross_cycle", "retrospective")).sum()),
        "limitations": [
            "The 28-row set is a heuristic candidate bucket, not a random sample and not a sensitivity check for false negatives.",
            "Evidence basis is recorded row by row; rows based on corpus extraction/abstract are lower-strength than locally inspected full text.",
            "The no-confirmed count applies only to the strict country-level future ILSA-cycle mean-score definition used here; it does not mean no multi-cycle or student-level forecasts exist.",
            "The PISA 2015-to-2018 study is a cross-cycle transfer exercise with a target-cycle covariate availability ambiguity; it is not counted as a verified pre-cycle country forecast.",
        ],
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(f"Wrote {OUT.relative_to(ROOT)} ({len(out)} rows)")
    print(f"Wrote {MANIFEST.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
