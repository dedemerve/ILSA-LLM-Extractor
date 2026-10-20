#!/usr/bin/env python3
"""Build reproducible ILSA algorithm-frequency tables from the coded corpus."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "analysis/02_prediction_taxonomy/study_classification.csv"
OUT_CSV = ROOT / "analysis/03_data_feasibility/audit/ilsa_ml_method_frequency_v1.csv"
OUT_MD = ROOT / "analysis/03_data_feasibility/phase3_11_corpus_method_frequency.md"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_all_method_token(token: str) -> str:
    normalized = " ".join(token.casefold().strip().split())
    if normalized.startswith("random forest"):
        return "Random Forest"
    if normalized in {"xgboost", "xgb", "xgbregressor", "extreme gradient boosting"}:
        return "XGBoost"
    return token.strip()


def main() -> None:
    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)
    if OUT_MD.exists():
        raise FileExistsError("Refusing to overwrite a prior corpus-frequency report")
    data = pd.read_csv(SOURCE, dtype={"file_name": str})
    if len(data) != 1266:
        raise ValueError(f"Expected 1,266 study records; found {len(data)}")
    if data["file_name"].nunique(dropna=False) != len(data):
        raise ValueError("Study file_name key is not unique")

    scopes = {
        "all_records_with_primary_method": data["ml_primary"].notna(),
        "predictive_studies_with_primary_method": data["research_design_type"].eq("predictive") & data["ml_primary"].notna(),
        "country_system_studies_with_primary_method": data["prediction_target_level"].eq("country_system") & data["ml_primary"].notna(),
        "country_level_prediction_task_D": data["primary_task_code"].eq("D") & data["ml_primary"].notna(),
    }
    rows: list[dict] = []
    for scope, mask in scopes.items():
        denominator = int(mask.sum())
        counts = data.loc[mask, "ml_primary"].value_counts(dropna=True)
        for method, count in counts.items():
            rows.append(
                {
                    "field": "ml_primary",
                    "scope": scope,
                    "scope_coded_n": denominator,
                    "method": method,
                    "study_count": int(count),
                    "share_of_coded_scope": float(count / denominator),
                    "share_of_all_1266_records": float(count / len(data)),
                    "counting_rule": "One coded primary algorithm per study record; raw label preserved.",
                }
            )

    all_tokens = data["ml_all_techniques"].fillna("").map(
        lambda value: {canonical_all_method_token(token) for token in value.split(",") if token.strip()}
    )
    technique_scopes = {
        "all_records_with_any_technique": pd.Series(True, index=data.index),
        "predictive_studies": data["research_design_type"].eq("predictive"),
        "country_system_studies": data["prediction_target_level"].eq("country_system"),
        "country_level_prediction_task_D": data["primary_task_code"].eq("D"),
    }
    for scope, mask in technique_scopes.items():
        coded = mask & data["ml_all_techniques"].notna()
        denominator = int(coded.sum())
        counts: dict[str, int] = {}
        for idx in data.index[coded]:
            for method in all_tokens.loc[idx]:
                counts[method] = counts.get(method, 0) + 1
        for method, count in sorted(counts.items(), key=lambda item: (-item[1], item[0].casefold())):
            rows.append(
                {
                    "field": "ml_all_techniques",
                    "scope": scope,
                    "scope_coded_n": denominator,
                    "method": method,
                    "study_count": int(count),
                    "share_of_coded_scope": float(count / denominator) if denominator else 0.0,
                    "share_of_all_1266_records": float(count / len(data)),
                    "counting_rule": "Each normalized technique is counted at most once per study; comma-delimited tokens; RF and XGBoost aliases normalized conservatively.",
                }
            )

    table = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    primary_counts = data.loc[data["research_design_type"].eq("predictive"), "ml_primary"].value_counts()
    exact_rf = int(primary_counts.get("Random Forest", 0))
    exact_xgb = int(primary_counts.get("XGBoost", 0))
    predictive_n = int(data["research_design_type"].eq("predictive").sum())
    coded_predictive_n = int((data["research_design_type"].eq("predictive") & data["ml_primary"].notna()).sum())
    country = data.loc[data["primary_task_code"].eq("D") & data["ml_primary"].notna(), "ml_primary"].value_counts()
    status = data["temporal_forecasting_status"].value_counts(dropna=False)
    status_table = "| Temporal-forecasting status | Study count |\n|---|---:|\n" + "\n".join(
        f"| {str(label).replace('|', '\\|')} | {int(count)} |" for label, count in status.items()
    )
    report = f"""# Phase 3.11 — ILSA Algorithm Frequency Audit

**Source:** `analysis/02_prediction_taxonomy/study_classification.csv`  
**Records:** {len(data)} unique study records  
**Source SHA-256:** `{sha256(SOURCE)}`  
**Output table:** `analysis/03_data_feasibility/audit/ilsa_ml_method_frequency_v1.csv`

## Primary algorithm field

`ml_primary` contains a coded primary method for 167/1,266 records. Among 166 studies classified as predictive, 147 have a coded primary method. In this predictive, algorithm-coded subset the exact primary labels are:

| Primary method | Studies | Share of 147 coded predictive studies | Share of all 166 predictive studies |
|---|---:|---:|---:|
| Random Forest | {exact_rf} | {exact_rf/coded_predictive_n:.1%} | {exact_rf/predictive_n:.1%} |
| XGBoost | {exact_xgb} | {exact_xgb/coded_predictive_n:.1%} | {exact_xgb/predictive_n:.1%} |
| Neural Network | {int(primary_counts.get('Neural Network',0))} | {int(primary_counts.get('Neural Network',0))/coded_predictive_n:.1%} | {int(primary_counts.get('Neural Network',0))/predictive_n:.1%} |
| SVM | {int(primary_counts.get('SVM',0))} | {int(primary_counts.get('SVM',0))/coded_predictive_n:.1%} | {int(primary_counts.get('SVM',0))/predictive_n:.1%} |

RF and XGBoost are the two most frequent exact `ml_primary` labels among predictive records. Counts over all records with a primary algorithm are 44 RF and 18 XGBoost (out of 167 coded records). These are literature-motivated general predictive baselines, not evidence that they dominate genuine future-cycle ILSA forecasting.

## Country-level scope check

For primary task code D (country-level prediction), the algorithm-coded subset contains {int(data.primary_task_code.eq('D').sum())} records and its leading primary methods are: SVM {int(country.get('SVM',0))}, Random Forest {int(country.get('Random Forest',0))}, and XGBoost {int(country.get('XGBoost',0))}. In that narrower scope, RF and XGBoost are not the two most frequent methods. The corpus audit has identified zero confirmed genuine future-cycle country-level ILSA forecasts, with 28 cross-cycle studies unresolved; therefore no algorithm frequency ranking can be claimed specifically for genuine ILSA forecasting.

## Any-method field and coding limits

The companion `ml_all_techniques` table counts each technique at most once per study and normalizes only transparent Random Forest/XGBoost label variants. It is an any-method count, not the primary-method count; multi-method studies contribute to several method totals. The exact raw primary field remains unaltered. The table preserves the two fields as separate analyses so these denominators are not mixed.

The study labels are the output of the existing automated classification pipeline. This audit verifies frequencies in that coded file; it does not independently re-read source documents or establish inter-rater reliability. Before manuscript claims, review ambiguous algorithms and the 28 unresolved cross-cycle studies at source.

## Machine-readable frequency table

The CSV contains every method count for four primary-method scopes and four any-technique scopes, including the denominator, study share, and counting rule. The temporal-status codes in the input remain:

{status_table}
"""
    if OUT_CSV.exists():
        existing = pd.read_csv(OUT_CSV)
        try:
            pd.testing.assert_frame_equal(existing, table, check_exact=False, rtol=1e-12, atol=1e-12)
        except AssertionError as error:
            raise ValueError("Existing frequency CSV does not match current source-derived values") from error
    else:
        table.to_csv(OUT_CSV, index=False)
    OUT_MD.write_text(report, encoding="utf-8")
    print(f"Primary predictive denominator: {coded_predictive_n}/{predictive_n}")
    print(f"Predictive primary counts: Random Forest={exact_rf}, XGBoost={exact_xgb}")
    print(f"Country-level task D: SVM={country.get('SVM',0)}, RF={country.get('Random Forest',0)}, XGBoost={country.get('XGBoost',0)}")
    print(f"Wrote {OUT_CSV} and {OUT_MD}")


if __name__ == "__main__":
    main()
