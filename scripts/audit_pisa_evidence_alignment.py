#!/usr/bin/env python3
"""Read-only integrity and claim checks for the PISA forecasting evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PASS_COUNT = 0


def check(label: str, condition: bool, detail: object = "") -> None:
    global PASS_COUNT
    if not condition:
        raise AssertionError(f"FAIL: {label}: {detail}")
    PASS_COUNT += 1


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_cycles(value: object) -> list[int]:
    if pd.isna(value) or str(value).strip() in {"", "nan"}:
        return []
    return [int(float(part)) for part in str(value).split(";") if part and part != "nan"]


def check_manifest_outputs(directory: Path, manifest_name: str = "run_manifest.json") -> None:
    manifest = json.loads((directory / manifest_name).read_text())
    outputs = manifest.get("outputs", {})
    for filename, expected_hash in outputs.items():
        if not isinstance(expected_hash, str) or len(expected_hash) != 64:
            continue
        check(
            f"{directory.name}/{filename} output SHA-256",
            sha256(directory / filename) == expected_hash,
        )


def main() -> None:
    panel_dir = ROOT / "outputs/pisa_all_models_quadratic_trend_v1"
    hist_path = panel_dir / "historical_all_models_predictions.csv"
    p25_path = panel_dir / "pisa2025_all_models_predictions.csv"
    hist = pd.read_csv(hist_path)
    p25 = pd.read_csv(p25_path)

    hist_keys = ["target_cycle", "domain", "country_iso3", "model"]
    p25_keys = hist_keys
    check("historical prediction row count", len(hist) == 5_616, len(hist))
    check("historical prediction keys unique", not hist.duplicated(hist_keys).any())
    check("historical target cycles", set(hist.target_cycle) == {2012, 2015, 2018, 2022})
    check("nine historical methods", hist.model.nunique() == 9, sorted(hist.model.unique()))
    expected_country_counts = {2012: 40, 2015: 51, 2018: 57, 2022: 60}
    for target, expected_n in expected_country_counts.items():
        per_domain = hist.loc[hist.target_cycle.eq(target)].groupby("domain").country_iso3.nunique()
        check(f"{target} common three-domain country count", len(per_domain) == 3 and per_domain.eq(expected_n).all(), per_domain.to_dict())
    check(
        "historical inputs precede every target",
        all(all(year < int(target) for year in parse_cycles(cycles)) for cycles, target in zip(hist.context_cycles, hist.target_cycle)),
    )
    check("historical recorded contexts are complete", hist.groupby("model").context_cycles.apply(lambda values: values.notna().all() or values.isna().all()).all())
    check(
        "historical target score agrees across methods",
        hist.groupby(["target_cycle", "domain", "country_iso3"]).true_score.nunique().eq(1).all(),
    )
    tabular_source = (ROOT / "scripts/run_pisa_tabular_benchmarks.py").read_text()
    check("tabular training code truncates at target", "if target_year >= cutoff:\n                break" in tabular_source)
    check("tabular test context excludes target", 'observed["cycle"] < target_cycle' in tabular_source and 'observed["cycle"] == target_cycle' in tabular_source)

    expected_historical_mae = {
        "mathematics": {"Persistence": 9.328, "LinearTrend": 11.297, "LinearQuadraticTrend": 11.564, "Ridge": 9.961, "RandomForest": 9.811, "XGBoost": 9.563, "TimesFM3_zero_shot": 10.629, "TimesFM2_5_zero_shot": 9.873, "TimesFM2_5_ILSA_LoRA": 9.773},
        "reading": {"Persistence": 10.156, "LinearTrend": 13.333, "LinearQuadraticTrend": 14.099, "Ridge": 10.834, "RandomForest": 10.759, "XGBoost": 10.379, "TimesFM3_zero_shot": 11.495, "TimesFM2_5_zero_shot": 10.385, "TimesFM2_5_ILSA_LoRA": 10.273},
        "science": {"Persistence": 8.442, "LinearTrend": 10.799, "LinearQuadraticTrend": 11.671, "Ridge": 9.124, "RandomForest": 9.168, "XGBoost": 8.565, "TimesFM3_zero_shot": 9.704, "TimesFM2_5_zero_shot": 8.850, "TimesFM2_5_ILSA_LoRA": 8.783},
    }
    fold_mae = hist.groupby(["domain", "model", "target_cycle"]).absolute_error.mean()
    equal_fold_mae = fold_mae.groupby(["domain", "model"]).mean()
    for domain, model_values in expected_historical_mae.items():
        for model, expected_value in model_values.items():
            actual = float(equal_fold_mae.loc[domain, model])
            check(f"historical MAE {domain}/{model}", round(actual, 3) == expected_value, actual)

    check("2025 prediction row count", len(p25) == 1_782, len(p25))
    check("2025 prediction keys unique", not p25.duplicated(p25_keys).any())
    check("2025 target cycle and nine methods", set(p25.target_cycle) == {2025} and p25.model.nunique() == 9)
    country_counts = p25.groupby("domain").country_iso3.nunique()
    check("2025 has 66 systems per domain", len(country_counts) == 3 and country_counts.eq(66).all(), country_counts.to_dict())
    common_countries = [set(group.country_iso3) for _, group in p25.groupby("domain")]
    check("2025 common country set across domains", len(common_countries) == 3 and common_countries[0] == common_countries[1] == common_countries[2])
    check("2025 contexts exclude the target", all(all(year < 2025 for year in parse_cycles(cycles)) for cycles in p25.context_cycles))
    check(
        "2025 target score agrees across methods",
        p25.groupby(["domain", "country_iso3"]).true_score.nunique().eq(1).all(),
    )
    p25_mae = p25.groupby(["domain", "model"]).absolute_error.mean()
    expected_leaders = {"mathematics": ("XGBoost", 9.318), "reading": ("Ridge", 11.357), "science": ("Persistence", 9.378)}
    for domain, (method, expected_value) in expected_leaders.items():
        actual = float(p25_mae.loc[domain, method])
        check(f"2025 leader {domain}/{method}", round(actual, 3) == expected_value and actual == float(p25_mae.loc[domain].min()), actual)

    check_manifest_outputs(panel_dir)

    context_panels = {
        "historical TimesFM 3.0": ("outputs/pisa_timesfm_historical_zeroshot_v2/timesfm_predictions.csv", 624),
        "historical TimesFM 2.5 adaptation": ("outputs/pisa_timesfm25_ils_adapter_v1/timesfm25_adaptation_predictions.csv", 1_248),
        "2025 tabular": ("outputs/pisa_2025_retrospective_tabular_v1/tabular_predictions.csv", 990),
        "2025 trend selection": ("outputs/pisa_2025_quadratic_trend_retrospective_v1/tabular_predictions.csv", 198),
        "2025 TimesFM": ("outputs/pisa_2025_retrospective_timesfm_v1/timesfm_predictions.csv", 594),
    }
    for label, (relative_path, expected_rows) in context_panels.items():
        frame = pd.read_csv(ROOT / relative_path)
        check(f"{label} forecast row count", len(frame) == expected_rows, len(frame))
        check(f"{label} context cutoff", frame.context_cycles.notna().all() and all(all(year < int(target) for year in parse_cycles(cycles)) for cycles, target in zip(frame.context_cycles, frame.target_cycle)))

    pisa_panel = pd.read_csv(ROOT / "outputs/stage5/pisa_forecasting_panel_v2.csv", dtype={"country_iso3": str})
    pisa_panel = pisa_panel.loc[pisa_panel.program.eq("PISA")].copy()
    pisa_panel["cycle"] = pd.to_numeric(pisa_panel.cycle, errors="raise").astype(int)
    for label, (relative_path, _) in context_panels.items():
        frame = pd.read_csv(ROOT / relative_path)
        frame = frame.drop_duplicates(["target_cycle", "domain", "country_iso3", "context_cycles"])
        for row in frame.itertuples(index=False):
            score_column = f"PISA_{row.domain}"
            expected_cycles = sorted(
                pisa_panel.loc[
                    pisa_panel.country_iso3.eq(row.country_iso3)
                    & pisa_panel.domain.eq(row.domain)
                    & pisa_panel.cycle.lt(int(row.target_cycle))
                    & pisa_panel[score_column].notna(),
                    "cycle",
                ].astype(int).tolist()
            )
            check(
                f"{label} uses all eligible pre-target PISA cycles",
                parse_cycles(row.context_cycles) == expected_cycles,
                f"{row.country_iso3}/{row.domain}/{row.target_cycle}",
            )

    adapter_manifest = json.loads((ROOT / "outputs/pisa_timesfm25_ils_adapter_v1/run_manifest.json").read_text())
    check(
        "historical adaptation inner validation cutoffs",
        all(int(fold["validation_cycle"]) < int(fold["outer_target_cycle"]) and fold["training_windows_for_selection"] > 0 and fold["refit_training_windows"] > 0 for fold in adapter_manifest["folds"]),
    )
    timesfm_2025_manifest = json.loads((ROOT / "outputs/pisa_2025_retrospective_timesfm_v1/run_manifest.json").read_text())
    check(
        "2025 TimesFM training excludes target",
        "strictly < 2025" in timesfm_2025_manifest["pisa_2025_outcome_training_rule"],
    )
    tabular_2025_manifest = json.loads((ROOT / "outputs/pisa_2025_retrospective_tabular_v1/run_manifest.json").read_text())
    check("2025 tabular training cutoff is documented", "strictly before 2025" in tabular_2025_manifest["cutoff_rule"])

    oecd_dir = ROOT / "outputs/pisa2025_oecd_mean_audit_v3"
    oecd_manifest = json.loads((oecd_dir / "audit_manifest.json").read_text())
    oecd_rows = pd.read_csv(oecd_dir / "country_domain_comparison.csv")
    check("OECD 2025 target rows matched", oecd_manifest["matched_rows"] == 198 and oecd_manifest["unmatched_rows"] == 0)
    check("OECD target means align to scoring panel", oecd_manifest["evaluation_panel_target_max_abs_difference"] == 0.0)
    check("OECD audit output row count", len(oecd_rows) == 198)

    input_dir = ROOT / "outputs/reproducibility_check_v3/pisa_model_input_uncertainty_v4"
    input_manifest = json.loads((input_dir / "run_manifest.json").read_text())
    check("model-input uncertainty draw count", input_manifest["n_draws"] == 2_000)
    check("model-input uncertainty manifest script hash", sha256(ROOT / "scripts/run_pisa_model_input_uncertainty_v4.py") == input_manifest["script_sha256"])
    summary_frames: dict[str, pd.DataFrame] = {}
    for key in ("metrics", "paired", "winner_prob"):
        output = ROOT / input_manifest["outputs"][key]
        check(f"model-input uncertainty {key} output hash", sha256(output) == input_manifest["output_sha256"][key])
        summary_frames[key] = pd.read_csv(output)
    checkpoints = sorted((input_dir / "checkpoints").glob("*.csv"))
    check("model-input uncertainty checkpoint count", len(checkpoints) == 80, len(checkpoints))
    checkpoint_rows = sum(sum(1 for _ in path.open()) - 1 for path in checkpoints)
    check("model-input uncertainty fold rows", checkpoint_rows == 120_000, checkpoint_rows)
    check("model-input uncertainty metric summary rows", len(summary_frames["metrics"]) == 15)
    check("model-input uncertainty paired rows", len(summary_frames["paired"]) == 12)
    winners = summary_frames["winner_prob"]
    check("model-input uncertainty winner rows", len(winners) == 12)
    check(
        "model-input uncertainty winner probabilities sum to one",
        all(abs(float(group.winner_prob.sum()) - 1.0) < 1e-12 for _, group in winners.groupby("domain")),
    )

    uncertainty_dir = ROOT / "outputs/pisa_joint_score_country_uncertainty_v3"
    uncertainty_manifest = json.loads((uncertainty_dir / "run_manifest.json").read_text())
    for filename, expected_hash in uncertainty_manifest["output_sha256"].items():
        check(f"country-cluster/target-score output hash {filename}", sha256(uncertainty_dir / filename) == expected_hash)

    frequency_path = ROOT / "analysis/03_data_feasibility/audit/ilsa_ml_method_frequency_v1.csv"
    frequency = pd.read_csv(frequency_path)
    predictive = frequency.loc[frequency.scope.eq("predictive_studies_with_primary_method")].set_index("method").study_count
    country_scope = frequency.loc[frequency.scope.eq("country_level_prediction_task_D")].drop_duplicates("method").set_index("method").study_count
    check("literature frequency predictive RF/XGBoost", int(predictive["Random Forest"]) == 42 and int(predictive["XGBoost"]) == 17)
    check("country-level frequency differs from broad scope", int(country_scope["SVM"]) == 4 and int(country_scope["Random Forest"]) == 3 and int(country_scope["XGBoost"]) == 2)

    print(f"PASS: {PASS_COUNT} read-only evidence-alignment checks")
    print("Historical country counts:", expected_country_counts)
    print("Historical equal-fold leaders: Persistence in mathematics, reading, and science")
    print("PISA 2025 point-MAE leaders: XGBoost / Ridge / Persistence")
    print("PISA 2025 target rows: 198/198 aligned to OECD table")
    print("Model-input uncertainty: 2,000 draws; 80 checkpoints; 120,000 fold rows")


if __name__ == "__main__":
    main()
