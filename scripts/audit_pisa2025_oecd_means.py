#!/usr/bin/env python3
"""Compare project PISA 2025 country means with OECD Volume I StatLink tables.

This is an independent source audit, not a model evaluation. It preserves all
source files and writes a new versioned audit directory only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import importlib.metadata
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from openpyxl import load_workbook


ROOT = Path(__file__).resolve().parents[1]
PISA_PANEL = ROOT / "outputs/stage5/pisa_forecasting_panel_v2.csv"
EVALUATION = ROOT / "outputs/pisa_2025_retrospective_timesfm_v1/all_models_predictions.csv"
COUNTRY_NAMES = ROOT / "analysis/03_data_feasibility/audit/pisa2025_se_validation.csv"
OUTPUT = ROOT / "outputs/pisa2025_oecd_mean_audit_v3"
TABLES = {
    "science": "Table I.B1.2a.1",
    "reading": "Table I.B1.2a.2",
    "mathematics": "Table I.B1.2a.3",
}
COUNTRY_NAME_OVERRIDES = {
    # ISO-like panel key -> OECD 2025 published country/economy label.
    "CZE": "Czechia",
    "HKG": "Hong Kong (China)",
    "MAC": "Macao (China)",
    "QAT": "Qatar",
    "TUR": "Türkiye",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def normalize_name(value: str) -> str:
    value = value.casefold().replace("*", "")
    value = re.sub(r"\([^)]*\)", "", value)
    return re.sub(r"[^a-z0-9]+", "", value)


def read_official_table(
    workbook, sheet_name: str
) -> dict[str, tuple[str, float, float | None]]:
    sheet = workbook[sheet_name]
    # The OECD StatLink workbook has country/economy names in column A,
    # unrounded mean in B, and its sampling standard error in C.
    result: dict[str, tuple[str, float, float | None]] = {}
    for row in range(10, sheet.max_row + 1):
        name = sheet.cell(row, 1).value
        mean = sheet.cell(row, 2).value
        se = sheet.cell(row, 3).value
        if not isinstance(name, str) or not isinstance(mean, (int, float)):
            continue
        key = normalize_name(name)
        if key in result:
            raise ValueError(f"Duplicate OECD country key {name!r} in {sheet_name}")
        result[key] = (
            name,
            float(mean),
            float(se) if isinstance(se, (int, float)) else None,
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workbook", type=Path, required=True, help="Downloaded OECD StatLink xlsx")
    parser.add_argument(
        "--output", type=Path, default=OUTPUT,
        help="New output directory; existing directories are never overwritten",
    )
    args = parser.parse_args()
    workbook_path = args.workbook.resolve()
    if not workbook_path.is_file():
        raise FileNotFoundError(workbook_path)
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing audit directory: {args.output}")
    if not PISA_PANEL.is_file() or not EVALUATION.is_file() or not COUNTRY_NAMES.is_file():
        raise FileNotFoundError("Required project prediction or country-name input is missing")

    evaluation = pd.read_csv(EVALUATION)
    predictions = evaluation.loc[
        evaluation["target_cycle"].eq(2025) & evaluation["model"].eq("Persistence"),
        ["country_iso3", "domain", "true_score"],
    ].copy()
    if predictions.duplicated(["country_iso3", "domain"]).any():
        raise ValueError("Retrospective evaluation outcome keys are not unique")
    if predictions.groupby("domain")["country_iso3"].nunique().to_dict() != {
        domain: 66 for domain in TABLES
    }:
        raise ValueError("Expected 66 evaluation countries in each target domain")

    panel = pd.read_csv(PISA_PANEL, dtype={"country_iso3": str})
    panel = panel.loc[
        panel["program"].eq("PISA") & pd.to_numeric(panel["cycle"], errors="raise").eq(2025),
        ["country_iso3", "cycle", *[f"PISA_{domain}" for domain in TABLES]],
    ].copy()
    country_domain_rows = []
    target_countries = set(predictions["country_iso3"])
    for domain in TABLES:
        score_col = f"PISA_{domain}"
        domain_panel = panel.loc[panel["country_iso3"].isin(target_countries)]
        consistency = domain_panel.groupby("country_iso3")[score_col].nunique(dropna=True)
        if consistency.gt(1).any():
            raise ValueError(f"Inconsistent repeated panel targets in {domain}: {consistency[consistency.gt(1)].index.tolist()}")
        values = domain_panel[["country_iso3", score_col]].dropna().drop_duplicates("country_iso3")
        values = values.rename(columns={score_col: "true_score"})
        values["domain"] = domain
        country_domain_rows.append(values)
    panel_targets = pd.concat(country_domain_rows, ignore_index=True)
    if panel_targets.duplicated(["country_iso3", "domain"]).any():
        raise ValueError("PISA panel targets are not unique by country/domain")
    if len(panel_targets) != 66 * 3:
        raise ValueError(f"Expected 198 panel outcomes for evaluation sample; got {len(panel_targets)}")
    outcomes = predictions.merge(
        panel_targets,
        on=["country_iso3", "domain"],
        how="outer",
        validate="one_to_one",
        suffixes=("_evaluation", "_panel"),
        indicator=True,
    )
    if not outcomes["_merge"].eq("both").all():
        raise ValueError("Evaluation and canonical PISA panel outcome keys differ")
    target_consistency_max_abs = float(
        (outcomes["true_score_evaluation"] - outcomes["true_score_panel"]).abs().max()
    )
    if target_consistency_max_abs > 1e-10:
        raise ValueError(f"Evaluation target differs from canonical panel by {target_consistency_max_abs}")
    predictions = panel_targets.rename(columns={"true_score": "project_mean"})

    names = pd.read_csv(COUNTRY_NAMES, dtype={"iso3": str})[["iso3", "country_name"]]
    if names["iso3"].duplicated().any():
        raise ValueError("Country-name crosswalk contains duplicate ISO keys")
    predictions = predictions.merge(
        names, left_on="country_iso3", right_on="iso3", how="left", validate="many_to_one"
    )
    predictions["country_name"] = predictions.apply(
        lambda row: COUNTRY_NAME_OVERRIDES.get(row["country_iso3"], row["country_name"]),
        axis=1,
    )
    if predictions["country_name"].isna().any():
        missing = predictions.loc[predictions["country_name"].isna(), "country_iso3"].unique()
        raise ValueError(f"No source name for country codes: {missing.tolist()}")

    workbook = load_workbook(workbook_path, read_only=True, data_only=True)
    records = []
    unmatched = []
    for domain, sheet_name in TABLES.items():
        official = read_official_table(workbook, sheet_name)
        subset = predictions.loc[predictions["domain"].eq(domain)]
        for row in subset.itertuples(index=False):
            key = normalize_name(row.country_name)
            if key not in official:
                unmatched.append(
                    {
                        "country_iso3": row.country_iso3,
                        "country_name_project": row.country_name,
                        "domain": domain,
                        "oecd_sheet": sheet_name,
                    }
                )
                continue
            official_name, oecd_mean, oecd_se = official[key]
            project_mean = float(row.project_mean)
            difference = project_mean - oecd_mean
            records.append(
                {
                    "country_iso3": row.country_iso3,
                    "country_name_project": row.country_name,
                    "country_name_oecd": official_name,
                    "domain": domain,
                    "project_mean": project_mean,
                    "oecd_mean_unrounded": oecd_mean,
                    "oecd_mean_rounded_1dp": round(oecd_mean, 1),
                    "oecd_mean_rounded_integer": round(oecd_mean),
                    "oecd_sampling_se": oecd_se,
                    "project_minus_oecd": difference,
                    "absolute_difference": abs(difference),
                    "within_0_5_points": abs(difference) <= 0.5,
                    "within_1_0_points": abs(difference) <= 1.0,
                }
            )
    workbook.close()

    audit = pd.DataFrame(records)
    expected = set(TABLES)
    if set(audit["domain"].unique()) != expected:
        raise ValueError(f"Missing comparison domains: {expected - set(audit['domain'].unique())}")
    counts = audit.groupby("domain")["country_iso3"].nunique().to_dict()
    if counts != {domain: 66 for domain in TABLES}:
        raise ValueError(f"Expected 66 independently matched countries per domain; got {counts}")
    if unmatched:
        # Preserve unmatched rows for review; do not silently drop them.
        unmatched_frame = pd.DataFrame(unmatched)
    else:
        unmatched_frame = pd.DataFrame(
            columns=["country_iso3", "country_name_project", "domain", "oecd_sheet"]
        )

    summaries = []
    for domain, group in audit.groupby("domain", sort=True):
        delta = group["project_minus_oecd"]
        summaries.append(
            {
                "domain": domain,
                "n_matched": int(group["country_iso3"].nunique()),
                "mean_signed_difference": float(delta.mean()),
                "mae_difference": float(delta.abs().mean()),
                "rmse_difference": float((delta.pow(2).mean()) ** 0.5),
                "max_absolute_difference": float(delta.abs().max()),
                "n_within_0_5": int(group["within_0_5_points"].sum()),
                "n_within_1_0": int(group["within_1_0_points"].sum()),
            }
        )

    args.output.mkdir(parents=True)
    archived_workbook = args.output / "oecd_pisa2025_performance_tables.xlsx"
    shutil.copy2(workbook_path, archived_workbook)
    audit.to_csv(args.output / "country_domain_comparison.csv", index=False)
    unmatched_frame.to_csv(args.output / "unmatched_country_keys.csv", index=False)
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_url": "https://stat.link/mrq53f",
        "source_workbook": display_path(archived_workbook),
        "source_workbook_sha256": sha256(archived_workbook),
        "project_score_panel": str(PISA_PANEL.relative_to(ROOT)),
        "project_score_panel_sha256": sha256(PISA_PANEL),
        "evaluation_target_source": str(EVALUATION.relative_to(ROOT)),
        "evaluation_target_source_sha256": sha256(EVALUATION),
        "evaluation_panel_target_max_abs_difference": target_consistency_max_abs,
        "country_name_crosswalk": str(COUNTRY_NAMES.relative_to(ROOT)),
        "country_name_crosswalk_sha256": sha256(COUNTRY_NAMES),
        "audit_script_sha256": sha256(Path(__file__).resolve()),
        "openpyxl_version": importlib.metadata.version("openpyxl"),
        "domain_sheets": TABLES,
        "comparison": "Project estimate versus OECD unrounded national mean; official published integer is also retained.",
        "matched_rows": len(audit),
        "unmatched_rows": len(unmatched_frame),
        "counts_by_domain": counts,
        "summary_by_domain": summaries,
    }
    (args.output / "audit_manifest.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
