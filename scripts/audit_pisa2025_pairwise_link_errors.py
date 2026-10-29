#!/usr/bin/env python3
"""Reconstruct PISA 2025 pairwise scale-link error SEs from OECD tables.

This is a read-only audit of the OECD performance-tables workbook. It uses
published score SEs and cross-cycle difference SEs, then validates the
country-invariant results against the values published by NCES for PISA 2025.
It does not construct a cross-cycle covariance matrix.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
import xml.etree.ElementTree as ET
import zipfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "outputs/pisa2025_oecd_mean_audit_v3/oecd_pisa2025_performance_tables.xlsx"
DEFAULT_OUTPUT = ROOT / "outputs/pisa2025_pairwise_link_error_v1"
OECD_TABLES_URL = (
    "https://stat.link/mrq53f"
)
NCES_METHOD_URL = "https://nces.ed.gov/surveys/pisa/pisa2025/technical-notes/index.asp"
OECD_TECH_NOTES_URL = (
    "https://www.oecd.org/en/publications/pisa-2025-results-volume-i_73451bc5-en/"
    "full-report/technical-notes-on-analyses-in-this-volume_46a7001c.html"
)

# Domain: (worksheet, target-2025 mean SE column, prior cycle fields)
# Prior cycle fields: (cycle, prior mean SE column, 2025-minus-prior difference SE column)
TABLES = {
    "science": (
        "Table I.B1.2a.36",
        "O",
        [(2006, "C", "Q"), (2009, "E", "S"), (2012, "G", "U"),
         (2015, "I", "W"), (2018, "K", "Y"), (2022, "M", "AA")],
    ),
    "reading": (
        "Table I.B1.2a.37",
        "S",
        [(2000, "C", "U"), (2003, "E", "W"), (2006, "G", "Y"),
         (2009, "I", "AA"), (2012, "K", "AC"), (2015, "M", "AE"),
         (2018, "O", "AG"), (2022, "Q", "AI")],
    ),
    "mathematics": (
        "Table I.B1.2a.38",
        "Q",
        [(2003, "C", "S"), (2006, "E", "U"), (2009, "G", "W"),
         (2012, "I", "Y"), (2015, "K", "AA"), (2018, "M", "AC"),
         (2022, "O", "AE")],
    ),
}

# NCES PISA 2025 Methodology and Technical Notes, Table 4, rounded to 0.001.
EXPECTED = {
    ("science", 2006): 4.822, ("science", 2009): 6.690,
    ("science", 2012): 6.062, ("science", 2015): 3.408,
    ("science", 2018): 3.507, ("science", 2022): 3.116,
    ("reading", 2000): 6.759, ("reading", 2003): 5.363,
    ("reading", 2006): 8.630, ("reading", 2009): 4.787,
    ("reading", 2012): 6.109, ("reading", 2015): 3.791,
    ("reading", 2018): 1.832, ("reading", 2022): 1.094,
    ("mathematics", 2003): 5.673, ("mathematics", 2006): 4.268,
    ("mathematics", 2009): 4.450, ("mathematics", 2012): 3.782,
    ("mathematics", 2015): 2.999, ("mathematics", 2018): 2.551,
    ("mathematics", 2022): 1.220,
}

NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def column_name(cell_reference: str) -> str:
    match = re.match(r"([A-Z]+)", cell_reference)
    if not match:
        raise ValueError(f"Unexpected spreadsheet cell reference: {cell_reference}")
    return match.group(1)


def workbook_parts(book: zipfile.ZipFile) -> tuple[list[str], dict[str, str]]:
    strings: list[str] = []
    if "xl/sharedStrings.xml" in book.namelist():
        shared = ET.fromstring(book.read("xl/sharedStrings.xml"))
        for item in shared.findall("m:si", NS):
            strings.append("".join(node.text or "" for node in item.findall(".//m:t", NS)))

    workbook = ET.fromstring(book.read("xl/workbook.xml"))
    relationships = ET.fromstring(book.read("xl/_rels/workbook.xml.rels"))
    targets = {rel.attrib["Id"]: rel.attrib["Target"] for rel in relationships}
    sheet_paths: dict[str, str] = {}
    for sheet in workbook.find("m:sheets", NS):
        target = targets[sheet.attrib[f"{{{REL_NS}}}id"]]
        if target.startswith("/"):
            path = target.lstrip("/")
        else:
            path = "xl/" + target.lstrip("/")
        sheet_paths[sheet.attrib["name"]] = path
    return strings, sheet_paths


def cell_values(row: ET.Element, shared_strings: list[str]) -> dict[str, str]:
    values: dict[str, str] = {}
    for cell in row.findall("m:c", NS):
        value = cell.find("m:v", NS)
        if value is None:
            continue
        raw = value.text or ""
        if cell.attrib.get("t") == "s":
            raw = shared_strings[int(raw)]
        values[column_name(cell.attrib["r"])] = raw
    return values


def parse_number(values: dict[str, str], column: str) -> float | None:
    raw = values.get(column, "")
    if not raw or raw.strip().lower() in {"m", "..", "n"}:
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def reconstruct(input_path: Path) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    detail: list[dict[str, object]] = []
    grouped: dict[tuple[str, int], list[float]] = defaultdict(list)
    with zipfile.ZipFile(input_path) as book:
        shared_strings, sheet_paths = workbook_parts(book)
        for domain, (sheet_name, target_se_col, prior_fields) in TABLES.items():
            if sheet_name not in sheet_paths:
                raise ValueError(f"Missing expected OECD worksheet: {sheet_name}")
            sheet = ET.fromstring(book.read(sheet_paths[sheet_name]))
            rows = sheet.findall(".//m:sheetData/m:row", NS)
            header_row = next((r for r in rows if r.attrib.get("r") == "9"), None)
            if header_row is None:
                raise ValueError(f"Missing expected header row 9 in {sheet_name}")
            headers = cell_values(header_row, shared_strings)
            for cycle, prior_se_col, diff_se_col in prior_fields:
                if headers.get(prior_se_col) != "S.E." or headers.get(diff_se_col) != "S.E.":
                    raise ValueError(
                        f"Unexpected column header in {sheet_name}: "
                        f"{prior_se_col}={headers.get(prior_se_col)!r}, "
                        f"{diff_se_col}={headers.get(diff_se_col)!r}"
                    )
                for row in rows:
                    if int(row.attrib.get("r", "0")) < 10:
                        continue
                    values = cell_values(row, shared_strings)
                    entity = values.get("A", "").strip()
                    if not entity or entity.lower().startswith("oecd average"):
                        continue
                    prior_se = parse_number(values, prior_se_col)
                    target_se = parse_number(values, target_se_col)
                    change_se = parse_number(values, diff_se_col)
                    if prior_se is None or target_se is None or change_se is None:
                        continue
                    variance = change_se**2 - prior_se**2 - target_se**2
                    if variance < -1e-6:
                        raise ValueError(
                            f"Negative inferred link-error variance ({variance}) "
                            f"for {domain}/{cycle}/{entity}"
                        )
                    link_se = math.sqrt(max(variance, 0.0))
                    grouped[(domain, cycle)].append(link_se)
                    detail.append({
                        "domain": domain,
                        "previous_cycle": cycle,
                        "country_or_economy": entity,
                        "se_previous_score": prior_se,
                        "se_2025_score": target_se,
                        "se_2025_minus_previous": change_se,
                        "inferred_link_error_se": link_se,
                    })

    summary: list[dict[str, object]] = []
    for key, expected in EXPECTED.items():
        values = grouped.get(key, [])
        if not values:
            raise ValueError(f"No country/economy rows reconstructed for {key}")
        median = sorted(values)[len(values) // 2] if len(values) % 2 else (
            sorted(values)[len(values) // 2 - 1] + sorted(values)[len(values) // 2]
        ) / 2
        max_abs_deviation = max(abs(value - median) for value in values)
        expected_deviation = abs(median - expected)
        summary.append({
            "domain": key[0],
            "previous_cycle": key[1],
            "n_country_economy_rows": len(values),
            "inferred_link_error_se_median": median,
            "inferred_link_error_se_min": min(values),
            "inferred_link_error_se_max": max(values),
            "within_table_max_abs_deviation_from_median": max_abs_deviation,
            "nces_reference_link_error_se_rounded_3dp": expected,
            "abs_deviation_from_nces_3dp": expected_deviation,
            "matches_nces_3dp_within_0_0005": expected_deviation <= 0.0005,
        })
        if expected_deviation > 0.0005:
            raise ValueError(f"Reconstruction does not match NCES reference for {key}")
        if max_abs_deviation > 0.0001:
            raise ValueError(f"Country-invariant link SE check failed for {key}")
    return detail, summary


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty result: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    input_path = args.input.resolve()
    output_dir = args.output_dir.resolve()
    if not input_path.is_file():
        parser.error(f"Input workbook not found: {input_path}")
    if output_dir.exists():
        parser.error(f"Output already exists; refusing to overwrite: {output_dir}")

    detail, summary = reconstruct(input_path)
    output_dir.mkdir(parents=True)
    detail_path = output_dir / "country_pairwise_link_error_reconstruction.csv"
    summary_path = output_dir / "pairwise_link_error_summary.csv"
    write_csv(detail_path, detail)
    write_csv(summary_path, summary)
    manifest = {
        "analysis": "reconstruct PISA 2025 pairwise scale-link error SEs from OECD cycle-mean and difference SE tables",
        "formula": "sqrt(SE(2025 - t)^2 - SE(t)^2 - SE(2025)^2)",
        "interpretation": "marginal pairwise link-error SEs only; no cross-cycle covariance matrix is estimated",
        "source_workbook": str(input_path),
        "source_workbook_sha256": sha256(input_path),
        "source_tables": [config[0] for config in TABLES.values()],
        "reference_url_nces": NCES_METHOD_URL,
        "reference_url_oecd_notes": OECD_TECH_NOTES_URL,
        "reference_url_oecd_workbook_statlink": OECD_TABLES_URL,
        "script_sha256": sha256(Path(__file__).resolve()),
        "python_version": sys.version,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "detail_rows": len(detail),
        "summary_rows": len(summary),
        "checks": {
            "all_21_domain_cycle_pairs_reconstructed": len(summary) == 21,
            "country_invariant_within_table": all(
                float(row["within_table_max_abs_deviation_from_median"]) <= 0.0001
                for row in summary
            ),
            "matches_nces_reference_to_3dp": all(
                row["matches_nces_3dp_within_0_0005"] for row in summary
            ),
            "covariance_matrix_estimated": False,
        },
        "outputs": [detail_path.name, summary_path.name],
        "output_sha256": {
            detail_path.name: sha256(detail_path),
            summary_path.name: sha256(summary_path),
        },
    }
    manifest_path = output_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output_dir": str(output_dir),
        "detail_rows": len(detail),
        "summary_rows": len(summary),
        "checks": manifest["checks"],
        "summary_csv_sha256": sha256(summary_path),
        "manifest_sha256": sha256(manifest_path),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
