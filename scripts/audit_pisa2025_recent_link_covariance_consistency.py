#!/usr/bin/env python3
"""Check whether recent PISA link marginals and trend-link SEs admit a covariance.

For the 2018, 2022, 2025 window, this solves the covariance between the two
pre-2025 link offsets implied by OECD's decennial-trend link-error SE. It then
checks the necessary Cauchy bound. It is a consistency diagnostic, not a
covariance estimator for model-error simulation.
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
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LINK_SUMMARY = ROOT / "outputs/pisa2025_pairwise_link_error_v1/pairwise_link_error_summary.csv"
DEFAULT_OECD_WORKBOOK = ROOT / "outputs/pisa2025_oecd_mean_audit_v3/oecd_pisa2025_performance_tables.xlsx"
DEFAULT_OUTPUT = ROOT / "outputs/pisa2025_recent_link_covariance_diagnostic_v1"

OECD_A3_URL = (
    "https://www.oecd.org/en/publications/pisa-2025-results-volume-i_73451bc5-en/"
    "full-report/technical-notes-on-analyses-in-this-volume_46a7001c.html"
)
OECD_A3_STATLINK = "https://stat.link/p7oc9v"
OECD_PERFORMANCE_STATLINK = "https://stat.link/mrq53f"
OECD_2022_A7_URL = (
    "https://www.oecd.org/en/publications/2023/12/pisa-2022-results-volume-i_76772a36/"
    "full-report/comparing-mathematics-reading-and-science-performance-across-pisa-assessments_1c417d6f.html"
)

# Table I.A3.1: link-error SE for the linear/decennial trend using 2018, 2022, 2025.
# Values are score points per decade; source table is linked above.
TREND_LINK_SE_2018_2022_2025 = {
    "mathematics": 3.652,
    "reading": 2.617,
    "science": 4.741,
}

NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_link_errors(path: Path) -> dict[tuple[str, int], float]:
    result: dict[tuple[str, int], float] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            result[(row["domain"], int(row["previous_cycle"]))] = float(
                row["inferred_link_error_se_median"]
            )
    required = {(domain, year) for domain in TREND_LINK_SE_2018_2022_2025 for year in (2018, 2022)}
    missing = required - result.keys()
    if missing:
        raise ValueError(f"Missing pairwise link-error values: {sorted(missing)}")
    return result


def read_science_oecd_35_trend(path: Path) -> tuple[float, float]:
    """Read OECD-35 2015/18/22/25 science scores and published trend from row 12."""
    ns = NS
    with zipfile.ZipFile(path) as book:
        shared: list[str] = []
        if "xl/sharedStrings.xml" in book.namelist():
            root = ET.fromstring(book.read("xl/sharedStrings.xml"))
            shared = [
                "".join(node.text or "" for node in item.findall(".//m:t", ns))
                for item in root.findall("m:si", ns)
            ]
        workbook = ET.fromstring(book.read("xl/workbook.xml"))
        relationships = ET.fromstring(book.read("xl/_rels/workbook.xml.rels"))
        relmap = {node.attrib["Id"]: node.attrib["Target"] for node in relationships}
        target = None
        for sheet in workbook.find("m:sheets", ns):
            if sheet.attrib["name"] == "Table I.B1.2a.36":
                target = relmap[sheet.attrib[f"{{{REL_NS}}}id"]]
                break
        if target is None:
            raise ValueError("OECD workbook is missing Table I.B1.2a.36")
        sheet_path = target.lstrip("/") if target.startswith("/") else "xl/" + target.lstrip("/")
        sheet = ET.fromstring(book.read(sheet_path))
        row = next((r for r in sheet.findall(".//m:sheetData/m:row", ns) if r.attrib.get("r") == "12"), None)
        if row is None:
            raise ValueError("OECD science table is missing OECD-35 row 12")
        values: dict[str, str] = {}
        for cell in row.findall("m:c", ns):
            match = re.match(r"([A-Z]+)", cell.attrib["r"])
            value = cell.find("m:v", ns)
            if match and value is not None:
                raw = value.text or ""
                if cell.attrib.get("t") == "s":
                    raw = shared[int(raw)]
                values[match.group(1)] = raw
    # H, J, L, N are the means for 2015, 2018, 2022, 2025; AB is the
    # published average decennial science trend for this row.
    means = [float(values[column]) for column in ("H", "J", "L", "N")]
    published_trend = float(values["AB"])
    return means[0], published_trend


def calendar_year_ols_weights(years: tuple[int, ...], target_year: int = 2025) -> dict[int, float]:
    x = [float(year) for year in years]
    mean_x = sum(x) / len(x)
    denominator = sum((value - mean_x) ** 2 for value in x)
    if denominator <= 0:
        raise ValueError("OLS time values must have non-zero variance")
    return {year: 10.0 * (float(year) - mean_x) / denominator for year in years}


def calculate(link_errors: dict[tuple[str, int], float]) -> list[dict[str, object]]:
    weights = calendar_year_ols_weights((2018, 2022, 2025))
    w18, w22 = weights[2018], weights[2022]
    base_variance_coefficient = 2.0 * w18 * w22
    if base_variance_coefficient == 0:
        raise ValueError("The two historical cycle weights must both be non-zero")
    results: list[dict[str, object]] = []
    for domain, trend_se in TREND_LINK_SE_2018_2022_2025.items():
        se18 = link_errors[(domain, 2018)]
        se22 = link_errors[(domain, 2022)]
        base_variance = (w18 * se18) ** 2 + (w22 * se22) ** 2
        implied_covariance = (trend_se**2 - base_variance) / base_variance_coefficient
        cauchy_limit = se18 * se22
        min_variance = base_variance - abs(base_variance_coefficient) * cauchy_limit
        max_variance = base_variance + abs(base_variance_coefficient) * cauchy_limit
        min_feasible_se = math.sqrt(max(min_variance, 0.0))
        max_feasible_se = math.sqrt(max(max_variance, 0.0))
        results.append({
            "domain": domain,
            "se_link_2018_to_2025": se18,
            "se_link_2022_to_2025": se22,
            "reported_trend_link_se_2018_2022_2025": trend_se,
            "ols_weight_2018_per_decade": w18,
            "ols_weight_2022_per_decade": w22,
            "covariance_2018_2022_implied_by_trend": implied_covariance,
            "maximum_absolute_covariance_by_cauchy": cauchy_limit,
            "minimum_feasible_trend_link_se": min_feasible_se,
            "maximum_feasible_trend_link_se": max_feasible_se,
            "reported_trend_se_within_feasible_range": (
                min_feasible_se - 0.0005 <= trend_se <= max_feasible_se + 0.0005
            ),
            "cauchy_excess": max(abs(implied_covariance) - cauchy_limit, 0.0),
        })
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--link-summary", type=Path, default=DEFAULT_LINK_SUMMARY)
    parser.add_argument("--oecd-workbook", type=Path, default=DEFAULT_OECD_WORKBOOK)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    link_path = args.link_summary.resolve()
    workbook_path = args.oecd_workbook.resolve()
    output_dir = args.output_dir.resolve()
    if not link_path.is_file() or not workbook_path.is_file():
        parser.error("A required source file is missing")
    if output_dir.exists():
        parser.error(f"Output already exists; refusing to overwrite: {output_dir}")

    # Calendar-year OLS weights are checked against the OECD-35 science
    # 2015-2025 trend in the source workbook before the subset covariance test.
    _, published_point_trend = read_science_oecd_35_trend(workbook_path)
    trend_weights = calendar_year_ols_weights((2015, 2018, 2022, 2025))
    oecd35_science_means: tuple[float, ...]
    with zipfile.ZipFile(workbook_path) as book:
        shared: list[str] = []
        if "xl/sharedStrings.xml" in book.namelist():
            root = ET.fromstring(book.read("xl/sharedStrings.xml"))
            shared = [
                "".join(node.text or "" for node in item.findall(".//m:t", NS))
                for item in root.findall("m:si", NS)
            ]
        workbook = ET.fromstring(book.read("xl/workbook.xml"))
        relationships = ET.fromstring(book.read("xl/_rels/workbook.xml.rels"))
        relmap = {node.attrib["Id"]: node.attrib["Target"] for node in relationships}
        target = next(
            relmap[s.attrib[f"{{{REL_NS}}}id"]]
            for s in workbook.find("m:sheets", NS)
            if s.attrib["name"] == "Table I.B1.2a.36"
        )
        sheet_path = target.lstrip("/") if target.startswith("/") else "xl/" + target.lstrip("/")
        sheet = ET.fromstring(book.read(sheet_path))
        row = next(r for r in sheet.findall(".//m:sheetData/m:row", NS) if r.attrib.get("r") == "12")
        cells: dict[str, str] = {}
        for cell in row.findall("m:c", NS):
            match = re.match(r"([A-Z]+)", cell.attrib["r"])
            value = cell.find("m:v", NS)
            if match and value is not None:
                raw = value.text or ""
                if cell.attrib.get("t") == "s":
                    raw = shared[int(raw)]
                cells[match.group(1)] = raw
        oecd35_science_means = tuple(float(cells[c]) for c in ("H", "J", "L", "N"))
    calculated_point_trend = sum(
        trend_weights[year] * score
        for year, score in zip((2015, 2018, 2022, 2025), oecd35_science_means)
    )
    point_trend_abs_difference = abs(calculated_point_trend - published_point_trend)
    if point_trend_abs_difference > 0.001:
        raise ValueError(
            "Calendar-year OLS weights do not reproduce the OECD-35 science trend: "
            f"difference={point_trend_abs_difference}"
        )

    results = calculate(load_link_errors(link_path))
    output_dir.mkdir(parents=True)
    result_path = output_dir / "recent_link_covariance_consistency.csv"
    with result_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    manifest = {
        "analysis": "Cauchy-feasibility check for 2018/2022/2025 PISA link-error covariance implied by official trend-link SEs",
        "interpretation": "consistency diagnostic only; no covariance matrix is estimated or used in model uncertainty draws",
        "formula": "Var(T) = w18^2*SE18^2 + w22^2*SE22^2 + 2*w18*w22*Cov18_22; 2025 is the reference offset",
        "trend_link_se_source_values": TREND_LINK_SE_2018_2022_2025,
        "trend_link_se_source": OECD_A3_STATLINK,
        "method_reference": OECD_A3_URL,
        "multi_cycle_method_reference": OECD_2022_A7_URL,
        "pairwise_link_error_source_statlink": OECD_PERFORMANCE_STATLINK,
        "pairwise_link_error_summary": str(link_path),
        "pairwise_link_error_summary_sha256": sha256(link_path),
        "oecd_performance_workbook": str(workbook_path),
        "oecd_performance_workbook_sha256": sha256(workbook_path),
        "script_sha256": sha256(Path(__file__).resolve()),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "calendar_year_ols_validation": {
            "check": "reproduce OECD-35 science 2015-2025 decennial trend in Table I.B1.2a.36 row 12",
            "calculated": calculated_point_trend,
            "published": published_point_trend,
            "absolute_difference": point_trend_abs_difference,
            "passed": point_trend_abs_difference <= 0.001,
        },
        "checks": {
            "calendar_year_ols_weights_reproduce_published_point_trend": point_trend_abs_difference <= 0.001,
            "math_covariance_feasible": bool(results[0]["reported_trend_se_within_feasible_range"]),
            "reading_covariance_feasible": bool(results[1]["reported_trend_se_within_feasible_range"]),
            "science_covariance_feasible": bool(results[2]["reported_trend_se_within_feasible_range"]),
            "covariance_matrix_estimated": False,
        },
        "outputs": [result_path.name],
    }
    manifest["output_sha256"] = {result_path.name: sha256(result_path)}
    manifest_path = output_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output_dir": str(output_dir),
        "checks": manifest["checks"],
        "diagnostics": results,
        "output_sha256": manifest["output_sha256"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
