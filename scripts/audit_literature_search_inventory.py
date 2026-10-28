#!/usr/bin/env python3
"""Inventory saved Scopus/WoS extraction JSONs and exact-match them to the ILSA corpus.

This is a provenance audit of files currently present in the workspace. It does
not recreate database searches, establish the date they were run, or infer why
a record was included/excluded. Outputs are versioned and never overwritten.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "analysis/02_prediction_taxonomy/study_classification.csv"
SEARCH_ROOTS = {
    "Scopus": ROOT / "outputs/Scopus",
    "Web of Science": ROOT / "outputs/Web of Science",
}
OUT_DIR = ROOT / "analysis/03_data_feasibility/audit/literature_search_inventory_v1"
INVENTORY = OUT_DIR / "saved_search_json_inventory.csv"
REPORT = OUT_DIR / "inventory_report.json"
MANIFEST = OUT_DIR / "manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_doi(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip().lower()
    text = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", text)
    text = re.sub(r"^doi:\s*", "", text)
    return text.rstrip(" .;,")


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    if OUT_DIR.exists() or INVENTORY.exists() or REPORT.exists() or MANIFEST.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {OUT_DIR}")
    if not CORPUS.is_file():
        raise FileNotFoundError(CORPUS)

    corpus_rows = read_csv_rows(CORPUS)
    corpus_by_doi: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in corpus_rows:
        doi = normalize_doi(row.get("doi"))
        if doi:
            corpus_by_doi[doi].append(row)

    output_rows: list[dict[str, object]] = []
    for database, search_root in SEARCH_ROOTS.items():
        if not search_root.is_dir():
            raise FileNotFoundError(search_root)
        json_paths = sorted(search_root.rglob("*.json"), key=lambda p: p.as_posix().casefold())
        for path in json_paths:
            with path.open("r", encoding="utf-8") as stream:
                payload = json.load(stream)
            metadata = payload.get("metadata") or {}
            doi = normalize_doi(metadata.get("doi"))
            matched = corpus_by_doi.get(doi, []) if doi else []
            relpath = path.relative_to(ROOT).as_posix()
            query_folder = path.relative_to(search_root).parts[0]
            output_rows.append(
                {
                    "database": database,
                    "query_folder": query_folder,
                    "relative_json_path": relpath,
                    "json_sha256": sha256(path),
                    "title": metadata.get("title") or "",
                    "year": metadata.get("year") or "",
                    "doi_raw": metadata.get("doi") or "",
                    "doi_normalized": doi,
                    "source_category": metadata.get("source_category") or "",
                    "exact_doi_match_in_1266": bool(matched),
                    "matched_corpus_records": len(matched),
                    "matched_corpus_file_names": " | ".join(
                        row.get("file_name", "") for row in matched
                    ),
                    "matched_corpus_sources": " | ".join(
                        row.get("source_category", "") for row in matched
                    ),
                }
            )

    OUT_DIR.mkdir(parents=True, exist_ok=False)
    fieldnames = list(output_rows[0]) if output_rows else []
    with INVENTORY.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    by_db: dict[str, dict[str, object]] = {}
    for database in SEARCH_ROOTS:
        rows = [row for row in output_rows if row["database"] == database]
        dois = [str(row["doi_normalized"]) for row in rows if row["doi_normalized"]]
        by_db[database] = {
            "saved_json_files": len(rows),
            "with_doi": sum(bool(row["doi_normalized"]) for row in rows),
            "distinct_normalized_dois": len(set(dois)),
            "duplicate_doi_rows_within_database": len(dois) - len(set(dois)),
            "exact_doi_matched_json_rows": sum(bool(row["exact_doi_match_in_1266"]) for row in rows),
            "query_folder_counts": dict(sorted(Counter(str(row["query_folder"]) for row in rows).items())),
        }

    all_dois = {str(row["doi_normalized"]) for row in output_rows if row["doi_normalized"]}
    corpus_dois = set(corpus_by_doi)
    matched_json_dois = all_dois & corpus_dois
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Inventory of locally saved extraction JSONs; not a reconstruction of database execution or screening.",
        "corpus": {
            "path": CORPUS.relative_to(ROOT).as_posix(),
            "rows": len(corpus_rows),
            "doi_records": sum(bool(normalize_doi(row.get("doi"))) for row in corpus_rows),
            "unique_normalized_dois": len(corpus_dois),
            "sha256": sha256(CORPUS),
        },
        "saved_search_json_files": len(output_rows),
        "json_rows_with_doi": sum(bool(row["doi_normalized"]) for row in output_rows),
        "distinct_normalized_dois_across_databases": len(all_dois),
        "distinct_search_dois_exactly_matching_corpus": len(matched_json_dois),
        "distinct_search_dois_not_matching_corpus": sorted(all_dois - corpus_dois),
        "distinct_corpus_dois_not_in_saved_search_jsons": len(corpus_dois - all_dois),
        "database_summary": by_db,
        "limitations": [
            "Saved query-folder names retain a nominal query string but not a database execution timestamp or raw export count.",
            "JSON files are project extraction records, not raw Scopus/Web of Science exports.",
            "Exact DOI matching does not establish title-level matches for records without a DOI.",
            "The inventory contains no inclusion/exclusion decisions or reasons.",
        ],
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    MANIFEST.write_text(
        json.dumps(
            {
                "created_utc": report["created_utc"],
                "script": Path(__file__).relative_to(ROOT).as_posix(),
                "script_sha256": sha256(Path(__file__)),
                "inputs": {
                    "corpus": {"path": report["corpus"]["path"], "sha256": report["corpus"]["sha256"]},
                    "search_roots": [p.relative_to(ROOT).as_posix() for p in SEARCH_ROOTS.values()],
                    "search_json_file_count": len(output_rows),
                },
                "outputs": {
                    "inventory": {"path": INVENTORY.relative_to(ROOT).as_posix(), "sha256": sha256(INVENTORY), "rows": len(output_rows)},
                    "report": {"path": REPORT.relative_to(ROOT).as_posix(), "sha256": sha256(REPORT)},
                },
                "overwrite_policy": "Output directory must not exist before execution.",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
