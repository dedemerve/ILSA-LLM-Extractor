"""Path mapping and discovery for run_batch_folders (no API)."""

from pathlib import Path

import pytest

from scripts.run_batch_folders import (
    discover_jobs,
    json_output_path,
    list_immediate_subdirs,
    safe_json_stem,
    should_skip_pdf,
    source_database_for_subdir,
)
from utils.storage import EXTRACTION_FAILED_PREFIX


def test_safe_json_stem_truncates_and_sanitizes():
    long = "a" * 120
    pdf = Path(f"/tmp/{long}.pdf")
    assert len(safe_json_stem(pdf)) == 80
    pdf2 = Path("/tmp/foo/bar/baz.pdf")
    assert "/" not in safe_json_stem(pdf2)


def test_json_output_path_layout():
    top = Path("/data/Scopus")
    pdf = top / "Scopus 2020-2026" / "TIMSS" / "TIMSS 2019" / "paper.pdf"
    out = json_output_path(pdf, top, "Scopus", Path("/proj/outputs"))
    assert out == Path(
        "/proj/outputs/Scopus/Scopus 2020-2026/TIMSS/TIMSS 2019/json/paper.json"
    )


def test_json_output_path_long_stem():
    stem = "x" * 100
    top = Path("/data/IEA")
    pdf = top / "nested" / f"{stem}.pdf"
    out = json_output_path(pdf, top, "IEA", Path("/out"))
    assert out.name == f"{'x' * 80}.json"
    assert out.parent == Path("/out/IEA/nested/json")


def test_source_database_for_subdir():
    assert source_database_for_subdir("Web of Science") == "web_of_science"


def test_list_immediate_subdirs(tmp_path: Path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    (tmp_path / "skip.pdf").write_bytes(b"%PDF")
    (tmp_path / ".hidden").mkdir()
    names = [p.name for p in list_immediate_subdirs(tmp_path)]
    assert names == ["a", "b"]


def test_discover_jobs_skips_complete_json(tmp_path: Path):
    root = tmp_path / "root"
    sub = root / "WoS"
    sub.mkdir(parents=True)
    pdf = sub / "article.pdf"
    pdf.write_bytes(b"%PDF-1.4")

    out_root = tmp_path / "outputs"
    json_dir = out_root / "WoS" / "json"
    json_dir.mkdir(parents=True)
    payload = {
        "metadata": {"file_name": "article.pdf", "authors": []},
        "data": {
            "survey_design": {
                "student_weights_used": False,
                "weight_fields_interpretation": "fixture",
            },
            "plausible_values_handling": "not_reported",
            "missing_data_handling": "not_reported",
            "sample_details": {"countries": [], "sample_filtering_criteria": "test"},
            "ml_techniques": {"primary": None, "all_techniques": []},
            "confounders_identified": [],
            "main_findings": [],
            "outcome_summary": "fixture",
        },
    }
    (json_dir / "article.json").write_text(
        __import__("json").dumps(payload), encoding="utf-8"
    )

    jobs = discover_jobs(root, outputs_root=out_root)
    assert jobs == []


def test_should_skip_pdf_failure_payload(tmp_path: Path):
    path = tmp_path / "fail.json"
    path.write_text(
        '{"metadata":{"file_name":"x.pdf"},"data":{"main_findings":[],"outcome_summary":"'
        + EXTRACTION_FAILED_PREFIX
        + 'oops"}}',
        encoding="utf-8",
    )
    assert should_skip_pdf(path) is False


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
