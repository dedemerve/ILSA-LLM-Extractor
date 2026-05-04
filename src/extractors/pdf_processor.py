from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF


SECTION_PATTERNS = {
    "abstract": re.compile(
        r"^\s*(?:abstract|summary)\s*$",
        re.IGNORECASE | re.MULTILINE,
    ),
    "introduction": re.compile(
        r"^\s*(?:1\.?\s+)?introduction\s*$",
        re.IGNORECASE | re.MULTILINE,
    ),
    "methods": re.compile(
        r"^\s*(?:\d\.?\s+)?(?:methods?|methodology|materials\s+and\s+methods|"
        r"research\s+design|research\s+methodology|data\s+and\s+methods)\s*$",
        re.IGNORECASE | re.MULTILINE,
    ),
    "results": re.compile(
        r"^\s*(?:\d\.?\s+)?(?:results?|findings?|analysis)\s*$",
        re.IGNORECASE | re.MULTILINE,
    ),
    "discussion": re.compile(
        r"^\s*(?:\d\.?\s+)?(?:discussion|discussion\s+and\s+conclusions?)\s*$",
        re.IGNORECASE | re.MULTILINE,
    ),
    "conclusion": re.compile(
        r"^\s*(?:\d\.?\s+)?(?:conclusions?|concluding\s+remarks)\s*$",
        re.IGNORECASE | re.MULTILINE,
    ),
    "references": re.compile(
        r"^\s*(?:references?|bibliography|works\s+cited)\s*$",
        re.IGNORECASE | re.MULTILINE,
    ),
}


@dataclass
class ProcessedPDF:
    """Container for the parsed content of a single PDF document."""

    file_path: Path
    file_name: str
    source_database: str
    total_pages: int
    raw_text: str
    sections: dict[str, str] = field(default_factory=dict)
    used_smart_sections: bool = False
    extraction_text: str = ""
    estimated_tokens: int = 0
    parse_errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialize processed PDF metadata for logging or persistence."""
        return {
            "file_name": self.file_name,
            "source_database": self.source_database,
            "total_pages": self.total_pages,
            "sections_found": list(self.sections.keys()),
            "used_smart_sections": self.used_smart_sections,
            "estimated_tokens": self.estimated_tokens,
            "parse_errors": self.parse_errors,
        }


def extract_raw_text(pdf_path: Path) -> tuple[str, int, list[str]]:
    """
    Extract raw text from a PDF file using PyMuPDF.

    Returns a tuple of (full_text, page_count, errors). Errors are collected
    rather than raised so that batch processing can continue.
    """
    errors: list[str] = []
    try:
        doc = fitz.open(pdf_path)
    except Exception as exc:
        errors.append(f"Failed to open PDF: {exc}")
        return "", 0, errors

    text_parts: list[str] = []
    page_count = doc.page_count
    for page_index in range(page_count):
        try:
            page = doc.load_page(page_index)
            text_parts.append(page.get_text("text"))
        except Exception as exc:
            errors.append(f"Failed to extract page {page_index}: {exc}")
    doc.close()
    return "\n".join(text_parts), page_count, errors


def detect_sections(raw_text: str) -> dict[str, tuple[int, int]]:
    """
    Detect IMRaD section boundaries within the raw text.

    Returns a dict mapping section name to (start_offset, end_offset) tuples.
    The end_offset of a section is the start_offset of the next section.
    Returns an empty dict if fewer than 3 distinct sections are reliably detected.
    """
    matches: list[tuple[str, int]] = []
    for section_name, pattern in SECTION_PATTERNS.items():
        for match in pattern.finditer(raw_text):
            matches.append((section_name, match.start()))

    if len(matches) < 3:
        return {}

    matches.sort(key=lambda pair: pair[1])

    seen: set[str] = set()
    deduped: list[tuple[str, int]] = []
    for name, start in matches:
        if name not in seen:
            deduped.append((name, start))
            seen.add(name)

    if len(deduped) < 3:
        return {}

    boundaries: dict[str, tuple[int, int]] = {}
    for index, (name, start) in enumerate(deduped):
        end = deduped[index + 1][1] if index + 1 < len(deduped) else len(raw_text)
        boundaries[name] = (start, end)
    return boundaries


def build_extraction_text(
    raw_text: str,
    sections: dict[str, tuple[int, int]],
) -> tuple[str, bool]:
    """
    Build the text payload that will be sent to the LLM.

    If three or more IMRaD sections are detected, concatenate the relevant
    sections (abstract through conclusion) and exclude references. Otherwise,
    fall back to the full raw text with the references tail removed if
    detectable.

    Returns (extraction_text, used_smart_sections).
    """
    informative_keys = ("abstract", "introduction", "methods", "results", "discussion", "conclusion")

    if sections:
        parts: list[str] = []
        for key in informative_keys:
            if key in sections:
                start, end = sections[key]
                parts.append(raw_text[start:end].strip())
        if parts:
            return "\n\n".join(parts), True

    references_match = SECTION_PATTERNS["references"].search(raw_text)
    if references_match:
        return raw_text[: references_match.start()].strip(), False
    return raw_text.strip(), False


def estimate_token_count(text: str) -> int:
    """
    Rough estimate of token count using a 4-characters-per-token heuristic.

    Accurate enough for batch budgeting; replace with tiktoken for precise
    accounting once a model is fixed.
    """
    return max(1, len(text) // 4)


def process_pdf(pdf_path: Path, source_database: str) -> ProcessedPDF:
    """
    Run the full PDF processing pipeline on a single file.

    Steps: open and extract raw text, detect IMRaD sections, build the
    extraction text payload, and estimate token usage. All errors are
    collected on the returned object; this function does not raise on
    parse failures.
    """
    raw_text, page_count, errors = extract_raw_text(pdf_path)
    sections_with_bounds = detect_sections(raw_text)
    extraction_text, used_smart = build_extraction_text(raw_text, sections_with_bounds)

    sections_text: dict[str, str] = {}
    for name, (start, end) in sections_with_bounds.items():
        sections_text[name] = raw_text[start:end].strip()

    return ProcessedPDF(
        file_path=pdf_path,
        file_name=pdf_path.name,
        source_database=source_database,
        total_pages=page_count,
        raw_text=raw_text,
        sections=sections_text,
        used_smart_sections=used_smart,
        extraction_text=extraction_text,
        estimated_tokens=estimate_token_count(extraction_text),
        parse_errors=errors,
    )


def discover_pdfs(raw_pdfs_root: Path) -> list[tuple[Path, str]]:
    """
    Walk the raw_pdfs root directory and return all PDFs paired with their
    source database name (the immediate parent folder name).

    Expected layout:
        raw_pdfs_root/
            wos/*.pdf
            scopus/*.pdf
            oecd/*.pdf
            iea/*.pdf
    """
    valid_sources = {"wos", "scopus", "oecd", "iea"}
    discovered: list[tuple[Path, str]] = []
    for source_dir in raw_pdfs_root.iterdir():
        if not source_dir.is_dir():
            continue
        source_name = source_dir.name.lower()
        if source_name not in valid_sources:
            continue
        for pdf_file in source_dir.rglob("*.pdf"):
            discovered.append((pdf_file, source_name))
    return discovered


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    raw_pdfs_root = project_root / "data" / "raw_pdfs"

    pdfs = discover_pdfs(raw_pdfs_root)
    print(f"Discovered {len(pdfs)} PDFs across all source databases.")

    if not pdfs:
        print("No PDFs found. Place files under data/raw_pdfs/{wos,scopus,oecd,iea}/ to test.")
    else:
        sample_path, sample_source = pdfs[0]
        print(f"\nProcessing sample: {sample_path.name} (source: {sample_source})")
        result = process_pdf(sample_path, sample_source)
        print(f"\nProcessing summary: {result.to_dict()}")
        preview_chars = 500
        print(f"\nExtraction text preview (first {preview_chars} chars):")
        print(result.extraction_text[:preview_chars])
        print(f"\n...")
        print(f"\nTotal extraction text length: {len(result.extraction_text)} chars")
        print(f"Estimated tokens: {result.estimated_tokens}")
        if result.parse_errors:
            print(f"\nParse errors encountered: {result.parse_errors}")