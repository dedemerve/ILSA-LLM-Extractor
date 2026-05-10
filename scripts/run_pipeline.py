#!/usr/bin/env python3
"""
ILSA Literature Pipeline - Enhanced with checkpoint and cost tracking.
"""

import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import fitz  # PyMuPDF

# Allow imports from ilsa_pipeline/ when running from scripts/
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ilsa_pipeline"))

from extractors.gpt_extractor import GPTExtractor
from utils.storage import StorageManager, save_json


@dataclass
class ProcessedPDF:
    file_name: str
    extracted_text: str
    extraction_method: str = "pymupdf"
    sections_found: list[str] = field(default_factory=list)
    error: str | None = None


def setup_logging(output_dir: Path) -> logging.Logger:
    log_file = output_dir / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger(__name__)


def get_processed_files(json_dir: Path) -> set[str]:
    """Return stems of PDFs that already have a JSON output."""
    if not json_dir.exists():
        return set()
    return {f.stem for f in json_dir.glob("*.json")}


def read_pdf(pdf_path: Path) -> ProcessedPDF:
    """Extract full text from a PDF using PyMuPDF."""
    try:
        doc = fitz.open(pdf_path)
        text = "\n\n".join(page.get_text() for page in doc)
        doc.close()
        if not text.strip():
            return ProcessedPDF(file_name=pdf_path.name, extracted_text="",
                                error="PDF produced no extractable text")
        return ProcessedPDF(file_name=pdf_path.name, extracted_text=text)
    except Exception as e:
        return ProcessedPDF(file_name=pdf_path.name, extracted_text="", error=str(e))


def main():
    project_root = Path(__file__).resolve().parents[1]
    output_dir = project_root / "outputs"
    json_dir = output_dir / "json"
    output_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(output_dir)
    logger.info("=" * 60)
    logger.info("ILSA AI Literature Pipeline")
    logger.info("=" * 60)

    raw_pdfs_root = project_root / "data" / "raw_pdfs"
    if not raw_pdfs_root.exists():
        logger.error(f"PDF directory not found: {raw_pdfs_root}")
        return

    processed_files = get_processed_files(json_dir)
    all_pdfs = list(raw_pdfs_root.rglob("*.pdf"))
    remaining = [p for p in all_pdfs if p.stem not in processed_files]
    logger.info(f"Found {len(all_pdfs)} PDFs — {len(processed_files)} already done, "
                f"{len(remaining)} to process")

    extractor = GPTExtractor()
    storage = StorageManager(output_dir / "articles.db")

    total_cost = 0.0
    success_count = 0
    fail_count = 0

    for pdf_path in remaining:
        logger.info(f"Processing: {pdf_path.name}")

        processed = read_pdf(pdf_path)
        result = extractor.extract(processed)

        save_json(result, json_dir)
        total_cost += result.cost_usd

        if result.success:
            try:
                storage.insert_article(result)
                success_count += 1
                logger.info(f"  Saved: {pdf_path.name} (${result.cost_usd:.4f})")
            except Exception as e:
                fail_count += 1
                logger.error(f"  DB insert failed for {pdf_path.name}: {e}")
        else:
            fail_count += 1
            logger.error(f"  Extraction failed for {pdf_path.name}: {result.error}")

    storage.close()

    logger.info("=" * 60)
    logger.info(f"Done — {success_count} saved, {fail_count} failed, "
                f"total cost: ${total_cost:.4f}")
    logger.info("=" * 60)

    cost_report = {
        "total_cost_usd": round(total_cost, 6),
        "pdfs_processed": success_count,
        "pdfs_failed": fail_count,
        "timestamp": datetime.now().isoformat(),
    }
    (output_dir / "cost_report.json").write_text(
        __import__("json").dumps(cost_report, indent=2)
    )


if __name__ == "__main__":
    main()
