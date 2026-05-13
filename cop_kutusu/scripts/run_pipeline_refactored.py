#!/usr/bin/env python3
"""
ILSA Literature Pipeline - Refactored (v2.0)
=============================================
Production-ready extraction pipeline with anti-hallucination constraints,
strict typing, explainability layers, and optimized document handling.

This script:
1. Loads PDFs using PyMuPDF (efficient, low-dependency)
2. Truncates context to first 15 pages (prevents nano LLM degradation)
3. Calls Gemini 2.5-Flash (or similar nano LLM) with anti-hallucination prompt
4. Validates via Pydantic with Optional fields (graceful error handling)
5. Saves extraction results as JSON with full provenance
6. Logs quality metrics and processing statistics

Requirements:
    pip install pydantic google-generativeai pymupdf

Environment Variables:
    GOOGLE_API_KEY: Your Gemini API key
    ILSA_OUTPUT_DIR: Override output directory (default: ./outputs)
    MAX_CONTEXT_PAGES: Override page truncation (default: 15)
"""

import json
import logging
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

try:
    import fitz  # PyMuPDF
except ImportError:
    print("Error: PyMuPDF not installed. Install with: pip install pymupdf")
    sys.exit(1)

from src.extractors.gpt_extractor_refactored import ILSAPaperExtractor, ExtractionResult
from src.schemas.article_schema_refactored import PaperExtraction


# ============================================================================
# CONFIGURATION & LOGGING
# ============================================================================

@dataclass
class PipelineConfig:
    """Pipeline configuration with sensible defaults."""
    output_dir: Path = Path("./outputs")
    json_dir: Path = field(init=False)
    logs_dir: Path = field(init=False)
    model_name: str = "gemini-2.5-flash"  # Nano LLM
    max_context_pages: int = 15
    skip_processed: bool = True
    dry_run: bool = False
    
    def __post_init__(self):
        self.output_dir = Path(self.output_dir)
        self.json_dir = self.output_dir / "json"
        self.logs_dir = self.output_dir / "logs"
        self.json_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)


def setup_logging(config: PipelineConfig) -> logging.Logger:
    """
    Configure logging to both file and console.
    
    Logs include:
    - Processing status (started, completed, failed)
    - Extraction quality metrics
    - Performance statistics
    """
    log_file = config.logs_dir / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # File handler
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # Root logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logging.getLogger(__name__)


# ============================================================================
# PDF PROCESSING
# ============================================================================

@dataclass
class ProcessedPDF:
    """Result of PDF text extraction."""
    file_name: str
    file_path: Path
    extracted_text: str
    num_pages: int = 0
    char_count: int = 0
    error: Optional[str] = None
    
    @property
    def success(self) -> bool:
        return self.error is None and len(self.extracted_text.strip()) > 0


def read_pdf_pymupdf(pdf_path: Path) -> ProcessedPDF:
    """
    Extract full text from PDF using PyMuPDF (fitz).
    
    PyMuPDF is:
    - Fast and efficient
    - Handles complex PDFs well
    - Low dependency footprint
    - Better OCR quality than PyPDF2 for scanned documents
    
    Args:
        pdf_path: Path to PDF file
    
    Returns:
        ProcessedPDF with extracted text or error message
    """
    try:
        doc = fitz.open(pdf_path)
        
        # Extract text with preserved structure
        text_parts = []
        for page_num, page in enumerate(doc):
            # page.get_text() returns text with layout preserved
            page_text = page.get_text()
            text_parts.append(f"[Page {page_num + 1}]\n{page_text}")
        
        doc.close()
        
        full_text = "\n\n".join(text_parts)
        
        if not full_text.strip():
            return ProcessedPDF(
                file_name=pdf_path.name,
                file_path=pdf_path,
                extracted_text="",
                num_pages=len(doc),
                error="PDF produced no extractable text (possibly encrypted or blank)"
            )
        
        return ProcessedPDF(
            file_name=pdf_path.name,
            file_path=pdf_path,
            extracted_text=full_text,
            num_pages=len(doc),
            char_count=len(full_text)
        )
    
    except Exception as e:
        return ProcessedPDF(
            file_name=pdf_path.name,
            file_path=pdf_path,
            extracted_text="",
            error=f"PDF extraction failed: {type(e).__name__}: {str(e)}"
        )


# ============================================================================
# EXTRACTION & STORAGE
# ============================================================================

def save_extraction_json(
    result: ExtractionResult,
    output_dir: Path,
    pdf_path: Path,
) -> Optional[Path]:
    """
    Save extraction result to JSON file.
    
    Includes:
    - Extracted paper metadata
    - Processing metadata (timestamp, model, quality flags)
    - Raw LLM response (first 1000 chars for debugging)
    - Processing notes
    
    Args:
        result: ExtractionResult from the pipeline
        output_dir: Directory to save JSON files
        pdf_path: Original PDF path (for filename mapping)
    
    Returns:
        Path to saved JSON file, or None if save failed
    """
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine output filename (same stem as PDF)
        json_path = output_dir / f"{pdf_path.stem}.json"
        
        # Build output structure
        output_data = {
            "extraction_status": "success" if result.success else "failed",
            "extraction_timestamp": datetime.utcnow().isoformat() + "Z",
            "model_used": result.response.model_used if result.response else None,
            "input_pdf": str(pdf_path),
            "output_json": str(json_path),
        }
        
        if result.success:
            output_data["paper"] = result.paper.model_dump(exclude_none=False, mode="json")
            if result.processing_notes:
                output_data["processing_notes"] = result.processing_notes
        else:
            output_data["error"] = result.error
            if result.processing_notes:
                output_data["processing_notes"] = result.processing_notes
            # Include first 500 chars of raw response for debugging
            if result.raw_response:
                output_data["raw_response_sample"] = result.raw_response[:500]
        
        # Write to file
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        return json_path
    
    except Exception as e:
        logging.error(f"Failed to save JSON for {pdf_path.name}: {str(e)}")
        return None


def get_processed_files(json_dir: Path) -> set[str]:
    """
    Return set of PDF stems that already have JSON output.
    Allows resuming interrupted pipelines without re-processing.
    
    Args:
        json_dir: Directory containing JSON outputs
    
    Returns:
        Set of stems (filenames without extension) of processed PDFs
    """
    if not json_dir.exists():
        return set()
    return {f.stem for f in json_dir.glob("*.json")}


# ============================================================================
# STATISTICS & REPORTING
# ============================================================================

@dataclass
class PipelineStatistics:
    """Track pipeline execution metrics."""
    total_pdfs: int = 0
    processed: int = 0
    successful: int = 0
    failed: int = 0
    skipped: int = 0
    start_time: datetime = field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None
    
    @property
    def elapsed_seconds(self) -> float:
        end = self.end_time or datetime.utcnow()
        return (end - self.start_time).total_seconds()
    
    @property
    def success_rate(self) -> float:
        if self.processed == 0:
            return 0.0
        return (self.successful / self.processed) * 100
    
    def log_summary(self, logger: logging.Logger):
        """Print summary statistics."""
        logger.info("=" * 70)
        logger.info("PIPELINE EXECUTION SUMMARY")
        logger.info("=" * 70)
        logger.info(f"Total PDFs found:        {self.total_pdfs}")
        logger.info(f"Processed:               {self.processed}")
        logger.info(f"Successful:              {self.successful}")
        logger.info(f"Failed:                  {self.failed}")
        logger.info(f"Skipped (already done):  {self.skipped}")
        logger.info(f"Success rate:            {self.success_rate:.1f}%")
        logger.info(f"Elapsed time:            {self.elapsed_seconds:.1f}s")
        logger.info(f"Avg time per PDF:        {self.elapsed_seconds / max(self.processed, 1):.1f}s")
        logger.info("=" * 70)


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main(
    pdf_root: Optional[Path] = None,
    config: Optional[PipelineConfig] = None,
) -> int:
    """
    Main pipeline execution.
    
    Args:
        pdf_root: Root directory containing PDF files (searches recursively)
        config: Pipeline configuration; uses defaults if None
    
    Returns:
        Exit code (0 for success, 1 for errors)
    """
    if config is None:
        config = PipelineConfig()
    
    logger = setup_logging(config)
    stats = PipelineStatistics()
    
    logger.info("=" * 70)
    logger.info("ILSA LITERATURE PIPELINE v2.0 (REFACTORED)")
    logger.info("=" * 70)
    logger.info(f"Model:                   {config.model_name}")
    logger.info(f"Output directory:        {config.output_dir}")
    logger.info(f"Max context pages:       {config.max_context_pages}")
    logger.info(f"Skip already processed:  {config.skip_processed}")
    logger.info(f"Dry run mode:            {config.dry_run}")
    logger.info("=" * 70)
    
    # Determine PDF root
    if pdf_root is None:
        project_root = Path(__file__).resolve().parents[2]  # Up from scripts/ to project root
        pdf_root = project_root / "data" / "raw_pdfs"
    
    if not pdf_root.exists():
        logger.error(f"PDF directory not found: {pdf_root}")
        return 1
    
    logger.info(f"Searching for PDFs in: {pdf_root}")
    
    # Find all PDFs
    all_pdfs = list(pdf_root.rglob("*.pdf"))
    if not all_pdfs:
        logger.warning(f"No PDF files found in {pdf_root}")
        return 0
    
    stats.total_pdfs = len(all_pdfs)
    
    # Filter already-processed PDFs
    processed_stems = get_processed_files(config.json_dir) if config.skip_processed else set()
    remaining_pdfs = [
        p for p in all_pdfs
        if p.stem not in processed_stems
    ]
    
    stats.skipped = len(all_pdfs) - len(remaining_pdfs)
    
    logger.info(f"Found {len(all_pdfs)} total PDFs")
    logger.info(f"Skipping {stats.skipped} already processed")
    logger.info(f"Will process {len(remaining_pdfs)} new PDFs")
    
    if config.dry_run:
        logger.warning("DRY RUN MODE: No extractions will be performed")
        for pdf_path in remaining_pdfs[:5]:  # Show first 5
            logger.info(f"  [DRY] Would process: {pdf_path.name}")
        if len(remaining_pdfs) > 5:
            logger.info(f"  [DRY] ... and {len(remaining_pdfs) - 5} more")
        return 0
    
    # Initialize extractor
    try:
        extractor = ILSAPaperExtractor(
            model=config.model_name,
            temperature=0.0,  # Deterministic extraction
        )
    except Exception as e:
        logger.error(f"Failed to initialize extractor: {str(e)}")
        logger.error("Check API key and model availability")
        return 1
    
    # Process PDFs
    for idx, pdf_path in enumerate(remaining_pdfs, 1):
        logger.info(f"\n[{idx}/{len(remaining_pdfs)}] Processing: {pdf_path.name}")
        
        # Step 1: Extract text from PDF
        processed_pdf = read_pdf_pymupdf(pdf_path)
        
        if not processed_pdf.success:
            logger.error(f"  ✗ Failed to extract PDF text: {processed_pdf.error}")
            stats.failed += 1
            continue
        
        logger.info(f"  ✓ Extracted {processed_pdf.num_pages} pages ({processed_pdf.char_count} chars)")
        
        # Step 2: Call LLM for metadata extraction
        try:
            result = extractor.extract_from_pdf(
                pdf_text=processed_pdf.extracted_text,
                file_name=processed_pdf.file_name,
                max_context_pages=config.max_context_pages,
            )
        except Exception as e:
            logger.error(f"  ✗ LLM extraction failed: {str(e)}")
            stats.failed += 1
            continue
        
        # Step 3: Log extraction result
        if result.success:
            logger.info(f"  ✓ Extraction successful")
            
            # Log key extracted fields
            if result.paper.reasoning_scratchpad:
                logger.debug(f"    Reasoning: {result.paper.reasoning_scratchpad}")
            
            if result.paper.research_design_type:
                logger.info(f"    Research design: {result.paper.research_design_type}")
            
            if result.paper.missing_data_handling:
                logger.info(f"    Missing data handling: {result.paper.missing_data_handling}")
            
            if result.paper.needs_human_review:
                logger.warning(f"    ⚠ Flagged for human review")
            
            stats.successful += 1
        else:
            logger.error(f"  ✗ Extraction failed: {result.error}")
            if result.raw_response:
                logger.debug(f"    LLM response (first 200 chars): {result.raw_response[:200]}")
            stats.failed += 1
            continue
        
        # Step 4: Save result to JSON
        json_path = save_extraction_json(result, config.json_dir, pdf_path)
        if json_path:
            logger.info(f"  ✓ Saved to: {json_path.name}")
        else:
            logger.error(f"  ✗ Failed to save extraction JSON")
            stats.failed += 1
            continue
        
        # If processing notes exist, log them
        if result.processing_notes:
            logger.debug(f"    Notes: {result.processing_notes}")
        
        stats.processed += 1
    
    # Print summary
    stats.end_time = datetime.utcnow()
    stats.log_summary(logger)
    
    return 0 if stats.failed == 0 else 1


if __name__ == "__main__":
    import os
    
    # Check for API key
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: Please set GOOGLE_API_KEY or OPENAI_API_KEY environment variable")
        print("  export GOOGLE_API_KEY=your_key_here")
        print("  or")
        print("  export OPENAI_API_KEY=your_key_here")
        sys.exit(1)
    
    # Build config from environment or defaults
    config = PipelineConfig(
        output_dir=Path(os.getenv("ILSA_OUTPUT_DIR", "./outputs")),
        model_name=os.getenv("ILSA_MODEL", "gemini-2.5-flash"),
        max_context_pages=int(os.getenv("MAX_CONTEXT_PAGES", "15")),
        skip_processed=os.getenv("ILSA_SKIP_PROCESSED", "true").lower() == "true",
        dry_run=os.getenv("ILSA_DRY_RUN", "false").lower() == "true",
    )
    
    exit_code = main(config=config)
    sys.exit(exit_code)
