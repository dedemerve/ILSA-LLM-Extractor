"""
Extract 5 papers from data/ through the full pipeline and save JSON outputs.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from src.extractors.pdf_processor import process_pdf
from src.extractors.gpt_extractor import GPTExtractor

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "deneme2" / "json"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(DATA_DIR.glob("*.pdf"))[:5]
    if not pdfs:
        print("No PDFs found in data/ directory.")
        return

    print(f"Found {len(pdfs)} PDFs to process:\n")
    for i, p in enumerate(pdfs, 1):
        print(f"  {i}. {p.name}")

    extractor = GPTExtractor()
    total_cost = 0.0
    total_time = 0.0

    for i, pdf_path in enumerate(pdfs, 1):
        print(f"\n{'='*70}")
        print(f"[{i}/{len(pdfs)}] Processing: {pdf_path.name}")
        print(f"{'='*70}")

        t0 = time.perf_counter()
        processed = process_pdf(pdf_path, source_database="manual_test")

        if not processed.extraction_text:
            print(f"  SKIP: No text extracted ({processed.parse_errors})")
            continue

        print(f"  Pages: {processed.total_pages}")
        print(f"  Tokens (est): {processed.estimated_tokens}")
        print(f"  Sections: {', '.join(processed.sections.keys()) or 'none'}")

        result = extractor.extract(processed)
        elapsed = time.perf_counter() - t0

        if not result.success:
            print(f"  FAILED: {result.error}")
            continue

        total_cost += result.cost_usd
        total_time += elapsed

        output = result.extraction.model_dump(mode="json")
        stem = pdf_path.stem
        safe_name = stem[:80].replace("/", "_").replace("\\", "_")
        out_path = OUTPUT_DIR / f"{safe_name}.json"
        out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")

        print(f"  Duration: {elapsed:.1f}s")
        print(f"  Tokens: {result.input_tokens} in / {result.output_tokens} out")
        print(f"  Cost: ${result.cost_usd:.4f}")
        print(f"  Primary ML: {output['data']['ml_techniques']['primary']}")
        print(f"  Design: {output['data']['research_design_type']}")
        print(f"  PV handling: {output['data']['plausible_values_handling']}")
        print(f"  Missing data: {output['data']['missing_data_handling']}")
        print(f"  Weights: {output['data']['survey_design']['student_weights_used']}")
        print(f"  Saved: {out_path.name}")

    print(f"\n{'='*70}")
    print(f"DONE — {len(pdfs)} papers processed")
    print(f"Total cost: ${total_cost:.4f}")
    print(f"Total time: {total_time:.1f}s")
    print(f"Output dir: {OUTPUT_DIR}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
