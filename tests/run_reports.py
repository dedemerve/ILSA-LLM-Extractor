"""Extract PDFs from ~/Desktop/reports/ and save JSON outputs."""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from src.extractors.pdf_processor import process_pdf
from src.extractors.gpt_extractor import GPTExtractor

REPORTS_DIR = Path.home() / "Desktop" / "reports"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "reports" / "json"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(REPORTS_DIR.glob("*.pdf"))
    if not pdfs:
        print("No PDFs found in ~/Desktop/reports/")
        return

    print(f"Found {len(pdfs)} PDFs in {REPORTS_DIR}:\n")
    for i, p in enumerate(pdfs, 1):
        print(f"  {i}. {p.name}")

    extractor = GPTExtractor()
    total_cost = 0.0
    total_time = 0.0
    success_count = 0

    for i, pdf_path in enumerate(pdfs, 1):
        print(f"\n{'='*70}")
        print(f"[{i}/{len(pdfs)}] Processing: {pdf_path.name}")
        print("=" * 70, flush=True)

        t0 = time.perf_counter()
        processed = process_pdf(pdf_path, source_database="reports")

        if not processed.extraction_text:
            print(f"  SKIP: No text extracted ({processed.parse_errors})")
            continue

        print(f"  Pages: {processed.total_pages}")
        print(f"  Tokens (est): {processed.estimated_tokens}")
        print(f"  Sections: {', '.join(processed.sections.keys()) or 'none'}", flush=True)

        result = extractor.extract(processed)
        elapsed = time.perf_counter() - t0

        if not result.success:
            print(f"  FAILED: {result.error}")
            continue

        total_cost += result.cost_usd
        total_time += elapsed
        success_count += 1

        output = result.extraction.model_dump(mode="json")
        safe_name = pdf_path.stem[:80]
        out_path = OUTPUT_DIR / f"{safe_name}.json"
        out_path.write_text(
            json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        d = output["data"]
        sd = d["survey_design"]
        ml = d["ml_techniques"]
        meta = output["metadata"]
        conf = d["confounders_identified"]
        wfi = sd["weight_fields_interpretation"] or ""

        print(f"  Duration: {elapsed:.1f}s")
        print(f"  Tokens: {result.input_tokens} in / {result.output_tokens} out")
        print(f"  Cost: ${result.cost_usd:.4f}")
        print(f"  Title: {(meta.get('title') or 'N/A')[:80]}")
        print(f"  DOI: {meta.get('doi')}")
        print(f"  Primary ML: {ml['primary']}")
        print(f"  All ML: {ml['all_techniques']}")
        print(f"  Design: {d['research_design_type']}")
        print(f"  PV handling: {d['plausible_values_handling']}")
        print(f"  Missing data: {d['missing_data_handling']}")
        print(f"  Weights: {sd['student_weights_used']}")
        print(f"  Weight summary: {wfi[:160]}...")
        print(f"  Confounders ({len(conf)}): {[c[:50] for c in conf[:3]]}...")
        print(f"  Total students: {d['sample_details']['total_students']}")
        countries = d["sample_details"]["countries"]
        print(f"  Countries ({len(countries)}): {[(c['country_code'], c['n_students']) for c in countries[:5]]}")
        print(f"  Saved: {out_path.name}", flush=True)

    print(f"\n{'='*70}")
    print(f"DONE -- {success_count}/{len(pdfs)} papers processed successfully")
    print(f"Total cost: ${total_cost:.4f}")
    print(f"Total time: {total_time:.1f}s")
    print(f"Output dir: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
