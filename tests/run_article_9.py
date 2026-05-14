"""Re-extract article 9 (Alkan et al.) which timed out earlier."""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from src.extractors.pdf_processor import process_pdf
from src.extractors.gpt_extractor import GPTExtractor

ARTICLES_DIR = Path.home() / "Desktop" / "articles"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "articles" / "json"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pdf_path = None
    for p in sorted(ARTICLES_DIR.glob("*.pdf")):
        if p.name.startswith("9."):
            pdf_path = p
            break

    if not pdf_path:
        print("Article 9 not found!")
        return

    print(f"Processing: {pdf_path.name}", flush=True)

    extractor = GPTExtractor()
    t0 = time.perf_counter()
    processed = process_pdf(pdf_path, source_database="articles")

    if not processed.extraction_text:
        print(f"SKIP: No text ({processed.parse_errors})")
        return

    print(f"Pages: {processed.total_pages}, Tokens: {processed.estimated_tokens}", flush=True)

    result = extractor.extract(processed)
    elapsed = time.perf_counter() - t0

    if not result.success:
        print(f"FAILED: {result.error}")
        return

    output = result.extraction.model_dump(mode="json")
    safe_name = pdf_path.stem[:80].replace("/", "_").replace("\\", "_")
    out_path = OUTPUT_DIR / f"{safe_name}.json"
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")

    d = output["data"]
    sd = d["survey_design"]
    ml = d["ml_techniques"]
    meta = output["metadata"]
    conf = d["confounders_identified"]
    wfi = sd["weight_fields_interpretation"] or ""
    hnre = d.get("handling_not_reported_explanation")

    print(f"\nDuration: {elapsed:.1f}s | Cost: ${result.cost_usd:.4f}")
    print(f"Title: {(meta.get('title') or 'N/A')[:80]}")
    print(f"DOI: {meta.get('doi')}")
    print(f"Primary ML: {ml['primary']} | All: {ml['all_techniques']}")
    print(f"Design: {d['research_design_type']}")
    print(f"PV: {d['plausible_values_handling']} | Missing: {d['missing_data_handling']}")
    print(f"Not-reported expl: {(hnre[:200] + '...') if hnre else '(null)'}")
    print(f"Weights: {sd['student_weights_used']}")
    print(f"Weight summary: {wfi[:200]}...")
    print(f"Confounders ({len(conf)}): {[c[:50] for c in conf[:5]]}...")
    print(f"N: {d['sample_details']['total_students']}")
    countries = d["sample_details"]["countries"]
    print(f"Countries ({len(countries)}): {[(c['country_code'], c['n_students']) for c in countries[:6]]}")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
