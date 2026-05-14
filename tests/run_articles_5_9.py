"""Extract articles 5-9 from ~/Desktop/articles/ and save JSON outputs."""

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

    targets = []
    for p in sorted(ARTICLES_DIR.glob("*.pdf")):
        num = p.name.split(".")[0].strip()
        if num in ("5", "6", "7", "8", "9"):
            targets.append(p)

    print(f"Found {len(targets)} PDFs to process:\n")
    for i, p in enumerate(targets, 1):
        print(f"  {i}. {p.name}")

    extractor = GPTExtractor()
    total_cost = 0.0
    total_time = 0.0

    for i, pdf_path in enumerate(targets, 1):
        print(f"\n{'='*70}")
        print(f"[{i}/{len(targets)}] Processing: {pdf_path.name}")
        print("=" * 70, flush=True)

        t0 = time.perf_counter()
        processed = process_pdf(pdf_path, source_database="articles")

        if not processed.extraction_text:
            print(f"  SKIP: No text ({processed.parse_errors})")
            continue

        print(f"  Pages: {processed.total_pages}, Tokens: {processed.estimated_tokens}", flush=True)

        result = extractor.extract(processed)
        elapsed = time.perf_counter() - t0

        if not result.success:
            print(f"  FAILED: {result.error}")
            continue

        total_cost += result.cost_usd
        total_time += elapsed

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

        print(f"  Duration: {elapsed:.1f}s | Cost: ${result.cost_usd:.4f}")
        print(f"  Title: {(meta.get('title') or 'N/A')[:75]}")
        print(f"  DOI: {meta.get('doi')}")
        print(f"  Primary ML: {ml['primary']} | All: {ml['all_techniques']}")
        print(f"  Design: {d['research_design_type']}")
        print(f"  PV: {d['plausible_values_handling']} | Missing: {d['missing_data_handling']}")
        print(f"  Not-reported expl: {(hnre[:150] + '...') if hnre else '(null)'}")
        print(f"  Weights: {sd['student_weights_used']}")
        print(f"  Weight summary: {wfi[:150]}...")
        print(f"  Confounders ({len(conf)}): {[c[:45] for c in conf[:3]]}...")
        print(f"  N: {d['sample_details']['total_students']}")
        countries = d["sample_details"]["countries"]
        print(f"  Countries ({len(countries)}): {[(c['country_code'], c['n_students']) for c in countries[:5]]}")
        print(f"  Saved: {out_path.name}", flush=True)

    print(f"\n{'='*70}")
    print(f"DONE -- {len(targets)} papers | ${total_cost:.4f} | {total_time:.1f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
