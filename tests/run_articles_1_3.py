"""Extract articles 1, 2, 3 from ~/Desktop/articles/ into json_v3."""

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
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "articles" / "json_v3"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    targets = []
    for p in sorted(ARTICLES_DIR.glob("*.pdf")):
        num = p.name.split(".")[0].strip()
        if num in ("1", "2", "3"):
            targets.append(p)

    print(f"Processing {len(targets)} PDFs → {OUTPUT_DIR}\n")
    extractor = GPTExtractor()
    total_cost = 0.0

    for i, pdf_path in enumerate(targets, 1):
        print(f"{'='*70}")
        print(f"[{i}/{len(targets)}] {pdf_path.name[:70]}")
        print("=" * 70, flush=True)

        t0 = time.perf_counter()
        processed = process_pdf(pdf_path, source_database="articles")
        if not processed.extraction_text:
            print(f"  SKIP: No text")
            continue

        result = extractor.extract(processed)
        elapsed = time.perf_counter() - t0

        if not result.success:
            print(f"  FAILED: {result.error}")
            continue

        total_cost += result.cost_usd
        output = result.extraction.model_dump(mode="json")
        safe_name = pdf_path.stem[:80]
        out_path = OUTPUT_DIR / f"{safe_name}.json"
        out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")

        d = output["data"]
        conf = d["confounders_identified"]
        meta = output["metadata"]

        print(f"  Duration: {elapsed:.1f}s | Cost: ${result.cost_usd:.4f}")
        print(f"  Title: {(meta.get('title') or 'N/A')[:75]}")
        print(f"  DOI: {meta.get('doi')}")
        print(f"  Confounders ({len(conf)}):")

        cats = {}
        for c in conf:
            cat = c["category"]
            cats[cat] = cats.get(cat, 0) + 1
            print(f"    [{cat:20s}] {c['variable_code']:20s} → {c['variable_name']}")

        print(f"\n  Category distribution:")
        for cat, count in sorted(cats.items(), key=lambda x: -x[1]):
            print(f"    {cat}: {count}")
        print(f"\n  Saved: {out_path.name}", flush=True)

    print(f"\n{'='*70}")
    print(f"DONE — {len(targets)} papers | ${total_cost:.4f}")


if __name__ == "__main__":
    main()
