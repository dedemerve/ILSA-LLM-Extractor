"""Quick test: extract 2 articles to verify structured confounders."""

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
        if num in ("6", "8"):
            targets.append(p)

    print(f"Testing structured confounders with {len(targets)} PDFs\n")
    extractor = GPTExtractor()

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

        output = result.extraction.model_dump(mode="json")
        safe_name = pdf_path.stem[:80]
        out_path = OUTPUT_DIR / f"{safe_name}.json"
        out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")

        d = output["data"]
        conf = d["confounders_identified"]
        print(f"  Duration: {elapsed:.1f}s | Cost: ${result.cost_usd:.4f}")
        print(f"  Confounders ({len(conf)}):")

        cats = {}
        for c in conf:
            cat = c["category"]
            cats[cat] = cats.get(cat, 0) + 1
            code_str = c["variable_code"] or "(custom)"
            print(f"    [{cat:20s}] {code_str:20s} → {c['variable_name']}")

        print(f"\n  Category distribution:")
        for cat, count in sorted(cats.items(), key=lambda x: -x[1]):
            print(f"    {cat}: {count}")
        print(f"\n  Saved: {out_path.name}", flush=True)

    print(f"\n{'='*70}")
    print("DONE")


if __name__ == "__main__":
    main()
