"""Extract articles 1, 3, 5, 7, 9 from ~/Desktop/articles/ into outputs/articles/json/."""

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
TARGET_NUMS = ("1", "3", "5", "7", "9")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    targets = []
    for p in sorted(ARTICLES_DIR.glob("*.pdf")):
        num = p.name.split(".")[0].strip()
        if num in TARGET_NUMS:
            targets.append(p)

    print(f"Processing {len(targets)} PDFs → {OUTPUT_DIR}\n")
    for i, p in enumerate(targets, 1):
        print(f"  {i}. {p.name[:75]}")

    extractor = GPTExtractor()
    total_cost = 0.0
    ok = 0

    for i, pdf_path in enumerate(targets, 1):
        print(f"\n{'='*70}")
        print(f"[{i}/{len(targets)}] {pdf_path.name[:70]}")
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
        ok += 1

        output = result.extraction.model_dump(mode="json")
        safe_name = pdf_path.stem[:80].replace("/", "_").replace("\\", "_")
        out_path = OUTPUT_DIR / f"{safe_name}.json"
        out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")

        d = output["data"]
        meta = output["metadata"]
        conf = d["confounders_identified"]
        na = sum(1 for c in conf if str(c.get("variable_code", "")).upper() in ("N/A", "NA"))

        print(f"  Duration: {elapsed:.1f}s | Cost: ${result.cost_usd:.4f}")
        print(f"  Title: {(meta.get('title') or 'N/A')[:75]}")
        print(f"  DOI: {meta.get('doi')}")
        print(f"  Primary ML: {d['ml_techniques']['primary']}")
        print(f"  Confounders: {len(conf)} | N/A codes: {na}")
        print(f"  N: {d['sample_details']['total_students']}")
        print(f"  Saved: {out_path.name}", flush=True)

    print(f"\n{'='*70}")
    print(f"DONE — {ok}/{len(targets)} ok | ${total_cost:.4f}")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
