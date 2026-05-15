"""Re-extract article 1 (Lee & Lee) into json_v3."""

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
    pdf_path = next(p for p in ARTICLES_DIR.glob("*.pdf") if p.name.startswith("1."))
    extractor = GPTExtractor()
    processed = process_pdf(pdf_path, source_database="articles")
    result = extractor.extract(processed)

    if not result.success:
        print(f"FAILED: {result.error}")
        return

    output = result.extraction.model_dump(mode="json")
    out_path = OUTPUT_DIR / f"{pdf_path.stem[:80]}.json"
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")

    conf = output["data"]["confounders_identified"]
    print(f"Saved: {out_path}")
    for c in conf:
        print(f"  [{c['category']}] {c['variable_code']} → {c['variable_name']}")
    na = sum(1 for c in conf if c["variable_code"].upper() in ("N/A", "NA"))
    print(f"N/A count: {na}")


if __name__ == "__main__":
    main()
