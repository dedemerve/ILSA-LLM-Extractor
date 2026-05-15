"""Extract articles 3, 5, 6, 8, 10 from ~/Desktop/articles/ → outputs/articles/json/."""

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
TARGET_NUMS = ("3", "5", "6", "8", "10")


def _article_num(pdf_path: Path) -> str:
    return pdf_path.name.split(".")[0].strip()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(
        p for p in ARTICLES_DIR.glob("*.pdf")
        if _article_num(p) in TARGET_NUMS
    )
    if not pdfs:
        print(f"No PDFs found for articles {TARGET_NUMS} in {ARTICLES_DIR}")
        return

    print(f"Extracting {len(pdfs)} articles → {OUTPUT_DIR}\n")
    for i, p in enumerate(pdfs, 1):
        print(f"  {i}. [{_article_num(p)}] {p.name[:70]}")

    extractor = GPTExtractor()
    total_cost = 0.0
    success_count = 0

    for i, pdf_path in enumerate(pdfs, 1):
        print(f"\n{'=' * 70}")
        print(f"[{i}/{len(pdfs)}] {_article_num(pdf_path)} — {pdf_path.name[:65]}")
        print("=" * 70, flush=True)

        t0 = time.perf_counter()
        processed = process_pdf(pdf_path, source_database="articles")
        if not processed.extraction_text:
            print(f"  SKIP: No text ({processed.parse_errors})")
            continue

        result = extractor.extract(processed)
        elapsed = time.perf_counter() - t0

        if not result.success:
            print(f"  FAILED: {result.error}")
            continue

        total_cost += result.cost_usd
        success_count += 1

        output = result.extraction.model_dump(mode="json")
        safe_name = pdf_path.stem[:80].replace("/", "_").replace("\\", "_")
        out_path = OUTPUT_DIR / f"{safe_name}.json"
        out_path.write_text(
            json.dumps(output, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        d = output["data"]
        conf = d["confounders_identified"]
        findings = d.get("main_findings") or []
        na = sum(
            1 for c in conf
            if str(c.get("variable_code", "")).upper() in ("N/A", "NA", "")
        )

        print(f"  Duration: {elapsed:.1f}s | Cost: ${result.cost_usd:.4f}")
        print(f"  Title: {(output['metadata'].get('title') or 'N/A')[:75]}")
        print(f"  DOI: {output['metadata'].get('doi') or '(null)'}")
        sd = d.get("sample_details") or {}
        print(f"  N: {sd.get('total_students')} | Countries: {len(sd.get('countries') or [])}")
        sfc = (sd.get("sample_filtering_criteria") or "")[:120]
        print(f"  Filtering: {sfc}{'…' if len(sd.get('sample_filtering_criteria') or '') > 120 else ''}")
        print(f"  ML primary: {d['ml_techniques']['primary']}")
        print(f"  Confounders: {len(conf)} | N/A codes: {na}")
        osum = (d.get("outcome_summary") or "")[:160]
        print(f"  Outcome summary: {osum}{'…' if len(d.get('outcome_summary') or '') > 160 else ''}")
        print(f"  Main findings: {len(findings)}")
        for j, f in enumerate(findings, 1):
            print(f"    [{j}] dataset: {f.get('dataset_used', '?')[:55]}")
            print(f"        target: {f.get('target_variable', '?')[:60]}")
            preds = f.get("top_predictors") or []
            print(f"        predictors: {', '.join(preds[:4])}{'…' if len(preds) > 4 else ''}")
            print(f"        metrics: {(f.get('performance_metrics') or '')[:70]}")
        print(f"  Saved: {out_path.name}", flush=True)

    print(f"\n{'=' * 70}")
    print(f"DONE — {success_count}/{len(pdfs)} OK | Total cost: ${total_cost:.4f}")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
