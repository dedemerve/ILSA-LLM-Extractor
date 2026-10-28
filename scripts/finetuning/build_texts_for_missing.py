"""
PDF → .txt extraction for studies in full_source_inventory.csv where
text_extracted=False but pdf_available=True.

Reads each pdf_path, extracts plain text with pymupdf, writes to
outputs/finetuning/source_recovery/texts/<file_name>.txt, then updates
the inventory CSV in place.

Usage:
  python3 scripts/finetuning/build_texts_for_missing.py
  python3 scripts/finetuning/build_texts_for_missing.py --dry-run   # just count
  python3 scripts/finetuning/build_texts_for_missing.py --limit 10  # first N only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INVENTORY = ROOT / "outputs/finetuning/source_recovery/full_source_inventory.csv"
TEXTS_DIR = ROOT / "outputs/finetuning/source_recovery/texts"

MIN_CHARS = 200   # files shorter than this are treated as extraction failures

# Search roots: try the original path first, then fall back to these folders by filename
PDF_SEARCH_ROOTS = [
    Path.home() / "Desktop/ILSA Documents for LLM Extraction ",
    Path.home() / "Desktop/ILSA Survey",
    Path.home() / "Desktop/ILSA Datasets for LLM Extraction",
    Path.home() / "Desktop/ILSA Documents LLM Extraction",
    Path.home() / "Desktop/ILSA Papers & Reports LLM ",
    Path.home() / "Desktop/ILSA-Survey-Analysis Code/new_papers",
]

FILENAME_ALIASES = {
    "72. Teig. (2024). Uncovering Student Strategies for Solving Scientific Inquiry Tasks: Insights from Student Process Data in PISA.pdf":
        "72. Teig (2024). Uncovering Student Strategies for Solving Scientific Inquiry Tasks: Insights from Student Process Data in PISA.pdf",
}


def build_pdf_index() -> dict[str, Path]:
    """Map filename → path by scanning PDF_SEARCH_ROOTS."""
    idx: dict[str, Path] = {}
    for root in PDF_SEARCH_ROOTS:
        if root.exists():
            for p in root.rglob("*.pdf"):
                idx.setdefault(p.name, p)   # first found wins
    return idx


def extract_text(pdf_path: Path) -> str | None:
    try:
        import fitz  # pymupdf
        doc = fitz.open(str(pdf_path))
        pages = [page.get_text() for page in doc]
        doc.close()
        text = "\n".join(pages)
        return text if len(text.strip()) >= MIN_CHARS else None
    except Exception as e:
        print(f"  pymupdf error: {e}", file=sys.stderr)
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true",
                    help="report counts only, do not write files or modify inventory")
    ap.add_argument("--limit", type=int, default=0,
                    help="process at most N studies (0 = all)")
    args = ap.parse_args()

    import pandas as pd

    print("Building PDF index from search roots...", end=" ", flush=True)
    pdf_index = build_pdf_index()
    print(f"{len(pdf_index)} PDFs indexed.")

    inv = pd.read_csv(INVENTORY, dtype=str)
    todo = inv[inv["text_extracted"].str.strip().str.lower() != "true"].copy()

    print(f"Inventory: {len(inv)} total, {len(todo)} without text")

    # Resolve actual path: original pdf_path first, then filename lookup
    def resolve(row):
        orig = row.get("pdf_path")
        if orig and isinstance(orig, str) and orig.strip() and Path(orig).exists():
            return Path(orig)
        name = row["file_name"]
        return pdf_index.get(name) or pdf_index.get(FILENAME_ALIASES.get(name, ""))

    todo["_resolved"] = todo.apply(resolve, axis=1)
    found = todo[todo["_resolved"].notna()]
    not_found = todo[todo["_resolved"].isna()]

    print(f"  Found on disk (original path or filename match): {len(found)}")
    print(f"  Not found anywhere: {len(not_found)}")

    if args.limit:
        found = found.head(args.limit)
        print(f"  (limited to first {args.limit})")

    if args.dry_run:
        if len(not_found):
            print("\nNot found (sample):")
            for _, r in not_found.head(5).iterrows():
                print(f"  {r['file_name']}")
        print("\n--dry-run: stopping here, no files written.")
        return

    TEXTS_DIR.mkdir(parents=True, exist_ok=True)

    n_ok = n_fail = n_missing = 0
    updated_rows = []

    for idx, row in found.iterrows():
        pdf_path: Path = row["_resolved"]
        file_name = row["file_name"]
        stem = Path(file_name).stem
        txt_out = TEXTS_DIR / f"{stem}.txt"

        if not pdf_path.exists():
            n_missing += 1
            continue

        print(f"  [{n_ok + n_fail + 1}] {file_name[:70]}", end=" ... ")
        text = extract_text(pdf_path)
        if text is None:
            print("FAIL (empty/error)")
            n_fail += 1
            continue

        txt_out.write_text(text, encoding="utf-8")
        inv.at[idx, "text_extracted"] = "True"
        inv.at[idx, "txt_path"] = str(txt_out.relative_to(ROOT))
        inv.at[idx, "pdf_path"] = str(pdf_path)  # update to resolved path
        updated_rows.append(idx)
        n_ok += 1
        print(f"OK ({len(text):,} chars)")

    if updated_rows:
        inv.to_csv(INVENTORY, index=False)
        print(f"\nInventory updated: {len(updated_rows)} rows written to {INVENTORY.name}")

    print(f"\nDone — ok: {n_ok}  fail: {n_fail}  pdf_not_on_disk: {n_missing}")
    print(f"New texts/ total: "
          f"{len(list(TEXTS_DIR.glob('*.txt')))} files")


if __name__ == "__main__":
    main()
