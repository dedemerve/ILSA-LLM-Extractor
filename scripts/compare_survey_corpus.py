#!/usr/bin/env python3
"""Compare survey article JSON to ILSA Documents outputs; print deduped counts."""
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SURVEY_DIR = Path("/Users/mrved/Desktop/ilsa_survey_articles_json/json")
OUTPUTS = ROOT / "outputs"
CORPORA = ["Web of Science", "Scopus", "IEA", "OECD"]


def norm(s: str) -> str:
    if not s:
        return ""
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return " ".join(s.split())


def load_indexes():
    by_doi: dict[str, str] = {}
    by_title: dict[str, str] = {}
    counts: dict[str, int] = {}
    for top in CORPORA:
        counts[top] = 0
        for jp in (OUTPUTS / top).rglob("*.json"):
            counts[top] += 1
            try:
                meta = json.loads(jp.read_text(encoding="utf-8")).get("metadata") or {}
            except Exception:
                continue
            doi = (meta.get("doi") or "").strip().lower()
            title_n = norm(meta.get("title") or "")
            if doi:
                by_doi.setdefault(doi, top)
            if title_n:
                by_title.setdefault(title_n, top)
    return by_doi, by_title, counts


def match_folder(meta: dict, by_doi: dict, by_title: dict) -> str | None:
    doi = (meta.get("doi") or "").strip().lower()
    if doi and doi in by_doi:
        return by_doi[doi]
    title_n = norm(meta.get("title") or "")
    if title_n and title_n in by_title:
        return by_title[title_n]
    return None


def main() -> None:
    by_doi, by_title, counts = load_indexes()
    rows = []
    for jp in sorted(SURVEY_DIR.glob("*.json")):
        meta = json.loads(jp.read_text(encoding="utf-8")).get("metadata") or {}
        m = re.match(r"^(\d+)\.", jp.stem)
        num = int(m.group(1)) if m else None
        rows.append(
            {
                "num": num,
                "title": meta.get("title") or "",
                "hit": match_folder(meta, by_doi, by_title),
            }
        )

    wos = [r for r in rows if r["hit"] == "Web of Science"]
    scopus = [r for r in rows if r["hit"] == "Scopus"]
    other = [r for r in rows if not r["hit"]]

    nums = {r["num"] for r in rows if r["num"]}
    num_counts = Counter(r["num"] for r in rows)

    print("=== Documents corpus (outputs/) ===")
    for top in CORPORA:
        print(f"  {top}: {counts[top]}")
    print(f"  Toplam: {sum(counts.values())}")

    print("\n=== Survey (ilsa_survey_articles_json/json) ===")
    print(f"  JSON dosyası: {len(rows)}")
    print(f"  Benzersiz numara (1–131): {len(nums)}")
    print(f"  Eksik numara: {sorted(set(range(1, 132)) - nums)}")
    print(f"  Çift numara: {[n for n, c in num_counts.items() if c > 1]}")

    print("\n=== Survey kaynak (Documents ile eşleşme) ===")
    print(f"  Zaten Web of Science çıktısında: {len(wos)}")
    print(f"  Zaten Scopus çıktısında (WoS’ta yok): {len(scopus)}")
    print(f"  Google Scholar / diğer (Documents’ta yok): {len(other)}")

    print("\n=== Birleşik (çift sayım yok, DOI/başlık) ===")
    overlap = len(wos) + len(scopus)
    print(f"  Survey ∩ Documents: {overlap}")
    print(f"  Survey’e özgü eklenecek: {len(other)}")
    print(f"  Grand total benzersiz makale: {sum(counts.values()) + len(other)}")

    print("\n--- Google Scholar / diğer (eklenen) ---")
    for r in sorted(other, key=lambda x: x["num"] or 0):
        print(f"  #{r['num']:3} {r['title'][:80]}")


if __name__ == "__main__":
    main()
