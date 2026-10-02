"""
RAG context builder — iki path:
  1. Dense  : ChromaDB (all-MiniLM-L6-v2) — embedding benzerliği
  2. Keyword: SKB JSON üzerinde token eşleşmesi (ChromaDB yoksa veya yetersizse fallback)

Her path bağımsız çalışır; sonuçlar birleştirilip tekrarlar temizlenir.
"""
from __future__ import annotations

import json
import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
SKB_PATH   = ROOT / "outputs" / "semantic_knowledge_base.json"
CHROMA_DIR = ROOT / "outputs" / "chroma_ilsa_synthesis"
COLLECTION = "ilsa_knowledge_synthesis"
EMBED_MODEL = "all-MiniLM-L6-v2"


# ---------------------------------------------------------------------------
# SKB yükleyici
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _load_skb() -> dict:
    for p in (SKB_PATH, ROOT / "outputs" / "semantic_knowledge_base.json"):
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    return {}


# ---------------------------------------------------------------------------
# Dense path — ChromaDB
# ---------------------------------------------------------------------------

def _dense_retrieve(query: str, top_k: int = 12) -> list[str]:
    """ChromaDB'den embedding benzerliğiyle top_k sonuç döndürür."""
    try:
        import chromadb
        from chromadb.utils import embedding_functions
    except ImportError:
        log.debug("chromadb yüklü değil; dense path atlanıyor.")
        return []

    if not CHROMA_DIR.exists():
        log.debug("ChromaDB dizini bulunamadı: %s", CHROMA_DIR)
        return []

    try:
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBED_MODEL
        )
        col = client.get_collection(name=COLLECTION, embedding_function=ef)
        results = col.query(query_texts=[query], n_results=min(top_k, col.count()))
        docs      = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas",  [[]])[0]
        distances = results.get("distances",  [[]])[0]

        lines: list[str] = []
        for doc, meta, dist in zip(docs, metadatas, distances):
            score = round(1 - dist, 3)          # cosine similarity
            tag   = meta.get("Metadata_Filter_Flag", "") if meta else ""
            snippet = str(doc)[:220].replace("\n", " ")
            lines.append(f"  [dense|sim={score}|{tag}] {snippet}")
        return lines

    except Exception as exc:
        log.warning("ChromaDB sorgusu başarısız: %s", exc)
        return []


# ---------------------------------------------------------------------------
# Keyword path — SKB JSON
# ---------------------------------------------------------------------------

def _keyword_retrieve(
    query: str,
    ilsa: str | None,
    domain: str | None,
    top_k: int = 8,
) -> list[str]:
    """SKB JSON üzerinde token eşleşmesiyle top_k sonuç döndürür.

    SKB v4 yapısı:
      codebook        → list[dict] — Canonical_Category, Original_Source_Examples
      synthesis_records → list[dict] — Canonical_Method, Canonical_Variable,
                                        Aggregate_Effect_Trend, Study_Count, Metadata_Filter_Flag
      taxonomy.canonical_categories → list[str]
    """
    skb = _load_skb()
    if not skb:
        return []

    tokens = set(re.findall(r"[a-zA-Z]{4,}", query.lower()))
    if ilsa:
        tokens.add(ilsa.lower())
    if domain:
        tokens.add(domain.lower())

    hits: list[tuple[int, str]] = []

    # --- codebook: 39 canonical predictor kategorisi ---
    for entry in skb.get("codebook", []):
        if not isinstance(entry, dict):
            continue
        cat  = str(entry.get("Canonical_Category", ""))
        blob = (cat + " " + str(entry.get("Original_Source_Examples", ""))).lower()
        score = sum(1 for t in tokens if t in blob)
        if score > 0:
            hits.append((score, f"  [keyword|{cat}] {blob[:180]}"))

    # --- synthesis_records: 174 aggregated effect kayıt ---
    for rec in skb.get("synthesis_records", []):
        if not isinstance(rec, dict):
            continue
        var    = str(rec.get("Canonical_Variable", ""))
        method = str(rec.get("Canonical_Method", ""))
        trend  = str(rec.get("Aggregate_Effect_Trend", ""))
        n      = rec.get("Study_Count", "?")
        flag   = rec.get("Metadata_Filter_Flag", "")
        blob   = f"{var} {method} {trend}".lower()
        score  = sum(1 for t in tokens if t in blob)
        if score > 0:
            snippet = f"{var} | method={method} | trend={trend} | n={n} | {flag}"
            hits.append((score, f"  [keyword|synthesis] {snippet[:200]}"))

    # --- taxonomy canonical categories ---
    for cat in skb.get("taxonomy", {}).get("canonical_categories", []):
        if not isinstance(cat, str):
            continue
        score = sum(1 for t in tokens if t in cat.lower())
        if score > 0:
            hits.append((score, f"  [keyword|taxonomy] {cat}"))

    hits.sort(key=lambda x: -x[0])
    return [h[1] for h in hits[:top_k]]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class RAGContextBuilder:
    """Dense + keyword retrieval; sonuçları birleştirip tekrarları temizler."""

    def build(
        self,
        query: str,
        ilsa: str | None = None,
        domain: str | None = None,
        top_k: int = 8,
        use_dense: bool = True,
    ) -> str:
        if not query:
            return ""

        lines: list[str] = []

        if use_dense:
            dense = _dense_retrieve(query, top_k=12)
            lines.extend(dense)

        keyword = _keyword_retrieve(query, ilsa, domain, top_k=top_k)
        # Keyword sonuçlarından dense'te zaten geçen snippet'leri çıkar
        dense_text = " ".join(lines).lower()
        for kw in keyword:
            core = kw[kw.find("]") + 2 : kw.find("]") + 60].lower()
            if core and core not in dense_text:
                lines.append(kw)

        if not lines:
            return ""

        header = "RAG Evidence"
        if use_dense and any("[dense" in l for l in lines):
            header += " (dense + keyword)"
        else:
            header += " (keyword)"

        return header + ":\n" + "\n".join(lines[: top_k + 4])
