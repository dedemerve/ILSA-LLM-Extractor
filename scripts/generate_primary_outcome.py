"""
primary_outcome_summary kolonunu Groq llama-3.1-8b-instant ile üretir
ve sadece articles_master + articles_full parquet dosyalarını HF'e yükler.
"""

import json
import os
import pathlib
import re
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi

try:
    from dotenv import load_dotenv
    load_dotenv(pathlib.Path(__file__).resolve().parents[1] / ".env")
except ImportError:
    pass

REPO_ID  = "dedemerve/ILSA-LLM-Extractor-Dataset"
OUTPUTS  = pathlib.Path(__file__).resolve().parents[1] / "outputs"
CACHE    = OUTPUTS / "primary_outcome_cache.json"
MODEL    = "llama-3.1-8b-instant"
WORKERS  = 1
api      = HfApi()

_DROP_COLS = [
    "file_name", "corpus_source", "authors", "weight_variable_name",
    "plausible_values_handling", "handling_not_reported_explanation",
    "ml_primary", "ml_all_techniques", "open_access",
]

_SYSTEM_PROMPT = """You are an Expert Academic Summarization Agent. Extract the single most critical research finding from the outcome_summary.

RULES:
1. Synthesize — do NOT truncate. Write a completely new, whole sentence.
2. Ignore methodology (models, sampling, missing-data handling).
3. Focus on the bottom-line relationship or impact.
4. 20-40 words, always a complete sentence.
5. No filler phrases like "The study concludes that...".
6. If no clear conclusion, state the primary relationship plainly.

Return ONLY the single sentence. No quotes, no markdown."""


def _load_cache() -> dict:
    if CACHE.exists():
        return json.loads(CACHE.read_text(encoding="utf-8"))
    return {}


def _save_cache(cache: dict):
    CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def _call_llm(text: str, retries: int = 5) -> str | None:
    try:
        from openai import OpenAI, RateLimitError
    except ImportError:
        return None
    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key:
        return None
    client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user",   "content": str(text)},
                ],
                temperature=0.0,
                max_tokens=120,
            )
            result = (resp.choices[0].message.content or "").strip().strip('"\'`')
            return result if result else None
        except RateLimitError as e:
            err = str(e)
            print(f"  RATE LIMIT DETAY: {err[:400]}", flush=True)
            import re as _re
            m = _re.search(r"try again in ([\d.]+)s", err)
            wait = float(m.group(1)) + 2 if m else 60
            print(f"  Rate limit — {wait:.0f}s bekleniyor...", flush=True)
            time.sleep(wait)
        except Exception as e:
            print(f"  Hata: {e}", flush=True)
            return None
    return None


def generate_pos(series: pd.Series, cache: dict) -> pd.Series:
    results = dict(cache)
    pending = [(i, v) for i, v in series.items() if str(i) not in cache]
    total   = len(series)
    done    = total - len(pending)

    print(f"  Cache'den yüklendi: {done}/{total}")
    print(f"  İşlenecek: {len(pending)}")

    def process(idx_text):
        idx, text = idx_text
        if pd.isna(text):
            return str(idx), None
        s = str(text).strip()
        if not s or s.lower() in ("nan", "none", "null"):
            return str(idx), None
        return str(idx), _call_llm(s)

    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        futures = {executor.submit(process, item): item[0] for item in pending}
        for future in as_completed(futures):
            key, result = future.result()
            results[key] = result
            done += 1
            if done % 50 == 0 or done == total:
                print(f"  {done}/{total} işlendi", flush=True)
                _save_cache(results)

    _save_cache(results)
    return pd.Series({int(k): v for k, v in results.items()})


def drop_hf_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns=[c for c in _DROP_COLS if c in df.columns])
    if "year" in df.columns:
        df["year"] = (
            pd.to_numeric(df["year"], errors="coerce")
            .astype("Int64").astype(str).replace("<NA>", None)
        )
    for col in ("student_weights_used", "replicate_weights_used"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("boolean")
    if "total_students" in df.columns:
        df["total_students"] = pd.to_numeric(df["total_students"], errors="coerce").astype("Int64")
    if "countries_json" in df.columns:
        df["n_countries"] = df["countries_json"].apply(
            lambda x: len(json.loads(x)) if isinstance(x, str) and x not in ("null", "") else None
        ).astype("Int64")
        df = df.drop(columns=["countries_json"])
    if "countries_formatted" in df.columns:
        df = df.drop(columns=["countries_formatted"])
    return df


def df_to_parquet(df: pd.DataFrame, path: pathlib.Path):
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == object:
            sample = df[col].dropna().head(20)
            if sample.apply(lambda x: isinstance(x, (list, dict))).any():
                df[col] = df[col].apply(
                    lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, (list, dict)) else x
                )
            elif sample.apply(lambda x: isinstance(x, (int, float))).any():
                df[col] = df[col].astype(str)
        if df[col].dtype == object:
            df[col] = df[col].where(df[col].isna(), df[col].astype(str))
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), path, compression="zstd")
    print(f"  ✓ {path.name}  ({len(df):,} satır × {len(df.columns)} sütun)")


if __name__ == "__main__":
    print("Excel okunuyor...")
    df_articles = pd.read_excel(OUTPUTS / "ILSA_Meta_Analysis_Dataset_CLEAN.xlsx", sheet_name="1_Articles_Master")
    df_master   = pd.read_excel(OUTPUTS / "master_structured_table.xlsx", sheet_name="Papers")

    cache = _load_cache()

    print("\n[1/2] articles_master için primary_outcome_summary üretiliyor...")
    df_articles = drop_hf_cols(df_articles)
    pos = generate_pos(df_articles["outcome_summary"], cache)
    df_articles.insert(df_articles.columns.get_loc("outcome_summary") + 1, "primary_outcome_summary", pos)
    filled = df_articles["primary_outcome_summary"].notna().sum()
    print(f"  Tamamlandı: {filled}/{len(df_articles)} dolu")

    print("\n[2/2] articles_full için primary_outcome_summary üretiliyor...")
    df_master = drop_hf_cols(df_master)
    if "outcome_summary" in df_master.columns:
        pos2 = generate_pos(df_master["outcome_summary"], cache)
        df_master.insert(df_master.columns.get_loc("outcome_summary") + 1, "primary_outcome_summary", pos2)
        filled2 = df_master["primary_outcome_summary"].notna().sum()
        print(f"  Tamamlandı: {filled2}/{len(df_master)} dolu")

    print("\nParquet hazırlanıyor ve HF'e yükleniyor...")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = pathlib.Path(tmp)
        df_to_parquet(df_articles, tmp / "articles_master.parquet")
        df_to_parquet(df_master,   tmp / "articles_full.parquet")
        api.upload_folder(
            folder_path=str(tmp),
            repo_id=REPO_ID,
            repo_type="dataset",
            path_in_repo="data/processed",
        )

    print("\n✅ Tamamlandı!")
    print(f"Cache: {CACHE}")
