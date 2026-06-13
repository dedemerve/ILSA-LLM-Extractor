"""
Upload ILSA extraction outputs to HuggingFace dataset repo.
Target: dedemerve/ILSA-LLM-Extractor-Dataset
"""

import json
import pathlib
import shutil
import tempfile

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi

REPO_ID = "dedemerve/ILSA-LLM-Extractor-Dataset"
OUTPUTS = pathlib.Path("/Users/mrved/Desktop/ILSA_LLMs/outputs")
api = HfApi()


def df_to_parquet(df: pd.DataFrame, path: pathlib.Path):
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == object:
            sample = df[col].dropna().head(20)
            # list/dict → JSON string
            if sample.apply(lambda x: isinstance(x, (list, dict))).any():
                df[col] = df[col].apply(
                    lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, (list, dict)) else x
                )
            # mixed int/str → tümünü string'e çevir
            elif sample.apply(lambda x: isinstance(x, (int, float))).any():
                df[col] = df[col].astype(str)
        # object kolonu tamamen string'e normalize et (mixed type güvencesi)
        if df[col].dtype == object:
            df[col] = df[col].where(df[col].isna(), df[col].astype(str))
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, path, compression="zstd")
    print(f"  ✓ {path.name}  ({len(df):,} satır × {len(df.columns)} sütun)")


def upload_folder(local_dir: pathlib.Path, path_in_repo: str):
    api.upload_folder(
        folder_path=str(local_dir),
        repo_id=REPO_ID,
        repo_type="dataset",
        path_in_repo=path_in_repo,
    )
    print(f"  ✓ Yüklendi → {path_in_repo}/")


with tempfile.TemporaryDirectory() as tmp:
    tmp = pathlib.Path(tmp)

    # ── 1. PROCESSED TABLES ──────────────────────────────────────────────────
    print("\n[1/3] Processed tablolar hazırlanıyor...")
    proc = tmp / "processed"
    proc.mkdir()

    clean_path = OUTPUTS / "ILSA_Meta_Analysis_Dataset_CLEAN.xlsx"
    master_path = OUTPUTS / "master_structured_table.xlsx"

    df_articles = pd.read_excel(clean_path, sheet_name="1_Articles_Master")
    df_findings = pd.read_excel(clean_path, sheet_name="2_Main_Findings")
    df_conf     = pd.read_excel(clean_path, sheet_name="3_Confounders")
    df_master   = pd.read_excel(master_path, sheet_name="Papers")
    df_mfindings= pd.read_excel(master_path, sheet_name="Findings")
    df_mconf    = pd.read_excel(master_path, sheet_name="Confounders")

    df_to_parquet(df_articles,  proc / "articles_master.parquet")
    df_to_parquet(df_findings,  proc / "findings.parquet")
    df_to_parquet(df_conf,      proc / "confounders.parquet")
    df_to_parquet(df_master,    proc / "articles_full.parquet")
    df_to_parquet(df_mfindings, proc / "findings_full.parquet")
    df_to_parquet(df_mconf,     proc / "confounders_full.parquet")

    upload_folder(proc, "data/processed")

    # ── 2. REFERENCE FILES ───────────────────────────────────────────────────
    print("\n[2/3] Referans dosyalar hazırlanıyor...")
    ref = tmp / "reference"
    ref.mkdir()

    df_cb = pd.read_csv(OUTPUTS / "canonical_codebook.csv")
    df_to_parquet(df_cb, ref / "canonical_codebook.parquet")
    shutil.copy(OUTPUTS / "canonical_codebook.md", ref / "canonical_codebook.md")

    upload_folder(ref, "data/reference")

    # ── 3. RAW JSON FILES ────────────────────────────────────────────────────
    print("\n[3/3] Ham JSON dosyalar yükleniyor (1756 dosya)...")
    raw = tmp / "raw"
    raw.mkdir()

    source_map = {
        "scopus":  OUTPUTS / "Scopus",
        "oecd":    OUTPUTS / "OECD",
        "iea":     OUTPUTS / "IEA",
        "wos":     OUTPUTS / "Web of Science",
        "survey":  OUTPUTS / "ilsa_survey_articles",
    }

    total = 0
    for name, src in source_map.items():
        dest = raw / name
        dest.mkdir()
        files = list(src.rglob("*.json")) if src.exists() else []
        for f in files:
            shutil.copy(f, dest / f.name)
        total += len(files)
        print(f"  {name}: {len(files)} dosya")

    print(f"  Toplam: {total} JSON")
    upload_folder(raw, "data/raw")

print("\n✅ Tamamlandı! → https://huggingface.co/datasets/dedemerve/ILSA-LLM-Extractor-Dataset")
