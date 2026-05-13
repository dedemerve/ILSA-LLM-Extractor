# ILSA Literature Extraction Pipeline

Academic PDF’lerden (PISA, TIMSS, vb. ILSA + makine öğrenmesi) yapılandırılmış metadata çıkaran hat. Çekirdek: **PyMuPDF** ile metin, **OpenAI** ile JSON extraction, **Pydantic** şema (`src/schemas/models.py`).

## Kurulum

```bash
conda activate ilsa-literature-review   # veya kendi ortamınız
pip install -r ilsa_pipeline/requirements.txt
cp ilsa_pipeline/.env.example ilsa_pipeline/.env
# OPENAI_API_KEY ekleyin
```

Proje kökündeki `requirements.txt` tam conda çözümlemesidir; sadece extraction için `ilsa_pipeline/requirements.txt` yeterlidir.

## Çalıştırma

Ana orkestrasyon (çoklu PDF, JSON + isteğe bağlı SQLite):

```bash
cd /path/to/ILSA_LLMs
python ilsa_pipeline/scripts/run_pipeline.py \
  --pdf-dir ./data/pdfs \
  --output-dir ./output \
  --workers 3 \
  --resume
```

Belirli makale seti için: `ilsa_pipeline/scripts/extract_targeted.py`

## Çıktılar

- `output/json/*.json`: Her PDF için `file_name`, `success`, token/maliyet süreleri ve `extraction` (şemaya uygun nesne).
- Parquet / SQLite üretimi: `ilsa_pipeline/utils/storage.py` içindeki `build_master_parquet`, `build_sqlite_database`, `StorageManager`.

Eski yedekler ve alternatif şemalar `cop_kutusu/` altında tutulur.
