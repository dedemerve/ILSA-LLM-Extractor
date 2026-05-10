# ILSA Literature Extraction Pipeline

Yıldız Teknik Üniversitesi - Veri Bilimi ve Büyük Veri programı kapsamında, ~1.800 ILSA-AI akademik makalesinin GPT-4o Structured Outputs API'si ile yapılandırılmış metadata'ya dönüştürülmesi için tasarlanmış pipeline.

## Kurulum

```bash
# Conda ortamı aktif
conda activate ilsa-literature-review

# Bağımlılıkları kur
pip install -r requirements.txt

# API anahtarı
cp .env.example .env
# .env dosyasını düzenleyip OPENAI_API_KEY'i girin
```

## Kullanım

### Test koşumu (20 makale)

```bash
python scripts/run_pipeline.py \
    --pdf-dir ../data/pdfs \
    --output-dir ./output_test \
    --limit 20 \
    --workers 3
```

### Tam koşum (1800 makale)

```bash
python scripts/run_pipeline.py \
    --pdf-dir ../data/pdfs \
    --output-dir ./output \
    --workers 5 \
    --resume
```

### Bilgi tabanı sorguları

```bash
python scripts/example_queries.py --db ./output/ilsa_knowledge_base.db
```

## Maliyet

**1.800 makale ≈ $80-90** (akıllı bölüm seçimi sayesinde)

## Çıktılar

- `output/json/*.json` - Her makale için ayrı JSON
- `output/ilsa_master.parquet` - Tek dosya, ML/analiz için
- `output/ilsa_knowledge_base.db` - SQLite, SQL sorguları için
