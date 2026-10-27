# Human Validation Protocol
## LLM Etiketleme Güvenilirliği — Cohen's Kappa Analizi

**Dosya**: `analysis/human_validation_protocol.md`  
**Gold kaynak**: `outputs/finetuning/data/gold_standard_full_corpus_v1.4.1.csv`  
**Toplam kayıt**: 246 satır / 42 çalışma  
**Amaç**: LLM'in otomatik etiketlediği `canonical_construct` değerlerini bağımsız insan kodlamasıyla karşılaştırarak inter-rater güvenilirliği (Cohen's Kappa) hesaplamak.

---

## Neden Human Validation Gerekli?

RQ2 için kullanılan bilgi tabanı (`predictor_weights_v2.csv`) LLM tarafından ILSA literatüründen otomatik çıkarıldı. Tez savunmasında "LLM etiketleme kalitesi neydi?" sorusu sorulacak. Cohen's Kappa bu kaliteyi ölçen standart metriktir (κ > 0.60 = kabul edilebilir, κ > 0.80 = güçlü).

---

## Adım 1 — Örnekleme Betiğini Çalıştır

```bash
cd /Users/mrved/Desktop/ILSA_LLMs
python3 scripts/sample_for_human_validation.py
```

Bu betik henüz yok; aşağıdaki komutu terminalden çalıştır:

```python
import pandas as pd, numpy as np

df = pd.read_csv("outputs/finetuning/data/gold_standard_full_corpus_v1.4.1.csv")

# Stratified sample: her construct kategorisinden proportional seçim
# Hedef: 30 çalışma = ~72 satır (ortalama 2.4 satır/çalışma)
np.random.seed(42)

# Construct grupları (UNMATCHED ayrı bir kategori olarak)
df["construct_group"] = df["canonical_construct"].apply(
    lambda x: "UNMATCHED" if "UNMATCHED" in str(x) else "MATCHED"
)

# Çalışma bazında stratify (çalışma = study_id)
studies = df.groupby(["construct_group"])["study_id"].unique()

sampled_studies = []
for group, study_list in studies.items():
    n = max(1, round(len(study_list) * 30 / 42))  # proportional
    chosen = np.random.choice(study_list, size=min(n, len(study_list)), replace=False)
    sampled_studies.extend(chosen.tolist())

sampled_studies = list(set(sampled_studies))[:30]
sample_df = df[df["study_id"].isin(sampled_studies)].copy()

# Validation formu: LLM etiketlerini gizle
validation_form = sample_df[[
    "record_id", "study_id", "doi", "title", "year", "program",
    "predictor",          # ham değişken adı (kaynak metinden)
    "source_location",    # abstract/methods/results
]].copy()
validation_form["human_canonical_construct"] = ""  # doldurulacak
validation_form["human_notes"] = ""

validation_form.to_csv(
    "outputs/finetuning/audit/human_validation_sample_v1.csv",
    index=False
)
print(f"Kayıt sayısı: {len(validation_form)}, Çalışma sayısı: {len(sampled_studies)}")
```

**Beklenen çıktı**: `outputs/finetuning/audit/human_validation_sample_v1.csv`  
(~30 çalışma, ~72 satır)

---

## Adım 2 — Construct Referans Listesini Hazırla

Etiketleme sırasında kullanılacak geçerli canonical construct listesi:

```bash
python3 -c "
import pandas as pd
df = pd.read_csv('outputs/stage5/predictor_weights_v2.csv')
print(df[['predictor_canonical', 'w_norm']].to_string())
"
```

Ayrıca gold dataset'teki tüm unique değerler:
```bash
python3 -c "
import pandas as pd
df = pd.read_csv('outputs/finetuning/data/gold_standard_full_corpus_v1.4.1.csv')
print(sorted(df['canonical_construct'].unique()))
"
```

---

## Adım 3 — İnsan Kodlaması (Tahminen 2–3 saat)

`outputs/finetuning/audit/human_validation_sample_v1.csv` dosyasını Excel/Numbers'ta aç.

Her satır için:
1. `predictor` sütunundaki ham değişken adını oku
2. `title` ve `program` bilgisini dikkate al
3. `source_location` sütununa göre makalenin ilgili bölümüne bak (gerekirse DOI ile makaleye ulaş)
4. `human_canonical_construct` sütununa en uygun canonical değeri yaz
5. Belirsiz ise `human_notes` sütununa açıklama ekle

**Kurallar**:
- Listede karşılık bulamazsan: `UNMATCHED` yaz
- Eğitimsel bağlama özgü ise: `UNMATCHED_TEACHER_*` formatını kullan
- Birden fazla construct uyuyorsa: en spesifik olanı seç

---

## Adım 4 — Kappa Hesaplama Betiğini Çalıştır

```bash
cd /Users/mrved/Desktop/ILSA_LLMs
python3 scripts/compute_human_validation_kappa.py
```

Bu betik henüz yok; aşağıdaki kodu `scripts/compute_human_validation_kappa.py` olarak kaydet:

```python
#!/usr/bin/env python3
"""Human validation Cohen's Kappa hesaplama."""
import pandas as pd
from sklearn.metrics import cohen_kappa_score

GOLD = "outputs/finetuning/data/gold_standard_full_corpus_v1.4.1.csv"
HUMAN = "outputs/finetuning/audit/human_validation_sample_v1.csv"

gold_df = pd.read_csv(GOLD)[["record_id", "canonical_construct"]]
human_df = pd.read_csv(HUMAN)[["record_id", "human_canonical_construct"]]

# Boş satırları at
human_df = human_df[human_df["human_canonical_construct"].notna() &
                    (human_df["human_canonical_construct"] != "")]

merged = human_df.merge(gold_df, on="record_id", how="inner")
print(f"Karşılaştırılan kayıt: {len(merged)}")

kappa = cohen_kappa_score(
    merged["canonical_construct"],
    merged["human_canonical_construct"]
)
agreement = (merged["canonical_construct"] == merged["human_canonical_construct"]).mean()

print(f"\nCohen's Kappa   : {kappa:.4f}")
print(f"Basit uyum oranı: {agreement:.4f} ({agreement*100:.1f}%)")
print()
print("Yorumlama:")
if kappa >= 0.80:
    print("  κ ≥ 0.80 → Güçlü uyum (tez için yeterli)")
elif kappa >= 0.60:
    print("  0.60 ≤ κ < 0.80 → Kabul edilebilir uyum (sınır)")
else:
    print("  κ < 0.60 → Zayıf uyum — etiketleme protokolünü gözden geçir")

# Uyuşmazlık analizi
disagreements = merged[merged["canonical_construct"] != merged["human_canonical_construct"]]
if not disagreements.empty:
    print(f"\nUyuşmazlık sayısı: {len(disagreements)}")
    print(disagreements[["record_id", "canonical_construct",
                          "human_canonical_construct"]].to_string(index=False))

merged.to_csv("outputs/finetuning/audit/kappa_comparison_v1.csv", index=False)
print("\nKarşılaştırma tablosu kaydedildi: outputs/finetuning/audit/kappa_comparison_v1.csv")
```

---

## Adım 5 — Tez Metnine Yaz

Kappa sonucu elde edildikten sonra tezin **Methodology** bölümüne (bilgi tabanı alt bölümü) eklenecek paragraf:

```
Inter-rater reliability of the LLM-assisted label extraction was assessed by 
independently re-coding a stratified random sample of 30 studies (N = XX records) 
drawn from the gold-standard corpus (v1.4.1). Cohen's kappa between the LLM-assigned 
canonical construct labels and the independent human coder was κ = [DEĞER] 
(simple agreement = XX%), indicating [güçlü / kabul edilebilir / zayıf] agreement 
(Landis & Koch, 1977). Disagreements primarily arose in [UNMATCHED kategorileri / 
teacher-level constructs / vb.], which were resolved by [açıklama].
```

---

## Özet Tablo

| Adım | Ne yapılır | Tahmini süre | Araç |
|------|------------|-------------|------|
| 1 | Örnekleme betiğini çalıştır | 2 dakika | Terminal |
| 2 | Construct listesini yazdır | 2 dakika | Terminal |
| 3 | 30 çalışma için manuel etiketleme | 2–3 saat | Excel/Numbers |
| 4 | Kappa betiğini çalıştır | 2 dakika | Terminal |
| 5 | Tez metnine paragraf ekle | 15 dakika | Word |

**Toplam tahmini süre**: ~3.5 saat (manuel kodlama dahil)

---

## Notlar

- Tüm etiketler `Merve_Dede` tarafından yapılmış (`gold_verified` veya `gold_verified_after_adjudication`). Human validation için **bağımsız bir ikinci kodlayıcı** olmak üzere kendin kodlarsın — aynı makaleye daha önce baktığını unutmaya çalış (blind review).
- 30 çalışma pratikte yeterlidir; bazı güvenilirlik çalışmaları %20 örnekle çalışır (burada %71).
- UNMATCHED etiketleri kappa hesabına dahil edilir — bu LLM'in sınırını ölçmenin parçası.
