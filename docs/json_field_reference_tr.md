# ILSA Makale JSON Alan Sözlüğü

Bu belge, `outputs/**/json/*.json` dosyalarındaki her alanın ne anlama geldiğini açıklar. Şema kaynağı: `src/schemas/models.py` (`ILSAArticleMetadata`).

Her JSON dosyası **tek bir PDF** ile eşleşir. Üst düzey yapı:

```json
{
  "metadata": { ... },
  "data": { ... }
}
```

---

## 1. `metadata` — Bibliyografik kimlik

| Alan | Tür | Ne ifade eder |
|------|-----|----------------|
| **`file_name`** | metin (zorunlu) | Kaynak PDF dosya adı. Tüm Excel tablolarında **birincil anahtar** (`file_name` ile birleştirme). |
| **`title`** | metin veya `null` | Makalenin/raporun tam başlığı. |
| **`authors`** | metin listesi veya `null` | Yazar adları (sıralı liste). |
| **`year`** | tam sayı veya `null` | Yayın yılı (4 haneli). |
| **`doi`** | metin veya `null` | DOI (`10.xxxx/...`); URL öneki olmadan. PDF’de varsa çıkarılması beklenir. |
| **`venue`** | metin veya `null` | Dergi, konferans, yayınevi veya kurum adı. |
| **`publication_type`** | sabit liste veya `null` | Yayın biçimi: `journal`, `conference`, `book_chapter`, `preprint`, `report`, `thesis`. |
| **`open_access`** | `true` / `false` / `null` | Açık erişim olup olmadığı; bilinmiyorsa `null`. |
| **`source_category`** | sabit liste veya `null` | Araştırma türü: `peer_reviewed_research`, `review_article`, `methodology_paper`, `technical_report`. Excel’de `study_filter_type` ve bulgu kurallarını etkiler. |

---

## 2. `data.survey_design` — Örnekleme ve ağırlık

| Alan | Tür | Ne ifade eder |
|------|-----|----------------|
| **`student_weights_used`** | `true` / `false` / `null` | Öğrenci/örnekleme ağırlığı kullanıldı mı (ör. `W_FSTUWT`). |
| **`replicate_weights_used`** | `true` / `false` / `null` | BRR, jackknife vb. replike ağırlık / varyans yöntemi kullanıldı mı. |
| **`weight_variable_name`** | metin veya `null` | Metinde geçen ağırlık değişken adı (ör. `W_FSTUWT`, `TOTWGT`). |
| **`weight_fields_interpretation`** | metin (zorunlu) | 3–4 cümle: hangi veri seti, örnek seçimi, temizleme, ağırlık stratejisi. Teknik raporlarda bile doldurulur; boş bırakılmaz. Excel’de `weights_filter` ve ağırlık notları için kaynak. |

---

## 3. `data.sample_details` — Örneklem

| Alan | Tür | Ne ifade eder |
|------|-----|----------------|
| **`total_students`** | tam sayı veya `null` | Analitik örnekteki toplam öğrenci sayısı. |
| **`countries`** | liste | Ülke bazlı örnek: her öğe `{ "country_code": "TUR", "n_students": 4500 }`. `country_code` = ISO 3166-1 alpha-3. |
| **`sample_filtering_criteria`** | metin (zorunlu) | Yazarların ILSA verisini nasıl filtrelediği (sınıf, eksik veri, alt görev vb.). Filtre yoksa: tam örnek ifadesi. |

---

## 4. `data` — Ölçüm ve analiz yöntemi

| Alan | Tür | Ne ifade eder |
|------|-----|----------------|
| **`plausible_values_handling`** | sabit liste | Plausible value (PV) kullanımı: `rubin_rules`, `single_pv`, `average_pv`, `all_pv`, `mitml`, `wle`, `irt_theta`, `not_applicable`, `not_reported`. Teknik rapor / user guide → çoğunlukla `not_applicable`. |
| **`missing_data_handling`** | sabit liste | Eksik veri: `listwise_deletion`, `pairwise_deletion`, `mean_imputation`, `single_imputation`, `knn_imputation`, `multiple_imputation`, `not_reported`. |
| **`handling_not_reported_explanation`** | metin veya `null` | PV veya eksik veri `not_reported` / `not_applicable` ise **neden** (2–3 cümle). Her ikisi de açıkça raporlandıysa `null`. |
| **`research_design_type`** | sabit liste veya `null` | `predictive`, `causal_observational`, `causal_experimental`, `exploratory`. |
| **`null_fields_interpretation`** | metin veya `null` | Çıkarım çok seyrekse (örneklem yok, ML yok): neden boş kaldığının teşhisi. Dolu kayıtlarda `null`. |

---

## 5. `data.ml_techniques` — Makine öğrenmesi

| Alan | Tür | Ne ifade eder |
|------|-----|----------------|
| **`primary`** | metin veya `null` | Ana / en iyi performans gösteren algoritma (ör. `XGBoost`). `all_techniques` doluysa null olmamalı. |
| **`all_techniques`** | metin listesi | Denenen tüm ML algoritmaları (ön işleme / klasik istatistik değil). Excel’de `ml_techniques`, `ml_family` türetilir. |

Geleneksel istatistik çalışmalarında liste boş veya `primary` null olabilir.

---

## 6. `data.confounders_identified[]` — Bağımsız değişkenler

Her liste öğesi **tek bir kavramsal değişken** (ML feature sütunu değil).

| Alan | Tür | Ne ifade eder |
|------|-----|----------------|
| **`variable_code`** | metin | Tablolama kodu: resmi ILSA kodu (`ESCS`), yapı adı veya `snake_case` slug. `N/A` yazılmaz. |
| **`variable_name`** | metin | Kısa İngilizce etiket (en fazla ~8 kelime): örn. `Gender`, `ESCS`. |
| **`category`** | sabit liste (13 değer) | Değişken alanı; `other` yok. |

**`category` değerleri:**

| Kod | Tipik içerik |
|-----|----------------|
| `socioeconomic` | ESCS, HOMEPOS, ev kitapları, ebeveyn eğitimi |
| `demographic` | Cinsiyet, göçmenlik, dil, sınıf |
| `student_attitude` | Öz-yeterlik, kaygı, motivasyon |
| `student_behavior` | Ödev süresi, devamsızlık, okuma alışkanlığı |
| `teacher` | Öğretmen nitelikleri, deneyim, öğretim uygulamaları |
| `school` | Okul türü, kaynak, sınıf mevcudu, iklim |
| `ict` | Bilgisayar kullanımı, ICT kaynakları |
| `curriculum` | Müfredat, ders saati, içerik kapsamı |
| `parent_home` | Aile desteği, ev ortamı |
| `process_data` | Yalnızca üst düzey süreç özetleri (toplam süre, VOTAT); ham tıklama/n-gram değil |
| `prior_achievement` | Önceki başarı, diğer alan PV’leri kontrol olarak |
| `peer_effects` | Sınıf iklimi, akran zorbalığı |
| `system_level` | Ülke düzeyi GDP, politika değişkenleri |

Excel’de her confounder **ayrı satır** (`3_Confounders`).

---

## 7. `data.main_findings[]` — Yapılandırılmış bulgular

Her öğe = **bir hedef değişken / bir analitik sonuç hattı**. Empirik ML makalelerinde birden fazla satır olabilir (Math vs Science).

| Alan | Tür | Ne ifade eder |
|------|-----|----------------|
| **`dataset_used`** | metin | ILSA programı, yıl, sınıf, alan (ör. `PISA 2012 Grade 15 Mathematics`). |
| **`target_variable`** | metin | Bağımlı değişken / sonuç (ör. `Mathematics achievement (PVs)`). |
| **`top_predictors`** | metin listesi | En önemli 3–5 öngörücü; mümkünse `confounders_identified` ile uyumlu isimler. |
| **`performance_metrics`** | metin | Model metrikleri (`R²`, accuracy, AUC…). Yoksa `Not reported`. |
| **`standardized_conclusion`** | metin | 2–3 cümle: veri seti → öngörücüler → hedef → bulgu → (varsa) politika/ yorum. |

**Boş `main_findings` ne zaman normal?**

- Resmi **teknik rapor**, **user guide**, **framework** (`source_category`: `technical_report` / `methodology_paper`).
- Sistematik derleme / teori, öngörücü sonuç yok.

**Ne zaman sorun sayılır?**

- Empirik ML makalesi, anlamlı `outcome_summary`, ama JSON’da `main_findings: []` → çıkarım veya doğrulama eksikliği.

Excel’de her bulgu **ayrı satır** (`2_Main_Findings`).

---

## 8. `data.outcome_summary` — Özet anlatı

| Alan | Tür | Ne ifade eder |
|------|-----|----------------|
| **`outcome_summary`** | metin (zorunlu) | 4–5 cümle (~120 kelime): makalenin ana sonuçları, en iyi model, sınırlamalar (ağırlık, nedensellik). `main_findings` tablosunun prose özeti; alan yorumları için `null_fields_interpretation` kullanılır. |

Özel önekler:

- `__EXTRACTION_FAILED__: ...` → PDF çıkarımı başarısız; pipeline `--resume` ile yeniden denenebilir.
- Çok kısa veya anlamsız metin → `substantive_outcome_summary` false; bulgu zorunluluğu tetiklenmeyebilir.

---

## 9. JSON → Excel eşlemesi (özet)

| JSON | Excel (CLEAN) |
|------|----------------|
| `metadata.*` + türetilen sınıflar | `1_Articles_Master` (1 satır / makale) |
| `data.main_findings[]` | `2_Main_Findings` (N satır / makale) |
| `data.confounders_identified[]` | `3_Confounders` (M satır / makale) |
| Taxonomy sonrası birleşik görünüm | `Canonical_View` |

Excel’de boş JSON alanları sık şu **sentinel** değerlere dönüşür (veri silinmez, filtrelenir):

| Sentinel | Anlamı |
|----------|--------|
| `N/A: Technical Report` | Teknik / çerçeve belge; empirik bulgu beklenmez |
| `N/A: Descriptive Report` | Betimleyici ulusal/uluslararası rapor |
| `Not Reported by Authors` | Empirik makale; yazar metrik/alan bildirmemiş |
| `Not Reported` | Genel eksiklik |

---

## 10. Dosya adı ≠ farklı şema

`Supplement_1.json`, `p1_tr_book.json`, `Processes.json` gibi isimler yalnızca **kaynak PDF adını** yansıtır. Hepsi aynı `metadata` + `data` yapısını kullanır; içerik türü `source_category` ve `publication_type` ile ayrılır.

| Örnek dosya | Tipik `source_category` | Beklenti |
|-------------|-------------------------|----------|
| `Supplement_*.json` | `methodology_paper` | Anket–değişken eşlemesi; empirik ML yok |
| `p1_tr_book.json` | `technical_report` | Örnekleme, ölçek, ağırlık kuralları |
| `Processes.json` | `technical_report` / rapor | Ulusal sonuçlar, PV/JRR anlatımı |
| Scopus/WoS empirik makale | `peer_reviewed_research` | Dolu `main_findings`, confounders |

---

## 11. Doğrulama ve yeniden işleme

| Komut / dosya | İşlev |
|---------------|--------|
| `validate_public_article_json()` | JSON’un şemaya uygunluğu |
| `scripts/resanitize_json_outputs.py` | Prompt/kural değişince tüm JSON’ları yeniden sanitize (API yok) |
| `scripts/enrich_json_for_excel.py` | Eksik alanları kural tabanlı doldurma |
| `scripts/build_tabular_dataset.py` | JSON → Excel |

İnsan doğrulaması sonrası düzeltmeler **doğrudan ilgili `*.json` dosyasına** yazılır; Excel pipeline ile yeniden üretilir.

---

## 12. İlgili dosyalar

- Şema: `src/schemas/models.py`
- Bulgu zorunluluğu: `src/schemas/findings_validation.py`
- Post-process / sanitize: `src/extractors/gpt_extractor.py`
- Excel sentinel ve sheet’ler: `scripts/build_tabular_dataset.py`
- Phase 0 (doğrulama mimarisi): `docs/phase0_methodological_specification.md`

*Son güncelleme: şema ile uyumlu referans belgesi; alan tanımları `models.py` Field açıklamalarından türetilmiştir.*
