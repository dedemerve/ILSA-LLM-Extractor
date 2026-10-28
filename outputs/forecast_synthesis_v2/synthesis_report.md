# PISA Tahmin Sistemi — Kapsamlı Sentez Raporu

**Tarih:** 2026-10-24  
**Dalı:** `feat/finetuning-v2-pipeline`  
**Hazırlayan:** Merve Dede  

---

## Ana bulgu

Persistence dört tarihsel hedef döngüsünde üç alanın tamamında en düşük eşit-ağırlıklı MAE'yi vermiştir. PISA 2025 sonuçlarına maruz kalınmış retrospektif karşılaştırmada ise alan liderleri ayrışmıştır. Bulgular ülke/ekonomi ortalama puanları içindir; öğrenci düzeyinde tahmin veya ileriye dönük bağımsız doğrulama değildir.

## 1. Araştırma Sorusu ve Yaklaşım

Bu çalışma, uluslararası öğrenci değerlendirmelerinden (ILSA) biri olan PISA'nın 2025 döngüsündeki matematik, okuma ve fen alanı ülke ortalamalarını tahmin etmek için tabular yöntemleri ve TimesFM'i karşılaştırmaktadır. Literatür ağırlıklı Stage 5/LOCO ve extraction fine-tuning analizleri ayrı araştırma hatlarıdır; bu dokuz-modelli PISA benchmark'ının katmanları değildir:

1. **Tabular ML** — Literatürde en sık kullanılan yöntemler (RF, XGBoost, Ridge)
2. **Zaman Serisi Modeli** — Google TimesFM (2.5 ve 3.0), sıfır-atış ve ILSA adaptasyonlu

Tarihsel rolling-origin hedefleri 2012, 2015, 2018 ve 2022'dir; her tahminde hedef öncesindeki döngüler kullanılır. Tabular modeller havuzlanmış geçmiş geçişleri gecikme özellikleriyle kullanırken TimesFM mevcut sıralı geçmişi alır. 2025 tahminleri 2000–2022 geçmişine dayanır ve gerçekleşen hedeflerle sonuçlara maruz kalınmış retrospektif olarak karşılaştırılır.

---

## 2. Literatür Korpusu Yöntem Frekansı

1.266 kayıtlık kodlanmış korpusta birincil yöntemi bulunan 147 tahminsel çalışma analiz edilmiştir. Özgün Scopus/WoS arama geçmişi, ham veri dışa aktarımları ve tarama kararları bulunmadığından bu sayım eksiksizliği doğrulanmış sistematik bir tarama değildir; sınırdaki çalışmalar için bağımsız ikinci kodlayıcı da beklenmektedir.

### Tablo 1 — Birincil Yöntem Frekansı (tahminsel çalışmalar, N=147)

| Yöntem | Çalışma N | 147 kodlu tahminsel çalışmanın %'si |
| --- | --- | --- |
| Random Forest | 42 | 28.6 |
| XGBoost | 17 | 11.6 |
| Neural Network | 16 | 10.9 |
| SVM | 14 | 9.5 |
| Logistic Regression | 6 | 4.1 |
| Decision Tree | 5 | 3.4 |
| LASSO | 5 | 3.4 |
| Gradient Boosting | 5 | 3.4 |
| LightGBM | 4 | 2.7 |
| Elastic Net | 4 | 2.7 |
| CatBoost | 2 | 1.4 |
| Stacking | 2 | 1.4 |

### Tablo 1b — Ülke Düzeyinde Tahmin Alt-Kümesindeki Yöntemler (N=17)

| Yöntem | Çalışma N |
| --- | --- |
| SVM | 4 |
| Random Forest | 3 |
| XGBoost | 2 |

Geniş tahminsel altkümede RF (42; %28,6) ve XGBoost (17; %11,6) en sık kodlanan yöntemlerdir. Dar ülke düzeyi altkümede SVM 4, RF 3 ve XGBoost 2 çalışmayla öndedir. RF/XGBoost genel karşılaştırıcı olarak gerekçelendirilebilir; doğrulanmış ülke düzeyi gelecek-döngü ILSA tahminlerinde en sık yöntemler oldukları söylenemez.

---

## 3. Tarihsel Backtesting (2012–2022)

Dört rolling-origin hedefi (2012, 2015, 2018, 2022), üç alan ve ortak örnekler kullanılmıştır: hedef başına sırasıyla 40, 51, 57 ve 60 ülke/ekonomi. Genel ortalama, 12 alan×hedef MAE hücresinin eşit ağırlıklı ortalamasıdır; standart sapma bu 12 betimsel hücre üzerindedir ve 12 bağımsız zaman dönemi anlamına gelmez.

### Tablo 2 — Tarihsel Backtesting Özeti (MAE, N=12 fold)

| Model | Ort. MAE | Std | Min | Max |
| --- | --- | --- | --- | --- |
| Persistence (baseline) | 9.309 | 2.385 | 6.353 | 14.482 |
| XGBoost | 9.502 | 2.559 | 5.996 | 14.655 |
| TimesFM 2.5 + ILSA LoRA | 9.61 | 2.416 | 6.746 | 14.822 |
| TimesFM 2.5 (zero-shot) | 9.703 | 2.415 | 6.759 | 14.825 |
| Random Forest | 9.912 | 2.692 | 6.638 | 15.399 |
| Ridge | 9.973 | 2.532 | 6.006 | 14.715 |
| TimesFM 3.0 (zero-shot) | 10.61 | 2.345 | 7.444 | 15.83 |
| Linear Trend | 11.81 | 2.713 | 8.537 | 16.975 |
| Linear-Quadratic Trend | 12.444 | 2.885 | 8.924 | 17.236 |

### Tablo 3 — Domain Bazında Tarihsel Ortalama MAE

| Model | Matematik MAE | Okuma MAE | Fen MAE |
| --- | --- | --- | --- |
| Persistence (baseline) | 9.328 | 10.156 | 8.442 |
| XGBoost | 9.563 | 10.379 | 8.565 |
| TimesFM 2.5 + ILSA LoRA | 9.773 | 10.273 | 8.783 |
| Random Forest | 9.811 | 10.759 | 9.168 |
| TimesFM 2.5 (zero-shot) | 9.873 | 10.385 | 8.85 |
| Ridge | 9.961 | 10.834 | 9.124 |
| TimesFM 3.0 (zero-shot) | 10.629 | 11.495 | 9.704 |
| Linear Trend | 11.297 | 13.333 | 10.799 |
| Linear-Quadratic Trend | 11.564 | 14.099 | 11.671 |

### Tablo 4 — Döngü Bazında Tarihsel Ortalama MAE

| Model | 2012 | 2015 | 2018 | 2022 |
| --- | --- | --- | --- | --- |
| Persistence (baseline) | 8.28 | 9.83 | 7.238 | 11.887 |
| Ridge | 9.127 | 11.403 | 7.415 | 11.947 |
| XGBoost | 8.61 | 10.261 | 7.022 | 12.117 |
| TimesFM 2.5 + ILSA LoRA | 8.6 | 10.079 | 7.592 | 12.168 |
| TimesFM 2.5 (zero-shot) | 8.711 | 10.322 | 7.602 | 12.176 |
| Random Forest | 9.159 | 10.63 | 7.289 | 12.571 |
| TimesFM 3.0 (zero-shot) | 9.744 | 11.084 | 8.833 | 12.777 |
| Linear Trend | 10.355 | 11.504 | 10.305 | 15.074 |
| Linear-Quadratic Trend | 10.131 | 13.208 | 10.711 | 15.728 |

**Bulgular:**
- Persistence üç alanın tamamında en düşük eşit-fold MAE'yi verdi; bu protokolde karmaşık yöntemler son-gözlem baseline'ını geçemedi.
- XGBoost Persistence'a yakın olsa da tarihsel alan ortalamalarının hiçbirinde Persistence'ı geçmedi.
- TimesFM 2.5 LoRA (ILSA adaptasyonlu), zero-shot'a göre ortalama ~0.09 MAE iyileştirme sağlar.
- LoRA iyileşmesi her alanda 0.2 puanın altındadır. TimesFM 2.5 LoRA tarihselde Ridge ve RF'den daha düşük ortalama MAE verirken Persistence ve XGBoost'u geçememiştir; yöntem sıralaması modele ve alana göre değişir.
- Becker–Coyle'un yayımlanmış 2022 doğrusal/kuadratik trend seçimi kodundan yeniden kurulan tahminler, ortak 60 ülke örneğinde üç alanda da Persistence ve karşılaştırılan tabular modellerden daha yüksek MAE vermiştir. Bu bir supplement-kodu rekonstrüksiyonudur; yayımlanmış hazır tahmin tablosu veya ileriye dönük doğrulama değildir.

---

## 4. PISA 2025 Nihai Karşılaştırma

> **Dikkat — sonuçlara maruz kalınmış retrospektif değerlendirme:** Kod ve manifestlerde 2025 hedefi model girdisi/eğitim hedefi yapılmamıştır; ancak 2025 sonuçları ve önceki model sonuçları proje ekibinin erişimine açıktı. Bu nedenle bu karşılaştırma kör ve bağımsız ileriye dönük doğrulama değildir.

### Tablo 5 — PISA 2025 Model Karşılaştırması (66 ülke/ekonomi; alan başına)

| Model | Matematik MAE | Okuma MAE | Fen MAE | Ortalama MAE |
| --- | --- | --- | --- | --- |
| Ridge | 9.457 | 11.357 | 9.506 | 10.107 |
| XGBoost | 9.318 | 12.932 | 9.731 | 10.66 |
| Random Forest | 11.557 | 13.032 | 9.752 | 11.447 |
| Persistence (baseline) | 11.584 | 15.216 | 9.378 | 12.059 |
| TimesFM 3.0 (zero-shot) | 12.864 | 16.75 | 10.076 | 13.23 |
| Linear-Quadratic Trend | 12.728 | 18.277 | 10.194 | 13.733 |
| TimesFM 2.5 + ILSA LoRA | 14.831 | 17.717 | 10.6 | 14.383 |
| TimesFM 2.5 (zero-shot) | 14.989 | 17.849 | 10.767 | 14.535 |
| Linear Trend | 15.188 | 19.584 | 10.582 | 15.118 |

**Ana Bulgular:**
- Alan liderleri farklıdır: matematikte XGBoost (9.318), okumada Ridge (11.357), fende Persistence (9.378).
- Üç alan MAE'sinin basit ortalamasında Ridge (10.107), XGBoost'tan (10.660) düşüktür; bu sıralama da seçili örneklem ve retrospektif tasarıma özgüdür.
- TimesFM 2.5 LoRA'nın üç-alan ortalama MAE'si Ridge'den 4.276 puan yüksektir; bu fark 2025 retrospektif karşılaştırmasına özgüdür.
- **ILSA LoRA'nın TimesFM'e katkısı:** +0.152 PISA puanı (LoRA lehine)
- Üç alan ve 66 ülke birlikte havuzlandığında dokuz modelin ortalama signed error'ı pozitiftir; alan bazında bu tüm modeller için geçerli değildir. Bu betimsel desen COVID-19 gibi nedenlere ilişkin nedensel açıklama sağlamaz.

### Belirsizlik kapsamı
Hedef puan örnekleme hatası/ülke-kümesi duyarlılığı (Phase 3.25) ve tarihsel girdi belirsizliğiyle model yeniden eğitimi (Phase 3.26) ayrı koşullu analizlerdir; birleşik aralık oluşturmaz. Phase 3.26'da 2.000 marjinal-SE çekiminde Persistence'ın en düşük eşit-fold MAE sıklığı matematikte %97,45, okumada %98,55, fende %99,75'tir; bu gelecek döngüler için olasılık değildir ve TimesFM yeniden eğitim belirsizliğini kapsamaz. Tam tarihsel ölçek-link kovaryansı doğrulanamadığı için link hatası da dahil değildir; sonuçlar tam belirsizlik yayılımı olarak sunulmamalıdır.

---

## 5. Keşfedici ülke-eşleştirmeli testler

> Bu Wilcoxon testleri 2025'e maruz kalınmış tek bir değerlendirmedeki ülke bazlı mutlak hataları eşleştirir. p değerleri çoklu karşılaştırma düzeltmesizdir; ülkeler arası bağımlılık, hedef puan örnekleme belirsizliği ve ölçek-link belirsizliği modellenmemiştir. Bu nedenle p değerleri doğrulayıcı üstünlük kanıtı sayılmamalı ve ana rapordaki koşullu küme-bootstrap/puan-SE duyarlılık analizlerinin yerine geçmemelidir.

### Tablo 6 — Wilcoxon Signed-Rank Testleri (ülke bazında hata farkı)

| Domain | Model A | Model B | N Ülke | Ortalama Δ (A−B) | Düşük ort. mutlak hata | W | p | Ham p<0.05? |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mathematics | XGBoost | TimesFM 2.5 + ILSA LoRA | 66 | -5.513 | A: lower mean AE | 385.0 | 4.17224e-06 | True |
| mathematics | XGBoost | Random Forest | 66 | -2.239 | A: lower mean AE | 529.0 | 0.000230734 | True |
| mathematics | XGBoost | Persistence (baseline) | 66 | -2.2659 | A: lower mean AE | 568.0 | 0.000595636 | True |
| mathematics | TimesFM 2.5 + ILSA LoRA | TimesFM 2.5 (zero-shot) | 66 | -0.1585 | A: lower mean AE | 354.0 | 1.58147e-06 | True |
| reading | XGBoost | TimesFM 2.5 + ILSA LoRA | 66 | -4.7848 | A: lower mean AE | 438.0 | 2.0076e-05 | True |
| reading | XGBoost | Random Forest | 66 | -0.0999 | A: lower mean AE | 1088.0 | 0.910989 | False |
| reading | XGBoost | Persistence (baseline) | 66 | -2.2843 | A: lower mean AE | 609.0 | 0.00151559 | True |
| reading | TimesFM 2.5 + ILSA LoRA | TimesFM 2.5 (zero-shot) | 66 | -0.132 | A: lower mean AE | 654.0 | 0.00392386 | True |
| science | XGBoost | TimesFM 2.5 + ILSA LoRA | 66 | -0.8691 | A: lower mean AE | 913.0 | 0.218807 | False |
| science | XGBoost | Random Forest | 66 | -0.0211 | A: lower mean AE | 1069.5 | 0.818114 | False |
| science | XGBoost | Persistence (baseline) | 66 | 0.3531 | B: lower mean AE | 1069.0 | 0.815633 | False |
| science | TimesFM 2.5 + ILSA LoRA | TimesFM 2.5 (zero-shot) | 66 | -0.1662 | A: lower mean AE | 662.0 | 0.00460959 | True |

---

## 6. Ayrı araştırma hattı: extraction → W_j → Stage 5 LOCO

### 6.1 W_j Ağırlık Farkı (Downstream Evaluation)

Bu bölüm ana dokuz-modelli PISA ülke ortalaması benchmark'ından farklı bir downstream deneydir; sonuçları PISA 2025 yöntem sıralamasına eklenmemelidir.

| Metrik | Değer |
|---|---|
| Oracle MAE (altın etki satırlarıyla) | 21.7017 |
| Stage 5 Frozen MAE (mevcut ağırlıklarla) | 22.0963 |
| ΔMAE (oracle − frozen) | -0.3946 |
| L1 uzaklığı (W frozen → W oracle) | 1.2463 |
| Spearman ρ (W frozen, W oracle) | 0.3354 |
| Yön uyumu (W frozen, W oracle) | 0.1818 |
| W_j'ye eşlenen kanıt satırı (oracle) | 66 |

**Yorum:** ΔMAE = oracle − frozen = −0.3946 olduğundan oracle MAE 0.3946 puan daha düşüktür (daha iyidir); eski yorumdaki 'oracle daha kötü' ifadesi ters işaretlidir. Oracle koşulu değerlendirilen split'in altın etiketlerini kullandığından deploy edilebilir extraction kolu veya bağımsız genelleme kanıtı değildir. Ağırlık benzerliği ölçüleri bu ayrı pipeline'a özgüdür.

### 6.2 Noise Dose-Response (Yön Çevirme)

### Tablo 7 — Yön Flip Pertürbasyonu (0–100%; 100 tekrar/düzey)

| Flip oranı | MAE ort. | 2.5% yüzdeliği | 97.5% yüzdeliği |
| --- | --- | --- | --- |
| 0.0 | 21.7017 | 21.7017 | 21.7017 |
| 0.1 | 21.7165 | 21.7017 | 21.7972 |
| 0.25 | 21.721 | 21.7017 | 21.8044 |
| 0.5 | 21.7297 | 21.7017 | 21.8044 |
| 1.0 | 21.7017 | 21.7017 | 21.7017 |

### Tablo 8 — Çift Düşürme Pertürbasyonu (0–100%; 100 tekrar/düzey)

| Drop oranı | MAE ort. | 2.5% yüzdeliği | 97.5% yüzdeliği |
| --- | --- | --- | --- |
| 0.0 | 21.7017 | 21.7017 | 21.7017 |
| 0.1 | 21.7761 | 21.6707 | 21.931 |
| 0.25 | 21.8208 | 21.5846 | 22.0792 |
| 0.5 | 21.8632 | 21.564 | 22.1854 |
| 1.0 | 21.7647 | 21.7647 | 21.7647 |

**Yorum:** Bu sentetik pertürbasyon tasarımında flip/drop ortalamaları baseline çevresindedir; %100 flip sonucunun baseline ile aynı olması pertürbasyon semantiğinin raporlanmasını gerektirir. Bu sonuç gerçek extraction hatalarının dağılımına genellenemez ve düşük duyarlılığı kanıtlamaz. Construct swap pertürbasyonu ise MAE'yi ~2 puan düşürmektedir — bu bir artifakt: yanlış construct seçimi tahmin dışı kalan bir predictor ağırlığını arttırarak bazen MAE'yi iyileştirebilir.

---

## 7. Ayrı araştırma hattı: extraction fine-tuning

| Kol | Model | Durum | Val Strict F1 |
|---|---|---|---|
| Oracle ceiling | — | ✅ Tamamlandı | 0.778 |
| A0 base_qwen3_chunked | Qwen3-8B | ⬜ Yerel v2 değerlendirme çıktısı yok | — |
| A1 qwen3_lora_selected | Qwen3-8B + LoRA | ⬜ Yerel v2 değerlendirme çıktısı yok | — |
| A2 base_mistral_chunked | Mistral-7B | ⬜ Yerel v2 değerlendirme çıktısı yok | — |
| A3 mistral_lora_selected | Mistral-7B + LoRA | ⬜ Yerel v2 değerlendirme çıktısı yok | — |
| P3 zeroshot_qwen27b | qwen/qwen3.8-27b (sağlayıcı çıktı kaydında yok) | ✅ Tamamlandı | **0.000** |
| Stage 7/9 LoRA (geçersiz) | Mistral-7B | ❌ 0/256 tensor yüklendi | — |

**Yorum sınırı:** Zero-shot Qwen kolundaki strict F1=0.000 tek başına domain fine-tuning'in gerekli ya da üstün olduğunu kanıtlamaz. Aynı tahminlerin matcher tanımına göre farklı F1 alması ölçüm duyarlılığı gösterir. Geçerli base/LoRA kıyası için aynı split, aynı çıkarım/değerlendirme ayarları ve tamamlanmış eşleşik kollar gerekir. Yerel v2 değerlendirme envanterinde oracle ve zero-shot Qwen çıktıları bulunuyor; eşleşik base/LoRA kolları görünmüyor.

**Sonraki adımlar:**
1. Önce doğrulama bölmesinde aynı ayarlarla eşleşik base, retrieval/prompt ve LoRA kollarını tamamla.
2. Eğitim kollarını GPU'da çalıştır ve her adapter'ın yüklendiğini doğrula.
3. Checkpoint'i validation strict extraction metriğiyle seç; kolları aynı test ayarlarında değerlendir.
4. Test bölümünü yalnızca protokol, model ve kod kilitlendikten sonra bir kez çalıştır.

---

## 8. Metodolojik Değerlendirme

### ✅ Doğru Yapılanlar
- Literatürden en sık kullanılan yöntemler (RF, XGBoost) seçilmiş ve uygulanmış
- Retrospektif backtesting prensibi doğru: hedef döngüden önceki verilerle eğitim
- TimesFM'e ILSA bağlamı verilmiş (unified_ilsa_panel ile LoRA adaptasyonu)
- 2025 tahminleri 66 ülke için üretilmiş
- Ayrı bir downstream deneyde sentetik W_j/noise duyarlılığı incelenmiş; sonuçları ana PISA benchmark'ından ayrılmıştır
- Fine-tuning v2 protokolü belgelenmiş; önceden dışarıda kayıt altına alındığı doğrulanmamıştır; geçersiz Stage 7/9 sonucu ayrıca belgelenmiştir

### ⚠️ Dikkat Edilmesi Gereken Sorunlar
- **TimesFM LoRA katkısı marjinal (~0.09 MAE)** — ILSA context injection yeterli iyileştirme sağlamıyor
- **Pooled overprediction betimseldir** — domain bazında bütün yöntemler pozitif sapma göstermiyor; bu sonuçtan neden/etki çıkarılamaz
- **W_j/noise sonuçları ayrı exploratory downstream tasarıma özgüdür**; PISA benchmark'ına doğrudan aktarılmaz
- **Fine-tuning kolları eksiktir** — yerel v2 envanteri eşleşik base/LoRA karşılaştırmasını göstermiyor

### ❌ Geçersiz Sayılan Sonuçlar
- Stage 7/9 Mistral LoRA F1 = 0.379 → base model sonucu, LoRA değil
- P3 zero-shot F1 = 0.000 → yalnızca belirtilen strict matcher ve kol için sonuç; fine-tuning üstünlüğü kanıtı değildir

---

## 9. Özet

| Soru | Cevap |
|---|---|
| Tahminsel literatürde en sık kodlananlar? | RF 42/147, XGBoost 17/147; ülke düzeyi altkümede SVM 4/17 |
| 2025 alan liderleri? | Matematik XGBoost; okuma Ridge; fen Persistence; üç-alan ortalaması Ridge |
| TimesFM ILSA LoRA işe yarıyor mu? | Tarihsel eşleşik MAE farkı <0.2; güçlü üstünlük kanıtı yok |
| Extraction kalitesi forecasting'i etkiliyor mu? | Mevcut sentetik-noise deneyi bunu kesinleştirmiyor |
| Fine-tuning tamamlandı mı? | Eşleşik base/LoRA kolları yerel envanterde tamamlanmış görünmüyor |
| 2025 tahminleri var mı? | Evet — 66 ülke/ekonomi × 3 alan; yöntem-alan MAE aralığı 9.318–19.584 |
| Bağımsız ileriye dönük doğrulama var mı? | Hayır; PISA 2029 için kod, veri, model ve tahminler sonuçlar açılmadan önce dondurulmalı |

## Kaynak ve yeniden üretim
- Ana tahmin panelleri: `outputs/pisa_all_models_quadratic_trend_v1/`.
- Literatür sıklığı/kapsam notu: `analysis/03_data_feasibility/phase3_11_corpus_method_frequency.md`.
- Araştırma amacı ve kanıt denetimi: `analysis/03_data_feasibility/phase3_33_objective_alignment_audit.md`.
- Bu sentezi üreten betik: `scripts/synthesize_forecast_comparison.py`.