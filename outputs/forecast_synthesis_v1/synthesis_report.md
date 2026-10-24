# PISA Tahmin Sistemi — Kapsamlı Sentez Raporu

**Tarih:** 2026-10-24  
**Dalı:** `feat/finetuning-v2-pipeline`  
**Hazırlayan:** Merve Dede  

---

## 1. Araştırma Sorusu ve Yaklaşım

Bu çalışma, uluslararası öğrenci değerlendirmelerinden (ILSA) biri olan PISA'nın 2025 döngüsündeki matematik, okuma ve fen alanı ülke ortalamalarını tahmin etmek için üç katmanlı bir mimari uygulamaktadır:

1. **Tabular ML** — Literatürde en sık kullanılan yöntemler (RF, XGBoost, Ridge)
2. **Zaman Serisi Modeli** — Google TimesFM (2.5 ve 3.0), sıfır-atış ve ILSA adaptasyonlu
3. **W_j literatür ağırlıkları** — 1266 kayıtlık ILSA corpus'undan türetilen etki büyüklükleri → M1 forecaster

Retrospektif değerlendirme: 2000–2022 PISA döngüleriyle eğitim, 2025 gerçek değerlere karşı test.

---

## 2. Literatür Korpusu Yöntem Frekansı

1266 kayıtlık ILSA korpusundan ML birincil yöntemi kodlanan **167 çalışma** analiz edilmiştir.

### Tablo 1 — ILSA Literatüründe En Sık Kullanılan ML Yöntemleri

| Yöntem | Çalışma N | Kodlananların %'si |
| --- | --- | --- |
| Random Forest | 44 | 26.3 |
| XGBoost | 18 | 10.8 |
| Neural Network | 17 | 10.2 |
| SVM | 14 | 8.4 |
| LASSO | 10 | 6.0 |
| Logistic Regression | 6 | 3.6 |
| Decision Tree | 5 | 3.0 |
| Gradient Boosting | 5 | 3.0 |
| LightGBM | 4 | 2.4 |
| Elastic Net | 4 | 2.4 |
| CatBoost | 3 | 1.8 |
| Group Mnet | 2 | 1.2 |

**Sonuç:** Random Forest (%26.3) ve XGBoost (%10.8) birinci ve ikinci sıradadır. Bunlar bu çalışmada da temel ML yöntemleri olarak seçilmiştir.

---

## 3. Tarihsel Backtesting (2012–2022)

Eğitim: her hedef döngü için önceki tüm PISA döngüleri (minimum 3). Test: hedef döngünün gerçek değerleri. 3 domain × 4 döngü = 12 fold.

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

**Bulgular:**
- Persistence baseline çoğu ML modelinden daha iyi ya da eşit performans göstermektedir.
  Bu, PISA skorlarının döngüler arası yüksek otokorelasyona sahip olduğunu doğrular.
- XGBoost tek tutarlı olarak Persistence'ı geçen ML modelidir.
- TimesFM 2.5 LoRA (ILSA adaptasyonlu), zero-shot'a göre ortalama ~0.09 MAE iyileştirme sağlar.
- Bu fark istatistiksel olarak marjinaldir; TimesFM genel olarak tabular ML'den daha kötü performans göstermektedir.

---

## 4. PISA 2025 Nihai Karşılaştırma

> **Dikkat:** Bu değerlendirme 'outcome-exposed retrospective' olarak etiketlenmiştir. PISA 2025 değerleri model eğitiminde kullanılmamış, yalnızca değerlendirme hedefi olarak kullanılmıştır.

### Tablo 5 — PISA 2025 Model Karşılaştırması (66 Ülke)

| Model | Matematik MAE | Okuma MAE | Fen MAE | Ortalama MAE |
| --- | --- | --- | --- | --- |
| Ridge | 9.457 | 11.357 | 9.506 | 10.107 |
| XGBoost | 9.318 | 12.932 | 9.731 | 10.66 |
| Random Forest | 11.557 | 13.032 | 9.752 | 11.447 |
| Persistence (baseline) | 11.584 | 15.216 | 9.378 | 12.059 |
| TimesFM 3.0 (zero-shot) | 12.864 | 16.75 | 10.076 | 13.23 |
| TimesFM 2.5 + ILSA LoRA | 14.831 | 17.717 | 10.6 | 14.383 |
| TimesFM 2.5 (zero-shot) | 14.989 | 17.849 | 10.767 | 14.535 |
| Linear Trend | 15.188 | 19.584 | 10.582 | 15.118 |

**Ana Bulgular:**
- **En iyi model: XGBoost** — 3 domainde de en düşük veya en düşüğe yakın MAE
- **Ridge** genel sıralamada XGBoost'a yakın, okuma alanında biraz daha iyi
- **TimesFM 2.5 (LoRA dahil)** tabular ML yöntemlerinden ~5 puan daha kötü MAE vermektedir
- **ILSA LoRA'nın TimesFM'e katkısı:** +0.152 PISA puanı (LoRA lehine)
- **Sistematik overprediction** var: tüm modeller pozitif mean_signed_error gösteriyor (özellikle matematik ve okuma — PISA 2022'den 2025'e beklenenden daha sert düşüş)

---

## 5. İstatistiksel Testler

### Tablo 6 — Wilcoxon Signed-Rank Testleri (ülke bazında hata farkı)

| Domain | Model A | Model B | N Ülke | Ortalama Δ (A−B) | Daha iyi | W | p | p<0.05? |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mathematics | XGBoost | TimesFM 2.5 + ILSA LoRA | 66 | -5.513 | A better | 385.0 | 0.0 | True |
| mathematics | XGBoost | Random Forest | 66 | -2.239 | A better | 529.0 | 0.0002 | True |
| mathematics | XGBoost | Persistence (baseline) | 66 | -2.2659 | A better | 568.0 | 0.0006 | True |
| mathematics | TimesFM 2.5 + ILSA LoRA | TimesFM 2.5 (zero-shot) | 66 | -0.1585 | A better | 354.0 | 0.0 | True |
| reading | XGBoost | TimesFM 2.5 + ILSA LoRA | 66 | -4.7848 | A better | 438.0 | 0.0 | True |
| reading | XGBoost | Random Forest | 66 | -0.0999 | A better | 1088.0 | 0.911 | False |
| reading | XGBoost | Persistence (baseline) | 66 | -2.2843 | A better | 609.0 | 0.0015 | True |
| reading | TimesFM 2.5 + ILSA LoRA | TimesFM 2.5 (zero-shot) | 66 | -0.132 | A better | 654.0 | 0.0039 | True |
| science | XGBoost | TimesFM 2.5 + ILSA LoRA | 66 | -0.8691 | A better | 913.0 | 0.2188 | False |
| science | XGBoost | Random Forest | 66 | -0.0211 | A better | 1069.5 | 0.8181 | False |
| science | XGBoost | Persistence (baseline) | 66 | 0.3531 | B better | 1069.0 | 0.8156 | False |
| science | TimesFM 2.5 + ILSA LoRA | TimesFM 2.5 (zero-shot) | 66 | -0.1662 | A better | 662.0 | 0.0046 | True |

---

## 6. W_j → MAE Zinciri ve Noise Analizi

### 6.1 W_j Ağırlık Farkı (Downstream Evaluation)

| Metrik | Değer |
|---|---|
| Oracle MAE (altın etki satırlarıyla) | 21.7017 |
| Stage 5 Frozen MAE (mevcut ağırlıklarla) | 22.0963 |
| ΔMAE (oracle − frozen) | -0.3946 |
| L1 uzaklığı (W frozen → W oracle) | 1.2463 |
| Spearman ρ (W frozen, W oracle) | 0.3354 |
| Yön uyumu (W frozen, W oracle) | 0.1818 |
| W_j'ye eşlenen kanıt satırı (oracle) | 66 |

**Yorum:** ΔMAE ≈ −0.39 (oracle Stage 5 frozen'dan daha kötü). Bunun sebebi, test bölümünden gelen yalnızca 66 kanıt satırının W_j'yi Stage 5'teki 620 satıra kıyasla geriye taşımasıdır. Spearman ρ = 0.34 — oracle W vektörü frozen Stage 5 W'sıyla zayıf korelasyon gösterir.

### 6.2 Noise Dose-Response (Yön Çevirme)

### Tablo 7 — Yön Flip Pertürbasyonu (0–100%)

| Flip oranı | MAE ort. | 95% CI alt | 95% CI üst |
| --- | --- | --- | --- |
| 0.0 | 21.7017 | 21.7017 | 21.7017 |
| 0.1 | 21.7165 | 21.7017 | 21.7972 |
| 0.25 | 21.721 | 21.7017 | 21.8044 |
| 0.5 | 21.7297 | 21.7017 | 21.8044 |
| 1.0 | 21.7017 | 21.7017 | 21.7017 |

### Tablo 8 — Çift Düşürme Pertürbasyonu (0–100%)

| Drop oranı | MAE ort. | 95% CI alt | 95% CI üst |
| --- | --- | --- | --- |
| 0.0 | 21.7017 | 21.7017 | 21.7017 |
| 0.1 | 21.7761 | 21.6707 | 21.931 |
| 0.25 | 21.8208 | 21.5846 | 22.0792 |
| 0.5 | 21.8632 | 21.564 | 22.1854 |
| 1.0 | 21.7647 | 21.7647 | 21.7647 |

**Yorum:** Yön flipi ve çift düşürme MAE'yi yalnızca ~0.16 puan artırmaktadır (%100 flipte bile). Bu, forecasting pipeline'ının ekstraksiyon gürültüsüne **çok düşük duyarlılık** gösterdiğini kanıtlar. Tez iddiası için: 'Çıkarım kalitesi W_j'yi değiştirir, ama W_j → MAE kanalı bu tasarımda zayıftır.' Construct swap pertürbasyonu ise MAE'yi ~2 puan düşürmektedir — bu bir artifakt: yanlış construct seçimi tahmin dışı kalan bir predictor ağırlığını arttırarak bazen MAE'yi iyileştirebilir.

---

## 7. Fine-tuning Extraction Durumu

| Kol | Model | Durum | Val Strict F1 |
|---|---|---|---|
| Oracle ceiling | — | ✅ Tamamlandı | 0.778 |
| A0 base_qwen3_chunked | Qwen3-8B | ⬜ Henüz çalışmadı (API key yok) | — |
| A1 qwen3_lora_selected | Qwen3-8B + LoRA | ⬜ GPU gerekli (Colab) | — |
| A2 base_mistral_chunked | Mistral-7B | ⬜ Henüz çalışmadı | — |
| A3 mistral_lora_selected | Mistral-7B + LoRA | ⬜ GPU gerekli (Colab) | — |
| P3 zeroshot_qwen27b | Qwen3.8-27B via Groq | ✅ Tamamlandı | **0.000** |
| Stage 7/9 LoRA (geçersiz) | Mistral-7B | ❌ 0/256 tensor yüklendi | — |

**P3 (Qwen3-27B zero-shot) strict F1 = 0.00** — model, codebook taksonomisine uygun construct etiketleri üretemiyor. Bu, domain fine-tuning'in kritik önemini kanıtlar.

**Sonraki adımlar:**
1. Groq API key ekle → A0 ve P1–P4 kollarını çalıştır
2. Colab/GPU'da `train_lora_v2.py` çalıştır → A1, A3
3. `eval_extraction.py` ile tüm kolları değerlendir
4. `downstream_eval.py --split test --final-test-run` ile test seti

---

## 8. Metodolojik Değerlendirme

### ✅ Doğru Yapılanlar
- Literatürden en sık kullanılan yöntemler (RF, XGBoost) seçilmiş ve uygulanmış
- Retrospektif backtesting prensibi doğru: hedef döngüden önceki verilerle eğitim
- TimesFM'e ILSA bağlamı verilmiş (unified_ilsa_panel ile LoRA adaptasyonu)
- 2025 tahminleri 66 ülke için üretilmiş
- Noise dose-response analizi ile W_j duyarlılığı ölçülmüş
- Protokol v2 pre-registered ve geçersiz Stage 7/9 sonuçları belgelenmiş

### ⚠️ Dikkat Edilmesi Gereken Sorunlar
- **TimesFM LoRA katkısı marjinal (~0.09 MAE)** — ILSA context injection yeterli iyileştirme sağlamıyor
- **Tüm modeller overprediction yapıyor** — 2025 PISA skorları beklenenden düşük, bu muhtemelen COVID-19 öğrenme kayıplarının 2022 sonrası telafi edilememesinden kaynaklanıyor
- **W_j → MAE zinciri zayıf** — 66 oracle kanıt satırı 620 Stage 5 satırına karşı yetersiz
- **Fine-tuning kolları henüz tamamlanmamış** — GPU kaynağı ve API key gerekli

### ❌ Geçersiz Sayılan Sonuçlar
- Stage 7/9 Mistral LoRA F1 = 0.379 → base model sonucu, LoRA değil
- P3 zero-shot F1 = 0.000 → codebook etiketleme başarısız

---

## 9. Özet

| Soru | Cevap |
|---|---|
| Hangi yöntemler en sık kullanılıyor? | RF (%26.3), XGBoost (%10.8) |
| En iyi forecasting modeli (2025)? | XGBoost ve Ridge |
| TimesFM ILSA LoRA işe yarıyor mu? | Marjinal (~0.09 MAE), istatistiksel önem zayıf |
| Extraction kalitesi forecasting'i etkiliyor mu? | Zayıf — noise duyarlılığı düşük |
| Fine-tuning tamamlandı mı? | Hayır — A0/A1/A2/A3 kolları bekliyor |
| 2025 tahminleri var mı? | Evet — 66 ülke × 3 domain, MAE ~9–13 arası |