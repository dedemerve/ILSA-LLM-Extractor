# RQ1–RQ2 Uygulama ve Değerlendirme Protokolü

## Araştırma soruları

**RQ1.** Which machine learning models are most frequently used to predict student achievement in ILSA research, and how does the forecasting performance of the most prevalent models, particularly Random Forest and XGBoost, compare with that of time-series models in predicting country-level mean achievement scores across ILSA programs and domains?

**RQ2.** To what extent can a structured knowledge base derived from the ILSA research corpus be integrated with time-series forecasting models to improve predictions of country-level mean achievement scores across ILSA programs and domains?

Bu protokol, kapsamı yerel veri ve model çıktılarının gerçekten desteklediği düzeyde tanımlar. Sonuçlardan önce belirlenmiş başarı eşiği veya nedensel iddia üretmez.

## RQ1: Literatür yöntemi sıklığı

- `analysis/03_data_feasibility/audit/ilsa_ml_method_frequency_v1.csv` içindeki `ml_primary` frekanslarını iki ayrı paydayla raporla:
  1. Genel predictive çalışmalarda birincil yöntemi kodlanmış **147** kayıt: Random Forest 42, XGBoost 17, Neural Network 16, SVM 14.
  2. Dar ülke-düzeyi görev-D alt kümesindeki **17** kayıt: SVM 4, Random Forest 3, XGBoost 2.
- Bu iki kapsamı birbirine karıştırma. Genel tahmin korpusundaki RF/XGBoost sıklığını gelecek döngüye yönelik ülke ortalama skoru tahmini literatürünün sıklığı diye sunma.
- Kodlanmış korpusun eksiksiz bir sistematik tarama olduğu kanıtlanmış değildir. Arama sorgusu, yürütme tarihi, ham dışa aktarımlar, hit sayıları ve tarama kayıtları eksiktir. Ek bulunan kaynaklar eski frekans paydasına sessizce eklenemez; yeni, tarihli ve sürümlenmiş tarama gerekir.

## RQ1: Eşleştirilmiş performans karşılaştırması

- Kaynaklar: `outputs/ilsa_tabular_benchmark_v1/predictions.csv` ve PISA için `outputs/pisa_timesfm25_ils_adapter_v1/all_models_historical_predictions.csv`.
- Karşılaştırmayı program × alan × hedef döngü × ülke ortak anahtarlarında yap. Her grup içinde tüm kıyaslanan modeller için eksiksiz ortak ülke hedeflerini kullan.
- Programlar: PISA, TIMSS Grade 4, TIMSS Grade 8, PIRLS, ICCS, ICILS. PIAAC iki döngüyle çoklu rolling-origin karşılaştırmasına uygun olmadığı için dışarıda bırakılır. TALIS’in başarı puanı hedefi yoktur.
- Tüm programlarda Persistence, LinearTrend, Ridge, RandomForest ve XGBoost değerlendirilir. PISA’da aynı dört ortak tarihsel fold üzerinde TimesFM 2.5 zero-shot, TimesFM 2.5 ILSA-LoRA ve TimesFM 3.0 zero-shot eklenir. TimesFM kapsamını diğer programlara genelleme.
- Program-alan düzeyinde fold-başına MAE/RMSE, eşit-ağırlıklı fold MAE/RMSE, ortak örneklem büyüklüğü, ülke sayısı ve hedef döngüleri ver. Persistence’ın her grupta birinci olmasını veya RF/XGBoost’un belirli bir sıralama örüntüsünü ön koşul yapma.
- Persistence ve LinearTrend’e karşı RF/XGBoost ve PISA TimesFM kontrastları için ülke-küme paired bootstrap %95 aralıklarını üret. Bunlar mevcut tahmin ve fold’lara koşullu aralıklardır; model yeniden eğitimi, skor ölçüm/link hatası, döngüler arası belirsizlik veya çoklu karşılaştırma düzeltmesi içermez. Tek fold ve çok az ortak ülke bulunan ICCS/ICILS sonuçları özellikle ihtiyatlı yorumlanır.

## RQ2: Literatür bilgisi entegrasyonu

- Yapılandırılmış bilgi kaynağı `outputs/stage5/predictor_weights_v2.csv` ve eşleştirilmiş lagged ILSA bağlamsal değişkenleridir.
- Ana müdahale, mevcut LOCO tahmin betiğindeki M1 literatür ağırlıklı Ridge modelidir. M0 ağırlıksız Ridge ablation’ıdır; M2 Persistence zaman-serisi temel karşılaştırmasıdır. M1’de literatür ağırlığı `sqrt(w_j)` ölçeklemesiyle uygun lag kovaryatlarına uygulanır; skor/geçmiş omurgası nötr kalır.
- RQ2 performans örneklemini gerçek ülke düzeyindeki başarı sonuçlarıyla sınırla: PISA math/reading/science, TIMSS G4 ve G8 math/science, PIRLS reading, ICCS civic knowledge, ICILS computer literacy. ICILS ebeveyn eğitimi, SES ve evde okuryazarlık etkinlikleri gibi tahmin kovaryatlarını sonuç alanı sayma.
- M0–M1’i aynı ülke-hedef satırlarında karşılaştır; ayrıca M1–Persistence kıyaslamasını ayrı ortak örneklemle ver. Fold metriklerini eşit ağırlıkla birleştir; MAE/RMSE, fark ve M1 lehine göreli MAE değişimini raporla.
- Eşleştirilmiş MAE farkları için ülke-küme bootstrap %95 aralıklarını üret. Sonuçlar gözlemsel tahmin kıyaslamasıdır; nedensellik göstermez.
- TimesFM LoRA yalnızca ILSA döngü puanlarıyla yapılan adaptasyondur; predictor-weight bilgi tabanı entegrasyonu değildir. PISA LoRA–zero-shot sonuçlarını tamamlayıcı, ayrı bir analiz olarak tut.
- TimesFM 2.5 entegrasyonunu ayrıca sınamak için `scripts/run_timesfm_kb_integration.py` çalıştır. Kaynak zero-shot tahminleri `outputs/pisa_timesfm25_ils_adapter_v1/all_models_historical_predictions.csv` içinden al; 2022 PISA üç alan hedefinde aynı ülke-alan çiftlerini kullan; `lag_ESCS` ve `lag_HOMEPOS` ağırlıklarını `predictor_weights_v2.csv` içinden al. Her alan için yalnızca önceki hedef döngülerinde bulunan eksiksiz kovaryat ve TimesFM hata kayıtlarıyla ağırlıksız ve literatür-ağırlıklı Ridge hata düzeltmesini eğit. Baseline, bias-only düzeltme ve TimesFM zero-shot olmalı. Eğitim/ölçekleme hedefe göre geçmiş veride fit edilmeli; aynı ortak test çiftlerinde MAE/RMSE ve ülke-küme bootstrap raporlanmalı.
- TimesFM 2.5 resmi paketinin `forecast_with_covariates()` API’si native XReg sağlar; bu API’yi native XReg olarak raporlamak için resmi checkpoint gerçekten yüklenip tahmin üretilmiş olmalıdır. `scripts/run_timesfm_kb_integration.py` mevcut TimesFM 2.5 tahminleri üstünde XReg-benzeri hata düzeltme pilotudur ve native XReg değildir. Mevcut koşuda yalnızca 2022 PISA test edilebilmiş; kovaryat kapsamı kalibrasyonu 2015 hedef yılına sınırlamıştır. Literatür ağırlıklı Ridge, bu pilotta ağırlıksız Ridge’i iyileştirmemiştir. Sonucu tüm ILSA programlarına veya tüm zaman-serisi modellerine genelleme.
- Çalıştırma ve doğrulama kaydı `outputs/timesfm_kb_integration_v3/` altındadır. Betik sözdizimi kontrolünden geçirilmiş, v3 çıktısı üretilmiş; hedef puanı/panel eşleşmesi (1e-6 tolerans), tekil anahtarlar, dört kol için ortak test anahtarları, hata aritmetiği, girdi hash’leri ve MAE/RMSE bağımsız yeniden hesabı doğrulanmıştır. Yeniden çalıştırmalar v1–v3 aynı tahminleri ve metrikleri vermiştir.

## Uygulama ve çıktılar

- `scripts/merge_benchmark_results.py` v1’i koruyarak toplu RQ tablolarını `outputs/ilsa_benchmark_merged_v2/` içine üretir; var olan hedef klasör üzerine yazılmaz.
- `scripts/build_rq_evidence.py` mevcut model tahminlerini yeniden eğitmeden eşleştirilmiş performans ve belirsizlik tablolarını üretir. Girdi hash’leri ve yöntem manifestiyle izlenebilir bir `outputs/rq_evidence_vN/` sürümü oluşturur; var olan klasörü değiştirmez.
- RQ kanıt çıktıları: model/fold metrikleri, paired country-cluster bootstrap kontrastları, 10 başarı program-alan çiftinde M0/M1 ve M1/Persistence karşılaştırmaları, kapsam ayrımlı yöntem frekansları ve JSON özet.
- Son sentez: bulguları RQ1-A literatür sıklığı, RQ1-B eşleştirilmiş performans, RQ2 M0–M1 entegrasyon etkisi, M1–Persistence karşılaştırması, TimesFM bilgi entegrasyonu pilotu ve ayrı LoRA analizi olarak sun. Veri kapsamı, ortak örneklem, belirsizlik ve dış doğrulama sınırlılıklarını belirt.

## Bu protokoldeki tamamlanma ve açık kalan kapılar

Yerel analiz tamamlanmış sayılabilmesi için eşleştirilmiş tahmin anahtarları tekil olmalı, v1 kaynak çıktılar korunmalı, RQ1/RQ2 tablolarındaki sayılar girdi tahminlerinden yeniden hesaplanabilir olmalı ve metodolojik sınırlılıklar manifest/raporda yer almalıdır.

Özgün veri tabanı arama kayıtları yerel depoda yoksa literatürün eksiksizliği doğrulanmış sayılamaz. Bağımsız ikinci insan taraması gerçek bir ikinci gözden geçirici olmadan tamamlanmış sayılamaz. Bu kapılar raporda açık kalır; yerel analiz sonucu gibi gösterilmez.
