# RQ1 ve RQ2 Kanıt Raporu

**Tarih:** 25 Ekim 2026  
**Durum:** Yerel yeniden analiz, sınırlı tamamlayıcı literatür taraması ve TimesFM bilgi entegrasyonu pilotu tamamlandı. Sistematik literatür taraması, bağımsız ikinci insan taraması ve native TimesFM XReg doğrulaması açık kapılardır.

## Kapsam ve yöntem

Tahmin performansı, kaydedilmiş ülke düzeyi tahminler üzerinden yeniden hesaplandı; modeller yeniden eğitilmedi ve kaynak tahmin dosyaları değiştirilmedi. RQ1’de her program-alan için tüm kıyaslanan modellerin ortak ülke-hedef döngüleri kullanıldı. MAE ve RMSE önce hedef döngü başına hesaplandı, ardından fold’lara eşit ağırlık verildi. Eşleştirilmiş MAE farklarının belirsizliği, ülkeleri kümeler halinde yeniden örnekleyen 5.000 tekrarlı bootstrap ile özetlendi.

RQ2’de yalnızca asıl başarı sonuçları tutuldu: ICCS civic knowledge; ICILS computer literacy; PIRLS reading; PISA math/reading/science; TIMSS Grade 4 ve Grade 8 math/science. ICILS SES, ebeveyn eğitimi ve ev etkinlikleri gibi kovaryat hedefleri dışlandı. M0–M1 ve M1–Persistence karşılaştırmaları ayrı ortak örneklemlerde hesaplandı.

Makinece okunabilir tablolar, girdi SHA-256 değerleri ve hesaplama açıklamaları [rq_evidence_v6](/Users/mrved/Desktop/ILSA_LLMs/outputs/rq_evidence_v6) içindedir. Uygulama [build_rq_evidence.py](/Users/mrved/Desktop/ILSA_LLMs/scripts/build_rq_evidence.py) ile tekrarlanabilir. Birleştirme çıktıları [ilsa_benchmark_merged_v2](/Users/mrved/Desktop/ILSA_LLMs/outputs/ilsa_benchmark_merged_v2) içinde korunur.

## RQ1 — Kullanılan yöntemlerin sıklığı

Projenin kodlanmış korpusunda 1.266 çalışma kaydı bulunuyor; birincil yöntem kodu olan predictive çalışmaların 147’sinde Random Forest 42 (%28,6), XGBoost 17 (%11,6), Neural Network 16 (%10,9), SVM 14 (%9,5) ile ilk sıradaki yöntemlerdir. Bu, **kodlanmış genel predictive çalışma kümesinin** frekansıdır.

Daha dar `country_level_prediction_task_D` alt kümesinde birincil algoritması kodlanmış 17 kayıt arasında SVM 4, Random Forest 3, XGBoost 2 kayda sahiptir. Bu 17 çalışma bile doğrulanmış geleceğe dönük ülke-ortalaması tahminlerinden oluşan temiz bir alt küme değildir. Dolayısıyla “RF ve XGBoost, ülke ortalama puan tahmini literatüründe en sık yöntemlerdir” sonucu desteklenmiyor. Daha güvenli ifade: **RF ve XGBoost kodlanmış genel predictive ILSA korpusunda en sık iki birincil algoritma etiketidir; dar ülke-düzeyi görev alt kümesinde SVM öndedir.**

Sınırlı birincil kaynak taraması da dar literatürün seyrek ve heterojen olduğunu gösterdi. Öne çıkan örnekler, PISA 2015’ten PISA 2018 bilim puanlarını kestiren RF/XGBoost/SVR çalışması ([Acisli-Celik & Yesilkanat, 2023](https://doi.org/10.1007/s00521-023-08901-6)); sonraki TIMSS Grade 8 ülke ortalamalarını kalite kontrol amacıyla tahmin eden sinir ağı yaklaşımı ([Braun, von Davier & Chen, 2025](https://doi.org/10.1186/s40536-025-00262-x)); ve PISA 2025 matematik ilerlemesini modelleyen büyüme/stacking çalışmasıdır ([Kaplan et al., 2025](https://doi.org/10.1017/psy.2025.2)). Bunlar tahmin hedefi, zaman bölmesi ve değerlendirme tasarımı bakımından aynı deney değildir. Ayrıntılar ve arama sınırları [literatür notlarında](</Users/mrved/Desktop/ILSA_LLMs/research_notes/ILSA forecast literature update/country_mean_forecasts.md>) yer alır.

### RQ1 — Ortak test örneklemindeki performans

PISA’da Persistence sekiz yöntem arasından matematik, okuma ve fen alanlarının üçünde de en düşük eşit-fold MAE’yi verdi. Diğer program-alanlarda en düşük MAE 10 grubun 9’unda Persistence, ICILS computer literacy grubunda LinearTrend oldu. Aşağıdaki değerler puan cinsindedir; program ve alanlar arası ham MAE kıyaslanmamalıdır.

| Program / alan | En düşük MAE | Eşit-fold MAE | Fold | Ortak tahmin | Ülke |
|---|---|---:|---:|---:|---:|
| ICCS civic knowledge | Persistence | 13,04 | 1 | 7 | 7 |
| ICILS computer literacy | LinearTrend | 7,37 | 1 | 3 | 3 |
| PIRLS reading | Persistence | 12,90 | 2 | 50 | 36 |
| PISA mathematics | Persistence | 9,33 | 4 | 208 | 64 |
| PISA reading | Persistence | 10,16 | 4 | 208 | 64 |
| PISA science | Persistence | 8,44 | 4 | 208 | 64 |
| TIMSS Grade 4 mathematics | Persistence | 14,57 | 4 | 110 | 51 |
| TIMSS Grade 4 science | Persistence | 14,91 | 4 | 109 | 50 |
| TIMSS Grade 8 mathematics | Persistence | 14,61 | 5 | 122 | 48 |
| TIMSS Grade 8 science | Persistence | 15,35 | 5 | 122 | 48 |

PISA’da RF ve XGBoost, LinearTrend’ı üç alanın tamamında nokta tahmininde geçti; ülke-küme bootstrap aralıkları bu farklar için pozitifti. Buna karşılık RF ve XGBoost’un hiçbiri Persistence’tan daha düşük nokta MAE vermedi. RF/XGBoost ile Persistence arasındaki 20 eşleştirilmiş kontrastın 14’ünde aralık Persistence lehine sıfırın altındaydı; altısında belirsizlik sıfırı içerdi. TimesFM modelleri de PISA’da Persistence’ı geçmedi. Bu sonuçlar dört tarihsel PISA fold’una ve mevcut ülkelere koşulludur.

ICCS’te yalnızca 7, ICILS’te 3 ortak ülke ve tek fold vardır; bu iki program için sıralamalar genellenebilir kanıt sayılmaz. PIAAC iki döngüsü nedeniyle çoklu rolling-origin analize alınmadı; TALIS başarı puanı değerlendirmesi sunmadığı için dışarıda kaldı. TimesFM yalnızca PISA’da, 2012/2015/2018/2022 fold’larında değerlendirildi.

Ülke-küme aralıkları model tahminlerini sabit tutar ve yalnızca ülke örneklemi değişkenliğini yansıtır. Ölçme ve ölçek-link belirsizliği, tahmin modeli yeniden eğitimi ve model seçimi belirsizliği bu bootstrap’a dahil değildir. Aralıklar çoklu karşılaştırma için düzeltilmemiştir.

## RQ2 — Literatür bilgisi entegrasyonu

Mevcut müdahale, literatürden türetilmiş predictor ağırlıklarını uygun gecikmeli ILSA bağlam değişkenlerine uygulayan M1 Ridge modelidir. M0 aynı LOCO çerçevesindeki ağırlıksız Ridge ablation’ı; M2 Persistence zaman-serisi temelidir. Bu, bilgi tabanının tahmin modeline **sabit özellik ağırlıkları** yoluyla entegrasyonunu sınar. TimesFM LoRA ise ILSA puan serileriyle model adaptasyonudur; predictor ağırlığı bilgi tabanı müdahalesi değildir.

M1, 10 başarı program-alan çiftinin 9’unda M0’dan düşük MAE verdi; ICILS’te eşit sonuç aldı. Göreli MAE değişimi alanlara göre yaklaşık %0–%3,23 arasındadır. Beş çiftte ülke-küme bootstrap %95 aralığı M1 lehine sıfırın dışındaydı; diğer çiftlerde aralık sıfırı içerdi. Göreli kazanım PISA matematikte %0,10, okumada %3,23, fende %2,24; TIMSS Grade 4 matematikte %0,15 ve fende %0,76; TIMSS Grade 8 matematikte %0,53 ve fende %0,28; PIRLS okumada %1,20; ICCS’te %0,01’di. ICILS’te tahminler eşitti.

M1’in M0’a karşı MAE kazanımı her durumda RMSE kazanımına dönüşmedi: TIMSS Grade 4 matematik ve fende M1’in MAE’si düşükken RMSE’si M0’dan daha yüksekti. Bu, ortalama mutlak hatadaki sınırlı iyileşmenin büyük hataları azaltmayı garanti etmediğini gösterir.

M1, Persistence’ı 10 alanın hiçbirinde nokta MAE’de geçmedi; 8 alanda ülke-küme bootstrap aralığı Persistence lehine sıfırın dışındaydı. Bu nedenle RQ2’ye verilebilecek yanıt sınırlıdır: **Literatür ağırlıkları Ridge tabanlı, kovaryat kullanan zaman-serisi tahminine uygulanabilir ve mevcut LOCO sonuçlarında ağırlıksız Ridge’e göre çoğunlukla küçük iyileşmeler göstermiştir; ancak üstünlük tutarlı biçimde anlamlı değildir ve basit Persistence temelini aşmamıştır.** Sonuç tüm zaman-serisi mimarilerine veya TimesFM’e genellenemez.

### TimesFM entegrasyonu — 2022 PISA pilotu

TimesFM 2.5’in resmi Google Research paketinde `forecast_with_covariates()` XReg API’si vardır. Bu makinede kurulu pakette API mevcut olsa da API’nin beklediği resmi PyTorch checkpoint’i yerel önbellekte yoktu ve ağ erişimi sağlanmadı. Bu nedenle native XReg tahmini çalıştırılmadı. Bunun yerine, mevcut TimesFM 2.5 zero-shot tahminlerinin hatalarını önceki PISA döngülerinde öğrenen zaman-sıralı bir kovaryat düzeltme deneyi yapıldı. Bu deney gerçek zaman serisi tahminlerini girdi olarak kullanır fakat resmi XReg API’sinin çalıştırılmasıyla eşdeğer değildir. [TimesFM 2.5 resmi XReg açıklaması](https://github.com/google-research/timesfm).

2022 hedef döngüsündeki üç PISA alanında ortak, iki kovaryatı da mevcut **55 ülke (165 ülke-alan tahmini)** kullanıldı. Her alan için düzeltme modelleri yalnızca önceki hedef döngülerden eksiksiz kovaryatlı veriyi aldı; veri mevcudiyeti nedeniyle eğitim her alan için yalnızca 2015’e (51 ülke) düşüyordu. Kullanılan özellikler, 2022 satırında önceki PISA döngüsüne ait olan `lag_ESCS` ve `lag_HOMEPOS` idi; corpus ağırlıkları sırasıyla 1,000 ve 0,7175’tir. Esas karşılaştırma ağırlıksız kovaryat Ridge ablation’ı ile aynı Ridge’in literatür ağırlıklı biçimidir; `M0_bias_correction` basit hata-düzeltme kontrolüdür.

| Tahmin kolu | MAE | RMSE | TimesFM zero-shot’a göre MAE farkı* |
|---|---:|---:|---:|
| TimesFM 2.5 zero-shot | 11,581 | 15,669 | — |
| Sadece geçmiş ortalama hata düzeltmesi | 10,883 | 14,894 | 0,698 |
| Ağırlıksız kovaryat hata düzeltmesi | 10,910 | 14,968 | 0,670 |
| Literatür ağırlıklı kovaryat hata düzeltmesi | 10,911 | 14,967 | 0,669 |

*Pozitif fark, referans MAE’sinin model MAE’sinden yüksek olduğunu gösterir. Ülke kümeli eşleştirilmiş bootstrap ile literatür ağırlıklı kolun TimesFM zero-shot’a karşı farkı %95 aralıkta 0,126–1,234’tür. Ancak literatür ağırlıklı ve aynı kovaryatlı ağırlıksız kol arasındaki fark yalnızca −0,0009 puandır; %95 aralık −0,0062 ile 0,0047 arasındadır. Dolayısıyla bu pilotta hata düzeltmesi tahminleri iyileştirmiş görünse de, kazanım basit bias düzeltmesiyle de elde edilmekte ve **literatür ağırlığı ek fayda göstermemektedir**.

Bu kanıt yalnızca PISA 2022’ye, 55 ülkeye ve iki mevcut kovaryata koşulludur; 2015’ten öğrenilen düzeltme tek bir geçmiş döngüye dayanır. Tüm ILSA programlarına/alanlarına genelleme veya native XReg’in performans iddiası değildir. Makinece okunabilir, yeniden üretilmiş çıktılar ve girdi hash’leri [timesfm_kb_integration_v3](/Users/mrved/Desktop/ILSA_LLMs/outputs/timesfm_kb_integration_v3), yeniden çalıştırma betiği [run_timesfm_kb_integration.py](/Users/mrved/Desktop/ILSA_LLMs/scripts/run_timesfm_kb_integration.py) içindedir. Betik, kaydedilmiş hedef puanlarını PISA panelindeki puanlarla ve model kollarındaki ülke-alan anahtarlarıyla ayrıca doğrular.

## Sonuç ve kalan açık kapılar

Yerel veriyle yapılabilecek ortak-örneklem performans karşılaştırması, kapsam ayrımlı yöntem sıklığı, RQ2 ağırlık ablation’ı ve ülke-küme belirsizliği hesaplandı. Raporlanan dosyalar mevcut model tahminlerini yeniden kullanır; önceden üretilmiş modelleri yeniden eğitmez.

İki literatür kapısı tamamlanmış sayılamaz: (1) özgün Scopus/WoS sorgu geçmişi, ham sonuç dışa aktarımları, tarama ölçütleri ve eleme kayıtları yerel depoda bulunmadığından korpusun eksiksizliği doğrulanamıyor; (2) bağımsız ikinci insan taraması yapılmadı. Tamamlayıcı web taraması yayımlanmış örnekleri buldu ama sistematik veritabanı araması değildir. Bunları kapatmak için özgün dışa aktarımlar veya tarihli, sürümlenmiş yeni arama ve bağımsız insan kodlayıcı gerekir.

Ayrıca RQ2’nin geniş kapsamlı yanıtı için TimesFM entegrasyon kanıtı yalnızca tek bir PISA hedef döngüsünü kapsayan pilot düzeyindedir. Resmi TimesFM XReg checkpoint’i erişilebilir olduğunda aynı eşleştirilmiş hedeflerde native XReg yeniden çalıştırılmalıdır. Yeni bir bağımsız gelecek-döngü doğrulaması da sonuç yayımlanmadan önce kilitlenmiş tahmin ve daha sonraki bir hedef döngü gerektirir. Bu koşulları mevcut tarihsel veriyi değiştirerek tamamlamak yöntemsel olarak doğru olmaz.

## İzlenebilirlik

- Frekans ve arama sınırı: [frequency scope note](</Users/mrved/Desktop/ILSA_LLMs/research_notes/ILSA forecast literature update/method_frequency_scope.md>), [country-mean forecast review](</Users/mrved/Desktop/ILSA_LLMs/research_notes/ILSA forecast literature update/country_mean_forecasts.md>).
- Bilgi tabanı entegrasyonu metodolojisi: [knowledge base integration note](</Users/mrved/Desktop/ILSA_LLMs/research_notes/ILSA forecast literature update/knowledge_base_integration.md>).
- Çalıştırma protokolü: [RQ alignment prompt](</Users/mrved/Desktop/ILSA_LLMs/analysis/rq_alignment_implementation_prompt.md>).
