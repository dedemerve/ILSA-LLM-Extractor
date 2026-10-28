# PISA tahminlerinde hedef puan belirsizliği: örnekleme hatası ve ülke kümeli duyarlılık

**Durum:** Tamamlandı — bu adım, tahminleri yeniden eğitmeden mevcut sabit tahminlere uygulanmış koşullu bir duyarlılık analizidir. Tam araştırma doğrulaması değildir.

## Amaç ve kapsam

Ülke ortalama PISA puanları tahmin değil, örneklemden hesaplanan puan tahminleridir. Önceki değerlendirmeler nokta puanlarını gerçek değer gibi kullanıyordu. Bu ek analiz, yayımlanmış ülke puanı standart hatalarının (SE) tarihsel PISA hedeflerini ve 2025 OECD hedeflerini oynattığımızda model hata ölçütlerini ne kadar değiştirdiğini hesaplar.

İki ayrı analiz üretildi:

1. **Hedef-puan SE duyarlılığı:** Hedef puanları, ülke ve alan bazında `Normal(yayınlanmış ortalama, yayımlanmış SE)` çekilişleriyle değiştirildi; aynı çekiliş aynı ülke/dönemde bütün modellere uygulandı. Modeller ve tahminleri sabit tutuldu. Her senaryoda ülke mutlak hataları ve MAE yeniden hesaplandı.
2. **Birleşik ülke-kümesi + hedef-puan duyarlılığı:** Her tekrar içinde ülkeler alan bazında yerine koyarak yeniden örneklendi; her ülke/hedef-döngü puanı da yayımlanmış SE kullanılarak çekildi. Aynı ülkenin tekrarlanan döngüleri aynı bootstrap kümesinde kaldı ve her puan çekilişi modeller arasında eşleştirildi. Tarihsel sonuçta dört dış hedef döngüsünün MAE ortalaması eşit ağırlıklandı.

Her analiz 20.000 çekiliş, sabit tohum ve yüzde 2,5–97,5 persentil aralıklarını kullanır. `probability_better` veya `winner_probability`, belirtilen simülasyon/yeniden örnekleme kuralında elde edilen ampirik oranlardır; Bayesçi sonsal olasılık ya da doğrulayıcı p-değeri değildir. Çoklu model karşılaştırmaları için düzeltme yapılmadı.

## Veriler ve denetimler

- Tarihsel değerlendirme: PISA 2012, 2015, 2018 ve 2022 dış katlama tahminleri; model tahminleri [kaydedilmiş tarihsel tahmin dosyasından](../../outputs/pisa_timesfm25_ils_adapter_v1/all_models_historical_predictions.csv) alındı.
- Tarihsel SE: [ülke puanı tahmin dosyasından](../../outputs/stage4/country_estimates.csv) döngü, alan ve ülke anahtarıyla eşlendi. PISA 2000 okuma için doğrulanmış sayısal-ISO kod geçiş tablosu kullanıldı; bu değerlendirme hedefleri PISA 2000 olmadığı için tahmine katılmadı.
- PISA 2025: 66 ortak ülke/ekonomi ve üç alan için tahminler, OECD’nin hassas ülke ortalamaları ve yayımlanmış SE’leriyle [OECD ortalama denetim tablosundan](../../outputs/pisa2025_oecd_mean_audit_v3/country_domain_comparison.csv) eşlendi.
- Tarihsel hedef SE eşleşmeleri eksiksiz; 2025 OECD SE eşleşmeleri eksiksiz; model-ülke-döngü anahtarları benzersiz; 2025’te her alanda aynı 66 ülke/ekonomi var.
- Kaydedilmiş tarihsel tahminlerden yeniden hesaplanan nokta eşit-katlama MAE’leri ile birleşik analizdeki nokta tahminleri tam eşleşti. 2025 sonuçlarında OECD hassas hedef ortalamasına göre MAE’ler yeniden hesaplandı; panelin yuvarlanmış hedeflerine göre fark en çok 0,00018 MAE puanı.
- 2025 ülke hedef puanlarının eski proje panelindeki değerlerinden en büyük fark 0,01155 puan; bu nedenle doğrulama toleransı 0,012 puan olarak açıkça belirlendi.

## Bulgular

### Tarihsel dış katlamalar

Nokta tahminlerine göre Persistence üç alanda da en düşük eşit-katlama MAE’ye sahip: matematik 9,33; okuma 10,16; fen 8,44 puan. Hedef-puan SE’sini tek başına oynattığımızda Persistence’ın en iyi model olma sıklığı matematikte %95,8, okumada %94,1, fende %91,1 oldu. Ancak bu tek başına ülke örneklem değişkenliğini içermez.

Ülke-kümesi ve hedef-puan SE’sini birlikte hesaba kattığımızda Persistence’ın en iyi model olma sıklığı matematikte %64,4, okumada %50,8 ve fende %64,8’e indi. Tarihsel model belirsizliği tek başına hedef puan SE’siyle görünenden daha büyüktür; özellikle okuma alanında XGBoost (%23,5), ILSA-LoRA ile uyarlanmış TimesFM (%17,7) ve Ridge (%7,4) da birinci olabildi.

Uyarlanmış TimesFM’in kendi sıfır-atış TimesFM 2.5 karşısındaki hedef-puan SE-only MAE farkı (LoRA eksi zero-shot) üç alanda da negatiftir. Fakat birleşik analizin yüzde 95 aralığı:

| Alan | MAE farkı | Birleşik yüzde 95 aralığı | Simülasyonlarda daha iyi olma oranı |
|---|---:|---:|---:|
| Matematik | −0,100 | [−0,203; −0,013] | %98,7 |
| Okuma | −0,112 | [−0,184; −0,015] | %99,0 |
| Fen | −0,067 | [−0,150; +0,029] | %89,1 |

Dolayısıyla bu dört tarihsel katlama ve kullanılan bootstrap tasarımı altında matematik ve okumadaki küçük iyileşme tutarlı görünüyor; fende belirsizlik aralığı sıfırı kapsıyor. Bu, genellenebilir/prospektif üstünlük kanıtı değildir: model eğitimi, özellik belirsizliği, döngü belirsizliği ve ölçek-link hatası henüz yoktur.

### PISA 2025 retrospektif karşılaştırması

2025 nokta MAE’si en düşük modeller matematikte XGBoost (9,32), okumada Ridge (11,36), fende Persistence (9,38). Ülke bootstrap + OECD puan SE birleşik duyarlılığında birinci olma oranları:

| Alan | En sık birinci model | Birinci olma oranı | Diğer kayda değer aday |
|---|---|---:|---|
| Matematik | XGBoost | %54,3 | Ridge %45,7 |
| Okuma | Ridge | %99,6 | Random Forest %0,3 |
| Fen | Persistence | %42,5 | Ridge %26,7; XGBoost %12,5; Random Forest %6,8 |

Kalite bayraklı altı ülke/ekonomi çıkarılıp ortak 60 ülkelik analiz tekrarlandığında tablo benzer kaldı: matematikte XGBoost %53,0 ve Ridge %46,9; okumada Ridge %99,1 ile en sık birinci; fende Persistence %42,9, Ridge %18,5 ve XGBoost %16,4. Bu, ana bulguların bu altı ülke tarafından tek başına yaratılmadığını gösterir; fakat PISA 2025'in sonuçları önceden görüldüğünden kör/prospektif değerlendirme sorununu çözmez.

**Bunlar 2025 için kör test sonuçları değildir.** PISA 2025 sonuçları model çalışması başlamadan önce zaten projede mevcuttu; bu nedenle bu bölüm yalnızca sonuçları görüldükten sonraki retrospektif karşılaştırmadır.
Fen alanındaki liderlik belirgin değil. Matematikte XGBoost ve Ridge birbirine yakın; okuma sonucu bu duyarlılık tasarımında daha kararlı. **Bunlar 2025 için kör test sonuçları değildir.** PISA 2025 sonuçları model çalışması başlamadan önce zaten projede mevcuttu; bu nedenle bu bölüm yalnızca sonuçları görüldükten sonraki retrospektif karşılaştırmadır.

## Yorum ve yöntem sınırları

1. **Puan SE’si, link hatası değildir.** OECD, farklı değerlendirme yılları arasındaki ölçek karşılaştırılabilirliği için link hatasını örnekleme hatasından ayrı tanımlar. PISA 2025 rehberi, fen için 2022–2025 skor karşılaştırmasında 2,9 puanlık link hatası bildiriyor; bu, bu 66 ülkenin ortalama yayınlı fen puan SE’si olan yaklaşık 2,48 puandan büyüktür. Şu anki sayısal analiz link hatasını kapsamıyor. Dolayısıyla zaman içinde ölçek kaymasının model hatası ve sıralaması üzerindeki etkisi hâlâ çözülmesi gereken temel kapıdır. OECD’nin 2022 yöntem eki ayrıca birden fazla döngü için link hataları arasındaki kovaryansı hesaba katar; bağımsız, ülke başına rastgele gürültü eklemek doğru ikame değildir. [OECD PISA 2025 Reader’s Guide](https://www.oecd.org/en/publications/pisa-2025-results-volume-i_73451bc5-en/full-report/reader-s-guide_0364240f.html), [OECD PISA 2022 trend ve link hata yöntemi](https://www.oecd.org/en/publications/2023/12/pisa-2022-results-volume-i_76772a36/full-report/comparing-mathematics-reading-and-science-performance-across-pisa-assessments_1c417d6f.html), [OECD PISA 2022 Technical Report, Annex Table 14.A.19](https://www.oecd.org/content/dam/oecd/en/publications/reports/2024/03/pisa-2022-technical-report_599753f0/01820d6d-en.pdf).
2. **Bu analiz yalnızca hedef puanı değiştiriyor.** Tahminleri üreten geçmiş PISA bağlam puanları, ILSA özellikleri, model kurulumları ve tahminler sabit. Özellikle TimesFM eğitim/uyarlamasındaki eğitim örnekleri ve girdi puanlarının belirsizliği burada taşınmıyor. Tam belirsizlik yayılımında girdi puanlarını da uygun döngü/alan kovaryansıyla yeniden çekmek ve modelleri her tekrarda yeniden eğitmek gerekir.
3. **Normal ve bağımsız puan çekilişleri yaklaşımdır.** Ülke/döngü ortalama SE’siyle normal çekiliş, nokta tahmin çevresinde bir duyarlılık dağılımıdır; resmi öğrenci düzeyi PV ve BRR/Fay yeniden üretiminin yerini tutmaz. PISA puanları makul değerler (PV), final/replicate ağırlıkları ve anket tasarımını gözeterek analiz edilmelidir. OECD’nin [PISA veri hazırlama ve analiz kılavuzu](https://www.oecd.org/en/about/programmes/pisa/how-to-prepare-and-analyse-the-pisa-database.html) bu gereklilikleri açıklar.
4. **Ülke kümesi bootstrap’ı sınırlı ülke kümesine koşulludur.** Bu örneklem bütün eğitim sistemlerinin rastgele seçimi değildir. Tarihsel katlama sayısı yalnızca dört; yeni assessment döngülerine genelleme belirsizliği hesaplanmıyor.
5. **2025 karşılaştırması kalite bayraklı alt grupları da içeriyor.** Projedeki ayrı 2025 duyarlılık çalışması altı ülke/ekonomi (ALB, CAN, NLD, NOR, NZL, USA) için kalite bayrağı raporluyor. Bu birleşik SE/bootstrap analizi 66’nın tamamını kullanıyor; aynı birleşik yöntem kalite-temiz 60 ve diğer önceden tanımlı alt gruplarda tekrar edilmelidir.
6. **Sonuç, seçim yanlılığı ve çoklu deneme düzeltmesi içermiyor.** Modeller ve dört katlama önceden tek bir nihai protokol gibi dondurulmadığından bu aralıklar keşifseldir. Sonuçları “istatistiksel olarak kanıtlanmış üstünlük” şeklinde raporlamayın.

## Yeniden üretim ve dosyalar

- [Hedef SE duyarlılık betiği](../../scripts/summarize_pisa_target_sampling_uncertainty.py)
- [Birleşik puan SE + ülke kümeli bootstrap betiği](../../scripts/summarize_pisa_joint_score_country_uncertainty.py)
- [Hedef SE-only çıktıları ve manifest](../../outputs/pisa_target_sampling_uncertainty_v2/)
- [Birleşik belirsizlik çıktıları ve manifest](../../outputs/pisa_joint_score_country_uncertainty_v2/)

Her iki betik de çıktı dizini önceden varsa çalışmayı durdurur; önceki deney dosyaları veya tahminler değiştirilmedi. Çalıştırma manifestleri girdi SHA-256 özetlerini, tohumları, yöntem ve kapsam kontrollerini kaydeder.

## Sıradaki gerekli kapılar

1. OECD PISA 2025 ve önceki teknik raporlardaki alan/döngü link-hata değerlerini ve gerekli kovaryans tablolarını çıkar; tanımı, ölçek yönünü ve hangi döngü kıyaslarına uygulanabileceğini doğrula.
2. Link belirsizliğini, hedef ülke örneklemesi ve puan-SE bootstrap’ıyla ortak simülasyona kat. 2025 trend/karşılaştırma kuralları dışında tutulan ülkeleri bu analizden çıkar veya ayrı raporla; özellikle OECD’nin trend kullanım uyarılarını uygula.
3. 2025 kalite-temiz 60 ülke ve diğer önceden belirlenen ülke alt grupları için birleşik duyarlılığı aynı dondurulmuş kodla üret.
4. Tahmin girdisi/uyarlama belirsizliğini ayrıca ele al: her tekrarda hedef dışı geçmiş PISA puanlarını ve kullanılabilir ILSA ülke özetlerini belirsizliklerine göre yeniden üret, tüm preprocessing ve modelleri yalnızca ilgili eğitim döngülerinde yeniden fit et.
5. Tüm belirsizlik türlerini tek aralıkta birleştirmeden önce her bileşeni ayrı raporla; sonra bileşenler arası bağımlılıkların desteklenebildiği bir birleşik prosedür tanımla ve simülasyon kapsama kontrolleri yap.
6. Nihai önceden belirlenmiş model seçimi ve değerlendirme planını dondur; PISA 2025’i sadece retrospektif sonuç olarak etiketle. Gerçek ileriye dönük doğrulama için sonuçları görülmemiş bir sonraki PISA döngüsü veya tarihe göre kilitlenmiş ayrı bir değerlendirme kümesi gerekir.
