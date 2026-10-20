# Phase 3.21 — PISA ölçek-link hatası envanteri ve kovaryans ön kontrolü

**Tarih:** 2026-10-15
**Amaç:** Phase 3.19’da açık kalan ölçek-link belirsizliğinin kaynaklarını sayısal olarak belirlemek ve yanlış bir kovaryans varsayımının önüne geçmek.

## Kaynaklardan doğrulanan link hata değerleri

OECD PISA 2022 Technical Report Ek Tablo 14.A.19, önceki döngü ile 2022 arasındaki alan bazlı robust link-error SE’lerini verir. Aşağıda tahmin tasarımında kullanılan önemli döngüler gösterilmiştir; hücreler `önceki döngü → sonraki döngü` yönündedir.

| Döngü çifti | Matematik | Okuma | Fen |
|---|---:|---:|---:|
| 2012 → 2022 | 3,58 | 6,01 | 5,20 |
| 2015 → 2022 | 2,74 | 3,63 | 1,38 |
| 2018 → 2022 | 2,24 | 1,47 | 1,61 |

Birden fazla döngü kullanan tarihsel değerlendirmede aynı tablo, erken PISA puanları için daha yüksek link hata ölçekleri olduğunu da gösterir: örneğin 2003→2022 matematik 5,55/okuma 5,25; 2006→2022 matematik 4,09/okuma 8,56/fen 3,68; 2009→2022 matematik 4,28/okuma 4,66/fen 5,92 puandır. PISA 2018 sonuç tablosu da 2012→2018 için matematik 3,34, okuma 3,74 ve fen 4,01 puan; PISA 2015 teknik raporu 2012→2015 için matematik 3,5462, okuma 5,2535 ve fen 3,9228 puan verir. [OECD PISA 2022 Technical Report](https://www.oecd.org/content/dam/oecd/en/publications/reports/2024/03/pisa-2022-technical-report_599753f0/01820d6d-en.pdf), [OECD PISA 2018 trend yöntemi ve tablosu](https://www.oecd.org/en/publications/pisa-2018-results-volume-i_5f07c754-en/full-report/component-24.html), [OECD PISA 2015 Technical Report](https://www.oecd.org/content/dam/oecd/en/about/programmes/edu/pisa/publications/technical-report/PISA2015_TechRep_Final.pdf).

PISA 2025 sonuçlarının Ek A3 tablosu, ayrı bir çıktı olarak önceki döngü bileşimleri ile 2025’i içeren **doğrusal trend link hatalarını** verir. Örneğin 2015, 2018, 2022 ve 2025 döngülerini içeren on yıllık trend için matematik 3,001, okuma 3,234, fen 3,034; 2018, 2022 ve 2025 bileşimi için sırasıyla 3,652, 2,617 ve 4,741 puandır. Bunlar 2018→2025 veya 2022→2025 ikili link hataları değildir ve bu çalışmanın ülke puanı forecast MAE’sine doğrudan eklenemez. OECD Reader’s Guide, PISA 2025 ile 2022 fen karşılaştırması için 2,9 puanlık ikili link hata örneği verir. OECD, PISA 2025 Technical Report’u mevcut sonuç sayfasında “forthcoming” olarak tanımlıyor; bu yüzden 2025’in tüm alan/döngü pairwise covariance yapısı burada doğrulanmış değildir. [OECD PISA 2025 Annex A3 teknik notları](https://www.oecd.org/en/publications/pisa-2025-results-volume-i_73451bc5-en/full-report/technical-notes-on-analyses-in-this-volume_46a7001c.html), [OECD PISA 2025 Reader’s Guide](https://www.oecd.org/en/publications/pisa-2025-results-volume-i_73451bc5-en/full-report/reader-s-guide_0364240f.html).

## Kovaryansın neden şu anda kullanılamadığı

OECD çoklu döngü trend standart hatalarında link hatalarının kovaryansını hesaba kattığını açıkça söyler. PISA 2012 teknik raporunda bazı link hata kovaryansları Tablo 12.32’de doğrudan mevcuttur. 2015/2018/2022 ve 2025 döngülerini bir arada kapsayan aynı ölçek yönelimi ve aynı kalibrasyon tanımında doğrulanmış tam kovaryans matrisi bu incelemede bulunmadı.

Bu eksikliği test etmek için 2012, 2015 ve 2018 puanlarının 2022 ölçeğine göre göreli link hataları için, kaynak tablolardaki her çiftin marjinal SE’sini kullanarak standart varyogram özdeşliğinden kovaryans türetildi:

`Cov(L_i, L_j) = [Var(L_i) + Var(L_j) − Var(L_i − L_j)] / 2`

Ancak elde edilen matematik kovaryans matrisinin özdeğerleri tümü pozitifken (yaklaşık 2,47; 6,16; 16,71), okuma (−2,59) ve fen (−0,12) matrislerinde negatif özdeğer oluştu. Bu, söz konusu matrislerin geçerli kovaryans matrisi olarak kullanılamadığını gösterir. Olası nedenler arasında rapor tablolarının farklı link tanımları/yöntemleri, raporlama yuvarlaması veya ikili hataların tek bir ortak gizil ölçek-offset sürecine tam uymaması bulunabilir; hangisinin geçerli olduğu henüz doğrulanmadı. Negatif özdeğerleri sıfıra kırpmak ya da bağımsızlık varsaymak bilimsel olarak gerekçelendirilmiş çözüm değildir.

Sıra her matriste `[2012, 2015, 2018]` döngüsüdür; sayılar puan² birimindedir:

```text
Matematik
[[12.8164,  3.8742,  3.3392],
 [ 3.8742,  7.5076,  3.5482],
 [ 3.3392,  3.5482,  5.0176]]

Okuma
[[36.1201, 10.8489, 12.1467],
 [10.8489, 13.1769, -0.0536],
 [12.1467, -0.0536,  2.1609]]

Fen
[[27.0400, 6.7780, 6.7760],
 [ 6.7780, 1.9044, 1.1082],
 [ 6.7760, 1.1082, 2.5921]]
```

## Sonuç ve gerekli sonraki adım

- PISA 2022’ye kadar olan önemli çiftler için link-error **standart sapmaları** birincil OECD kaynaklarından doğrulandı. Bu nedenle “link hatası kaynaklarda yok” demek doğru olmaz.
- Tüm model girdilerine/çıktılarına uygulanabilecek tutarlı, pozitif-yarı-tanımlı ve PISA 2025’i de kapsayan **ortak kovaryans yapısı henüz kurulamadı**. Bu nedenle Phase 3.19’daki hedef-SE ve ülke bootstrap aralıklarına link hatası eklenmedi.
- Bir sonraki teknik iş; PISA 2012/2015/2018/2022 teknik raporlarındaki link re-scaling tanımlarını, covariance tables/technical-report ek dosyalarını, PISA 2025 trend tablolarının türetim kuralını bir araya getirip her alan için ölçek zincirini resmi referans yönüne sabitlemektir. Ardından tüm çiftwise değerlerden kovaryans türetmek yerine varsa OECD’nin doğrudan kovaryans girdileri/2000 senaryoları kullanılmalı; matris simetrisi, pozitif yarı-tanımlılık ve her kaynak marjinal varyansının geri üretimi test edilmelidir.
- Bu eşik geçilene kadar mevcut MAE belirsizlik analizleri **örnekleme-SE + ülke kümeli koşullu duyarlılık** olarak adlandırılmalı; PISA dönemleri arası tam skor/link belirsizliği olarak sunulmamalıdır.

## Kaynak izi

- PISA 2012 Technical Report: Tablo 12.31 pairwise link-error SE; Tablo 12.32 doğrudan link hata kovaryansları. [OECD PDF](https://www.oecd.org/content/dam/oecd/en/about/programmes/edu/pisa/publications/technical-report/PISA-2012-technical-report-final.pdf)
- PISA 2015 Technical Report: Tablo 12.8 pairwise link-error değerleri. [OECD PDF](https://www.oecd.org/content/dam/oecd/en/about/programmes/edu/pisa/publications/technical-report/PISA2015_TechRep_Final.pdf)
- PISA 2018 Results, Annex A7: 2018 ve eski döngüler pairwise link errors; çoklu trend için kovaryans gerektiği açıklaması. [OECD sayfası](https://www.oecd.org/en/publications/pisa-2018-results-volume-i_5f07c754-en/full-report/component-24.html)
- PISA 2022 Technical Report: Tablo 14.A.19, 2022’ye göre pairwise link errors. [OECD PDF](https://www.oecd.org/content/dam/oecd/en/publications/reports/2024/03/pisa-2022-technical-report_599753f0/01820d6d-en.pdf)
- PISA 2025 Results Volume I: Ek A3 çoklu-döngü trend link errors; teknik raporun yayımlanma durumu ve 2025 Reader’s Guide. [OECD Annex A3](https://www.oecd.org/en/publications/pisa-2025-results-volume-i_73451bc5-en/full-report/technical-notes-on-analyses-in-this-volume_46a7001c.html), [Reader’s Guide](https://www.oecd.org/en/publications/pisa-2025-results-volume-i_73451bc5-en/full-report/reader-s-guide_0364240f.html).
