# Faz 0 — Köken (provenance) ve açık sorular

**Tarih:** 27 Ekim 2026
**Git:** `feat/finetuning-v2-pipeline` @ `ad7b10049fb998afeb9e2b0875bb601163ad2957`
**Girdi manifesti:** [00_input_manifest.json](00_input_manifest.json) — 22 girdi dosyası için boyut, mtime, sha256, git takip durumu.
**Kapsam:** Yalnızca okuma ve doğrulama. Hiçbir model yeniden eğitilmedi, mevcut hiçbir çıktı değiştirilmedi. Bu fazda yazılan dosyalar yalnızca bu belge ve manifesttir.

Etiketler: **VERIFIED** = bu fazda çalıştırılan kodla doğrulandı; **ASSUMPTION** = depo dışı bilgi veya doğrulanmamış çıkarım; **RECOMMENDATION** = öneri.

## 0.1 PISA 2025 değerlerinin kaynağı

**VERIFIED.** `unified_panel.csv` içindeki PISA 2025 ülke ortalamaları, resmi OECD kamu kullanım dosyasından (PUF) hesaplanmış gerçek tahminlerdir; model çıktısı değildir.

- Kaynak dosya: `/Users/mrved/Desktop/ILSA Datasets for LLM Extraction/PISA Datasets/PISA 2025 Data/CY09_MS_STU_PUF.sav` (2 168 666 883 bayt, mtime 2026-08-21, sha256 `6f6fa1e1ecce…`, tam değer manifestte).
- Kontrol: 10 PV ve `W_FSTUWT` ile ağırlıklı ülke ortalamaları yeniden hesaplandı (matematik). 89 ülkenin tamamında saklanan değerle fark 0.0000.
- Tahmin yöntemi `build_country_estimates.py` içinde `BRR_FAY`, 80 replikasyon, `W_FSTURWT`.
- Aynı klasörde `*-PISA-2025-Technical Report-AnnexTables_14092026.xlsx` dosyaları var; tablolar 14 Eylül 2026 tarihli.

**ASSUMPTION.** PISA 2025 sonuçlarının resmi yayın tarihi depodan doğrulanamadı. PUF dosyasının tarihi (Ağustos 2026) ve teknik rapor eklerinin tarihi (Eylül 2026), verinin bu tarihlerden önce kamuya açık olduğunu düşündürüyor.

**Sonuç ve RECOMMENDATION.** 2025 test fold'undan çıkarılmayacak; değerler resmi veriden geliyor. Ancak `pisa_forecasting_panel_v2_manifest.json` 2025'i `outcome_exposed_retrospective_benchmark_only` olarak işaretliyor. Ağırlık tablosu (25 Ekim) ve model kararları 2025 sonuçları görüldükten sonra düzenlendi. Bu yüzden 2025 temiz bir holdout değil. Faz 2–4'te ana sonuçlar hem 2025 dahil hem de 2025 hariç raporlanmalı.

## 0.2 Kovaryat zamanlaması

### Doğrulanan kaynak döngüleri (VERIFIED)

Her satır, paneldeki `lag_*` değerinin hangi kaynak döngüyle birebir eşleştiğini gösterir (max |fark| < 1e-6). Kontroller `covariate_estimates.csv`, `talis_covariate_estimates.csv` ve PISA 2018/2022/2025 mikroverisi üzerinden yapıldı.

| Hedef program | Hedef döngü | Kovaryat | Kaynak | Gecikme (yıl) | Not |
|---|---|---|---|---|---|
| PISA | 2015 | lag_ESCS, lag_HOMEPOS | PISA 2009 | 6 | 2012 kovaryatı hesaplanmamış |
| PISA | 2015 | lag_BSDGEDUP | TIMSS G8 2011 | 4 | programlar arası |
| PISA | 2015 | lag_ASDHEDUP | TIMSS G4 2011 | 4 | programlar arası |
| PISA | 2015 | lag_SECLSS, lag_TCDISCS | TALIS 2013 | 2 | |
| PISA | 2022 | lag_ESCS, lag_HOMEPOS, lag_BELONG | PISA 2015 | 7 | 2018 kovaryatı hesaplanmamış; mikroveri kontrolüyle 2018 ve 2022 ile eşleşmiyor |
| PISA | 2022 | TALIS (SECLSS, TCDISCS, TEFFPROS, TJSPROS) | TALIS 2018 | 4 | bkz. bulgu B |
| PISA | 2025 | lag_ESCS, lag_HOMEPOS, lag_BELONG | PISA 2022 | 3 | 2022 mikroverisiyle tam eşleşme; 2025 ile eşleşmiyor |
| PISA | 2025 | TALIS (4 değişken) | TALIS 2018 (2022 değerinin kopyası) | 7 | TALIS 2024 kullanılmamış; bkz. bulgu C |
| TIMSS G8 | 2007 | lag_BSDGEDUP | TIMSS 2003 | 4 | |
| TIMSS G8 | 2015 | lag_ESCS | PISA 2009 | 6 | programlar arası |
| TIMSS G8 | 2019 | lag_BSDGEDUP | TIMSS 2015 | 4 | |
| TIMSS G8 | 2023 | lag_BSDGEDUP | yok | — | 2023 için kovaryat yok |
| TIMSS G4 | 2015 | lag_ESCS | PISA 2009 | 6 | programlar arası |
| TIMSS G4 | 2019 | lag_ASDHEDUP | TIMSS G4 2015 | 4 | |
| PIRLS | 2011 | lag_BSDGEDUP | TIMSS G8 2007 | 4 | programlar arası |
| PIRLS | 2016 | lag_ASDHEDUP | PIRLS 2011 | 5 | |
| PIRLS | 2021 | lag_ASDHEDUP, lag_ASDHELA | PIRLS 2016 | 5 | |
| ICCS | 2016 | lag_ASDHEDUP | PIRLS 2011 | 5 | programlar arası, n=2 |
| ICCS | 2022 | lag_ESCS | PISA 2015 | 7 | programlar arası |
| ICCS | 2022 | TALIS | TALIS 2018 | 4 | |

**VERIFIED.** Hiçbir kovaryat, hedef döngüyle aynı yıldan veya hedef döngüden sonraki bir yıldan gelmiyor. Aynı yıldan veri sızıntısı yok.

**VERIFIED.** Gecikme sabit değil (2 ile 7 yıl arası) ve programlar arası eşleşmeler `groupby().first()` ile örtük olarak oluşuyor (örneğin TIMSS 2015 satırları PISA 2009 ESCS değerini alıyor). Model, gecikme uzunluğunu bilmiyor.

### Tahmin başlangıç noktasına (forecast origin) göre kontrol

Bir kovaryatın kullanılabilir olup olmadığı, tahminin hangi tarihte yapıldığı varsayımına bağlıdır. İki tanım mümkün:

- **Tanım A (katı tahmin):** Origin = hedef programın bir önceki döngüsünün sonuçlarının yayımlandığı tarih.
- **Tanım B (yayın öncesi nowcast):** Origin = hedef döngünün sonuçlarının yayımlanmasından hemen önce.

Yayın tarihleri **ASSUMPTION**'dır (depo dışı bilgiye dayanıyor; resmi kaynaklardan teyit edilmeli): PISA 2009 Ara 2010, PISA 2012 Ara 2013, PISA 2015 Ara 2016, PISA 2018 Ara 2019, PISA 2022 Ara 2023; TIMSS 2003 Ara 2004, 2007 Ara 2008, 2011 Ara 2012, 2015 Kas 2016, 2019 Ara 2020; PIRLS 2011 Ara 2012, 2016 Ara 2017; ICCS 2009 Kas 2010, 2016 Kas 2017; TALIS 2013 Haz 2014, TALIS 2018 Haz 2019, TALIS 2024 Eki 2025.

Tanım A'ya göre origin'den **sonra** yayımlanmış kovaryatlar (yayın tarihleri ASSUMPTION olduğu için bu liste de koşullu):

| Hedef | Origin (Tanım A) | Kovaryat | Kaynak yayın tarihi | Durum |
|---|---|---|---|---|
| PISA 2015 | Ara 2013 | TALIS 2013 | Haz 2014 | origin'den sonra |
| TIMSS G8/G4 2015 | Ara 2012 | TALIS 2013 | Haz 2014 | origin'den sonra |
| ICCS 2016 | Kas 2010 | PIRLS 2011 | Ara 2012 | origin'den sonra |
| ICCS 2022 | Kas 2017 | TALIS 2018 | Haz 2019 | origin'den sonra |

Tanım B'ye göre tüm kovaryatlar hedef sonuçlar yayımlanmadan önce kamuya açıktı.

**RECOMMENDATION.** Tezde Tanım A kullanılmalı ("bir önceki döngüden sonra bir sonraki döngüyü tahmin et"); bu, RQ1'in "forecasting" çerçevesine uyuyor. Tanım A ile tablodaki dört kombinasyon Faz 2 düzeneğinde maskelenmeli. Aynı kontrol programlar arası **skor** özelliklerine de uygulanmalı (örneğin TIMSS 2019 hedefi için PISA 2015 skorları, TIMSS 2015 yayınından birkaç gün sonra yayımlandı). Bu karar kullanıcıya aittir; aşağıdaki açık sorulara bakınız.

## Faz 0'da ortaya çıkan yeni bulgular

**A. Korunan bir veri dosyasının üzerine yazılmış (VERIFIED).** `outputs/stage4/country_estimates.csv`, `ad7b100` commit'inde 3 601 satırdan (7 program, tüm döngüler) 237 satıra (yalnızca PISA 2018) indi. Mevcut `unified_panel.csv` dosyası hâlâ eski tam veriyle tutarlı (PISA 2025 matematikte fark 0). Ancak:
- `build_unified_panel.py` şimdi çalıştırılırsa panel 237 satırlık dosyadan yeniden üretilir ve veri kaybolur.
- `run_loco_forecasting.py`, `se_true` değerini bu dosyadan alıyor; PISA 2018 dışındaki tüm satırlarda `se_true` boş kalır.
- Tam sürüm git geçmişinde duruyor: `git show ad7b100^:outputs/stage4/country_estimates.csv` (bu oturumda yalnızca scratchpad'e salt-okunur kopyası alındı).

**B. TALIS 2018 ülke tahminleri bozuk (VERIFIED).** `talis_covariate_estimates.csv` içinde TALIS 2018 ölçeklerinden yedisinin (`T3SECLS`, `T3SEINS`, `T3SEENG`, `T3DISC`, `T3JSENV`, `T3JSPRO`, `T3COLES`) ülkeler arası standart sapması 0.001; örneğin `T3SECLS` aralığı 12.636–12.641. Aynı yapıların TALIS 2013 sürümlerinde SD 0.53–0.96. Yalnızca `T3EFFPD` normal varyansa sahip (SD 0.965). Gerçek ülke farkları bu kadar küçük değildir; hesaplamada bir hata olması muhtemel (nedeni **ASSUMPTION**: yanlış değişken, standartlaştırılmış uluslararası değer veya ağırlık hatası). Ridge'de `StandardScaler` bu neredeyse sabit sütunları birim varyansa şişirir; yani PISA 2022/2025 ve ICCS 2022 fold'larında model gürültüye katsayı veriyor. Bu durum, kovaryatların neden zarar verdiğini ve neden küçültmenin (shrinkage) yardımcı olduğunu kısmen açıklayabilir (**ASSUMPTION**, Faz 2'de test edilecek).

**C. PISA 2025'teki TALIS kovaryatları eski değerlerin kopyası (VERIFIED).** PISA 2025 satırlarındaki dört TALIS değişkeni, PISA 2022 satırlarındaki değerlerle 27/27 ülkede birebir aynı. `talis_covariate_estimates.csv` TALIS 2024 için yalnızca üç değişken içeriyor (`T4SECLS`, `T4SEINS`, `T4SEENG`; gerçek varyansla, aralık 10.8–14.5); bunlar panele hiç girmemiş.

**D. `covariate_estimates.csv` dosyasında tekrarlanan anahtarlar var (VERIFIED).** (program, döngü, değişken, ülke) anahtarında 212 satır tekrarlı: TIMSS `BSDGEDUP` 49, TIMSS `ITSEX` 49, TIMSS G4 `ASDHEDUP` 62, PIRLS `ASDHEDUP` 26, `ASDHELA` 26. Pivotlama `aggfunc="first"` ile bunlardan birini sessizce seçiyor.

**E. TimesFM PISA girdileri git'te değil (VERIFIED).** `outputs/pisa_timesfm25_ils_adapter_v1/` (merge scriptinin okuduğu `equal_fold_summary.csv` dahil) izlenmiyor. Manifestte sha256 değerleri kayıtlı; RECOMMENDATION: tez için arşivlenmeli.

## 0.3 Manifest

[00_input_manifest.json](00_input_manifest.json) şunları içerir: git HEAD, dal adı, ve her girdi için yol, boyut, mtime, sha256, git takip durumu ve HEAD'e göre değişiklik. 22 dosyanın tamamı mevcut; izlenen dosyaların hiçbiri HEAD'e göre değiştirilmemiş. Dosya yeniden yazmayı reddeden bir kontrolle oluşturuldu.

## Çalıştırılan kontroller

- PISA 2025 PUF'tan ağırlıklı PV ortalamaları ile saklanan tahminlerin karşılaştırılması (`/usr/local/bin/python3`, pyreadstat).
- PISA 2018 (`CY07_MSU_STU_QQQ.sav`), 2022 (`CY08MSP_STU_QQQ.SAV`) ve 2025 mikroverisinden ESCS, HOMEPOS ve BELONG ülke ortalamaları; paneldeki `lag_*` değerleriyle karşılaştırma.
- `covariate_estimates.csv` ve `talis_covariate_estimates.csv` üzerinden kaynak döngü eşleştirmesi ve dağılım istatistikleri.
- `git show ad7b100^` ile `country_estimates.csv` dosyasının önceki sürümünün okunması.

## Onay bekleyen açık sorular

1. **Tahmin başlangıç noktası (origin) tanımı:** A (katı, önerilen) mı, B mi? Faz 2'deki maskeleme buna bağlı.
2. **`country_estimates.csv`:** `ad7b100^` sürümü yeni bir dosyaya mı geri yüklensin (örneğin `outputs/stage4/country_estimates_full_v2.csv`, mevcut dosyaya dokunmadan), yoksa üzerine yazma bilinçli bir karar mıydı?
3. **TALIS 2018 tahminleri:** Faz 1'de `talis_covariate_estimates` yeniden hesaplansın mı, yoksa bozuk TALIS sütunları Faz 2'de tamamen dışarıda mı bırakılsın?
4. **PISA 2025:** "Outcome-exposed" olduğu için ana sonuçlar 2025 hariç, duyarlılık analizi 2025 dahil olarak mı raporlansın?
