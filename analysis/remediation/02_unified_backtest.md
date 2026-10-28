# Faz 2 — Tek değerlendirme düzeneği

**Tarih:** 27 Ekim 2026
**Git:** `feat/finetuning-v2-pipeline` @ `ad7b100` (commit yok; tüm çıktılar yeni dosya)
**Script:** `scripts/run_unified_backtest.py`
**Çıktı:** `outputs/unified_backtest_v1/` (+ `manifest.json`: girdi sha256'ları, sürümler, tasarım parametreleri)
Etiketler: **VERIFIED**, **ASSUMPTION**, **RECOMMENDATION**.

## Tasarım

| Öğe | Karar |
|---|---|
| Doğrulama şeması | Genişleyen pencere (rolling-origin) backtest; "LOCO" adı kullanılmıyor |
| Origin | Tanım A: aynı program-alanın bir önceki döngüsünün yayın tarihi. Yayın tarihleri koddaki `RELEASE` sözlüğünde (**ASSUMPTION**, resmi kaynaktan teyit edilmeli) |
| Eğitim | Hedef döngüsü < t olan tüm (ülke, döngü) satırları; her satırın özellikleri kendi origin'ine göre hesaplanıyor |
| Test kümesi | (ülke, t): hedefi ve en az bir önceki kendi skoru olanlar. Önceki skoru olmayan 580 satır `no_history.csv` dosyasında; ana metriklere girmiyor |
| Tam fold | Her model her test satırını tahmin ettiyse. Eğitim satırı < 10 olan fold eksik sayılıyor; özetlere girmiyor, `folds.csv` içinde raporlanıyor |
| Skor özellikleri | `lag1`, `lag2f` (yoksa lag1), `delta`, `has_lag2`, `gap` (yıl) |
| Kovaryatlar | Yalnızca kendi programınınki, ve origin'de yayımlanmış en son kaynak döngü. PISA için TALIS (TALIS-PISA bağlantılı tasarım). Başka programlar arası özellik yok; `groupby().first()` birleştirmesi kullanılmıyor |
| KB | `kb_weights.KBWeights` (v3, katı arama). TALIS 2018 SECLSS, TCDISCS ve TJSPROS kaynakları maskelendi (29/30/30 satır) |
| Ridge alpha | İç zaman sıralı doğrulama: son eğitim döngüsü doğrulama kümesi. Yetersizse alpha=10 (`fallback_fixed`). Tüm seçimler `ridge_alpha_log.csv` içinde |
| RF/XGB | Önceki tabular benchmark ile aynı sabit hiperparametreler, ayarlama yok, `n_jobs=1`, sabit seed |
| TimesFM | Kaydedilmiş zero-shot tahminler. Yalnızca mevcut olduğu (ülke, döngü) alt kümesinde, tüm modeller aynı alt kümede yeniden puanlanarak karşılaştırılıyor |
| 2025 | Ana sonuçlar 2025 hariç, duyarlılık analizi 2025 dahil |

### Veri işlemleri (VERIFIED, `manifest.json` → `data_log`)

- Skorlar `country_estimates_full_v2.csv` dosyasından okunuyor.
- PISA 2000'deki 43 sayısal ülke kodu, mevcut crosswalk (`analysis/03_data_feasibility/audit/pisa_2000_reading_country_crosswalk.csv`) ile ISO3'e çevrildi.
- 8 benchmark/alt-ulusal kod çıkarıldı: `PIRLS:2001:SE3`, `PIRLS:2006:IS5`, `PIRLS:2006:NO5`, `PIRLS:2011:MA6`, `PIRLS:2016:ZA5`, `TIMSS_G4:2011:YE6`, `TIMSS_G4:2015:NO4`, `TIMSS_G8:2015:NO8`.
- PIAAC'ın `2017` etiketli 84 satırı 2022 olarak düzeltildi (Cycle 2; `fix_piaac_cycle_year.py` ile aynı mantık).
- `covariate_estimates.csv` içindeki 118 tekrarlı anahtarda en büyük `n_total` değerli satır seçildi. Bu, ana örneklemi bridge/ePIRLS gibi alt örneklemlere tercih ediyor (**ASSUMPTION**).
- TimesFM `true_score` değerleri bizim hedeflerimizle 0.01 toleransla eşleşiyor (uyuşmazlık 0).

## Başarı kriterleri (VERIFIED)

| Kriter | Sonuç |
|---|---|
| Aynı girdiyle yeniden çalıştırma byte düzeyinde aynı mı | Evet: 9 CSV'nin tamamı `cmp` ile birebir aynı (ikinci çalıştırma scratchpad'e yazıldı) |
| Program-alan başına tek persistence değeri | Evet (özet tablolarında 1 satır) |
| Bir gruptaki her modelin n'i aynı mı | Evet: her özet tablosunda program-alan başına benzersiz n sayısı 1. Script, tam fold'larda bunu `AssertionError` ile zorluyor |
| Sonlu tahmin | Evet (assertion) |

**Teknik not (VERIFIED).** macOS Accelerate ile numpy 2.1.3, iyi koşullu matris çarpımlarında bile sahte `matmul` RuntimeWarning veriyor. Ridge ve OLS çıktıları kapalı form çözümle karşılaştırıldı (max fark < 1e-12). Uyarı scriptte gerekçesiyle susturuldu.

## Fold yapısı (VERIFIED, `folds.csv`)

- Ana program-alanların tam fold sayısı (2025 hariç): PISA matematik 5, okuma 6, fen 5; TIMSS G8 6+6; TIMSS G4 5+5; PIRLS 3. Her serinin ilk fold'u eğitim verisi olmadığı için eksik.
- Ek programlar:
  - ICCS: yalnızca 2022 fold'u tam (n_train=10).
  - ICILS: 2018 ve 2023 fold'ları eksik (n_train 0 ve 3).
  - PIAAC: tek fold, eğitim çifti yok; yalnızca temel modeller raporlanabiliyor. Bu nedenle aynı dalgadan özellik sızıntısı (F1) yapısal olarak ortadan kalktı.
- Kovaryat kapsamı fold'a göre değişiyor:
  - PISA: 2009–2018 fold'larında ESCS, HOMEPOS ve BELONG; 2022 ve 2025'te ayrıca TALIS (Tanım A ile TALIS 2013 değerleri; TALIS 2018'den yalnızca TEFFPROS).
  - TIMSS ve PIRLS: kovaryat ancak 2011/2019/2021'den itibaren eğitim sinyali taşıyor.
  - ICT değişkenlerinin eğitimde hiç sinyali yok.
- Alpha seçimleri: 135 Ridge uyumunun 59'unda iç doğrulama ızgaranın alt sınırını (0.01, neredeyse OLS) seçti, 49'unda iç değer, 27'sinde yedek alpha=10 kullanıldı. Üst sınır (1000) hiç seçilmedi. Alpha küçük olduğunda ridge ölçek değişikliklerinden etkilenmez; bu yüzden birçok fold'da M1 ≈ M0 (Faz 3 için önemli).

## Sonuçlar — ana tablo (2025 hariç, tam fold'lar, MAE puan)

| Model | PIRLS oku | PISA mat | PISA oku | PISA fen | G4 mat | G4 fen | G8 mat | G8 fen |
|---|---|---|---|---|---|---|---|---|
| Persistence | **14.04** | **9.63** | **11.31** | 9.02 | 17.65 | 18.34 | 16.97 | 17.93 |
| Drift | 14.52 | 10.23 | 11.33 | **8.95** | 17.88 | 20.37 | 17.04 | 18.35 |
| AR1 | 15.16 | 9.80 | 11.96 | 9.22 | 18.37 | 19.49 | **16.13** | **17.78** |
| LinearTrend | 17.77 | 11.81 | 15.05 | 11.73 | 22.77 | 22.94 | 23.24 | 24.43 |
| InternationalMean | 39.29 | 45.70 | 42.04 | 43.11 | 49.37 | 49.56 | 50.42 | 45.45 |
| Ridge_M0A | 15.60 | 10.27 | 12.41 | 9.88 | 16.57 | 17.58 | 17.30 | 18.05 |
| Ridge_M0 | 15.99 | 12.02 | 14.01 | 11.62 | **16.50** | 17.45 | 17.35 | 17.91 |
| Ridge_M1 | 15.99 | 11.97 | 14.00 | 11.53 | **16.50** | **17.45** | 17.35 | 17.91 |
| RF_score | 18.03 | 11.48 | 12.78 | 10.72 | 21.56 | 23.98 | 20.95 | 21.43 |
| RF_cov | 18.03 | 11.54 | 12.74 | 10.85 | 21.50 | 23.94 | 21.01 | 21.48 |
| XGB_score | 16.54 | 11.30 | 12.80 | 10.30 | 20.90 | 23.69 | 20.69 | 20.51 |
| XGB_cov | 16.59 | 11.48 | 12.79 | 10.63 | 20.94 | 23.56 | 20.79 | 21.24 |
| n / fold | 100 / 3 | 320 / 5 | 367 / 6 | 320 / 5 | 169 / 5 | 168 / 5 | 182 / 6 | 182 / 6 |

Göreli MAE (persistence = 1) ve sıralamalar `summary_primary_excl2025.csv` içinde.

**Betimsel gözlemler (VERIFIED; belirsizlik ve test Faz 4'te):**
1. 8 program-alanın 6'sında en iyi model persistence, drift veya AR(1); yani basit zaman serisi modelleri.
2. TIMSS G4'te Ridge modelleri persistence'tan yaklaşık %5–6.5 daha iyi (0.935 ve 0.952). TIMSS G8 fende Ridge_M0 ve M1, persistence ile neredeyse eşit (0.999).
3. RF ve XGBoost hiçbir program-alanda persistence'ı geçmiyor (göreli MAE 1.10–1.31). Kovaryat eklemek ağaç modellerini iyileştirmiyor.
4. PISA'da kovaryat eklemek Ridge'i kötüleştiriyor: M0A'ya göre M0 1.60–1.75 puan daha kötü. KB ölçeklemesi (M1) bu kaybı yalnızca 0.01–0.09 puan geri alıyor. TIMSS ve PIRLS'te M1 ile M0 aynı (tek kovaryat ve düşük alpha).
5. Eski RQ1 tablosundaki "Ridge persistence'tan çok kötü" bulgusunun bir kısmı, geçmişi olmayan ülkelerin ortalamayla doldurulmasından geliyordu. Ortak test kümesinde Ridge_M0A, persistence'a göre PISA'da %7–10, PIRLS'te %11 daha kötü. Eski tabloda bu oranlar PISA'da %19–36, PIRLS'te %80'di (Ridge_M0A / Persistence_M2).

## Duyarlılık: 2025 dahil

PISA'da 2025 fold'u eklenince n 401/448/402 oluyor; TIMSS ve PIRLS değişmiyor.

| Model | PISA mat | PISA oku | PISA fen |
|---|---|---|---|
| Persistence | 10.06 | 12.01 | 9.58 |
| Drift | 10.22 | **11.50** | **9.53** |
| Ridge_M0A | 11.20 | 13.13 | 10.42 |
| Ridge_M0 | 12.66 | 14.53 | 11.82 |
| Ridge_M1 | 13.45 | 15.02 | 12.10 |

**VERIFIED.** 2025 dahil edildiğinde M1, PISA'nın üç alanında da M0'dan daha kötü (0.28–0.79 puan). Eski `rq2_m0_vs_m1.csv` tablosundaki PISA "iyileşmesi" (en büyüğü okuma, 0.52) bu düzenekte tekrar etmiyor. Bu bir test sonucu değil, betimsel bir gözlemdir; çıkarım Faz 4'te yapılacak.

## TimesFM alt kümesi (2025 hariç)

Kaydedilmiş TimesFM tahminleri PISA 2012–2022, TIMSS, PIRLS ve ICCS için var. Alt küme n: PISA 208×3, G4 115/114, G8 132/132, PIRLS 61. Göreli MAE (persistence = 1):

| | PIRLS | PISA mat | PISA oku | PISA fen | G4 mat | G4 fen | G8 mat | G8 fen |
|---|---|---|---|---|---|---|---|---|
| TimesFM 2.5 zero-shot | **0.952** | 1.057 | 1.021 | 1.047 | 1.167 | 1.132 | 1.080 | 1.031 |
| Ridge_M0A | 1.228 | 1.077 | 1.099 | 1.128 | 0.917 | 1.030 | 1.066 | 1.093 |

**VERIFIED.** Ortak alt kümede TimesFM yalnızca PIRLS'te persistence'ı geçiyor (n=61). Eski RQ1 tablosunda TimesFM'in PISA'da birinci görünmesi farklı fold ve ülke kümelerinden kaynaklanıyordu.

## Ek tablolar

- `summary_appendix_complete_folds.csv`: ICCS 2022 (n=15). Persistence 12.75, diğer tüm modeller 19–34.
- `summary_appendix_baselines_incomplete_folds.csv`: ICCS 2016, ICILS 2018/2023, PIAAC 2022. Yalnızca temel modeller var.
  - PIAAC literacy persistence 10.51, numeracy 10.14 (n=15).
  - ICILS'te uluslararası ortalama persistence'tan iyi (0.85). Bu, ICILS 2023'teki ölçek veya katılım değişimine işaret edebilir (**ASSUMPTION**).
- Bu satırlar tek fold'dur; genelleme için kullanılmamalı.

## Önceki bulgularla ilişki

| Bulgu | Bu düzenekte durum |
|---|---|
| F1 PIAAC sızıntısı | Giderildi: aynı dalgadan özellik yok; eğitim çifti olmadığı için yalnızca temel modeller |
| F6 farklı test kümeleri | Giderildi: fold başına tek test kümesi, n eşitliği zorlanıyor |
| F6 geçmişi olmayan ülkeleri ortalamayla doldurma | Giderildi: 580 satır ayrı dosyada |
| F6 kovaryatların hedef sayılması | Giderildi: yalnızca sonuç alanları hedef |
| F9 tamamen NaN sütunlar ve programlar arası sızıntı | Giderildi: fold başına sinyalsiz sütun çıkarılıyor; programlar arası özellik yok |
| Faz 0: Tanım A ihlalleri | Giderildi: origin'den sonra yayımlanan kaynak maskeleniyor; fold başına `covariate_sources` JSON'unda `masked_after_origin` listesi var |

## Açık noktalar ve sonraki adım

- **ASSUMPTION.** `RELEASE` tarihleri resmi raporlardan teyit edilmeli; özellikle PIRLS 2021 (Mayıs 2023) ve PISA 2025 (Eylül 2026). Bir tarih değişirse yalnızca o fold'un kovaryat kapsamı değişir.
- **RECOMMENDATION.** Ridge alpha ızgarasının alt sınırı sık seçiliyor. Bu, ızgaranın yetersiz olduğu anlamına gelmiyor (0.01 zaten neredeyse OLS); ama M1'in M0'dan ayrışma imkânını sınırlıyor. Faz 3'te bunu raporlamak, M1'i yapay olarak avantajlı kılacak şekilde alpha'yı sabitlemekten daha dürüst.
- Faz 3 (RQ2 yeniden tasarımı): M1'e karşı M0A ve M0; tekdüze küçültme, 200 permütasyon ve veri güdümlü özellik başına ceza ile placebo testleri; KB'ye dayalı özellik seçimi; önceden kaydedilmiş karar kuralı. Hepsi bu düzeneğin aynı fold'ları üzerinde yapılacak.
