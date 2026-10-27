# Faz 4 — Çıkarım: güven aralıkları, TOST, Holm

**Tarih:** 27 Ekim 2026
**Git:** `feat/finetuning-v2-pipeline` @ `ad7b100` (commit yok; tüm çıktılar yeni dosya)
**Script:** `scripts/run_inference_v1.py`
**Çıktı:** `outputs/inference_v1/` (`contrasts_all.csv`, `rq2_preregistered_decision.csv`, `scale_context.csv`, `manifest.json`)
Etiketler: **VERIFIED**, **ASSUMPTION**, **RECOMMENDATION**.

Promptta çıktı yeri olarak `outputs/rq2_v3/` geçiyordu. O klasör Faz 3'te oluşturulduğu ve script'ler üzerine yazmayı reddettiği için ayrı bir klasör (`inference_v1`) kullanıldı.

## Yöntem

- **Fark:** Her test satırı için eşleştirilmiş kayıp farkı d_i = |e_ref| − |e_model|; Δ = ortalama(d). Pozitif değer, modelin referanstan daha iyi olduğu anlamına gelir.
- **Bootstrap (B = 5 000, ağırlıklı satır yeniden örneklemesi):**
  - **country (birincil):** Program-alan içinde ülkeler yeniden örnekleniyor; bir ülkenin tüm fold'lardaki satırları birlikte alınıyor.
  - **cycle (duyarlılık):** Test döngüleri yeniden örnekleniyor. Yalnızca 3–6 küme olduğu için kararsız; güven aralıkları bazen dejenere (genişlik ≈ 0). Bu sonuçlar yalnızca yön kontrolü olarak okunmalı.
  - **twoway (duyarlılık):** Ülkeler ve döngüler bağımsız örnekleniyor; satır ağırlığı iki çarpanın çarpımı (pigeonhole bootstrap, Owen 2007).
- **Testler (country şeması):**
  - Üstünlük: tek yönlü, H1: Δ > 0; sıfıra kaydırılmış bootstrap p.
  - Fark: çift yönlü p (RQ1).
  - TOST: ±1 puan marj. 90% güven aralığı (−1, 1) içindeyse eşdeğer; bootstrap TOST p = iki tek yönlü p'nin büyüğü.
  - Holm düzeltmesi: her karşılaştırma için 8 program-alan üzerinden. TOST p'nin alt sınırı 8/(B+1) ≈ 0.002.
- **Seed'ler:** Sabit; her akış SHA-256 ile türetiliyor. **VERIFIED:** İkinci çalıştırmada 3 CSV'nin tamamı byte düzeyinde aynı.
- **Ana analiz:** 2025 hariç. **Duyarlılık:** 2025 dahil.

## Eşdeğerlik marjı (±1 puan) için ölçek bağlamı (VERIFIED)

| Program-alan | Ülke ortalaması medyan SE | Persistence için örneklem gürültüsü tabanı (beklenen MAE) | Gerçekleşen persistence MAE |
|---|---|---|---|
| PISA mat / oku / fen | 2.51 / 2.69 / 2.53 | 2.99 / 3.15 / 2.95 | 9.63 / 11.31 / 9.02 |
| TIMSS G4 mat / fen | 2.73 / 2.89 | 3.27 / 3.60 | 17.65 / 18.34 |
| TIMSS G8 mat / fen | 3.29 / 3.17 | 3.91 / 3.89 | 16.97 / 17.93 |
| PIRLS oku | 2.51 | 3.01 | 14.04 |

- Örneklem gürültüsü tabanı: E|e| = sqrt(2/π) · sqrt(SE_t² + SE_(t−1)²). Gerçek skorlar hiç değişmeseydi persistence'ın yalnızca örneklem hatasından beklenen MAE'si budur. Link error bu tabana dahil değil.
- SE'si bilinen satır oranı: PIRLS 72/100, TIMSS 128–132/168–182, PISA 310–348/320–367.
- **Resmi link error değerleri (VERIFIED, PISA 2025 Teknik Rapor, Annex Table 15.A.20, 14 Eylül 2026):** PISA 2022→2025 için fen 3.12, matematik 1.22, okuma 1.09. Daha eski karşılaştırmalarda değerler daha yüksek (örneğin 2015→2025: fen 3.41, matematik 3.00, okuma 3.79).
- Ardışık eski döngüler arasındaki link error değerleri (örneğin 2018→2022) yerel dosyalarda bulunamadı; PISA 2022 Teknik Raporu'ndan eklenmeli (**ASSUMPTION**: benzer büyüklükte).

**Marjın gerekçesi:** 1 puan, tek bir ülke ortalamasının medyan SE'sinin yaklaşık %40'ı ve PISA 2022→2025 matematik ve okuma link error değerlerinin (1.1–1.2) altında. Bu büyüklükteki MAE farkları, ülke ortalamalarının ölçüm hassasiyetinin altında kalıyor. Persistence MAE'si örneklem tabanının 3–5 katı olduğu için tahmin hatasının büyük kısmı gerçek değişimden ve link error'dan geliyor; modellerin ayırt edebileceği alan dar.

## RQ2 — önceden kayıtlı karar (VERIFIED, ana analiz, country şeması)

| Program-alan | Δ(M0A − M1) [%95 GA] | K1 | K2 | K3 | KB iyileştiriyor mu | M1 ≈ M0A (TOST, Holm) | M1 ≈ M0 (TOST, Holm) |
|---|---|---|---|---|---|---|---|
| PISA mat | −1.70 [−2.25, −1.16] | Hayır | Hayır | Evet | **Hayır** | Hayır (M1 daha kötü) | Evet |
| PISA oku | −1.59 [−2.36, −0.88] | Hayır | Hayır | Evet | **Hayır** | Hayır (M1 daha kötü) | Evet |
| PISA fen | −1.65 [−2.34, −0.93] | Hayır | Hayır | Evet | **Hayır** | Hayır (M1 daha kötü) | Evet |
| G4 mat | +0.07 [−0.38, +0.53] | Hayır | Hayır | Hayır | **Hayır** | Evet | Evet |
| G4 fen | +0.13 [−0.23, +0.48] | Hayır | Hayır | Evet | **Hayır** | Evet | Evet |
| G8 mat | −0.05 [−0.65, +0.55] | Hayır | Hayır | Hayır | **Hayır** | Evet | Evet |
| G8 fen | +0.15 [−0.26, +0.54] | Hayır | Hayır | Hayır | **Hayır** | Evet | Evet |
| PIRLS oku | −0.39 [−0.81, +0.00] | Hayır | Hayır | Evet | **Hayır** | Evet | Evet |

**Önceden kayıtlı kurala göre sonuç: 0/8.** Holm düzeltmeli tek yönlü p (M1 > M0A) tüm program-alanlarda 1.0.

**Önceden kayıtlı ifade:** "KB entegrasyonunun tahmine katkısı olduğuna dair kanıt yok." TOST sonuçlarına göre şu ek yapılır:
- M1, M0 ile **8/8** program-alanda ±1 puan içinde pratik olarak eşdeğer.
- M1, M0A ile TIMSS ve PIRLS'te **5/5** program-alanda eşdeğer.
- PISA'da M1, kovaryatsız M0A'dan anlamlı olarak **daha kötü** (1.6–1.7 puan, güven aralıkları sıfırı dışlıyor).

**M1 − M0 farkının boyutu (VERIFIED):** Country şemasında PISA matematik (+0.05 [0.02, 0.08]) ve fen (+0.09 [0.04, 0.12]) farkları sıfırdan ayrışıyor. Ancak:
- Cycle ve twoway şemalarında güven aralıkları sıfırı içeriyor (twoway: matematik [−0.01, 0.17], fen [−0.01, 0.18]).
- Büyüklükleri marjın %5–9'u kadar.
- Faz 3'e göre bu kazanç, KB içeriğinden değil genel kovaryat küçültmesinden geliyor; tekdüze w=0.25 daha fazla kazanç sağlıyor.

**Duyarlılık (2025 dahil, PISA):**
- M1, M0'dan daha kötü:
  - Matematik: −0.79 [−1.06, −0.52]; TOST Holm p = 0.059, yani eşdeğerlik gösterilemiyor.
  - Okuma: −0.49 [−0.73, −0.26]; eşdeğer.
  - Fen: −0.28 [−0.59, +0.02]; eşdeğer.
- M1, M0A'dan 1.7–2.2 puan daha kötü.
- Karar değişmiyor.

**Bootstrap şemalarına duyarlılık:** M1'in M0A'dan kötü olduğu PISA sonucu country şemasında sağlam. Cycle ve twoway şemalarında (yalnızca 5–6 döngü kümesi) güven aralıkları genişliyor ve okuma ile fende sıfırı içeriyor. Kural hiçbir şemada sağlanmıyor.

## RQ1 — her model persistence'a karşı (VERIFIED, ana analiz, country şeması)

Hücrelerde Δ = MAE(Persistence) − MAE(model) ve %95 güven aralığı var; pozitif değer, modelin persistence'tan daha iyi olduğu anlamına gelir. `*` = Holm düzeltmeli çift yönlü p < .05; `≈` = TOST ile ±1 puan içinde eşdeğer (Holm p < .05).

| Model | PIRLS oku | PISA mat | PISA oku | PISA fen | G4 mat | G4 fen | G8 mat | G8 fen |
|---|---|---|---|---|---|---|---|---|
| AR1 | −1.12 [−2.34, 0.12] | −0.17 [−0.72, 0.41] ≈ | −0.65 [−1.23, −0.10] | −0.20 [−0.50, 0.10] ≈ | −0.72 [−3.28, 1.87] | −1.15 [−3.36, 1.20] | +0.84 [−0.71, 2.45] | +0.15 [−1.65, 2.03] |
| Drift | −0.48 [−1.77, 0.85] | −0.60 [−0.82, −0.37] *≈ | −0.02 [−0.41, 0.37] ≈ | +0.07 [−0.20, 0.35] ≈ | −0.23 [−2.00, 1.55] | −2.03 [−3.40, −0.56] * | −0.07 [−1.10, 0.92] | −0.42 [−1.31, 0.48] |
| LinearTrend | −3.73 * | −2.18 * | −3.74 * | −2.71 * | −5.12 * | −4.61 * | −6.27 * | −6.50 * |
| InternationalMean | −25.3 * | −36.1 * | −30.7 * | −34.1 * | −31.7 * | −31.2 * | −33.5 * | −27.5 * |
| Ridge_M0A | −1.56 [−3.02, −0.01] | −0.64 [−1.37, 0.07] | −1.11 [−1.88, −0.34] * | −0.87 [−1.57, −0.18] | +1.08 [−1.68, 3.99] | +0.76 [−2.51, 4.09] | −0.33 [−2.26, 1.55] | −0.12 [−2.13, 1.91] |
| Ridge_M0 | −1.95 [−3.51, −0.29] | −2.39 * | −2.70 * | −2.60 * | +1.15 [−1.55, 3.91] | +0.88 [−2.44, 4.24] | −0.37 [−2.51, 1.66] | +0.02 [−2.03, 2.12] |
| Ridge_M1 | −1.95 [−3.51, −0.29] | −2.35 * | −2.70 * | −2.51 * | +1.15 [−1.55, 3.91] | +0.88 [−2.43, 4.25] | −0.38 [−2.49, 1.66] | +0.02 [−2.03, 2.12] |
| RF_score | −3.99 [−7.17, −1.21] | −1.86 * | −1.47 * | −1.71 * | −3.91 [−9.00, 0.55] | −5.64 [−11.06, −1.26] | −3.97 [−7.35, −0.72] | −3.50 [−6.88, −0.79] |
| RF_cov | −3.98 [−7.15, −1.21] | −1.92 * | −1.43 * | −1.84 * | −3.86 [−8.93, 0.59] | −5.60 [−11.07, −1.23] | −4.04 [−7.43, −0.82] | −3.55 [−6.95, −0.80] |
| XGB_score | −2.50 [−5.42, 0.26] | −1.67 * | −1.49 [−2.60, −0.44] | −1.29 * | −3.25 [−7.90, 0.96] | −5.36 [−10.53, −1.05] | −3.72 [−6.67, −0.87] | −2.58 [−5.87, 0.15] |
| XGB_cov | −2.55 [−5.47, 0.22] | −1.85 * | −1.48 [−2.54, −0.44] | −1.61 * | −3.29 [−7.96, 0.92] | −5.22 [−10.43, −0.86] | −3.81 [−6.81, −0.93] | −3.32 [−6.70, −0.56] |
| TimesFM 2.5 (alt küme) | +0.57 [−0.55, 1.87] | −0.54 [−1.21, 0.10] | −0.22 [−0.80, 0.38] ≈ | −0.40 [−0.94, 0.14] | −2.28 [−3.55, −1.14] * | −1.91 [−3.47, −0.47] | −1.18 [−2.30, −0.16] | −0.47 [−1.80, 0.80] |

**RQ1 bulguları (VERIFIED):**
1. **Hiçbir model persistence'ı anlamlı olarak geçemiyor.** 96 model × program-alan karşılaştırmasının hiçbirinde Holm düzeltmeli anlamlı üstünlük yok. Bu sonuç üç bootstrap şemasında da aynı.
2. **RF ve XGBoost persistence'tan kötü.** Hiçbir program-alanda nokta tahmin bile persistence'tan iyi değil. Holm sonrası anlamlı olarak kötü oldukları yerler: RF, PISA'nın üç alanında (1.4–1.9 puan); XGBoost, PISA matematik ve fende (1.3–1.9 puan). XGBoost PISA okumada da yaklaşık 1.5 puan kötü, ama Holm sonrası anlamlı değil. TIMSS ve PIRLS'te nokta tahminleri 2.5–5.6 puan kötü, ama güven aralıkları geniş olduğu için Holm sonrası anlamlı değil.
3. **Basit zaman serisi modelleri persistence ile eşdeğer.** AR(1) PISA matematik ve fende, drift PISA'nın üç alanında, TimesFM ise PISA okumada ±1 puan içinde eşdeğer. Doğrusal trend ve uluslararası ortalama 8/8 program-alanda anlamlı olarak kötü.
4. **Ridge:** PISA'da kovaryatlı Ridge (M0, M1) anlamlı olarak kötü. Skor-tek Ridge (M0A) yalnızca PISA okumada anlamlı kötü. TIMSS G4'te Ridge'in +0.8–1.2 puanlık avantajı anlamlı değil.
5. **TimesFM** (ortak alt kümede): PIRLS'te +0.57 ama anlamlı değil. TIMSS G4 matematikte anlamlı olarak kötü (−2.28).
6. **Güç sınırı:** TIMSS ve PIRLS'te güven aralıkları ±2–5 puan genişliğinde (3–6 fold, fold başına 15–50 ülke). Bu programlarda 2 puanın altındaki farklar ne gösterilebiliyor ne de dışlanabiliyor.

**Duyarlılık şemaları:**
- Cycle şeması 27/96, twoway şeması 16/96 karşılaştırmada "anlamlı olarak kötü" diyor. Anlamlı üstünlük her ikisinde de yok.
- Cycle şeması 3–6 kümeyle güvenilir değil; anlamlılık sayısındaki fark buradan kaynaklanıyor.

## Tezde kullanılabilecek özet ifadeler (RECOMMENDATION)

**RQ1:**
> *"In a common rolling-origin backtest (8 program-domains, 3–6 origins each), no model significantly outperformed persistence after Holm correction. Random forest and XGBoost were never better than persistence; random forest was significantly worse in all three PISA domains and XGBoost in PISA mathematics and science (by 1.3–1.9 points). Persistence, drift and AR(1) were practically equivalent (±1 point) where precision allowed; differences below ~2 points could not be resolved for TIMSS and PIRLS."*

**RQ2:**
> *"Under the pre-registered rule, the literature-derived knowledge base did not improve forecasts in any of the 8 program-domains. Literature-weighted Ridge was practically equivalent to unweighted Ridge in all 8 (TOST, ±1 point) and significantly worse than the score-only model for PISA (by 1.6–1.7 points). Random reassignments of the knowledge-base weights performed as well or better, indicating that the content of the knowledge base did not carry forecast-relevant information at the country level."*

## Sınırlılıklar

- 3–6 origin, dolayısıyla döngü düzeyinde çıkarım zayıf. Ülke kümeli bootstrap, ortak döngü şoklarını (örneğin COVID dönemi 2022 düşüşü) hafife alabilir; twoway şeması bunun duyarlılık kontrolü.
- TOST eşdeğerliği "MAE farkı ±1 puan içinde" anlamına geliyor; tek tek ülke tahminlerinin eşdeğer olduğu anlamına gelmiyor.
- Link error yalnızca PISA 2025 karşılaştırmaları için yerelde mevcut; diğer program ve döngüler için Limitations bölümünde kaynak gösterilmeli.

## Sonraki adım

Faz 5 (raporlama ve tez metni):
- RQ1 tablolarını program-alan başına, sıralama belirsizliğiyle birlikte üretmek.
- Tezdeki her değişecek iddia için eski cümle / yeni cümle / kanıt dosyası farkını önermek.

Tez dosyalarında onayınız olmadan değişiklik yapılmayacak.
