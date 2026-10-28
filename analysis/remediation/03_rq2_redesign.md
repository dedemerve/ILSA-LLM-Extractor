# Faz 3 — RQ2 yeniden tasarımı

**Tarih:** 27 Ekim 2026
**Git:** `feat/finetuning-v2-pipeline` @ `ad7b100` (commit yok; tüm çıktılar yeni dosya)
**Önceden kayıt:** [03_preregistration.md](03_preregistration.md); sha256 `1713cbb6276797b8dd750d1b53ac2666677f388ef6d83df11e0073eaafa4fdb8`, 2026-10-27T14:49:38Z, analizden önce kilitlendi. Script, dosya değişmişse çalışmayı reddediyor.
**Script:** `scripts/run_rq2_v3.py`
**Çıktı:** `outputs/rq2_v3/`
Etiketler: **VERIFIED**, **ASSUMPTION**, **RECOMMENDATION**.

## Yöntem özeti

- Fold'lar, test satırları, ön işleme ve alpha seçimi `run_unified_backtest.py` dosyasından import ediliyor.
- **VERIFIED.** Ridge_M0A, M0 ve M1 tahminleri düzenek çıktısıyla 1e-5 toleransla eşleşiyor; script bunu assertion ile zorluyor. 8 ana program-alan, 44 tam fold.
- 44 fold'un 13'ünde eğitimde sinyali olan kovaryat yok (erken TIMSS, PIRLS ve PISA okuma 2006). Bu fold'larda tüm ağırlıklandırma varyantları M0A ile aynı tahmini veriyor. Bu, KB'nin etki edebileceği alanı yapısal olarak daraltıyor.
- Varyantlar önceden kayıttaki tanımlarla birebir uygulandı:
  - Uniform_c
  - 200 permütasyon (12 pozitif KB ağırlığından iadesiz atama, program-alan içinde sabit)
  - DataDriven_block (c ve alpha birlikte iç zaman sıralı doğrulamayla)
  - DataDriven_adaptive
  - KB_top_k ve 200 rastgele sıralamalı Random_top_k
- **Tek uygulama ayrıntısı (ASSUMPTION değil, belgeleme):** DataDriven_adaptive'in ikinci aşama alpha seçimi, ağırlıkları tüm eğitim verisinden öğrendikten sonra yapılıyor. Bu, test verisine değil yalnızca eğitim içi doğrulamaya hafif bir iyimserlik katabilir.

## Sonuçlar — ana analiz (2025 hariç, MAE puan)

| Program-alan | n | M0A | M0 | M1 | Uni 0.75 | Uni 0.5 | Uni 0.25 | DD block | DD adaptive | KB top1 | KB top2 | KB top3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| PISA mat | 320 | **10.27** | 12.02 | 11.97 | 11.97 | 11.90 | 11.77 | 12.02 | 12.00 | 10.47 | 10.56 | 10.65 |
| PISA oku | 367 | 12.41 | 14.01 | 14.00 | 14.01 | 14.00 | 14.00 | 13.88 | 14.01 | 12.44 | 12.33 | **12.25** |
| PISA fen | 320 | **9.88** | 11.62 | 11.53 | 11.53 | 11.43 | 11.29 | 11.57 | 11.55 | 10.08 | 10.22 | 10.22 |
| G8 mat | 182 | 17.30 | 17.35 | 17.35 | 17.35 | 17.35 | 17.36 | **17.25** | 17.35 | — | — | — |
| G8 fen | 182 | 18.05 | **17.91** | **17.91** | 17.91 | 17.91 | 17.91 | 17.92 | 17.91 | — | — | — |
| G4 mat | 169 | 16.57 | **16.50** | **16.50** | 16.50 | 16.50 | 16.50 | 16.52 | 16.50 | — | — | — |
| G4 fen | 168 | 17.58 | **17.45** | **17.45** | 17.45 | 17.45 | 17.46 | 17.57 | 17.45 | — | — | — |
| PIRLS oku | 100 | **15.60** | 15.99 | 15.99 | 15.99 | 15.99 | 15.99 | **15.60** | 15.99 | 15.78 | — | — |

"—": programın kovaryat sayısı k'dan fazla değil; seçim tanımsız (önceden kayıttaki kural).

### Karar kuralı koşulları (VERIFIED, ana analiz)

| Program-alan | ΔMAE(M0A − M1) | ΔMAE(M0 − M1) | Permütasyonlarda M1'den iyi veya eşit oran | En iyi placebo − M1 | K1 (nokta, ≥ 1) | K2 (< perm %5) | K3 (> M0) |
|---|---|---|---|---|---|---|---|
| PISA mat | −1.70 | +0.05 | %99.0 | −0.20 (Uniform 0.25) | Hayır | Hayır | Evet |
| PISA oku | −1.59 | +0.01 | %42.0 | −0.13 (DD block) | Hayır | Hayır | Evet |
| PISA fen | −1.65 | +0.09 | %93.5 | −0.24 (Uniform 0.25) | Hayır | Hayır | Evet |
| G8 mat | −0.05 | −0.00 | %55.5 | −0.10 (DD block) | Hayır | Hayır | Hayır |
| G8 fen | +0.15 | −0.00 | %55.5 | −0.00 | Hayır | Hayır | Hayır |
| G4 mat | +0.07 | −0.00 | %57.0 | −0.00 | Hayır | Hayır | Hayır |
| G4 fen | +0.13 | +0.00 | %48.0 | −0.00 | Hayır | Hayır | Evet |
| PIRLS oku | −0.39 | +0.00 | %23.0 | −0.39 (DD block) | Hayır | Hayır | Evet |

**Önceden kayıtlı kurala göre karar (VERIFIED):** Üç koşulun da sağlanması gerekiyor. K1'in nokta kısmı (M0A'ya göre en az 1 puan kazanç) ve K2 (permütasyonların %95'inden iyi olmak) 8 program-alanın hiçbirinde sağlanmıyor. Faz 4'teki güven aralıkları K1'i yalnızca daha da zorlaştırabilir, kolaylaştıramaz. Bu nedenle kural **0/8** program-alanda sağlanıyor.

Önceden kayıtlı sonuç ifadesi: **"KB entegrasyonunun tahmine katkısı olduğuna dair kanıt yok."** "Pratik olarak eşdeğer" eki Faz 4'teki TOST sonucuna bağlı.

### Ayrıntılı gözlemler (VERIFIED)

1. **KB içeriği rastgeleden iyi değil.** PISA matematikte rastgele ağırlık atamalarının %99'u, fende %93.5'i KB ağırlıkları kadar veya daha iyi sonuç veriyor. Yani KB'nin sıralaması bu iki alanda rastgele bir sıralamadan **daha kötü**. Okumada %42.
2. **Genel küçültme KB'den iyi.** PISA'nın üç alanında da en az bir placebo M1'den iyi: tekdüze w=0.25 matematikte 0.20, fende 0.24 puan; DataDriven_block okumada 0.13 puan. M1'in M0'a karşı küçük kazancı (0.01–0.09) KB içeriğinden değil, kovaryatların genel olarak küçültülmesinden kaynaklanıyor.
3. **Asıl karşılaştırma M0A.** PISA'da kovaryat eklemek her ağırlıklandırma şemasında zararlı (M0A'ya göre +1.4–1.7 puan). PISA'da her kovaryatlı varyantın en iyisi bile M0A'yı geçemiyor.
4. **Veri, KB sıralamasıyla uyuşmuyor.** DataDriven_adaptive'in eğitimden öğrendiği ağırlıklarda HOMEPOS genellikle 1.0, ESCS çoğu fold'da düşük (örneğin PISA matematik 2022: 0.03). KB'de ise ESCS en yüksek (1.0). Standartlaştırılmış katsayılar çoklu doğrusal bağlantı altında kararsız olduğundan bu karşılaştırma betimseldir (`datadriven_adaptive_weights.csv`).
5. **DataDriven_block çoğunlukla c=0 seçiyor.** Kovaryatı olan fold'larda iç doğrulama sıkça kovaryatları tamamen sıfırlamayı seçiyor; PISA'nın 2025 fold'larının hepsinde c=0. Kovaryatı olmayan fold'larda c'nin değeri anlamsız (eşitlikte ızgaranın ilk değeri seçiliyor) (`datadriven_block_log.csv`).
6. **KB'ye dayalı özellik seçimi (ikincil yol) de başarısız.** KB_top_k, rastgele top-k sıralamalarının %17.5–72'sinden daha iyi değil. Hiçbir durumda M0A'dan 1 puan iyi değil (en iyisi PISA okuma top-3: M0A'ya göre −0.16 puan, yani 0.16 puan daha iyi; marjın altında). İkincil yol koşulu sağlanmıyor.
7. **Testin gücü sınırlı.** TIMSS ve PIRLS'te 1–2 kovaryat ve küçük alpha nedeniyle permütasyon dağılımı çok dar (örneğin G8 fen: 17.908–17.908; PIRLS: 15.993–15.994). Bu programlarda KB ağırlıklarının tahmini değiştirme kapasitesi neredeyse sıfır. Buradaki "kanıt yok" sonucu, "KB zararlı" değil, "KB'nin etkisi ölçülemeyecek kadar küçük" anlamına geliyor.

## Duyarlılık: 2025 dahil

| PISA | M0A | M0 | M1 | En iyi placebo | Permütasyonlarda M1'den iyi veya eşit oran |
|---|---|---|---|---|---|
| Matematik | **11.20** | 12.66 | 13.45 | 12.59 (DD block) | %96.5 |
| Okuma | **13.13** | 14.53 | 15.02 | 14.33 (DD block) | %93.0 |
| Fen | **10.42** | 11.82 | 12.10 | 11.75 (Uniform 0.75) | %99.0 |

**VERIFIED.** 2025 dahil edildiğinde M1, PISA'nın üç alanında da M0'dan kötü (0.28–0.79 puan) ve rastgele ağırlıkların %93–99'u M1'den iyi veya eşit. TIMSS ve PIRLS'te 2025 fold'u olmadığı için değişiklik yok. Karar değişmiyor.

## Önceki iddialarla ilişki

| Önceki iddia | Bu fazdaki durum |
|---|---|
| "M1 shows consistent improvements" (9/10 program-alan) | Desteklenmiyor. Ortak düzenekte M1 − M0 farkı ≤ 0.09 puan; placebo'lar daha iyi; KB sıralaması rastgeleden iyi değil |
| Eski PISA okuma ΔMAE = 0.52 | Tekrar etmiyor: ana analizde 0.01, 2025 dahil −0.49 |
| TimesFM-KB pilotunda ağırlıklı − ağırsız düzeltme ≈ 0 | Tutarlı: KB ağırlıkları her iki entegrasyon yolunda da etkisiz |

## Tekrar üretilebilirlik

**VERIFIED.** Aynı girdilerle ikinci kez çalıştırıldı (scratchpad'e yazıldı). 7 CSV'nin tamamı `cmp` ile byte düzeyinde aynı: `contrasts_primary_excl2025`, `contrasts_sensitivity_incl2025`, `datadriven_adaptive_weights`, `datadriven_block_log`, `permutation_null`, `predictions`, `random_topk_null`.

## Çıktı dosyaları

`outputs/rq2_v3/`:
- `predictions.csv`: ülke düzeyinde tüm deterministik varyantlar; Faz 4 bootstrap'i için.
- `contrasts_primary_excl2025.csv`, `contrasts_sensitivity_incl2025.csv`
- `permutation_null.csv`: 8 × 200 çekiliş, atamalar JSON olarak.
- `random_topk_null.csv`
- `datadriven_block_log.csv`, `datadriven_adaptive_weights.csv`
- `manifest.json`: önceden kayıt hash'i, düzenek tahminlerinin hash'i, seed'ler.

## Faz 4'e devredilenler

- K1'in güven aralığı kısmı: M0A − M1 için ülke kümeli bootstrap %95 güven aralığı ve Holm düzeltmesi. Nokta tahmin zaten marjın altında olduğu için kararı değiştirmez, ama raporlanmalı.
- TOST (±1 puan): TIMSS ve PIRLS'te M1 − M0A ve M1 − M0 farklarının eşdeğerlik aralığında olup olmadığı. PISA'da M1, M0A'dan 1.6–1.7 puan kötü; eşdeğerlik beklenmiyor, ama güven aralığı raporlanmalı.
- RQ1 için persistence'a karşı tüm modellerin eşleştirilmiş farkları ve güven aralıkları.
