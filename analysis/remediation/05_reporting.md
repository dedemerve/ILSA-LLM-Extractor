# Faz 5 — Raporlama tabloları ve tez iddiaları için önerilen farklar

**Tarih:** 27 Ekim 2026
**Git:** `feat/finetuning-v2-pipeline` @ `ad7b100` (commit yok; tüm çıktılar yeni dosya)
**Etiketler:** VERIFIED, ASSUMPTION, RECOMMENDATION.
**Kısıt:** Hiçbir tez veya mevcut rapor dosyası değiştirilmedi. Aşağıdakiler yalnızca önerilerdir.

**Kaynak notu.** Tez el yazması (Word/PDF) depoda ve masaüstünde bulunamadı. Bu yüzden farklar iki tür kaynağa göre yazıldı:
- **[P]:** İlk inceleme isteğinde makaleye ait olduğu belirtilen iddialar.
- Depodaki belgeler:
  - **[E]** `analysis/04_rq1_rq2_evidence_report.md`
  - **[R]** `README.md`
  - **[H]** `analysis/human_validation_protocol.md`
  - **[C]** `CHANGELOG.md` ve `ad7b100` commit mesajı

Tez dosyası paylaşılırsa her öneri cümle düzeyinde eşlenebilir.

## 5.1 RQ1 tabloları

**Script:** `scripts/build_rq1_tables_v1.py`
**Çıktı:** `outputs/rq1_tables_v1/`
- `rq1_tables.md`: tez ekine hazır Markdown tablolar.
- `rq1_per_program_domain.csv`
- `rq1_aggregate_relative_mae.csv`

**Tasarım:**
- Her program-alan için tek tablo; tablodaki her model aynı satırlarda puanlanıyor.
- MAE ve göreli MAE (model / persistence) için ülke kümeli bootstrap %95 güven aralığı (B = 2 000).
- Sıralama belirsizliği: her bootstrap tekrarında tüm modeller birlikte yeniden puanlanıyor; %95 sıralama aralığı ve P(1. sıra) raporlanıyor.
- Persistence'a karşı anlamlılık ve eşdeğerlik etiketleri Faz 4'ten (Holm, TOST ±1).
- TimesFM, kendi alt kümesindeki ayrı tablolarda.
- Program-alanlar arası ham MAE ortalaması alınmıyor; toplu sıralama göreli MAE'nin geometrik ortalamasıyla yapılıyor.
- Ana analiz 2025 hariç; duyarlılık analizi CSV'de.

**VERIFIED.** İkinci çalıştırmada 3 dosyanın tamamı byte düzeyinde aynı.

### Toplu tablo (ana analiz, 8 program-alan, ortak test kümesi)

| Model | Göreli MAE (geometrik ortalama) | Nokta sıralamasında 1. | Ortalama P(1. sıra) | Anlamlı daha iyi | Anlamlı daha kötü | ±1 içinde eşdeğer |
|---|---|---|---|---|---|---|
| Persistence | 1.000 | 3 | 0.39 | — | — | — |
| AR1 | 1.027 | 2 | 0.18 | 0 | 0 | 2 |
| Drift | 1.030 | 1 | 0.19 | 0 | 2 | 3 |
| Ridge_M0A | 1.035 | 0 | 0.07 | 0 | 1 | 0 |
| Ridge_M1 | 1.093 | 1 | 0.03 | 0 | 3 | 0 |
| Ridge_M0 | 1.095 | 1 | 0.13 | 0 | 3 | 0 |
| XGB_score | 1.182 | 0 | 0.00 | 0 | 2 | 0 |
| XGB_cov | 1.195 | 0 | 0.00 | 0 | 2 | 0 |
| RF_score | 1.218 | 0 | 0.00 | 0 | 3 | 0 |
| RF_cov | 1.220 | 0 | 0.00 | 0 | 3 | 0 |
| LinearTrend | 1.299 | 0 | 0.00 | 0 | 8 | 0 |
| InternationalMean | 3.281 | 0 | 0.00 | 0 | 8 | 0 |

TIMSS G4 matematikte M0 ve M1 berabere oldukları için nokta sıralamasında ikisi de 1. sayılıyor.

**Sıralama belirsizliği (VERIFIED):** Nokta tahmininde birinci olan modelin P(1. sıra) değeri 0.20 ile 0.79 arasında.
- Persistence: PISA matematik 0.72, PISA okuma 0.53, PIRLS 0.71.
- Drift: PISA fen 0.67.
- AR1: TIMSS G8 matematik 0.79, TIMSS G8 fen 0.32.
- Ridge: TIMSS G4 matematik 0.45, TIMSS G4 fen 0.20.

Persistence'ın %95 sıralama aralığı her program-alanda 1. sırayı içeriyor (alt sınır her yerde 1). Yani hiçbir program-alanda "en iyi model" tekil olarak belirlenemiyor. Ama persistence hiçbir yerde güvenle dışlanamıyor.

## 5.2–5.4 İddia farkları

Biçim: **Eski** (kaynak) → **Yeni** (önerilen İngilizce tez cümlesi) → **Kanıt**.

### A. Yöntem ve tasarım

**A1. Doğrulama şemasının adı**
- **Eski:** "Leave-One-Cycle-Out (LOCO)" [P], [R] satır 25, 56, 83; [E] RQ2 bölümü.
- **Yeni:** *"Models were evaluated in an expanding-window (rolling-origin) backtest: each cycle t of a program-domain was forecast using only cycles before t, with information restricted to what had been publicly released when the previous cycle's results appeared (forecast origin)."*
- **Kanıt:** `analysis/remediation/02_unified_backtest.md`; `scripts/run_unified_backtest.py`; `outputs/unified_backtest_v1/manifest.json`.

**A2. RQ1 karşılaştırma tablosu**
- **Eski:** Farklı fold tasarımlarını birleştiren tek tablo (`outputs/ilsa_benchmark_merged_v2/rq1_comparison.csv`, `rq1_summary_by_model.csv`) [P].
- **Yeni:** *"All models were scored on the same country–cycle pairs within each program-domain; countries without a prior score in the same program were excluded from the comparison and reported separately (580 country-cycles)."*
- **Kanıt:** `outputs/rq1_tables_v1/rq1_tables.md`; `outputs/unified_backtest_v1/no_history.csv`.
- **Not:** `ilsa_benchmark_merged_v2` tezde kullanılmamalı. Persistence aynı program-alanda farklı değerler alıyor (9.69 ve 10.03) ve kovaryat alanları hedef sayılmış.

**A3. M1'in tanımı**
- **Eski:** "Predictor features scaled by sqrt(w_j), w_j = normalized citation frequency from a KB of 246 annotated studies" [P].
- **Yeni:** *"In M1, standardized contextual covariates were multiplied by sqrt(w_j) before Ridge estimation, which is equivalent to a covariate-specific penalty α/w_j (a generalized ridge prior). The weights w_j were computed by a scripted composite of study frequency, program, cycle and country diversity, and the share of predictive evidence, from an LLM-extracted evidence matrix of 80 studies (530 evidence rows)."*
- **Kanıt:** `scripts/build_predictor_weights_v3.py`; `outputs/stage5/predictor_weights_v3.csv` ve `_manifest.json`; `analysis/remediation/01_kb_integrity.md` §1.1.

**A4. Ağırlık tablosu**
- **Eski:** `predictor_weights_v2.csv` (w_norm elle kopyalanmış; HOME_LITERACY_ACTIVITIES satırı HOMEPOS'tan kopya) [P].
- **Yeni:** v3 ağırlıkları. Methods bölümüne eklenecek cümle: *"Weights are reproducible from their stored components; two variable-to-construct mappings were corrected (PIRLS ASDHELA → home literacy activities; TALIS SECLSS → teacher classroom-management self-efficacy), and degenerate TALIS 2018 country estimates (between-country SD ≈ 0.001) were excluded."*
- **Kanıt:** `analysis/remediation/00_provenance.md` bulgu B; `01_kb_integrity.md` §1.1–1.2; `outputs/stage5/kb_feature_map_v3.csv`.

**A5. İstatistiksel test**
- **Eski:** "Diebold–Mariano test (HLN corrected) compares M1 vs M2" [R] satır 83.
- **Yeni:** *"Paired differences in absolute error were assessed with country-cluster bootstrap confidence intervals (B = 5,000), with cycle-cluster and two-way (country × cycle) bootstraps as sensitivity analyses; p-values were Holm-adjusted across program-domains, and practical equivalence was tested with TOST using a ±1-point margin."*
- **Kanıt:** `analysis/remediation/04_inference.md`; `scripts/run_inference_v1.py`.

**A6. Eşdeğerlik marjının gerekçesi (yeni cümle)**
- *"The 1-point margin corresponds to about 40% of the median standard error of a country mean (2.5–3.3 points) and is below the published PISA 2022→2025 link errors for mathematics (1.22) and reading (1.09)."*
- **Kanıt:** `outputs/inference_v1/scale_context.csv`; PISA 2025 Technical Report, Annex Table 15.A.20.

**A7. PISA 2025**
- **Eski:** 2025, diğer fold'larla eşit bir test fold'u [P].
- **Yeni:** *"PISA 2025 outcomes were available before the analysis was finalized; the 2025 fold is therefore reported only as a sensitivity analysis, and primary results exclude it."*
- **Kanıt:** `00_provenance.md` §0.1 (PUF'tan yeniden hesaplama, fark 0); `pisa_forecasting_panel_v2_manifest.json` (`outcome_exposed`).

**A8. Veri kaynağı**
- **Eski:** `outputs/stage4/country_estimates.csv` (mevcut sürüm yalnızca PISA 2018, 237 satır).
- **Yeni:** Methods bölümünde `country_estimates_full_v2.csv` (3 601 satır; `ad7b100^` sürümünden geri yüklendi) kaynak gösterilmeli.
- **Kanıt:** `01_kb_integrity.md` §1.0.

### B. RQ1 sonuçları

**B1. En iyi model**
- **Eski:** TimesFM LoRA'nın ortalama sırası 1.33, Persistence_M2'ninki 2.62 vb. (`rq1_summary_by_model.csv`) [P]. [E]'de: "PISA'da Persistence üç alanın üçünde de en düşük MAE … diğer program-alanlarda 10 grubun 9'unda Persistence."
- **Yeni:** *"No model significantly outperformed persistence in any of the eight program-domains after Holm correction, under any of the three bootstrap schemes. Persistence had the lowest point MAE in 3 of 8 program-domains, AR(1) in 2, drift in 1 and Ridge in 2 (TIMSS Grade 4). Persistence's 95% rank interval included rank 1 everywhere, and the most likely winner had a bootstrap probability of rank 1 between 0.20 and 0.79."*
- **Kanıt:** `outputs/rq1_tables_v1/rq1_aggregate_relative_mae.csv`, `rq1_per_program_domain.csv`; `04_inference.md` RQ1 tablosu.

**B2. RF/XGBoost ile zaman serisi modelleri**
- **Eski:** [E]'de "RF/XGBoost ile Persistence arasındaki 20 kontrastın 14'ünde aralık Persistence lehine…" (çoklu karşılaştırma düzeltmesi yok).
- **Yeni:** *"Random forest and XGBoost never had lower point MAE than persistence (geometric-mean relative MAE 1.18–1.22). After Holm correction, random forest was significantly worse in all three PISA domains and XGBoost in PISA mathematics and science (by 1.3–1.9 points); for TIMSS and PIRLS, point estimates were 2.5–5.6 points worse but confidence intervals were too wide (±2–5 points) for significance. Adding contextual covariates did not improve either tree model."*
- **Kanıt:** `04_inference.md`; `rq1_tables.md`.

**B3. TimesFM**
- **Eski:** "TimesFM LoRA rank 1 in 2/3, zero-shot mean rank 1.9" [P].
- **Yeni:** *"On the subset with saved TimesFM 2.5 zero-shot forecasts, TimesFM did not significantly outperform persistence in any program-domain; it was practically equivalent in PISA reading and significantly worse in TIMSS Grade 4 mathematics (−2.3 points)."*
- **Kanıt:** `rq1_tables.md` (TimesFM subset tabloları); `04_inference.md`.
- **Not:** LoRA karşılaştırması bu düzenekte yeniden yapılmadı. LoRA iddiası yalnızca `reports/final_synthesis_report.md` içindeki PISA tasarımıyla sınırlı kalmalı.

**B4. ICCS, ICILS, PIAAC**
- **Eski:** Ana RQ1 tablosunda sıralanıyorlar. PIAAC'ta M0/M1 persistence'ı geçiyor (literacy 8.28 vs 10.51; numeracy 7.26–7.33 vs 10.14) [P].
- **Yeni:** *"ICCS, ICILS and PIAAC offer one usable forecast origin each and are reported descriptively in an appendix. For PIAAC, no regression model could be trained without same-wave information; persistence MAE was 10.5 (literacy) and 10.1 (numeracy), n = 15."*
- **Kanıt:** `outputs/unified_backtest_v1/summary_appendix_*.csv`.
- **Not:** Önceki PIAAC "üstünlüğü" aynı dalgadan sızıntıdan kaynaklanıyordu (numeracy literacy'yi, literacy numeracy'yi tahmin ediyordu); bu sonuç tezden çıkarılmalı.

**B5. Kapsam cümlesi (RQ1'in literatür kısmı)**
- [E]'deki "RF ve XGBoost genel predictive korpusta en sık iki etiket; dar ülke-düzeyi alt kümede SVM önde" ifadesi bu düzeltme sürecinde yeniden incelenmedi. Olduğu gibi kalabilir (**ASSUMPTION**: sayımlar değişmedi).

### C. RQ2 sonuçları

**C1. Ana RQ2 iddiası**
- **Eski:** "M1 shows consistent improvements" (10 program-alanın 9'unda, ortalama ΔMAE ≈ 0.3) [P]. [E]: "…ağırlıksız Ridge'e göre çoğunlukla küçük iyileşmeler göstermiştir…"
- **Yeni:** *"Under a decision rule pre-registered before the placebo and permutation analyses, the literature-derived knowledge base did not improve forecasts in any of the eight program-domains. Literature-weighted Ridge (M1) was practically equivalent to unweighted Ridge (M0) in all eight (TOST, ±1 point) and significantly worse than the score-only Ridge (M0A) for all three PISA domains (by 1.6–1.7 points)."*
- **Kanıt:** `analysis/remediation/03_preregistration.md` (sha256 `1713cbb6…`); `03_rq2_redesign.md`; `outputs/inference_v1/rq2_preregistered_decision.csv`.
- **Not:** Eski "ortalama ΔMAE ≈ 0.3" değeri, eski tabloda bile yanlıştı (gerçek değer 0.16; iyileşen satırlarda 0.18).

**C2. KB içeriğinin rolü (yeni cümle)**
- *"Random reassignments of the knowledge-base weights performed as well as or better than the actual weights in 23–99% of 200 draws per program-domain (99% in PISA mathematics), and uniform or data-driven shrinkage of covariates matched or exceeded M1. The small M1-versus-M0 gains (≤ 0.09 points) therefore reflect generic covariate shrinkage, not the content of the knowledge base."*
- **Kanıt:** `outputs/rq2_v3/contrasts_primary_excl2025.csv`, `permutation_null.csv`.

**C3. KB'ye dayalı özellik seçimi (yeni cümle)**
- *"Selecting the top-k covariates by knowledge-base weight was not better than random selection and did not reach the 1-point margin over the score-only model."*
- **Kanıt:** `outputs/rq2_v3/random_topk_null.csv`.

**C4. TimesFM-KB pilotu**
- [E] ile tutarlı (ağırlıklı − ağırlıksız = −0.0009); değişiklik gerekmiyor.
- Model adlarındaki `M1_unweighted` ve `M2_literature_weighted` tezdeki M1/M2 ile karışmamalı; metinde "unweighted/weighted covariate correction" denmeli.
- **Kanıt:** `outputs/timesfm_kb_integration_v3/`.

**C5. Sıfır sonucun yorumu (yeni paragraf)**
- *"Three features of the evidence explain why literature importance did not transfer to forecast utility: (i) 486 of 530 evidence rows were student-level and only 4 were country-level, so the weights reflect how often a construct is studied, not its ability to predict changes in country means; (ii) contextual covariates change slowly between cycles relative to score changes, and inner validation frequently chose to shrink them to zero; and (iii) the extraction itself was noisy for constructs and effect types (see D2)."*
- **Kanıt:** `01_kb_integrity.md` §"KB kanıtının analiz düzeyi" ve §1.5; `outputs/rq2_v3/datadriven_block_log.csv`.

### D. KB kalitesi ve güvenirlik

**D1. κ = 0.69**
- **Eski:** "κ = 0.69 (substantial agreement)" [P]; [C] commit mesajı; [H] satır 7: "bağımsız insan kodlaması… inter-rater güvenilirliği".
- **Yeni:** *"Re-coding of 173 gold-corpus records (30 studies) by the same researcher who had verified the gold labels yielded κ = 0.65 (95% study-bootstrap CI 0.45–0.78) against the pre-adjudication labels; this is intra-rater agreement."*
- **Kanıt:** `outputs/kb_integrity_v1/kappa_summary.json`; `01_kb_integrity.md` §1.3.
- **Not:** 0.69, 5 etiketin insan koduna göre düzeltilmesinden sonra hesaplanmıştı; güvenirlik ölçüsü olarak geçersiz.

**D2. KB'nin kendisinin doğrulanması (yeni paragraf)**
- *"For the evidence matrix that produced the weights, a blind expert coding of 51 study–construct pairs (35 studies; single coder) agreed with the LLM construct labels at κ = 0.70 (95% CI 0.57–0.83) but agreed poorly on effect type (κ = 0.21) and direction (LLM-assigned signs confirmed in 1 of 10 cases). All six sampled pairs labelled teacher classroom-management self-efficacy—the construct with the second-highest weight—were actually student self-efficacy."*
- **Kanıt:** `outputs/kb_integrity_v1/evidence_validation_agreement_v1.json`, `evidence_validation_per_construct_v1.csv`; `01_kb_integrity.md` §1.5.

**D3. Corpus büyüklüğü**
- **Eski:** "246 annotated studies" [P]; W_CORPUS yorumu "132 articles".
- **Yeni:** *"The gold corpus comprises 246 annotated rows from 42 studies and was used for validation; the knowledge base used for weighting comprises 530 evidence rows from 80 studies."*
- **Kanıt:** `outputs/finetuning/data/gold_standard_full_corpus_v1.4.1.csv` (`study_id` sayısı); `outputs/stage5/evidence_matrix.csv`.

### E. Sınırlılıklar bölümüne eklenecek maddeler

1. Program-alan başına 3–6 tahmin başlangıcı (origin); döngü düzeyinde çıkarım zayıf. Ülke kümeli bootstrap ortak döngü şoklarını hafife alabilir; twoway şeması duyarlılık kontrolü.
2. TIMSS ve PIRLS'te güven aralıkları ±2–5 puan; 2 puanın altındaki farklar çözümlenemiyor.
3. KB, sonuç verildikten sonra yayımlanan literatürden kuruldu; erken fold'lar için hafif bir ileriye bakış (look-ahead) içeriyor.
4. Tek kodlayıcı: hem gold doğrulamada hem KB doğrulamada. İnsan–insan güvenirliği raporlanmadı (RECOMMENDATION: danışman yaklaşık 15 kaydı kör kodlasın).
5. PISA 2025 sonuçları analiz tamamlanmadan önce görüldü; yalnızca duyarlılık analizinde kullanıldı.
6. PISA 2015 mod değişikliği (kâğıttan bilgisayara) ve yeniden ölçekleme; trend karşılaştırmalarında link error. Yalnızca PISA 2025 link error değerleri yerelde doğrulandı; diğerleri teknik raporlardan kaynak gösterilmeli.
7. Yayın tarihleri dış bilgiye dayanıyor. Duyarlılık testi yalnızca TALIS 2013 ve TALIS 2018 tarihlerinin sonucu değiştirebildiğini gösterdi (±6–12 ay); bu iki tarih kaynak gösterilmeli.
8. TALIS 2018 ülke tahminleri bozuk olduğu için dışlandı; PISA 2022/2025 fold'larında TALIS bilgisi yalnızca TALIS 2013 ve TEFFPROS'tan geliyor.
9. Tekrarlı kovaryat satırlarında (118) en büyük örneklemli satır seçildi (ASSUMPTION: bu, ana örneklemdir).

### F. Depo belgeleri için öneriler (tez dışı)

- **[R] `README.md`:** "LOCO" ve Diebold–Mariano anlatımı güncellenmeli; yeni scriptlerin çalıştırma sırası eklenmeli:
  1. `build_predictor_weights_v3.py`
  2. `run_unified_backtest.py`
  3. `run_rq2_v3.py`
  4. `run_inference_v1.py`
  5. `build_rq1_tables_v1.py`
  6. `compute_kappa_v2.py`
  7. `compute_evidence_validation_agreement.py`
- **[H] `human_validation_protocol.md`:**
  - Satır 7'deki "bağımsız insan kodlaması / inter-rater" ifadesi düzeltilmeli.
  - Satır 13'teki `predictor_weights_v2.csv` referansı v3 olmalı.
- **[E] `04_rq1_rq2_evidence_report.md`:** RQ1 ve RQ2 bölümlerinin bu düzeltme serisiyle geçersiz kılındığını belirten bir not eklenmeli. Dosyanın üzerine yazılmamalı; yeni sürüm ayrı bir dosya olmalı.
- **`reports/final_synthesis_report.md`:** Yalnızca PISA ve ayrı bir model kümesi içeriyor. "Persistence lowest equal-fold MAE in all three domains" sonucu bu düzenekle çelişmiyor; değişiklik gerekmiyor.

## Açık noktalar

- Tez el yazmasının dosyası paylaşılırsa önerileri cümle düzeyinde eşlerim.
- Hangi önerilerin tez metnine uygulanacağı kullanıcının kararı; hiçbir tez dosyası değiştirilmedi.
