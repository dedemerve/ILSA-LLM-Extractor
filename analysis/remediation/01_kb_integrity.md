# Faz 1 — Veri ve bilgi tabanı (KB) bütünlüğü

**Tarih:** 27 Ekim 2026
**Git:** `feat/finetuning-v2-pipeline` @ `ad7b100` (commit yok; tüm çıktılar yeni dosya)
Etiketler: **VERIFIED**, **ASSUMPTION**, **RECOMMENDATION** (Faz 0 ile aynı anlam).

## Kullanıcı kararları (Faz 0 sonrası)

| Karar | Uygulama |
|---|---|
| Origin tanımı A (katı tahmin) | Faz 2 düzeneğinde origin'den sonra yayımlanan kovaryatlar maskelenecek (00_provenance.md, tablo 0.2) |
| Tam `country_estimates` yeni dosyaya | Yapıldı, aşağıya bakınız |
| Bozuk TALIS 2018 sütunları dışarıda | `kb_feature_map_v3.csv` içinde `excluded_source_cycles=TALIS:2018` (SECLSS, TCDISCS, TJSPROS); Faz 2'de maskelenecek |
| 2025 hariç ana sonuç, 2025 dahil duyarlılık | Faz 2–4 raporlama kuralı |

**VERIFIED.** Origin tanımı A altında PISA 2025'teki TALIS değerlerinin 2022 kopyası olması (Faz 0 bulgu C) bir hata değil: Aralık 2023 origin'inde en son erişilebilir TALIS, TALIS 2018'dir (TALIS 2024 Ekim 2025'te yayımlandı, yayın tarihi ASSUMPTION). Bu değerler zaten TALIS 2018 dışlaması kapsamında maskeleniyor; yalnızca `TEFFPROS` (normal varyanslı T3EFFPD) kalıyor.

## 1.0 `country_estimates` geri yükleme

- Yeni dosya: `outputs/stage4/country_estimates_full_v2.csv` = `git show ad7b100^:outputs/stage4/country_estimates.csv` (3 601 satır, sha256 `cbc86d45fa16…`). Mevcut 237 satırlık dosyaya dokunulmadı.
- **VERIFIED.** 3 517 (program, alan, ülke, döngü) değeri `unified_panel.csv` ile karşılaştırıldı; uyuşmazlık 0. Panel bu tam sürümle üretilmiş.
- **RECOMMENDATION.** Faz 2 düzeneği `se_true` ve skorları bu dosyadan okumalı; `build_unified_panel.py` mevcut `ESTIMATES_CSV` yoluyla yeniden çalıştırılmamalı.

## 1.1 KB ağırlıkları v3

### Kaynak seçimi

Depoda üç ayrı KB kaynağı var:

| Kaynak | Kapsam | Ağırlık hesaplayan kod | M1'de kullanıldı mı | İnsan doğrulaması |
|---|---|---|---|---|
| `evidence_matrix.csv` → `literature_priority.csv` | 80 çalışma, 530 satır | `build_literature_priority.py` | Evet (değerler `predictor_weights_v2.csv`'ye elle kopyalanmış) | Yok |
| `effect_sizes.csv` | 577 makale, 3 083 kayıt | `compute_predictor_weights_v2.py` | Hayır (yalnızca F/D/C/E sütunları dosyada, `w_norm` değil) | Yok |
| Gold corpus v1.4.1 | 42 çalışma, 246 satır | — | Hayır | κ (aşağıda) |

**VERIFIED.** `literature_priority.csv`, `build_literature_priority.py` fonksiyonları bellekte çalıştırılınca birebir yeniden üretiliyor (19 yapının tamamında fark 0). Gold corpus ile `evidence_matrix` arasında `study_id` örtüşmesi 0, DOI örtüşmesi 4.

**Seçim: `evidence_matrix.csv` + `build_literature_priority.compute_weights`.** Gerekçeler:
1. Tezde raporlanan M1 müdahalesi bu ağırlıklarla yapıldı; v3 aynı sayıları üretiyor, ama artık elle değil scriptle.
2. Formül belgeli ve kodda: sıklık, program, döngü ve ülke çeşitliliğinin geometrik ortalaması × öngörücü kanıt katsayısı.
3. Gold corpus'ta sayısal etki büyüklüğü yok; tek başına ağırlık formülünü besleyemez.
4. `effect_sizes.csv` yolunun koddaki formülü (min-max) en düşük yapıya ağırlık 0 veriyor ve bu formül hiçbir zaman M1'e girmedi.

Sınırlılık: seçilen kaynak insan tarafından doğrulanmamış (1.3'e bakınız).

### Çıktılar

- `scripts/build_predictor_weights_v3.py` → `outputs/stage5/predictor_weights_v3.csv` ve `predictor_weights_v3_manifest.json`. Formül import ediliyor, kopyalanmıyor. Script üzerine yazmayı reddediyor.
- **VERIFIED.** Dosyadaki `w`, saklanan bileşen sütunlarından (`w_F, w_P, w_C, w_N, w_E, forecastable`) yeniden hesaplanıyor; script bunu `AssertionError` ile zorluyor. NaN ağırlık yok.
- Kanıt satırı olmayan üç yapı (TEACHER_SELF_EFFICACY_ENGAGEMENT, PROF_COLLABORATION, JOB_SATISFACTION_ENVIRONMENT) tabloda yer almıyor.

### v2 koduna göre değişen ağırlıklar (VERIFIED)

| Kovaryat | v2 kodundaki eşleme ve ağırlık | v3 eşleme ve ağırlık |
|---|---|---|
| lag_ASDHELA (PIRLS) | PARENTAL_EDU, 0.4559 | HOME_LITERACY_ACTIVITIES, 0.1029 |
| ICILS home_literacy_activities | HOME_LITERACY_ACTIVITIES, 0.7175 (HOMEPOS satırından kopya) | HOME_LITERACY_ACTIVITIES, 0.1029 |
| lag_SECLSS | TEACHER_QUALITY, 0.8543 | TEACHER_SELF_EFFICACY_CLS_MGMT, 0.8543 (değer aynı; TALIS 2018 kaynakları maskelenir) |
| lag_TCDISCS, lag_TJSPROS | 0.4910, 0.0894 | aynı; TALIS 2018 kaynakları maskelenir |
| Diğer kovaryatlar | | değişiklik yok |

## 1.2 Katı ağırlık arama

- `outputs/stage5/kb_feature_map_v3.csv`: ILSA değişkeni → kanonik yapı, `status` (`weighted` / `excluded`), `excluded_source_cycles`, not.
- `scripts/kb_weights.py` (`KBWeights`): tek ad alanı (önek olmadan ILSA değişken adı). Eşlenmemiş kovaryat, bilinmeyen yapı, açıkça dışlanmış değişken veya pozitif olmayan ağırlık durumunda `KeyError` fırlatıyor. Skor geçmişi özellikleri yalnızca çağıran taraf `neutral` olarak bildirirse 1.0 alıyor.
- **VERIFIED.** `unified_panel.csv` içindeki 12 `lag_*` sütununun hepsi eşleniyor; `lag_FOO` ve `PIAAC_gender` beklendiği gibi hata veriyor.
- Açık dışlamalar: `PIAAC_gender` ve TIMSS `ITSEX` (GENDER; KB'de forecastable değil, `w=0`).
- Mevcut scriptlere (`run_loco_forecasting.py`, `run_piaac_loco.py`, `run_icils_loco.py`) dokunulmadı; Faz 2 düzeneği bu modülü kullanacak.

## 1.3 Uyum istatistikleri

`scripts/compute_kappa_v2.py` → `outputs/kb_integrity_v1/` (`kappa_summary.json`, `per_construct_agreement.csv`, `confusion_matrix.csv`). Referans etiketler git'ten okunuyor (`ad7b100^` = `b39eb0f`).

| Ölçü | Değer |
|---|---|
| n kayıt / çalışma / insan kategorisi | 173 / 30 / 21 |
| κ (düzeltme öncesi etiketler) | **0.653** |
| Çalışma kümeli bootstrap %95 GA (5 000 tekrar) | **[0.450, 0.780]** |
| Basit uyum | %70.5 |
| κ (düzeltme sonrası, güvenirlik ölçüsü olarak geçersiz) | 0.687 |

**VERIFIED.** Gold etiketler ve yeniden kodlama aynı kişi (Merve_Dede) tarafından yapılmış (`annotator` sütunu 246/246; `human_validation_protocol.md` notu). Bu ölçü bağımsız iki kodlayıcı arasındaki uyum (inter-rater) değil, aynı kodlayıcının tekrarlı kodlama uyumudur (intra-rater). Tezde "independent human coder" ifadesi kullanılmamalı.

**VERIFIED.** Yapı bazında en düşük uyum (n ≥ 3):

| İnsan etiketi | n | Uyum |
|---|---|---|
| SELF_CONCEPT | 5 | 0.00 |
| TEACHER_SELF_EFFICACY_CLS_MGMT | 4 | 0.00 (LLM 4/4'ünü SELF_CONCEPT etiketlemiş) |
| EDUCATIONAL_ATTAINMENT | 5 | 0.20 |
| TEACHING_QUALITY | 4 | 0.25 |
| MOTIVATION, PROCESS_DATA_BEHAVIOR | 6, 9 | 0.33 |

SES_COMPOSITE 10/10 doğru. KB'de ikinci en yüksek ağırlığa (0.8543) sahip TEACHER_SELF_EFFICACY_CLS_MGMT, LLM'in sistematik olarak karıştırdığı yapı (DISCIPLINARY_CLIMATE ve SELF_CONCEPT ile). Bu bulgu gold corpus etiketlemesine ait; aynı hatanın `evidence_matrix`'te de olup olmadığı bir **ASSUMPTION**, aşağıdaki örneklemle test edilecek.

### KB'nin kendisi için doğrulama örneklemi

- `scripts/sample_evidence_matrix_validation.py` → `outputs/kb_integrity_v1/evidence_matrix_validation_sample_v1.csv`: 51 (çalışma, yapı) çifti, 35 çalışma; `w > 0` olan her yapıdan en fazla 6 çift (küçük yapılarda hepsi). Sıra karıştırılmış.
- Kodlanacak alanlar: `human_construct_correct`, `human_canonical_construct`, `human_direction`, `human_effect_type`, `human_analysis_level`, `human_programs_cycles_ok`, `human_notes`. Bunlar ağırlık formülünü belirleyen alanlar (yapı, program ve döngü kapsamı, öngörücü/ilişkisel tür).
- **RECOMMENDATION.** Bu örneklemi ikinci, bağımsız bir kodlayıcı (danışman veya bir meslektaş) kodlamalı. En az 30–50 çift yeterli. Yapı doğruluğu ve `effect_type` için κ ile birlikte, yanlış etiketler düzeltildiğinde ağırlıkların ne kadar değiştiği de raporlanmalı.

### KB kanıtının analiz düzeyi (VERIFIED)

`evidence_matrix.csv` içinde 530 satırın 486'sı öğrenci, 37'si öğretmen, 3'ü okul ve yalnızca 4'ü ülke düzeyinde. M1'de kullanılan yapılara ait 365 satırın yalnızca 3'ü ülke düzeyinde. Yani ağırlıklar öğrenci düzeyinde "ne sık çalışılmış" bilgisini ifade ediyor; ülke ortalamalarının döngüler arası değişimini öngörme gücünü değil. Bu, RQ2'de beklenen sıfır sonucun (Faz 3) kuramsal açıklamasıdır.

## 1.4 Düzeltme (adjudication) prosedürü

**VERIFIED (mevcut durum).**
- Gold corpus'taki 81 satır `gold_verified_after_adjudication`, 165 satır `gold_verified` durumunda; hepsi aynı kişi tarafından.
- `ad7b100` commit'inde insan yeniden kodlamasından sonra 5 etiket insan koduna çevrildi (G0113, G0182 → TEACHING_QUALITY; G0172 → EDUCATIONAL_ATTAINMENT; G0232, G0233 → HOME_LITERACY_ACTIVITIES). κ_v1 bu düzeltilmiş etiketlerle hesaplanmıştı.

**Bundan sonraki kural (RECOMMENDATION, tezde Method bölümüne yazılmalı):**
1. Uyum istatistikleri her zaman dondurulmuş, düzeltme öncesi etiket sürümüne karşı hesaplanır; sürüm git commit'iyle belirtilir.
2. Uyuşmazlıklar ayrı bir dosyada kaydedilir (`record_id`, iki etiket, karar, gerekçe, karar veren, tarih).
3. Düzeltilmiş etiketler KB'yi besleyebilir, ama yeni bir corpus sürümü olarak (örneğin v1.4.2) ve CHANGELOG notuyla.
4. Mümkünse uyuşmazlıklar üçüncü bir kişi veya danışman tarafından karara bağlanır; aynı kişinin hem kodlayıp hem karar vermesi Limitations bölümünde belirtilir.

## 1.5 KB'nin kendisi için uzman doğrulaması (27 Ekim 2026 eki)

**Tasarım.** `evidence_matrix_validation_sample_v1_blind.csv` dosyasındaki 51 (çalışma, yapı) çifti, 35 çalışma. Merve_Dede tarafından PDF'lerden, LLM etiketleri gizli olarak kodlandı (kodlayıcı beyanı). Tek uzman kodlayıcı; ölçülen şey LLM çıkarımı ile uzman kodlaması arasındaki uyum, iki insan arasındaki güvenirlik değil.
**Script:** `scripts/compute_evidence_validation_agreement.py` → `outputs/kb_integrity_v1/evidence_validation_*_v1.*`. Girdilerin sha256 değerleri JSON'da kayıtlı; bootstrap çalışma kümeli, 5 000 tekrar.

### Uyum (VERIFIED)

| Alan | κ | %95 GA (çalışma bootstrap) | Uyum |
|---|---|---|---|
| Yapı (canonical_construct) | **0.70** | [0.57, 0.83] | %72.5 |
| Etki türü (predictive/associative) | 0.21 | [0.00, 0.44] | %55 |
| Analiz düzeyi | 0.38 | [0.00, 0.71] | %84 |
| Yön | 0.01 | [−0.02, 0.05] | %2 |

### Bulgular (VERIFIED)

1. **KB'de ikinci en yüksek ağırlığa sahip yapı büyük ölçüde yanlış etiketlenmiş.** LLM'in TEACHER_SELF_EFFICACY_CLS_MGMT dediği 6 kaydın 6'sı da öğrenci öz-yeterliği (uzman: UNMATCHED). Gold corpus'taki 0/4 bulgusuyla tutarlı. Bu yapının ağırlığı (0.8543) sistematik bir etiket hatasına dayanıyor.
2. **Diğer yapıların doğruluğu:**
   - SES_COMPOSITE 3/6 doğru; yanlışlar karışık değişken grupları (örneğin metabiliş + ev eşyaları + okul kaçırma) veya başka yapılar.
   - ICT_HOME_ACCESS ve SCHOOL_BELONGING 4/6 doğru.
   - HOME_LITERACY_ACTIVITIES 0/1 doğru.
   - PARENTAL_EDUCATION, HOME_RESOURCES ve DISCIPLINARY_CLIMATE 6/6; EFFECTIVE_PD ve ICT_SCHOOL_ACCESS 3/3 doğru.
3. **LLM'in atadığı yönler doğrulanmıyor.** LLM 51 kaydın yalnızca 10'una yön atamış; uzman bunlardan yalnızca 1'ini doğruluyor (EV049, +). Diğer 9'unda makale işaretli bir etki raporlamıyor. Uzmanın 47/51 `unclear` kodu, çalışmaların çoğunun işaretsiz değişken önemi (feature importance) raporladığını gösteriyor. KB'deki yön bilgisi kullanılabilir değil.
4. **Etki türü sistematik olarak yanlış sınıflanmış.** LLM'in "associative" dediği 35 kaydın 21'ini uzman "predictive" olarak kodladı. Bu, ağırlık formülündeki w_E (öngörücü kanıt payı) bileşenini doğrudan etkiliyor.
5. **Analiz düzeyi:** LLM'in "teacher" dediği 8 kaydın 7'si, öğretmen değişkeninin öğrenci sonucuyla analiz edildiği öğrenci düzeyi çalışmalar.
6. **Sonuç değişkeni:** Uzman notlarına göre en az 9 kayıtta sonuç akademik başarı değil (iyi oluş, akademik yılmazlık, sınıf tekrarı, yurttaşlık katılımı, öğretmen ekip yenilikçiliği, dijital spor katılımı). `build_literature_priority.py` sonuç türüne göre filtre uygulamıyor. KB, başarıyı yordamayan çalışmaları da sayıyor.
7. **Program/döngü bilgisi** 51 kaydın 48'inde doğru. Hatalar: EV004 ve EV051'de TIMSS 2019 eksik; EV001'de program INVALSI (ILSA değil).

### Ağırlıklara etkisi (VERIFIED, `predictor_weights_v3_sample_corrected.csv`)

Yalnızca örneklenen 51 çift uzman etiketine göre düzeltildi (UNMATCHED çiftler çıkarıldı; 33 kanıt satırının etiketi değişti), formül aynı.

| Yapı | n çalışma (v3 → düzeltilmiş) | w (v3 → düzeltilmiş) |
|---|---|---|
| TEACHER_SELF_EFFICACY_CLS_MGMT | 21 → 15 | 0.854 → 0.677 |
| HOME_LITERACY_ACTIVITIES | 1 → 0 | 0.103 → (yok) |
| ICT_SCHOOL_ACCESS | 3 → 4 | 0.282 → 0.312 |
| PRIOR_ACHIEVEMENT | 1 → 2 | 0.048 → 0.149 |
| Diğerleri | | değişim ≤ 0.006 |

Örneklem kanıt satırlarının yaklaşık %10'unu kapsıyor. TEACHER_SELF_EFFICACY_CLS_MGMT'de örneklenen 6 çiftin hepsi yanlış olduğu için, tüm corpus düzeltilse bu yapının ağırlığının çok daha fazla düşmesi beklenir (**ASSUMPTION**: örneklem dışındaki 15 çalışmanın da benzer oranda yanlış olduğu varsayımına dayanıyor).

### RQ2 için anlamı

- Faz 3'teki permütasyon testi, KB ağırlıklarının herhangi bir atamasının rastgele atamalardan iyi olmadığını gösterdi. Bu yüzden etiket düzeltmelerinin RQ2 kararını (0/8) değiştirmesi beklenmiyor (**ASSUMPTION**; istenirse düzeltilmiş ağırlıklarla M1 yeniden çalıştırılabilir).
- Bu doğrulama, RQ2'deki sıfır sonuca ikinci bir açıklama ekliyor: KB yalnızca öğrenci düzeyinde değil, etiket düzeyinde de gürültülü. En yüksek ağırlıklı yapılardan biri yanlış etiketlenmiş, yön ve etki türü bilgisi güvenilir değil.

**Tezde kullanılacak ifade (RECOMMENDATION):** *"Agreement between LLM-extracted construct labels and blind expert coding (single coder, 51 study–construct pairs from 35 studies) was κ = 0.70 (95% study-bootstrap CI 0.57–0.83). Agreement was low for effect type (κ = 0.21) and direction (LLM-assigned signs were confirmed in 1 of 10 cases). The construct with the second-highest weight (teacher classroom-management self-efficacy) was mislabelled in 6 of 6 sampled pairs."* Limitations bölümünde tek kodlayıcı kullanıldığı belirtilmeli.

## Bu fazda yazılan dosyalar (hepsi yeni)

- `outputs/stage4/country_estimates_full_v2.csv`
- `scripts/build_predictor_weights_v3.py`, `outputs/stage5/predictor_weights_v3.csv`, `outputs/stage5/predictor_weights_v3_manifest.json`
- `outputs/stage5/kb_feature_map_v3.csv`, `scripts/kb_weights.py`
- `scripts/compute_kappa_v2.py`, `outputs/kb_integrity_v1/{kappa_summary.json, per_construct_agreement.csv, confusion_matrix.csv}`
- `scripts/sample_evidence_matrix_validation.py`, `outputs/kb_integrity_v1/evidence_matrix_validation_sample_v1.csv`

## Açık noktalar

- **ASSUMPTION.** `ASDHELA`'nın PIRLS'teki anlamı ("okula başlamadan önceki erken okuryazarlık etkinlikleri") PIRLS kullanıcı rehberinden teyit edilmeli; v3 eşlemesi buna dayanıyor.
- `W_CORPUS` yorumunda "132 articles" yazıyor; `evidence_matrix.csv` 80 çalışma içeriyor. Tezde doğru sayı 80 olmalı (VERIFIED, dosyadan).
- Doğrulama örneklemi 27 Ekim 2026'da kodlandı (bkz. 1.5). İnsan–insan güvenirliği için danışmanın yaklaşık 15 kaydı kör kodlaması hâlâ önerilir.
