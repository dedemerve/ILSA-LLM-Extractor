# Provenance Audit — Final Frozen Results
**Date:** 2026-10-05  
**Branch:** fix/stage5-dual-source-build-xy

---

## 1. Evidence Corpus Provenance: 577 → 620 Discrepancy

### What each number represents

| Sayı | Tanım | Kaynak |
|-----:|-------|--------|
| **1458** | Ham JSON dosya sayısı (tüm klasörler) | 302 WoS + 423 Scopus + 591 OECD + 142 ilsa_survey |
| **1349** | ILSA-relevant JSON'lar (anahtar kelime filtresi) | WoS:294 + Scopus:386 + OECD:545 + survey:124 |
| **577** | WoS+Scopus ILSA-relevant unique DOI (OECD/survey hariç, pre-dedup) | `complete_pipeline.py` ITEM 6 — OECD genişletmesinden önce |
| **1144** | Evidence taşıyan (main_findings alanı dolu) ham study sayısı | WoS:284 + Scopus:374 + OECD:364 + survey:122 |
| **820** | Unique study (herhangi bir predictor, DOI/başlık dedup) | `provenance_audit.py` sayımı |
| **620** | **Canonical-mapped** unique study_id (RAW_TO_CANONICAL eşlemesi başarılı) | `expand_evidence_matrix.py` — W_{j,p,d} hesabına giren |

### Pipeline akışı

```
N_json_total         = 1458
       ↓ ILSA keyword filter
N_ilsa_relevant      = 1349
       ↓ main_findings field populated
N_evidence_bearing   = 1144  (ham, cross-source dedup öncesi)
       ↓ DOI / normalized-title dedup
N_unique_any         = 820   (herhangi bir predictor adı ile)
       ↓ RAW_TO_CANONICAL dict filter (expand_evidence_matrix.py)
N_canonical_mapped   = 620   ← TEZ'DE KULLANILACAK SAYI
```

### 577 neden ortaya çıktı?

`complete_pipeline.py` ITEM 6'da kaynak kapsamı raporlanırken **OECD ve ilsa_survey klasörleri hariç**, yalnızca WoS (294) + Scopus (386) ILSA-relevant dokümanlarının DOI-bazlı deduplikasyonu ≈577 verdi. Bu sayı N_evidence_bearing değil, N_ilsa_relevant'ın bir alt kümesidir. `expand_evidence_matrix.py`'ye OECD ve survey eklenince:
- N_unique_any → 820 (herhangi bir predictor adı olan)
- N_canonical_mapped → 620 (W_{j,p,d} hesabına katkıda bulunan)

200 çalışma (820−620) `main_findings` içeriyor ancak predictor isimleri `RAW_TO_CANONICAL` sözlüğüyle eşleşmiyor (örn. etnik köken, yabancı dil, okul türü gibi yapılar). Bu çalışmalar W ağırlıklarına katkıda bulunmuyor; `provenance_audit.md`'de sınırlılık olarak raporlanmalı.

### Tez için doğru ifade

> "The evidence corpus was compiled from four bibliographic databases: Web of Science (N=302), Scopus (N=423), OECD iLibrary (N=591), and a targeted survey of ILSA-specific ML studies (N=142), yielding 1,458 JSON records in total. After excluding non-ILSA records via keyword filter (N=1,349 retained), 1,144 documents contained structured predictor evidence (main_findings field populated). Cross-database deduplication by DOI and normalized title reduced the pool to 820 unique evidence-bearing studies. Of these, **N=620 unique studies** included at least one predictor mappable to the canonical construct vocabulary (RAW_TO_CANONICAL); these 620 studies contributed to the W_{j,p,d} weight computation. The remaining 200 studies (primarily reporting ethnicity, language background, or school-type predictors outside the canonical set) were retained in the database but did not enter the weighting model."

---

## 2. p-value Discrepancy: p=0.252 → p=0.017

### İki sonucun tam olarak ne olduğu

| Sonuç | Tarih | Giriş verisi | p-değeri |
|-------|-------|-------------|---------|
| **Pre-fix** | TIMSS_G4 düzeltmesinden ÖNCE | `evidence_matrix_full.csv` → `predictor_weights_v2_pd.csv` (hatalı) | 0.252 (n.s.) |
| **Post-fix** | TIMSS_G4 düzeltmesinden SONRA | `evidence_matrix_full.csv` → `predictor_weights_v2_pd.csv` (düzeltilmiş) | **0.017*** |

### Sorunun kökü

`build_evidence_matrix_v2.py`'deki program normalizasyon bloğu, `"TIMSS"` etiketli kanıtları panel'deki `"TIMSS_G4"` programıyla eşleştirmiyordu. Bu nedenle TIMSS_G4×science grubu sıfır kanıt görüyor, W_{j,TIMSS_G4,science} = 1.0 (düz M0 eşdeğeri) oluyordu. Düzeltmeden önce bu gruptaki M1_pd performansı M1'den kötüydü (Δ = −0.16); düzeltme sonrası +0.13'e döndü.

```python
# HATA (pre-fix): "TIMSS" kanıtları TIMSS_G4'e geçmiyordu
prog_list.append(p)  # sadece "TIMSS" ekliyordu

# DÜZELTME (post-fix): build_evidence_matrix_v2.py satır 111-118
for p in raw_progs:
    if p == "TIMSS":
        prog_list.extend(["TIMSS", "TIMSS_G4"])   # ikisine de
    elif p == "TIMSS_G4":
        prog_list.extend(["TIMSS", "TIMSS_G4"])
    else:
        prog_list.append(p)
prog_list = list(dict.fromkeys(prog_list))
```

### Aynı 45 katlama, farklı ağırlık girdisi

Her iki çalıştırmada da `m0_m1_m1pd_results.csv` **tamamen aynı 45 LOCO katlama, aynı panel satırları, aynı model kodu** kullanıldı. Tek fark: M1_pd'nin feature scaling'inde kullanılan `predictor_weights_v2_pd.csv` dosyası — hatalı vs düzeltilmiş.

Bu nedenle:
- p=0.252 → **artifact of a data bug**, tez'de raporlanmamalı
- p=0.017 → **final, correct result**, tezde raporlanacak tek değer

### Reproducibility

Mevcut `outputs/stage5/m0_m1_m1pd_results.csv` post-fix çıktısıdır. `validate_m1pd.py` bu dosyayı okur ve p=0.017 üretir. Pre-fix sonuçları kayıtlarda bulunmuyor (aynı dosya üzerine yazıldı) — bu bir kayıp değil; yanlış sonucun silinmesi intended davranış.

---

## 3. Final Frozen Results (post-fix)

### Model karşılaştırması — full panel (N=45 fold)

| Model | MAE | Açıklama |
|-------|-------|---------|
| M2 (persistence baseline) | 40.19 | Naive: bir önceki gözlemi tahmin |
| M0 (unweighted Ridge) | 17.08 | W_{j,p,d} = 1 (düz Ridge) |
| M1 (global W_j) | 17.02 | Program-agnostic literature weights |
| **M1_pd (global W_{j,p,d})** | **16.94** | Program×domain specific weights |

### H1 Test sonucu

```
n folds        = 45
mean(ΔMAE)     = +0.077  (pozitif = M1_pd kazanıyor)
95% CI         = [+0.003, +0.151]
wins/ties/loss = 28 / 4 / 13
p_perm         = 0.017 *
```

### Robustness checks

| Test | Sonuç |
|------|-------|
| Balanced panel (18 fold) | M2>M0>M1>M1_pd sıralaması korunuyor (23.98>13.05>12.82>12.67) |
| LOSO stability | 32/32 grup CV<25%, ort. CV=6.6% |
| External stability (W_v2 vs W_v3) | Pearson r=0.9933, MAE=0.028 |
| Evidence coverage | 7/8 matched pd×domain'da M1_pd < M1 |

---

## 4. Sensitivity Analyses (Ana sonuçlardan bağımsız)

Aşağıdaki analizler **ana pipeline dışında** kalmaktadır; sınırlılık/duyarlılık tartışmasına dahil edilecek:

- **Survey-weighted means**: Mikroveri kapsamı %2.4 (kısıtlı parquet dosyaları)
- **TALIS covariates**: %9 satır kapsamı; 16 ek değişken
- **Lag-residual feedback**: 1017 gözlem; leakage-safe walk-forward
- **PIAAC**: Tek katlama, persistence-only (Ridge için yetersiz N)
