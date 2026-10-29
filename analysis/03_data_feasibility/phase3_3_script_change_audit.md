# Phase 3.3 — Script Change Audit

**Tarih:** 2026-10-14  
**Yetkilendirilen değişiklikler:** İki (2)  
**Değiştirilen dosya:** `scripts/step2_timesfm_zeroshot.py`  
**Yeni dosya:** `analysis/03_data_feasibility/phase3_3_script_change_audit.md` (bu dosya)  
**Değerlendirme ÇALIŞTIRILMADI.**

---

## 1. Değiştirilen Dosyalar

| Dosya | Değişiklik türü |
|-------|----------------|
| `scripts/step2_timesfm_zeroshot.py` | İki yetkili değişiklik uygulandı (aşağıda ayrıntılar) |
| `analysis/03_data_feasibility/phase3_3_script_change_audit.md` | Yeni denetim raporu (bu dosya) |

Başka hiçbir üretim scripti, çıktı dosyası veya rapor değiştirilmedi.  
Mevcut takip edilen dosyalardaki değişiklikler (`phase3_data_feasibility_audit.md`, `scripts/finetuning/extract_v2.py`) korundu.

---

## 2. Değişiklik 1 — Çıktı Dizini Yalıtımı

### Eski davranış
```python
OUT_DIR  = ROOT / "outputs" / "step2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TEST_CYCLE  = 2022
PROGRAM     = "PISA"
DOMAIN      = "mathematics"
TARGET_COL  = "PISA_mathematics"
MIN_CONTEXT = 3
```
`TEST_CYCLE = 2025` yapılsaydı, script `outputs/step2/` altına yazardı ve mevcut 2022 sonuçları silinirdi.

### Yeni davranış
```python
# KONFİGÜRASYON bloğu OUT_DIR'den ÖNCE tanımlanıyor
TEST_CYCLE  = 2022
PROGRAM     = "PISA"
DOMAIN      = "mathematics"
TARGET_COL  = "PISA_mathematics"
MIN_CONTEXT = 3

_LEGACY_2022_DIR = ROOT / "outputs" / "step2"
OUT_DIR = _LEGACY_2022_DIR if TEST_CYCLE == 2022 else ROOT / "outputs" / f"step2_{TEST_CYCLE}"
assert not (TEST_CYCLE != 2022 and OUT_DIR == _LEGACY_2022_DIR), (
    f"Güvenlik hatası: TEST_CYCLE={TEST_CYCLE} için {_LEGACY_2022_DIR} dizinine yazılamaz; "
    "mevcut PISA 2022 sonuçları korunmalıdır."
)
OUT_DIR.mkdir(parents=True, exist_ok=True)
```

### Davranış matrisi (statik doğrulandı)

| TEST_CYCLE | OUT_DIR | Legacy 2022 üzerine yazar mı? | Assertion geçer mi? |
|-----------|---------|------------------------------|---------------------|
| 2022 | `outputs/step2/` | Hayır (aynı dizin, zaten 2022 çalıştırması) | EVET |
| 2025 | `outputs/step2_2025/` | **Hayır** (farklı dizin) | EVET |
| Diğer yıl | `outputs/step2_<yıl>/` | **Hayır** | EVET |

### Korunan 2022 dosyaları (SHA-256 doğrulandı)

| Dosya | Boyut | SHA-256 (ilk 12 hane) |
|-------|-------|-----------------------|
| `outputs/step2/zeroshot_results.csv` | 5088 B | `5b189e2b0b8c` |
| `outputs/step2/metrics_summary.csv` | 60 B | `cdbffdac89df` |
| `outputs/step2/zeroshot_scatter.png` | 132607 B | `bacc69ca51af` |

---

## 3. Değişiklik 2 — Doğrusal Trend Baseline

### Formül ve uygulama

```python
def _linear_trend_pred(context_years, context_scores, target_year):
    """
    OLS doğrusal trend: puan = kesişim + eğim × yıl
    context_years: gerçek değerlendirme yılları (indeks değil)
    Döndürür: (intercept, slope, prediction)
    Bağlam sayısı < 2 ise (nan, nan, nan) döner.
    """
    if len(context_years) < 2:
        return (np.nan, np.nan, np.nan)
    cy = np.array(context_years, dtype=np.float64)
    cs = np.array(context_scores, dtype=np.float64)
    slope, intercept = np.polyfit(cy, cs, 1)
    prediction = intercept + slope * float(target_year)
    return (float(intercept), float(slope), prediction)
```

**Doğrulama:** `numpy.polyfit(cy, cs, 1)` OLS çözümünü (en küçük kareler) verir.  
Zaman değişkeni: gerçek yıl sayısı (örn. 2003, 2006, ... 2022) — düzensiz aralıklar doğru şekilde ele alınır.  
Hedef yıl: `TEST_CYCLE` değeri (2025 çalıştırması için 2025).

### Statik fonksiyon testi

| Test senaryosu | Girdi | Çıktı |
|----------------|-------|-------|
| LUX 6 döngü (2003–2018 → 2025 tahmini) | `[2003,2006,2009,2012,2015,2018]`, `[493.21,490.00,489.07,489.85,485.77,483.42]` | eğim=−0.5796, pred=480.15 |
| 3 nokta minimum | `[2015,2018,2022]`, `[350,360,355]` | eğim=0.6081, pred=359.05 |
| 1 nokta (yetersiz) | `[2022]`, `[400]` | pred=`nan` (doğru) |

### Uygulanma konumu

`_linear_trend_pred` fonksiyonu her iki sonuç dalında çağrılır (`USE_TIMESFM=True` ve `USE_TIMESFM=False`). Her ülke kaydı için:
```python
lt_intercept, lt_slope, lt_pred = _linear_trend_pred(
    r["context_cycles"], r["context_scores"], TEST_CYCLE
)
```

### Yeni çıktı sütunları

Her ülke satırına eklenen sütunlar:

| Sütun | Tanım |
|-------|-------|
| `linear_trend` | OLS ile tahmin edilen puan (hedef yıl) |
| `linear_intercept` | OLS kesişim noktası |
| `linear_slope` | OLS eğimi (puan/yıl) |
| `err_linear` | `abs(true_score − linear_trend)` |

**Uyarı:** `err_linear`, `err_persistence` ve `err_timesfm` gibi mutlak hatadır.  
Protokol gereği **işaretli hata** sütunları `prediction − actual` formülüyle sonradan hesaplanmalıdır:
```python
df["signed_err_persistence"] = df["persistence"] - df["true_score"]
df["signed_err_linear"]      = df["linear_trend"] - df["true_score"]
df["signed_err_timesfm"]     = df["timesfm_zeroshot"] - df["true_score"]
```

### Mevcut 2022 davranışı üzerindeki etkisi

`TEST_CYCLE=2022` çalıştırmasında `_linear_trend_pred` de çağrılacak ve çıktı CSV'ye `linear_trend`, `linear_intercept`, `linear_slope`, `err_linear` sütunları eklenecektir. Mevcut `persistence`, `timesfm_zeroshot`, `err_persistence`, `err_timesfm` sütunları değişmez. Eklemeler geriye dönük uyumludur.

---

## 4. Ortak Ülke Karşılaştırması — Uygulama

Protokol Bölüm 7: Herhangi bir yöntem farklı sayıda ülke üretirse, karşılaştırma **kesişim** üzerinden yapılır.

**`_linear_trend_pred` hiçbir ülkeyi sessizce dışlamaz:** bağlam sayısı < 2 olan ülke için `linear_trend = nan` döner ve bu ülke `err_linear` hesabına dahil edilmez. `MIN_CONTEXT=3` filtresi geçen tüm ülkeler ≥ 3 tarihi gözleme sahip olduğundan, `nan` durumu oluşmaması beklenir.

**Ortak ülke mantığı:** `metrics_summary.csv`'deki `n` değeri her yöntem için ayrı raporlanır. Bir ülke TimesFM için `nan` üretirse (yükleme hatası), MAE sadece TimesFM için değişir; persistence ve linear trend MAE'leri etkilenmez. Bu durum raporda belirtilmelidir.

---

## 5. Adil Karşılaştırma Gereksinimleri Uyum Kontrolü

| Gereksinim | Sağlandı mı? |
|-----------|-------------|
| Aynı hedef döngü (TEST_CYCLE) | EVET — tüm yöntemler aynı TEST_CYCLE'ı kullanır |
| Aynı hedef değişken (PISA_mathematics) | EVET — tek kaynak: `unified_panel.csv`, `PISA_mathematics` sütunu |
| Aynı ülke seti (n=66) | EVET — MIN_CONTEXT=3 filtresi tüm yöntemler için aynı |
| Aynı gerçek değerler | EVET — `r["true_score"]` tüm yöntemler için paylaşılır |
| Aynı hata tanımı | EVET — `abs(true_score − pred)` tutarlı |
| Kalite işaretli ülkeler dahil (n=66) | EVET — filtreleme yapılmaz; değerlendirme raporunda ayrı katman olarak belirtilecek |
| TimesFM fine-tune yapılmaz | EVET — `predict_batch` çağrısı değişmedi |
| PISA 2025 sonuçları model seçimine dahil edilmez | EVET — değerlendirme çalıştırılmadı |

---

## 6. Çıktı Şeması Özeti

`zeroshot_results.csv` (güncellenmiş şema):

```
country, n_context, last_cycle, true_score,
persistence, linear_trend, linear_intercept, linear_slope,
timesfm_zeroshot,
err_persistence, err_linear, err_timesfm
```

`metrics_summary.csv` (güncellenmiş şema):
```
model, MAE, n
Persistence, <değer>, <n>
Linear_trend, <değer>, <n>        ← yeni satır
TimesFM_zero-shot, <değer>, <n>
```

---

## 7. Gerçekleştirilen Statik Kontroller

| Kontrol | Sonuç |
|---------|-------|
| Python sözdizimi (`py_compile`) | GEÇTI |
| `_linear_trend_pred` fonksiyonu tanımlı mı? | GEÇTI |
| Yeni sütun adları sonuç dict'lerinde tutarlı mı? | GEÇTI (`linear_trend` ×2, `linear_intercept` ×2, `linear_slope` ×2, `err_linear` ×6) |
| `OUT_DIR` guard mevcut mu? | GEÇTI |
| Safety assertion mevcut mu? | GEÇTI |
| TEST_CYCLE=2022 → `outputs/step2/` | GEÇTI (statik simülasyon) |
| TEST_CYCLE=2025 → `outputs/step2_2025/` | GEÇTI (statik simülasyon) |
| TEST_CYCLE=diğer → legacy'ye yazmaz | GEÇTI (statik simülasyon) |
| 2022 çıktı dosyaları değişmedi (SHA-256) | GEÇTI |
| `outputs/step2_2025/` oluşturulmadı | GEÇTI |
| Takip edilen diğer dosyalar değişmedi | GEÇTI |
| `_linear_trend_pred` LUX örneği: slope=−0.5796, pred=480.15 | GEÇTI |
| `_linear_trend_pred` tek nokta → nan | GEÇTI |

---

## 8. Gerçekleştirilemeyen Kontroller (Değerlendirme Çalıştırılmadığı İçin)

| Kontrol | Neden gerçekleştirilemedi? |
|---------|--------------------------|
| Runtime davranışı: 66 ülkenin tamamı için linear_trend hesaplanıyor mu? | Değerlendirme çalıştırılmadı |
| TimesFM yükleniyor ve `predict_batch` çalışıyor mu? | Değerlendirme çalıştırılmadı |
| `outputs/step2_2025/` altında doğru dosyalar oluşuyor mu? | Değerlendirme çalıştırılmadı |
| 3 yöntemin MAE'leri sayısal olarak doğru mu? | Değerlendirme çalıştırılmadı |
| Ortak ülke seti: 3 yöntem için n değerleri eşleşiyor mu? | Değerlendirme çalıştırılmadı |

---

## 9. Kalan Sınırlamalar

1. **LUX protokol tutarsızlığı:** `phase3_3_final_evaluation_protocol.md` Bölüm 5.1, LUX için 2022 verisi olduğunu belirtir, ancak `unified_panel.csv`'de LUX'un 2022 döngüsü yoktur. Persistence ve linear trend için LUX'un son bağlam yılı 2018 olacaktır. Değerlendirme raporunda belirtilmelidir.

2. **Resmi ortalama doğrulaması eksik:** OECD Results Volume I'e erişim sağlanamadı (HTTP 403). Tahminler SE doğrulamasından geçmiştir ancak ülke ortalamaları resmi yayınla karşılaştırılamamıştır.

3. **Script provenance belirsiz:** `build_country_estimates.py` path hatası düzeltilmedi. Mevcut tahminler SE doğrulamasına dayalı olarak kullanılabilir durumdadır.

4. **İşaretli hatalar script tarafından kaydedilmiyor:** `err_*` sütunları mutlak değerdir. Protokol uyumu için işaretli hatalar değerlendirme sonrasında `prediction − actual` ile hesaplanmalıdır.

---

## 10. PISA 2025 Değerlendirmesini Çalıştırmak İçin Yapılması Gerekenler

Aşağıdaki komut, yetkilendirme verildikten sonra çalıştırılabilir:

```bash
# Önce: scripts/step2_timesfm_zeroshot.py içinde TEST_CYCLE = 2025 olarak ayarla
# Sonra:
python3 scripts/step2_timesfm_zeroshot.py
```

Çıktılar şuraya yazılır:
```
outputs/step2_2025/zeroshot_results.csv
outputs/step2_2025/metrics_summary.csv
outputs/step2_2025/zeroshot_scatter.png
```

`outputs/step2/` altındaki 2022 sonuçları dokunulmaz kalır.

---

**Script değişiklikleri uygulandı; PISA 2025 değerlendirmesi ÇALIŞTIRILMADI.**
