# Phase 3.3 — Final Static Safety and Comparability Audit

**Tarih:** 2026-10-15
**Önceki denetim raporu:** `phase3_3_script_change_audit.md` (bu dosya onu GEÇERSİZ KILMAZ; tamamlayıcı niteliktedir)
**Değiştirilen dosya:** `scripts/step2_timesfm_zeroshot.py`
**Değerlendirme ÇALIŞTIRILMADI.**

---

## 1. Bu Oturumda Uygulanan Değişiklikler

| Görev | Durum | Uygulanan değişiklik |
|-------|-------|----------------------|
| Görev 1 — Fail-fast üzerine yazma koruması | TAMAMLANDI | `FileExistsError` bloğu `OUT_DIR.mkdir()` sonrasına eklendi |
| Görev 2 — Ortak ülke metrikleri | TAMAMLANDI | Bölüm 4'e örnek sayısı, `df_common` kesişimi, ortak MAE + işaretli hata eklendi |
| Görev 3 — Doğrusal trend uygunluk takibi | TAMAMLANDI | `skipped_countries` listesi Bölüm 1'e eklendi; her ülke kaydına `context_cycles_str` eklendi |
| Görev 4 — Statik doğrulama | TAMAMLANDI | `py_compile` geçti; sütun kontrolleri doğrulandı; 2022 SHA-256 değerleri değişmedi |
| Görev 5 — Bu denetim raporu | TAMAMLANDI | Bu dosya |

---

## 2. Görev 1 — Fail-Fast Üzerine Yazma Koruması

### Uygulanan değişiklik

```python
# Fail-fast: hedef dizinde çıktı dosyaları zaten varsa dur (üzerine yazmayı önle)
_PLANNED_OUTPUTS = [
    OUT_DIR / "zeroshot_results.csv",
    OUT_DIR / "metrics_summary.csv",
    OUT_DIR / "zeroshot_scatter.png",
]
_existing_outputs = [str(p) for p in _PLANNED_OUTPUTS if p.exists()]
if _existing_outputs:
    raise FileExistsError(
        f"Çıktı dosyaları zaten mevcut — üzerine yazmak yerine duruyorum:\n"
        + "\n".join(f"  {p}" for p in _existing_outputs)
        + f"\nÇözüm: mevcut dosyaları taşıyın ya da TEST_CYCLE değerini değiştirin."
    )
```

### Konum

`OUT_DIR.mkdir(parents=True, exist_ok=True)` satırından hemen sonra.

### Kapsam

| Senaryo | Davranış |
|---------|----------|
| `TEST_CYCLE=2022`, `outputs/step2/` zaten CSV içeriyor | `FileExistsError` fırlatılır — 2022 sonuçları korunur |
| `TEST_CYCLE=2025`, `outputs/step2_2025/` boş/yok | Devam eder |
| `TEST_CYCLE=2025`, `outputs/step2_2025/` zaten CSV içeriyor | `FileExistsError` fırlatılır |

### Önceki koruma (mevcut ve değişmedi)

`OUT_DIR` yalnızlık mantığı ve `assert` deyimi (`phase3_3_script_change_audit.md` Bölüm 2) değişmeden kaldı. Bu fail-fast bloğu, o korumanın üzerine eklenen ikinci bir katmandır.

---

## 3. Görev 2 — Ortak Ülke Metrikleri

### Uygulanan değişiklik (Bölüm 4 başlangıcı)

```python
# Yöntem başına örnek sayısı
n_persist_valid = df["persistence"].notna().sum()
n_linear_valid  = df["linear_trend"].notna().sum()
n_tfm_valid     = df["timesfm_zeroshot"].notna().sum()
print(f"  Geçerli tahmin sayısı — Persistence: {n_persist_valid}, "
      f"Linear trend: {n_linear_valid}, TimesFM: {n_tfm_valid}")

# Ortak küme: üç yöntem için de geçerli tahmin
_common_cols = ["persistence", "linear_trend", "timesfm_zeroshot"]
df_common = df.dropna(subset=_common_cols).copy()
if len(df_common) == 0:
    raise RuntimeError(
        "Ortak ülke kümesi BOŞ — üç yöntem için de geçerli tahmin yok. "
        "Adil karşılaştırma yapılamaz; değerlendirme durduruluyor."
    )

# Ortak kümede işaretli hatalar (prediction − actual)
df_common["signed_err_persistence"] = df_common["persistence"]      - df_common["true_score"]
df_common["signed_err_linear"]      = df_common["linear_trend"]     - df_common["true_score"]
df_common["signed_err_timesfm"]     = df_common["timesfm_zeroshot"] - df_common["true_score"]
```

### Protokol uyum kontrolü

| Protokol gereksinimi | Sağlandı mı? |
|----------------------|-------------|
| Her yöntem için uygun ülke sayısı raporlanır | EVET — `n_persist_valid`, `n_linear_valid`, `n_tfm_valid` yazdırılır |
| Karşılaştırma yalnızca ortak kümede yapılır | EVET — `df_common = df.dropna(subset=_common_cols)` |
| Boş ortak küme açıkça reddedilir | EVET — `RuntimeError` fırlatılır |
| `signed_error = prediction − actual` tanımı | EVET — her sütun bu formülle hesaplanır |
| Ortak küme n ve ülke listesi raporlanır | EVET — `len(df_common)` ve `_missing_from_common` yazdırılır |

### `err_*` sütunları hakkında not

`err_persistence`, `err_linear`, `err_timesfm` sütunları MUTLAK hata içerir (`abs(…)`).
Bunlar artık yalnızca ülke düzeyinde inceleme için kullanılır; MAE hesabı `df_common` üzerinde bu sütunlardan yapılabilir.
Ortak küme işaretli hata analizi için `signed_err_*` sütunları kullanılır.

---

## 4. Görev 3 — Doğrusal Trend Uygunluk Takibi

### 4a. Dışlanan ülke takibi (Bölüm 1)

```python
skipped_countries = []  # dışlanan ülkeler: (iso3, sebep)
for country in test_countries:
    ...
    if len(target_row) == 0:
        skipped_countries.append((country, "hedef döngü verisi yok"))
        continue
    if len(context_rows) < MIN_CONTEXT:
        skipped_countries.append((
            country,
            f"yetersiz geçmiş: {len(context_rows)} gözlem (gerekli: {MIN_CONTEXT})"
        ))
        continue
    records.append(...)

if skipped_countries:
    print(f"  Dışlanan ülkeler ({len(skipped_countries)}):")
    for iso3, reason in skipped_countries:
        print(f"    {iso3}: {reason}")
```

Dışlama sebepleri iki kategoriye ayrılır:
1. `"hedef döngü verisi yok"` — ülke TEST_CYCLE'da panel'de yok
2. `"yetersiz geçmiş: N gözlem (gerekli: MIN_CONTEXT)"` — kesin sayı belirtilir

### 4b. Bağlam yılları takibi (`context_cycles_str`)

Her ülke kaydına aşağıdaki sütun eklendi:
```python
"context_cycles_str": ";".join(str(c) for c in r["context_cycles"]),
```

Bu sütun:
- Noktasayım değil, gerçek değerlendirme yıllarını depolar (örn. `"2003;2006;2009;2012;2015;2018"`)
- OLS fit'te hangi yılların kullanıldığını doğrulama amacıyla kullanılabilir
- LUX gibi düzensiz gözlem örüntüsü olan ülkeler için kritiktir

### 4c. Doğrusal trend uygunluk kuralları — statik doğrulama

| Kural | Sağlandı mı? | Kanıt |
|-------|-------------|-------|
| Hedef yıl fit'ten dışlandı | EVET | `context_rows = cdata[cdata["cycle"] < TEST_CYCLE]` — TEST_CYCLE dahil değil |
| Gerçek yıllar kullanıldı (indeks değil) | EVET | `r["context_cycles"]` = `context_rows["cycle"].tolist()` |
| Eksik puanlar sayılmadı | EVET | `sub = sub.dropna(subset=[TARGET_COL])` üst kısımda |
| < 2 nokta → nan | EVET | `_linear_trend_pred` içinde `if len(context_years) < 2: return (nan, nan, nan)` |
| MIN_CONTEXT=3 garantisi | EVET | Records döngüsünde `len(context_rows) < MIN_CONTEXT` kontrolü; 3 nokta ≥ 2 kuralını otomatik karşılar |

---

## 5. Görev 4 — Statik Doğrulama

### 5a. Python sözdizimi

```
python3 -m py_compile scripts/step2_timesfm_zeroshot.py
→ GEÇTI
```

### 5b. Yeni sütun kontrolü

| Sütun | Konumlar |
|-------|---------|
| `context_cycles_str` | Her iki sonuç dalında (USE_TIMESFM=True ve False) |
| `signed_err_persistence` | `df_common` üzerinde Bölüm 4'te |
| `signed_err_linear` | `df_common` üzerinde Bölüm 4'te |
| `signed_err_timesfm` | `df_common` üzerinde Bölüm 4'te |

AST denetimi (Python `ast` modülü ile):
- `context_cycles_str` sözlük anahtarı: **2 adet** (her iki dal)
- `df_common` başvurusu: **22 adet**
- `raise` deyimi: **2 adet** (OUT_DIR assertion + boş ortak küme RuntimeError)
- `signed_err_*` sütun atamaları: **3 adet**

### 5c. PISA 2022 çıktıları değişmedi (SHA-256)

| Dosya | SHA-256 (ilk 12 hane) | Durum |
|-------|----------------------|-------|
| `outputs/step2/zeroshot_results.csv` | `5b189e2b0b8c` | KORUNDU |
| `outputs/step2/metrics_summary.csv` | `cdbffdac89df` | KORUNDU |
| `outputs/step2/zeroshot_scatter.png` | `bacc69ca51af` | KORUNDU |

`outputs/step2_2025/` dizini oluşturulmadı.

### 5d. Değerlendirme çalıştırılmadı

`python3 scripts/step2_timesfm_zeroshot.py` komutu bu oturumda çalıştırılmadı.
`outputs/step2_2025/` altında herhangi bir çıktı dosyası mevcut değil.

---

## 6. Tüm Oturum Boyunca Yapılan Değişikliklerin Özeti

`scripts/step2_timesfm_zeroshot.py` üzerinde bu oturum boyunca uygulanan değişiklikler:

| Sıra | Değişiklik | Yetkilendirme |
|------|-----------|---------------|
| 1 | `OUT_DIR` döngü bağımlı mantığı + `assert` | `phase3_3_script_change_audit.md` |
| 2 | `_linear_trend_pred` fonksiyonu | `phase3_3_script_change_audit.md` |
| 3 | Sonuç dict'lerine `linear_trend`, `linear_intercept`, `linear_slope`, `err_linear` eklendi | `phase3_3_script_change_audit.md` |
| 4 | Fail-fast `FileExistsError` bloğu | Bu oturum — Görev 1 |
| 5 | `skipped_countries` takibi + dışlama sebebi yazdırma | Bu oturum — Görev 3 |
| 6 | Sonuç dict'lerine `context_cycles_str` eklendi | Bu oturum — Görev 3 |
| 7 | Bölüm 4'e örnek sayıları, `df_common` kesişimi, ortak MAE + işaretli hata eklendi | Bu oturum — Görev 2 |

Başka hiçbir üretim scripti, veri dosyası veya denetim raporu değiştirilmedi.

---

## 7. Kalan Sınırlamalar (Değişmedi)

1. **Resmi ortalama doğrulaması eksik:** OECD Results Volume I HTTP 403 nedeniyle erişilemiyor. SE doğrulaması (MAD=0.000026, n=79) güçlü bir dolaylı kanıt sağlamaktadır ancak nokta tahmin doğruluğunu resmi olarak onaylamıyor.

2. **Script provenance belirsiz:** `build_country_estimates.py` path hatası düzeltilmedi (bu oturumda yetkilendirilmedi).

3. **LOCO n_test=75 ≠ değerlendirme n=66:** LOCO MAE değerleri farklı ülke kümesi kullandığından doğrudan karşılaştırılamaz.

4. **LUX 2022 eksikliği:** LUX için persistence 2018 değerini kullanacak. `context_cycles_str` sütunu bunu doğrulanabilir kılıyor.

---

## 8. PISA 2025 Değerlendirmesini Çalıştırmak İçin Yapılması Gerekenler

1. `TEST_CYCLE = 2025` olarak ayarla
2. `python3 scripts/step2_timesfm_zeroshot.py` çalıştır (explicit kullanıcı yetkilendirmesi gerekir)

Çıktılar `outputs/step2_2025/` altına yazılır. Mevcut `outputs/step2/` korunur.
Fail-fast guard, `outputs/step2_2025/` üzerine tekrar çalıştırmayı da engeller.

---

**Statik düzeltmeler tamamlandı; PISA 2025 değerlendirmesi ÇALIŞTIRILMADI.**
