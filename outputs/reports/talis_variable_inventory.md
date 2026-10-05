# TALIS Variable Inventory
**Oluşturulma tarihi:** 2026-10-04  
**Kaynak:** Arşiv incelemesi + pyreadstat metadata (BTG teacher files)

---

## 1. Dosya yapısı ve ülke kapsamı

| Cycle | Prefix | Level | Dosya/ülke | Country col | Weight col |
|-------|--------|-------|-----------|-------------|------------|
| 2008 | BTG / BCG | ISCED 2 teacher / school | 18 ülke | IDCNTRY (string "AUS - Australia") | TCHWGT |
| 2013 | BTG / BCG / ATG / ACG / PTG / PCG | ISCED 1+2+upper-sec, principal | 29 ülke | CNTRY (ISO3) | TCHWGT |
| 2018 | BTG / BCG / ATG / ACG / PTG / PCG | ISCED 1+2+upper-sec, principal | 33 ülke | CNTRY (ISO3) | TCHWGT |
| 2024 | BTG / BCG (INT aggregate only) | ISCED 2 | ~40 ülke bekleniyor; arşivde 1 dosya | CNTRY | — |

> **2008 notu:** IDCNTRY "ISO3 - Country Name" formatında → kod çıkarımı ilk 3 karakter.  
> **2024 notu:** Yalnızca uluslararası aggregate dosyası mevcut; ülke başına ayrı dosya henüz yayınlanmamış.

---

## 2. Cross-cycle composite scale harmonization

| Construct | Tier | 2008 | 2013 | 2018 | 2024 | Cross-cycle kullanılabilir mi? |
|-----------|------|------|------|------|------|-------------------------------|
| Self-Efficacy: Classroom Mgmt | **A** | N/A | `SECLSS` | `T3SECLS` | `T4SECLS` | 2013–2024 ✓ |
| Self-Efficacy: Instruction | **A** | N/A | `SEINSS` | `T3SEINS` | `T4SEINS` | 2013–2024 ✓ |
| Self-Efficacy: Student Engagement | **A** | N/A | `SEENGS` | `T3SEENG` | `T4SEENG` | 2013–2024 ✓ |
| Disciplinary Climate | **A** | N/A | `TCDISCS` | `T3DISC` | `T4CLSDIS` | 2013–2024 (label farklı, construct yakın) |
| Professional Collaboration | **B** | N/A | `TCCOLLS` | `T3COLES` | `T4COLES` | 2013–2024 ✓ |
| Job Satisfaction: Work Environment | **B** | N/A | `TJSENVS` | `T3JSENV` | `T4JSENVT` | 2013–2024 ✓ |
| Job Satisfaction: Profession | **B** | N/A | `TJSPROS` | `T3JSPRO` | `T4JSPROT` | 2013–2024 ✓ |
| Effective Prof. Development | **B** | N/A | `TEFFPROS` | `T3EFFPD` | `T4PDBR`? | 2013–2018 (2024 construct değişti) |
| Constructivist Beliefs | **C** | `TBCONS` | `TCONSBS` | N/A | N/A | Yalnızca 2008–2013 |
| Workplace Well-being | **C** | N/A | N/A | N/A | `T4WELS` | Yalnızca 2024 |
| Teacher Self-Efficacy (overall) | **B** | N/A | `TSELEFFS` | → 3'e ayrıldı | `T4SELF` | 2013 composite, 2018 split |
| Instructional Leadership | **B** | N/A | N/A | N/A | `T4INSTLE` | Yalnızca 2024 |

**Tier tanımları:**  
- **A** — Ülke düzeyinde güvenilir, çok-döngülü, forecasting'e doğrudan aday  
- **B** — Cross-program contextual feature; belirli program/temporal alignment gerektirir  
- **C** — Descriptive; ülke düzeyinde kararlı feature üretilemez veya cycle kapsam eksik

---

## 3. Temporal alignment (TALIS → PISA/TIMSS lag feature)

```
TALIS 2008 (ISCED 2) → TIMSS 2011 G8   [t=2008 → t+3]
TALIS 2013 (ISCED 2) → PISA 2015        [t=2013 → t+2]
TALIS 2013 (ISCED 2) → TIMSS 2015 G8   [t=2013 → t+2]
TALIS 2018 (ISCED 2) → PISA 2022        [t=2018 → t+4]
TALIS 2018 (ISCED 2) → TIMSS 2019 G8   [t=2018 → t+1]
```

> **Leakage kontrolü:** TALIS verisini yalnızca hedef döngüden *önceki* year ile eşleştir.  
> Örnek: TIMSS 2015'i tahmin ederken TALIS 2013 kullanılabilir; TALIS 2018 kullanılamaz.

---

## 4. Aggregation güvenilirliği

| Koşul | Durum |
|-------|-------|
| Ağırlıklı ortalama (`TCHWGT`) | ✓ 2008/2013/2018 BTG dosyalarında mevcut |
| Survey design (BRR replicate weights) | 2013/2018'de `TRWGT1-100` mevcut |
| PV yapısı | TALIS'te PV yok — scale score doğrudan kullanılabilir |
| Alt-ulusal birimler | 2008: `BFL` (Belçika Flamanca), `INT` (aggregate) → filtrele |
| Minimum n threshold | Ülke başına < 30 öğretmenli ülkeler raporlanmıyor (OECD kuralı) |

---

## 5. Önerilen entegrasyon değişkenleri

Pipeline'a eklenecek TALIS kovaryatları (`program="TALIS"` olarak kaydedilecek):

```python
TALIS_VARS = {
    2013: {
        "SECLSS":   "self_efficacy_cls_mgmt",
        "SEINSS":   "self_efficacy_instruction",
        "SEENGS":   "self_efficacy_engagement",
        "TCDISCS":  "disciplinary_climate",
        "TCCOLLS":  "prof_collaboration",
        "TJSENVS":  "job_sat_environment",
        "TJSPROS":  "job_sat_profession",
    },
    2018: {
        "T3SECLS":  "self_efficacy_cls_mgmt",
        "T3SEINS":  "self_efficacy_instruction",
        "T3SEENG":  "self_efficacy_engagement",
        "T3DISC":   "disciplinary_climate",
        "T3COLES":  "prof_collaboration",
        "T3JSENV":  "job_sat_environment",
        "T3JSPRO":  "job_sat_profession",
    },
}
```

> **2008 notu:** Scale değişkenleri eksik (yalnızca `TBCONS`). 2008'den yalnızca contextual/descriptive feature üretilir.

---

## 6. Veri eksiklikleri (DATA GAP)

| Gap | Açıklama |
|-----|----------|
| TALIS 2024 | Yalnızca uluslararası INT dosyası arşivde; ülke başına ayrı release bekleniyor |
| TALIS 2008 self-efficacy | Composite scale yok; sadece madde düzeyinde sorular |
| TALIS 2008 ISO3 format | `IDCNTRY = "AUS - Australia"` → ilk 3 karakter çıkarımı; `BFL`, `MLT`, `MYS` PISA/TIMSS'te yok |
