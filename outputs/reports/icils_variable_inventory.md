# ICILS Variable Inventory
**Oluşturulma tarihi:** 2026-10-04  
**Kaynak:** Arşiv incelemesi + pyreadstat metadata (BSG student, BCG school files)

---

## 1. Dosya yapısı ve ülke kapsamı

| Cycle | Prefixler | PV | Country col | Weight col | Ülke (standard) |
|-------|-----------|----|-----------  |-----------|-----------------|
| 2013 | BSG / BCG / BTG | PV1–5CIL (5) | CNTRY | TOTWGTS | 18 (14 standard¹) |
| 2018 | BSG / BCG / BTG | PV1–5CIL + PV1–5CT | CNTRY | TOTWGTS | 12 (10 standard²) |
| 2023 | BSG / BCG / BTG | PV1–5CIL + PV1–5CT | CNTRY | TOTWGTS / HOUWGTS | 25 (21 standard³) |

> ¹ **2013 sub-national/special:** ABA (Buenos Aires), CNL (Kanada Newfoundland), COT (Kanada Ontario), HKG (Hong Kong)  
> ² **2018 sub-national:** DNW (Kuzey Ren-Vestfalya), RMO (Moskova)  
> ³ **2023 sub-national:** BFL (Belçika Flamanca), DNW (Kuzey Ren-Vestfalya), MLT (Malta—küçük ülke, dahil edilebilir)

---

## 2. Outcome değişkenleri (PV yapısı)

| Construct | Cycle | Değişkenler | Tier |
|-----------|-------|-------------|------|
| CIL (Computer & Information Literacy) | 2013 / 2018 / 2023 | `PV1CIL`–`PV5CIL` | **A** |
| CT (Computational Thinking) | 2018 / 2023 | `PV1CT`–`PV5CT` | **A** (2018→) |

> CIL ölçeği tüm döngülerde karşılaştırılabilir (linking yapıldı).  
> CT yalnızca 2018'den itibaren; 2013 ile CT karşılaştırması yapılamaz.

---

## 3. Student background scales (BSG, S_ prefix)

| Construct | Tier | 2013 | 2018 | 2023 | Cross-cycle? |
|-----------|------|------|------|------|--------------|
| SES: Highest ISCED (parents) | **A** | `S_HISCED` | `S_HISCED` | `S_HISCED` | ✓ 2013–2023 |
| SES: Highest ISEI (parents) | **A** | `S_HISEI` | `S_HISEI` | `S_HISEI` | ✓ 2013–2023 |
| Home literacy index | **A** | `S_HOMLIT` | `S_HOMLIT` | `S_HOMLIT` | ✓ 2013–2023 |
| National SES index | **A** | `S_NISB` | `S_NISB` | N/A | 2013/2018 only |
| ICT self-efficacy: general | **A** | `S_BASEFF` | `S_GENEFF` | `S_GENEFF` | B/G rename → map |
| ICT self-efficacy: specialist | **A** | `S_ADVEFF` | `S_SPECEFF` | `S_SPECEFF` | A/S rename → map |
| Use of ICT for study | **B** | `S_USESTD` | `S_USESTD` | N/A | 2013/2018 only |
| Use of ICT for social comm. | **B** | `S_USECOM` | `S_USECOM` | N/A | 2013/2018 only |
| ICT at school (tasks) | **B** | `S_TSKLRN` | `S_ICTLRN` | N/A | rename → map |
| Positive perceptions of ICT | **B** | N/A | `S_ICTPOS` | `S_ICTPOSG` | 2018/2023 |
| Negative perceptions of ICT | **B** | N/A | `S_ICTNEG` | `S_ICTNEG` | 2018/2023 |
| Immigration status | **C** | `S_IMMBGR` | `S_IMMBGR` | `S_IMMBGR` | ✓ (contextual) |

---

## 4. School-level variables (BCG, C_ prefix — 2023)

| Construct | Tier | Değişken | Not |
|-----------|------|---------|-----|
| ICT device availability (ratio) | **B** | `C_RATSTD` | Ülke ortalaması aggregable |
| ICT resource availability | **B** | `C_ICTRES` | Principal raporlaması |
| Pedagogical hindrances | **C** | `C_HINPED` | Descriptive |
| Resource hindrances | **C** | `C_HINRES` | Descriptive |

---

## 5. Temporal alignment (ICILS → PISA/TIMSS)

```
ICILS 2013 → PISA 2015   [t+2] — CIL; 14 standard ülke
ICILS 2018 → PISA 2022   [t+4] — CIL + CT; 10 standard ülke
ICILS 2018 → TIMSS 2019  [t+1] — CIL; 10 standard ülke
ICILS 2023 → PISA 2025   [t+2] — CIL + CT; 21 standard ülke
```

> **Leakage notu:** ICILS değerleri yalnızca hedef cycle'dan önce geliyor; PISA 2022 tahmini için ICILS 2018 kullanılabilir.

---

## 6. Cross-program country overlap

| Çift | Örtüşen ülke |
|------|-------------|
| ICILS 2013 ∩ PISA | ~12 |
| ICILS 2018 ∩ PISA | ~10 |
| ICILS 2023 ∩ PISA | ~18 |

---

## 7. Önerilen entegrasyon değişkenleri (forecasting panel)

```python
ICILS_VARS = {
    "CIL":     ("cil_score",        "Computer & Information Literacy (mean PV)"),
    "CT":      ("ct_score",         "Computational Thinking (mean PV, 2018+)"),
    "S_HISCED":("parental_educ",    "Highest ISCED of parents"),
    "S_HISEI": ("parental_ses",     "Highest ISEI of parents"),
    "S_HOMLIT":("home_literacy",    "Home literacy index"),
    "S_NISB":  ("nat_ses_index",    "National SES index (2013/2018)"),
    "S_GENEFF":("ict_self_eff_gen", "ICT self-efficacy general"),
    "S_SPECEFF":("ict_self_eff_sp", "ICT self-efficacy specialist"),
}
# 2013 rename: S_BASEFF → S_GENEFF, S_ADVEFF → S_SPECEFF
```

> PV ortalaması (mean of PV1–PV5) ülke düzeyinde aggregation için kullanılacak.  
> TOTWGTS ağırlığı ile ağırlıklı ortalama; sub-national birimler filtrelenecek.

---

## 8. Veri eksiklikleri (DATA GAP)

| Gap | Açıklama |
|-----|----------|
| 2013'te CT yok | CT testi 2018'de eklendi |
| 2013 ICILS ülke sayısı az | 14 standard ülke — küçük örtüşme |
| ICILS 2020 | Pilot/özel study — tek dosya (MER), raporlama yok |
| `S_USESTD`, `S_USECOM`, `S_TSKLRN` 2023'te yok | Construct değişikliği; crosswalk eksik |
