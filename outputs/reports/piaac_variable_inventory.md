# PIAAC Variable Inventory
**Oluşturulma tarihi:** 2026-10-04  
**Kaynak:** Arşiv incelemesi + pandas CSV metadata (PRG* country files)

---

## 1. Dosya yapısı

| Cycle | Format | Separator | Ülke sayısı | Country col | Weight col |
|-------|--------|-----------|-------------|-------------|------------|
| Cycle 1 (~2012–2017) | CSV (per country) | comma | 27 (25 unique¹) | `CNTRYID` | `SPFWT0` |
| Cycle 2 (~2022–2023) | CSV (per country) | **semicolon** | 21 | `CNTRYID` | `SPFWT0` |

> ¹ USA iki ayrı dosyada: `prgusap1_2012.csv` + `prgusap1_2017.csv`

---

## 2. Outcome değişkenleri (PV yapısı)

| Construct | Cycle 1 | Cycle 2 | Not |
|-----------|---------|---------|-----|
| Literacy | `PVLIT1`–`PVLIT10` | `PVLIT1`–`PVLIT10` | Her iki cycle karşılaştırılabilir |
| Numeracy | `PVNUM1`–`PVNUM10` | `PVNUM1`–`PVNUM10` | Her iki cycle karşılaştırılabilir |
| Problem Solving (ICT) | `PVPSL1`–`PVPSL10` | N/A (discontinued) | Yalnızca Cycle 1 |
| Adaptive Problem Solving | N/A | `PVAPS1`–`PVAPS10` | Yalnızca Cycle 2 (yeni construct) |

> **Önemli:** PV sayısı 10'dur (PISA/TIMSS'in 5 PV'sinden farklı). Ülke ortalaması için mean(PV1–PV10) kullanılmalı.

---

## 3. Background değişkenleri

| Construct | Cycle 1 değişken | Cycle 2 değişken | Tier |
|-----------|-----------------|-----------------|------|
| Eğitim düzeyi (ISCED) | `B_Q01a` | `B2_Q01` | **A** — aggregable |
| Yıl bazlı eğitim | `YRSQUAL` | `YRSQUAL` | **A** |
| Cinsiyet | `GENDER_R` | `GENDER_R` | **C** |
| İstihdam durumu | `C_D05` | `C2_D05` | **B** |
| Mesleki kategori (ISCO) | `D_Q03` | `D2_Q03` | **B** |
| ICT kullanımı (iş) | `F_Q03a` | (benzer) | **B** |
| Okuma aktivitesi (work) | `F_Q02a` | (benzer) | **B** |
| Sayısal aktivite (work) | `G_Q01a` | (benzer) | **B** |
| Ülke doğumluluğu | `IMGEN` | `IMGEN` | **C** |

---

## 4. Metodolojik kısıtlar

### PIAAC LOCO'ya dahil edilmemeli

PIAAC yalnızca 2 cycle içeriyor (Cycle 1: ~2012, Cycle 2: ~2022). Bu, standart LOCO expanding window yapısına uymuyor:

- Minimum LOCO için en az 3 cycle gerekir (1 train, 1 val, 1 test)
- 2 cycle ile yalnızca tek bir hold-out test fold üretilebilir
- Forecasting için yalnızca `Cycle 1 → Cycle 2` lag mümkün

### Uygun kullanım biçimi

```
PIAAC Cycle 1 → PISA 2015/2018 adult literacy → student achievement cross-level predictor
PIAAC Cycle 1 → TIMSS 2015 G8 → adult numeracy as country-level background predictor
PIAAC Cycle 2 → PISA 2025 → lag feature (t+3)
```

> **Tezde doğru çerçeve:** PIAAC, öğrenci başarısının bağlamsal (adult human capital) tahmin edicisi olarak kullanılabilir; doğrudan LOCO hedefi değil.

---

## 5. Cycle 1 ülkeleri

```
BEL CAN CHL CZE DNK ESP EST FIN FRA GBR GRC IRL JPN KAZ LTU MEX NOR NZL PER POL RUS SGP SVK SWE TUR USA
```
(27 dosya, 25 unique ülke — USA iki wave)

## 6. Cycle 2 ülkeleri

```
AUT CAN CHE CHL CZE DEU FIN FRA GBR HUN IRL JPN LTU NOR NZL POL PRT SGP SVK SWE USA
```
(21 ülke)

---

## 7. Önerilen entegrasyon değişkenleri

```python
PIAAC_VARS = {
    "PVLIT":   ("literacy_score",   "Adult Literacy (mean PV1–PV10)"),
    "PVNUM":   ("numeracy_score",   "Adult Numeracy (mean PV1–PV10)"),
    "PVPSL":   ("ps_ict_score",     "Problem Solving in ICT env. (Cycle 1 only)"),
    "PVAPS":   ("adapt_ps_score",   "Adaptive Problem Solving (Cycle 2 only)"),
    "YRSQUAL": ("years_education",  "Years of formal education"),
}
```

> PV pooling: `mean(PV1, ..., PV10)` per individual, then weighted country mean (SPFWT0).

---

## 8. Veri eksiklikleri (DATA GAP)

| Gap | Açıklama |
|-----|----------|
| Yalnızca 2 cycle | LOCO için yetersiz — contextual predictor olarak kullan |
| PSL discontinued | Cycle 2'de yok; APS farklı construct |
| Cycle 2 semicolon separator | `sep=';'` ile okunmalı (comma varsayımı hata verir) |
| RUS Cycle 1 | OECD yayınından çekildi; veri politika gereği kısıtlı |
| KAZ, PER, MEX | Cycle 2'de yok — kapsam daralması |
