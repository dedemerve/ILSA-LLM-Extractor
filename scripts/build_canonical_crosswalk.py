#!/usr/bin/env python3
"""
7-ILSA Canonical Predictor Crosswalk

Ham değişken → canonical construct eşleştirmesi.
Her satır benzersiz (program, raw_variable) anahtarına sahiptir.

Sütunlar:
  program, cycle_availability, raw_variable, canonical_construct,
  level, direction, direction_note, weight_type, pv_structure,
  forecast_role, tier, cross_cycle_comparable, notes

Direction:
  + : yüksek değer = daha iyi/daha fazla (achievement yönüyle pozitif korelasyon beklenir)
  - : yüksek değer = daha düşük/daha kötü (ters skala; modele eklemeden önce işaret değiştir veya notla)
  0 : yönsüz / nominal

Çıktı: outputs/stage5/canonical_crosswalk.csv
"""
import pathlib
import pandas as pd

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT_CSV = PROJECT_ROOT / "outputs" / "stage5" / "canonical_crosswalk.csv"

# ── Crosswalk kaydı ───────────────────────────────────────────────────────────
# Her kayıt: (program, cycle_availability, raw_variable, canonical_construct,
#             level, direction, direction_note, weight_type, pv_structure,
#             forecast_role, tier, cross_cycle_comparable, notes)

ROWS = [
    # ── PISA ──────────────────────────────────────────────────────────────────
    ("PISA", "2003/2009/2015/2022/2025", "ESCS",
     "SES_COMPOSITE", "student", "+",
     "Higher=better SES",
     "W_FSTUWT (BRR)", "none",
     "primary_predictor", "A", "yes",
     "PISA'nın kendi SES composite'i; 2022 reference cycle; 2003/2009 TXT parser"),

    ("PISA", "2015/2022/2025", "HOMEPOS",
     "HOME_RESOURCES", "student", "+",
     "Higher=more home educational resources",
     "W_FSTUWT (BRR)", "none",
     "primary_predictor", "A", "yes",
     "ESCS alt bileşeni; bağımsız predictor olarak tutulabilir"),

    ("PISA", "2003/2015/2022/2025", "BELONG",
     "SCHOOL_BELONGING", "student", "+",
     "Higher=stronger sense of belonging",
     "W_FSTUWT (BRR)", "none",
     "primary_predictor", "A", "partial",
     "2009'da kodebook dışı; 2003 mevcut; construct tutarlı"),

    ("PISA", "2022/2025", "ICTAVHOM",
     "ICT_HOME_ACCESS", "student", "+",
     "Higher=more ICT access at home",
     "W_FSTUWT (BRR)", "none",
     "primary_predictor", "B", "partial",
     "Yalnızca 2022/2025; 2015 ve öncesinde farklı format"),

    ("PISA", "2022/2025", "ICTAVSCH",
     "ICT_SCHOOL_ACCESS", "student", "+",
     "Higher=more ICT access at school",
     "W_FSTUWT (BRR)", "none",
     "primary_predictor", "B", "partial",
     "Yalnızca 2022/2025"),

    # ── TIMSS G8 ──────────────────────────────────────────────────────────────
    ("TIMSS", "2003/2007/2011/2015/2019/2023", "BSDGEDUP",
     "PARENTAL_EDUCATION", "home", "-",
     "1=University+, 5=Below primary → INVERSE: yüksek kod = düşük eğitim",
     "TOTWGT (JK2)", "none",
     "primary_predictor", "A", "yes",
     "Tüm TIMSS G8 döngülerinde tutarlı. Modelde negatif korelasyon beklenir."),

    # ── TIMSS G4 ──────────────────────────────────────────────────────────────
    ("TIMSS_G4", "2011/2015/2019/2023", "ASDHEDUP",
     "PARENTAL_EDUCATION", "home", "-",
     "1=University+, 5=Below primary → INVERSE: yüksek kod = düşük eğitim",
     "TOTWGT (JK2)", "none",
     "primary_predictor", "A", "yes",
     "TIMSS G4 2007/2003'te bu construct mevcut değil (ASH yok)."),

    # ── PIRLS ─────────────────────────────────────────────────────────────────
    ("PIRLS", "2011/2016/2021", "ASDHEDUP",
     "PARENTAL_EDUCATION", "home", "-",
     "1=University+, 5=Below primary → INVERSE",
     "TOTWGT (JK2)", "none",
     "primary_predictor", "A", "yes",
     "PIRLS ASH home questionnaire; eşit ağırlık kullanılıyor (TOTWGT yok)"),

    ("PIRLS", "2011/2016/2021", "ASDHELA",
     "HOME_LITERACY_ACTIVITIES", "home", "-",
     "1=Often, 3=Never → INVERSE: yüksek = daha az okuma aktivitesi",
     "TOTWGT (JK2)", "none",
     "primary_predictor", "B", "yes",
     "Ebeveynlerin okuma aktivitesi; 3 kategorili ters skala"),

    # ── ICCS ──────────────────────────────────────────────────────────────────
    ("ICCS", "2009", "NISB",
     "SES_COMPOSITE", "student", "+",
     "Higher=better national index of socioeconomic background",
     "TOTWGT (JK2)", "none",
     "primary_predictor", "A", "partial",
     "2009 IDB uses unstandardized NISB; S_NISB/ISESCS absent in this release"),

    ("ICCS", "2016/2022", "S_NISB",
     "SES_COMPOSITE", "student", "+",
     "Higher=better national SES index",
     "TOTWGT (JK2)", "none",
     "primary_predictor", "A", "yes",
     "Preferred SES composite for ICCS 2016/2022"),

    ("ICCS", "2016/2022", "S_ECOB",
     "SES_COMPOSITE", "student", "+",
     "Higher=better economic/cultural background index",
     "TOTWGT (JK2)", "none",
     "primary_predictor", "B", "partial",
     "Fallback SES proxy when S_NISB missing"),

    ("ICCS", "2009", "HISCED",
     "PARENTAL_EDUCATION", "home", "+",
     "Higher HISCED = higher parental education (ISCED-like)",
     "TOTWGT (JK2)", "none",
     "primary_predictor", "A", "partial",
     "2009 parental education; positive direction (unlike TIMSS BSDGEDUP)"),

    ("ICCS", "2016/2022", "S_HISCED",
     "PARENTAL_EDUCATION", "home", "+",
     "Higher S_HISCED = higher parental education",
     "TOTWGT (JK2)", "none",
     "primary_predictor", "A", "yes",
     "Standardized parental education in ICCS 2016/2022 ISG files"),

    ("ICCS", "2009/2016", "PARED",
     "PARENTAL_EDUCATION", "home", "+",
     "Higher=more parental education",
     "TOTWGT (JK2)", "none",
     "primary_predictor", "B", "partial",
     "Legacy/fallback parental education label in some ICCS releases"),

    ("ICCS", "2009/2016/2022", "PV1CIV/PV*CIV",
     "CIVIC_KNOWLEDGE", "student", "+",
     "Higher=better civic knowledge",
     "TOTWGT (JK2)", "PV civic knowledge",
     "outcome_and_predictor", "A", "yes",
     "Primary ICCS outcome; used as lag score in LOCO/forward"),

    # ── TALIS 2013 ────────────────────────────────────────────────────────────
    ("TALIS", "2013", "SECLSS",
     "TEACHER_SELF_EFFICACY_CLS_MGMT", "teacher", "+",
     "Higher=more confident in classroom management",
     "TCHWGT (BRR100)", "none",
     "contextual_predictor", "A", "yes_2013_2018",
     "2013 karşılığı SECLSS; 2018'de T3SECLS olarak yeniden adlandırıldı. WLE scale ~8-16."),

    ("TALIS", "2013", "SEINSS",
     "TEACHER_SELF_EFFICACY_INSTRUCTION", "teacher", "+",
     "Higher=more confident in instruction",
     "TCHWGT (BRR100)", "none",
     "contextual_predictor", "A", "yes_2013_2018",
     "2018'de T3SEINS"),

    ("TALIS", "2013", "SEENGS",
     "TEACHER_SELF_EFFICACY_ENGAGEMENT", "teacher", "+",
     "Higher=more confident in student engagement",
     "TCHWGT (BRR100)", "none",
     "contextual_predictor", "A", "yes_2013_2018",
     "2018'de T3SEENG"),

    ("TALIS", "2013", "TCDISCS",
     "DISCIPLINARY_CLIMATE", "classroom", "-",
     "NEED FOR DISCIPLINE: yüksek = daha fazla disiplin sorunu → TERS YÖN",
     "TCHWGT (BRR100)", "none",
     "contextual_predictor", "A", "CAUTION_direction_flip",
     "KRİTİK: 2013 TCDISCS 'need for discipline' (yüksek=kötü) vs 2018 T3DISC 'climate' (yüksek=iyi). "
     "Cross-cycle karşılaştırma için 2013'ü negatif yönde dönüştür veya ayrı tutarak raporla."),

    ("TALIS", "2013", "TCCOLLS",
     "PROF_COLLABORATION", "teacher", "+",
     "Higher=more professional collaboration",
     "TCHWGT (BRR100)", "none",
     "contextual_predictor", "B", "partial",
     "2018'de T3COLES; construct yakın fakat tam eşdeğerlik sınırlı"),

    ("TALIS", "2013", "TJSENVS",
     "JOB_SATISFACTION_ENVIRONMENT", "teacher", "+",
     "Higher=more satisfied with work environment",
     "TCHWGT (BRR100)", "none",
     "contextual_predictor", "B", "yes_2013_2018",
     "2018'de T3JSENV"),

    ("TALIS", "2013", "TJSPROS",
     "JOB_SATISFACTION_PROFESSION", "teacher", "+",
     "Higher=more satisfied with teaching as profession",
     "TCHWGT (BRR100)", "none",
     "contextual_predictor", "B", "yes_2013_2018",
     "2018'de T3JSPRO"),

    ("TALIS", "2013", "TEFFPROS",
     "EFFECTIVE_PD", "teacher", "+",
     "Higher=professional development perceived as more effective",
     "TCHWGT (BRR100)", "none",
     "contextual_predictor", "B", "partial",
     "2018'de T3EFFPD; 2024'te construct değişti"),

    # ── TALIS 2018 ────────────────────────────────────────────────────────────
    ("TALIS", "2018", "T3SECLS",
     "TEACHER_SELF_EFFICACY_CLS_MGMT", "teacher", "+",
     "Higher=more confident in classroom management",
     "TCHWGT (BRR100)", "none",
     "contextual_predictor", "A", "yes_2013_2018",
     "Canonical'i SECLSS ile paylaşır"),

    ("TALIS", "2018", "T3SEINS",
     "TEACHER_SELF_EFFICACY_INSTRUCTION", "teacher", "+",
     "Higher=more confident in instruction",
     "TCHWGT (BRR100)", "none",
     "contextual_predictor", "A", "yes_2013_2018", ""),

    ("TALIS", "2018", "T3SEENG",
     "TEACHER_SELF_EFFICACY_ENGAGEMENT", "teacher", "+",
     "Higher=more confident in student engagement",
     "TCHWGT (BRR100)", "none",
     "contextual_predictor", "A", "yes_2013_2018", ""),

    ("TALIS", "2018", "T3DISC",
     "DISCIPLINARY_CLIMATE", "classroom", "+",
     "Higher=better disciplinary climate (POZITIF — 2013 TCDISCS'in TERSİ)",
     "TCHWGT (BRR100)", "none",
     "contextual_predictor", "A", "CAUTION_direction_flip",
     "KRİTİK: 2013 TCDISCS ile zıt yön. Birleştirmeden önce 2013'ü negatif dönüştür."),

    ("TALIS", "2018", "T3COLES",
     "PROF_COLLABORATION", "teacher", "+",
     "Higher=more professional collaboration",
     "TCHWGT (BRR100)", "none",
     "contextual_predictor", "B", "partial", ""),

    ("TALIS", "2018", "T3JSENV",
     "JOB_SATISFACTION_ENVIRONMENT", "teacher", "+",
     "Higher=more satisfied with work environment",
     "TCHWGT (BRR100)", "none",
     "contextual_predictor", "B", "yes_2013_2018", ""),

    ("TALIS", "2018", "T3JSPRO",
     "JOB_SATISFACTION_PROFESSION", "teacher", "+",
     "Higher=more satisfied with teaching as profession",
     "TCHWGT (BRR100)", "none",
     "contextual_predictor", "B", "yes_2013_2018", ""),

    ("TALIS", "2018", "T3EFFPD",
     "EFFECTIVE_PD", "teacher", "+",
     "Higher=professional development perceived as more effective",
     "TCHWGT (BRR100)", "none",
     "contextual_predictor", "B", "partial", ""),

    # ── ICILS ─────────────────────────────────────────────────────────────────
    ("ICILS", "2013/2018/2023", "PV1-5CIL",
     "CIL_SCORE", "student", "+",
     "Higher=better Computer & Information Literacy",
     "TOTWGTS", "PV1-5CIL (5 PVs)",
     "outcome_and_predictor", "A", "yes",
     "Outcome olarak kullanılabilir; cross-program'da PISA/TIMSS outcome'unu tahmin eden predictor"),

    ("ICILS", "2018/2023", "PV1-5CT",
     "CT_SCORE", "student", "+",
     "Higher=better Computational Thinking",
     "TOTWGTS", "PV1-5CT (5 PVs)",
     "outcome_and_predictor", "A", "partial",
     "Yalnızca 2018+; 2013 ile CT karşılaştırması mümkün değil"),

    ("ICILS", "2013/2018/2023", "S_HISCED",
     "PARENTAL_EDUCATION_ISCED", "home", "+",
     "Higher ISCED = higher education (BSDGEDUP'un tersine POZİTİF yön)",
     "TOTWGTS", "none",
     "primary_predictor", "A", "yes",
     "ICILS'te ISCED kodu doğrudan yüksek=iyi — BSDGEDUP ile farklı yön"),

    ("ICILS", "2013/2018/2023", "S_HISEI",
     "PARENTAL_OCCUPATION_ISEI", "home", "+",
     "Higher=higher occupational status",
     "TOTWGTS", "none",
     "primary_predictor", "A", "yes", ""),

    ("ICILS", "2013/2018/2023", "S_HOMLIT",
     "HOME_LITERACY_INDEX", "home", "+",
     "Higher=more books/literacy resources at home",
     "TOTWGTS", "none",
     "primary_predictor", "A", "yes", ""),

    ("ICILS", "2013/2018", "S_NISB",
     "NAT_SES_INDEX", "student", "+",
     "National SES composite; higher=better SES",
     "TOTWGTS", "none",
     "primary_predictor", "A", "partial",
     "2013/2018 only; 2023'te kaldırıldı"),

    ("ICILS", "2013", "S_BASEFF",
     "ICT_SELF_EFF_GENERAL", "student", "+",
     "Higher=more confident in basic ICT use",
     "TOTWGTS", "none",
     "primary_predictor", "A", "CAUTION_rename",
     "2018/2023'te S_GENEFF olarak yeniden adlandırıldı"),

    ("ICILS", "2018/2023", "S_GENEFF",
     "ICT_SELF_EFF_GENERAL", "student", "+",
     "Higher=more confident in general ICT applications",
     "TOTWGTS", "none",
     "primary_predictor", "A", "yes_2018_2023",
     "2013'teki S_BASEFF karşılığı"),

    # ── PIAAC ─────────────────────────────────────────────────────────────────
    ("PIAAC", "Cycle1 (~2012)", "PVLIT1-10",
     "ADULT_LITERACY", "adult_population", "+",
     "Higher=better literacy skills",
     "SPFWT0 (BRR80)", "PV1-10 (10 PVs)",
     "contextual_predictor", "A", "yes_C1_C2",
     "Ülke ortalaması; öğrenci başarısı için human capital proxy"),

    ("PIAAC", "Cycle1 (~2012)", "PVNUM1-10",
     "ADULT_NUMERACY", "adult_population", "+",
     "Higher=better numeracy skills",
     "SPFWT0 (BRR80)", "PV1-10 (10 PVs)",
     "contextual_predictor", "A", "yes_C1_C2", ""),

    ("PIAAC", "Cycle1 (~2012)", "PVPSL1-10",
     "PS_ICT_SCORE", "adult_population", "+",
     "Problem Solving in ICT environments; higher=better",
     "SPFWT0 (BRR80)", "PV1-10 (10 PVs)",
     "contextual_predictor", "B", "no",
     "Cycle 1 only; discontinued in Cycle 2"),

    ("PIAAC", "Cycle2 (~2022)", "PVAPS1-10",
     "ADAPTIVE_PROBLEM_SOLVING", "adult_population", "+",
     "Higher=better adaptive problem solving",
     "SPFWT0 (BRR80)", "PV1-10 (10 PVs)",
     "contextual_predictor", "B", "no",
     "Cycle 2 only; new construct — PSL ile karşılaştırılamaz"),

    ("PIAAC", "Cycle1+2", "YRSQUAL",
     "YEARS_EDUCATION", "adult_population", "+",
     "Higher=more years of formal education",
     "SPFWT0 (BRR80)", "none",
     "contextual_predictor", "A", "yes_C1_C2", ""),
]

COLUMNS = [
    "program", "cycle_availability", "raw_variable", "canonical_construct",
    "level", "direction", "direction_note", "weight_type", "pv_structure",
    "forecast_role", "tier", "cross_cycle_comparable",
    "direction_original", "direction_canonical", "transformation",
    "notes",
]


# ── Transformation meta tablosu ──────────────────────────────────────────────
# (program, raw_variable): (direction_original, direction_canonical, transformation)
# transformation seçenekleri:
#   "none"       → değer olduğu gibi kullanılır (positif yönlü)
#   "negate"     → value * -1 (ordinal ters skala; yüksek kod = kötü)
#   "drop_near_zero_variance" → ülkeler arası varyans çok düşük; predictor olarak kullanılamaz
#   "rename_only" → değer korunur; sadece değişken adı değişti (ICILS S_BASEFF→S_GENEFF)

TRANSFORM_META = {
    ("TIMSS",    "BSDGEDUP"):  ("-", "+", "negate"),
    ("TIMSS_G4", "ASDHEDUP"):  ("-", "+", "negate"),
    ("PIRLS",    "ASDHEDUP"):  ("-", "+", "negate"),
    ("PIRLS",    "ASDHELA"):   ("-", "+", "negate"),
    ("TALIS",    "TCDISCS"):   ("-", "+", "negate"),
    # T3DISC (2018): uluslararası kalibrasyon → ülkeler arası varyans ~0 → kullanılamaz
    ("TALIS",    "T3DISC"):    ("+", "+", "drop_near_zero_variance"),
    # ICILS rename: S_BASEFF → S_GENEFF (2018+); değer korunur
    ("ICILS",    "S_BASEFF"):  ("+", "+", "rename_only"),
    # PISA, diğer TALIS, ICILS, PIAAC → dönüşüm yok
}


def main():
    # ROWS'da 13 alan: COLUMNS'un transformation öncesi versiyonu
    base_cols = [c for c in COLUMNS if c not in ("direction_original","direction_canonical","transformation")]
    df = pd.DataFrame(ROWS, columns=base_cols)

    # Transformation meta ekle
    df["direction_original"]  = df.apply(
        lambda r: TRANSFORM_META.get((r["program"], r["raw_variable"]), (r["direction"], r["direction"], "none"))[0], axis=1)
    df["direction_canonical"] = df.apply(
        lambda r: TRANSFORM_META.get((r["program"], r["raw_variable"]), (r["direction"], r["direction"], "none"))[1], axis=1)
    df["transformation"] = df.apply(
        lambda r: TRANSFORM_META.get((r["program"], r["raw_variable"]), (r["direction"], r["direction"], "none"))[2], axis=1)

    # Sütun sırası: notes en sona
    notes_col = df.pop("notes")
    df["notes"] = notes_col

    # Duplicate key kontrolü
    dup = df.duplicated(subset=["program", "raw_variable"])
    if dup.any():
        print("DUPLICATE:", df[dup][["program","raw_variable"]].values.tolist())

    df.to_csv(OUT_CSV, index=False)
    print(f"Kaydedildi: {OUT_CSV}  ({len(df)} satır)")

    # Özet
    print("\n=== Direction özeti ===")
    print(df.groupby(["program","direction"])["raw_variable"].count().to_string())

    print("\n=== Kritik direction uyarıları ===")
    warns = df[df["direction"] == "-"][["program","raw_variable","canonical_construct","direction_note"]]
    print(warns.to_string(index=False))

    print("\n=== Cross-cycle karşılaştırma sorunları ===")
    caution = df[df["cross_cycle_comparable"].str.startswith("CAUTION")][
        ["program","raw_variable","canonical_construct","cross_cycle_comparable","notes"]]
    print(caution.to_string(index=False))


if __name__ == "__main__":
    main()
