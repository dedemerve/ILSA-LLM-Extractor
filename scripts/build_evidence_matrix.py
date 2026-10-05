#!/usr/bin/env python3
"""
FAZ 3 — LLM Evidence Matrix

Mevcut JSON extraction'ını genişletilmiş schema ile yeniden işler:

  study × program × cycle × country × dataset × variable
        × construct × outcome × method × effect

Her makale için:
  1. Mevcut JSON'daki main_findings/confounders/top_predictors bilgisini çıkar
  2. Canonical construct crosswalk ile ILSA değişkenine bağla
  3. Effect type'ı sınıflandır (associative / predictive / cross-cycle / cross-country)
  4. Analysis level'i kayıt altına al (student / school / country)

Çıktılar:
  outputs/stage5/evidence_matrix.csv        — tam evidence satırları
  outputs/stage5/evidence_matrix_summary.csv — construct × program özeti
"""
from __future__ import annotations

import json
import logging
import pathlib
import re
from typing import Optional

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT_DIR      = PROJECT_ROOT / "outputs" / "stage5"

JSON_ROOTS = [
    PROJECT_ROOT / "outputs" / "OECD",
    PROJECT_ROOT / "outputs" / "IEA",
    PROJECT_ROOT / "outputs" / "Scopus",
    PROJECT_ROOT / "outputs" / "Web of Science",
    PROJECT_ROOT / "outputs" / "ilsa_survey_articles" / "json",
]

try:
    from scripts.canonical_predictor_map import canonical_map as _canonical_map
except ImportError:
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.canonical_predictor_map import canonical_map as _canonical_map

# ── Canonical construct → ILSA variable crosswalk ────────────────────────────
# Serbest metin predictor'ları → canonical construct eşleme
PREDICTOR_TO_CONSTRUCT: dict[str, str] = {
    # SES / home background
    "socioeconomic": "SES_COMPOSITE",
    "ses": "SES_COMPOSITE",
    "escs": "SES_COMPOSITE",
    "economic, social and cultural status": "SES_COMPOSITE",
    "family ses": "SES_COMPOSITE",
    "socio-economic": "SES_COMPOSITE",
    "economic background": "SES_COMPOSITE",
    "family economic": "SES_COMPOSITE",
    "home socioeconomic": "SES_COMPOSITE",

    # Parental education
    "parental education": "PARENTAL_EDUCATION",
    "parents' education": "PARENTAL_EDUCATION",
    "mother's education": "PARENTAL_EDUCATION",
    "father's education": "PARENTAL_EDUCATION",
    "highest parental education": "PARENTAL_EDUCATION",
    "parental educational attainment": "PARENTAL_EDUCATION",

    # Home resources
    "home educational resources": "HOME_RESOURCES",
    "home resources": "HOME_RESOURCES",
    "homepos": "HOME_RESOURCES",
    "possessions": "HOME_RESOURCES",
    "books at home": "HOME_RESOURCES",
    "cultural possessions": "HOME_RESOURCES",

    # School belonging / climate
    "sense of belonging": "SCHOOL_BELONGING",
    "belonging": "SCHOOL_BELONGING",
    "school belonging": "SCHOOL_BELONGING",
    "feel like i belong": "SCHOOL_BELONGING",
    "school climate": "SCHOOL_BELONGING",
    "disciplinary climate": "DISCIPLINARY_CLIMATE",
    "classroom discipline": "DISCIPLINARY_CLIMATE",
    "classroom management": "DISCIPLINARY_CLIMATE",
    "orderly environment": "DISCIPLINARY_CLIMATE",

    # Home literacy
    "home literacy": "HOME_LITERACY_ACTIVITIES",
    "early literacy": "HOME_LITERACY_ACTIVITIES",
    "reading at home": "HOME_LITERACY_ACTIVITIES",
    "literacy activities": "HOME_LITERACY_ACTIVITIES",
    "home literacy activities": "HOME_LITERACY_ACTIVITIES",

    # ICT
    "ict access": "ICT_HOME_ACCESS",
    "computer access": "ICT_HOME_ACCESS",
    "internet access": "ICT_HOME_ACCESS",
    "digital resources": "ICT_HOME_ACCESS",
    "school ict": "ICT_SCHOOL_ACCESS",
    "ict at school": "ICT_SCHOOL_ACCESS",
    "ict use": "ICT_HOME_ACCESS",

    # Teacher / TALIS
    "teacher self-efficacy": "TEACHER_SELF_EFFICACY_CLS_MGMT",
    "self-efficacy": "TEACHER_SELF_EFFICACY_CLS_MGMT",
    "teacher confidence": "TEACHER_SELF_EFFICACY_INSTRUCTION",
    "professional collaboration": "PROF_COLLABORATION",
    "teacher collaboration": "PROF_COLLABORATION",
    "job satisfaction": "JOB_SATISFACTION_PROFESSION",
    "professional development": "EFFECTIVE_PD",

    # Motivation / attitudes
    "motivation": "MOTIVATION",
    "intrinsic motivation": "MOTIVATION",
    "academic motivation": "MOTIVATION",
    "interest in science": "SCIENCE_INTEREST",
    "science confidence": "SCIENCE_SELF_EFFICACY",
    "math confidence": "MATH_SELF_EFFICACY",
    "self-concept": "SELF_CONCEPT",
    "academic self-concept": "SELF_CONCEPT",
    "value of education": "MOTIVATION",

    # Prior achievement
    "prior achievement": "PRIOR_ACHIEVEMENT",
    "previous score": "PRIOR_ACHIEVEMENT",
    "baseline score": "PRIOR_ACHIEVEMENT",
    "prior performance": "PRIOR_ACHIEVEMENT",
    "prior score": "PRIOR_ACHIEVEMENT",

    # Instructional / curriculum
    "instructional time": "INSTRUCTIONAL_TIME",
    "teaching time": "INSTRUCTIONAL_TIME",
    "curriculum": "CURRICULUM_TYPE",
    "teaching quality": "TEACHING_QUALITY",
    "quality of instruction": "TEACHING_QUALITY",
    "instructional quality": "TEACHING_QUALITY",

    # Gender / migration
    "gender": "GENDER",
    "sex": "GENDER",
    "female": "GENDER",
    "immigrant": "IMMIGRANT_STATUS",
    "immigration": "IMMIGRANT_STATUS",
    "language background": "LANGUAGE_BACKGROUND",
    "language at home": "LANGUAGE_BACKGROUND",
}

PROGRAM_KEYWORDS: dict[str, list[str]] = {
    "PISA":     ["pisa", "programme for international student assessment"],
    "TIMSS":    ["timss", "trends in international mathematics and science"],
    "TIMSS_G4": ["timss grade 4", "timss g4", "timss fourth", "timss 4th"],
    "PIRLS":    ["pirls", "progress in international reading literacy"],
    "TALIS":    ["talis", "teaching and learning international survey"],
    "ICILS":    ["icils", "international computer and information literacy"],
    "ICCS":     ["iccs", "international civic and citizenship education"],
    "PIAAC":    ["piaac", "programme for the international assessment of adult competencies"],
}

OUTCOME_KEYWORDS: dict[str, str] = {
    "mathematics": "mathematics_achievement",
    "math": "mathematics_achievement",
    "science": "science_achievement",
    "reading": "reading_achievement",
    "literacy": "literacy_achievement",
    "numeracy": "numeracy_achievement",
    "civic knowledge": "civic_knowledge",
    "cil": "computer_information_literacy",
    "computer literacy": "computer_information_literacy",
    "adult skills": "adult_skills",
}

EFFECT_SIGNALS: dict[str, str] = {
    "predict": "predictive",
    "forecast": "predictive",
    "regression": "associative",
    "correlation": "associative",
    "association": "associative",
    "related to": "associative",
    "associated with": "associative",
    "cross-cycle": "cross_cycle",
    "longitudinal": "cross_cycle",
    "temporal": "cross_cycle",
    "cross-country": "cross_country",
    "international": "cross_country",
}


def _normalize(text: str) -> str:
    return text.lower().strip() if text else ""


def _match_construct(predictor_text: str, variable_code: str = "", category: str = "") -> str:
    canon = _canonical_map(predictor_text, variable_code=variable_code or None, category=category or None)
    if canon == "OTHER":
        pt = _normalize(predictor_text)
        for kw, construct in PREDICTOR_TO_CONSTRUCT.items():
            if kw in pt:
                return construct
        return "UNMATCHED"
    return canon


def _detect_programs(text: str) -> list[str]:
    tl = _normalize(text)
    found = []
    # G4 önce kontrol edilmeli (TIMSS'in alt kümesi)
    for prog in ["TIMSS_G4", "PISA", "TIMSS", "PIRLS", "TALIS", "ICILS", "ICCS", "PIAAC"]:
        for kw in PROGRAM_KEYWORDS[prog]:
            if kw in tl:
                found.append(prog)
                break
    return list(dict.fromkeys(found))  # deduplicate, order preserved


def _detect_outcome(text: str) -> str:
    tl = _normalize(text)
    for kw, outcome in OUTCOME_KEYWORDS.items():
        if kw in tl:
            return outcome
    return "unspecified"


def _detect_effect_type(methods_text: str, design_text: str) -> str:
    combined = _normalize(f"{methods_text} {design_text}")
    for kw, etype in EFFECT_SIGNALS.items():
        if kw in combined:
            return etype
    return "associative"


def _detect_analysis_level(text: str) -> str:
    tl = _normalize(text)
    if "country-level" in tl or "country level" in tl or "national" in tl:
        return "country"
    if "school-level" in tl or "school level" in tl:
        return "school"
    if "teacher" in tl:
        return "teacher"
    return "student"


def _extract_cycles(text: str) -> list[int]:
    """Metindeki 4-haneli ILSA cycle yıllarını çıkar."""
    years = [int(y) for y in re.findall(r"\b(199\d|200\d|201\d|202\d)\b", text)]
    valid  = [y for y in years if y in {
        1995,1999,2000,2001,2003,2006,2007,2008,2009,2011,2012,2013,2015,
        2016,2018,2019,2021,2022,2023,2024,2025
    }]
    return sorted(set(valid))


def _extract_countries(sample_details: dict) -> list[str]:
    countries = sample_details.get("countries", [])
    if isinstance(countries, list):
        return [c.get("country_code", "") if isinstance(c, dict) else str(c)
                for c in countries]
    return []


def _extract_direction(text: str) -> Optional[str]:
    tl = _normalize(text)
    neg_words = ["negative", "negatively", "inverse", "decrease", "lower", "worse"]
    pos_words = ["positive", "positively", "higher", "increase", "better", "improve"]
    n = sum(1 for w in neg_words if w in tl)
    p = sum(1 for w in pos_words if w in tl)
    if p > n:
        return "+"
    if n > p:
        return "-"
    return None  # ambiguous


def process_article(json_path: pathlib.Path) -> list[dict]:
    """Tek bir article JSON'ından evidence satırları üretir."""
    try:
        d = json.load(open(json_path, encoding="utf-8"))
    except Exception as e:
        log.warning("JSON okunamadı: %s — %s", json_path.name, e)
        return []

    meta = d.get("metadata", {})
    data = d.get("data", {})

    study_id  = meta.get("file_name", json_path.stem)
    title     = str(meta.get("title") or "")
    year      = meta.get("year")
    doi       = meta.get("doi", "")
    venue     = meta.get("venue", "")

    # Program ve cycle tespiti (title + main_findings birleşimi)
    full_text_for_detection = str(title or "")
    main_findings = data.get("main_findings", [])
    if isinstance(main_findings, list):
        for mf in main_findings:
            if isinstance(mf, dict):
                full_text_for_detection += " " + str(mf.get("dataset_used") or "")
                full_text_for_detection += " " + str(mf.get("standardized_conclusion") or "")

    programs = _detect_programs(full_text_for_detection)
    cycles   = _extract_cycles(full_text_for_detection)
    countries = _extract_countries(data.get("sample_details", {}))

    methods_text = str(data.get("ml_techniques", ""))
    design_text  = str(data.get("research_design_type", ""))
    effect_type  = _detect_effect_type(methods_text, design_text)

    rows = []

    # ── main_findings'ten predictor çıkar ────────────────────────────────────
    if isinstance(main_findings, list):
        for mf in main_findings:
            if not isinstance(mf, dict):
                continue

            dataset_used    = mf.get("dataset_used", "")
            outcome_raw     = mf.get("target_variable", "")
            outcome_type    = _detect_outcome(f"{outcome_raw} {dataset_used}")
            mf_programs     = _detect_programs(dataset_used) or programs
            mf_cycles       = _extract_cycles(dataset_used) or cycles

            conclusion = mf.get("standardized_conclusion", "")
            analysis_level = _detect_analysis_level(conclusion)

            top_preds = mf.get("top_predictors", [])
            if isinstance(top_preds, list):
                for rank, pred in enumerate(top_preds, 1):
                    construct = _match_construct(str(pred))
                    direction = _extract_direction(conclusion)
                    rows.append({
                        "study_id":       study_id,
                        "title":          title[:120],
                        "year":           year,
                        "doi":            doi[:80] if doi else "",
                        "venue":          venue[:80] if venue else "",
                        "programs":       "|".join(mf_programs),
                        "cycles":         "|".join(str(c) for c in mf_cycles),
                        "n_countries":    len(countries),
                        "countries_sample": "|".join(countries[:10]),
                        "predictor_raw":  str(pred)[:100],
                        "canonical_construct": construct,
                        "outcome_raw":    outcome_raw[:100],
                        "outcome_type":   outcome_type,
                        "effect_type":    effect_type,
                        "analysis_level": analysis_level,
                        "rank_in_study":  rank,
                        "direction":      direction,
                        "dataset_used":   dataset_used[:120],
                    })

    # ── confounders_identified'dan ek predictor ───────────────────────────────
    confounders = data.get("confounders_identified", [])
    if isinstance(confounders, list):
        for conf in confounders:
            if not isinstance(conf, dict):
                continue
            vname    = conf.get("variable_name", conf.get("variable_code", ""))
            vcode    = conf.get("variable_code", "")
            category = conf.get("category", "")
            construct = _match_construct(vname, variable_code=vcode, category=category)
            # Sadece zaten ana satır olarak eklenmemişleri ekle
            if construct != "UNMATCHED":
                rows.append({
                    "study_id":            study_id,
                    "title":               title[:120],
                    "year":                year,
                    "doi":                 doi[:80] if doi else "",
                    "venue":               venue[:80] if venue else "",
                    "programs":            "|".join(programs),
                    "cycles":              "|".join(str(c) for c in cycles),
                    "n_countries":         len(countries),
                    "countries_sample":    "|".join(countries[:10]),
                    "predictor_raw":       vname[:100],
                    "canonical_construct": construct,
                    "outcome_raw":         "from_confounders",
                    "outcome_type":        "unspecified",
                    "effect_type":         "associative",
                    "analysis_level":      _detect_analysis_level(category),
                    "rank_in_study":       None,
                    "direction":           None,
                    "dataset_used":        "",
                })

    return rows


def _collect_corpus_json() -> list[pathlib.Path]:
    seen_keys: set[str] = set()
    selected: list[pathlib.Path] = []
    for root in JSON_ROOTS:
        if not root.exists():
            continue
        if root.name == "json":
            candidates = sorted(root.glob("*.json"))
        else:
            candidates = sorted(root.rglob("json/*.json"))
        for fpath in candidates:
            if "(1)" in fpath.name:
                continue
            try:
                meta = json.load(open(fpath, encoding="utf-8")).get("metadata", {})
                key = (meta.get("doi") or meta.get("title") or fpath.stem)[:120]
            except Exception:
                key = fpath.stem
            if key in seen_keys:
                continue
            seen_keys.add(key)
            selected.append(fpath)
    return selected


def main():
    selected = _collect_corpus_json()
    log.info("%d benzersiz corpus JSON bulundu (OECD+IEA+Scopus+WoS+survey)", len(selected))

    all_rows = []
    for fpath in selected:
        rows = process_article(fpath)
        all_rows.extend(rows)
        log.info("  %s → %d satır", fpath.name[:60], len(rows))

    df = pd.DataFrame(all_rows)
    if df.empty:
        log.warning("Hiç satır üretilemedi.")
        return

    # Unmatched constructs'ları kayıt için tut ama filtrele
    unmatched = df[df["canonical_construct"] == "UNMATCHED"]
    df_matched = df[df["canonical_construct"] != "UNMATCHED"].copy()

    df.to_csv(OUT_DIR / "evidence_matrix_full.csv", index=False)
    df_matched.to_csv(OUT_DIR / "evidence_matrix.csv", index=False)
    log.info("Kaydedildi: evidence_matrix.csv (%d satır), evidence_matrix_full.csv (%d satır)",
             len(df_matched), len(df))
    if len(unmatched):
        log.info("  Unmatched predictor satırları: %d (evidence_matrix_full'da var)",
                 len(unmatched))

    # ── Özet ─────────────────────────────────────────────────────────────────
    print("\n=== Construct × Program özeti (matched rows) ===")
    summary = (
        df_matched.groupby(["canonical_construct","programs"])
        .agg(n_studies=("study_id","nunique"), n_rows=("study_id","count"),
             years=("year", lambda x: f"{int(x.min())}-{int(x.max())}"))
        .reset_index()
        .sort_values(["n_studies","canonical_construct"], ascending=[False,True])
    )
    summary.to_csv(OUT_DIR / "evidence_matrix_summary.csv", index=False)
    print(summary.to_string(index=False))

    print(f"\n=== Unmatched predictor örnekleri (ilk 20) ===")
    if len(unmatched):
        print(unmatched[["study_id","predictor_raw"]].drop_duplicates()
              .head(20).to_string(index=False))

    print(f"\nToplam: {len(df_matched)} matched evidence satırı")
    print(f"  Unique studies: {df_matched['study_id'].nunique()}")
    print(f"  Unique constructs: {df_matched['canonical_construct'].nunique()}")
    print(f"  Effect types: {df_matched['effect_type'].value_counts().to_dict()}")

    # Construct frequency (w_F için temel)
    print("\n=== Construct görünme frekansı (w_F temeli) ===")
    freq = (df_matched.groupby("canonical_construct")["study_id"]
            .nunique().sort_values(ascending=False).reset_index()
            .rename(columns={"study_id": "n_studies"}))
    freq["w_F_raw"] = freq["n_studies"] / freq["n_studies"].max()
    print(freq.to_string(index=False))
    freq.to_csv(OUT_DIR / "construct_frequency.csv", index=False)


if __name__ == "__main__":
    main()
