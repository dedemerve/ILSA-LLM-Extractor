#!/usr/bin/env python3
"""
Stage 5 — Modül 1: Makale Etki Büyüklüğü Çıkarımı

Her makale JSON'undan LLM ile:
  - predictor → effect_size (standardized β, r, R², odds ratio)
  - predictor → direction (positive/negative/mixed)
  - quality score (metodoloji kalitesi 1-5)

Çıktı: outputs/stage5/effect_sizes.csv
  Sütunlar: paper_id, title, year, program, domain, predictor_canonical,
            effect_type, effect_value, direction, n_countries, n_students,
            pv_correct, weight_correct, quality_score

Ardından compute_predictor_weights_v2.py bu dosyayı kullanır.
"""
from __future__ import annotations

import json
import logging
import os
import pathlib
import re
import time

import pandas as pd

try:
    from openai import OpenAI as _OpenAI
    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False

try:
    import anthropic as _anthropic
    _ANTHROPIC_AVAILABLE = True
except ImportError:
    _ANTHROPIC_AVAILABLE = False

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
OUTPUTS_DIR  = PROJECT_ROOT / "outputs"
OUT_DIR      = PROJECT_ROOT / "outputs" / "stage5"
OUT_CSV      = OUT_DIR / "effect_sizes.csv"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Tüm JSON kaynak dizinleri
JSON_SOURCES = [
    OUTPUTS_DIR / "ilsa_survey_articles" / "json",  # survey makaleleri (132)
    OUTPUTS_DIR / "Scopus",                          # Scopus (423)
    OUTPUTS_DIR / "Web of Science",                  # WoS (302)
]

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Canonical predictor taxonomy — ILSA forecasting'e uygun değişkenler
# ---------------------------------------------------------------------------
CANONICAL_PREDICTORS = {
    # SES ailesi
    "SES_COMPOSITE":      ["escs", "ses", "socioeconomic", "socio-economic"],
    "HOME_RESOURCES":     ["homepos", "home resources", "hedres", "books at home", "cultural possessions"],
    "PARENTAL_EDUCATION": ["parental education", "misced", "fisced", "mother education", "father education"],
    # Motivasyon / tutum
    "BELONGING":          ["belonging", "school belonging", "sense of belonging"],
    "MOTIVATION":         ["motivation", "intrinsic motivation", "student motivation"],
    "SELF_EFFICACY":      ["self-efficacy", "self efficacy", "academic self"],
    "ANXIETY":            ["anxiety", "test anxiety", "math anxiety"],
    # Okul / öğretmen
    "TEACHER_QUALITY":    ["teacher quality", "teacher effectiveness", "instructional quality"],
    "SCHOOL_CLIMATE":     ["school climate", "disciplinary climate", "school environment"],
    "SCHOOL_RESOURCES":   ["school resources", "infrastructure", "facilities"],
    # Teknoloji
    "ICT_ACCESS":         ["ict", "computer", "internet access", "digital"],
    # Cinsiyet
    "GENDER":             ["gender", "sex", "male", "female", "girls", "boys"],
    # Öğretim
    "INSTRUCTIONAL_TIME": ["instructional time", "teaching time", "homework"],
    # Çapraz-program/geçmiş başarı
    "PRIOR_ACHIEVEMENT":  ["prior achievement", "previous score", "lag", "past performance"],
    # Ülke düzeyi
    "GDP_EXPENDITURE":    ["gdp", "expenditure", "education spending", "economic"],
}

def canonical_map(predictor_str: str) -> str:
    """Serbest metin predictor → canonical etiket"""
    s = predictor_str.lower()
    for canon, kws in CANONICAL_PREDICTORS.items():
        if any(kw in s for kw in kws):
            return canon
    return "OTHER"


# ---------------------------------------------------------------------------
# LLM çağrısı
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are an expert in educational measurement and psychometrics specializing in ILSA studies (PISA, TIMSS, PIRLS).
You extract structured effect size information from study summaries for a meta-analysis on country-level achievement forecasting."""

EXTRACTION_PROMPT = """Analyze this study's JSON summary and extract ALL predictor-outcome relationships with quantitative effect sizes.

Study JSON:
{json_str}

Return a JSON array. Each element must have:
{{
  "predictor": "exact variable name as used in study",
  "predictor_category": "one of: SES/home_resources/parental_education/belonging/motivation/self_efficacy/anxiety/teacher_quality/school_climate/school_resources/ICT/gender/instructional_time/prior_achievement/GDP_expenditure/other",
  "effect_type": "one of: standardized_beta / r_correlation / R2 / accuracy / F1 / AUC / odds_ratio / other",
  "effect_value": numeric value or null if not reported,
  "direction": "positive / negative / null / mixed",
  "outcome": "the target variable being predicted",
  "program": "PISA/TIMSS/PIRLS/PIAAC/other",
  "domain": "mathematics/reading/science/other",
  "level": "student/school/country/other",
  "n_countries": integer or null,
  "n_students": integer or null,
  "pv_correct": true/false/null (did study handle plausible values correctly?),
  "weight_correct": true/false/null (did study use sampling weights?),
  "quality_flag": "high/medium/low (based on methodological rigor: correct PV handling, weights, replication)"
}}

Rules:
- If no quantitative effect size, set effect_value to null but still include direction
- Only include predictors that predict student achievement outcomes (skip purely descriptive statistics)
- For R²: divide by 100 if reported as percentage (e.g., 91.2% → 0.912)
- If the study has multiple models, include the best-performing model's metrics
- If predictor is a composite index (e.g., ESCS), categorize as SES
- Return ONLY the JSON array, no other text"""


def _slim_json(paper_json: dict) -> str:
    slim = {
        "metadata": paper_json.get("metadata", {}),
        "data": {
            k: paper_json.get("data", {}).get(k)
            for k in ["survey_design", "sample_details", "ml_techniques",
                       "confounders_identified", "main_findings", "outcome_summary",
                       "plausible_values_handling"]
        }
    }
    return json.dumps(slim, ensure_ascii=False, indent=2)[:8000]


def extract_effects_openai(client, paper_json: dict) -> list[dict]:
    """OpenAI gpt-4o ile etki büyüklüğü çıkarımı"""
    json_str = _slim_json(paper_json)
    resp = client.chat.completions.create(
        model="gpt-4o",
        max_tokens=2000,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": EXTRACTION_PROMPT.format(json_str=json_str)},
        ],
        temperature=0,
    )
    text = resp.choices[0].message.content.strip()
    m = re.search(r'\[.*\]', text, re.DOTALL)
    if m:
        return json.loads(m.group())
    return []


def extract_effects_anthropic(client, paper_json: dict) -> list[dict]:
    """Anthropic Claude ile etki büyüklüğü çıkarımı"""
    json_str = _slim_json(paper_json)
    msg = client.messages.create(
        model="claude-opus-5-5",
        max_tokens=2000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": EXTRACTION_PROMPT.format(json_str=json_str)}]
    )
    text = msg.content[0].text.strip()
    m = re.search(r'\[.*\]', text, re.DOTALL)
    if m:
        return json.loads(m.group())
    return []


def parse_paper_id(filename: str) -> tuple[str, str, int]:
    """Dosya adından: paper_id, başlık parçası, yıl"""
    name = filename.replace(".json", "")
    # "1. Lee & Lee. (2025). ..."
    m = re.match(r'^(\d+)\.\s+(.+?)\.\s+\((\d{4})\)', name)
    if m:
        return m.group(1), m.group(2), int(m.group(3))
    return name[:10], name, 0


# ---------------------------------------------------------------------------
# Ana döngü
# ---------------------------------------------------------------------------
def main():
    openai_key    = os.environ.get("OPENAI_API_KEY")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")

    if openai_key and _OPENAI_AVAILABLE:
        client = _OpenAI(api_key=openai_key)
        extract_fn = extract_effects_openai
        log.info("Backend: OpenAI gpt-4o")
    elif anthropic_key and _ANTHROPIC_AVAILABLE:
        client = _anthropic.Anthropic(api_key=anthropic_key)
        extract_fn = extract_effects_anthropic
        log.info("Backend: Anthropic Claude")
    else:
        raise RuntimeError(
            "Ne OPENAI_API_KEY ne ANTHROPIC_API_KEY ayarlanmamış.\n"
            "  export OPENAI_API_KEY='sk-proj-...'\n"
            "  ya da\n"
            "  export ANTHROPIC_API_KEY='sk-ant-...'"
        )

    # Zaten işlenmiş makaleleri atla
    done_ids: set[str] = set()
    if OUT_CSV.exists():
        existing = pd.read_csv(OUT_CSV)
        done_ids = set(existing["paper_id"].astype(str))
        log.info("Mevcut: %d satır, %d makale işlenmiş", len(existing), len(done_ids))

    # Tüm kaynaklardan JSON topla — doi/title bazlı deduplicate
    seen_titles: set[str] = set()
    files: list[pathlib.Path] = []
    for src in JSON_SOURCES:
        if not src.exists():
            continue
        for fpath in sorted(src.rglob("*.json")):
            if "(1)" in fpath.name:
                continue
            try:
                meta = json.loads(fpath.read_text(errors="ignore")).get("metadata", {})
                title_key = (meta.get("doi") or meta.get("title", fpath.stem))[:120]
            except Exception:
                title_key = fpath.stem[:120]
            if title_key in seen_titles:
                continue
            seen_titles.add(title_key)
            files.append(fpath)

    log.info("Toplam benzersiz makale: %d (survey=%d, scopus=%d, wos=%d)",
             len(files),
             sum(1 for f in files if "ilsa_survey_articles" in str(f)),
             sum(1 for f in files if "/Scopus/" in str(f)),
             sum(1 for f in files if "Web of Science" in str(f)),
    )

    SAVE_EVERY = 10  # Her 10 makalede bir kaydet
    pending: list[dict] = []
    total_saved = len(existing) if OUT_CSV.exists() else 0

    def flush(force: bool = False) -> None:
        nonlocal total_saved
        if not pending:
            return
        if not force and len(pending) < SAVE_EVERY:
            return
        new_df = pd.DataFrame(pending)
        if OUT_CSV.exists():
            old_df = pd.read_csv(OUT_CSV)
            combined = pd.concat([old_df, new_df], ignore_index=True)
        else:
            combined = new_df
        combined.to_csv(OUT_CSV, index=False)
        total_saved = len(combined)
        log.info("  [KAYIT] %d makale işlendi, toplam %d satır → %s",
                 len(pending), total_saved, OUT_CSV.name)
        pending.clear()

    for fpath in files:
        pid, title_part, year = parse_paper_id(fpath.stem)
        # Yıl meta'dan al (Scopus/WoS dosyalarında prefix numarası yok)
        if year == 0:
            try:
                meta_y = json.loads(fpath.read_text(errors="ignore")).get("metadata", {})
                year = int(meta_y.get("year", 0) or 0)
                title_part = str(meta_y.get("title", fpath.stem))[:60]
                pid = str(meta_y.get("doi", fpath.stem))[:40] or fpath.stem[:20]
            except Exception:
                pass
        if pid in done_ids:
            log.debug("Atlanıyor (zaten işlenmiş): %s", pid)
            continue

        log.info("İşleniyor [%s] %s (%d)", pid, title_part[:50], year)
        try:
            paper_json = json.loads(fpath.read_text())
            meta = paper_json.get("metadata", {})
            effects = extract_fn(client, paper_json)

            for eff in effects:
                pending.append({
                    "paper_id":           pid,
                    "title":              meta.get("title", title_part),
                    "year":               year,
                    "program":            eff.get("program", ""),
                    "domain":             eff.get("domain", ""),
                    "predictor_raw":      eff.get("predictor", ""),
                    "predictor_canonical": canonical_map(eff.get("predictor", "")),
                    "predictor_category": eff.get("predictor_category", ""),
                    "effect_type":        eff.get("effect_type", ""),
                    "effect_value":       eff.get("effect_value"),
                    "direction":          eff.get("direction", ""),
                    "outcome":            eff.get("outcome", ""),
                    "level":              eff.get("level", ""),
                    "n_countries":        eff.get("n_countries"),
                    "n_students":         eff.get("n_students"),
                    "pv_correct":         eff.get("pv_correct"),
                    "weight_correct":     eff.get("weight_correct"),
                    "quality_flag":       eff.get("quality_flag", ""),
                })
            done_ids.add(pid)
            log.info("  → %d etki büyüklüğü çıkarıldı", len(effects))
            time.sleep(0.5)  # rate limit

        except Exception as e:
            log.error("  HATA [%s]: %s", pid, e)
            continue

        flush()

    flush(force=True)
    log.info("Tamamlandı. Toplam kayıt: %d", total_saved)


if __name__ == "__main__":
    main()
