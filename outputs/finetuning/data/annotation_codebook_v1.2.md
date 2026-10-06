# ILSA Annotation Codebook v1.2

**Version:** 1.2  
**Date:** 2026-10-06  
**Supersedes:** v1.1  
**Annotator:** Merve Dede  
**Branch:** experiment/ilsa-domain-finetuning

---

## Changelog v1.1 → v1.2

This version results from a systematic construct audit of 111 `UNMATCHED` and `UNMATCHED_TEACHER_ICT` evidence rows in gold_standard_split_v4.csv.

### Constructs ADDED

| Construct | Rows | Studies | Programs | Justification |
|-----------|-----:|--------:|----------|---------------|
| `SCHOOL_ICT_INFRASTRUCTURE` | 10 | 1 | ICILS | School-level ICT infrastructure/availability — distinct from teacher-level ICT use |
| `AGE_GROUP` | 2 | 2 | PIAAC | Age as primary labor market predictor (age-earnings profile); multiple sources |
| `TEACHER_EXPERIENCE` | 2 | 2 | TIMSS | Years of teaching experience as predictor of student outcomes |
| `PRIVATE_TUTORING` | 2 | 2 | TIMSS | Private supplementary lessons as treatment variable in causal inference studies |

### Constructs EXPLICITLY NOT ADDED

| Candidate | Reason NOT Added |
|-----------|-----------------|
| `TEACHER_ICT_TRAINING` | Single study (ICILS adapted, Turkish sample). Insufficient independent evidence. Retained as `UNMATCHED_TEACHER_ICT` for future corpus expansion. |
| `TEACHER_ICT_PROFICIENCY` | Single study. `ICT use effectiveness` is ambiguous (proficiency? attitude? quality?). Not defensible. |
| `ICT_ATTITUDE_TEACHER` | 1–3 rows, single study. Too narrow. |
| `TEACHER_ICT_COLLABORATION` | 1 row. No codebook basis. |
| `HOME_LANGUAGE` | 1 PIRLS study. Use `IMMIGRANT_STATUS` when language spoken at home proxies immigrant background. |
| `PARENTAL_SUPPORT` | 1 PISA study. `Parents' emotional support` not yet operationalized in ILSA canonical vocabulary. |
| `ACTIVE_LEARNING` | 1 TIMSS study. `Active Learning index` is a composite; insufficient evidence for a stable construct. |
| `EDUCATIONAL_MISMATCH` | 1 PIAAC study. Potentially valuable but needs cross-cycle/multi-country support. |
| `NATIONAL_EXAM_SYSTEM` | 1 PIAAC study. System-level policy variable — not a student-level predictor construct. |
| `TEACHER_WORKLOAD` | 1 TIMSS study. Insufficient evidence. |
| `SCIENCE_VALUE_ATTITUDE` | 1 TIMSS study. Subsumed under `MOTIVATION` or future `SCIENCE_ATTITUDE` construct. |

### Status Field Added

A new `construct_status` field is introduced (NOT part of the canonical construct vocabulary):

| Value | Meaning |
|-------|---------|
| `FORECASTABLE` | This evidence row represents a construct that can, in principle, be mapped to a country-level temporal predictor in the forecasting feature space. |
| `NOT_FORECASTABLE` | This evidence row is conceptually or operationally unsuitable for country-level temporal forecasting. The row is retained for provenance but excluded from the forecastable canonical predictor set. |

**Important:** `NOT_FORECASTABLE` is a **status flag**, NOT a canonical construct. Do not add it to the construct vocabulary.

### Existing Misclassifications Corrected

| Predictor | Old Label | Corrected Label |
|-----------|-----------|-----------------|
| `home device` (PIRLS) | `UNMATCHED` | `ICT_HOME_ACCESS` |
| `student confidence index` | `UNMATCHED` | `SELF_CONCEPT` |
| `Students confident in reading` | `UNMATCHED` | `SELF_CONCEPT` |
| `Confidence in science` | `UNMATCHED` | `SELF_CONCEPT` |
| `student discipline` (TIMSS) | `UNMATCHED` | `DISCIPLINARY_CLIMATE` |
| `educational attainment` | `UNMATCHED` | `EDUCATIONAL_ATTAINMENT` |
| `Firm training participation` (PIAAC) | `UNMATCHED` | `JOB_TRAINING` |
| `parental education` | `UNMATCHED` | `PARENTAL_EDUCATION` |
| `Use of ICT in class` (TALIS) | `UNMATCHED_TEACHER_ICT` | `TEACHER_ICT_USE` |

---

## Construct Definitions — v1.2 Additions

---

### `SCHOOL_ICT_INFRASTRUCTURE`

**Definition:** School-level availability, quality, or provision of ICT equipment and digital resources, as a contextual predictor of student or teacher outcomes.

**Canonical source variables:**
- ICILS: `ITISSCL` (ICT availability scale at school level), `ITISSCH` (school emphasis on ICT use), school ICT level composites

**Inclusion criteria:**
- The predictor represents school-level ICT availability or infrastructure (not individual teacher behavior or attitude)
- The analysis operates at or above the school level (or uses school-level ICT as a contextual predictor)
- The construct is explicitly framed as school-level resource provision

**Exclusion criteria:**
- Individual teacher's ICT behavior → `TEACHER_ICT_USE`
- Teacher attitude toward ICT → `UNMATCHED_TEACHER_ICT` (until further evidence)
- Teacher ICT training participation → `UNMATCHED_TEACHER_ICT` (until further evidence)
- Student home ICT access → `ICT_HOME_ACCESS`

**Current evidence:**
- 10 rows, 1 study (ICILS 2018 adapted), 1 program
- **Provenance note:** Evidence is concentrated in a single study. The construct is theoretically justified within ICILS but empirical support is currently limited to one source.

---

### `AGE_GROUP`

**Definition:** Age or age group as a primary predictor of labor market outcomes, cognitive ability, or educational achievement, where age represents a substantive forecastable dimension (not merely a demographic control).

**Mapping rule:**
- `age` as a primary predictor of earnings, productivity, or cognitive outcomes in PIAAC → `AGE_GROUP`
- `age` as a demographic control in a multivariate model with other primary predictors → `NOT_FORECASTABLE`
- `Student age` in PIRLS/TIMSS (child development control) → `NOT_FORECASTABLE`
- `age fixed effects` → `NOT_FORECASTABLE`

**Canonical source variables:**
- PIAAC: respondent age / birth year (used as substantive predictor in age-earnings profiles or cognitive aging studies)

**Current evidence:**
- 2 rows, 2 PIAAC studies, 2 cycles (2012, Cycle 2)

---

### `TEACHER_EXPERIENCE`

**Definition:** Years of teaching experience or professional tenure as a predictor of student achievement or teacher outcomes.

**Inclusion criteria:**
- Predictor is explicitly labeled as teaching years, years of teaching experience, or equivalent
- Used as a substantive predictor (not as a control among many without specific theoretical motivation)

**Exclusion criteria:**
- Generic "teacher characteristics" composite → not mappable
- "PD" alone (professional development) → insufficient; keep `UNMATCHED` pending further disambiguation
- Work experience in non-teaching contexts → `WORK_EXPERIENCE` (not yet in codebook)
- Subject specialization → not mapped

**Current evidence:**
- 2 rows, 2 TIMSS studies (TIMSS 2015, TIMSS 2019)

---

### `PRIVATE_TUTORING`

**Definition:** Private supplementary instruction or private tutoring as a predictor of student academic achievement, typically operationalized as participation in private lessons outside of school.

**Inclusion criteria:**
- Predictor is explicitly labeled as private lessons, private tutoring, or shadow education
- Used as a treatment variable or primary predictor, not as a background control

**Exclusion criteria:**
- Parental homework help → `UNMATCHED` or future construct
- After-school programs (school-organized) → not mapped
- Study time at home → `HOMEWORK_TIME`

**Current evidence:**
- 2 rows, 2 TIMSS studies (both TIMSS 2015, causal inference designs)

---

## NOT_FORECASTABLE Operationalization

The following predictor types should receive `construct_status = NOT_FORECASTABLE`:

| Type | Examples |
|------|---------|
| Fixed effects (country, student, year) | `country FE`, `student FE`, `graduation year FE` |
| Statistical interaction terms | `interaction terms`, `interactions` |
| Aggregate macro-economic indicators | `R&D expenditure`, `STEM skill ratio`, `firm size` |
| Industry/sector dummies | `industry` |
| Composite control bundles | `12 student/school covariates`, `and school background variables (400 features)` |
| Process data signatures | `log signature (PLS)`, `action signature (AS)`, `time signature (TS)` |
| Outcome-derived labels used as predictors | `Reading achievement too low` |
| Demographic controls without forecastable construct value | `Student age` (child age in primary school), `migration` (as incidental control) |
| Country-specific trend terms | `country-specific trends` |

**Decision rule:** A variable is `NOT_FORECASTABLE` only when it is **conceptually or operationally unsuitable** for country-level temporal forecasting — not merely because it is difficult to map. If a variable has a forecastable construct but the mapping is not yet resolved, it stays `UNMATCHED + FORECASTABLE`.

---

## Unresolved Evidence — Preserved for Future Annotation

The following predictor types remain `UNMATCHED + FORECASTABLE` pending future corpus expansion or codebook extension:

| Predictor type | Program | Example | Future candidate |
|----------------|---------|---------|-----------------|
| Parental emotional/social support | PISA | `Parents' emotional support` | `PARENTAL_SUPPORT` |
| Reading attitude/motivation | PIRLS | `Reading is boring attitude` | `MOTIVATION` (expand) |
| Language spoken at home | PIRLS | `language at home` | `HOME_LANGUAGE` or `IMMIGRANT_STATUS` |
| PIAAC labor market mismatches | PIAAC | `undereducation`, `mismatch` | `EDUCATIONAL_MISMATCH` |
| Science valuation | TIMSS | `Valuing science` | `MOTIVATION` subtype |
| Teacher professional development | TIMSS | `PD` | `TEACHER_PD` (pending) |
| Active learning pedagogy | TIMSS | `Active Learning index` | pending |

---

## Retained UNMATCHED_TEACHER_ICT

The following evidence types remain under `UNMATCHED_TEACHER_ICT` by explicit decision:

| Predictor | Rows | Reason |
|-----------|-----:|--------|
| `Attending ICT–CT training` | 7–10 | Single study; `TEACHER_ICT_TRAINING` not yet justified |
| `ICT use effectiveness` | 9 | Ambiguous (proficiency vs. quality vs. attitude); single study |
| `Teachers' perceptions/views regarding ICT` | 3 | Too narrow; single study |
| `Teachers' collaboration for ICT use` | 1 | Insufficient evidence |

---

## Inheritance from v1.1

All rules, construct definitions, and exclusion criteria from v1.1 remain in force. See `annotation_codebook.md` for the base version.

Key rules carried forward:
- `ICT_HOME_ACCESS` does NOT apply to teacher-level ICT in TALIS
- `direction=0` in evidence matrix ≠ confirmed null result
- OECD country notes → excluded from gold
- Measurement invariance studies → excluded or descriptive
- `rubin_rules`/`average_pv` → `all_pvs` (HIGH confidence, no gold barrier)

---

*Codebook v1.2 — Phase 3C construct audit — 2026-10-06*
