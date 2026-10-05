# Gold Standard Quality Report — Phase 3 Pilot

**Date:** 2026-10-05  
**Status:** PILOT COMPLETE — Full gold annotation PENDING  
**Branch:** experiment/ilsa-domain-finetuning

---

## Terminology

| Label | Definition |
|-------|-----------|
| gold_verified | Source text read and all Tier-1/2 fields confirmed by annotator |
| gold_verified_after_adjudication | Source text read; labels corrected; adjudication recorded |
| ambiguous | Source text read; fields cannot be reliably assigned |
| excluded | Source text confirms this is NOT an ML prediction study |
| annotation_pending | Not yet annotated |

**No records are labelled gold from automatic validation alone.**

---

## Pilot Statistics (18 records annotated)

| Status | Studies | Evidence rows |
|--------|--------:|-------------:|
| gold_verified | 8 | 51 |
| gold_verified_after_adjudication | 4 | 77 |
| excluded | 5 | 5 |
| ambiguous | 1 | 1 |
| **Total annotated** | **18** | **134** |

**Usable gold rows: 128** (gold_verified + gold_verified_after_adjudication)  
**Usable gold studies: 12**

---

## Program Distribution (12 verified studies)

| Program | N studies | % |
|---------|----------:|--:|
| PISA | 3 | 25.0% |
| TALIS | 3 | 25.0% |
| TIMSS | 1 | 8.3% |
| PIRLS | 2 | 16.7% |
| ICILS | 2 | 16.7% |
| ICCS | 1 | 8.3% |
| PIAAC | 0 | 0% |

PISA is at 25% (vs 54% in silver candidates) — program balance successfully maintained in pilot design.  
PIAAC is at 0% — all 3 pilot PIAAC records were excluded as descriptive reports.

---

## Key Pilot Findings

### 1. PIAAC inflation problem
All 3 PIAAC silver candidates examined in pilot were OECD country notes or working papers — descriptive reports, not ML prediction studies. This inflates the 49-count PIAAC silver pool.

**Implication:** The ~49 PIAAC silver candidates need to be filtered to journal articles only before annotation. Expected viable PIAAC gold studies: 5–10.

### 2. TALIS ICT canonical error
Teacher-level ICT variables in TALIS studies are systematically mapped to `ICT_HOME_ACCESS` (student home access) — a cross-domain error. All TALIS ICT-related evidence rows require canonical correction: teacher ICT → UNMATCHED_TEACHER_ICT.

**Impact:** ~60 evidence rows across TALIS silver candidates require canonical correction.

### 3. Direction encoding ambiguity
`direction=0` in the evidence matrix means either "null result" (0) or "not reported" (NaN misencoded). Without source text confirmation, direction=0 cannot be trusted. In 5 of 18 pilot records, direction=0 was found to mean "not_reported."

### 4. Outcome placeholder
`target_variable = "Primary study outcome (heuristic migration)"` appears in ~40% of evidence matrix records. This is an LLM fallback when the primary outcome could not be determined. All gold records must have the real outcome extracted from source text.

### 5. Excluded study types
From the pilot, the following types are systematically not fine-tuning viable:
- OECD country notes / policy briefs (5 excluded)
- Macroeconomic aggregate studies (GDP growth as outcome)
- Data harmonization papers
- Measurement invariance studies

---

## Construct Distribution (gold_verified + adjudicated)

| Construct | Evidence rows |
|-----------|-------------:|
| ICT_HOME_ACCESS | 22 |
| GENDER | 20 |
| SES_COMPOSITE | 18 |
| SELF_CONCEPT | 11 |
| IMMIGRANT_STATUS | 9 |
| ANXIETY | 6 |
| MOTIVATION | 5 |
| SCHOOL_BELONGING | 3 |
| UNMATCHED | 22 |
| GDP_EXPENDITURE | 4 |
| SES_COMPOSITE (duplicate predictor) | 3 |
| Other | 5 |

---

## Annotation Agreement (Pilot)

Single annotator (Merve Dede) — no second annotator available for this pilot.

| Category | Assessment |
|----------|-----------|
| Program assignment | Deterministic — confirmed in 17/18 records |
| Cycle assignment | Deterministic — confirmed in 16/18 records |
| Target variable | Confirmed in 13/18 (5 had LLM placeholder) |
| Canonical construct | Confirmed with corrections in 15/18 (3 had errors) |
| Direction | 12/18 confirmed; 5 direction=0 ambiguous; 1 N/A |
| Effect type | 13/18 confirmed; 3 corrections; 2 excluded |

**No second-pass intra-rater statistics** — washout period required (≥7 days).

---

## Difficult Cases Covered

From the protocol's 20-category difficulty taxonomy:

| Category | Example |
|----------|---------|
| A. predictor vs confounder | Pejic_StanicMolcer: time-on-task is predictor not SES |
| D. direction reversal | Creative self-efficacy direction in Spanish PISA |
| E. multiple predictors | 1-s2.0-S2666374025001359: 27 evidence rows |
| G. multiple programs | Test-scores-GDP: TIMSS+PISA → excluded (macro) |
| L. canonical mapping | UNMATCHED_TEACHER_ICT issue across TALIS |
| M. plausible values | Rubin's rules vs single PV |
| N. student weights | Confirmed in TALIS records |
| Q. methodological papers | data-10-00130 excluded |
| R. technical reports | PIAAC country notes excluded |
| T. non-ML described as ML | Are-life-goals (MGCFA) flagged as ambiguous |

10 of 20 difficulty categories covered in pilot.

---

## Remaining Limitations

1. **N=12 gold studies is insufficient** for fine-tuning. Target: 51–68 studies across programs.
2. **PIAAC gap:** 0 viable gold studies from pilot. Requires journal-article-only filtering.
3. **Second-pass reliability not conducted.** Intra-rater consistency statistics pending.
4. **200+ manual-review records not yet assessed.** Many are promotable after normalization.
5. **Full gold dataset not final** — clearly marked annotation_pending for remaining 42 studies.

---

*Generated: 2026-10-05 | Phase 3 pilot complete | Full annotation pending*
