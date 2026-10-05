# Annotation Codebook Changelog

---

## v1.0 — 2026-10-05

**Status:** Initial release  
**Based on:** Pilot annotation of 18 records across 7 programs  

### New rules established

| Rule | Affected fields | Reason |
|------|----------------|---------|
| ICT_HOME_ACCESS does NOT apply to teacher-level ICT in TALIS | canonical_construct | Pilot: 10+15959+RPGE+EN.pdf, Patterns..., ICILS teachers |
| direction=0 in evidence matrix = not_reported unless source confirms null | direction_canonical | Pilot: systematic ambiguity in direction encoding |
| OECD country notes → excluded (descriptive reports) | gold_status | Pilot: 09034ae3, 5jlpq7tg3hxs, 5kmhds6czqzq all excluded |
| "Primary study outcome (heuristic migration)" = LLM placeholder → must find real outcome | target_variable | Systematic LLM failure mode across multiple studies |
| Measurement invariance studies (CFA/MGCFA) → excluded OR descriptive | gold_status, effect_type | Pilot: Are-life-goals-invariant... ambiguous |
| Macroeconomic aggregate studies → excluded | gold_status | Pilot: Test scores and economic growth excluded |
| Data harmonization papers → excluded | gold_status | Pilot: data-10-00130 excluded |
| rubin_rules → all_pvs (HIGH confidence) — does not block silver | plausible_values_handling | Normalization script validation |
| irt_theta / wle → code as single_pv with flag | plausible_values_handling | Neither is the standard PV procedure |
| direction=not_reported (not +/-/0) when NaN in evidence matrix | direction_canonical | Explicit rule to avoid misinterpretation |

### Fields added to gold schema (not in v0)
- ambiguity_flag (multiple values with | separator)
- adjudication_note
- source_supported_claim

### Fields modified
- effect_type: added "descriptive" as explicit value (was not in original protocol)
- plausible_values_handling: irt_theta and wle guidance added

### Records affected
All 18 pilot records. Future annotations inherit these rules from v1.0.

---

## v1.1 — 2026-10-05

**Status:** Phase 3B update  
**Scope:** Three targeted rule changes based on PIAAC eligibility analysis, TALIS ICT corpus review, and gold annotation pilot findings.

---

## Change 1: TEACHER_ICT_USE — New canonical construct for TALIS teacher-level ICT

- **Field:** canonical_construct
- **Old rule:** TALIS teacher-level ICT variables coded as UNMATCHED_TEACHER_ICT (no canonical mapping). Rule: ICT_HOME_ACCESS does NOT apply to teacher-level ICT in TALIS.
- **New rule:** TALIS teacher-level ICT use variables (e.g., "use of ICT in teaching", "frequency of ICT use in class") → code as TEACHER_ICT_USE. This is a new canonical construct. ICT_HOME_ACCESS still applies ONLY to student home access variables (PISA, PIRLS). UNMATCHED_TEACHER_ICT is deprecated — recode existing instances to TEACHER_ICT_USE.
- **Reason:** 31 gold evidence rows carry UNMATCHED_TEACHER_ICT (the largest single source of UNMATCHED in the gold set). This unresolved mapping degrades LLM training signal for TALIS fine-tuning. A dedicated canonical construct enables proper construct-level evaluation.
- **Examples:** "Frequency of ICT use in class" (TALIS 2018) → TEACHER_ICT_USE; "Use of digital tools for instruction" → TEACHER_ICT_USE; "Home computer access for students" (PISA 2018) → ICT_HOME_ACCESS (unchanged).
- **Source basis:** Gold pilot data — 31 instances of UNMATCHED_TEACHER_ICT across TALIS studies in gold_standard_full_corpus.csv. Codebook v1.0 note: "ICT_HOME_ACCESS does NOT apply to teacher-level ICT in TALIS."
- **Affected records:** ~31 gold evidence rows (TALIS studies); estimated 15-25 silver records in TALIS pool.

---

## Change 2: PIAAC literacy/numeracy coding rules

- **Field:** canonical_construct, target_variable
- **Old rule:** No explicit PIAAC-specific rule. Literacy proficiency was coded inconsistently as UNMATCHED or mixed with student-level constructs.
- **New rule:**
  - PIAAC adult literacy proficiency as OUTCOME (dependent variable): code target_variable as "Literacy proficiency score (PIAAC)" and do NOT use a predictor-level canonical construct for the outcome.
  - PIAAC literacy used as a PREDICTOR: code canonical_construct as LITERACY_PROFICIENCY_PIAAC.
  - PIAAC numeracy as OUTCOME: target_variable = "Numeracy proficiency score (PIAAC)".
  - PIAAC numeracy as PREDICTOR: canonical_construct = NUMERACY_PROFICIENCY_PIAAC.
  - PIAAC problem-solving in technology-rich environments (PSTRE) as outcome/predictor: PSTRE_PIAAC.
- **Reason:** PIAAC is structurally different from school-based ILSAs — the assessment measures adult workforce skills, not student academic achievement. The literacy/numeracy distinction matters for forecasting model interpretability. 0/12 current gold studies cover PIAAC; explicit rules needed before PIAAC annotation begins.
- **Examples:** Study predicting earnings from literacy score → predictor = LITERACY_PROFICIENCY_PIAAC; Study predicting literacy from education level → target_variable = "Literacy proficiency score (PIAAC)", predictor construct = PARENTAL_EDUCATION.
- **Source basis:** PIAAC eligibility filter (Phase 3B) identified 52 PIAAC silver records with no coverage in gold. Codebook gap confirmed by Phase 3 pilot.
- **Affected records:** ~52 PIAAC silver records; all future PIAAC annotations.

---

## Change 3: SELF_REGULATION vs SELF_CONCEPT — treat as distinct constructs

- **Field:** canonical_construct
- **Old rule:** No explicit disambiguation rule. Risk of conflation between self-regulation (behavioral control strategies) and self-concept (self-perception of academic ability).
- **New rule:**
  - SELF_CONCEPT: applies when the study measures students' self-perceived ability, competence beliefs, or academic self-concept (e.g., "I am good at math", PISA self-concept scales). Cognitive appraisal of own ability.
  - SELF_REGULATION: applies when the study measures learning strategy use, metacognitive control, or behavioral self-management (e.g., time management, elaboration strategies, goal-setting). Behavioral/strategic dimension.
  - If a scale conflates both dimensions, code as SELF_CONCEPT if the primary items are ability-belief items; code as SELF_REGULATION if the primary items are strategy items.
  - Never merge the two. Flag ambiguous cases with ambiguity_flag=SELF_REG_CONCEPT_OVERLAP.
- **Reason:** In PISA and TALIS, self-concept and self-regulation are assessed by distinct scale batteries. Conflating them degrades construct-level forecasting validity. Gold pilot data shows SELF_CONCEPT appears in 10 gold evidence rows — all correctly coded as self-concept (no conflation found in existing gold).
- **Examples:** PISA "SCMAT" (math self-concept) → SELF_CONCEPT; PISA "METASUM" (metacognitive strategies) → SELF_REGULATION.
- **Source basis:** 10 existing gold evidence rows with SELF_CONCEPT; Phase 3B construct coverage analysis.
- **Affected records:** Existing gold: 10 rows (all correct); future annotations with self-regulation constructs.

---

## Summary of v1.1 changes

| Change | Type | New construct(s) | Deprecations | Affected records |
|--------|------|-----------------|--------------|-----------------|
| 1. TEACHER_ICT_USE | New canonical construct | TEACHER_ICT_USE | UNMATCHED_TEACHER_ICT | ~31 gold + ~20 silver |
| 2. PIAAC literacy/numeracy | New coding rules | LITERACY_PROFICIENCY_PIAAC, NUMERACY_PROFICIENCY_PIAAC, PSTRE_PIAAC | none | ~52 silver |
| 3. SELF_REGULATION vs SELF_CONCEPT | Disambiguation rule | none (clarification only) | none | future annotations |

---

*Document initialized: 2026-10-05 (v1.0)*  
*v1.1 update: 2026-10-05*
