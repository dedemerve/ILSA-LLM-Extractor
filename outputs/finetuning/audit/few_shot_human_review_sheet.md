# Few-Shot Examples — Human Review Sheet

**File under review:** `outputs/finetuning/baseline/few_shot_examples_frozen.json`  
**Review date:** 2026-10-06  
**Reviewer:** (researcher decision required)  
**Reliability status:** CLEARED (macro κ = 0.871 ≥ 0.61, 2026-10-06)  
**Protocol reference:** EVALUATION_PROTOCOL_v1.md §6.1

This sheet is generated from existing repository data only. No modifications have been made to the JSON file or gold standard. Decisions APPROVE / REJECT must be supplied by the researcher.

---

## Selection Criteria (from EVALUATION_PROTOCOL_v1.md §6.1)

The protocol requires 3–5 demonstrations stratified as:
- 1 PISA study — HIGH construct, clear direction
- 1 TIMSS study — MID construct, rubin_rules PV handling
- 1 PIAAC study — labor market outcome, NOT_FORECASTABLE example included
- Optional: 1 PIRLS or TALIS study for diversity

All demonstrations must be from the **training split only**.

---

## Cross-Cutting Concerns (applies to all shots)

The evidence rows in the JSON are structured as gold annotation records, not as the 10-field protocol output schema (§3.2). Specific mismatches:

| Protocol field (§3.2) | JSON field | Present? | Note |
|-----------------------|-----------|---------|------|
| `predictor` | `predictor` | ✓ | |
| `ilsa_variable` | — | ✗ | Missing; nullable per protocol |
| `canonical_construct` | `canonical_construct` | ✓ | |
| `outcome` | `target_variable` | ✓ (different name) | Field name differs from schema |
| `effect_type` | `effect_type` | ✓ | |
| `direction` | `direction_canonical` | ✓ (different name) | Field name differs; shot_1 uses "1" not "positive" |
| `effect_size` | — | ✗ | Missing |
| `plausible_values_handling` | `plausible_values_handling` | ✓ | |
| `student_weights_used` | — | ✗ | Missing |
| `evidence_text` | — | ✗ | Missing; mandatory per protocol |

**Implication:** These rows serve as gold reference labels (study_id → expected output), not as formatted demonstration pairs. The baseline prompt must specify how to translate these records into (source_text → extraction) demonstration pairs. This does not block the selection choice but must be addressed in the prompt design.

---

## Shot 1 — PISA (Candidate)

| Field | Value |
|-------|-------|
| Shot key | `shot_1_PISA` |
| Study ID | `1-s2.0-S2666374025001359-main.pdf` |
| Title | Creative thinking, mathematics achievement, and well-being: A multilevel analysis of PISA 2022 |
| Program | PISA |
| Cycle | PISA 2022 |
| Protocol criteria match | PISA ✓ |
| Selection rationale | PISA 2022; HIGH constructs (SES_COMPOSITE, GENDER, IMMIGRANT_STATUS, SELF_CONCEPT); single_pv; E_predictive |
| Training split | ✓ confirmed |
| Current status | CANDIDATE |

**Evidence rows (27 total):**

| record_id | predictor | canonical_construct | construct_status | direction | ambiguity_flag | human_label_status |
|-----------|-----------|-------------------|-----------------|-----------|---------------|-------------------|
| G0013 | Creative self-efficacy | SELF_CONCEPT | FORECASTABLE | 1 | pv_handling_mixed | gold_verified_after_adjudication |
| G0014 | Socioeconomic status (ESCS) | SES_COMPOSITE | FORECASTABLE | 1 | pv_handling_mixed | gold_verified_after_adjudication |
| G0015 | Gender | GENDER | FORECASTABLE | 1 | pv_handling_mixed | gold_verified_after_adjudication |
| G0016 | Immigration background | IMMIGRANT_STATUS | FORECASTABLE | 1 | pv_handling_mixed | gold_verified_after_adjudication |
| G0017–G0039 | (23 additional rows) | SES_COMPOSITE / GENDER / IMMIGRANT_STATUS / SELF_CONCEPT | FORECASTABLE | 1 | pv_handling_mixed | gold_verified_after_adjudication |

**Concerns flagged (for researcher decision):**

1. **Volume:** 27 rows in a single shot is very large. Most context windows will include all rows, but this may crowd out the other shots or the actual extraction task. The protocol says "3–5 demonstrations" — it is unclear whether this means 3–5 studies or 3–5 evidence rows total.

2. **Ambiguity flag:** All 27 rows carry `pv_handling_mixed` ambiguity. The study uses mixed PV handling (single_pv in some models, multi-level in others). This is a legitimate annotation edge case. The protocol says the PISA shot should have "clear direction" — all rows here have direction="1" (positive), which after canonicalization is consistent.

3. **`direction_canonical` = "1":** The raw value "1" appears in the JSON instead of "positive". Per the v1.4 adjudication note, 1→positive. This must be canonicalized in the prompt format.

4. **All rows `gold_verified_after_adjudication`:** This is the status after second-pass reliability for this study. The annotations are valid but were all reviewed and adjudicated.

5. **Protocol criterion check:** The protocol specifies "HIGH construct, clear direction." SELF_CONCEPT, SES_COMPOSITE, GENDER, IMMIGRANT_STATUS are all HIGH constructs. Direction="1" (positive) is clear once canonicalized. Criteria MET, pending direction canonicalization.

**Researcher decision required:** APPROVE / REJECT

---

## Shot 2 — TIMSS (Candidate)

| Field | Value |
|-------|-------|
| Shot key | `shot_2_TIMSS` |
| Study ID | `thinking-skills-in-teaching-practices.pdf` |
| Title | (TIMSS+PISA cross-program study on teaching practices) |
| Program | TIMSS+PISA |
| Cycle | TIMSS 2011 + PISA 2012 |
| Protocol criteria match | TIMSS ✓ (cross-program; primary TIMSS dataset used) |
| Selection rationale | TIMSS+PISA; TEACHING_QUALITY, SES_COMPOSITE; rubin_rules; E_assoc; mixed direction |
| Training split | ✓ confirmed |
| Current status | CANDIDATE |

**Evidence rows (4 total):**

| record_id | predictor | canonical_construct | construct_status | direction | ambiguity_flag | human_label_status |
|-----------|-----------|-------------------|-----------------|-----------|---------------|-------------------|
| G0162 | Lower-order teaching practices | TEACHING_QUALITY | FORECASTABLE | mixed | none | gold_verified |
| G0163 | Higher-order teaching practices | TEACHING_QUALITY | FORECASTABLE | mixed | none | gold_verified |
| G0164 | Student SES | SES_COMPOSITE | FORECASTABLE | mixed | none | gold_verified |
| G0165 | School/teacher characteristics | UNMATCHED | NOT_FORECASTABLE | mixed | none | gold_verified |

**Concerns flagged:**

1. **Program is TIMSS+PISA, not pure TIMSS:** The protocol asks for "1 TIMSS study." This is a cross-program study using both TIMSS 2011 and PISA 2012 data. It demonstrates cross-program extraction. Minor deviation from the protocol criterion.

2. **Includes 1 NOT_FORECASTABLE row (G0165):** The protocol specifies a NOT_FORECASTABLE example for the PIAAC shot (shot 3). Having one also in shot 2 is additional positive coverage, not a protocol violation.

3. **4 rows with no ambiguity flags, all gold_verified:** Clean, unambiguous annotations. Well-suited as a demonstration.

4. **rubin_rules PV handling:** Meets the protocol criterion exactly.

5. **TEACHING_QUALITY is a MID-stratum construct:** Meets protocol criterion.

**No critical concerns.** Study selection is well-justified.

**Researcher decision required:** APPROVE / REJECT

---

## Shot 3 — PIAAC (Candidate)

| Field | Value |
|-------|-------|
| Shot key | `shot_3_PIAAC` |
| Study ID | `jwl-04-2021-0047.pdf` |
| Title | (PIAAC study on informal learning and workplace autonomy) |
| Program | PIAAC |
| Cycle | PIAAC Cycle 1 (Rounds 1–3) |
| Protocol criteria match | PIAAC ✓; labor market outcome ✓; NOT_FORECASTABLE included ✓ |
| Selection rationale | PIAAC labor market; WORKPLACE_AUTONOMY; not_applicable PV; UNMATCHED included (readiness to learn) |
| Training split | ✓ confirmed |
| Current status | CANDIDATE |

**Evidence rows (4 total):**

| record_id | predictor | canonical_construct | construct_status | direction | ambiguity_flag | human_label_status |
|-----------|-----------|-------------------|-----------------|-----------|---------------|-------------------|
| G0197 | Workplace autonomy | WORKPLACE_AUTONOMY | FORECASTABLE | positive | none | gold_verified |
| G0198 | Interaction in the workplace | UNMATCHED | NOT_FORECASTABLE | positive | none | gold_verified |
| G0199 | Readiness to learn | UNMATCHED | FORECASTABLE | positive | none | gold_verified |
| G0200 | Demographic controls | UNMATCHED | NOT_FORECASTABLE | positive | none | gold_verified |

**Concerns flagged:**

1. **G0199 is UNMATCHED + FORECASTABLE:** "Readiness to learn" is a forecastable predictor concept but lacks a canonical construct label in the current codebook. This is a valid edge case to demonstrate — the model learns to output UNMATCHED when a predictor is forecastable in principle but lacks a codebook entry.

2. **WORKPLACE_AUTONOMY is a SINGLE-stratum construct** (1 row total in corpus): This demonstrates a low-frequency construct, which the protocol §4.4 notes must be reported separately. Good diversity.

3. **not_applicable PV handling:** Meets protocol criterion (PIAAC uses skills assessment scores, not plausible values in the same sense).

4. **3 of 4 rows are UNMATCHED:** Appropriate for a PIAAC labor market study where many constructs fall outside the ILSA-specific codebook. Demonstrates UNMATCHED behavior.

5. **No ambiguity flags, all gold_verified:** Clean annotations.

**No critical concerns.** Well-matched to protocol criteria.

**Researcher decision required:** APPROVE / REJECT

---

## Shot 4 — PIRLS (Candidate)

| Field | Value |
|-------|-------|
| Shot key | `shot_4_PIRLS` |
| Study ID | `1-s2.0-S0022096522001059-main.pdf` |
| Title | (PIRLS 2006 longitudinal study — Theory of Mind and reading) |
| Program | PIRLS |
| Cycle | PIRLS 2006 |
| Protocol criteria match | PIRLS ✓ (optional diversity study) |
| Selection rationale | PIRLS reading; MOTIVATION, GENDER; not_applicable PV; E_predictive; LOW-frequency construct example |
| Training split | ✓ confirmed |
| Current status | CANDIDATE |

**Evidence rows (6 total):**

| record_id | predictor | canonical_construct | construct_status | direction | ambiguity_flag | human_label_status |
|-----------|-----------|-------------------|-----------------|-----------|---------------|-------------------|
| G0069 | Reading motivation | MOTIVATION | FORECASTABLE | not_reported | direction_zero_ambiguous | gold_verified |
| G0070 | Gender | GENDER | FORECASTABLE | not_reported | direction_zero_ambiguous | gold_verified |
| G0071 | Reading motivation | MOTIVATION | FORECASTABLE | not_reported | direction_zero_ambiguous | gold_verified |
| G0072 | Gender | GENDER | FORECASTABLE | not_reported | direction_zero_ambiguous | gold_verified |
| G0073 | Parental education | PARENTAL_EDUCATION | FORECASTABLE | not_reported | direction_zero_ambiguous | gold_verified |
| G0074 | Socioeconomic status (SES) | SES_COMPOSITE | FORECASTABLE | not_reported | direction_zero_ambiguous | gold_verified |

**Concerns flagged:**

1. **All 6 rows have `direction_zero_ambiguous` ambiguity flag:** The direction was originally coded 0 and canonicalized to `not_reported`. This is a legitimate edge case (study did not report signed effects clearly), but using it as a demonstration means the model learns to output `not_reported` direction for this study pattern. This is appropriate behavior per the codebook.

2. **All directions `not_reported`:** Uniform direction across all 6 rows — demonstrates that `not_reported` is a valid output, not a fallback. Appropriate for this longitudinal design.

3. **Longitudinal outcome (Theory of Mind):** Unusual outcome for PIRLS (typically reading literacy). Valid PIRLS 2006 study. Does not affect construct annotation quality.

4. **PARENTAL_EDUCATION is a LOW-stratum construct:** Good diversity within the demonstration.

5. **All gold_verified, no adjudication:** Clean, primary annotations.

**No critical concerns.**

**Researcher decision required:** APPROVE / REJECT

---

## Summary — Decisions Pending

| Shot | Program | Protocol criterion match | Critical concerns | Researcher decision |
|------|---------|------------------------|------------------|---------------------|
| shot_1_PISA | PISA 2022 | ✓ HIGH constructs, clear direction (canonicalization needed) | Large (27 rows); all `pv_handling_mixed`; direction="1" needs canonical form | **PENDING** |
| shot_2_TIMSS | TIMSS+PISA | ✓ MID construct, rubin_rules (cross-program deviation) | Cross-program study, not pure TIMSS | **PENDING** |
| shot_3_PIAAC | PIAAC | ✓ Labor market, NOT_FORECASTABLE included | None critical | **PENDING** |
| shot_4_PIRLS | PIRLS | ✓ Optional diversity | All `direction_zero_ambiguous` | **PENDING** |

**FEW_SHOT_EXAMPLES status: NOT_FROZEN — awaiting researcher decisions above.**

No changes have been made to `few_shot_examples_frozen.json`. The file remains CANDIDATE status.
