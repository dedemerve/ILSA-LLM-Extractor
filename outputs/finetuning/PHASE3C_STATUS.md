# Phase 3C Status: Annotation Batch Preparation and Gold Freeze

**GOLD_STATUS = ANNOTATION_STILL_REQUIRED**
**Date:** 2026-10-05
**Branch:** experiment/ilsa-domain-finetuning
**Git Commit:** 9253e2e

---

## 1. Objective

Phase 3C prepares a structured annotation batch of 42 records to expand the gold standard corpus, with PIAAC-first prioritization to close the key program coverage gap. Since actual source-text reading by human annotator (Merve Dede) cannot occur in a computational session, this phase: (a) structures the 42 records into priority batches with all available LLM-extracted data pre-loaded, (b) freezes the gold corpus at v3 = 12 studies / 128 rows pending human annotation, and (c) documents all integrity checks and readiness blockers.

---

## 2. Initial Gold Status

| Item | Value |
|------|-------|
| Gold version | v2 → v3 (unchanged, 12 studies) |
| Evidence rows | 128 |
| Studies | 12 |
| PISA | 3 (25%) |
| TALIS | 3 (25%) |
| PIRLS | 2 (17%) |
| ICILS | 2 (17%) |
| TIMSS | 1 (8%) |
| ICCS | 1 (8%) |
| PIAAC | 0 (0%) |

The PIAAC gap is the primary blocker. PISA share = 25%, within the ≤33% target.

---

## 3. Annotation Batches

The 42 selected records (decision='A' in annotation_pending_decision.csv) are structured into priority batches:

| Batch | Priority | Records | Description |
|-------|----------|---------|-------------|
| 1 | PIAAC-first | 18 | PIAAC primary candidates from piaac_gold_eligibility.csv (annotation_priority=1) |
| 2 | Underrepresented | 24 | TIMSS (17 records), PIRLS (6 records), multi-program (1 record) |
| 3 | Difficult cases | 0 | None of the 42 annotate-decision records are in difficult_case_inventory.csv |
| 4 | Remaining | 0 | All records covered by batches 1-2 |

All 42 are journal articles (publication_type=journal). 5 records have LLM JSON extractions available; the remaining 37 require full source-text annotation.

See: `outputs/finetuning/audit/phase3c_annotation_batch.csv`

---

## 4. PIAAC Annotation

- 52 PIAAC records in piaac_gold_eligibility.csv
- 18 records marked annotation_priority=1 (primary_candidates)
- All 18 are journal articles (eligible for gold)
- OECD descriptive reports excluded from gold (not empirical studies)
- 5 of 18 have LLM JSON extraction available
- All 18 require human source-text verification before gold inclusion
- **Status: QUEUED FOR HUMAN REVIEW**

---

## 5. Underrepresented Program Coverage

Programs with <3 gold studies at phase start: TIMSS (1), PIRLS (2), ICCS (1), PIAAC (0).

| Program | Current Gold Studies | Batch 2 Records Queued |
|---------|---------------------|----------------------|
| TIMSS | 1 | 17 (TIMSS/TIMSS_G4) |
| PIRLS | 2 | 6 |
| PIAAC | 0 | 18 (Batch 1) |
| ICCS | 1 | 0 |

---

## 6. Difficult-Case Coverage

64 records in difficult_case_inventory.csv. Of the 42 annotate-decision records, 0 overlap with the difficult-case inventory directly (the difficult cases were separately tracked; the 42-record batch was drawn from silver_verified_candidates by program coverage gap, not from the difficult-case pool).

The difficult_case_inventory.csv should be revisited in Phase 3D after the 42-record batch is human-verified.

---

## 7. Human Annotation Procedure

Source-text-first protocol (codebook v1.1, 14-step process):

1. Retrieve source PDF from outputs/ilsa_cache/ or journal DOI
2. Confirm article identity (title, authors, year, DOI)
3. Confirm program (ILSA acronym from abstract/methods)
4. Identify assessment cycle year from data section
5. Extract target variable (dependent variable)
6. Map to canonical construct using codebook mapping table
7. Extract predictor variables
8. Determine effect type (predictive/associative/unspecified)
9. Determine direction from reported coefficients or conclusions
10. Extract evidence type
11. Note plausible values and weighting handling
12. Record ML techniques used
13. Flag ambiguities for adjudication
14. Assign human_label_status = gold_verified

All field values must be directly supported by source text. No inference beyond what is written. If a field cannot be determined from source text, mark as ambiguous_flag=TRUE.

---

## 8. Annotation Disagreements

In this computational pass, no human-vs-LLM disagreements have been resolved (no human annotation performed). The human_annotation_decision_log.csv flags all 42 records as requiring human source-text verification.

For the 5 records with JSON extractions, LLM field values are pre-loaded but all fields are marked requires_source_text_review. This is intentional: LLM extraction is a starting point, not a gold label.

---

## 9. Intra-Rater Reliability

- Annotator: Merve Dede (single annotator)
- Second-pass intra-rater reliability: NOT YET CONDUCTED
- Washout period: 7 days required from annotation date
- Planned second-pass date: ≥2026-10-12 (after 42-record batch annotation)
- Previous reliability (Phase 3B, 12 studies): not formally reported; all 12 studies passed QC

---

## 10. Final Gold Distribution (v3 = v2, 12 studies)

**By Program (study-level):**
PISA=3, TALIS=3, PIRLS=2, ICILS=2, TIMSS=1, ICCS=1, PIAAC=0

**By Program (evidence row-level):**
ICILS=47, PISA=39, TALIS=19, PIRLS=12, ICCS=6, TIMSS=5

**By Publication Type:**
journal=123, conference=5

**By Effect Type:**
predictive=97, associative=22, unspecified=9

**By Evidence Type:**
predictive=97, associative=22, unspecified=9

**Top Canonical Constructs:**
UNMATCHED_TEACHER_ICT=31, GENDER=18, SES_COMPOSITE=18, UNMATCHED=10, SELF_CONCEPT=10, ICT_HOME_ACCESS=10, DISCIPLINARY_CLIMATE=9, IMMIGRANT_STATUS=8

---

## 11. Gold Freeze

- **gold_standard_full_corpus_v3.csv** = identical to v2 (12 studies, 128 rows)
- v3 is the current frozen gold corpus
- The 42-record annotation batch is tracked in phase3c_annotation_batch.csv (NOT in the gold file)
- Records are added to gold ONLY after human source-text verification
- Previous versions preserved: gold_standard_full_corpus.csv (v1), gold_standard_full_corpus_v2.csv (v2)

---

## 12. Limitations

1. **PIAAC gap**: No PIAAC records in gold. 18 primary candidates queued but not yet verified.
2. **42-record batch not human-verified**: All 42 records require Merve Dede to read source PDFs. This cannot be done computationally.
3. **No second-pass reliability**: Intra-rater reliability has not been computed for any gold records. Washout period applies.
4. **Only 5/42 records have LLM JSON extractions**: 37 records have no pre-extracted data; annotator must work entirely from source PDFs.
5. **Difficult cases deferred**: The 64-record difficult_case_inventory.csv was not drawn from for the 42-record batch; a separate difficult-case annotation round is needed.

---

## 13. Fine-Tuning Readiness

**NOT READY**

Blockers:
- [ ] PIAAC coverage: 0/1+ studies needed
- [ ] 42-record annotation batch: requires human source-text reading
- [ ] Second-pass intra-rater reliability: not conducted
- [ ] Minimum gold threshold not evaluated (depends on annotation batch outcome)
- [ ] Train/val/test split: explicitly prohibited until gold is complete

---

## 14. Next Phase

**Phase 3D: Human Annotation + Reliability**

1. Merve Dede reads source PDFs for all 42 queued records (using phase3c_annotation_batch.csv as guide)
2. Priority: Batch 1 (PIAAC, 18 records) → Batch 2 (TIMSS/PIRLS, 24 records)
3. For each record: complete 14-step annotation protocol, update human_annotation_decision_log.csv
4. After ≥7-day washout: conduct second-pass reliability on original 12 gold studies
5. Add newly verified records to gold_standard_full_corpus_v4.csv
6. Re-evaluate gold stop condition (PISA≤33%, PIAAC≥1 study, minimum rows met)
7. If stop condition met: proceed to fine-tuning data preparation

**Estimated Phase 3D timeline:** 2–3 hours human annotation + 7-day washout for reliability

