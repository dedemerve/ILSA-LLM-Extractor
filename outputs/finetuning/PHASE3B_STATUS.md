# Phase 3B Status Report — ILSA Gold Annotation Sampling and PIAAC Coverage

**Date:** 2026-10-05  
**Branch:** experiment/ilsa-domain-finetuning  
**Phase:** 3B — Audit, Coverage Analysis, and Annotation Decision Framework

---

## 1. Objective

Phase 3B extends the Phase 3 pilot by:
- Auditing the 12 existing human-verified gold studies for completeness
- Filtering and profiling PIAAC silver candidates for gold eligibility
- Profiling the full 526-record silver pool (v2)
- Analyzing 4-level program coverage (corpus → source-text → silver → gold)
- Auditing construct coverage and identifying difficult methodological cases
- Building an annotation decision framework for the next ~42 records
- Updating the annotation codebook to v1.1 with three targeted rule changes
- Creating a versioned gold corpus (v2) filtered to human-verified records only

---

## 2. Current Corpus

| Level | Total |
|-------|-------|
| Full corpus (1_Articles_Master) | 1,266 |
| Source-text available | 577 |
| Silver (all statuses) | 618 |
| Silver verified candidates | 526 |
| Silver text manual review | 51 |
| Silver only | 41 |

The 42 annotation-pending records are NOT in gold and were NOT annotated in this phase.

---

## 3. Current Silver Pool

**Silver Pool v2** (outputs/finetuning/audit/silver_pool_v2_profile.csv):
- Total: 526 silver_verified_candidate records
- Original 341 (profiled in Phase 3): matched to old silver_candidate_profile.csv by DOI
- Promoted 185 (Phase 3 promotion): lack program info from old profile; marked `unknown_check_manually`

The 185 promoted records require manual program verification before gold annotation.

---

## 4. PIAAC Filtering

**File:** outputs/finetuning/audit/piaac_gold_eligibility.csv

- Total PIAAC records in silver pool: 52
- Journal articles (empirical): 3
- Reports (descriptive): 49
- Source text available: 52
- Eligible gold candidates (primary_candidate): 3
- Excluded: 49 (48 descriptive reports; 1 missing evidence)

**Critical gap:** PIAAC is unrepresented in the gold set (0/12 studies). The 3 PIAAC journal articles are top-priority annotation candidates. Annotation codebook v1.1 adds explicit PIAAC literacy/numeracy coding rules before this annotation begins.

---

## 5. Program Coverage

**File:** outputs/finetuning/audit/gold_sampling_coverage.csv

| Program | Full corpus | Source-text | Silver 526 | Gold 12 | Gap |
|---------|------------|-------------|------------|---------|-----|
| PISA | 358 (28.3%) | 337 (58.4%) | 175 (33.3%) | 3 (25%) | Adequate |
| TALIS | 69 (5.5%) | 69 (12.0%) | 63 (12.0%) | 3 (25%) | Over-represented (intentional) |
| PIAAC | 59 (4.7%) | 55 (9.5%) | 49 (9.3%) | 0 (0%) | **CRITICAL GAP** |
| TIMSS | 56 (4.4%) | 55 (9.5%) | 19 (3.6%) | 1 (8.3%) | Underrepresented |
| PIRLS | 31 (2.4%) | 29 (5.0%) | 13 (2.5%) | 2 (16.7%) | Acceptable |
| ICCS | 15 (1.2%) | 14 (2.4%) | 8 (1.5%) | 1 (8.3%) | Underrepresented |
| ICILS | 18 (1.4%) | 17 (2.9%) | 9 (1.7%) | 2 (16.7%) | Acceptable |

Note: Silver 526 program counts are partial (185 promoted records have unknown program).

---

## 6. Construct Coverage

**File:** outputs/finetuning/audit/gold_construct_coverage.csv

Gold evidence rows by canonical construct (top 5):
- UNMATCHED_TEACHER_ICT: 31 (highest — deprecated in v1.1 → TEACHER_ICT_USE)
- GENDER: 18
- SES_COMPOSITE: 18
- UNMATCHED: 10
- SELF_CONCEPT: 10

High-priority constructs with 0 gold instances:
- Numerous constructs from the silver pool not yet represented in gold

---

## 7. Difficult-Case Coverage

**File:** outputs/finetuning/audit/difficult_case_inventory.csv

- 64 records flagged as difficult in the old 341 profile
- Difficulty categories: multi_program, has_unmatched, complex_pv, knn_imputation
- These records are annotatable but require extra care; marked for review before annotation

---

## 8. Gold Sampling Strategy

**File:** outputs/finetuning/audit/annotation_pending_decision.csv

- 514 silver_verified records not yet in gold
- 42 selected for next annotation batch (decision=A)
- Selection priority:
  1. PIAAC journal articles (critical gap)
  2. TIMSS journal articles (underrepresented)
  3. PIRLS journal articles
  4. ICCS/ICILS journal articles
  5. Diverse construct coverage from remaining programs

---

## 9. Existing Gold Audit

**File:** outputs/finetuning/audit/existing_gold_audit.csv

All 12 human-verified gold studies audited:
- All 12: source_text_available=TRUE
- Tier-1 completeness: all fields present
- Source location recorded: all studies have source_location annotated
- No corrections required for any of the 12 studies

**One structural issue identified:** UNMATCHED_TEACHER_ICT (31 gold evidence rows across TALIS studies). This is addressed in codebook v1.1 by introducing TEACHER_ICT_USE as a canonical construct.

---

## 10. New Human Annotation

No new human annotations were performed in Phase 3B. The 42 annotation-pending records are designated for the next phase. Codebook v1.1 rules must be applied to all future annotations.

---

## 11. Annotation Quality

**File:** outputs/finetuning/audit/gold_annotation_qc.csv

QC framework initialized with 12 existing gold studies (all qc_pass=TRUE). Template columns are defined for future annotated records.

---

## 12. Final Gold Distribution

**File:** outputs/finetuning/audit/final_gold_distribution.csv

Gold v2 (128 evidence rows, 12 studies):
- Programs: PISA=3, TALIS=3, TIMSS=1, PIRLS=2, ICCS=1, ICILS=2
- Publication types: journal=12
- Dominant constructs: UNMATCHED_TEACHER_ICT (31), GENDER (18), SES_COMPOSITE (18)

Gold v2 file: outputs/finetuning/data/gold_standard_full_corpus_v2.csv (128 rows, excludes 5 excluded + 1 ambiguous)

---

## 13. Limitations

1. **185 promoted records lack program info.** The silver pool v2 profile marks these as `unknown_check_manually`. Program distribution for silver 526 is partial.
2. **PIAAC gap is critical.** 0/12 gold studies cover PIAAC. This must be addressed in the next annotation batch.
3. **UNMATCHED_TEACHER_ICT.** 31 gold evidence rows carry this deprecated label. These should be recoded to TEACHER_ICT_USE per v1.1 rules in a future cleanup pass (does not block current use).
4. **No train/test splits.** Downstream fine-tuning pipeline has not been run. Gold corpus is pre-split.
5. **Phase 3B did not annotate any records.** All counts reflect existing human annotations only.

---

## 14. Fine-Tuning Readiness

| Component | Status |
|-----------|--------|
| Gold corpus (v2) | Ready: 128 evidence rows, 12 studies |
| Silver pool profiled | Ready: 526 records in v2 profile |
| Annotation codebook | v1.1 released |
| QC framework | Initialized |
| PIAAC coverage | NOT ready — 0 PIAAC gold studies |
| Train/test split | NOT created (Phase 4) |
| Fine-tuning pipeline | NOT run (Phase 4) |

Fine-tuning is NOT recommended until at least 3-5 PIAAC and additional TIMSS/PIRLS gold studies are annotated.

---

## 15. Next Phase

**Phase 3C — Human Annotation Batch 2:**
1. Annotate ~42 selected records (annotation_pending_decision.csv, decision=A)
2. Prioritize: PIAAC journals (3) → TIMSS journals → PIRLS journals → ICCS/ICILS
3. Apply codebook v1.1 rules (TEACHER_ICT_USE, PIAAC literacy, SELF_REG vs SELF_CONCEPT)
4. Recode existing UNMATCHED_TEACHER_ICT → TEACHER_ICT_USE in gold v2
5. Update gold_annotation_qc.csv with new QC records
6. After batch: update gold_standard_full_corpus_v2.csv
7. After 30+ gold studies: proceed to Phase 4 (train/test split + fine-tuning)

---

*Phase 3B complete: 2026-10-05*
