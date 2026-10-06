# Phase 3C Status — Human Gold Annotation Complete

**Date:** 2026-10-06
**Branch:** experiment/ilsa-domain-finetuning
**GOLD_STATUS:** READY_FOR_SPLIT

---

## 1. Objective
Complete human source-text annotation of the 42-record batch, freeze the gold set at v4, and evaluate readiness for the next phase.

---

## 2. Initial Gold Status (before Phase 3C annotation)
| Metric | Value |
|--------|------:|
| Studies | 12 |
| Evidence rows | 128 |
| Programs | PISA=3, TALIS=3, PIRLS=2, ICILS=2, TIMSS=1, ICCS=1, **PIAAC=0** |

---

## 3. Annotation Batches

**Batch 1 — PIAAC (14 studies annotated)**
All eligible PIAAC journal articles from the silver pool annotated. PIAAC gap resolved.

**Batch 2 — TIMSS / PIRLS (15 + 2 studies annotated)**
Underrepresented programs prioritized. 3 PIRLS records excluded (no ILSA microdata — adapted items only or aggregate statistics).

**Batch 3 — Excluded (4 records)**
- 1.pdf: PIRLS adapted items, no ILSA microdata
- 4.pdf: PIRLS adapted items, no ILSA microdata
- 2.pdf: PISA/PIRLS aggregate state-level statistics
- data-10-00130.pdf: Country-level aggregate harmonization

---

## 4. PIAAC Annotation

| Metric | Value |
|--------|------:|
| PIAAC records in silver pool | 52 |
| PIAAC eligible journal articles (Phase 3B) | 18 |
| PIAAC gold after annotation | **14** |
| PIAAC excluded | 0 (all 14 annotation decisions = gold_verified) |

PIAAC coverage includes: labor market outcomes (wages, automation, productivity), adult competency measurement (literacy, numeracy, problem-solving), process data (PSTRE log files), and personality/workplace learning.

---

## 5. Underrepresented Program Coverage

| Program | Phase 3B gold | Phase 3C additions | Final gold |
|---------|-------------:|-------------------:|-----------:|
| PISA    | 3            | 0                  | 3          |
| TALIS   | 3            | 0                  | 3          |
| TIMSS   | 1            | 14                 | **15**     |
| PIRLS   | 2            | 2                  | **4**      |
| ICILS   | 2            | 0                  | 2          |
| ICCS    | 1            | 0                  | 1          |
| PIAAC   | 0            | **14**             | **14**     |

---

## 6. Difficult-Case Coverage

Represented in gold v4:
- Causal inference methods (IV, DiD, PSM, causal forests, BART, TMLE): 5 studies
- Process data / log files (HMM, path signatures, Word2Vec, LCS): 6 studies
- Multi-program (TIMSS+PISA, PIAAC+PISA+TIMSS): 2 studies
- Multi-cycle (PIRLS 5 cycles, PIAAC 3 rounds): 4 studies
- Automated item scoring (CNNs, ANNs, Bag-of-Words): 4 studies
- Null results: 3 studies (non-significant effects)
- Single-PV usage: 3 studies (flagged for codebook attention)

---

## 7. Human Annotation Procedure

Annotator: Merve Dede
Protocol: 14-step annotation per codebook v1.1
Source text: All gold_verified records annotated from source PDF
Date: 2026-10-06

---

## 8. Annotation Disagreements

4 records excluded:
- Reason A (3 records): PIRLS items adapted for independent studies — no ILSA microdata
- Reason B (1 record): Country-level aggregate harmonization — not student-level ILSA

---

## 9. Intra-Rater Reliability

| Status |
|--------|
| Single annotator (Merve Dede) |
| Second-pass: NOT YET CONDUCTED |
| Washout period from initial pilot (2026-10-05): 7-day minimum → second pass from 2026-10-12 |
| Recommended subset: 30% of gold studies (~13 studies) |

---

## 10. Final Gold Distribution (v4)

| Program | N studies | % |
|---------|----------:|--:|
| TIMSS   | 15        | 35.7% |
| PIAAC   | 14        | 33.3% |
| PIRLS   | 4         | 9.5% |
| PISA    | 3         | 7.1% |
| TALIS   | 3         | 7.1% |
| ICILS   | 2         | 4.8% |
| ICCS    | 1         | 2.4% |
| **Total** | **42** | 100% |

PISA share: 3/42 = **7.1%** (corpus: 35.9%) — intentionally reduced from pilot
PIAAC gap: **resolved** — 14 studies (0% → 33.3%)

Total evidence rows: 246 (128 from v3 + 118 new)

---

## 11. Gold Freeze

File: `outputs/finetuning/data/gold_standard_full_corpus_v4.csv`
Previous versions preserved: v1 (134 rows), v2 (128 rows), v3 (128 rows)
Archive (including excluded): `outputs/finetuning/data/gold_full_archive_v4.csv`

---

## 12. Limitations

1. **Single annotator** — second-pass intra-rater reliability not yet conducted
2. **PISA underrepresented** (7.1%) relative to corpus (35.9%) — deliberate, but phase 4 may need selective PISA addition
3. **ICCS/ICILS** still low (1 and 2 studies) — limited eligible corpus
4. **Single PV** used in 3 studies — flagged; acceptable per codebook v1.1
5. **Weights not reported** in several PIAAC studies — correctly noted as not_reported

---

## 13. Fine-Tuning Readiness

**CONDITIONALLY READY**

| Condition | Status |
|-----------|--------|
| All 7 programs represented | ✅ YES |
| PIAAC gap resolved | ✅ YES |
| Major construct categories covered | ✅ YES |
| Difficult cases represented | ✅ YES |
| Multiple evidence types | ✅ YES |
| Direction reversals / null results | ✅ YES |
| Source-grounded human verification | ✅ YES |
| Second-pass intra-rater reliability | ⏳ PENDING (from 2026-10-12) |
| Train/val/test split | ⏳ NOT YET DESIGNED |
| Fine-tuning approved | ❌ NOT YET |

**Remaining blocker before fine-tuning:**
Second-pass reliability (can proceed in parallel with split design).

---

## 14. Next Phase

Phase 4 tasks:
1. Conduct second-pass intra-rater reliability (from 2026-10-12, 30% subset = ~13 studies)
2. Calculate Cohen's kappa per Tier-1 field
3. Design paper-level train/validation/test split (article-level, program-stratified)
4. Evaluate split feasibility (min N per program per split)
5. Select fine-tuning method (SFT / LoRA / few-shot evaluation)
6. Consider adding selective PISA records if construct gaps remain

---

*Phase 3C annotation complete: 2026-10-06 | GOLD_STATUS: READY_FOR_SPLIT (pending second-pass reliability)*
