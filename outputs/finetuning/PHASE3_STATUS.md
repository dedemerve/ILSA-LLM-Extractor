# Phase 3 Status — Human-Verified Gold-Standard Construction

**Date:** 2026-10-05  
**Branch:** experiment/ilsa-domain-finetuning  
**Status:** PILOT COMPLETE — Silver promotion COMPLETE — Full annotation IN PROGRESS

---

## 1. Objective

Construct a scientifically defensible human-verified gold-standard dataset from the full 1,266-record ILSA corpus for later domain-specific fine-tuning of the LLM extraction layer.

---

## 2. Full Corpus

| Metric | Count |
|--------|------:|
| Authoritative corpus | 1,266 |
| Unique by file_name | 1,266 |
| Source: `ILSA_Meta_Analysis_Dataset.xlsx` → `1_Articles_Master` | — |
| Source databases | 5 (Scopus, IEA, OECD, WoS, ilsa_survey) |
| Records in evidence matrix | 618 |
| Records with source text | 577 |

---

## 3. Silver Candidate Profile

| Pool | N | Notes |
|------|--:|-------|
| silver_verified_candidate | 526 | Source text + Tier-1/2 schema clean (341 original + 185 promoted) |
| silver_text_manual_review | 51 | Source text + non-promotable schema drift (wle/irt_theta/knn) |
| silver_only | 41 | No source text (all ilsa_survey) |
| **Total evidence-bearing** | **618** | — |

**Promotion completed 2026-10-05:** 185 of 236 manual-review records promoted to silver_verified_candidate (HIGH-confidence PV drift: rubin_rules/average_pv/all_pv → all_pvs). 51 records with non-promotable drift (wle=14, irt_theta=4, knn=19, single_imputation+mixed=14) remain as silver_text_manual_review.

**Silver ≠ gold. All 526 silver records are LLM-generated labels without human verification.**

---

## 4. Manual Review Profile

| Drift reason | N | Promoted? |
|-------------|--:|:----------|
| rubin_rules → all_pvs (pure) | 123+3(combo) | YES — all promoted |
| average_pv → all_pvs (pure) | 50+5(combo) | YES — all promoted |
| all_pv → all_pvs (pure) | 12+2(combo) | YES — all promoted |
| knn_imputation only | 19 | NO — MEDIUM confidence |
| wle only | 14 | NO — LOW confidence |
| irt_theta only | 4 | NO — LOW confidence |
| rubin_rules + knn (combo) | 3 | NO — mixed |
| average_pv + single_imputation (combo) | 3 | NO — mixed |
| rubin_rules + single_imputation (combo) | 2 | NO — mixed |
| average_pv + knn (combo) | 2 | NO — mixed |
| all_pv + single/knn (combo) | 2 | NO — mixed |

**185 records promoted 2026-10-05.** Remaining 51 require human review before promotion.

---

## 5. Program Coverage

| Program | Full corpus | Source text | Silver 341 | Gold pilot |
|---------|------------:|------------:|-----------:|-----------:|
| PISA | 358 (35.9%) | 337 (58.4%) | 183 (53.7%) | 3 (25.0%) |
| TALIS | 69 (6.9%) | 69 (12.0%) | 65 (19.1%) | 3 (25.0%) |
| PIAAC | 59 (5.9%) | 55 (9.5%) | 49 (14.4%) | 0 (0%) |
| TIMSS | 56 (5.6%) | 55 (9.5%) | 23 (6.7%) | 1 (8.3%) |
| PIRLS | 31 (3.1%) | 29 (5.0%) | 13 (3.8%) | 2 (16.7%) |
| ICCS | 15 (1.5%) | 14 (2.4%) | 8 (2.3%) | 1 (8.3%) |
| ICILS | 18 (1.8%) | 17 (2.9%) | 9 (2.6%) | 2 (16.7%) |

**Critical issue:** PIAAC silver candidates are largely OECD descriptive reports, not ML prediction studies. 0 of 3 pilot PIAAC records survived to gold_verified.

---

## 6. Annotation Codebook

**Version:** 1.0 (released 2026-10-05)  
**Location:** `outputs/finetuning/data/annotation_codebook.md`  
**Changelog:** `outputs/finetuning/audit/annotation_codebook_changelog.md`  

Key rules established in v1.0:
- ICT_HOME_ACCESS does NOT apply to teacher-level ICT in TALIS
- direction=0 in evidence matrix ≠ confirmed null result
- OECD country notes → excluded from gold
- Measurement invariance studies → excluded or descriptive
- Macroeconomic aggregate studies → excluded
- rubin_rules/average_pv → all_pvs (HIGH confidence, no gold barrier)

---

## 7. Pilot Annotation

**Records annotated:** 18 (2–3 per program, stratified)  
**Annotation method:** Source text reading (abstract + methods + results) + LLM extraction comparison  
**Annotator:** Merve Dede  

| Status | Studies | Evidence rows |
|--------|--------:|-------------:|
| gold_verified | 8 | 51 |
| gold_verified_after_adjudication | 4 | 77 |
| excluded | 5 | 5 |
| ambiguous | 1 | 1 |
| **Usable gold** | **12** | **128** |

---

## 8. Human Gold Construction

**Current gold count: 12 studies, 128 evidence rows**  
**Target gold count: 51–68 studies**  
**Remaining to annotate: ~40–56 studies**  

The annotation is ongoing. 42 more records from the 60-record sample are marked annotation_pending.

---

## 9. Annotation Reliability

| Metric | Status |
|--------|--------|
| Single annotator (Merve Dede) | Active |
| Second annotator | Not available |
| Intra-rater second pass | Pending (≥7 days washout required) |
| Program assignment agreement | ~17/18 (deterministic) |
| Canonical construct agreement | ~15/18 (3 errors corrected) |
| Direction agreement | ~12/18 (5 ambiguous) |

---

## 10. Gold Distribution

Gold pilot (12 studies) compared to silver 341:
- PISA reduced from 53.7% → 25.0%
- PIRLS raised from 3.8% → 16.7%
- ICILS raised from 2.6% → 16.7%
- PIAAC dropped to 0% (systematic exclusion of descriptive reports)

---

## 11. Remaining Limitations

1. N=12 gold studies → insufficient for fine-tuning; need 51–68 minimum
2. PIAAC gap: all pilot PIAAC candidates excluded — requires different sampling strategy
3. No second-pass intra-rater reliability (pending 7-day washout)
4. 197 promotable manual-review records not yet promoted or annotated
5. 40+ annotation-pending records in the 60-record sample need completion

---

## 12. Fine-Tuning Readiness

**Fine-tuning is NOT yet permitted.**

Blockers:
- Insufficient gold studies (12 of 51–68 target)
- No intra-rater reliability statistics
- PIAAC sampling strategy not resolved
- Train/validation/test split not designed
- Second-pass verification not conducted

---

## 13. Next Phase (Phase 4)

Phase 4 objectives:
1. Promote 197 HIGH-confidence manual-review records to silver_verified_candidate
2. Complete annotation of remaining 42 records in the 60-record sample
3. Design new PIAAC sample (journal articles only)
4. Expand to target of 51–68 gold studies
5. Conduct second-pass intra-rater verification (after 7-day washout)
6. Design train/validation/test split at article level
7. Assess fine-tuning method feasibility (SFT / LoRA / few-shot evaluation only)

---

## Integrity Checks

All 15 Phase 3 integrity checks: **PASS**  
`outputs/stage5/` unchanged  
W_j, W_jpd, LOCO folds, forecasting results: unchanged  
Gold set not used to optimize W or MAE  
Branch: experiment/ilsa-domain-finetuning — not merged  

---

*Phase 3 pilot complete: 2026-10-05 | Full annotation ongoing | Branch: experiment/ilsa-domain-finetuning*
