# Manual-Review Profile (236 Records)

**Date:** 2026-10-05  
**Source:** outputs/finetuning/data/full_corpus_silver_candidates.csv (gold_status = silver_text_manual_review)

---

## Summary

236 records have source text available but require manual schema review before gold annotation.

---

## Review Reason Distribution

| Reason | N | % | Normalizable? |
|--------|--:|--:|:-------------|
| pv_drift:rubin_rules | 128 | 54.2% | HIGH confidence — rubin_rules → all_pvs |
| pv_drift:average_pv | 55 | 23.3% | HIGH confidence — average_pv → all_pvs |
| pv_drift:all_pv | 14 | 5.9% | HIGH confidence — all_pv → all_pvs |
| missing_drift:knn_imputation | 25 | 10.6% | MEDIUM — knn_imputation → multiple_imputation |
| pv_drift:wle | 14 | 5.9% | LOW — WLE is a specific latent score method |
| missing_drift:single_imputation | 8 | 3.4% | MEDIUM — single_imputation → mean_imputation |
| pv_drift:irt_theta | 4 | 1.7% | LOW — IRT theta ≠ standard PV procedure |
| Multiple drift reasons | 12 | 5.1% | Varies |

---

## Promotable Records

**Key finding:** 128 + 55 + 14 = 197 records have `rubin_rules`, `average_pv`, or `all_pv` as their ONLY review trigger. The normalization script (`normalize_training_labels.py`) classifies these as HIGH confidence normalizations:

- `rubin_rules` → `all_pvs`
- `average_pv` → `all_pvs`
- `all_pv` → `all_pvs`

**These 197 records can be promoted to silver_verified_candidate** after applying the normalization with manual spot-check of ~10% of cases. This would expand the annotation candidate pool from 341 to ~538 records.

---

## ILSA Program Distribution

| Program | N studies | % |
|---------|----------:|--:|
| PISA | 156 | 66.1% |
| TIMSS | 38 | 16.1% |
| PIRLS | 20 | 8.5% |
| PIAAC | 10 | 4.2% |
| ICILS | 8 | 3.4% |
| TALIS | 8 | 3.4% |
| ICCS | 6 | 2.5% |

**Note:** The manual-review set is more TIMSS/PIRLS-heavy than the silver-verified set (PISA 53.7% vs 66.1%). This is because rubin_rules/average_pv are common in studies that properly use TIMSS/PIRLS PV procedures (more likely to use all PVs correctly than ML papers that often use single PV).

---

## Implication for Fine-Tuning

Promoting the 197 high-confidence records would:
1. Expand the gold candidate pool substantially
2. Improve TIMSS/PIRLS representation
3. Provide more examples of correct PV handling (all_pvs) vs incorrect (single_pv)

**Recommended action:** Apply normalization and promote rubin_rules/average_pv/all_pv records, then annotate from expanded pool.

---

*Generated: 2026-10-05 | Phase 3 | Branch: experiment/ilsa-domain-finetuning*
