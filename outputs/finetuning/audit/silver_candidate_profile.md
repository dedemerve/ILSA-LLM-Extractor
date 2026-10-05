# Silver-Verified Candidate Profile

**Date:** 2026-10-05  
**Source:** outputs/finetuning/data/full_corpus_silver_candidates.csv  
**N = 341** silver_verified_candidate records

---

## 1. Corpus Source Distribution

| corpus_source | N | % |
|--------------|--:|--:|
| Scopus | 150 | 44.0% |
| OECD | 105 | 30.8% |
| Web of Science | 85 | 24.9% |
| IEA | 1 | 0.3% |

**Note:** IEA records are largely technical reports without ML prediction findings; only 1 passes Tier-1/2 checks.

---

## 2. ILSA Program Distribution

| Program | N studies | % of 341 |
|---------|----------:|--------:|
| PISA | 183 | 53.7% |
| TALIS | 65 | 19.1% |
| PIAAC | 49 | 14.4% |
| TIMSS | 23 | 6.7% |
| PIRLS | 13 | 3.8% |
| ICILS | 9 | 2.6% |
| ICCS | 8 | 2.3% |
| Multi-program | 8 | 2.3% |

**PISA dominance: 53.7%.** Gold construction must explicitly counter this bias.

---

## 3. Publication Year

| Period | N |
|--------|--:|
| 2006–2015 | 42 |
| 2016–2020 | 45 |
| 2021–2022 | 66 |
| 2023–2024 | 86 |
| 2025–2026 | 93 |
| Unknown | 9 |

Recency bias: ~55% published 2021 or later.

---

## 4. Publication Type

| Type | N | % |
|------|--:|--:|
| Journal article | 232 | 68.0% |
| Report | 107 | 31.4% |
| Conference | 1 | 0.3% |
| Not reported | 1 | 0.3% |

---

## 5. Top Canonical Constructs (by evidence row count)

| Construct | Evidence rows | Studies |
|-----------|-------------:|--------:|
| GENDER | 667 | ~150 |
| SES_COMPOSITE | 522 | ~140 |
| ICT_HOME_ACCESS | 433 | ~90 |
| UNMATCHED | 185 | ~80 |
| SELF_CONCEPT | 134 | ~50 |
| LANGUAGE_BACKGROUND | 108 | ~40 |
| PARENTAL_EDUCATION | 89 | ~35 |
| IMMIGRANT_STATUS | 87 | ~35 |
| GDP_EXPENDITURE | 87 | ~30 |
| DISCIPLINARY_CLIMATE | 68 | ~25 |

---

## 6. Schema Fields

### Plausible Values Handling

| Value | N | % |
|-------|--:|--:|
| not_applicable | 308 | 90.3% |
| single_pv | 31 | 9.1% |
| not_reported | 2 | 0.6% |

All records have Tier-1/2 clean PV handling (no rubin_rules, wle, irt_theta, knn_imputation).

### Missing Data Handling

| Value | N | % |
|-------|--:|--:|
| not_reported | 231 | 67.7% |
| multiple_imputation | 63 | 18.5% |
| listwise_deletion | 37 | 10.9% |
| mean_imputation | 8 | 2.3% |
| pairwise_deletion | 2 | 0.6% |

---

## 7. Methodological Complexity

| Indicator | N |
|-----------|--:|
| Multi-program studies | 8 |
| Has UNMATCHED construct | 28 |
| N constructs > 3 per study (median) | median = 3 |
| Source text > 100 KB | ~50 |
| Source text 10–100 KB | ~270 |

---

## 8. Rare Categories and Class Imbalance Risks

- **ICCS (2.3%)**: only 8 studies — severely underrepresented
- **ICILS (2.6%)**: only 9 studies — severely underrepresented  
- **PIRLS (3.8%)**: limited representation
- **IEA records (0.3%)**: essentially absent
- **PIAAC (14.4%)**: mostly OECD working papers — many may be descriptive, not ML prediction

**Critical finding from pilot:** Several OECD/PIAAC records use placeholder predictor entries ("See outcome_summary...") indicating they are descriptive reports, not ML prediction studies. These inflate the silver count but must be excluded during annotation.

---

## 9. Duplicate and Near-Duplicate Risk

- All 341 records are unique by DOI
- Some studies may cover the same dataset/population across different analyses
- TIMSS macro-economic studies (e.g., Hanushek & Woessmann type) are substantively different from individual-level ML prediction studies

---

## 10. Source Text Lengths

- min: ~12 KB, max: ~2.9 MB, median: ~84 KB
- All 341 records have source text ≥ 10 KB (usable)

---

*Generated: 2026-10-05 | Phase 3 | Branch: experiment/ilsa-domain-finetuning*
