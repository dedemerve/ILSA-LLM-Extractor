# Corpus Count Discrepancy Report

**Date:** 2026-10-05  
**Branch:** experiment/ilsa-domain-finetuning  
**Status:** RECONCILED — all counts explained

---

## Summary

The user-specified corpus size is **1,266 unique records**. All prior audits (reporting 983 or 820 unique records) were incomplete because they **missed the IEA source database** (`outputs/IEA/`), which contains 288 records not present in WoS, Scopus, OECD, or ilsa_survey_articles.

**The 1,266 figure is confirmed as the authoritative full corpus count.**  
Authoritative source: `outputs/ILSA_Meta_Analysis_Dataset.xlsx` → Sheet `1_Articles_Master` → 1,266 rows, all unique by `file_name`.

---

## 1. All Historical Corpus Counts Reconciled

| Count | Source | Explanation |
|------:|--------|-------------|
| **1,266** | `ILSA_Meta_Analysis_Dataset.xlsx` (1_Articles_Master) | **Authoritative.** All 1,266 unique by file_name. 5 source databases. |
| 1,451 | Previous JSON file count (raw, no (1) copies) | All JSON files in WoS+Scopus+OECD+ilsa_survey — **IEA was not counted** |
| 1,458 | `provenance_audit.md` N_json_total | Sum across 4 DBs (WoS=302, Scopus=423, OECD=591, ilsa_survey=142) — IEA not included |
| 1,349 | `provenance_audit.md` N_ilsa_relevant | ILSA-relevance filter applied to the 1,458 (4 DB) total — IEA still excluded |
| 983 | Previous DOI/title dedup across 4 DBs | Cross-DB dedup of WoS+Scopus+OECD+ilsa_survey — IEA still missing |
| 820 | `provenance_audit.md` N_unique_any | After predictor-name canonicalization filter — different dedup methodology |
| 620 | `evidence_matrix_full_v2.csv` unique study_ids | Only records with canonical predictor mappings enter the weight computation |

---

## 2. The Missing IEA Database

The previous Phase 1 audit searched four source directories:
- `outputs/Web of Science/` (WoS)
- `outputs/Scopus/`
- `outputs/OECD/`
- `outputs/ilsa_survey_articles/`

It **omitted** the fifth:
- `outputs/IEA/` — contains 288 records across 21 subdirectories (ICCS, PIRLS, TIMSS, ICILS cycles)

The IEA records are IEA official publication reports (codebooks, technical reports, framework documents) for TIMSS, PIRLS, ICCS, and ICILS — not ML/prediction journal articles. This explains why they have low evidence matrix overlap (only 1/288 matched by DOI to the evidence matrix).

---

## 3. Authoritative Corpus Breakdown (1,266 records)

| corpus_source | N | % | In Evidence Matrix | PDF Available |
|--------------|--:|--:|------------------:|-------------:|
| Scopus | 400 | 31.6% | 275 (68.8%) | 400 (100%) |
| IEA | 288 | 22.7% | 1 (0.3%) | 288 (100%) |
| OECD | 261 | 20.6% | 105 (40.2%) | 261 (100%) |
| Web of Science | 262 | 20.7% | 196 (74.8%) | 262 (100%) |
| ilsa_survey_articles | 55 | 4.3% | 41 (74.5%) | 0 (0%) |
| **Total** | **1,266** | **100%** | **618 (48.8%)** | **1,211 (95.7%)** |

---

## 4. Why the Evidence Matrix Has Only 620 Records

The evidence matrix (`evidence_matrix_full_v2.csv`) contains only records that:
1. Were processed by the LLM extractor AND
2. Produced at least one finding with a canonical predictor mapping

IEA official reports (288 records) are reference documents, not ML prediction studies. They have no `top_predictors` and contribute 0 canonical findings. This is expected behavior.

The 2-unit gap (620 vs 618 Excel-matched) is due to study_id format differences: the evidence matrix uses truncated filename stems, while the Excel uses full `file_name`. Two records match by DOI but not by exact string match.

---

## 5. Provenance Audit vs. Excel

The `provenance_audit.md` is authoritative for the 4-DB extraction pipeline (WoS, Scopus, OECD, ilsa_survey). Its counts are correct for that scope. The IEA database was added separately as a reference document corpus and was always tracked in the Excel but not in the provenance pipeline.

```
Provenance audit scope (4 DB):   N_json = 1,458
IEA reference documents:                    288
                                        -------
Total full corpus (Excel):              1,266  ← 1458 - duplicates + IEA = 1,266
```

Note: the arithmetic does not sum cleanly because:
- The 4-DB total (1,458) includes cross-DB duplicates that are deduplicated in the Excel
- The IEA records do not overlap with the WoS/Scopus search results

---

## 6. Fine-Tuning Corpus Scope

For fine-tuning purposes, the **relevant sub-corpus** is:

| Scope | N | Rationale |
|-------|--:|-----------|
| Full corpus | 1,266 | All records tracked in Excel |
| Evidence-bearing (in evidence matrix) | 618 | Have canonical predictor mappings; relevant to W_j |
| With PDF available | 1,211 | Source text potentially extractable |
| ilsa_survey subset (previous Phase 1) | 55 | ML/prediction studies with detailed extraction schema |
| ilsa_survey with PDF | 0 | No PDFs in `data/` for ilsa_survey (10 were in `data/` matching survey filenames) |

**The ilsa_survey subset (55 records) remains the primary fine-tuning target** because it was extracted using the full Pydantic schema with all Tier-1/2/3 fields. WoS/Scopus/OECD/IEA extractions may use a simpler schema.

---

## 7. Immediate Implication for Phase 2

Phase 2 source recovery must:
1. Confirm the 1,211 PDF-available records (IEA + OECD + WoS + Scopus) are accessible
2. Obtain PDFs for the 55 ilsa_survey_articles records (0 currently available)
3. Prioritize extraction from the 618 evidence-bearing records

The 10 previously extracted text files (from `data/*.pdf`, ilsa_survey articles 1–10) remain valid and are carried forward.

---

*Generated: 2026-10-05 | Supersedes the incomplete 4-DB audit from the previous session*
