# Full Corpus Inventory

**Date:** 2026-10-05  
**Branch:** experiment/ilsa-domain-finetuning  
**Authoritative source:** `outputs/ILSA_Meta_Analysis_Dataset.xlsx` → `1_Articles_Master`

---

## 1. Corpus Overview

| Metric | Count |
|--------|------:|
| **Total unique records** | **1,266** |
| Unique by file_name | 1,266 |
| Unique by DOI (non-UNKNOWN) | 995 |
| Records with UNKNOWN DOI | 271 |
| Records in evidence matrix | 618 |
| Records with PDF available | 1,211 |
| Records with PDF missing | 55 |

---

## 2. Per-Source Breakdown

| corpus_source | N records | % | In Evidence Matrix | PDF Available | PDF Missing |
|--------------|----------:|--:|------------------:|-------------:|------------:|
| Scopus | 400 | 31.6% | 275 (68.8%) | 400 | 0 |
| IEA | 288 | 22.7% | 1 (0.3%) | 288 | 0 |
| OECD | 261 | 20.6% | 105 (40.2%) | 261 | 0 |
| Web of Science | 262 | 20.7% | 196 (74.8%) | 262 | 0 |
| ilsa_survey_articles | 55 | 4.3% | 41 (74.5%) | 0 | 55 |
| **Total** | **1,266** | **100%** | **618 (48.8%)** | **1,211 (95.7%)** | **55 (4.3%)** |

---

## 3. Recovery Priority Classification

For fine-tuning purposes, records are classified by recovery priority:

| Priority | Label | N | Description |
|----------|-------|--:|-------------|
| A2 | survey_pdf_missing | 55 | ilsa_survey records — full Pydantic schema, PDF not in local directories |
| B1 | evidence_pdf_available | 577 | In evidence matrix + PDF exists — highest value for cross-DB fine-tuning |
| C1 | nonevidence_pdf_available | 634 | PDF exists but not in evidence matrix — lower priority |

**Note:** No records fall into A1 (survey + pdf available) in the on-disk `data/` directory. The 10 previously recovered PDFs were matched from `data/*.pdf` to survey file_names via stem comparison. That data/ directory has the ilsa_survey filenames but they are named differently (numbered prefix format) vs. the Excel file_names (unnumbered format).

---

## 4. IEA Source — Special Note

The 288 IEA records are **official IEA technical publications** (cycle reports, codebooks, framework documents) for TIMSS, PIRLS, ICCS, and ICILS. They were extracted using the LLM pipeline but contain no ML prediction findings. Only 1 IEA record appears in the evidence matrix (by DOI match), and that match is incidental.

For fine-tuning purposes, IEA records are **not candidates** for the extraction fine-tuning task — they would produce negative examples (no predictors, no canonical mappings). They may be useful for document-type classification.

---

## 5. Publication Type Distribution

| publication_type | N | % |
|-----------------|--:|--:|
| journal | 703 | 55.5% |
| report | 538 | 42.5% |
| book_chapter | 14 | 1.1% |
| conference | 10 | 0.8% |
| Not reported | 1 | 0.1% |

---

## 6. Evidence Matrix Cross-Reference

The evidence matrix (`outputs/stage5/evidence_matrix_full_v2.csv`) contains 620 unique study_ids contributing to canonical predictor weights W_j and W_{j,p,d}. Of these, 618 are matched by DOI to the Excel master (2 discrepancy due to study_id format difference — truncated filename vs. full file_name).

The 1,266 - 618 = 648 records **not in the evidence matrix** are:
- 288 IEA reference documents (no ML findings)
- IEA/OECD/WoS/Scopus reports with no canonical predictor extraction
- Records failing ILSA-relevance or canonicalization filters

---

## 7. Silver-Standard Candidate Implications

The previous Phase 1 audit identified 9 silver-standard candidates from the 55-record ilsa_survey subset. Protocol 3 requires expanding the search to all 1,266 records.

**Revised silver-standard construction will target:**
1. All 618 evidence-bearing records with PDF available (577 records, priority B1)
2. All 55 ilsa_survey records once PDFs are obtained (priority A2)

The 9 previously identified silver candidates from the 55-record ilsa_survey subset remain valid — they are a proper subset of the full corpus. They must not be reported as the complete silver-candidate set.

---

## 8. Detailed File

Full per-record inventory: `outputs/finetuning/audit/full_corpus_inventory.csv` (1,266 rows)  
Source recovery inventory: `outputs/finetuning/source_recovery/full_source_inventory.csv` (1,266 rows)

---

*Generated: 2026-10-05 | Protocol 3 compliance | Branch: experiment/ilsa-domain-finetuning*
