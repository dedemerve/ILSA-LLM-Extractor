# Phase 2 Status — Source Recovery (Full Corpus)

**Date:** 2026-10-05  
**Branch:** experiment/ilsa-domain-finetuning  
**Status:** IN PROGRESS

---

## Phase 2 Goal

Obtain source texts for all viable fine-tuning candidates across the full 1,266-record corpus.

---

## Current Source Recovery State

| Metric | Count |
|--------|------:|
| Total corpus records | 1,266 |
| PDF available (on-disk) | 1,211 |
| PDF missing | 55 |
| Text extracted (pdftotext) | 10 |
| Text extraction pending | 1,201 |

---

## Priority Queue

### Priority A2 — ilsa_survey records (55 records, PDF missing)

These 55 records use the full Pydantic extraction schema (all Tier-1/2/3 fields), making them the highest-quality fine-tuning candidates. However, their source PDFs are not in the local `data/` directory or the three external PDF directories.

**Action required:** Locate PDFs for ilsa_survey article numbers 1–55 (numbered filenames in `outputs/ilsa_survey_articles/json/`).

Note: 10 PDFs were previously matched from `data/` using stem comparison (articles 1–10). The remaining 45 need to be sourced from institutional access or author repositories.

### Priority B1 — Evidence-bearing records with PDF (577 records)

577 records are in the evidence matrix AND have PDFs available in external directories:
- `/Users/mrved/Desktop/ILSA Documents LLM Extraction/` (684 PDFs)
- `/Users/mrved/Desktop/ILSA Papers & Reports LLM /` (1,619 PDFs)
- `/Users/mrved/Desktop/ILSA Documents copy/` (1,015 PDFs)

Text extraction using `pdftotext` is pending for all 577.

**Action required:** Run batch pdftotext extraction for all 577 B1-priority PDFs.

### Priority C1 — Non-evidence records with PDF (634 records)

634 records have PDFs but no evidence matrix presence (mostly IEA reports and non-ML studies). Lower priority for fine-tuning.

---

## Previously Recovered (Carried Forward)

10 text files from previous Phase 1 work:
- Located in `outputs/finetuning/source_recovery/texts/`
- Articles 1–10 of ilsa_survey (numbered prefix format)
- Total: ~912 KB
- All remain valid

---

## Next Steps

1. **Run batch pdftotext** on all 577 B1-priority PDFs → `outputs/finetuning/source_recovery/texts/`
2. **Update full_source_inventory.csv** with extraction results
3. **Re-run silver candidate construction** against full 618-record evidence corpus
4. **Report final text-available count** before proceeding to Phase 3

---

*Generated: 2026-10-05 | Branch: experiment/ilsa-domain-finetuning*
