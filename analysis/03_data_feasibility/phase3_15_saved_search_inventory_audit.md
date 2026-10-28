# Phase 3.15 — Saved Literature Search Inventory Audit

**Date:** 2026-10-15  
**Status:** Local Scopus/Web of Science extraction inventory built and hash-checked; search execution and screening provenance remain unresolved.  
**Script:** `scripts/audit_literature_search_inventory.py`  
**Outputs:** `analysis/03_data_feasibility/audit/literature_search_inventory_v1/`

## Purpose

This audit inventories the search-related JSON records currently saved under `outputs/Scopus/` and `outputs/Web of Science/` and exact-matches normalized DOIs to the fixed `analysis/02_prediction_taxonomy/study_classification.csv` corpus. It does not change the corpus, raw JSON records, method-frequency counts, model panels, or experiment results.

## Verified inventory

| Evidence | Scopus | Web of Science | Combined |
|---|---:|---:|---:|
| Saved JSON files | 423 | 302 | 725 |
| JSONs with a DOI | 423 | 302 | 725 |
| Distinct normalized DOI within source | 400 | 295 | — |
| Repeated DOI rows within source | 23 | 7 | — |
| Distinct DOI across both sources | — | — | 665 |
| Distinct source DOI matched in the 1,266-row corpus | — | — | 663 |
| Distinct source DOI not matched in the corpus | — | — | 2 |
| Distinct corpus DOI absent from these saved search JSONs | — | — | 332 |

The two unmatched source DOI values are `10.5209/rced.86027` and `10.1007/978-3-031-27781-8_11`. The row-level file records titles, years, query-folder names, relative paths, file hashes, and exact DOI-match results. Scopus contains 400 distinct DOI records and all 400 are represented in the fixed corpus; WoS has 295 distinct DOIs, of which 293 occur in the corpus. The 332 corpus DOIs absent from these two saved sets are not automatically exclusions or errors: the corpus also contains IEA/OECD material and records sourced outside these specific JSON directories.

## Query-folder evidence

Seven program-named folders are present for each source (ICCS, ICILS, PIAAC, PIRLS, PISA, TALIS, TIMSS). Their folder names preserve a nominal query label containing a `2020-2026` range and terms for artificial intelligence, machine learning, deep learning, time-series analysis, prediction, or correlation, combined with a program name. Several long folder names are truncated by filesystem path limits. These labels establish that a query description was saved in the directory name; they do **not** establish the exact database syntax, actual execution date, database filters, total platform result counts, export completeness, or search updates.

The files are article-extraction JSONs with project metadata and extracted study content, not raw Scopus/WoS exports. No RIS, BibTeX, or EndNote export was found in the workspace scan. No reproducible title/abstract screening log, full-text exclusion log, or reason-per-record decision table was identified in the current search inventory. Therefore this audit improves file-level provenance but does not make the 1,266-record corpus a reproducible systematic-search sample by itself.

## Reproducibility and integrity

- Inventory rows are sorted by source and relative path; every saved JSON has a SHA-256 hash.
- The exact-DOI join uses normalized lowercase DOI text and strips DOI URL prefixes and trailing punctuation.
- The corpus file hash and output hashes are in `audit/literature_search_inventory_v1/manifest.json`.
- The script refuses to overwrite an existing output directory.
- The run report is in `audit/literature_search_inventory_v1/inventory_report.json`.

This is an exact DOI audit. It does not perform fuzzy title matching, prove which JSON records were screened into the corpus, or classify the two unmatched DOI records as eligible/ineligible.

## Consequences for the study

1. Retain current algorithm-frequency outputs only as descriptive counts for the fixed coded corpus; do not present them as a complete census of ILSA forecasting methods.
2. Do not use the nominal 2020-2026 folder labels as proof of the original search date or full search protocol.
3. Do not change the locked 1,266-record corpus or frequency denominators based only on this join. First recover original inclusion criteria and inspect the two unmatched source records and external forecasting candidates.
4. Complete a new, date-stamped search protocol and screening log if literature-completeness claims are required. A separately versioned corpus release should be created only after eligibility and deduplication rules are applied and borderline cases independently checked.

## Next literature-audit actions

1. Retrieve raw database exports or query-history screenshots/logs if available; record exact fields, filters, dates, and result counts.
2. Inspect the two DOI-unmatched records and resolve their corpus disposition.
3. Run a targeted search for national/country PISA, TIMSS, and PIRLS mean-score forecasting, including traditional trend regression and terms such as forecast, projection, extrapolation, future cycle, and held-out cycle.
4. Screen all country-system predictive records and plausible false negatives outside the 28 heuristic flags against the project's strict forecast definition.
5. Double-code borderline studies; report agreement and adjudications.
6. Decide whether any newly verified study meets the recovered search scope. If yes, issue a new corpus release and recalculate denominators; otherwise retain an explicit scope-limited conclusion.

## Project-wide status after this audit

The predictive experiment is substantially executed: historical tabular baselines, TimesFM 3.0 zero-shot, TimesFM 2.5 ILSA LoRA with a matched zero-shot comparator, and an outcome-exposed PISA 2025 retrospective comparison are available. The saved 2025 country means in the 66-country, three-domain benchmark are independently checked against OECD tables. Remaining material work includes literature-search completeness, final cross-artifact/code audit, score-estimate uncertainty propagation where feasible, consolidated reporting, and an outcome-independent protocol for a future PISA cycle. PISA 2025 cannot be relabeled as a blind final test.
