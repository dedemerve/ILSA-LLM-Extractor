# Phase 3.28 — Literature Search Scope and Candidate Audit

**Date:** 2026-10-20  
**Status:** Local export metadata audit and five-row independent outcome adjudication complete. Becker–Coyle numerical comparison is superseded by the Phase 3.31 reconstruction; original search execution details remain **UNRESOLVED**.
**Scope:** Prompt 1 only. No corpus or prior audit file was changed.

## Summary

- **VERIFIED:** The 1,266-record corpus is an existing coded corpus, and local saved-search inventory counts 725 extraction JSONs: 423 under Scopus and 302 under Web of Science. They are not raw database result exports.
- **VERIFIED:** Query-folder labels encode seven named ILSA program searches in each database and a nominal 2020–2026 window. They also carry the terms artificial intelligence, machine learning, deep learning, time series analysis, prediction, or correlation.
- **UNRESOLVED:** Search platform syntax/fields, actual search execution date, result count at execution, applied filters, export completeness, and screening history cannot be reconstructed from the local files inspected.
- **VERIFIED:** On independent outcome coding, EXT-001 and EXT-002 describe country-level PISA mean-score forecasts. EXT-003 predicts minimum-proficiency shares, not mean scores. MIS-001 and MIS-002 do not forecast ILSA country mean scores.
- **SUPERSEDED:** At the time of this audit, the Becker–Coyle supplement had not been retrieved. Phase 3.31 later retrieved the publisher supplement, reconstructed the included R procedure, and compared its 2022 forecasts on the common project sample. This is a reconstruction rather than a publisher-supplied prediction table; see [Phase 3.31](phase3_31_becker_coyle_and_search_provenance.md).

## Methods and evidence inspected

I checked the requested target path before creating anything; `analysis/03_data_feasibility/phase3_28_literature_scope_audit.md` did not exist. I read the candidate register and Phase 3.27, inspected the Phase 3.15 saved-search inventory manifest, report and row-level CSV, checked for raw export formats and locally stored Becker–Coyle/supplement files, and reviewed the historical prediction CSV schema. Candidate output coding below is my independent adjudication based on primary publisher or preprint descriptions, rather than copying the disposition fields in the register.

Local source files:

- `analysis/03_data_feasibility/audit/literature_search_inventory_v1/manifest.json`
- `analysis/03_data_feasibility/audit/literature_search_inventory_v1/inventory_report.json`
- `analysis/03_data_feasibility/audit/literature_search_inventory_v1/saved_search_json_inventory.csv`
- `analysis/03_data_feasibility/audit/literature_candidate_adjudication_v1.csv`
- `analysis/03_data_feasibility/phase3_27_literature_candidate_adjudication.md`
- `outputs/pisa_tabular_benchmark_v1/historical_predictions.csv`
- `outputs/scopus_batch_run.log`

## Search scope: recoverable details

### Local saved-search inventory

**VERIFIED:** Inventory manifest reports 725 JSON files under the configured `outputs/Scopus` and `outputs/Web of Science` roots, with 423 Scopus and 302 WoS files. The inventory report identifies 725 JSON rows with DOI, 665 distinct normalized DOIs across databases, 663 exact DOI matches to corpus entries, and 332 corpus DOI records absent from these saved JSONs. These figures describe the saved extraction records and DOI matching; they are not database search result counts or proof of retrieval completeness.

**VERIFIED:** The 14 query-folder labels (seven per database) combine the displayed nominal year range `2020-2026` and the terms `“artificial intelligence” OR “machine learning” OR “deep learning” OR “time series analysis” OR “prediction” OR “correlation”` with one of these program terms:

| Database | Program folder groups and local extraction JSON count |
|---|---|
| Scopus | ICCS 29; ICILS 25; PIAAC 29; PIRLS 25; PISA 188; TALIS 66; TIMSS 61 (total 423) |
| Web of Science | ICCS 4; ICILS 7; PIAAC 19; PIRLS 21; PISA 184; TALIS 25; TIMSS 42 (total 302) |

The displayed program query terms in the folder labels are the program acronym and long name: ICCS / International Civic and Citizenship Education Study; ICILS / International Computer and Information Literacy Study; PIAAC / Programme for the International Assessment of Adult Competencies; PIRLS / Progress in International Reading Literacy Study; PISA / Programme for International Student Assessment; TALIS / The Teaching and Learning International Survey; and TIMSS / Trends in International Mathematics and Science Study. The full query-folder strings are preserved in the row-level inventory CSV. Counts above are JSON extraction files per folder group, not the number of results returned by Scopus or WoS.

**VERIFIED:** The inventory report explicitly describes its scope as saved extraction JSON inventory rather than database execution/screening reconstruction. It says folder names retain nominal query text but not the execution timestamp or raw export count, and that JSON files are project extraction records rather than raw database exports. The inventory includes individual title/year/DOI and file hash fields; it does not include database filters or screening decisions.

**VERIFIED, narrowly:** `outputs/scopus_batch_run.log` contains processing log timestamps in May 2026 and reports 423/423 extraction jobs as complete. That is an extraction-run record, not evidence of the Scopus or WoS query execution date. No corresponding complete WoS raw export or execution log was located among the inspected files.

### What cannot be recovered

**UNRESOLVED:** Exact Scopus and WoS search syntax in database-native field notation; index/search fields; language, document-type, subject-area, or other filters; the precise date each query was run; hit count at query execution; whether all hits were exported; pagination/export limits; deduplication sequence; and title/abstract screening and exclusion reasons. The raw Scopus/WoS export formats (for example CSV/RIS/BibTeX or native export files) were not found in the requested directories. The JSONs are enriched extraction records and cannot stand in for those exports.

**VERIFIED, as a descriptive inference only:** Available local data support a nominal search frame of Scopus and Web of Science; seven ILSA programs; terms around AI/ML/deep learning/time series/prediction/correlation; and a labeled 2020–2026 publication window. This is the frame implied by folder names, not a verified reconstruction of the executed query. The corpus’ 1,266-row count is a coded corpus count and cannot be treated as the number of database results.

## Independent adjudication of all five candidate rows

Operational coding question: does the source report or present a forecast/projection of a PISA, TIMSS, PIRLS (including PIRLS country average) **country-level mean score**? A country-level rate/proportion derived from score thresholds is not a mean score. This outcome coding is separate from original corpus membership and search eligibility.

| Candidate | Independent country mean-score forecast code | Adjudication |
|---|---|---|
| EXT-001 — Becker & Coyle (2026), PISA 2022 | **YES — PISA** | **VERIFIED.** The publisher describes per-country PISA mean scores from 2000–2022 and forecasts 2022 with trend regressions using earlier PISA waves; the methods describe linear and quadratic fits and the choice between them. This qualifies as a country mean-score forecast, irrespective of whether the study meets the project’s inclusion/search rules. The linked supplementary material is stated to contain the trend estimates and tests. [Publisher article](https://www.frontiersin.org/journals/education/articles/10.3389/feduc.2026.1734988/full). |
| EXT-002 — Kaplan et al. (2025), PISA mathematics | **YES — PISA, with an estimand qualification** | **VERIFIED.** The publisher-hosted paper models country-level PISA mathematics proficiency and presents one-cycle-ahead PISA 2025 projections by sex. This is a country-level forecast of mean performance, but the published projections are sex-stratified and should not be equated to an unstratified country-by-domain mean forecast without additional aggregation detail. [Cambridge publisher PDF](https://www.cambridge.org/core/services/aop-cambridge-core/content/view/D85FD94C994DB13028209C06CA323169/S003331232500002Xa.pdf/stacking_models_of_growth_a_methodology_for_predicting_the_pace_of_progress_to_the_education_sustainable_development_targets_using_international_large-scale_assessments.pdf). |
| EXT-003 — Kaplan et al. (arXiv:2511.06107) | **NO — threshold-share projection** | **VERIFIED.** The outcome is the country-level percentage attaining minimum proficiency in reading and mathematics. It is a projected country-level educational indicator based partly on PISA, but not a PISA country mean score. [arXiv abstract](https://arxiv.org/abs/2511.06107). |
| MIS-001 — Martín Ezpeleta et al. (2024) | **NO** | **VERIFIED.** The publisher record describes a creativity/metaphor study of Spanish teacher trainees; PISA is contextual motivation and is not a forecast outcome or data source for a country mean-score prediction. [Publisher record](https://revistas.ucm.es/index.php/RCED/en/article/view/86027). |
| MIS-002 — van Rijn & Ali (2023) | **NO** | **VERIFIED.** The publisher chapter is a psychometric speed-accuracy response model illustrated with listening-test data. A PIRLS query-folder label does not make it a PIRLS country-score forecast. [Springer chapter](https://link.springer.com/chapter/10.1007/978-3-031-27781-8_11). |

**UNRESOLVED:** Whether EXT-001 or EXT-002 was eligible under the original database search and screening protocol. Their absence from corpus matches cannot establish ineligibility or search omission because search execution timing, full exports, and inclusion criteria are unavailable. These source findings also do not change the fixed corpus or candidate register.

## Becker–Coyle supplement and historical-prediction comparison

**VERIFIED:** Becker & Coyle state that they modeled per-country PISA means and other score summaries, forecast 2022 from PISA waves through 2018, and supply linear/quadratic trends and significance tests in linked supplementary material. The article’s abstract reports forecast deviations for 64 participating countries. The relevant claims and supplement link appear in the [publisher article](https://www.frontiersin.org/journals/education/articles/10.3389/feduc.2026.1734988/full).

**VERIFIED:** The project file `outputs/pisa_tabular_benchmark_v1/historical_predictions.csv` has fields `target_cycle`, `domain`, `country_iso3`, `model`, `true_score`, and `prediction` (among others), so it is structurally suitable for a country/domain/cycle comparison if the supplement contains comparable 2022 values and country identifiers.

**SUPERSEDED by Phase 3.31:** The publisher supplement was subsequently obtained. It contains source R scripts and historical score data rather than a ready-made country prediction table. The R procedure was reconstructed, country and domain keys aligned, and comparison statistics produced on the 60-country common set for 2022. The result and method-compatibility limitations are in [Phase 3.31](phase3_31_becker_coyle_and_search_provenance.md).

## Limitations and next evidence needed

1. **UNRESOLVED — search protocol:** Recover original database export files or saved-search histories with query strings, search dates, filters, and hit counts; plus deduplication and screening logs. The current folder labels and extraction inventory do not establish those facts.
2. **UNRESOLVED — corpus coverage:** The inventory’s DOI intersection is not a recall estimate. It cannot determine how many eligible studies were missed, especially for records without DOI or absent from the saved extraction JSONs.
3. **UNRESOLVED — independent human second review:** This report supplies a separate source-based outcome adjudication, but does not constitute a second human reviewer. The CSV’s `second_reviewer_status` remains pending.
4. **COMPLETE — Becker–Coyle comparison:** See Phase 3.31 for the reconstructed procedure, common-sample metrics, and method-comparability qualification.

## Source trail

Local numerical and file claims are traceable to the listed local files and paths above. External candidate source claims use the primary publisher or preprint pages linked in the candidate table. Search-scope counts refer specifically to `inventory_report.json` and its row-level inventory, not original database result exports.
