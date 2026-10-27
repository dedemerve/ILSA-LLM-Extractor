# Changelog

All notable changes to this project will be documented in this file.

## [1.1.0] - 2026-10-25

### Changed
- TALIS pipeline (`build_talis_covariates.py`): added JRR variance estimation using TRWGT1–TRWGT100 replicate weights; SE column now populated for all four cycles (2008, 2013, 2018, 2024)
- PIAAC Cycle 2 year corrected from 2017 to 2022 in `build_icils_piaac_estimates.py` and downstream outputs
- TALIS 2008 and 2024 cycles added to covariate pipeline; output expanded from 2 to 4 cycles

### Performance (thesis run)
- Covariate records: 174 (TALIS output, post-correction)
- TALIS SE coverage: R=100 replicates across all cycles

## [1.0.0] - 2026-05-31

### Added
- Initial release of the ILSA-LLM-Extractor pipeline
- LLM-based structured extraction using GPT-5.4 nano (2026-03-17)
- Pydantic v2 schema for `ILSAArticleMetadata`
- Three-stage pipeline: Knowledge Extraction, Knowledge Synthesis, RAG-Based Agent
- Support for IEA (TIMSS, PIRLS, ICCS) and OECD (PISA, TALIS, PIAAC) documents
- Batch processing with `--resume` support for failed extractions
- Storage layer: JSON, Parquet, and SQLite outputs via `StorageManager`
- Deduplication via DOI matching and normalized title/author comparison
- 1,680 PDFs processed, 1,266 unique study records extracted
- Anti-hallucination validation rules
- Hugging Face dataset published: dedemerve/ILSA-LLM-Extractor-Dataset
- MIT License added
- CONTRIBUTING.md added

### Performance
- Classification accuracy: 99.2%
- Main findings rows: 1,893
- Confounder rows: 7,655
