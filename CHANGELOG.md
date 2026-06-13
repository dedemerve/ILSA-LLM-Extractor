# Changelog

All notable changes to this project will be documented in this file.

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
