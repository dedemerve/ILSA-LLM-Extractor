---
license: cc-by-4.0
language:
- en
task_categories:
- feature-extraction
- token-classification
annotations_creators:
- machine-generated
multilinguality:
- monolingual
source_datasets:
- original
tags:
- ILSA
- PISA
- TIMSS
- PIRLS
- TALIS
- PIAAC
- ICCS
- educational-assessment
- large-scale-assessment
- LLM
- RAG
- metadata-extraction
- systematic-review
- machine-learning
pretty_name: ILSA LLM Extractor Dataset
size_categories:
- 10K<n<100K
configs:
- config_name: articles_master
  data_files: data/processed/articles_master.parquet
- config_name: findings
  data_files: data/processed/findings.parquet
- config_name: confounders
  data_files: data/processed/confounders.parquet
- config_name: articles_full
  data_files: data/processed/articles_full.parquet
- config_name: findings_full
  data_files: data/processed/findings_full.parquet
- config_name: confounders_full
  data_files: data/processed/confounders_full.parquet
- config_name: canonical_codebook
  data_files: data/reference/canonical_codebook.parquet
- config_name: meta_analysis
  data_files: data/meta_analysis/train-00000-of-00001.parquet
- config_name: knowledge_synthesis
  data_files: data/knowledge_synthesis/train-00000-of-00001.parquet
- config_name: codebook
  data_files: data/codebook/train-00000-of-00001.parquet
- config_name: semantic_synthesis
  data_files: data/semantic_synthesis/train-00000-of-00001.parquet
- config_name: policy_taxonomy
  data_files: data/policy_taxonomy/train-00000-of-00001.parquet
---

# ILSA LLM Extractor Dataset

## Dataset Description

This dataset contains structured metadata automatically extracted from **1,756 peer-reviewed articles and reports** covering International Large-Scale Assessments (IEA: TIMSS, PIRLS, ICCS; OECD: PISA, TALIS, PIAAC). The extraction pipeline combines PDF parsing, LLM-based structured extraction, and RAG-based synthesis.

**Pipeline stages:**
- **Stage 1:** LLM-based structured extraction producing 1,266 unique deduplicated study records
- **Stage 2:** Knowledge synthesis — terminology alignment, method taxonomy mapping, variable standardization
- **Stage 3:** RAG-based analytical agent grounded in 1,266 studies

**System-level metrics:** 1,756 PDFs processed | 1,266 unique records | 2,128 findings | 8,336 confounders

---

## Dataset Contents

### Processed Tables (`data/processed/`) — Recommended for analysis

| File | Description | Rows | Columns |
|------|-------------|-----:|-------:|
| `articles_master.parquet` | Core article metadata — deduplicated, enriched (CLEAN version) | 1,266 | 37 |
| `findings.parquet` | Main findings per article — target variables, predictors, metrics | 2,128 | 16 |
| `confounders.parquet` | Confounders and covariates identified per study | 8,336 | 11 |
| `articles_full.parquet` | All articles including non-peer-reviewed (full pipeline output) | 1,756 | 23 |
| `findings_full.parquet` | Findings from full corpus | 2,552 | 13 |
| `confounders_full.parquet` | Confounders from full corpus | 9,833 | 9 |

### Reference (`data/reference/`)

| File | Description | Rows |
|------|-------------|-----:|
| `canonical_codebook.parquet` | Canonical variable category definitions with operational definitions | 39 |

### Raw JSON Extractions (`data/raw/`)

Per-article LLM extraction outputs organized by source corpus:

| Folder | Source | Files |
|--------|--------|------:|
| `raw/scopus/` | Scopus database | 423 |
| `raw/oecd/` | OECD iLibrary | 591 |
| `raw/iea/` | IEA Data Repository | 308 |
| `raw/wos/` | Web of Science | 302 |
| `raw/survey/` | ILSA survey articles | 132 |
| **Total** | | **1,756** |

---

## Full Column Descriptions

### `articles_master.parquet` (1,266 rows × 37 columns)

#### Bibliographic Fields

| Column | Type | Null% | Description |
|--------|------|------:|-------------|
| `file_name` | string | 0% | Source PDF filename. Primary unique identifier. |
| `doi` | string | 21% | Digital Object Identifier. Null for reports without DOI. |
| `title` | string | 1% | Article or report title as extracted from PDF. |
| `authors` | string | 33% | Author list. Null for institutional/anonymous reports. |
| `year` | float64 | 5% | Publication year (1990–2025). |
| `publication_type` | string | 0% | One of: `journal`, `report`, `book_chapter`, `conference`. |
| `source_category` | string | 0% | One of: `peer_reviewed_research`, `technical_report`, `methodology_paper`, `review_article`. |
| `venue` | string | 3% | Journal name or publisher (e.g. "Educational Psychology Review"). |
| `corpus_source` | string | 0% | Database of origin: `Scopus`, `IEA`, `Web of Science`, `OECD`, `ilsa_survey_articles`. |

#### Survey Methodology Fields

| Column | Type | Null% | Description |
|--------|------|------:|-------------|
| `student_weights_used` | float64 | 2% | 1.0 = study applied ILSA sampling weights (e.g. W_FSTUWT). 0.0 = not applied. |
| `replicate_weights_used` | float64 | 31% | 1.0 = BRR/JK replicate weights used for variance estimation. 0.0 = not used. Null = not determinable. |
| `weight_variable_name` | string | 57% | Specific weight variable name reported (e.g. `W_FSTUWT`, `TOTWGT`, `SENWT`). |
| `weight_fields_interpretation` | string | 0% | LLM-generated explanation of how weights were or were not applied in the study. |
| `plausible_values_handling` | string | 0% | How plausible values were handled. Values: `rubin_rules` (correct pooling across 5 PVs), `average_pv` (averaged — methodologically suboptimal), `single_pv` (single draw — incorrect), `all_pv` (all analyzed separately), `wle` (Warm's likelihood estimate), `irt_theta` (IRT-based score), `not_applicable`, `not_reported`. |
| `pv_correct` | boolean | 0% | Methodological quality flag for PV handling. `True` = correct (`rubin_rules`, `all_pv`, `not_applicable`, `wle`, `irt_theta`). `False` = incorrect (`single_pv`, `average_pv` — 104 studies). `<NA>` = indeterminate. Use to filter methodologically sound studies. |
| `pv_filter_label` | string | 0% | Human-readable label for `plausible_values_handling`. Values: `Pooled PVs (Rubin Rules)`, `Average PVs`, `Single PV Draw`, `WLE / IRT Theta`, `All PVs Analyzed Separately`, `Not Applicable (Framework)`, `Not Reported`. |
| `missing_data_handling` | string | 0% | Missing data strategy. Values: `multiple_imputation`, `listwise_deletion`, `knn_imputation`, `mean_imputation`, `single_imputation`, `pairwise_deletion`, `not_reported`. |
| `md_filter_label` | string | 0% | Human-readable label for `missing_data_handling`. Values: `Multiple Imputation`, `Listwise Deletion`, `KNN Imputation`, `Mean Imputation`, `Single Imputation`, `Pairwise Deletion`, `Not Reported`. |
| `handling_not_reported_explanation` | string | 15% | LLM-generated explanation when missing data handling was not reported. |
| `weights_filter` | string | 0% | Whether sampling weights were applied: `True`, `False`, `Unknown`. |

#### Sample and Design Fields

| Column | Type | Null% | Description |
|--------|------|------:|-------------|
| `research_design_type` | string | 0% | Study design. Values: `exploratory` (descriptive/factor analysis), `predictive` (regression/ML), `causal_observational` (quasi-experimental), `causal_experimental` (RCT). |
| `total_students` | float64 | 33% | Total student sample size as reported. Null when not explicitly stated. |
| `sample_size` | string | 0% | Sample size as extracted (may include "N/A" strings for non-empirical documents). |
| `sample_filtering_criteria` | string | 0% | LLM-generated description of how the study sample was selected/filtered. |
| `countries_formatted` | string | 19% | Comma-separated country names included in the study. |
| `countries_json` | string | 19% | JSON array of `{country_code, n_students}` objects. Country codes follow ISO 3166-1 alpha-3. Parse with `json.loads()` before use. |
| `country_codes` | string | 19% | Comma-separated ISO 3166-1 alpha-3 country codes (e.g. `TUR, DEU, FRA`). Extracted from `countries_json` for easy filtering. |

#### Classification and ML Fields

| Column | Type | Null% | Description |
|--------|------|------:|-------------|
| `document_class` | string | 0% | Top-level document classification: `empirical_article` or `technical_report`. |
| `study_filter_type` | string | 0% | Detailed study type. Values: `Empirical Study - Machine Learning`, `Empirical Study - Traditional Statistics`, `Technical/Assessment Framework`, `Descriptive National Report`. |
| `ml_techniques` | string | 0% | ML methods used (canonical form). "Not Reported: Likely Traditional Methods" for non-ML studies. |
| `ml_primary` | string | 87% | Primary ML technique (most prominent method). Null for non-ML studies (87% of corpus). |
| `ml_all_techniques` | string | 87% | Comma-separated list of all ML techniques used. Null for non-ML studies. |
| `ml_family` | string | 0% | ML method family. Values: `Tree-Based / Ensemble Learning`, `Deep Learning`, `Generalized Linear Models (GLM)`, `Other ML / Not Classified`, `Not Reported: Likely Traditional Methods`, `N/A: Technical Report`. |

#### Outcome and Synthesis Fields

| Column | Type | Null% | Description |
|--------|------|------:|-------------|
| `outcome_summary` | string | 0% | LLM-generated 2–4 sentence summary of the study's main contribution and findings. |
| `primary_finding` | string | 0% | LLM-generated single-sentence statement of the primary finding. |
| `effect_size` | string | 0% | Effect size reported by authors (e.g. "R²=0.42", "Cohen's d=0.31"). "Not Reported by Authors" when absent. |
| `confounders` | string | 0% | Whether confounders were identified: `present`, `Not Reported by Authors`, `N/A: Technical Report`. |
| `null_fields_interpretation` | string | 37% | LLM explanation for why fields are null (e.g. bibliographic-only excerpt, non-empirical document). |

---

### `findings.parquet` (2,128 rows × 16 columns)

One row per study finding. A single article may have multiple findings (different outcome variables or subgroup analyses).

| Column | Type | Null% | Description |
|--------|------|------:|-------------|
| `file_name` | string | 0% | Source PDF filename. Foreign key to `articles_master`. |
| `doi` | string | 15% | Article DOI. |
| `dataset_used` | string | 0% | ILSA dataset and cycle used (e.g. "PISA 2018", "TIMSS 2019", "PIRLS 2021"). |
| `target_variable` | string | 0% | Outcome variable of the finding (e.g. "Mathematics achievement", "Reading literacy"). |
| `top_predictors` | string | 0% | Key predictors identified in the finding (e.g. "SES, gender, school resources"). |
| `performance_metrics` | string | 11% | Model performance metrics as reported (e.g. "R²=0.42", "AUC=0.81", "RMSE=45.2"). |
| `standardized_conclusion` | string | 11% | Standardized LLM-generated conclusion sentence. |
| `primary_finding` | string | 0% | Full LLM-generated primary finding statement. |
| `publication_type` | string | 0% | Inherited from articles_master: `journal`, `report`, `book_chapter`, `conference`. |
| `source_category` | string | 0% | Inherited from articles_master. |
| `document_class` | string | 0% | Inherited from articles_master: `empirical_article` or `technical_report`. |
| `study_filter_type` | string | 0% | Inherited from articles_master. |
| `effect_size` | string | 0% | Effect size for this specific finding. "Not Reported by Authors" when absent. |
| `target_domain` | string | 0% | Subject domain of outcome variable. Values: `Mathematics`, `Reading`, `Science`, `Civic Education`, `Digital/Computer Literacy`, `Problem Solving`, `Non-Cognitive / Process Output`, `Composite / Multi-Domain`, `N/A: Technical Report`, `Other / Unspecified`. |
| `target_dimension` | string | 0% | Nature of the outcome measure. Values: `Cognitive Achievement`, `Attitudinal / Affective`, `Policy / System Outcome`, `Process Data / Log Metrics`, `Methodological (no DV)`, `Other`. |
| `predictor_filter_categories` | string | 18% | Semicolon-separated predictor categories (e.g. "Student: SES; School/Teacher: Context"). |

---

### `confounders.parquet` (8,336 rows × 11 columns)

One row per confounder/predictor variable per study. Multiple rows per article.

| Column | Type | Null% | Description |
|--------|------|------:|-------------|
| `file_name` | string | 0% | Source PDF filename. Foreign key to `articles_master`. |
| `doi` | string | 3% | Article DOI. |
| `variable_code` | string | 0% | Internal variable code assigned by pipeline (e.g. "sesi_j", "gender_s"). |
| `variable_name` | string | 0% | Human-readable variable name (e.g. "socioeconomic status (SES)", "school resources"). |
| `category` | string | 0% | Canonical variable category. Values: `socioeconomic`, `demographic`, `student_attitude`, `student_behavior`, `prior_achievement`, `parent_home`, `school`, `teacher`, `curriculum`, `ict`, `system_level`, `peer_effects`, `process_data`, `N/A: Technical Report`, `Not Reported by Authors`. |
| `predictor_level` | string | 0% | Level of the predictor in the multilevel structure. Values: `Student Level`, `School/Teacher Level`, `System/Country Level`, `Unspecified`, `N/A: Technical Report`. |
| `predictor_category` | string | 0% | Detailed predictor grouping. Values: `Student: SES`, `Student: Demographic`, `Student: Attitudinal/Behavioral`, `Student: Prior Achievement`, `Student: Process Data`, `School/Teacher: Context`, `School/Teacher: Practice`, `System: Policy/Context`, `N/A: Technical Report`, `Other`. |
| `publication_type` | string | 0% | Inherited from articles_master. |
| `source_category` | string | 0% | Inherited from articles_master. |
| `document_class` | string | 0% | Inherited from articles_master. |
| `study_filter_type` | string | 0% | Inherited from articles_master. |

---

## Known Limitations (Red Team Assessment)

The following limitations were identified through systematic internal review:

| Issue | Scope | Severity | Notes |
|-------|-------|----------|-------|
| `authors` null in 33% of records | articles_master | Medium | Institutional/anonymous OECD and IEA reports lack author attribution |
| `ml_primary` / `ml_all_techniques` null for 87% | articles_master | By design | 87% of corpus uses traditional statistics, not ML — nulls are correct |
| `year` stored as Int64 | articles_master | ✅ Fixed | Corrected from float64 |
| `open_access` column | articles_master | ✅ Fixed | Removed — not reliably extractable from PDF |
| `json_source_path` column | articles_master | ✅ Fixed | Removed — local machine path, not portable |
| `countries_json` as JSON string, not array | articles_master | ✅ Fixed | `country_codes` kolonu eklendi (ISO 3166-1 alpha-3, virgülle ayrılmış) |
| `pv_correct` flag added | articles_master | ✅ Fixed | 104 studies with incorrect PV handling flagged (`single_pv`, `average_pv`) |
| `performance_metrics` null in 11% of findings | findings | Medium | Not all studies report quantitative performance metrics |
| `confounders` = "N/A: Technical Report" (544 rows) | confounders | By design | Technical reports do not model predictors — expected |
| Human validation sample | All tables | High | Inter-rater validation performed on 8-paper sample only. Full corpus validation ongoing. |

---

## Quick Usage

```python
from datasets import load_dataset

# Cleaned article metadata (recommended)
articles = load_dataset("dedemerve/ILSA-LLM-Extractor-Dataset", "articles_master", split="train")

# Findings
findings = load_dataset("dedemerve/ILSA-LLM-Extractor-Dataset", "findings", split="train")

# Confounders
confounders = load_dataset("dedemerve/ILSA-LLM-Extractor-Dataset", "confounders", split="train")
```

```python
import pandas as pd

# Direct Parquet read
df = pd.read_parquet("hf://datasets/dedemerve/ILSA-LLM-Extractor-Dataset/data/processed/articles_master.parquet")

# Filter: ML studies only
ml_studies = df[df['study_filter_type'] == 'Empirical Study - Machine Learning']

# Filter: correct PV handling only
correct_pv = df[df['plausible_values_handling'] == 'rubin_rules']

# Parse countries
import json
df['countries_list'] = df['countries_json'].apply(lambda x: json.loads(x) if pd.notna(x) else [])
```

---

## Coverage

| Assessment | Organization | Corpus Source |
|-----------|-------------|--------------|
| PISA | OECD | Scopus, WoS, OECD iLibrary |
| TALIS | OECD | OECD iLibrary |
| PIAAC | OECD | OECD iLibrary |
| TIMSS | IEA | IEA Repository, Scopus, WoS |
| PIRLS | IEA | IEA Repository, Scopus, WoS |
| ICCS | IEA | IEA Repository, Scopus, WoS |

---

## Citation

```bibtex
@dataset{dede_cetinkaya_2026_ilsa,
  author    = {Dede, Merve and {\c{C}}etinkaya, Ekrem},
  title     = {ILSA LLM Extractor Dataset},
  year      = {2026},
  publisher = {Hugging Face},
  url       = {https://huggingface.co/datasets/dedemerve/ILSA-LLM-Extractor-Dataset}
}
```

## License

[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) — Raw ILSA microdata is **not** included. This dataset contains only LLM-extracted metadata from published research articles.
