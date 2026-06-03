# Codebook: ILSA Meta-Analysis Dataset

## LLM-Augmented Knowledge Extraction from the ILSA Literature Corpus

**Coverage:** 1,266 source documents (after quality filtering); sourced from Scopus, Web of Science, IEA, OECD, and the curated ILSA survey article collection (2000–2026)
**Extraction method:** Three-stage LLM-based pipeline — (1) structured JSON extraction from full-text PDFs, (2) schema normalization and controlled-vocabulary alignment, (3) filter label generation for Excel-native slicing
**Primary join key:** `file_name` (unique document identifier, consistent across all sheets)

---

## Sheet Overview

| Sheet | Rows | Unit of Analysis | Purpose |
|---|---|---|---|
| `0_Dashboard_Analysis_Control` | 36 | Metric | Summary statistics and pipeline status indicators |
| `1_Articles_Master` | 1,266 | Document | Bibliographic metadata, methodological quality flags, filter labels |
| `2_Main_Findings` | 2,128 | Finding per document | Empirical results, predictors, performance metrics, domain labels |
| `3_Confounders` | 8,336 | Control variable per document | Variable-level confounder taxonomy |
| `Canonical_View` | 10,464 | Record (finding or confounder) | Unified long-format view for cross-sheet filtering and RAG retrieval |
| `_FilterLists` | 40 | Controlled vocabulary item | Reference lists for all filter/slicer columns |

---

## Sheet 1: `1_Articles_Master`

Each row represents one source document. Contains bibliographic metadata, methodological design indicators, quality filter labels, and country encoding.

### Bibliographic Identifiers

| Variable | Type | Description | Controlled Vocabulary |
|---|---|---|---|
| `file_name` | string | Original PDF filename. Primary key shared across all sheets. | Unique per document |
| `doi` | string | Digital Object Identifier. Empty if unavailable. | Format: `10.XXXX/...` |
| `title` | string | Full publication title. | Free text |
| `authors` | string | Comma-delimited author names in publication order. | Free text |
| `year` | float | Publication year. Stored as float to accommodate missing values. | 4-digit integer; `NaN` if unavailable |
| `venue` | string | Journal name, conference title, or book series. | Free text |
| `publication_type` | string | Publication format. | `journal`, `book_chapter`, `conference`, `report` |
| `source_category` | string | Methodological role of the document in the literature. | `peer_reviewed_research`, `technical_report`, `methodology_paper`, `review_article` |
| `open_access` | float | Full-text accessibility status at time of retrieval. | `1.0` = open; `0.0` = restricted; `NaN` = unknown |
| `corpus_source` | string | Bibliographic database or collection the document was drawn from. | `IEA`, `OECD`, `Scopus`, `Web of Science`, `ilsa_survey_articles` |
| `json_source_path` | string | File path to the JSON extraction output for this document. Enables row-level traceability back to raw LLM output. | Absolute path string |

### Methodological Quality Indicators

| Variable | Type | Description | Controlled Vocabulary |
|---|---|---|---|
| `student_weights_used` | float | Whether student-level sampling weights were applied in the analysis. Required for population-representative inference from ILSA complex survey data. | `1.0` = yes; `0.0` = no; `NaN` = not determinable |
| `replicate_weights_used` | float | Whether replicate weights (jackknife or Balanced Repeated Replication) were used for variance estimation. These adjust standard errors for the multi-stage cluster sampling structure common to all ILSA programs. | `1.0` = yes; `0.0` = no; `NaN` = not determinable |
| `weight_variable_name` | string | Name of the weight variable used. Standard ILSA identifiers include `W_FSTUWT` (PISA), `TOTWGT`, `TOTWGTC`, `TOTWGTS`, `TOTWGTT` (TIMSS/PIRLS/ICCS). | e.g., `W_FSTUWT`, `TOTWGTC` |
| `weight_fields_interpretation` | string | Narrative description of how sampling weights were applied or interpreted in the study context. | Free text |
| `plausible_values_handling` | string | Method used to handle plausible values (PVs) — multiply-imputed latent proficiency scores in ILSA cognitive assessments. See **Special codes** below for `not_applicable` vs `not_reported`. | `rubin_rules`, `average_pv`, `single_pv`, `all_pv`, `irt_theta`, `wle`, `mitml`, `not_applicable`, `not_reported` |
| `missing_data_handling` | string | Strategy for missing item or background questionnaire data in the study's analysis. See **Special codes** below. | `listwise_deletion`, `pairwise_deletion`, `multiple_imputation`, `mean_imputation`, `single_imputation`, `knn_imputation`, `not_reported` |
| `handling_not_reported_explanation` | string | Required when PV or missing-data fields are `not_reported` or `not_applicable`: 2–3 sentences explaining *why* (reporting gap vs document type). | Free text |
| `null_fields_interpretation` | string | Explanation of why certain schema fields are empty for a given document — for example, because the document is a technical report without an ML modeling component, or because only front matter was available during extraction. Supports corpus quality auditing. | Free text |

### Sample Descriptors

| Variable | Type | Description | Controlled Vocabulary |
|---|---|---|---|
| `total_students` | float | Number of student-level observations in the analytic sample. | Integer ≥ 0; `NaN` if unavailable |
| `sample_size` | string / int | Alternative sample size field. Contains a numeric value for empirical studies or a status code for non-empirical documents. | Integer or `N/A: Technical Report` |
| `sample_filtering_criteria` | string | Inclusion and exclusion criteria applied to the ILSA microdata before analysis. Common filters include grade level, minimum school response rate, and exclusion of students with special needs. | Free text |
| `countries_formatted` | string | Comma-delimited list of countries included in the study. Not normalized to a single encoding standard. | Free text; e.g., `Turkey, Germany, Finland` |
| `countries_json` | string | JSON-serialized array of country objects with `country_code` (ISO 3166-1 alpha-3) and `n_students` fields. Intended for programmatic filtering. | JSON array; `n_students` may be `null` |

### Analytical Method Fields

| Variable | Type | Description | Controlled Vocabulary |
|---|---|---|---|
| `ml_primary` | string | Primary analytical or machine learning technique used in the study. | e.g., `OLS`, `HLM`, `Random Forest`, `XGBoost`, `SVM`, `Neural Network`, `Logistic Regression`, `Decision Tree` |
| `ml_techniques` | string | All analytical techniques used, including those applied for comparison or robustness purposes. | Free text or `Not Reported: Likely Traditional Methods`, `N/A: Technical Report` |
| `ml_all_techniques` | string | Comma-delimited list of all specific technique names found in the document. | Free text; comma-delimited |
| `research_design_type` | string | Epistemological orientation of the study. | `predictive`, `explanatory`, `exploratory`, `causal_observational`, `causal_experimental` |
| `outcome_summary` | string | Standardized paragraph summarizing the study's primary findings, generated to support cross-study comparability. Approximately 100–200 words. | Free text |
| `primary_finding` | string | One-sentence synthesis of the study's key result, following the template: *"Using [dataset], the study leveraged [predictors] to predict [outcome], finding that [conclusion]."* Formatted for rapid scanning and RAG retrieval. | Free text |
| `effect_size` | string | Quantitative effect size or model performance metric (e.g., R², explained variance, accuracy). | Free text; or `Not Reported by Authors`, `N/A: Technical Report`, `Descriptive statistics only` |
| `confounders` | string | Indicates whether control variables are documented for the study. | `present`, `Not Reported by Authors`, `N/A: Technical Report` |

### Filter / Slicer Labels

Controlled-vocabulary columns optimized for use as Excel column filters, PivotTable slicers, or query predicates. All values are drawn from the `_FilterLists` sheet.

| Variable | Type | Description | Controlled Vocabulary |
|---|---|---|---|
| `document_class` | string | Top-level classification separating empirical studies from non-empirical documents. | `empirical_article`, `technical_report` |
| `study_filter_type` | string | Study type combining empirical stance and method family. Primary filter for restricting the corpus to ML-focused papers. | `Empirical Study - Machine Learning`, `Empirical Study - Traditional Statistics`, `Technical/Assessment Framework`, `Descriptive National Report` |
| `ml_family` | string | Method family that groups individual technique names into analytically comparable clusters. | `Tree-Based / Ensemble Learning`, `Deep Learning`, `Generalized Linear Models (GLM)`, `Traditional Psychometrics / Multilevel Modeling`, `Clustering / Unsupervised Learning`, `Not Reported: Likely Traditional Methods`, `N/A: Technical Report` |
| `pv_filter_label` | string | Human-readable label for `plausible_values_handling`, formatted for filter display. | `Pooled PVs (Rubin Rules)`, `Average PVs`, `Single PV Draw`, `WLE / IRT Theta`, `Not Applicable (Framework)`, `Not Reported` |
| `md_filter_label` | string | Human-readable label for `missing_data_handling`, formatted for filter display. | `Listwise Deletion`, `Pairwise Deletion`, `Multiple Imputation`, `Mean Imputation`, `Single Imputation`, `KNN Imputation`, `Not Reported` |
| `weights_filter` | string | Simplified weight usage flag for one-click filtering. | `True`, `False`, `Unknown` |

### Special codes: `not_applicable` vs `not_reported` (and Excel sentinels)

These are **controlled vocabulary codes**, not missing cells. They record *why* a methodological field has no analytic value — the pipeline does not invent Rubin rules or imputation when the source text does not support it.

| Code | Field(s) | Meaning | Typical document |
|------|----------|---------|------------------|
| **`not_applicable`** | `plausible_values_handling` | The document does **not** perform achievement analysis using ILSA plausible values (no PV-based outcome modeling). | Technical report, user guide, framework, encyclopedia, questionnaire supplement, many descriptive national reports |
| **`not_reported`** | `missing_data_handling` (and rarely `plausible_values_handling`) | The study may be empirical, but authors **did not state** the strategy clearly enough to code (or extraction could not resolve it). | Peer-reviewed papers with sparse methods sections |
| **`rubin_rules`**, `multiple_imputation`, etc. | Either field | Authors **explicitly reported** the method; coded from the PDF text. | Empirical ILSA ML / multilevel studies |

**Important distinction (JSON and Excel):**

- **`not_applicable`** = “This question does not apply to this document type” (e.g. no achievement PV analysis in a PIRLS User Guide supplement).
- **`not_reported`** = “This question applies, but the paper did not report it (or it could not be extracted).”

Filter labels mirror this logic:

| Raw code | `pv_filter_label` | `md_filter_label` |
|----------|-------------------|-------------------|
| `not_applicable` | `Not Applicable (Framework)` | — |
| `not_reported` | `Not Reported` | `Not Reported` |

**Excel-only sentinels** (relational sheets, not JSON enum values) use a different convention:

| Sentinel | Meaning |
|----------|---------|
| `N/A: Technical Report` | Row belongs to a technical/framework document; empirical finding fields are intentionally inactive |
| `N/A: Descriptive Report` | Descriptive national/international report without ML outcome modeling |
| `Not Reported by Authors` | Empirical document; authors did not report that specific field |
| `Not Reported` | Generic absence on a non-critical column |

Audit columns: `handling_not_reported_explanation` and `null_fields_interpretation` (JSON) carry the narrative reason when codes are `not_*`. See also `docs/json_field_reference_tr.md` for per-field JSON definitions.

---

## Sheet 2: `2_Main_Findings`

Each row represents one empirical finding within a source document. A single document may contribute multiple rows when findings span different outcome variables, datasets, or country subgroups. Linked to `1_Articles_Master` via `file_name`.

| Variable | Type | Description | Controlled Vocabulary |
|---|---|---|---|
| `file_name` | string | Foreign key → `1_Articles_Master.file_name`. | Join key |
| `doi` | string | DOI of the parent document. Carried forward for convenience. | Inherited |
| `dataset_used` | string | ILSA dataset and cycle on which this finding is based. Enables filtering by survey program and wave. | e.g., `PISA 2018`, `TIMSS 2019`, `PIRLS 2021`, `ICCS 2016` |
| `target_variable` | string | Outcome or dependent variable being predicted or explained. In ILSA research this is typically a student achievement score, attitudinal construct, or behavioral indicator. | Free text; e.g., `reading literacy score`, `mathematics plausible value`, `sense of belonging` |
| `top_predictors` | string | Comma-delimited list of the most influential predictor variables, ordered by effect size or importance where available. | Free text; comma-delimited |
| `performance_metrics` | string | Model performance indicators for this finding (R², RMSE, AUC, accuracy, F1-score, explained variance, etc.). Supports meta-analytic comparison of model quality across studies. | Free text; e.g., `R² = 0.34`, `AUC = 0.81`; or `Not reported` |
| `standardized_conclusion` | string | Standardized narrative conclusion following the template: *dataset → predictors → outcome → magnitude → educational interpretation.* Formatted for direct text embedding in RAG pipelines. Approximately 80–150 words. | Free text |
| `effect_size` | string | Effect size or performance metric in condensed form. | Free text or `Not Reported by Authors` |
| `primary_finding` | string | One-sentence synthesis of this specific finding, following the same template as `1_Articles_Master.primary_finding`. | Free text |
| `publication_type` | string | Carried forward from the parent document. | See `1_Articles_Master.publication_type` |
| `source_category` | string | Carried forward from the parent document. | See `1_Articles_Master.source_category` |
| `document_class` | string | Carried forward from the parent document. | `empirical_article`, `technical_report` |
| `study_filter_type` | string | Carried forward from the parent document. | See `1_Articles_Master.study_filter_type` |
| `target_domain` | string | Normalized subject-area domain of the outcome variable. Enables aggregation across studies by literacy domain. | `Mathematics`, `Reading`, `Science`, `Digital/Computer Literacy`, `Civic Education`, `Other / Unspecified`, `N/A: Technical Report` |
| `target_dimension` | string | Type of construct measured by the outcome variable. | `Cognitive Achievement`, `Attitudinal / Affective`, `Process Data / Log Metrics`, `Policy / System Outcome`, `Methodological (no DV)` |
| `predictor_filter_categories` | string | Semicolon-delimited list of predictor category labels present in this finding's predictor set. Enables filtering by predictor family without joining to `3_Confounders`. | e.g., `Student: SES; Student: Demographic; School/Teacher: Context` |

---

## Sheet 3: `3_Confounders`

Each row represents one control or covariate variable documented in a source document. Multiple confounders per document appear as separate rows. Linked to `1_Articles_Master` via `file_name`.

| Variable | Type | Description | Controlled Vocabulary |
|---|---|---|---|
| `file_name` | string | Foreign key → `1_Articles_Master.file_name`. | Join key |
| `doi` | string | DOI of the parent document. Carried forward for convenience. | Inherited |
| `variable_code` | string | ILSA instrument variable code from the official survey codebook (e.g., `ESCS`, `ST013Q01TA`, `HISCED`). When only a composite index is referenced without item-level codes, the index identifier is recorded. | e.g., `ESCS`, `W_FSTUWT`, `sesi_j`, `ST218Q01HA` |
| `variable_name` | string | Human-readable construct label for the variable. | e.g., `Socioeconomic status (SES)`, `Number of books at home`, `Open classroom discussion` |
| `category` | string | Conceptual domain of the confounder. Supports aggregated analysis of which variable families are most commonly controlled for in the ILSA ML literature. | `socioeconomic`, `demographic`, `student_attitude`, `student_behavior`, `ict`, `home_environment`, `school_context`, `teacher_quality`, `country_level`, `other` |
| `publication_type` | string | Carried forward from the parent document. | See `1_Articles_Master.publication_type` |
| `source_category` | string | Carried forward from the parent document. | See `1_Articles_Master.source_category` |
| `document_class` | string | Carried forward from the parent document. | `empirical_article`, `technical_report` |
| `study_filter_type` | string | Carried forward from the parent document. | See `1_Articles_Master.study_filter_type` |
| `predictor_level` | string | Level of analysis at which the variable was measured or aggregated. | `Student Level`, `School/Teacher Level`, `System/Country Level`, `Unspecified`, `N/A: Technical Report` |
| `predictor_category` | string | Finer-grained theoretical category for taxonomic analysis of the predictor space across the corpus. | `Student: SES`, `Student: Demographic`, `Student: Attitudinal/Behavioral`, `Student: Prior Achievement`, `Student: Process Data`, `School/Teacher: Context`, `School/Teacher: Practice`, `System/Country Level` |

---

## Sheet 4: `Canonical_View`

A unified long-format table merging findings and confounders into a single queryable view. Designed for cross-sheet filtering, RAG index construction, and meta-analytic aggregation without multi-table joins.

| Variable | Type | Description | Controlled Vocabulary |
|---|---|---|---|
| `record_type` | string | Whether the row originates from the findings or confounders sheet. | `finding`, `confounder` |
| `file_name` | string | Foreign key → `1_Articles_Master.file_name`. | Join key |
| `study_filter_type` | string | Carried forward study type label. | See `1_Articles_Master.study_filter_type` |
| `canonical_method` | string | Normalized method label from the controlled taxonomy. Maps diverse technique names onto a consistent vocabulary for meta-analysis. | `Deep_Learning`, `Ensemble_Learning`, `Supervised_General`, `Traditional_Stats`, `Unsupervised`, `Theoretical_and_Meta_Synthesis`, `[IGNORE]` |
| `canonical_variable` | string | Normalized construct label from the controlled taxonomy. Maps diverse variable names and codes onto a consistent conceptual vocabulary. | e.g., `SES`, `Gender_Demographics`, `Civic_Achievement`, `Civic_Engagement`, `Academic_Resilience`, `Belonging_School_Climate` |
| `metadata_filter_flag` | string | High-level eligibility flag for meta-analytic inclusion. | `empirical_finding`, `theoretical_meta_synthesis`, `excluded` |
| `raw_label` | string | Original unenriched label from the source field, preserved for traceability. | Free text |
| `effect_trend` | string | Direction of the reported effect or association. | `Positive`, `Negative`, `Null`, `[IGNORE]` |
| `target_domain` | string | Normalized subject-area domain. | See `2_Main_Findings.target_domain` |
| `synthesis_excerpt` | string | Truncated standardized conclusion (first ~300 characters) for display in filter panels and preview panes. | Free text excerpt |

---

## Sheet 5: `_FilterLists`

Reference sheet containing all controlled vocabulary values used across filter and slicer columns. Intended for use as Excel named ranges, data validation lists, or lookup tables. Rows represent individual vocabulary items, not documents.

| Variable | Description |
|---|---|
| `analysis_level` | Unit of analysis: `student`, `school`, `country` |
| `canonical_method` | Normalized ML method family labels |
| `canonical_variable` | Normalized construct and variable labels |
| `catalog_match_confidence` | Confidence level of canonical label assignment: `High`, `Medium`, `Low` |
| `check_type` | Type of verification applicable to the record: `descriptive`, `weighted_mean`, `correlation_direction`, `correlation_magnitude`, `regression_coefficient_sign` |
| `effect_direction_consensus` | Cross-study consensus on effect direction: `Positive`, `Negative`, `Null`, `Unknown` |
| `effect_trend` | Effect direction for individual records: `Positive`, `Negative`, `Null`, `[IGNORE]` |
| `feature_store_status` | Variable extraction pipeline status: `Not_Started`, `Planned`, `Partial`, `Complete` |
| `match_status` | Comparison outcome between LLM-generated and published findings: `Full_Match`, `Partial_Match`, `Direction_Mismatch`, `Discrepancy`, `Flagged_Anomaly` |
| `md_filter_label` | Missing data handling labels (see `1_Articles_Master.md_filter_label`) |
| `metadata_filter_flag` | Inclusion eligibility: `empirical_finding`, `theoretical_meta_synthesis` |
| `microdata_ready` | Whether source ILSA microdata is available for reanalysis: `True`, `False` |
| `ml_family` | Method family groupings (see `1_Articles_Master.ml_family`) |
| `multi_output_pipeline_priority` | Priority tier for the multi-output RAG pipeline: `Critical`, `High`, `Medium`, `Low`, `Deferred` |
| `normalized_domain` | Normalized subject domain: `Mathematics`, `Science`, `Reading`, `Digital/Computer Literacy`, `Civic Education` |
| `normalized_isa` | Normalized ILSA program name: `PISA`, `TIMSS`, `PIRLS`, `ICCS`, `ICILS` |
| `paper_reported_direction` | Effect direction stated by the original authors: `Positive`, `Negative`, `Null`, `Unknown` |
| `paper_significance` | Statistical significance as reported: `Significant`, `Not_Significant`, `Not_Reported` |
| `predictor_category` | Predictor taxonomy labels (see `3_Confounders.predictor_category`) |
| `predictor_level` | Predictor analysis level (see `3_Confounders.predictor_level`) |
| `pv_filter_label` | Plausible values handling labels (see `1_Articles_Master.pv_filter_label`) |
| `pv_rule` / `pv_rule_applied` | Internal PV handling codes: `rubin_rules`, `single_pv`, `average_pv`, `all_pv_separate`, `not_applicable` |
| `record_type` | Row type in `Canonical_View`: `finding`, `confounder` |
| `replicated_direction` | Effect direction in any replication or validation study: `Positive`, `Negative`, `Null`, `Unknown` |
| `study_filter_type` | Study type labels (see `1_Articles_Master.study_filter_type`) |
| `target_dimension` | Construct type of the outcome variable (see `2_Main_Findings.target_dimension`) |
| `target_domain` | Subject domain of the outcome variable (see `2_Main_Findings.target_domain`) |
| `verification_eligible` | Whether the record is eligible for automated verification: `True`, `False` |
| `verification_priority` | Verification scheduling tier: `P1_pilot`, `P2_core`, `P3_extended`, `P4_deferred` |
| `weights_filter` | Simplified weight usage flag: `True`, `False`, `Unknown` |

---

## Dataset Construction Notes

### Extraction Pipeline

Documents were processed through a three-stage LLM-based pipeline:

1. **Stage 1 — Structured JSON Extraction:** A fixed-schema system prompt was applied to each PDF. The schema covers bibliographic metadata, survey design quality indicators (`student_weights_used`, `replicate_weights_used`, `plausible_values_handling`, `missing_data_handling`), sample descriptors, ML technique inventory, a `main_findings` array, and a `confounders_identified` array. Each output is stored as an individual `.json` file traceable via `json_source_path`.

2. **Stage 2 — Normalization and Controlled-Vocabulary Alignment:** Post-processing scripts traversed all JSON outputs, enforced schema consistency, removed invalid characters, and mapped extracted values onto the controlled vocabularies defined in this codebook. Filter label columns (`study_filter_type`, `ml_family`, `pv_filter_label`, etc.) were generated at this stage to enable Excel-native filtering without formula-based transformations.

3. **Stage 3 — Canonical View and Filter List Generation:** The `Canonical_View` sheet was built by merging findings and confounders into a unified long-format table with normalized method and variable labels (`canonical_method`, `canonical_variable`). The `_FilterLists` sheet was populated with all controlled vocabulary items for use as a validation reference.

### Corpus Statistics

| Metric | Value |
|---|---|
| Unique documents (after quality filtering) | 1,266 |
| Main findings rows | 2,128 |
| Confounder rows | 8,336 |
| Canonical view rows | 10,464 |
| Publication years | 2000–2026 |
| Source databases | Scopus, Web of Science, IEA, OECD, curated ILSA collection |

### Known Limitations

- **Extraction coverage:** LLM extraction may miss or misclassify content in documents with non-standard reporting formats, multilingual text, or incomplete full-text access. Fields that cannot be determined are coded as `not_reported` or `not_applicable`; the `null_fields_interpretation` and `handling_not_reported_explanation` columns provide document-level audit notes.
- **Variable code granularity:** Item-level ILSA variable codes are recorded only when cited in the source document. Studies referencing composite indices (e.g., `ESCS`) without naming constituent items will show only the index-level code.
- **Country encoding:** `countries_formatted` is not normalized to a single standard; `countries_json` provides ISO 3166-1 alpha-3 codes where resolvable.
- **Multilingual documents:** A subset of documents is in French, German, Spanish, Latvian, or Norwegian. Extraction quality for these is lower than for English-language documents.
- **Non-empirical documents:** Technical reports, assessment frameworks, and descriptive national reports are retained with `document_class = technical_report`. These should be excluded when restricting analyses to empirical ML studies by filtering `study_filter_type = Empirical Study - Machine Learning`.

### Intended Use

This dataset serves as the knowledge base layer in an LLM-augmented ILSA forecasting and interpretation framework. The `standardized_conclusion` field in `2_Main_Findings` and the `synthesis_excerpt` field in `Canonical_View` are formatted for text embedding and retrieval in Retrieval-Augmented Generation (RAG) architectures. The `3_Confounders` sheet supports systematic mapping of the predictor space across the ILSA ML literature, enabling structured prompting when applying the framework to new ILSA cycle data. The `match_status` and `verification_priority` fields in `_FilterLists` support the planned validation step in which LLM-generated country-level insights are compared against published findings.
