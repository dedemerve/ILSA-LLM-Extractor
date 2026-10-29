# Pre-Fine-Tuning Baseline Inventory Report

**Date:** 2026-10-05  
**Branch at audit time:** fix/stage5-dual-source-build-xy  
**Purpose:** Record the exact frozen baseline before any domain-specific fine-tuning experiment begins. Every number and file path here is authoritative for the comparison experiment.

---

## A. Existing Corpus

### Provenance (authoritative — from `outputs/stage5/provenance_audit.md`)

```
N_json_total         = 1,458   (WoS:302 + Scopus:423 + OECD:591 + ilsa_survey:142)
       ↓ ILSA keyword filter
N_ilsa_relevant      = 1,349   (WoS:294 + Scopus:386 + OECD:545 + survey:124)
       ↓ main_findings field populated
N_evidence_bearing   = 1,144   (cross-source dedup before this step)
       ↓ DOI / normalized-title dedup
N_unique_any         = 820     (any predictor name present)
       ↓ RAW_TO_CANONICAL filter
N_canonical_mapped   = 620     ← THESIS FIGURE / enters W_{j,p,d} computation
```

**Critical note:** The intermediate figure of 577 found in older pipeline comments represents only WoS+Scopus DOI-dedup *before* OECD and ilsa_survey expansion. It is NOT the evidence-bearing or canonical-mapped count. Use 620 as the authoritative canonical-mapped figure.

200 studies (820 − 620) contain main_findings but their predictor labels fall outside the RAW_TO_CANONICAL dictionary (ethnicity, school type, language-background variants, etc.). They are stored in the database but do NOT contribute to W_{j,p,d}.

### Source database distribution

| Source | N JSON | N ILSA-relevant | N Evidence-bearing | % ILSA | % Evidence |
|--------|-------:|----------------:|-------------------:|-------:|-----------:|
| Web of Science | 302 | 294 | 284 | 97.4% | 96.6% |
| Scopus | 423 | 386 | 374 | 91.3% | 96.9% |
| OECD iLibrary | 591 | 545 | 364 | 92.2% | 66.8% |
| ilsa_survey_articles | 142 | 124 | 122 | 87.3% | 98.4% |
| **Total** | **1,458** | **1,349** | **1,144** | | |

Source: `outputs/stage5/corpus_provenance_audit.csv`

### Evidence matrix counts

| File | Data rows (excl. header) |
|------|-------------------------:|
| `outputs/stage5/evidence_matrix.csv` | 530 |
| `outputs/stage5/evidence_matrix_v2.csv` | 570 |
| `outputs/stage5/evidence_matrix_full.csv` | 1,121 |
| `outputs/stage5/evidence_matrix_full_v2.csv` | 6,799 |

The `evidence_matrix_full_v2.csv` is the expanded matrix used for final W_{j,p,d} computation.

### DOI coverage

DOI-based deduplication is the primary dedup strategy. DOI field is present in `MetadataBlock.doi` (mandatory when on PDF). Secondary dedup uses normalized title. Cross-database overlap: Scopus/WoS overlap ~30 records; OECD/survey overlap ~75 records (from provenance audit).

---

## B. Existing Extraction Schema

Defined in `src/schemas/models.py` (Pydantic v2 models). Full class hierarchy:

**MetadataBlock** — top-level paper identity
- `file_name` (str, required)
- `title` (Optional[str])
- `authors` (Optional[List[str]])
- `year` (Optional[int])
- `doi` (Optional[str]) — validated, prefix stripped
- `venue` (Optional[str])
- `publication_type`: `journal|conference|book_chapter|preprint|report|thesis`
- `open_access` (Optional[bool])
- `source_category`: `peer_reviewed_research|official_report`

**SurveyDesign** — sampling/weighting methodology
- `student_weights_used` (Optional[bool], coerced)
- `replicate_weights_used` (Optional[bool], coerced)
- `weight_variable_name` (Optional[str])
- `weight_fields_interpretation` (str, mandatory)

**SampleDetails** — study sample
- `total_students` (Optional[int])
- `countries`: List[`CountrySample`{`country_code`, `n_students`}]
- `sample_filtering_criteria` (str, mandatory)

**Confounder** — one variable per object
- `variable_code` (str)
- `variable_name` (str)
- `category`: one of 13 Literal values:  
  `socioeconomic|demographic|student_attitude|student_behavior|teacher|school|ict|curriculum|parent_home|process_data|prior_achievement|peer_effects|system_level`

**StructuredFinding** — one model-run per object
- `dataset_used` (str)
- `target_variable` (str)
- `top_predictors` (List[str])
- `performance_metrics` (str)
- `standardized_conclusion` (str)

**MLTechniques**
- `primary` (Optional[str])
- `all_techniques` (List[str])

**DataBlock** — full data container
- `survey_design`: SurveyDesign
- `plausible_values_handling`: `single_pv|all_pvs|not_applicable|not_reported`
- `missing_data_handling`: `listwise_deletion|multiple_imputation|mean_imputation|not_reported|not_applicable`
- `handling_not_reported_explanation` (Optional[str])
- `sample_details`: SampleDetails
- `ml_techniques`: MLTechniques
- `confounders_identified`: List[Confounder]
- `main_findings`: List[StructuredFinding]
- `outcome_summary` (str)
- `research_design_type`: `cross_sectional|longitudinal|experimental|not_reported`
- `null_fields_interpretation` (Optional[str])

Extractor code: `src/extractors/gpt_extractor.py` (OpenAI API), `ilsa_pipeline/extractors/gpt_extractor.py`.

---

## C. Existing Canonical Vocabulary

### Canonical constructs (19 total, from `outputs/stage5/evidence_provenance_audit.csv`)

| Canonical Construct | N Studies | N Rows | Forecastable |
|--------------------|----------:|-------:|:-------------|
| SES_COMPOSITE | 54 | 157 | True |
| GENDER | 50 | 83 | False |
| SCHOOL_BELONGING | 27 | 47 | True |
| TEACHER_SELF_EFFICACY_CLS_MGMT | 21 | 41 | True |
| HOME_RESOURCES | 19 | 28 | True |
| DISCIPLINARY_CLIMATE | 17 | 21 | True |
| ICT_HOME_ACCESS | 15 | 42 | True |
| SELF_CONCEPT | 13 | 23 | False |
| MOTIVATION | 11 | 19 | False |
| IMMIGRANT_STATUS | 11 | 18 | False |
| PARENTAL_EDUCATION | 7 | 10 | True |
| CURRICULUM_TYPE | 4 | 9 | False |
| LANGUAGE_BACKGROUND | 3 | 6 | False |
| ICT_SCHOOL_ACCESS | 3 | 5 | True |
| EFFECTIVE_PD | 3 | 6 | True |
| INSTRUCTIONAL_TIME | 3 | 7 | False |
| JOB_SATISFACTION_PROFESSION | 1 | 1 | True |
| PRIOR_ACHIEVEMENT | 1 | 5 | True |
| HOME_LITERACY_ACTIVITIES | 1 | 2 | True |

### RAW_TO_CANONICAL mapping

Location: `scripts/expand_evidence_matrix.py`, dict `RAW_TO_CANONICAL` (also in `scripts/build_evidence_matrix_v2.py`).
Covers: raw predictor string → canonical construct label (lowercased match).

### Canonical crosswalk

Location: `outputs/stage5/canonical_crosswalk.csv` (38 rows)  
Fields: `program|cycle_availability|raw_variable|canonical_construct|level|direction|direction_note|weight_type|pv_structure|forecast_role|tier|cross_cycle_comparable|direction_original|direction_canonical|transformation|notes`

### Direction/transformation records

Explicitly documented for reversed scales (e.g., TIMSS BSDGEDUP: `direction_original=−`, `direction_canonical=+`, `transformation=negate`; TALIS TCDISCS: `CAUTION_direction_flip`).

---

## D. Existing Literature Priors

### W_j (global)

File: `outputs/stage5/predictor_weights_v3.csv`  
Rows: 19 (one per canonical construct)  
Fields: `predictor_canonical|feature_name|lit_freq_v3|lit_component|w_norm_v2|w_norm_v3|programs`

Formula components:
- `lit_freq_v3`: raw frequency count across canonical-mapped studies
- `lit_component = log(1 + freq) / max_log_freq` (frequency signal, normalized)
- `w_norm_v2`: diversity signal (program × domain breadth, normalized)
- `w_norm_v3 = (lit_component + w_norm_v2) / 2` — combined weight (final W_j)

Top W_j values: MOTIVATION=1.0, SES_COMPOSITE=0.8194, TEACHER_QUALITY=0.717, GENDER=0.757

### W_{j,p,d} (program × domain)

File: `outputs/stage5/predictor_weights_v3_pd.csv`  
Rows: 97  
Computed per `(predictor_canonical, program, domain)` cell using same formula restricted to matching studies.

### Stability results

- LOSO stability: 32/32 groups, CV < 25%, mean CV = 6.6% (`outputs/stage5/weight_loso_stability.csv`)
- External stability (W_v2 vs W_v3): Pearson r = 0.9933, MAE = 0.028

---

## E. Existing Forecasting

### Models

| Model | Description | Code |
|-------|-------------|------|
| M0 | Unweighted Ridge — data-driven only, W_{j,p,d} = 1 | `scripts/compare_m0_m1_m1pd.py` |
| M1 | Global W_j weighted Ridge | `scripts/compare_m0_m1_m1pd.py` |
| M1_pd | Program×domain W_{j,p,d} weighted Ridge | `scripts/validate_m1pd.py` |
| M2 | Persistence baseline | `scripts/compare_m0_m1_m1pd.py` |

Ridge specification: sklearn `Ridge`, hyperparameter selected via validation fold, feature scaling applied with W-weighted scaling.

### Validation design

- LOCO / temporal expanding-window
- Country as the leave-out unit
- 45 folds (full panel), 18 folds (balanced panel)
- Files: `outputs/stage5/unified_panel.csv`, `outputs/stage5/validation_design_summary.csv`

### Final results (full panel, N=45 folds)

| Model | MAE |
|-------|----:|
| M2 Persistence | 40.19 |
| M0 Ridge (no lit) | 17.08 |
| M1 Global W_j | 17.02 |
| **M1_pd W_{j,p,d}** | **16.94** |

Balanced panel (N=18): M1_pd MAE = 12.97

### H1 test

- mean(ΔMAE M1_pd − M0) = +0.077 (positive = M1_pd wins)
- 95% CI: [+0.003, +0.151]
- wins/ties/losses: 28/4/13
- p_permutation = 0.017 *

---

## F. Existing Audit Artifacts

| Artifact | Location |
|----------|----------|
| Provenance audit | `outputs/stage5/provenance_audit.md` |
| Corpus provenance | `outputs/stage5/corpus_provenance_audit.csv` |
| Evidence provenance | `outputs/stage5/evidence_provenance_audit.csv` |
| LOSO stability | `outputs/stage5/weight_loso_stability.csv` |
| Final model comparison | `outputs/stage5/final_model_comparison.csv` |
| p-value discrepancy | `outputs/stage5/pvalue_discrepancy_audit.csv` |
| Evidence coverage | `outputs/stage5/evidence_coverage.csv` |
| Source coverage | `outputs/stage5/source_coverage_report.csv` |
| Canonical crosswalk | `outputs/stage5/canonical_crosswalk.csv` |
| Construct frequency | `outputs/stage5/construct_frequency.csv` |

---

## G. Frozen Baseline — Code Paths

| Component | File(s) |
|-----------|---------|
| Extraction schema | `src/schemas/models.py`, `src/schemas/findings_validation.py` |
| LLM extractor (OpenAI API) | `src/extractors/gpt_extractor.py`, `ilsa_pipeline/extractors/gpt_extractor.py` |
| PDF processor | `src/extractors/pdf_processor.py`, `ilsa_pipeline/extractors/pdf_processor.py` |
| Evidence matrix build | `scripts/build_evidence_matrix_v2.py`, `scripts/expand_evidence_matrix.py` |
| Canonical vocabulary | `scripts/build_canonical_taxonomy.py`, RAW_TO_CANONICAL in expand_evidence_matrix.py |
| W_j / W_jpd computation | `scripts/expand_evidence_matrix.py` (sections 5–6), `scripts/compute_lit_weight_sensitivity.py` |
| Forecasting / LOCO | `scripts/compare_m0_m1_m1pd.py`, `scripts/validate_m1pd.py` |
| Paired tests / permutation | `scripts/compute_permutation_tests.py` |
| LOSO stability | `scripts/compute_lit_weight_sensitivity.py` (or equivalent) |
| Panel construction | `scripts/build_unified_panel.py`, `scripts/build_enriched_panel_v2.py` |

---

## H. Fine-Tuning Experiment — What Remains Frozen

The following MUST NOT be changed during any fine-tuning experiment:

1. `outputs/stage5/` — all files, byte-for-byte
2. `outputs/stage5/predictor_weights_v3.csv` — W_j
3. `outputs/stage5/predictor_weights_v3_pd.csv` — W_{j,p,d}
4. `outputs/stage5/evidence_matrix_full_v2.csv` — current evidence matrix
5. `outputs/stage5/final_model_comparison.csv` — baseline results
6. `outputs/stage5/unified_panel.csv` — forecasting panel
7. `src/schemas/models.py` — extraction schema (fine-tuning uses this schema, not a new one)
8. `scripts/expand_evidence_matrix.py` RAW_TO_CANONICAL dict — canonical vocabulary
9. All validation fold definitions

The fine-tuning experiment lives entirely under `outputs/finetuning/`.

---

## I. Source Text Availability Assessment

Raw JSON files contain LLM-extracted structured outputs, NOT the source PDF text.

**Source PDFs:**
- `outputs/ilsa_survey_articles/json/` — 142 JSON files (extracted from PDFs processed by `ilsa_pipeline/extractors/pdf_processor.py`)
- Original PDFs would be in `data/` or equivalent — need to verify

**Implication for fine-tuning:** To construct supervised training examples with `(source_text → structured_JSON)` pairs, the original PDF text segments must be recoverable. This requires either:
1. Re-processing PDFs to extract raw text alongside structured output
2. Using the `standardized_conclusion` and `outcome_summary` fields as weak supervision proxies
3. Treating the LLM-generated outputs as silver labels without source-text alignment

**This is the critical blocker for full supervised fine-tuning.** Section 8 of the fine-tuning protocol requires source-text availability. A separate source-availability audit must be conducted before training data construction.

---

*Generated by: pre_finetuning_baseline_audit.py (manual inspection)*  
*Git branch: fix/stage5-dual-source-build-xy*  
*Date: 2026-10-05*
