# Gold Standard Quality Report

**Date:** 2026-10-05  
**Scope:** ilsa_survey_articles corpus (targeted ILSA ML survey, N=142 JSON including duplicates; N=132 unique after removing `(1)` copy files)  
**Purpose:** Assess suitability of existing LLM-extracted JSON records as training labels for domain-specific fine-tuning.

---

## 1. Candidate Study Population

| Count | Description |
|------:|-------------|
| 142 | Total JSON files in `outputs/ilsa_survey_articles/json/` |
| 10 | Duplicate `(1)` copies (excluded) |
| **132** | Unique JSON records analyzed |
| 130 | Records with DOI present (98.5%) |
| 2 | Records without DOI |
| 128 | Records with `main_findings` populated (97.0%) |
| 4 | Records with empty `main_findings` |
| 112 | Records with non-empty `top_predictors` in at least one finding |
| 203 | Total finding rows across all 132 records |
| 105 | Survey JSONs mapping to canonical study_ids in evidence_matrix_full_v2 |
| 27 | Survey JSONs NOT appearing in canonical evidence matrix |

---

## 2. Label Source Assessment

The existing JSON extractions were produced by the GPT-based extractor (`src/extractors/gpt_extractor.py`) using the ILSAArticleMetadata schema (Pydantic v2). **No human annotation has been applied post-hoc.** The quality of labels depends on:

- Extraction prompt quality (structured, schema-constrained via Pydantic validation)
- Automated Pydantic validators (bool coercion, NA guard, category Literal enforcement)
- Deterministic post-processing in `build_evidence_matrix_v2.py` and `expand_evidence_matrix.py`

**No record in this corpus has been human-verified beyond automated validation.**

### Validation tiers applied

| Tier | Mechanism | Coverage |
|------|-----------|---------- |
| Schema validation | Pydantic field validators, Literal types | 100% |
| Bool coercion | `coerce_optional_bool()` | 100% |
| NA guard | `variable_code` cannot be 'N/A' | 100% (confounder records) |
| DOI prefix stripping | validator `strip_doi_prefix` | 130/132 |
| Category enforcement | 13-value Literal for confounder.category | 100% |
| Canonical mapping | RAW_TO_CANONICAL dict | ~79.5% (105/132) |
| Human review | None | 0% |

---

## 3. Field-Level Coverage

### Plausible values handling distribution (N=132)

| Value | N | % |
|-------|--:|--:|
| not_applicable | 79 | 59.8% |
| rubin_rules | 30 | 22.7% |
| single_pv | 16 | 12.1% |
| average_pv | 4 | 3.0% |
| irt_theta | 1 | 0.8% |
| wle | 1 | 0.8% |
| all_pv | 1 | 0.8% |

**Note:** `rubin_rules` and non-standard values (`irt_theta`, `wle`, `all_pv`) are outside the current schema Literal definition (`single_pv|all_pvs|not_applicable|not_reported`). These represent schema drift or LLM free-form output that passed at extraction time. They must be harmonized before use as training labels.

### Missing data handling distribution (N=132)

| Value | N | % |
|-------|--:|--:|
| not_reported | 65 | 49.2% |
| multiple_imputation | 26 | 19.7% |
| knn_imputation | 13 | 9.8% |
| listwise_deletion | 16 | 12.1% |
| mean_imputation | 8 | 6.1% |
| single_imputation | 4 | 3.0% |

**Note:** `knn_imputation` and `single_imputation` are outside current schema Literal definition. Schema drift present.

### Research design type distribution (N=132)

| Value | N | % |
|-------|--:|--:|
| predictive | 83 | 62.9% |
| exploratory | 47 | 35.6% |
| causal_observational | 2 | 1.5% |

**Note:** `causal_observational` is outside current Literal definition (`cross_sectional|longitudinal|experimental|not_reported`). Schema drift present.

### Student weights used (N=132)

| Value | N | % |
|-------|--:|--:|
| False (no weights) | 86 | 65.2% |
| True (weights applied) | 17 | 12.9% |
| Unknown/None | 29 | 22.0% |

### Program distribution in findings (N=203 finding rows)

| Program | N findings |
|---------|----------:|
| PISA | 143 |
| TIMSS | 15 |
| PIRLS | 9 |
| PIAAC | 8 |
| TALIS | 2 |
| ICCS | 1 |
| ICILS | 1 |

**Strong PISA dominance.** Non-PISA programs are rare in this sub-corpus.

---

## 4. Label Quality Assessment

### Issues identified

| Issue | Count | Action |
|-------|------:|--------|
| Schema drift (non-Literal PV values) | ~37 records | flag as `needs_review` |
| Schema drift (non-Literal missing_data) | ~17 records | flag as `needs_review` |
| Schema drift (non-Literal research_design) | 2 records | flag as `needs_review` |
| Not in canonical evidence matrix | 27 records | flag as `excluded_from_gold` |
| No main_findings | 4 records | `excluded_from_gold` |
| No DOI | 2 records | flag, keep if title dedup clean |
| Human verification | 0 records | No record is human-verified |

### Gold standard construction rules

Since NO records have human review, the gold standard is constructed by:

1. **Including** records that: (a) passed Pydantic schema validation, (b) have DOI, (c) have non-empty `top_predictors`, (d) map to the canonical evidence matrix (105/132), (e) have NO schema drift in Tier-1 fields (program, cycle, target_variable, predictor, direction)
2. **Flagging as `needs_review`**: records with non-standard Literal values in any Tier-1 field
3. **Excluding**: records with empty main_findings, not in canonical mapping, or with contradictory evidence

### Estimated gold standard size

| Category | N | Label |
|----------|--:|-------|
| Candidate records | 132 | — |
| No main_findings | 4 | `excluded_from_gold` |
| Not in canonical evidence matrix | 27 | `excluded_from_gold` |
| Schema drift (estimated overlap) | ~15 additional | `needs_review` |
| **Estimated gold standard** | **~86** | `gold` |
| **Needs review** | **~20** | `needs_review` |
| **Excluded** | **~26** | `excluded_from_gold` |

**This estimate requires verification.** The actual gold standard CSV must be built by running the construction script.

---

## 5. Source Text Availability — Critical Constraint

For supervised extraction fine-tuning, each training example requires a `(source_text, structured_output)` pair. The current corpus only stores the structured output (JSON). Original PDF text was consumed during extraction but is NOT stored in the JSON files.

| Source text status | N records (estimated) |
|--------------------|----------------------:|
| `source_available` (PDF re-processable) | Unknown — requires PDF file audit |
| `source_missing` (no PDF, JSON only) | Unknown |
| `source_partial` (summary fields only) | 132 (standardized_conclusion, outcome_summary available) |

**Without source text, supervised fine-tuning in the strict sense is not possible.** Alternative approaches:

1. **Re-process PDFs** if originals are present in `data/` directory
2. **Use silver-label self-training**: generate new LLM outputs on held-out PDFs, use existing extractions as reference
3. **Instruction fine-tuning on schema**: use structured JSON outputs alone as target, with schema description as context

This limitation MUST be resolved before constructing `train.jsonl`.

---

## 6. Recommended Next Steps

1. Audit PDF availability in `data/` to determine `source_available` count
2. Harmonize schema drift values before labeling
3. Build `outputs/finetuning/gold_standard.csv` with `label_quality` column
4. If source texts available: construct `(source_text → JSON)` pairs for ~86 gold records
5. If source texts unavailable: document as limitation, proceed with schema-only instruction format

---

*Generated by: manual inspection of ilsa_survey_articles corpus*  
*Date: 2026-10-05*
