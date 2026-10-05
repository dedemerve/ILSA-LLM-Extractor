# Human Annotation Reliability Report

**Date:** 2026-10-05  
**Status:** NOT YET CONDUCTED — this document records the required protocol for when annotation is performed

---

## 1. Current Annotation Status

**Human annotation has not been performed.**

The existing labels were generated entirely by the GPT-based LLM extractor and passed through automated Pydantic schema validators. No researcher has manually reviewed or verified any extraction record against its source text.

All 9 `silver_verified_candidate` records in `outputs/finetuning/data/gold_standard.csv` are:
- automatically validated by schema constraints
- matched to canonical evidence matrix by DOI
- NOT human-verified

They must be labelled "automatically validated silver-standard candidates" until human annotation is complete.

---

## 2. Annotation Codebook (Required Before Annotation)

The following fields must be annotated per record:

### Tier-1 — Forecasting-critical (primary annotation target)

| Field | Type | Values |
|-------|------|--------|
| `program` | Categorical | PISA \| TIMSS \| TIMSS_G4 \| PIRLS \| TALIS \| ICCS \| ICILS \| PIAAC |
| `cycle` | Integer | Year of ILSA cycle |
| `target_variable` | Free text | What outcome was predicted (e.g., "mathematics achievement", "reading literacy") |
| `predictor_raw` | Free text | Exact predictor label as used in the source |
| `canonical_construct` | Categorical | One of 19 canonical constructs (from evidence_provenance_audit.csv) |
| `direction_original` | `+` \| `-` \| `?` | Direction of association in the source |
| `direction_canonical` | `+` \| `-` \| `?` | After scale reversal if needed |
| `effect_type` | Categorical | predictive \| associative \| descriptive |
| `source_location` | Free text | Section and sentence in source confirming the claim |

### Tier-2 — Evidence strength

| Field | Type | Values |
|-------|------|--------|
| `performance_metrics` | Free text | Model accuracy, R², AUC, etc. |
| `ml_technique_primary` | Free text | Primary ML technique used |
| `standardized_conclusion` | Free text | 2–3 sentence summary |

### Tier-3 — Methodology

| Field | Type | Values |
|-------|------|--------|
| `plausible_values_handling` | Categorical | single_pv \| all_pvs \| not_applicable \| not_reported |
| `student_weights_used` | Boolean | true \| false \| null |
| `missing_data_handling` | Categorical | listwise_deletion \| multiple_imputation \| mean_imputation \| not_reported \| not_applicable |
| `research_design_type` | Categorical | cross_sectional \| longitudinal \| experimental \| not_reported |

### Annotation ground rules

- Do NOT infer information not stated in the source text
- Use `not_reported` when the source is silent
- Use `unknown` when the source is ambiguous
- Use `not_applicable` when the field does not apply to this study design
- For reversed scales: always record `direction_original` first, then `direction_canonical`
- A paper studying PISA 2018 data MUST be coded as `program=PISA` and `cycle=2018`
- Confounders that are not top-ranked predictors must be coded as `effect_type=associative`

---

## 3. Sampling Protocol

Human annotation should be stratified across:

| Dimension | Target representation |
|-----------|----------------------|
| PISA | ≤50% of gold set |
| TIMSS | ≥15% |
| PIRLS | ≥10% (if source available) |
| TALIS | ≥10% (if source available) |
| ICILS/ICCS/PIAAC | Best effort |
| Predictive design | ≥60% |
| PV handling: all_pvs | ≥20% |
| Direction: negative | ≥15% |
| Reversed-scale construct | ≥2 examples |

**Current gap:** No PIRLS, TALIS, ICILS, ICCS, PIAAC source texts available. If PDFs are obtained for these programs, prioritize them.

---

## 4. Reliability Protocol

### If two annotators available

1. Both annotate independently from source text
2. Calculate Cohen's κ for each categorical Tier-1 field
3. For direction: κ ≥ 0.80 required; below this, adjudicate
4. For canonical_construct: κ ≥ 0.75 required
5. For program/cycle: κ should be 1.0 (deterministic from source)
6. Disagreements: discuss to consensus; record adjudication rationale

### If single annotator (current situation)

1. Annotate all records (first pass)
2. Wait minimum 7 days (washout period)
3. Re-annotate without reference to first-pass labels (second pass)
4. Calculate self-agreement rate per field
5. Fields with self-agreement < 80%: flag as ambiguous, excluded from gold

**Do not call single-annotator reannotation "inter-rater reliability."**  
Report it as "intra-rater agreement after washout."

---

## 5. Adjudication Rules

| Conflict type | Rule |
|---------------|------|
| Program disagreement | Source title/abstract is deterministic → recheck source |
| Cycle disagreement | ILSA data year is deterministic → recheck source |
| Direction disagreement | Examine scale description in source; default to `?` if ambiguous |
| Canonical construct | Use most specific construct; if ambiguous, record both and exclude from gold |
| Effect type | "predict" language → predictive; "associated" → associative; "causal" → flag for review |

---

## 6. Expected Outcomes

Once annotation is complete, create:

```
outputs/finetuning/data/gold_standard.csv
```

with `gold_status = "gold_verified"` for records passing human review.

Report:
- Total annotated
- κ / self-agreement per field
- Disagreement types and frequencies
- Final gold count
- Excluded as ambiguous

---

*This document is a protocol template. Fill in actual statistics when annotation is performed.*  
*Date completed: [PENDING]*
