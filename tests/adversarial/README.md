# Adversarial Test Scenarios

This directory documents edge-case PDFs used to stress-test the extraction pipeline.
Actual PDFs are not committed (copyright); this README describes each scenario and
the expected JSON output so tests can be reproduced or verified manually.

---

## How to use

1. Obtain or create a PDF matching the scenario description below.
2. Run the extractor: `python -m src.extractors.gpt_extractor <pdf_path> --output tests/adversarial/outputs/`
3. Compare the resulting JSON against the **Expected behaviour** column.
4. Flag any deviation as a regression.

---

## Scenario catalogue

### S-01 · No weights reported

**Description:** Empirical ML paper using PISA data; the methods section never mentions
sampling weights, `W_FSTUWT`, or replicate weights.

**Trigger fields:** `student_weights_used`, `replicate_weights_used`, `weight_variable_name`

**Expected behaviour:**
- `student_weights_used: false`
- `replicate_weights_used: false`
- `weight_variable_name: null`
- `weight_fields_interpretation` contains an explicit statement that weights are absent,
  not an empty string or filler sentence.

---

### S-02 · Multiple ILSA datasets in one paper

**Description:** Paper analyses both PISA 2018 and TIMSS 2019 in parallel sections,
reporting separate models for each.

**Trigger fields:** `main_findings[]`, `dataset_used`

**Expected behaviour:**
- `main_findings` has at least two entries, one per dataset.
- Each entry carries a distinct `dataset_used` value (e.g. `PISA 2018 Grade 9 Reading`
  vs `TIMSS 2019 Grade 8 Mathematics`).
- `outcome_summary` mentions both datasets.

---

### S-03 · Technical report / user guide (no empirical findings)

**Description:** Official PISA or TIMSS technical report describing sampling design,
weighting procedures, and scaling methodology — no predictive modelling.

**Trigger fields:** `source_category`, `main_findings`, `null_fields_interpretation`

**Expected behaviour:**
- `source_category: "technical_report"`
- `main_findings: []`
- `null_fields_interpretation` explains why findings are absent (not null).
- `plausible_values_handling: "not_applicable"`

---

### S-04 · Truncated PDF (> 400 000 characters)

**Description:** A very long report whose raw text exceeds `MAX_CHARS = 400_000`.
The extractor must silently truncate at the character limit and still return a valid JSON.

**Trigger:** `pdf_processor.py MAX_CHARS` truncation path.

**Expected behaviour:**
- Extraction completes without error.
- `outcome_summary` does not begin with `__EXTRACTION_FAILED__`.
- A truncation note may appear in `null_fields_interpretation` if key sections were cut.

---

### S-05 · Confounders listed only in a table (no prose mention)

**Description:** All predictor variables appear solely in a results table; the methods
section never names them in running text.

**Trigger fields:** `confounders_identified[]`

**Expected behaviour:**
- `confounders_identified` is populated from the table, not left empty.
- Each confounder has a valid `category` (no fallback to `other`).
- `variable_code` uses ILSA official codes where recognisable (e.g. `ESCS`).

---

### S-06 · Implausible sample size (single country, n > 1 000 000)

**Description:** Paper reports a suspiciously large student count (data entry error or
combined multi-cycle pool).

**Trigger fields:** `total_students`, `countries[].n_students`

**Expected behaviour:**
- The extractor reproduces the number as stated by the authors.
- No silent capping or correction.
- Reviewers should flag this manually during human validation.

---

### S-07 · Mixed-language PDF (English abstract, Turkish body)

**Description:** A thesis or report written primarily in Turkish with an English
abstract and summary.

**Trigger fields:** all text fields

**Expected behaviour:**
- All extracted text fields are in English (translated or abstracted by the model).
- `title` reflects the English title if one exists; otherwise a translated version.
- Extraction does not fail or produce garbled unicode.

---

### S-08 · DOI absent, preprint URL present

**Description:** Preprint (arXiv, SSRN) with a URL but no registered DOI.

**Trigger fields:** `doi`, `publication_type`, `open_access`

**Expected behaviour:**
- `doi: null` (URL is not a DOI).
- `publication_type: "preprint"`
- `open_access: true`

---

### S-09 · Plausible values averaged (methodological error)

**Description:** Authors explicitly state they averaged the five PVs and ran a single
regression, rather than applying Rubin's rules.

**Trigger fields:** `plausible_values_handling`

**Expected behaviour:**
- `plausible_values_handling: "average_pv"` (not `rubin_rules`).
- `outcome_summary` notes this as a methodological limitation.

---

### S-10 · Zero or negative R² reported

**Description:** Paper reports a negative out-of-sample R² indicating the model
performs worse than the mean predictor.

**Trigger fields:** `performance_metrics`, `standardized_conclusion`

**Expected behaviour:**
- The negative value is reproduced verbatim in `performance_metrics`.
- `standardized_conclusion` does not interpret this as a positive result.

---

## Adding a new scenario

1. Add a row above with the next `S-NN` identifier.
2. Describe the PDF, the trigger fields, and the expected JSON output.
3. If a real PDF is available, place it in `tests/adversarial/pdfs/` (gitignored) and
   note its filename in the scenario description.
