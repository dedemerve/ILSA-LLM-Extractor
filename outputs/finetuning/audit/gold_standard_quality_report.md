# Gold Standard Quality Report (Updated)

**Date:** 2026-10-05  
**Version:** v2 — replaces earlier version from commit 40de9da  
**Data:** `outputs/finetuning/data/gold_standard.csv`

---

## ⚠ TERMINOLOGY NOTE

The previous version of this report used the term "gold standard" for automatically validated records. This was incorrect.

**Correct terminology from this version forward:**

- `silver_verified_candidate` — source available + canonical match + auto-clean Tier-1/2 schema
- `silver_only` — canonical match but source unavailable or schema requires manual review
- `excluded` — no main_findings or not in canonical evidence matrix
- `gold_verified` — ONLY after human annotation (not yet performed)

**No records are gold_verified as of 2026-10-05.**

---

## 1. Corpus Summary

| Metric | Count |
|--------|------:|
| Total unique JSON files analyzed | 132 |
| Records with main_findings | 128 |
| Records with top_predictors | 112 |
| Records with DOI | 130 |
| Records in canonical evidence matrix | 105 |
| Source PDFs recovered | 10 |
| Source texts extracted | 10 |
| **silver_verified_candidate** | **9** |
| silver_only (source unavailable) | 103 |
| excluded | 20 |

---

## 2. Exclusion Breakdown

| Reason | N |
|--------|--:|
| Not in canonical evidence matrix (DOI not in evidence_matrix_full_v2.csv) | 16 |
| No main_findings | 4 |
| Total excluded | **20** |

---

## 3. Silver-Only Breakdown

| Reason | N |
|--------|--:|
| Source text unavailable (no PDF) | 112 |
| Tier-1/2 schema normalization needs manual review | ~3 |
| No top_predictors | ~1 |
| Total silver_only | **103** |

Note: some records have multiple reasons (source unavailable AND schema drift).

---

## 4. Silver-Verified Candidates — Detailed

These 9 records pass all automated checks AND have source text available. They are candidates for human annotation. They are NOT yet gold_verified.

| Study | DOI | Programs | N findings | PV handling | Missing data |
|-------|-----|----------|----------:|-------------|--------------|
| 1. Lee & Lee (2025) | 10.5281/zenodo.16778072 | TIMSS | 1 | not_applicable | not_reported |
| 10. Aydoğan & Tat (2025) | 10.21031/epod.1525454 | PISA | 1 | not_applicable | listwise_deletion |
| 3. Zhou et al. (2024) | 10.1016/j.heliyon.2024.e35945 | PISA | 1 | not_applicable | not_reported |
| 4. Zheng et al. (2023) | 10.1007/s10639-023-11908-0 | PISA | 1 | not_applicable | not_reported |
| 5. Song & Cutumisu (2024) | 10.1080/09500693.2024.2359099 | TIMSS | 1 | not_applicable | not_reported |
| 6. Elouafi (2025) | 10.3390/info16060480 | TIMSS | 1 | not_applicable | not_reported |
| 7. Alas & Tezer (2024) | 10.3390/su16177806 | PISA | 1 | not_applicable | not_reported |
| 8. Acıslı-Celik & Yesilkanat (2023) | 10.1007/s00521-023-08901-6 | PISA | 1 | not_applicable | not_reported |
| 9. Alkan et al. (2025) | 10.1371/journal.pone.0326121 | PISA | 3 | not_applicable | not_reported |

### Program distribution

| Program | N studies | % |
|---------|----------:|--:|
| PISA | 7 | 77.8% |
| TIMSS | 3 | 33.3% |
| PIRLS | 0 | 0% |
| TALIS | 0 | 0% |
| ICILS | 0 | 0% |
| ICCS | 0 | 0% |
| PIAAC | 0 | 0% |

**PISA dominance: 77.8%.** Non-PISA programs are severely under-represented.

### PV handling distribution (silver-verified candidates)

All 9 records have `plausible_values_handling = not_applicable`. This means these studies did NOT use standard ILSA plausible values procedures — they typically used single PV, WLE scores, or dichotomized outcomes. This is itself a methodological signal relevant to canonical mapping.

---

## 5. Full Corpus Field Distributions (N=132)

### Plausible values handling

| Value | N | % | Schema valid |
|-------|--:|--:|:-------------|
| not_applicable | 79 | 59.8% | ✓ |
| rubin_rules | 30 | 22.7% | ✗ → normalized to all_pvs |
| single_pv | 16 | 12.1% | ✓ |
| average_pv | 4 | 3.0% | ✗ → normalized to all_pvs |
| irt_theta | 1 | 0.8% | ✗ → not_applicable (manual review) |
| wle | 1 | 0.8% | ✗ → not_applicable (manual review) |
| all_pv | 1 | 0.8% | ✗ → normalized to all_pvs |

### Missing data handling

| Value | N | % | Schema valid |
|-------|--:|--:|:-------------|
| not_reported | 65 | 49.2% | ✓ |
| multiple_imputation | 26 | 19.7% | ✓ |
| listwise_deletion | 16 | 12.1% | ✓ |
| knn_imputation | 13 | 9.8% | ✗ → normalized to multiple_imputation |
| mean_imputation | 8 | 6.1% | ✓ |
| single_imputation | 4 | 3.0% | ✗ → mean_imputation (manual review) |

### Research design type

| Value | N | % | Schema valid |
|-------|--:|--:|:-------------|
| predictive | 83 | 62.9% | ✗ → cross_sectional (manual review) |
| exploratory | 47 | 35.6% | ✗ → cross_sectional (manual review) |
| causal_observational | 2 | 1.5% | ✗ → cross_sectional (manual review) |

**All 132 records have schema-drift research_design_type.** This is a systemic LLM output issue — the current schema Literal values (`cross_sectional|longitudinal|experimental|not_reported`) do not match the labels the LLM learned to produce. This is a Tier-3 field and does not block silver_verified_candidate status.

---

## 6. Class Imbalance Assessment

### For the 9 silver-verified candidates

N = 9 is too small for train/validation/test splitting. Minimum viable training set is ~50–100 supervised examples for reliable SFT.

**Current situation is insufficient for standard supervised fine-tuning.**

### Recommended paths forward

| Option | Feasibility | Notes |
|--------|-------------|-------|
| Obtain additional PDFs from authors or institutional access | Preferred | Would expand N to 64–105 if successful |
| Use 9 candidates for few-shot prompt evaluation only | Viable now | No gradient-based training required |
| Use silver_only labels with schema normalization applied | Possible but risky | LLM labels without source-text verification |
| Schema-constrained instruction evaluation (no training) | Viable now | Baseline vs alternative prompts |

---

## 7. Rare Category Assessment

| Construct | N studies (canonical evidence) | In silver_verified |
|-----------|-------------------------------:|:--------------------|
| SES_COMPOSITE | 54 | yes |
| GENDER | 50 | partial |
| SCHOOL_BELONGING | 27 | no |
| TEACHER_SELF_EFFICACY_CLS_MGMT | 21 | no |
| PARENTAL_EDUCATION | 7 | yes (TIMSS studies) |
| HOME_LITERACY_ACTIVITIES | 1 | no |
| JOB_SATISFACTION_PROFESSION | 1 | no |
| PRIOR_ACHIEVEMENT | 1 | no |

Teacher constructs (TALIS), literacy constructs (PIRLS), civic constructs (ICCS/ICILS) are entirely absent from the silver-verified set.

---

## 8. Annotation Priority Queue

If human annotation is to proceed, prioritize:

1. Articles 2–9 (already have source text): 9 candidates
2. If additional PDFs obtained: TIMSS-heavy articles first (to reduce PISA dominance)
3. Any article covering PIRLS/TALIS/ICILS/ICCS/PIAAC

---

*Generated: 2026-10-05 | Supersedes audit/gold_standard_quality_report.md from commit 40de9da*
