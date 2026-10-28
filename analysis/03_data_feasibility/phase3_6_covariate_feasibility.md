# Phase 3.6 — PISA Predictor and Covariate Availability

**Date:** 2026-10-15  
**Input:** `outputs/stage5/pisa_forecasting_panel_v2.csv`  
**Status:** Initial inventory complete; no model features have been selected or used  
**Integrity:** Read-only analysis of the new derived panel.

---

## 1. Scope

The proposed RF/XGBoost plan names prior PISA scores, GDP per capita, public education spending, student-teacher ratio, and enrollment. Before specifying a feature matrix, this audit checks what is actually present in the current PISA forecasting panel.

The inventory below covers PISA rows from 2003 through 2022 (1,406 long-form rows across mathematics, reading, and science). It reports raw non-missing coverage, not the smaller coverage after applying each model fold's country eligibility.

## 2. Available Lagged Fields

| Current panel field | Non-missing rows | Coverage | Modeling note |
|---|---:|---:|---|
| `lag_ESCS` | 360 | 25.6% | Candidate only; fold-level coverage and source timing need review. |
| `lag_HOMEPOS` | 360 | 25.6% | Candidate only; fold-level coverage and source timing need review. |
| `lag_BELONG` | 177 | 12.6% | Candidate only; low coverage. |
| `lag_ICTAVHOM` | 0 | 0.0% | Not usable from the current panel. |
| `lag_ICTAVSCH` | 0 | 0.0% | Not usable from the current panel. |
| `lag_BSDGEDUP` | 27 | 1.9% | Candidate only; very low coverage and coding direction needs review. |
| `lag_ASDHEDUP` | 24 | 1.7% | Candidate only; very low coverage and coding direction needs review. |
| `lag_ASDHELA` | 0 | 0.0% | Not usable from the current panel. |
| `lag_SECLSS` | 147 | 10.5% | Candidate only; low coverage. |
| `lag_TCDISCS` | 147 | 10.5% | Candidate only; low coverage. |
| `lag_TEFFPROS` | 144 | 10.2% | Candidate only; low coverage. |
| `lag_TJSPROS` | 147 | 10.5% | Candidate only; low coverage. |

The score columns `PISA_mathematics`, `PISA_reading`, and `PISA_science` are available for 1,406, 1,404, and 1,406 rows respectively. Historical score lags can be derived per canonical country and domain; they must be generated using only cycles strictly earlier than the forecast target.

### Coverage on Historical Evaluation Folds

The following counts use each fold's domain-specific countries with a target score and at least three earlier cycles. They show how many eligible countries have each feature in the current panel.

| Target | Domain | Eligible n | `lag_ESCS` | `lag_HOMEPOS` | `lag_BELONG` | Any teacher fields (`lag_SECLSS`, `lag_TCDISCS`, `lag_TEFFPROS`, `lag_TJSPROS`) |
|---:|---|---:|---:|---:|---:|---:|
| 2012 | Math / Reading / Science | 40 / 45 / 40 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| 2015 | Math / Reading / Science | 51 / 53 / 51 | 51 / 52 / 51 | 51 / 52 / 51 | 0 / 0 / 0 | 21 / 21 / 21 |
| 2018 | Math / Reading / Science | 57 / 57 / 57 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| 2022 | Math / Reading / Science | 60 / 61 / 60 | 55 / 56 / 55 | 55 / 56 / 55 | 54 / 55 / 54 | 25 / 25 / 25 |

The current contextual panel cannot support a balanced contextual-covariate model across all four rolling-origin folds: 2012 and 2018 have no non-missing lagged contextual fields on the eligible set. A separate covariate arm can be compared only on folds where the features exist, or the project must assemble additional historical covariates. It must not silently compare different fold/country samples against the score-only arm.

## 3. Proposed Macro-Education Covariates

The current forecasting panel contains no integrated fields for GDP per capita (PPP), education spending, student-teacher ratio, or enrollment. The project file inventory also surfaced no harmonized macro-covariate panel ready for the proposed folds.

These covariates therefore cannot be included in the first baseline as written. If added later, each needs a source, country mapping, year alignment, release/as-of date, missing-data rule, and a rule for mapping annual values to PISA assessment cycles. A value published after the forecast origin cannot be used for a historical prediction.

## 4. Design Consequences

1. Start with an achievement-history-only benchmark using lagged PISA domain scores and simple baselines.
2. Treat contextual ILSA lag fields as a separate exploratory feature arm; quantify fold-specific availability before deciding whether complete-case analysis or training-only imputation is viable.
3. Keep macro-education covariates out of the first RF/XGBoost run until source and historical availability are established.
4. Build every feature as-of the target origin. Current-cycle outcomes from another PISA domain must not enter a target-domain forecast unless that information would have been available at the declared forecast time.
5. For every fold, report country counts before and after feature availability filtering so a covariate model is not compared against a score-only model on an undisclosed sample change.

## 5. Next Gate

Before fitting RF/XGBoost, prepare a fold-specific feature availability table and freeze two distinct experiment arms: (A) historical achievement scores only; (B) score history plus any contextual variables that pass availability and coverage review. No feature should be retained solely because it improves a result on the exposed PISA 2025 benchmark.
