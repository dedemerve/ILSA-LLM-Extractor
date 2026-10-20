# Phase 3.26 — Model-Input Uncertainty Analysis

**Date:** 2026-10-20
**Status:** COMPLETE — corrected full refit-aware run validated (2,000 draws)
**Script:** `scripts/run_pisa_model_input_uncertainty_v4.py`  
**Output dir:** `outputs/pisa_model_input_uncertainty_v4/`

The earlier v1 attempt did not produce a completed output. The v2 smoke run held fitted models fixed and is not the refit-aware analysis specified here. The v3 attempt is invalid: it failed to filter the long panel to the matching domain, duplicating country-cycle scores as artificial repeated observations. Its 125 completed draws are retained only for audit and must not be used. See `outputs/pisa_model_input_uncertainty_v3/INVALID_RUN.md`. The corrected v4 filters each domain before constructing country histories, perturbs training inputs and labels, refits tabular models within each draw, and scores against fixed published held-out targets. It writes resumable checkpoints every 25 draws.

---

## 1. Purpose

Prior uncertainty analyses (Phases 3.19–3.25) perturbed only the **target** scores (the PISA
country means being predicted) while holding model input scores fixed.  This analysis adds the
complementary component: perturbing the **context input** scores that the tabular models use to
construct prediction features (lag1, lag2, score_change, etc.).

This complements the held-out target-score analysis; the separate analyses do not by
themselves form a joint interval. This phase does **not** cover:
- OECD scale-link covariance across cycles (Phase 3.23: unresolved for science)
- TimesFM model-input uncertainty (GPU infeasible to refit per draw)
- Model specification or hyperparameter uncertainty
- Prospective or blind evaluation (PISA 2025 excluded throughout)

---

## 2. Design

| Item | Value |
|------|-------|
| Draws (N_DRAWS) | 2 000 |
| Seed | 20261015 |
| Input perturbation | One draw from N(published mean, marginal SE²) per country/domain/cycle; reused across folds and roles |
| SE source | `outputs/stage4/country_estimates.csv`, PISA rows, column `se` |
| Identifier correction | Verified 2000 reading crosswalk: `analysis/03_data_feasibility/audit/pisa_2000_reading_country_crosswalk.csv` |
| Target score | Held-out target fixed at published mean; historical training labels are perturbed |
| Historical folds | 2012, 2015, 2018, 2022; frozen common three-domain country sets |
| Models | Persistence, LinearTrend, Ridge, RandomForest, XGBoost |
| Refit | Ridge, RandomForest and XGBoost refit for each draw × domain × target fold |
| TimesFM | Excluded from this refit-aware input analysis |
| Primary metric | Equal-fold-weighted MAE, averaged across 4 folds |
| Comparison | Each model minus Persistence (paired per draw) |

Each draw samples the available historical score estimates from marginal normal distributions.
The same sampled country/domain/cycle score is reused when it appears as a training label or
historical context. Ridge, RF, and XGBoost are refit on that draw's perturbed training examples;
Persistence and LinearTrend are recomputed from the perturbed context. All methods are scored
against the fixed published mean for the held-out target. Single-threaded fitting was checked
against parallel fitting on the same seeded draw; the largest fold/model MAE difference was
3.2×10⁻¹⁴ points.

Preflight checks confirmed that the panel SHA-256 matches both its build manifest and the frozen
fold manifest. After filtering by domain, each country-cycle score key is unique. All 1,449
historical panel means exactly match the corresponding `country_estimates.csv` means after the
PISA 2000 reading crosswalk, and every one has a positive published SE. A two-draw smoke run
confirmed the corrected pipeline and complete model/fold keys; it is not included in result tables.

---

## 3. SE Coverage

SE coverage verified before run after applying the numeric-to-alpha-3 PISA 2000 reading crosswalk.

| Domain | Observations with SE > 0 | Total observations |
|--------|--------------------------|-------------------|
| Mathematics | 469 | 469 |
| Reading | 511 | 511 |
| Science | 469 | 469 |

---

## 4. Results

### 4.1 Equal-fold MAE distributions (2 000 draws)

Values below summarize the draw-wise equal-fold MAE distribution. “Point MAE” is the mean
across the 2,000 draws; q2.5, median, and q97.5 are Monte Carlo quantiles under the stated
marginal-SE sampling model. These are not full predictive confidence intervals.

| Domain | Model | Point MAE | 2.5th pct | Median | 97.5th pct |
|--------|-------|-----------|-----------|--------|------------|
| Mathematics | Persistence | 9.639 | 9.287 | 9.635 | 9.986 |
| Mathematics | Ridge | 10.164 | 9.651 | 10.161 | 10.714 |
| Mathematics | RandomForest | 10.140 | 9.668 | 10.141 | 10.628 |
| Mathematics | XGBoost | 10.768 | 10.049 | 10.766 | 11.539 |
| Mathematics | LinearTrend | 11.616 | 11.085 | 11.616 | 12.147 |
| Reading | Persistence | 10.435 | 10.077 | 10.436 | 10.783 |
| Reading | Ridge | 11.010 | 10.549 | 11.005 | 11.492 |
| Reading | RandomForest | 11.061 | 10.596 | 11.055 | 11.520 |
| Reading | XGBoost | 11.238 | 10.519 | 11.231 | 11.969 |
| Reading | LinearTrend | 13.658 | 13.176 | 13.651 | 14.170 |
| Science | Persistence | 8.730 | 8.384 | 8.725 | 9.078 |
| Science | Ridge | 9.366 | 8.874 | 9.353 | 9.929 |
| Science | RandomForest | 9.363 | 8.903 | 9.356 | 9.841 |
| Science | XGBoost | 9.769 | 9.128 | 9.768 | 10.449 |
| Science | LinearTrend | 11.127 | 10.615 | 11.119 | 11.673 |

### 4.2 Paired comparisons: model MAE minus Persistence (equal-fold, 2 000 draws)

Positive deltas indicate higher (worse) MAE than Persistence. “P(better)” is the proportion
of paired draws with a negative delta; it is conditional on this sampling design, not a
frequentist hypothesis-test p-value.

| Domain | Model | Point Δ | 2.5th pct | 97.5th pct | P(model better) |
|--------|-------|---------|-----------|------------|-----------------|
| Mathematics | Ridge | +0.525 | +0.063 | +1.020 | 1.30% |
| Mathematics | RandomForest | +0.501 | +0.074 | +0.958 | 1.15% |
| Mathematics | XGBoost | +1.129 | +0.421 | +1.831 | 0.05% |
| Mathematics | LinearTrend | +1.977 | +1.491 | +2.456 | 0.00% |
| Reading | Ridge | +0.575 | +0.193 | +0.990 | 0.10% |
| Reading | RandomForest | +0.626 | +0.232 | +1.025 | 0.20% |
| Reading | XGBoost | +0.803 | +0.112 | +1.524 | 1.25% |
| Reading | LinearTrend | +3.223 | +2.762 | +3.694 | 0.00% |
| Science | Ridge | +0.636 | +0.197 | +1.125 | 0.10% |
| Science | RandomForest | +0.633 | +0.245 | +1.026 | 0.10% |
| Science | XGBoost | +1.039 | +0.411 | +1.654 | 0.05% |
| Science | LinearTrend | +2.397 | +1.958 | +2.863 | 0.00% |

### 4.3 Winner probabilities

| Domain | Persistence | Ridge | RandomForest | XGBoost | LinearTrend |
|--------|-------------|-------|--------------|---------|-------------|
| Mathematics | 97.50% | 1.30% | 1.15% | 0.05% | 0.00% |
| Reading | 98.50% | 0.10% | 0.20% | 1.20% | 0.00% |
| Science | 99.75% | 0.10% | 0.10% | 0.05% | 0.00% |

Across these historical folds and this input-score sampling design, Persistence has the
lowest equal-fold MAE in nearly all draws. This is a robustness result conditional on the
fixed fold/sample/model setup; it does not establish universal superiority or prospective
performance. The tree and Ridge models were retrained on every sampled dataset, so their
variation includes the effect of perturbed historical features and training labels.

### 4.4 Run integrity and reproducibility

The completed run contains 80 batch checkpoints and 120,000 fold-level metrics
(2,000 draws × 4 target cycles × 3 domains × 5 models), covering draws 0–1,999. There are
no duplicate draw/fold/domain/model keys. Each summary row is based on 2,000 draws, and
winner probabilities sum to 1.0 within each domain. The final manifest records the input,
script, and output SHA-256 hashes and software versions. The checkpoint and summary files
are under `outputs/pisa_model_input_uncertainty_v4/`.

---

## 5. Relationship to prior uncertainty analyses

| Analysis | What is perturbed | Target fixed? | Models retrained? |
|----------|-------------------|---------------|-------------------|
| Phase 3.19–3.22 (target SE only) | Target scores | — | No |
| Phase 3.25 (joint) | Target scores + country cluster resampling | — | No |
| **Phase 3.26 (this)** | **Historical context and training-label scores** | **Yes** | **Yes (Ridge/RF/XGB)** |

These are separate conditional analyses, not a jointly propagated interval. Phase 3.25 perturbs
held-out targets while retaining saved predictions. Phase 3.26 perturbs historical training/context
scores and refits tabular models while keeping held-out targets fixed. The score draws use marginal
standard errors and do not represent known cross-cycle link-error covariance; the two analyses must
not be combined as if their components were independent.

---

## 6. Limits

- TimesFM input uncertainty is excluded (GPU infeasible).
- OECD scale-link covariance across cycles remains unresolved for science.
- The four historical folds share countries across time; they are not independent observations.
- 2025 retrospective is not re-evaluated here; all results are historical temporal validation only.
- Model specification (hyperparameters, feature set) is fixed; this is not a model-selection
  uncertainty analysis.

The 95% ranges in Sections 4.1–4.2 describe Monte Carlo variation induced by independently
sampling each historical country/domain/cycle mean from its published marginal normal
distribution. They do not include cross-country or cross-cycle dependence, OECD scale-link
covariance, held-out target-score uncertainty, or future-cycle variation. Accordingly, this
phase should be interpreted alongside Phase 3.25 as a separate conditional sensitivity
analysis; the two ranges must not be combined into a single interval.
