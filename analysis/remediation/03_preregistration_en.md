# Phase 3 — Pre-registered decision rule for RQ2 (English translation)

> **Translation notice.** This is an English translation of `analysis/remediation/03_preregistration.md`. The Turkish original is the binding document. Its SHA-256 hash, `1713cbb6276797b8dd750d1b53ac2666677f388ef6d83df11e0073eaafa4fdb8`, is recorded in `outputs/rq2_v3/manifest.json` and was checked against the file on 27 October 2026. The hash covers the Turkish original only; this translation is not covered by it. Where the translation and the original differ, the original prevails. Translator's notes are marked [TN].

**Written:** 27 October 2026; before the Phase 3 analyses (placebo, permutation, data-driven penalty, feature selection) were run.
**Lock:** The sha256 value of this file is written to `outputs/rq2_v3/manifest.json` as `preregistration_sha256` before the analysis starts. The analysis script refuses to run if the file has changed.

**Transparency note.** The descriptive results of Phase 2 (the point MAE values of M0A, M0 and M1) were seen before this rule was written. The results of the placebo, permutation, data-driven penalty and KB-based feature-selection analyses were not seen. The rule was written following the text of item 3.5 of the prompt; it was not changed in response to the Phase 2 results. [TN: "the prompt" refers to the written task specification for the remediation analyses.]

## Scope

- **Unit:** The main program-domains (8): PISA mathematics, reading and science; TIMSS G8 mathematics and science; TIMSS G4 mathematics and science; PIRLS reading.
- **Sample:** The same complete folds and the same test rows as in `run_unified_backtest.py`.
  - **Primary analysis:** 2025 test cycle excluded.
  - **Sensitivity analysis:** 2025 included. The decision is made on the primary analysis only.
- **Additional programs (ICCS, ICILS, PIAAC):** Not covered by the rule; descriptive only, because they have a single fold.
- **Metric:** MAE (points) pooled over all test rows within a program-domain.

## Models compared (all the same Ridge, with the same inner time-ordered alpha selection)

| Name | Covariate weighting |
|---|---|
| M0A | No covariates (score features only) |
| M0 | All covariates, weight 1 |
| M1 | Covariates with KB v3 weight w_j, scaled by sqrt(w_j) |
| Uniform_c | w = c for all covariates, c ∈ {0.75, 0.5, 0.25} |
| Permutation null | 200 draws. In each draw, every covariate of the program is assigned a value drawn at random without replacement from the 12 positive weights in v3; the assignment is fixed across all folds within a program-domain (seed 20261027). This definition was chosen so that the null distribution is not degenerate in programs with a single covariate |
| DataDriven_block | Common covariate multiplier c ∈ {0, 0.1, 0.25, 0.5, 0.75, 1}; selected jointly with alpha by inner time-ordered validation |
| DataDriven_adaptive | From M0's standardized coefficients on the training data, w_j = abs(β_j) / max abs(β), lower bound 0.01; then Ridge scaled by sqrt(w_j) (adaptive ridge). Only training data are used |
| KB_top_k | The first k of the program's covariates by KB weight, k ∈ {1, 2, 3} (not exceeding the number of covariates); in each fold, among covariates that carry training signal |
| Random_top_k null | 200 draws. The same top-k rule applied to a random ordering of the covariates |

## Decision rule (for each program-domain, primary analysis)

The KB is said to **improve forecasts** in a program-domain only if **all** three of the following conditions hold:

1. **M1 is significantly and practically better than M0A:** ΔMAE(M0A − M1) ≥ 1.0 point, and the lower bound of the country-cluster bootstrap 95% confidence interval of this difference > 0 (Phase 4; 2,000+ replicates, Holm correction across the 8 program-domains).
2. **The content of the KB matters:** MAE(M1) is smaller than the 5th percentile of the permutation null distribution; that is, M1 is better than at least 95% of random weight assignments.
3. **M1 is better than M0:** ΔMAE(M0 − M1) > 0.

**Overall RQ2 claim:**
- The statement "the KB consistently improves forecasts" is used only if the rule is met in at least 6 of the 8 program-domains.
- If the rule is met in 1–5 program-domains: "improvement only in these program-domains".
- If it is met in none: "no evidence that KB integration contributes to forecasting".
- If the TOST in Phase 4 shows that the M1 − M0A difference lies within ±1 point, the statement "practically equivalent" is added.

**Secondary route (feature selection):** If the MAE of KB_top_k is smaller than the 5th percentile of the Random_top_k null distribution and at least 1 point better than M0A, it is stated that "KB-based feature selection helps". The same Holm correction is applied. This route does not change the primary decision; it is reported separately.

**Interpretation of placebos (descriptive):** If one of the Uniform_c or DataDriven models is better than M1, any gain of M1 over M0 is attributed to generic covariate shrinkage, not to the content of the KB.

## Fixed details

- Bootstrap: country-cluster; countries are resampled within a program-domain and all of that country's rows across folds are taken together; seed 20261027.
- Multiple comparisons: Holm (m = 8) for the p-values of Condition 1.
- Equivalence margin: ±1.0 point. The rationale is documented in Phase 4 with the median SE of country means (about 2.5) and published link-error values; the margin does not change.
- If the rule is changed after the run, the change and its rationale are appended to this file; the original text is not deleted.
