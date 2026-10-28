# Prospective PISA forecast lock protocol — v1

**Status:** Prepared for PISA 2029 execution; not yet a forecast or preregistration. OECD schedules the PISA 2029 main survey for 2029 and results publication for 2030; the exact release date remains to be announced. This protocol becomes prospective only after its content, code commit, and forecast files are time-stamped and sealed before PISA 2029 target outcomes are accessible to the team. ([OECD PISA 2029 brochure](https://www.oecd.org/content/dam/oecd/en/about/programmes/edu/pisa/publications/brochures/PISA-2029-Brochure.pdf))

## 1. Target and estimand

- **Target:** PISA 2029, the next cycle after the PISA 2025 outcomes used in project development.
- **Unit:** country/economy × subject-domain mean score, not individual student score.
- **Domains:** mathematics, reading, and science; report each separately and an equally weighted three-domain summary.
- **Population:** participating systems meeting the eligibility rule below; do not generalize to all countries.
- **Horizon:** one PISA assessment cycle ahead.

Before sealing, record the exact forecast freeze timestamp, public data cutoff, and event that counts as outcomes becoming accessible. Current OECD timeline expects core PISA 2029 results in 2030, but the exact release date is not yet specified; update this field only from an official OECD announcement, before any target outcome is accessed.

## 2. Eligibility and missingness

1. Use the frozen panel's ISO3/system identifiers and audited crosswalk. Store source and normalized IDs in the prediction file.
2. Define eligible systems before fitting: at least three prior observed cycles for each domain; use a single shared three-domain sample for the primary comparison.
3. Do not silently impute a missing target score. Report the number and reasons for systems excluded by each rule.
4. Publish a secondary domain-specific available-case table, clearly distinguished from the common-sample primary analysis.

## 3. Frozen information set

For every forecast, only use information demonstrably public or available by the recorded forecast cutoff. Freeze source files and SHA-256 hashes. Do not use target-cycle scores, later-released context data, post-cutoff macroeconomic observations, or country mappings revised after the result is known. The primary feature set uses historical PISA domain means only; covariate models remain a separately registered secondary arm with a documented as-of date for every feature.

## 4. Model set and fitting

Freeze code, package versions, seeds, checkpoints, hyperparameters, and model-to-input mapping before target outcomes are accessed. The planned comparison is:

- Persistence (latest observed cycle), the mandatory reference model.
- Linear trend and the registered linear/quadratic trend-selection baseline.
- Ridge, Random Forest, and XGBoost with the currently fixed settings.
- TimesFM 3.0 zero-shot.
- TimesFM 2.5 PISA/ILSA-adapted arm with its matched zero-shot comparator, only if the exact checkpoint, adaptation pool, and executable environment can be archived and independently reproduced by the lock date.

No model, hyperparameter, feature, cohort, or selection rule may be changed after target outcomes become accessible. Record negative or infeasible model runs. Model ranks on 2025 remain retrospective development evidence and cannot serve as prospective validation.

## 5. Prediction sealing and access control

Before any target-cycle outcomes are inspected:

1. Save one row per eligible system/domain/model with point prediction, input cutoff, context cycles, source hashes, model/checkpoint ID, code commit, and run manifest.
2. Save an immutable archive of all prediction files and manifests; compute SHA-256 hashes and record a trusted timestamp (for example, a public repository tag or institutional repository deposit).
3. Store the sealed predictions separately from the target outcomes and document who can access each.
4. Do not amend the sealed forecast after release. Any corrected file must be a new version with the previous one retained and the reason documented before evaluation.

## 6. Primary evaluation and uncertainty

- **Primary metric:** MAE, first averaged equally across eligible systems within each domain, then equally across the three domains. Report the three domain MAEs and the three-domain macro-average.
- **Secondary metrics:** RMSE, mean signed error, and median absolute error.
- **Primary contrasts:** each model's paired absolute-error difference against Persistence on exactly the same system/domain rows. Report effect sizes and uncertainty intervals; do not select a winner from point estimates alone.
- **Resampling:** paired country/economy-cluster bootstrap, preserving the three domains jointly within a system. Publish the seed, draw count, and interval construction.
- **Score uncertainty:** include target-score sampling SE sensitivity where official SEs are available. Report this separately from forecast error.
- **Scale-link uncertainty:** do not claim full link-adjusted intervals unless the exact cross-cycle covariance inputs are validated, positive semidefinite, and reproduce official pairwise and trend link-error checks. Until then, state the omitted component explicitly and provide no independence-based substitute.

## 7. Evaluation lock and reporting

After results are officially released, first verify the target outcomes and country crosswalk against primary OECD material. Then run the sealed evaluation code without modifying forecasts or analysis choices. Report all eligible systems, excluded systems, model failures, uncertainty limits, and all registered metrics. Label the result as a single prospective cycle; a single future target cannot establish universal model superiority or performance across future cycles.

## 8. Pre-release checklist to complete

| Lock item | Required value before sealing |
|---|---|
| Target PISA cycle and expected release | PISA 2029; core results expected in 2030, exact date TBD |
| Forecast cutoff timestamp and timezone | TBD |
| Team members with outcome access | TBD |
| Frozen panel and source hashes | TBD |
| System eligibility list and crosswalk hash | TBD |
| Code commit and environment lockfile | TBD |
| Model/checkpoint hashes and seeds | TBD |
| Predictions archive SHA-256 and trusted timestamp | TBD |
| Evaluation script hash | TBD |
| Approved deviations (must be none after outcome access) | TBD |

Until every required value is completed and the prediction artifact is sealed, this is a prepared protocol, not independent prospective evidence.
