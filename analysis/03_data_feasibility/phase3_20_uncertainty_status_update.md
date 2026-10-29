# Phase 3.20 — Uncertainty Status Update

**Date:** 2026-10-15

**Current-state companion:** [Phase 3.18 evidence index](phase3_18_current_evidence_and_supersession_index.md)
**New evidence:** [Phase 3.19 PISA score uncertainty report](phase3_19_pisa_score_uncertainty.md)

This update preserves all earlier reports and outputs. Phase 3.19 is the new current evidence for the score-uncertainty gate listed in Phase 3.18; the other Phase 3.18 findings and completion limits still apply.

## Added and verified

- PISA published country score sampling SEs were joined to all historical held-out targets (2012, 2015, 2018, 2022) and to OECD's 2025 published country means/SEs for the 66-country three-domain comparison.
- A 20,000-draw target-score sampling-SE sensitivity analysis was run on saved predictions.
- A separate 20,000-repeat combined country-cluster bootstrap and target-score SE sensitivity was run for historical folds, PISA 2025 all 66, and a quality-clean 60-country PISA 2025 subset.
- Prediction keys, target joins, point MAE recomputations, and winner-probability normalization were checked. All passed. Existing prediction files were not modified.

## Updated interpretation

- Historical persistence remains the point-MAE leader for all domains. Under joint country-cluster and target-score uncertainty its winner frequency is approximately 64% in mathematics, 51% in reading, and 65% in science; it is not an unambiguous leader.
- TimesFM 2.5 LoRA's small reduction versus matched zero-shot remains uncertain in science once both sources of uncertainty are represented. The conditional bootstrap interval crosses zero there.
- PISA 2025 keeps the same domain-specific leaders under both 66- and 60-country joint sensitivity: mathematics is close between XGBoost and Ridge, reading favors Ridge, and science has no single clearly dominant model. All PISA 2025 findings remain outcome-exposed and retrospective.
- The score-uncertainty gate is **partially completed**. Published sampling SE is propagated, but link error and its cross-cycle covariance are not. OECD reports that 2025/2022 science score comparisons have a 2.9-point link error, separate from sampling SE; this is large enough that its omission materially limits temporal interpretation.

## Remaining open gates

1. Extract and validate domain-specific PISA scale-link errors and the required cross-cycle covariance from OECD technical documentation, then include them in a separate and combined sensitivity analysis. OECD's PISA 2022 report says trend uncertainty uses covariance between link errors across prior scales; independent score noise is not a substitute.
2. Propagate uncertainty in the historical PISA and ILSA context inputs through preprocessing, feature construction, model refitting, and forecast production; Phase 3.19 conditions on saved predictions.
3. Retain the distinct country-cluster, score-sampling, link-scale, and model-fit uncertainty components in reporting before any joint interval is presented.
4. Preserve the literature screening, second-reviewer adjudication, final full pipeline reproduction, manuscript consolidation, and future blind assessment-cycle evaluation gates from Phase 3.18.

## Artifacts

- Analysis code: [target score SE](../../scripts/summarize_pisa_target_sampling_uncertainty.py), [joint score-SE and country bootstrap](../../scripts/summarize_pisa_joint_score_country_uncertainty.py).
- Outputs: [target score SE results](../../outputs/pisa_target_sampling_uncertainty_v2/), [joint sensitivity results](../../outputs/pisa_joint_score_country_uncertainty_v2/).
- Primary OECD references: [PISA 2025 Reader's Guide](https://www.oecd.org/en/publications/pisa-2025-results-volume-i_73451bc5-en/full-report/reader-s-guide_0364240f.html), [PISA 2022 cross-cycle trend/link-error method](https://www.oecd.org/en/publications/2023/12/pisa-2022-results-volume-i_76772a36/full-report/comparing-mathematics-reading-and-science-performance-across-pisa-assessments_1c417d6f.html), [PISA 2022 Technical Report](https://www.oecd.org/content/dam/oecd/en/publications/reports/2024/03/pisa-2022-technical-report_599753f0/01820d6d-en.pdf).

**Readiness:** The executed retrospective benchmark remains useful for method comparison. It is not yet a complete prospective forecasting study, a fully uncertainty-propagated trend analysis, or an exhaustive literature review.
