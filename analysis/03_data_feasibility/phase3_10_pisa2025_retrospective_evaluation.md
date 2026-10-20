# Phase 3.10 — PISA 2025 Outcome-Exposed Retrospective Evaluation

**Date:** 2026-10-15  
**Status:** Executed under frozen historical model settings; not a blind test  
**Tabular run:** `outputs/pisa_2025_retrospective_tabular_v1/run_manifest.json`  
**TimesFM run:** `outputs/pisa_2025_retrospective_timesfm_v1/run_manifest.json`

**Validation update (2026-10-15):** The PISA 2025 point means in the 66-country evaluation sample have since been independently matched to OECD Volume I StatLink tables for all three domains (198/198; maximum absolute difference 0.01155 points). See `phase3_12_oecd_pisa2025_mean_validation.md` and `outputs/pisa2025_oecd_mean_audit_v3/audit_manifest.json`. This supersedes the source-validation caveat in the next paragraph; the exposure disclosure and retrospective interpretation remain in force.

## Exposure and interpretation

PISA 2025 country-level outcomes and prior model results were already present in the project before this run. Accordingly, this is an **outcome-exposed retrospective benchmark**, not a pristine held-out test. The run used the already-fixed historical RF, XGBoost, Ridge, TimesFM 2.5, and TimesFM 3.0 configurations; no 2025 metric was used to tune them in this run. This does not erase earlier project exposure and must be stated in all presentations or manuscripts.

The benchmark uses the common set of 66 countries with at least three pre-2025 PISA observations and a 2025 outcome in all three domains. The target is the country-level PISA score estimate, not individual student achievement. The panel's standard errors have prior validation evidence; the point means have now also been independently checked for these 66 countries, as documented in the validation update above.

## MAE and mean signed error

Positive signed error means overprediction. Lower MAE is better.

| Domain | Model | MAE | RMSE | Mean signed error |
|---|---|---:|---:|---:|
| Mathematics | XGBoost | **9.32** | 13.65 | +1.22 |
| Mathematics | Ridge | 9.46 | 12.63 | +3.31 |
| Mathematics | Random Forest | 11.56 | 15.17 | +5.93 |
| Mathematics | Persistence | 11.58 | 15.28 | +6.26 |
| Mathematics | TimesFM 3.0 zero-shot | 12.86 | 16.00 | +8.37 |
| Mathematics | Linear trend | 15.19 | 17.91 | +12.16 |
| Mathematics | TimesFM 2.5 zero-shot | 14.99 | 17.71 | +10.49 |
| Mathematics | TimesFM 2.5 ILSA LoRA | 14.83 | 17.56 | +10.40 |
| Reading | Ridge | **11.36** | 15.70 | +6.72 |
| Reading | XGBoost | 12.93 | 18.23 | +8.22 |
| Reading | Random Forest | 13.03 | 18.03 | +7.80 |
| Reading | Persistence | 15.22 | 19.32 | +10.64 |
| Reading | TimesFM 3.0 zero-shot | 16.75 | 20.44 | +13.09 |
| Reading | TimesFM 2.5 zero-shot | 17.85 | 22.08 | +15.07 |
| Reading | TimesFM 2.5 ILSA LoRA | 17.72 | 21.88 | +14.90 |
| Reading | Linear trend | 19.58 | 23.93 | +17.82 |
| Science | Persistence | **9.38** | 14.16 | −0.64 |
| Science | Ridge | 9.51 | 13.18 | −4.13 |
| Science | XGBoost | 9.73 | 13.92 | −3.37 |
| Science | Random Forest | 9.75 | 13.71 | −2.26 |
| Science | TimesFM 3.0 zero-shot | 10.08 | 13.88 | −0.31 |
| Science | TimesFM 2.5 zero-shot | 10.77 | 14.58 | +1.31 |
| Science | TimesFM 2.5 ILSA LoRA | 10.60 | 14.41 | +1.20 |
| Science | Linear trend | 10.58 | 13.72 | +0.70 |

## Main findings

1. The best method varies by domain: XGBoost leads mathematics, Ridge leads reading, and persistence leads science.
2. TimesFM 3.0 does not lead in any domain. It is relatively closer in mathematics, but has larger MAE than every tabular method except linear trend in reading and science.
3. TimesFM 2.5 ILSA LoRA has slightly lower MAE than its matched TimesFM 2.5 zero-shot comparator in all three domains: by 0.16 points in mathematics, 0.13 in reading, and 0.17 in science. Both TimesFM 2.5 arms trail the leading method in all three domains.
4. The adaptation effect is directionally consistent in the four historical folds and this outcome-exposed 2025 retrospective comparison, but remains under 0.2 MAE points. The paired country bootstrap is reported in Phase 3.13; it is exploratory and does not establish a practically important or generalizable transfer benefit.
5. The 2025 TimesFM 3.0 mathematics result reproduces the prior project output: MAE 12.864 versus the earlier rounded 12.86, and the maximum country-level prediction difference is about 0.00019. The point-score source has since been independently matched to OECD tables for all 66 evaluation countries and three domains; see Phase 3.12.

## Artifact validation

- 1,584 predictions: 8 methods × 3 domains × 66 countries.
- Every forecast is finite; country/domain/model keys are unique; all 24 domain/model metrics have `n=66`.
- The TimesFM 2.5 2025 adapter reloads from disk and reproduces a saved forecast within 0.001 score points.
- For every adaptation window, the target cycle is strictly before 2025. PISA 2025 scores are only attached after forecasts are produced for scoring; they are absent from contexts, training windows, and epoch selection.
- The 2025 adapter selects one epoch using 2023 as the inner validation target (141 examples), then refits on 1,433 eligible pre-2025 multi-ILSA windows.
- Run manifests record panel hashes, model checkpoint revisions, package versions, settings, eligible countries, and output names.

## Limitations and final claim

Historical panels provide only a handful of assessment steps per country, and different ILSA programs have different sampling designs, domains, and time intervals. TimesFM receives ordered score values but no explicit program/domain identifiers or elapsed-year features. The 2025 cohort is outcome-exposed; the point means for the 66-country evaluation sample are independently validated, while the original estimate-generation provenance remains incomplete. These results support a reproducible **retrospective comparison on this project panel**, not a blind forecast claim, individual-student prediction claim, causal claim, or universal ranking of models.
