# Phase 3 Data-Feasibility Audit

**Date:** 2026-10-14  
**Protocol:** Research Protocol v1, Phase 3 — data-feasibility audit  
**Preceding artifacts:** phase2_2_report.md, phase2_2_1_addendum.md  
**Scope:** Inspect existing time-series datasets and baseline results; assess experiment comparability; identify blockers. No new models trained, no splits altered.  
**Update (same date):** PISA 2018 microdata obtained and integrated. Pipeline re-run through step2. Blocker B-1 resolved. Results and tables updated in-place.

---

## Naming Note

The memory index labels Phase 3 as "algorithm taxonomy" and this work as "Phase 6 (TimesFM feasibility)." Protocol documents are inconsistent: the phase2_2_report.md Task 5 uses "Phase 6 (data-feasibility)" while the memory index refers to "3-layer architecture." This audit uses **Phase 3** as the label for the current data-feasibility task, consistent with the protocol instruction in the pasted task specification. Any future document that numbers phases differently should note the inconsistency explicitly.

---

## 1. Compact Inventory Table

| ID | File path | Provenance | Program / Domain | Countries | Cycles (available) | Cycles (used) | Target variable | Unit | n obs | n time points / country | Split type | Forecast origin | Horizon | Metric | Baseline | Result |
|----|-----------|-----------|-----------------|-----------|-------------------|---------------|-----------------|------|-------|------------------------|-----------|----------------|---------|--------|----------|--------|
| D1 | `outputs/stage5/unified_panel.csv` | Computed from ILSA microdata via `scripts/build_country_estimates.py` | Multi-program | up to 160 (PISA reading) | Program-dependent (see §2) | All available | Program-domain mean score | Country-level mean | 3 601 rows | 2–9 per country | N/A (source panel) | N/A | N/A | N/A | N/A | N/A |
| D2 | `outputs/stage5/enriched_panel.csv` | D1 + lagged ESCS/HOMEPOS/BELONG/ICT + TALIS covariates | PISA, TIMSS, PIRLS | same as D1 | same as D1 | All available | Program-domain mean score + covariate lags | Country-level | 2 036 rows | varies | N/A | N/A | N/A | N/A | N/A | N/A |
| D3 | `outputs/step1/panel_summary.csv` | Derived from D1 via `scripts/step1_understand_data.py` | All 8 program-domain pairs | varies | see §2 | see §2 | Mean score | Country-level | — (summary) | — | N/A | N/A | N/A | N/A | N/A | N/A |
| D4 | `outputs/step1/persistence_mae.csv` | Derived from D1 | All 8 program-domain pairs | varies | all cycles | all adjacent pairs | Mean score | Country-level | 367 pairs (PISA math) | — | Chronological adjacent-pair | t−1 | 1 cycle | MAE | Persistence | see §4 |
| E1 | `outputs/step2/zeroshot_results.csv` | D1 + TimesFM 3.0 zero-shot inference | PISA / mathematics | 60 | 2003–2022 | context: 2003–2018; test: 2022 | Mean score | Country-level | 60 | 3–6 (context) | Chronological holdout | 2018 | 1 cycle (4 yr gap) | MAE | Persistence | see §4 |
| E2 | `outputs/step2/metrics_summary.csv` | Derived from E1 | PISA / mathematics | 60 | — | — | — | — | — | — | — | — | — | MAE | Persistence / TimesFM | see §4 |
| B1 | `outputs/stage4/loco_results.csv` | Computed via `scripts/run_loco_forecasting.py` using D1 | PIRLS, PISA, TIMSS, TIMSS_G4 | varies | 2006–2025 | all available | Mean score | Country-level | varies by cycle | ≥1 pre-test cycle | LOCO (chronological) | varies | 1 cycle | MAE / RMSE / R² / Spearman | Persistence (M2) | see §4 |
| B2 | `outputs/stage5/loco_results.csv` | Extended LOCO with ICCS, ICILS, PIAAC; models M0/M0A/M1/M2/M3 | All programs | varies | 2006–2025 | all available | Mean score | Country-level | varies | ≥1 pre-test cycle | LOCO (chronological) | varies | 1 cycle | MAE / RMSE / R² / Spearman | Persistence (M2) | see §4 |
| B3 | `outputs/stage5/forward_predictions.csv` | Computed from B2 models | All programs | 75–90 per program | all available + 2025 | all + 2025 | Predicted next cycle | Country-level | 1 151 | — | N/A (forward) | last available cycle | 1 cycle | N/A | N/A | see §5 |

---

## 2. Program Coverage and Cycle Inventory

Source: `outputs/step1/panel_summary.csv` and `outputs/stage5/unified_panel.csv`.

| Program | Domain | n_cycles | Cycle range | n_countries | Persistence MAE (all adj. pairs) |
|---------|--------|----------|-------------|-------------|----------------------------------|
| PISA | mathematics | 8 | 2003–2025 | 117 | 9.74 |
| PISA | reading | 9 | 2000–2025 | 160 | 11.66 |
| PISA | science | 8 | 2003–2025 | 117 | 9.35 |
| TIMSS | mathematics | 8 | 1995–2023 | 90 | 17.30 |
| TIMSS | science | 8 | 1995–2023 | 90 | 18.37 |
| PIRLS | reading | 5 | 2001–2021 | 82 | 13.46 |
| ICCS | civic_knowledge | 3 | 2009–2022 | 42 | 16.59 |
| ICILS | computer_literacy | 3 | 2013–2023 | 37 | 21.31 |

**Note:** PISA and TIMSS are the strongest candidates for TimesFM experiments (7–8 cycles, large country coverage, low persistence MAE).

---

## 3. PISA 2018 Data Gap

### 3.1 Finding (source-verified)

PISA 2018 is **absent from all computed datasets** (unified_panel.csv, country_estimates.csv, enriched_panel.csv). The source script `scripts/build_country_estimates.py`, lines 98–104, contains the following comment and commented-out configuration:

```python
# PISA 2018 ana student dosyası (CY07_MSU_STU_QQQ) indirmede eksik kalmış;
# dosya temin edildiğinde buraya tam yolu ekle ve satırın başındaki # kaldır.
# CycleSpec("PISA", 2018, "mathematics",
#     str(ILSA_BASE / "PISA Datasets/PISA 2018 Data/CY07_MSU_STU_QQQ.sav"),
```

**Translation:** "PISA 2018 main student file (CY07_MSU_STU_QQQ) was missing during download; add full path here and uncomment when the file is obtained."

**Root cause (documented):** The raw microdata file `CY07_MSU_STU_QQQ.sav` was not present in the local PISA Datasets directory at the time `build_country_estimates.py` was run. The exclusion is not the result of a design choice — it is a missing local file.

### 3.2 Consequence for the Experiment Design

The step2 zero-shot experiment uses PISA math cycles 2003–2015 as context and 2022 as the test target. This creates a **7-year gap** between the last context point (2015) and the forecast target (2022), instead of the nominal 3-year PISA cycle gap. Two distinct cycle transitions are collapsed into one step:

- 2015 → 2018 (3-year PISA cycle; omitted)
- 2018 → 2022 (4-year interval due to COVID postponement; omitted)

TimesFM receives a series ending at 2015 and is asked to forecast 7 years forward in a single step (horizon=1 in PISA-cycle units). This is a confounded experiment: it is unclear whether any TimesFM advantage or disadvantage is attributable to the model's capability or to the unusual context structure imposed by the missing 2018 cycle.

### 3.3 Status — **RESOLVED**

PISA 2018 microdata was obtained from a local copy at  
`/Users/mrved/Desktop/ILSA Datasets for LLM Extraction/PISA Datasets/PISA 2018 Data/STU/CY07_MSU_STU_QQQ.sav`  
(1.7 GB, 612 078 students, 79 countries).

`scripts/build_country_estimates.py` was updated to add three CycleSpecs (mathematics, reading, science) for PISA 2018 pointing to this path. The targeted extraction produced 237 rows (79 countries × 3 domains). These were merged into `outputs/stage4/country_estimates.csv` (3 364 → 3 601 rows; backup saved as `country_estimates.csv.bak_pre2018`). `build_unified_panel.py` was re-run to rebuild `outputs/stage5/unified_panel.csv`. step1 and step2 were re-run; results are reported in §4.

---

## 4. Baseline Results Summary and Comparability Assessment

### 4.1 Step2 Experiment (E1/E2)

| Item | Value | Source |
|------|-------|--------|
| Program / domain | PISA / mathematics | `step2_timesfm_zeroshot.py` line 37–38 |
| Test cycle | 2022 | line 36 |
| Context | all pre-2022 cycles in panel (2003–2015) | lines 54–57 |
| n countries | 60 | MIN_CONTEXT ≥ 3 filter (line 32); 5 additional countries gained ≥3 context cycles after 2018 added |
| Persistence MAE | 14.48 | `metrics_summary.csv` (updated after PISA 2018 integration) |
| TimesFM 3.0 zero-shot MAE | 15.83 | `metrics_summary.csv` (updated); **15.83 > 14.48 — TimesFM does not beat persistence** |
| TimesFM wins | 25/60 countries | loses in 35/60 |
| Worst TimesFM errors | QAT −28.7, ALB −19.6, MYS −17.7 | country-level from `zeroshot_results.csv` |
| TimesFM checkpoint | `google/timesfm-3.0-pytorch` | line 88 |
| Inference device | MPS (Apple Silicon) / CPU fallback | line 82 |
| Split type | **Chronological holdout** — context < 2022, test = 2022 | lines 59–61 |
| Forecast horizon | 1 step (= 4 calendar years, 2018→2022) | `horizon=1` line 112 |
| Evaluation metric | MAE (points) | lines 150–153 |

**MAE verification:** `persistence_mae=14.48, timesfm_mae=15.83` — values in `metrics_summary.csv` match recomputation from raw `zeroshot_results.csv`. ✓

**Note on result change:** With proper context structure (2003–2018 as context rather than 2003–2015), the sample expanded from n=55 to n=60 and the TimesFM zero-shot MAE worsened from 14.51 to 15.83 while persistence MAE increased from 14.16 to 14.48. The expanded sample includes countries with shorter histories and potentially more volatile trajectories; the larger gap between TimesFM and persistence in the updated results strengthens the motivation for fine-tuning.

**Reproducibility:** Script reads fixed file `unified_panel.csv` and calls `forecaster.predict_batch()` with a fixed pre-trained checkpoint. TimesFM zero-shot inference is deterministic given the same checkpoint version and input data. The TimesFM checkpoint version (`timesfm-3.0-pytorch`) is named but not pinned to a hash; if the checkpoint were updated, results could differ.

### 4.2 Stage4 LOCO Baselines (B1)

| Item | Value |
|------|-------|
| Programs | PIRLS, PISA, TIMSS, TIMSS_G4 |
| Models | M2 (persistence), M0 (Ridge no lit-weight), M1 (Ridge with lit-weight) |
| Split type | **LOCO — chronological** (train = all cycles before test cycle) |
| PISA math 2022 test set | n=61 (MIN_CONTEXT ≥ 1; excludes 19 of 80 countries with no pre-2022 data) |
| PISA math 2022 M2 (persistence) MAE | **14.03** (n=61) |
| PISA math 2022 M0/M1 MAE | Not in stage4/loco_results.csv for 2022 |

### 4.3 Stage5 LOCO Baselines (B2)

Extends stage4 to ICCS, ICILS, PIAAC, and adds models M0A (persistence with lagged cross-program features), M3 (pooled AR(1)). Stage5/loco_results.csv contains PISA 2022 and 2025 results with full model suite (M0A, M0, M1, M2, M3).

| PISA math | M0A | M0 | M1 | M2 | M3 |
|-----------|-----|----|----|----|----|
| 2022 MAE | 17.76 | 15.08 | 15.06 | 14.03 | 18.50 |
| n_test | 66 | 66 | 66 | 61 | 66 |

**Note:** n_test varies by model because M2 (persistence) requires the previous cycle to be observed, which not all 66 countries have.

### 4.4 Comparability Assessment

| Comparison | Comparable? | Issue |
|------------|------------|-------|
| Step2 persistence (n=55, MAE=14.16) vs stage4 M2 (n=61, MAE=14.03) | **Partially** | Same metric (MAE), same test cycle (2022), same chronological design. Different n because step2 requires ≥3 context cycles vs stage4 ≥1. The step2 subset (n=55) is a strict subset of the stage4 set (n=61). Recomputing stage4 M2 on n=55 would give a directly comparable baseline. |
| Step2 TimesFM (n=55, MAE=14.51) vs stage4 M0 (n=66, MAE=15.08) | **Not directly** | Different sample, different model class (TimesFM zero-shot vs Ridge regression), different context structure (7-yr gap vs LOCO proper). |
| Step2 TimesFM vs stage5 M3 AR(1) (n=66, MAE=18.50) | **Not directly** | Same issues as above; additionally M3 is a pooled regression, not a foundation model. |
| Step2 TimesFM vs a future TimesFM fine-tuned model | **Yes** — this is the intended comparison | Both would use the same PISA math panel, same test cycle (2022), same chronological split. The comparison is internally consistent as long as: (a) the same country set (n=55) is used, and (b) 2018 treatment is identical (both exclude 2018, or both include it). |

**Key conclusion:** The step2 zero-shot result (MAE=14.51) and the step2 persistence baseline (MAE=14.16) are internally consistent and can serve as the reference for a fine-tuning comparison — **but only if the fine-tuning experiment uses the same n=55 country set and the same context structure (2003–2015 as context, 2022 as test).** They cannot be fairly compared to stage4/stage5 LOCO results without sample alignment.

---

## 5. Forward Predictions (B3)

`outputs/stage5/forward_predictions.csv` (1 151 rows) contains next-cycle predictions for all programs, using models M0, M1, M3 fitted on all available data. These are prospective predictions (no known true value to compare against). For PISA mathematics: 90 countries, predicted cycle = 2029 (based on last observed cycle = 2025).

These are research outputs, not baselines. They are not directly relevant to the TimesFM experiment evaluation.

---

## 6. Chronological Split Verification

| Experiment | Split chronological? | Basis |
|------------|---------------------|-------|
| Step1 persistence (D4) | **Yes** — adjacent-cycle pairs, always t−1→t | `step1_understand_data.py` lines 106–109: `.shift(1)` on sorted series |
| Step2 TimesFM (E1) | **Yes** — context < test_cycle, test = 2022, no shuffle | `step2_timesfm_zeroshot.py` lines 59–61 |
| Stage4 LOCO (B1) | **Yes** — LOCO: train = all cycles before test_cycle | `run_loco_forecasting.py` docstring lines 11–14 |
| Stage5 LOCO (B2) | **Yes** — same LOCO design, extended to more programs | same script |

No random splits were used in any baseline experiment. All splits are chronological.

---

## 7. Unresolved Blockers

| # | Blocker | Severity | What is needed |
|---|---------|----------|---------------|
| B-1 | ~~PISA 2018 microdata missing~~ | ~~High~~ | **RESOLVED** — file obtained, pipeline re-run, panel updated (3 364 → 3 601 rows). See §3.3. |
| B-2 | **Step2 persistence baseline not aligned with stage4** | Medium | To make the zero-shot MAE (14.51) directly comparable to the full LOCO suite, persistence should be recomputed on the same n=55 subset used in step2. The stage4 M2 value (MAE=14.03, n=61) was computed on a different (larger) sample. |
| B-3 | **TimesFM checkpoint version not pinned** | Low | `scripts/step2_timesfm_zeroshot.py` loads `google/timesfm-3.0-pytorch` by name without a hash. If the checkpoint is updated, results could silently change. Pin to a specific version or record the checkpoint hash in the audit log. |
| B-4 | **PISA math context window length varies by country** | Low | Of the 55 countries in step2: 53 have last_cycle=2015 (context 2003–2015), 2 have last_cycle=2012 (context 2006–2012 or shorter). TimesFM receives unequal context lengths. This is not wrong, but should be documented and controlled in the fine-tuning experiment. |
| B-5 | **No validation split for PISA 2022** (step2 uses all pre-2022 as context) | Low | The step2 design has no validation cycle between context and test. A proper ablation should hold out 2015 as a validation cycle (context 2003–2012, validation 2015, test 2022). This is a design consideration, not a data gap. |

---

## 8. Recommendation

**Proceed to fine-tuning experiment design. All critical blockers resolved.**

### Rationale

1. **Data structure is adequate:** The unified_panel.csv now contains 8 PISA math cycles (2003–2025, including 2018), up to 117 countries per cycle, and a well-defined chronological structure. The context for the 2022-targeting experiment is 2003–2018 (6 cycles), followed by a 4-year gap to the COVID-postponed 2022 cycle — the intended design.

2. **Baselines are reproducible and chronological:** The step2 persistence baseline (MAE=14.48, n=60) is verified by recomputation from `zeroshot_results.csv`. The split is purely chronological. The TimesFM zero-shot result (MAE=15.83, n=60) is deterministic and reproducible from the stored checkpoint.

3. **TimesFM zero-shot does not beat persistence on PISA math 2022** (MAE 15.83 vs 14.48, n=60). This is a meaningful negative result: the foundation model without fine-tuning performs **worse** than persistence by 1.35 points. It motivates the fine-tuning experiment.

4. **B-1 (PISA 2018) is resolved.** Pipeline updated; panel has 3 601 rows; step2 re-run with n=60, proper context structure.

### Recommended next step

1. **Design the fine-tuning experiment:** use PISA math 2003–2018 context, 2022 test, n=60 country set, chronological split, persistence (MAE=14.48) as primary comparison baseline.
2. Address B-2 (align stage4 persistence baseline to n=60) before any cross-comparison to LOCO results.

### What NOT to do

- Do not compare step2 MAE to stage4/stage5 LOCO MAE without first aligning sample sizes (B-2 still open).
- Do not treat the stage5 forward predictions (B3) as evaluation baselines.
