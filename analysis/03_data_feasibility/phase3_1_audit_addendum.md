# Phase 3.1 Audit Addendum: Reproducibility, Estimate Construction, and Evaluation Design

**Date:** 2026-10-14  
**Protocol:** Research Protocol v1, Phase 3.1 — addendum to phase3_data_feasibility_audit.md  
**Preceding artifact:** analysis/03_data_feasibility/phase3_data_feasibility_audit.md (updated same date)  
**Scope:** Reproducibility audit of step2 results; construction audit of country-level estimates; evaluation design options; PISA 2025 validity assessment. No models trained, no data splits altered, no files committed or pushed.

---

## Task 1 — Reconciliation of Old and Updated Step2 Results

### 1.1 Context window in the current results

The current `outputs/step2/zeroshot_results.csv` (n=60) was produced by re-running `scripts/step2_timesfm_zeroshot.py` on the updated `unified_panel.csv` (which now includes PISA 2018). Inspection of the `last_cycle` column confirms:

| n_context | last_cycle | countries |
|-----------|-----------|-----------|
| 3 | 2018 | 5 |
| 4 | 2018 | 7 |
| 5 | 2018 | 12 |
| 6 | 2018 | 36 |

**All 60 countries use 2018 as the final context point.** No country terminates context at 2015. The experiment is: context = all PISA math cycles < 2022 (up to and including 2018), test = 2022.

### 1.2 MAE recomputation

Recomputed directly from raw `zeroshot_results.csv`:

| Metric | Stored value | Recomputed | Match |
|--------|-------------|-----------|-------|
| Persistence MAE (n=60) | 14.48 | 14.4820 | ✓ |
| TimesFM zero-shot MAE (n=60) | 15.83 | 15.8296 | ✓ |
| Delta (TimesFM − Persistence) | +1.35 | +1.3476 | ✓ |

Both stored values are rounded to 2 decimal places from the recomputed figures. The results are reproducible from the raw CSV.

### 1.3 Explanation of discrepancy between old results (n=55) and updated results (n=60)

The old results were produced before PISA 2018 was added to the panel. They are no longer stored in any file — `metrics_summary.csv` and `zeroshot_results.csv` were overwritten by the re-run. They exist only in the Phase 3 audit report (§4.1, original version) and the prior session summary.

| Item | Old results | Updated results | Cause |
|------|-------------|-----------------|-------|
| n countries | 55 | 60 | 5 additional countries now have ≥ 3 pre-2022 context cycles after PISA 2018 added |
| Last context cycle | 2015 (for all) | 2018 (for all) | PISA 2018 data added to panel |
| Persistence MAE | 14.16 | 14.48 | Larger and slightly different sample |
| TimesFM MAE | 14.51 | 15.83 | Longer, now-correct context; additional 5 countries include harder-to-forecast cases |
| TimesFM − Persistence | +0.35 | +1.35 | TimesFM performs worse relative to persistence with proper context structure |

**Verified:** The old results reflected a confounded context (2015 as last point with a 7-year gap to 2022). The updated results reflect the intended design (2018 as last point with a 4-year COVID-postponed gap to 2022). The updated results are the ones that should be used going forward.

**Preservation note:** The old result figures (14.51, 14.16, n=55) are documented in this addendum and in the Phase 3 audit report §4.1 (labeled as pre-2018 results). They are not in any stored output file. Git history does not contain them because neither `zeroshot_results.csv` nor `metrics_summary.csv` was committed before the re-run.

---

## Task 2 — Country-Level Estimate Construction Audit

### 2.1 Method summary

The construction of country-level estimates in `scripts/build_country_estimates.py` follows the PISA technical manual (OECD 2020) for PISA and the IEA technical documentation for TIMSS/PIRLS.

**For PISA (all cycles including 2018 and 2025):**

| Component | Implementation | Source |
|-----------|---------------|--------|
| Achievement variables | PV1MATH–PV10MATH (10 plausible values) | PISA microdata |
| Main student weight | W_FSTUWT (final student weight) | PISA technical manual |
| Replicate weights | W_FSTURWT001–W_FSTURWT080 (80 BRR replicates) | PISA technical manual |
| Variance method | BRR_FAY (k=0.5, R=80, denom=R×(1−k)²=20) | PISA technical manual §8 |
| PV pooling | Rubin (1987): T = Ū + (1 + 1/m) × B, m=10 | IRT/PV standard |
| Country identifier | CNT (string ISO3 code) | PISA microdata |
| Missing exclusion | Students with NA on any PV or weight ≤ 0 dropped | Standard practice |
| Minimum sample | Countries with n_analytic < 30 excluded | `estimate_country_generic` line 513 |

**PV pooling formula implemented correctly:** each plausible value is independently weighted to give θ_k; the overall mean θ̄ = (1/m)Σθ_k; within-imputation variance Ū = (1/m)ΣU_k; between-imputation variance B = Var(θ_k); total variance T = Ū + (1+1/m)B.

**BRR_FAY formula implemented correctly:** variance contribution from each replicate r = (θ̂_r − θ̂_0)²; total variance = Σ_r(θ̂_r − θ̂_0)² / [R × (1−k)²] = Σ / 20.

### 2.2 PISA 2025 provenance

PISA 2025 country estimates were derived from `CY09_MS_STU_PUF.sav` using the same BRR_FAY pipeline as all other PISA cycles. They are **observed estimates from microdata**, not model-generated forward predictions. Evidence:

- `outputs/stage4/country_estimates.csv` contains 89 PISA 2025 math rows with `method = BRR_FAY`
- All 89 rows have non-null SE values (mean SE = 2.45 PISA points)
- All 89 rows have non-null n_analytic (range ~4 000–30 000 students per country)
- `outputs/stage5/forward_predictions.csv` contains PISA rows but with `predicted_cycle > 2025`, not cycle=2025 observations

**The `forward_predictions.csv` PISA rows are model-generated next-cycle predictions (predicted cycle 2029), not PISA 2025 observations.** PISA 2025 in the panel is entirely from microdata.

### 2.3 Validation against official published estimates

**LIMITATION: Official PISA 2025 published means (OECD December 2025 release) are outside the knowledge cutoff of this audit (August 2025). An independent manual comparison is required.**

Internal plausibility checks performed:

| Check | Result |
|-------|--------|
| Score range | All countries: 311–612 PISA points. Consistent with historical PISA math scale. |
| Historical rankings | Top performers: QCI (612), SGP (563), MAC (549), TAP (546), JPN (525), KOR (522) — consistent with PISA 2022 and 2018 rankings. |
| n_replicates | 80 for all PISA 2025 rows ✓ |
| m_pv | 10 for all PISA 2025 rows ✓ |
| SE magnitudes | Range 1.1–9.3 points; mean 2.45. Consistent with PISA 2022 (mean SE ~2.4). |
| n_analytic | Range ~4 600–30 000. Consistent with PISA 2022 sample sizes. |
| Q-prefixed entities | QCI (612), QUA (431), QAT (416), QTJ (382), QKI (339) — sub-national PISA entities, present in 2022 panel as well. |

**Anomaly flagged — Albania (ALB):** PISA math 2022 = 368.2, PISA math 2025 = 432.8; delta = +64.6 points. A 65-point increase in a single 3-year PISA cycle is without precedent in PISA history. This could indicate a sampling or data issue in the 2025 ALB microdata (e.g., changed target population, school exclusions, or a data error). **ALB should be treated as suspect until verified against the OECD Annex B tables.**

**Required action before treating PISA 2025 as a final test set:**
1. Compare our estimates for at least 20 countries to the official OECD Annex B tables from the PISA 2025 international report (December 2025).
2. Investigate the ALB anomaly specifically.

### 2.4 Methodological defensibility classification

Given that:
- The computation pipeline (BRR_FAY, Rubin pooling, 10 PVs, correct weight variables) is methodologically correct
- Internal plausibility checks pass for all countries except ALB
- Official validation against published estimates has not been performed

**Classification: `CONDITIONALLY_FEASIBLE`**

Condition for upgrade to `FEASIBLE`: user-performed comparison of ≥ 20 country estimates to official OECD PISA 2025 published means (Annex B, Table I.B1.a), with mean absolute discrepancy ≤ 5 PISA points and no systematic bias.

---

## Task 3 — Evaluation Design Options

### 3.1 Option A: 2022 as development/validation, 2025 as held-out final test

| Parameter | Specification |
|-----------|--------------|
| **Training data** | PISA math cycles 2003–2018 (context window for fine-tuning and zero-shot) |
| **Development/validation target** | 2022 (forecast origin: 2018; horizon: 4 years) |
| **Final test target** | 2025 (forecast origin: 2022; horizon: 3 years) |
| **Eligible country set** | 75 countries with valid observations in both 2022 and 2025 (2022∩2025 overlap) |
| **Minimum context for training** | ≥ 3 pre-2022 cycles (same as step2 filter) |
| **Baselines** | Persistence (last observed score), Linear trend (slope from all available pre-test cycles), TimesFM zero-shot |
| **Final models** | TimesFM fine-tuned, where fine-tuning settings are chosen using only 2022 performance |
| **Evaluation metric** | MAE (primary), RMSE (secondary), % within 10 pts, % within 20 pts |
| **Missing observations** | Country excluded from the evaluation cycle it is missing from; no imputation |
| **Leakage prevention** | Fine-tuning hyperparameters (epochs, learning rate, LoRA rank) locked before any 2025 evaluation. 2025 scores must not be accessed until final evaluation. |

**Advantages:** Clean chronological holdout; 2025 is genuinely prospective (administered after the context data); maximizes comparability between zero-shot and fine-tuned TimesFM on the same test set.

**Disadvantages:** Only one final test cycle; performance on 2025 conflates model quality with the particular characteristics of the 2022→2025 post-COVID transition.

### 3.2 Option B: Rolling-origin evaluation across multiple historical cycles

| Parameter | Specification |
|-----------|--------------|
| **Test cycles (PISA math)** | 2012, 2015, 2018, 2022 (four chronological holdouts) |
| **Training window for each test cycle t** | All PISA math cycles < t with ≥ 2 prior observations per country |
| **Country set per test cycle** | All countries with ≥ 1 pre-t cycle AND valid t observation |
| **Baselines** | Persistence, linear trend |
| **Models** | TimesFM zero-shot (no data needed for each origin); fine-tuned TimesFM fitted on pre-t cycles |
| **Evaluation metric** | MAE averaged across countries; separately reported per test cycle |
| **Missing observations** | Country excluded from the specific test cycle it is missing from |
| **Note on 2022** | May be used as both a training origin (for predicting 2025) and a test cycle |

**Advantages:** Provides 4 independent evaluations; separates model performance across different PISA intervals (3-year pre-COVID, 4-year COVID gap, 3-year post-COVID). More robust to any single atypical cycle.

**Disadvantages:** Fine-tuning for Option B requires retraining the model for each origin (or using the same pre-2003 weights for all), which is more complex than Option A. Cannot simultaneously hold out 2025 for all four origins.

### 3.3 Recommendation

**Primary design: Option A.** This directly addresses the research question (does fine-tuned TimesFM beat zero-shot and persistence on an unseen PISA cycle?) and preserves 2025 as a genuinely held-out test set. The key discipline is: **lock all fine-tuning settings using only 2022 MAE; do not access 2025 scores until final evaluation.**

**Option B as robustness check.** After completing the primary Option A experiment, run Option B on historical cycles (2012, 2015, 2018) using zero-shot and persistence only (no fine-tuning) to assess whether the zero-shot result generalises across cycles. Fine-tuned TimesFM from Option A can be evaluated on 2022 within Option B's framework as well.

### 3.4 Baselines to include

| Baseline | Description | Parameters needed |
|----------|-------------|------------------|
| **Persistence** | Predicted score = last observed score (= score at forecast origin) | None |
| **Linear trend** | OLS slope on all pre-test cycles; extrapolate one step | Country-specific OLS (estimated in-sample) |
| **TimesFM zero-shot** | Checkpoint `google/timesfm-3.0-pytorch`; context = all pre-test cycles; horizon=1 | None (pre-trained) |
| **TimesFM fine-tuned** | Same checkpoint, LoRA fine-tuned on PISA math series; settings locked on 2022 | LoRA hyperparameters |

**The linear trend baseline is missing from the current experiments** (step2 only reports persistence and TimesFM). It should be added before any fine-tuning comparison is reported.

---

## Task 4 — PISA 2025 Leakage Audit

### 4.1 Has PISA 2025 been used in any model training or selection decision?

| System | PISA 2025 involvement | Leakage risk |
|--------|----------------------|-------------|
| `build_country_estimates.py` | CycleSpec for 2025 extracts observed microdata | None — this is data construction, not model fitting |
| `build_unified_panel.py` | Incorporates 2025 from country_estimates.csv | None |
| `step1_understand_data.py` | Computes persistence MAE including 2025 adjacent pairs | Low — descriptive only; not used to select a model |
| `step2_timesfm_zeroshot.py` | `TEST_CYCLE = 2022`; 2025 data is in the panel but not accessed (context filter: cycle < 2022) | **None confirmed** — script filters context to < 2022 and tests on 2022 only |
| `stage4/loco_results.csv` | Contains `test_cycle = '2025'` rows for M0, M1, M2, M3 models | **Moderate concern** — LOCO models were fitted and evaluated on 2025. However: (a) fine-tuning has not started; (b) no hyperparameter selection for TimesFM has been based on these LOCO results; (c) the LOCO models (Ridge regression variants) are not TimesFM. No TimesFM parameters or settings have been chosen using 2025 outcomes. |
| `stage5/loco_results.csv` | Same as stage4, also includes 2025 test rows | Same as above |
| TimesFM fine-tuning | Not yet performed | Not applicable |

**Conclusion:** PISA 2025 data has been incorporated into the panel and used as a LOCO test cycle for Ridge regression baselines. It has **not** been used to select any TimesFM parameters, checkpoint, or fine-tuning configuration. There is no TimesFM-specific leakage.

**Required discipline going forward:** When designing the fine-tuning experiment, all choices (LoRA rank, learning rate, epochs, evaluation frequency) must be made using only 2022 outcomes. PISA 2025 scores must not be examined until final evaluation is run.

### 4.2 Country overlap

| Set | n countries |
|----|-------------|
| PISA math 2022 | 80 |
| PISA math 2025 | 89 |
| Overlap (2022 ∩ 2025) | **75** |
| Only in 2022 | 5 (JAM, PAN, QAZ, QUR, UZB) |
| Only in 2025 | 14 (ARM, AZE, ECU, KEN, KGZ, LBN, LUX, MUS, QCI, QKI, QTJ, QUA, RWA, ZMB) |

The evaluation country set for Option A (final test on 2025) must be restricted to the 75-country overlap. Final test n = 75 (subject to minimum context filter).

---

## Conclusions and Explicit Decisions

### Decision 1: Is the updated zero-shot result reproducible?

**Yes — confirmed.**

Recomputed from raw `zeroshot_results.csv`: persistence MAE = 14.482, TimesFM MAE = 15.830. Matches stored `metrics_summary.csv` (14.48 / 15.83 rounded). All 60 countries have last_cycle = 2018. The context window is correct.

The old results (n=55, persistence=14.16, TimesFM=14.51) are documented here but no longer in any stored file. They reflected a confounded context (2015 as last point). They should not be used going forward.

### Decision 2: Are the country-level estimates methodologically defensible?

**Partially — `CONDITIONALLY_FEASIBLE` for PISA 2025.**

- PISA 2003–2022: the pipeline is methodologically correct (BRR_FAY, Rubin pooling, 10 PVs). All these cycles are within the knowledge cutoff; PISA 2022 official values are available for comparison if desired.
- PISA 2025: the pipeline is applied identically, but the official published values (December 2025) cannot be independently verified here. The ALB anomaly (+65 points) must be resolved.
- TIMSS/PIRLS: JRR_ZONES variance method is implemented; not audited in this phase.

**Required user action:** Compare PISA 2025 estimates for ≥ 20 countries to official OECD Annex B tables. Investigate ALB. If mean discrepancy ≤ 5 PISA points and no systematic bias, upgrade to `FEASIBLE`.

### Decision 3: Which evaluation design is recommended?

**Option A (2022 as dev/val, 2025 as final test) — primary.**  
**Option B (rolling-origin over 2012, 2015, 2018, 2022) — robustness check, zero-shot and persistence only.**

The linear trend baseline must be added alongside persistence before reporting any comparison.

### Decision 4: Is the project ready for fine-tuning?

**Not yet — two blockers remain before fine-tuning begins.**

| Blocker | Status | Required action |
|---------|--------|-----------------|
| B-1: PISA 2018 missing | **Resolved** (Phase 3 pipeline repair) | — |
| B-2: Official validation of PISA 2025 | **Open** | User compares ≥ 20 country estimates to OECD Annex B tables |
| B-3: ALB anomaly in PISA 2025 | **Open** | Investigate or exclude ALB from 2025 evaluation set |
| B-4: Linear trend baseline missing | **Open** | Add to step2 script before reporting fine-tuning results |
| B-5: Fine-tuning hyperparameter lock protocol | **Open** | Document how epoch / LR / LoRA rank will be selected using only 2022 MAE |

**Fine-tuning may begin once B-2 and B-3 are resolved (or ALB is formally excluded).** B-4 and B-5 must be completed before final results are reported.

---

## Unresolved Issues (for future phases)

1. **PISA 2025 official validation (B-2, B-3):** Cannot be confirmed here. Requires user action.
2. **TIMSS/PIRLS 2025 data:** Not yet in the panel. If available, would extend the evaluation beyond PISA.
3. **Linear trend baseline (B-4):** Missing from all current experiments. Should be added to `step2_timesfm_zeroshot.py`.
4. **PISA 2018 SAV source path portability:** The 2018 CycleSpec uses a hardcoded Desktop path. This will fail on any other machine. Should be parameterised or documented.
5. **n_pv column in country_estimates.csv:** Always 10 for PISA; not validated for TIMSS (should be 5). Low priority.
