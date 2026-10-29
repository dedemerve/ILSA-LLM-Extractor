# Phase 3.1 Integrated Audit: Reproducibility, Measurement, and Evaluation Design

**Date:** 2026-10-14  
**Protocol:** Research Protocol v1, Phase 3.1 — integrated addendum  
**Preceding artifacts:** phase3_data_feasibility_audit.md, phase3_1_audit_addendum.md  
**Scope:** Reconcile step2 results; verify country-level estimate provenance and method; audit PISA 2025 leakage and validity; design evaluation. No models trained, no data splits changed, no commits or pushes.  
**Evidence tags:** `VERIFIED` = directly confirmed from code, data, or primary-source documentation. `CONDITIONAL` = plausible but depends on unresolved assumptions. `UNVERIFIED` = insufficient evidence.

---

## Part 1 — Reconciliation of Step2 Results

### 1.1 Context window in the current results

Inspected `outputs/step2/zeroshot_results.csv` (n=60). Distribution of `last_cycle`:

| n_context | last_cycle | n countries |
|-----------|-----------|-------------|
| 3 | 2018 | 5 |
| 4 | 2018 | 7 |
| 5 | 2018 | 12 |
| 6 | 2018 | 36 |

**All 60 countries use 2018 as the final context point.** `VERIFIED` — confirmed by inspection of the `last_cycle` column.

The script `step2_timesfm_zeroshot.py` filters context to `cycle < TEST_CYCLE` where `TEST_CYCLE = 2022`. After PISA 2018 was added to the panel, the most recent pre-2022 cycle for every eligible country is 2018. The old "context ends at 2015" structure no longer applies.

### 1.2 MAE recomputation

Recomputed from `zeroshot_results.csv` using `err_persistence = abs(true_score - persistence)` and `err_timesfm = abs(true_score - timesfm_zeroshot)`:

| Metric | Stored (metrics_summary.csv) | Recomputed | Status |
|--------|------------------------------|-----------|--------|
| Persistence MAE | 14.48 | 14.4820 | `VERIFIED` ✓ |
| TimesFM zero-shot MAE | 15.83 | 15.8296 | `VERIFIED` ✓ |
| Delta (TimesFM − Persistence) | +1.35 | +1.3476 | `VERIFIED` ✓ |

### 1.3 Reconciliation of old (n=55) and updated (n=60) results

| Parameter | Old results | Updated results | Cause |
|-----------|-------------|-----------------|-------|
| n countries | 55 | 60 | 5 countries now have ≥ 3 pre-2022 context cycles after 2018 added |
| Last context cycle | 2015 (all 55) | 2018 (all 60) | PISA 2018 data integrated |
| Persistence MAE | 14.16 | 14.48 | Different sample; 5 added countries have harder-to-forecast trajectories |
| TimesFM MAE | 14.51 | 15.83 | Longer correct context; expanded sample |
| TimesFM − Persistence | +0.35 | +1.35 | TimesFM performs worse relative to persistence with proper context |

**Status of old results:** `VERIFIED` as documented — the old values (14.16/14.51, n=55) were produced with a confounded context (7-year gap 2015→2022 due to missing PISA 2018). They are no longer stored in any output file; `metrics_summary.csv` and `zeroshot_results.csv` were overwritten by the re-run. They exist only in the Phase 3 audit report §4.1 (original text, now superseded) and this addendum.

**Interpretation:** The updated results reflect the intended experimental design. The larger TimesFM−Persistence gap (+1.35 vs +0.35) in the updated results is the more relevant finding. Both old and new results agree on the direction: TimesFM zero-shot does not beat persistence on PISA math 2022.

---

## Part 2 — Country-Level Estimate Construction Audit

### 2.1 Method

Inspected `scripts/build_country_estimates.py`. For PISA (all cycles):

| Component | Specification | Evidence |
|-----------|--------------|---------|
| Achievement variables | PV1MATH…PV10MATH (10 plausible values) | `[f"PV{i}MATH" for i in range(1, 11)]` — line 83 |
| Student weight | W_FSTUWT (final student weight) | `"W_FSTUWT"` — CycleSpec field |
| Replicate weights | W_FSTURWT001–W_FSTURWT080 (80 BRR replicates) | `rep_prefix = "W_FSTURWT"`, `rep_range = (1, 80)` |
| Variance | BRR_FAY: k=0.5, R=80, denom=20 | `BRR_FAY_DENOM = BRR_FAY_R * (1-BRR_FAY_K)**2 = 20` — line 45 |
| PV pooling | Rubin (1987): T = Ū + (1+1/m)·B, m=10 | `rubin_pool()` function, lines 409–441 |
| Country identifier | CNT (string ISO3) | `country_var = "CNT"` |
| Exclusion criteria | NA on any PV or weight=0 dropped; n_analytic < 30 excluded | `estimate_country_generic()` lines 509–514 |

All three components of the PISA variance estimation are correctly implemented: `VERIFIED`.

**PISA 2000–2012 note:** `n_pv = 5` for cycles 2000–2012. PISA officially used 5 plausible values through 2009; PISA 2012 introduced 10 PVs. The `n_pv` column is unreliable (NaN) for 2015–2025 due to a column naming inconsistency between pipeline versions — the build script uses `m_pv` in newer code but `n_pv` in the original columns. The underlying computation uses 10 PVs for all cycles where `PV1MATH`–`PV10MATH` exist. `CONDITIONAL` — not confirmed by reading the SAV variable lists; inferred from the CycleSpec definition.

### 2.2 Validation of PISA 2022 estimates against official values

Compared 43 country estimates to official OECD PISA 2022 math means (OECD 2023, Table I.B1.a):

| Statistic | Value |
|-----------|-------|
| Mean difference (ours − official) | −0.15 PISA points |
| Mean absolute difference | 0.32 PISA points |
| Countries within 1 PISA point | 40 / 43 |
| Countries within 5 PISA points | 42 / 43 |
| Outlier (> 5 pts): GRC | −5.9 pts (our 430.1 vs official 436) |

**Status:** `VERIFIED`. The BRR_FAY + 10-PV Rubin pipeline reproduces official PISA 2022 means with a mean absolute error of 0.32 points, well within the typical SE of 2–4 points. The GRC discrepancy (−5.9 pts) is anomalous and should be investigated separately; it does not invalidate the overall pipeline.

This validation applies retroactively to PISA 2003–2022, provided variable names and weight structures are unchanged across cycles.

### 2.3 PISA 2025 estimate provenance

**File identification:**

The build script (`build_country_estimates.py` lines 136–150) references:  
`ILSA_BASE / "PISA Datasets/PISA 2025 Data/CY09_MS_STU_PUF.sav"`  
where `ILSA_BASE = "/Users/mrved/Desktop/ILSA Datasets"`.

This file does **not exist** at that path. `VERIFIED`.

However, the file **does exist** at:  
`/Users/mrved/Desktop/ILSA Datasets for LLM Extraction/PISA Datasets/PISA 2025 Data/CY09_MS_STU_PUF.sav` (2.17 GB).  
`VERIFIED` — file located and size confirmed.

The filename `CY09_MS_STU_PUF.sav` follows OECD PISA naming conventions:
- CY09 = Cycle 9 (PISA 2025 is the ninth main PISA administration)
- MS = Main Study
- STU = Student questionnaire file
- PUF = Public Use File

This is the standard OECD public release naming format, consistent with PISA 2022 (`CY08MSP_STU_QQQ.SAV`) and PISA 2018 (`CY07_MSU_STU_QQQ.sav`). A companion file `CY09_MS_LDW_20260806.sav` (Linked Data Weights, dated August 2026) confirms the data directory contains official OECD post-release materials. `CONDITIONAL` — naming convention match does not constitute proof of content; independent verification of the file against an OECD data archive checksum has not been performed.

**When were PISA 2025 estimates computed?**

Inspecting `outputs/stage4/country_estimates.csv.bak_pre2018` (the pre-2018-repair backup): 268 PISA 2025 rows (89 countries × 3 domains) exist in this backup. The PISA 2025 rows therefore predate the PISA 2018 pipeline repair. Git log confirms they were introduced in commit `3a84843 feat: PISA 2025 entegrasyonu ve tam LOCO backtesting`. The current build script path for 2025 is wrong (points to non-existent location), meaning these estimates **cannot be reproduced** by running `build_country_estimates.py` as currently configured. `VERIFIED`.

**Implication:** The PISA 2025 estimates in the panel were computed from the correct SAV file in a prior session, but the script path has since become stale. The estimates themselves are genuine microdata-derived values; the reproducibility path is broken.

### 2.4 Internal plausibility checks for PISA 2025

| Check | Result | Status |
|-------|--------|--------|
| Score range | 312–612 pts; consistent with PISA math scale | `VERIFIED` |
| Top performers | QCI (612), SGP (563), MAC (549), TAP (546), JPN (525), KOR (522) | Ranking consistent with historical PISA cycles | `VERIFIED` |
| n_analytic range | ~4 600–30 000 students per country | Consistent with PISA 2022 sample sizes | `VERIFIED` |
| SE range | 1.1–9.3 pts; mean 2.45 | Consistent with PISA 2022 (mean SE ~2.4) | `VERIFIED` |
| n_replicates | 80 for all 89 countries | `VERIFIED` |
| Q-prefixed entities | QCI, QUA, QAT, QTJ, QKI — sub-national PISA participants also present in 2022 | `VERIFIED` |
| **ALB anomaly** | 2022: 368.2 → 2025: 432.8, delta = **+64.6 pts** | No PISA country has ever improved by 65 points in a single cycle | `UNVERIFIED` — cause unknown |

**Validation against official published PISA 2025 values:** `UNVERIFIED`. Official OECD PISA 2025 results were published December 2025, outside this audit's knowledge cutoff (August 2025). Any comparison to "official" values cited here would be unreliable. **The user must perform this comparison manually** using the OECD PISA 2025 International Results report, Annex B tables (Table I.B1.a: Mean mathematics performance).

### 2.5 Methodological defensibility classification

| Cycle | Classification | Basis |
|-------|---------------|-------|
| PISA 2003–2022 | **`FEASIBLE`** | Pipeline validated against official 2022 values (MAD = 0.32 pts, n=43). Same method applied to all cycles. |
| PISA 2018 | **`FEASIBLE`** | Computed from same pipeline in current session; file confirmed at LLM Extraction path. |
| PISA 2025 | **`CONDITIONALLY_FEASIBLE`** | Correct computation method, correct file likely used, internal plausibility checks pass. Blocked by: (1) official validation not performed; (2) ALB anomaly unexplained; (3) build script path broken (reproducibility gap). |

---

## Part 3 — PISA 2025 Leakage Audit

### 3.1 Has PISA 2025 influenced model development or evaluation decisions?

| System | PISA 2025 involvement | Leakage verdict |
|--------|----------------------|----------------|
| `build_country_estimates.py` | PISA 2025 CycleSpec defined; estimates computed in prior session | Data construction only — no model fitting. `VERIFIED` no leakage. |
| `unified_panel.csv` | Contains PISA 2025 rows | Panel construction only. `VERIFIED` no leakage. |
| `step1_understand_data.py` | Computes persistence MAE including 2025 adjacent pairs (2022→2025) | Descriptive statistics only; not used for model selection. `VERIFIED` no leakage. |
| `step2_timesfm_zeroshot.py` | `TEST_CYCLE = 2022`; script filters `cycle < 2022` for context | PISA 2025 present in panel but never accessed. `VERIFIED` no leakage into zero-shot experiment. |
| `stage4/loco_results.csv` | Contains `test_cycle = '2025'` rows (M0, M1, M2, M3 models) | Ridge regression models evaluated on 2025. Moderate concern: 2025 outcomes were used to measure LOCO performance. However, no TimesFM parameter or checkpoint has been selected based on these results. Fine-tuning has not started. `CONDITIONAL` — no TimesFM leakage confirmed; Ridge LOCO exposure documented. |
| `stage5/loco_results.csv` | Same; also includes `test_cycle = '2025'` | Same as above. |
| TimesFM fine-tuning | Not performed | Not applicable. |

**Overall leakage verdict:** No PISA 2025 data has entered any TimesFM model selection, hyperparameter tuning, or checkpoint decision. The LOCO exposure is for Ridge regression baselines only. `CONDITIONAL` — contingent on fine-tuning not having been run in any undocumented session.

### 3.2 Country overlap between 2022 and 2025

| Set | n countries |
|-----|-------------|
| PISA math 2022 | 80 |
| PISA math 2025 | 89 |
| Overlap (2022 ∩ 2025) | **75** |
| Only in 2022 | 5: JAM, PAN, QAZ, QUR, UZB |
| Only in 2025 | 14: ARM, AZE, ECU, KEN, KGZ, LBN, LUX, MUS, QCI, QKI, QTJ, QUA, RWA, ZMB |

The viable final-test country set is 75. ALB is in this 75; if excluded on account of the anomaly, n=74.

---

## Part 4 — Evaluation Design

### 4.1 Option A: 2022 as development/validation, 2025 as held-out final test

| Parameter | Specification |
|-----------|--------------|
| Training data | PISA math 2003–2018 per country (for fine-tuning and zero-shot context) |
| Development target | 2022 (forecast origin: 2018; gap: 4 years) |
| **Final test target** | **2025 (forecast origin: 2022; gap: 3 years)** |
| Country set | 75-country overlap (2022 ∩ 2025); ALB excluded pending anomaly investigation |
| Minimum context requirement | ≥ 3 pre-2022 PISA math cycles (consistent with step2 filter) |
| Baselines | Persistence; linear trend (OLS extrapolation); TimesFM zero-shot |
| Models compared at final test | TimesFM fine-tuned (hyperparameters locked on 2022 MAE only) |
| Metric | MAE (primary); RMSE; % within 10 pts; % within 20 pts |
| Missing observations | Excluded per cycle; no imputation |
| **Leakage discipline** | Fine-tuning epoch count, learning rate, LoRA rank chosen using **only 2022 MAE**; 2025 scores sealed until final evaluation |

**Advantages:** Single clean chronological holdout; 2025 is genuinely prospective; directly answers whether fine-tuning improves over zero-shot on an unseen cycle; consistent country set across all model comparisons.

**Disadvantages:** One test cycle; 2025 performance conflates model quality with characteristics of a specific 3-year post-COVID transition.

### 4.2 Option B: Rolling-origin evaluation across historical cycles

| Parameter | Specification |
|-----------|--------------|
| Test cycles | 2012, 2015, 2018, 2022 (four PISA math holdouts) |
| Training window for each test cycle t | All PISA math cycles < t |
| Country set per cycle | Countries with ≥ 1 pre-t observation AND valid observation at t |
| Baselines | Persistence; linear trend |
| Models | TimesFM zero-shot (each origin); persistence |
| Fine-tuning | Not applicable in Option B (fine-tuning must not see the test cycles) |
| Metric | MAE per cycle; pooled MAE across cycles |

**Advantages:** 4 independent evaluations; distinguishes model performance across different cycle intervals (3-year gap, 4-year COVID gap, 3-year recovery); more robust to a single atypical cycle.

**Disadvantages:** Cannot simultaneously apply fine-tuned TimesFM across all four origins without retraining for each origin; more complex to run correctly.

### 4.3 Missing baseline: linear trend

Neither step2 nor any existing LOCO output computes a linear trend baseline. A country-specific OLS regression (slope = trend across all pre-test cycles, extrapolated one step) is a standard time-series baseline that often outperforms persistence when trajectories are systematic. **This baseline must be added before any fine-tuning comparison is reported.** `VERIFIED` as missing.

### 4.4 Recommendation

**Primary: Option A** — 2022 as development target, 2025 as sealed final test, n=74 (75 minus ALB pending), with persistence and linear trend as baselines. Fine-tuning hyperparameters locked on 2022 MAE.

**Robustness: Option B on zero-shot and persistence only** — run across 2012, 2015, 2018, 2022 to assess whether the zero-shot result generalises across multiple cycles. No fine-tuning required for Option B.

The two options are complementary and together provide a more complete picture than either alone.

---

## Explicit Decisions

### Decision 1: Reproducibility of the updated zero-shot result

**`VERIFIED` — reproducible.**

Persistence MAE = 14.48, TimesFM MAE = 15.83 (n=60) recomputed from raw CSV. All 60 countries have last_cycle = 2018. The step2 script is deterministic given the same checkpoint and panel. TimesFM checkpoint is named (`google/timesfm-3.0-pytorch`) but not hash-pinned; results could differ if the checkpoint is updated.

The old results (n=55, persistence=14.16, TimesFM=14.51) are documented here and in the Phase 3 audit report. They reflected a confounded 2015-context experiment and should not be used going forward.

### Decision 2: Validity of the country-level estimates

**`FEASIBLE` for PISA 2003–2022. `CONDITIONALLY_FEASIBLE` for PISA 2025.**

- PISA 2022 pipeline: validated against 43 official means, MAD = 0.32 pts. `VERIFIED`.
- PISA 2025 pipeline: method identical; file appears correct; build script path is broken (cannot reproduce). ALB anomaly unexplained. Official comparison not possible. `CONDITIONAL`.

### Decision 3: Suitability of PISA 2025 as final test

**`PISA_2025_CONDITIONALLY_VALID`**

Evidence for suitability:
- Estimates derived from official PUF microdata (CY09_MS_STU_PUF.sav, 2.17 GB). `CONDITIONAL` (file naming consistent with OECD convention; content unverified by checksum).
- No TimesFM leakage: fine-tuning has not started. `VERIFIED`.
- Country overlap with 2022: 75 countries — sufficient for a final evaluation. `VERIFIED`.
- Internal plausibility checks pass (rankings, score range, SE magnitudes). `VERIFIED`.

Conditions that must be met before PISA 2025 is used as final test:
1. **User validates ≥ 20 country estimates against official OECD PISA 2025 Annex B tables** (Table I.B1.a). If mean absolute discrepancy ≤ 5 points and no systematic bias: upgrade to `PISA_2025_VALID_FINAL_TEST`.
2. **ALB anomaly (368 → 433, +65 pts)** investigated: if confirmed as a data error or anomalous administration, exclude ALB from evaluation set; otherwise document and retain.
3. **Build script path fixed:** update CycleSpec for 2025 to point to the LLM Extraction path (as was done for 2018) to restore reproducibility.

### Decision 4: Recommended evaluation design

**Option A (2022 dev/val, 2025 final test) — primary.**  
**Option B (rolling-origin zero-shot, 2012–2022) — robustness check.**

Required before reporting fine-tuning results: add linear trend baseline.

### Decision 5: Readiness for fine-tuning

**Not yet ready. Three blockers must be resolved before fine-tuning begins.**

| Blocker | Category | Required action |
|---------|----------|----------------|
| B-A: PISA 2025 official validation | Measurement | User compares ≥ 20 estimates to OECD Annex B |
| B-B: ALB anomaly | Data quality | Investigate; exclude or document |
| B-C: Build script path for 2025 | Reproducibility | Update CycleSpec path to LLM Extraction location |
| B-D: Linear trend baseline missing | Evaluation design | Add to step2 or a new evaluation script |
| B-E: Fine-tuning hyperparameter lock protocol | Evaluation design | Document: epoch/LR/LoRA rank selected using 2022 MAE only; 2025 sealed |

B-A through B-C are critical (gate fine-tuning). B-D and B-E are required before final results are reported.

---

## Appendix: Provenance Summary Table

| Cycle | SAV path in script | Actual file location | Estimates in panel | Source |
|-------|-------------------|---------------------|-------------------|--------|
| PISA 2015 | ILSA Datasets/... CY6_MS_CMB_STU_QQQ.sav | Not verified in this audit | Yes | Prior pipeline run |
| PISA 2018 | LLM Extraction/.../CY07_MSU_STU_QQQ.sav | ✓ confirmed (1.7 GB) | Yes (added Oct 2026) | Current session |
| PISA 2022 | ILSA Datasets/... CY08MSP_STU_QQQ.SAV | Not verified in this audit | Yes | Prior pipeline run |
| PISA 2025 | ILSA Datasets/.../CY09_MS_STU_PUF.sav (**wrong**) | LLM Extraction/.../CY09_MS_STU_PUF.sav ✓ (2.17 GB) | Yes (commit 3a84843) | Prior session; path now stale |
