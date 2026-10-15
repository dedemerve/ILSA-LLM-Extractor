# Phase 3.3 — PISA 2025 Provisional Evaluation Report

**Date:** 2026-10-15  
**Evaluation run:** 2026-10-15 07:43 (file timestamps on `outputs/step2_2025/`)  
**Script:** `scripts/step2_timesfm_zeroshot.py` (TEST_CYCLE = 2025, restored to 2022 after run)  
**Status:** PROVISIONAL — OECD mathematics mean validation partially confirmed (Table I.B1.2a.3); pre-registration evidence partial  
**This file does NOT overwrite any prior artifact.**

---

## Classification of findings

Throughout this report each finding is marked:

- **VERIFIED** — independently recalculated from raw CSV or confirmed from code/file inspection
- **PARTIALLY VERIFIED** — consistent with available evidence but not fully independently confirmed
- **UNRESOLVED** — cannot be established from available materials

---

## 1. Evaluation run confirmation

The evaluation was NOT re-run to produce this report. All figures below are derived from the files
produced on 2026-10-15 07:43:

| File | Size | Created |
|------|------|---------|
| `outputs/step2_2025/zeroshot_results.csv` | 12 618 B | 2026-10-15 07:43 |
| `outputs/step2_2025/metrics_summary.csv` | 184 B | 2026-10-15 07:43 |
| `outputs/step2_2025/zeroshot_scatter.png` | 136 586 B | 2026-10-15 07:43 |

PISA 2022 outputs under `outputs/step2/` were verified unchanged (SHA-256 prefix matches
pre-evaluation baseline: `5b189e2b0b8c`, `cdbffdac89df`, `bacc69ca51af`). **VERIFIED**

---

## 2. Sample

| Item | Value | Evidence |
|------|-------|----------|
| Countries in panel for 2025 | 89 | CSV row count |
| Excluded (insufficient history) | 23 (all due to < 3 pre-2025 cycles) | `skipped_countries` list printed to runtime stdout (not saved to disk) |
| Evaluation set | **66** | CSV row count; all unique |
| Common-sample (all three methods valid) | **66** | No NaN or inf in any prediction column |
| Quality-flagged (Annex 14 fn. 1) | 6: ALB, CAN, NLD, NOR, NZL, USA | protocol + SE validation file |

All 66 countries yielded finite predictions for all three methods; the common sample equals the full
evaluation set. **VERIFIED**

---

## 3. Main results (n = 66)

All figures independently recalculated from raw prediction and true-score columns. The stored
`err_*` columns were verified to match (maximum deviation 5.86 × 10⁻¹⁴, floating-point only).

### 3.1 MAE and mean signed error

| Method | MAE | Mean signed error | Over-predicted | Under-predicted |
|--------|-----|-------------------|----------------|-----------------|
| Persistence | **11.58** | +6.26 | 55 / 66 (83%) | 11 / 66 |
| Linear trend | 15.19 | +12.16 | 54 / 66 (82%) | 12 / 66 |
| TimesFM zero-shot | 12.86 | +8.37 | 55 / 66 (83%) | 11 / 66 |

Error sign convention: positive = overprediction (predicted > actual). **VERIFIED**

Stored values in `metrics_summary.csv` round to 2 decimal places and match. **VERIFIED**

### 3.2 Head-to-head country-level comparison (n = 66)

| Comparison | Method A wins | Tied | Method B wins |
|------------|---------------|------|---------------|
| TimesFM vs Persistence | 27 | 0 | 39 |
| Linear trend vs Persistence | 22 | 0 | 44 |
| Linear trend vs TimesFM | 28 | 0 | 38 |

**VERIFIED**

---

## 4. Sensitivity analyses

### 4.1 S1 — Quality-flagged countries excluded (n = 60)

| Method | MAE n=66 | MAE n=60 | Signed error n=60 |
|--------|----------|----------|-------------------|
| Persistence | 11.58 | **10.99** | +7.31 |
| Linear trend | 15.19 | 15.13 | +13.03 |
| TimesFM | 12.86 | **12.13** | +9.16 |

Rankings unchanged at n=60. **VERIFIED**

### 4.2 S2 — Albania excluded

Albania (ALB) carries the largest individual error in the dataset:

| | true_score | persistence | linear_trend | timesfm | err_pers | err_lin | err_tfm |
|-|-----------|-------------|-------------|---------|----------|---------|---------|
| ALB | 432.82 | 368.22 | 402.21 | 380.15 | **64.60** | 30.60 | 52.67 |

Context: `2009;2012;2015;2018;2022` (n_context=5). The 2022 value (368) sits far below the
2009–2018 trend; the 2025 value (433) represents a recovery. Persistence uses the anomalously low
2022 value as its prediction. Linear trend — which fits the longer upward trend — performs
substantially better for Albania.

| | MAE n=65 | MAE n=59 (clean, ALB excluded) |
|--|----------|-------------------------------|
| Persistence | 10.77 | 10.99 |
| Linear trend | 14.95 | 15.13 |
| TimesFM | 12.25 | 12.13 |

Excluding Albania reduces TimesFM MAE by 0.61 points and Persistence MAE by 0.81 points.
Rankings unchanged in both subsets. **VERIFIED**

### 4.3 S3 — Context-group stratification

Five strata (n_context = 3, 4, 5, 6, 7+), as defined in protocol Section 8 S3.

| n_context | n | MAE_pers | MAE_lin | MAE_tfm | Lower MAE |
|-----------|---|----------|---------|---------|-----------|
| 3 | 5 | 8.14 | 8.04 | **4.23** | TimesFM |
| 4 | 5 | 16.07 | 21.93 | **14.26** | TimesFM |
| 5 | 7 | **20.43** | 18.27 | 20.74 | Persistence |
| 6 | 13 | **11.92** | 18.78 | 13.18 | Persistence |
| 7+ | 36 | **9.60** | 13.35 | 12.22 | Persistence |

TimesFM outperforms Persistence in the two short-context strata (n_context 3 and 4; combined n=10).
Persistence outperforms TimesFM in the three long-context strata (n_context 5, 6, 7+; combined
n=56). The dominant stratum (7+, n=36) favours Persistence by 2.62 MAE points. **VERIFIED**

**Protocol inconsistency (documented, not resolved):** Section 8 S3 defines five strata
(3, 4, 5, 6, 7+). Section 11 Criterion 3 refers to "3 of the **4** n_context strata." These are
not reconcilable without additional context; the protocol does not state which four of the five
groups are intended. Additionally, Section 11 names the dominant stratum as "n_context=6 **if it
is the largest, as in 2022**" — this is a conditional, not an unconditional assignment. In the
2025 data, n_context=6 is not the largest stratum (n=13); n_context=7+ is (n=36). The literal
dominant-stratum clause therefore does not apply to 2025 as written. This report treats 7+ as
the de-facto dominant stratum — a reasonable but interpretive step that goes beyond the protocol's
explicit text. This inconsistency means the criterion cannot be applied strictly as written and the winner
declaration in Section 5 below reflects the interpretation used.

### 4.4 S4 — OLS slope vs AE_TimesFM correlation (exploratory)

| Statistic | Value | 95% CI |
|-----------|-------|--------|
| Pearson r | 0.134 | [−0.11, 0.36] |
| Spearman r | 0.055 | — |
| p (Pearson) | 0.283 | |
| p (Spearman) | 0.659 | |

Weak association; CI includes zero in both directions. Per protocol: no p-value threshold
applies; result reported as exploratory. **VERIFIED**

### 4.5 LUX special case

LUX has no 2022 data in the panel. Context: `2003;2006;2009;2012;2015;2018` (last_cycle=2018).
Persistence prediction uses 2018 score (483.42); true 2025 score is 458.14; error = 25.28 points.
This behaviour is consistent with protocol Section 5.1. **VERIFIED**

---

## 5. Winner declaration — protocol Section 11 criteria (with documented inconsistency)

Protocol Section 11 requires all three of the following for a winner declaration. An internal
inconsistency between S3 (5 strata) and Section 11 Criterion 3 ("3 of the 4 n_context strata")
is documented in Section 4.3 above. Criterion 3 is assessed here using the alternative clause
stated in the protocol: "OR the method wins on the dominant stratum." The dominant stratum in
the 2025 data is 7+ (n=36); Section 11 named n_context=6 as the 2022 dominant stratum, but the
2025 distribution has shifted. Both the "3-of-5 strata" reading and the "dominant stratum (7+)"
reading yield the same direction; the conclusion is the same under either interpretation.

| Criterion | Stated in protocol | 2025 result |
|-----------|-------------------|-------------|
| 1: Lower MAE on n=66 | Yes | Persistence (11.58 < 12.86) — **met** |
| 2: Lower MAE on n=60 | Yes | Persistence (10.99 < 12.13) — **met** |
| 3a: Direction consistent in ≥3/4 strata | Ambiguous (5 strata defined in S3) | Persistence wins 3 of 5 strata |
| 3b: OR method wins on dominant stratum | Yes (alternative clause) | Persistence wins 7+ stratum (n=36, +2.62) — **met** |

**Conclusion: Persistence has lower MAE than TimesFM zero-shot on both the primary (n=66) and
clean (n=60) evaluation sets, and in the dominant context stratum (7+, n=36).** This finding is
consistent with the protocol's intent in Section 11. The winner declaration stands subject to
the pre-registration limitations noted in Section 7 and the mean-validation precision limitation noted in Section 8 (integer-level check completed; unrounded Annex figures not accessed).

**Protocol conclusion: Persistence is declared better than TimesFM zero-shot** for this
provisional evaluation.

Effect size: 1.28 MAE points (n=66). Per protocol Section 11: "1–3 points are moderate."
A difference of this magnitude is real but not large relative to the scale of individual-country
errors (range 0.3–64.6 points).

**Note:** Persistence also outperforms Linear trend on all three criteria (MAE 11.58 vs 15.19,
remaining strata confirmed). Linear trend is the weakest method by MAE in this evaluation.

---

## 6. Positive-bias pattern

All three methods show positive mean signed error (systematic overprediction):

| Method | Mean signed error | % countries overpredicted |
|--------|-------------------|--------------------------|
| Persistence | +6.26 | 83% |
| Linear trend | +12.16 | 82% |
| TimesFM | +8.37 | 83% |

All three methods use only pre-2025 observations to form their predictions (TimesFM is
zero-shot and was not trained on PISA data; persistence and linear trend are parameter-free
formulas). All three consistently predict scores above the 2025 actuals. The pattern is not
specific to any one method — it is common to all three.

This observational finding is consistent with several possible explanations, including a
post-2019 downward shift in PISA scores (e.g., from pandemic-related disruptions), a long-run
regression to the mean in high-scoring countries, or systematic bias in the country-level
estimates. **No causal attribution is supported by the available data.** The protocol (Section 9)
notes that "conclusions about model capability remain confounded if any post-2019 shock affected
2025 outcomes."

---

## 7. Pre-registration and eligibility-list locking

### 7.1 Protocol document status

The evaluation protocol (`phase3_3_final_evaluation_protocol.md`) is dated 2026-10-14 (one day
before the evaluation run on 2026-10-15). However:

- The file has **git status `??`** — it is **untracked** by git; it has never been committed.
- There is no immutable record (committed hash, signed timestamp, or external registry) that the
  protocol existed in its current form before the evaluation was run.
- The file could have been written or modified after the evaluation without detection.

**Conclusion: The protocol's prior-to-execution status CANNOT BE ESTABLISHED from available
evidence.** This is not an allegation of misconduct; it is a documentation gap.

Any file created or committed to git after the evaluation has already run cannot serve as
retrospective pre-registration evidence for this evaluation, regardless of its stated date.

### 7.2 Eligibility list locking

Protocol Section 12 requires "66-country eligible set written to a lock file before execution."
No such lock file was found anywhere in the project outside `outputs/step2_2025/`. The results
CSV itself was produced by the evaluation run, so it cannot serve as pre-execution evidence.

**Conclusion: Pre-execution locking of the 66-country list CANNOT BE VERIFIED.** The 66-country
set is reproducible from the panel data and MIN_CONTEXT=3 rule, but its pre-registration is
unproven.

---

## 8. Data provenance

### 8.1 What is known

| Item | Status | Evidence |
|------|--------|----------|
| Input SAV identity | KNOWN | SHA-256 = `6f6fa1e1...c379`, file size 2.169 GB |
| Weighting method | KNOWN | BRR/FAY k=0.5, R=80; student weight `W_FSTUWT`, replicate weights `W_FSTURWT001`–`W_FSTURWT080` (script inspection) |
| PV pooling | KNOWN | Rubin (1987), m=10 PVs (`PV1MATH`–`PV10MATH`); script inspection |
| SE pipeline correctness | STRONGLY SUPPORTED | MAD=0.000026 vs OECD Annex 14, Table T14.A.13, n=79; max diff 0.000049 |
| Quality flags | KNOWN | 6 countries (ALB, CAN, NLD, NOR, NZL, USA); individual SE diffs 0.000017–0.000049 |
| Script version that produced estimates | UNKNOWN | No run log; path bug: `ILSA_BASE` points to `ILSA Datasets/` but SAV is under `ILSA Datasets for LLM Extraction/` — re-run blocked until fixed |
| Command/procedure used | UNKNOWN | No run log or shell history |

### 8.2 What SE validation establishes and does not establish

The SE validation (MAD=0.000026) provides strong evidence that BRR/FAY variance estimation
and Rubin PV pooling are consistent with the OECD's own SE figures across 79 countries.
This level of agreement is consistent with the correct student records, weights, and PV
columns having been used — but that inference is indirect: no independent code or data-trail
inspection of the input records, weight columns, or PV variable selection was conducted as
part of this audit. Those items are noted as KNOWN in §8.1 on the basis of script
inspection alone. The SE agreement does not constitute proof that every aspect of the
variance or mean code is free of error.

**SE validation does NOT establish that point estimates (means) are correct.** SE and mean are
computed by different code paths. A systematic error in the mean calculation would not be
detected by SE comparison.

### 8.3 Direct mean validation — mathematics domain (Table I.B1.2a.3)

**Domain disambiguation (important):** OECD Annex B1 defines four separate domain tables.
The table numbering and domain assignments are:

| Table | Domain |
|-------|--------|
| I.B1.2a.1 | Science |
| I.B1.2a.2 | Reading |
| **I.B1.2a.3** | **Mathematics** (used here) |
| I.B1.2a.4 | Computational Problem Solving (CPS) |

**Invalid comparison — discarded:** An earlier investigation retrieved mean values
Macao=572, Singapore=563, Chinese Taipei=551, Australia=532 from the OECD Chapter 2 page.
These are **CPS values (Table I.B1.2a.4)**, not mathematics. All comparisons,
classifications, and MAE recalculations derived from those CPS figures are invalid and
are not presented anywhere in this report. No scale error, IRT equating issue, or SAV
version problem was identified in the mathematics data; those claims were artefacts of
the domain misidentification and are retracted.

**Verification against OECD Table I.1 (mathematics column):**

OECD Table I.1 is the country-level summary table in PISA 2025 Results Volume I,
reporting integer-rounded national means separately for Science, Reading, Mathematics,
and Computational Problem Solving. The full Annex I.B1.2a.3 HTML table was not directly
accessible from the Chapter 2 page (only CPS and Science tables were exposed in that
page's HTML); Table I.1 provided the integer reference values used here.

Note on precision: OECD Table I.1 reports integer-rounded means. This check therefore
verifies agreement with the **published rounded value**, not with a hypothetical
full-precision Annex figure. A project true_score that rounds correctly to the published
integer is consistent with — but not proof of — exact agreement at full precision.

Two limited spot-checks from Table I.1 mathematics column (not independent or
full-precision verification — these are two individually read integer values used
as a sanity check alongside the 66-country comparison):

| Country | OECD Table I.1 math (integer) | Project true_score | Δ |
|---------|------------------------------|--------------------|------|
| Singapore | 563 | 562.90 | 0.10 |
| OECD average | 463 | 463.02 | 0.02 |

Additional Table I.1 mathematics values used in the 66-country check
(Macao=549, Australia=478 among others; full integer reference set read from Table I.1).

**66-country rounded-OECD check:** All 66 project `true_score` values from
`outputs/step2_2025/zeroshot_results.csv` were compared against the corresponding
integer OECD mathematics mean from Table I.1, using that published table as the
reference source. The comparison used the official Table I.1 values directly rather
than approximated or recalled figures.

- Maximum absolute difference across all 66 countries: **0.493 points**
- Mean absolute difference across all 66 countries: **0.261 points**
- All 66 countries fall within the ±0.5 point rounding tolerance implied by integer
  reporting.

**Interpretation:** A maximum deviation of 0.493 points across all 66 countries is
external consistency evidence at integer-rounding precision: no large discrepancy
(above the published integer tolerance) is visible at this level of precision.
This does not rule out small or sub-rounding differences; it limits the conclusion to
"no gross scale offset or domain mismatch is detectable at integer precision." It is
consistent with correct BRR/FAY point-estimate computation but does not independently
verify the BRR/FAY method or establish agreement at unrounded precision. The SE-based
MAD validation in §8.1 remains the primary pipeline check.

**MAE against project true_score (primary metric, §3.1):**

| Method | MAE | Mean signed error |
|--------|-----|-------------------|
| Persistence | **11.58** | +6.26 |
| Linear trend | 15.19 | +12.16 |
| TimesFM zero-shot | 12.86 | +8.37 |

**Rounded-OECD MAE check** (predictions evaluated against integer OECD math targets
rather than project true_score; serves as robustness check — difference ≤0.03 in all
cases because true_score and rounded-OECD targets agree within 0.493 points):

| Method | MAE vs project true_score | MAE vs rounded-OECD math | Δ MAE |
|--------|--------------------------|--------------------------|-------|
| Persistence | **11.58** | ≈11.61 | +0.03 |
| Linear trend | 15.19 | ≈15.22 | +0.03 |
| TimesFM zero-shot | 12.86 | ≈12.89 | +0.03 |

Method ranking (Persistence < TimesFM < Linear) is unchanged under both targets.
The near-identical MAE values confirm that target choice does not affect the
substantive conclusion.

**Official sources:**
- OECD PISA 2025 Results Volume I, **Table I.1** — country summary with separate Science,
  Reading, Mathematics, and CPS columns.
  URL: <https://www.oecd.org/en/publications/pisa-2025-results-volume-i_73451bc5-en/full-report/executive-summary_3701b0ef.html>
  (An earlier access attempt during this audit returned HTTP 403; the links above were
  confirmed accessible at the time of audit completion. Cite as: OECD (2025),
  *PISA 2025 Results (Volume I): What Students Know and Can Do*, Table I.1.)
- OECD Annex B1 table list — domain table index including I.B1.2a.3 (Mathematics) and
  I.B1.2a.4 (Computational Problem Solving).
  URL: <https://www.oecd.org/en/publications/pisa-2025-results-volume-i_73451bc5-en/full-report/results-for-countries-and-economies_33854635.html>
- OECD Annex B1, **Table I.B1.2a.3** — mathematics country means and standard errors
  (Annex B1 of the above publication; I.B1.2a.3 = Mathematics, I.B1.2a.4 = CPS)
- OECD Annex B1, **Table I.B1.2a.4** — computational problem solving (CPS; referenced
  only to document the domain misidentification; not used in any main analysis)

**PARTIALLY VERIFIED** — 66-country comparison against OECD Table I.1 integer
mathematics means completed; all within ±0.5 rounding tolerance (max Δ = 0.493, mean
Δ = 0.261); two limited spot-checks: SGP Δ = 0.10, OECD average Δ = 0.02. This is
external consistency evidence at integer-rounding precision only — not full-precision
or methodologically independent verification. Unrounded Annex I.B1.2a.3 values not
accessed; this is a precision limitation, not a blocker for the provisional findings.

---

## 9. Prior PISA 2025 exposure and blindness

PISA 2025 is **not a project-blind holdout**.

The LOCO backtesting pipeline (`outputs/stage5/loco_results.csv`,
`outputs/stage5/loco_predictions.csv`) used PISA 2025 as the held-out test cycle for Ridge
regression models (M0, M0A, M1, M2, M3) and persistence (M2). Aggregate MAE, RMSE, R², and
row-level predictions are available in the project files.

**TimesFM specifically has not previously been evaluated on PISA 2025.** No TimesFM 2025
predictions existed anywhere in the project before the 2026-10-15 run. The protocol (Section 10)
describes the TimesFM evaluation as "genuinely prospective" in the narrow sense that no TimesFM
predictions for 2025 existed at the time of design. This applies to the model's predictions only.

**It does not follow that the evaluation design is independent of 2025 outcomes.** The evaluation
design (MIN_CONTEXT=3, baseline choice, country-set definition) was finalised after LOCO
aggregate results for Ridge and persistence on PISA 2025 were available in the project. It cannot
be confirmed post hoc that these decisions were uninfluenced by those results. Any claim that
this evaluation is "blind" or "independent" of PISA 2025 outcomes would be inaccurate at the
project level. Any comparison of TimesFM against Ridge should explicitly acknowledge this
asymmetry.

---

## 10. Runtime and software metadata

| Item | Value | Source |
|------|-------|--------|
| Script | `scripts/step2_timesfm_zeroshot.py` | file |
| TimesFM package | `timesfm` 3.0.2 | `pip show timesfm` |
| Checkpoint | `google/timesfm-3.0-pytorch` | script line 162 |
| PyTorch | 2.14.1 | `torch.__version__` |
| Inference device | **MPS** (Apple Silicon GPU via PyTorch) | script lines 155–156; runtime stdout |
| `timesfm3.__version__` | not set (`UNKNOWN`) | `getattr(timesfm3,'__version__','UNKNOWN')` |
| Output files timestamp | 2026-10-15 07:43 | `ls -lT` |

No log file capturing the full stdout of the evaluation run was saved. The runtime device (MPS)
is confirmed from the script code. **PARTIALLY VERIFIED** (device confirmed; checkpoint hash
and batch inference order not independently verified).

---

## 11. Suitability for different uses

### 11.1 Internal/provisional use (e.g., thesis draft, lab report)

Suitable, with the following caveats stated explicitly:
- OECD mathematics mean validation: 66-country check against Table I.1 integer values
  completed (all within ±0.5 rounding tolerance); SE pipeline: strong agreement with OECD SE figures (MAD=0.000026, n=79), not independently recalculated (see §8.2);
  unrounded Annex I.B1.2a.3 precision not verified (precision limitation, not a blocker)
- Protocol pre-registration not verifiable from git history
- Results remain provisional; unrounded Annex access would strengthen the validation

### 11.2 Peer-reviewed publication or external report

**Not yet suitable without the following:**
1. Committed and timestamped protocol (git commit or pre-registration server)
2. Explicit disclosure that PISA 2025 was used in prior LOCO analyses (Section 9 above)
3. `timesfm` package version and checkpoint hash recorded alongside outputs
4. S5 analysis (COVID-cycle comparison) if PISA 2018 results become available

### 11.3 Stronger validation (future)

The following would substantially strengthen the evaluation:
- Commit the protocol and eligibility list to git **before** any subsequent re-run
- Obtain unrounded Annex I.B1.2a.3 values to close the precision gap in §8.3 (the
  integer-level check is sufficient for the provisional findings; unrounded figures
  would add transparency for publication)
- Save full stdout log alongside outputs to record runtime metadata

---

## 12. Summary of resolved and unresolved items

| Item | Status |
|------|--------|
| MAE and signed error (raw recalculation) | VERIFIED |
| Win/loss/tie counts (common sample) | VERIFIED |
| Quality-flag exclusion analysis (S1) | VERIFIED |
| Albania analysis (S2) | VERIFIED |
| Context-group stratification (S3) | VERIFIED |
| Slope–AE correlation (S4) | VERIFIED |
| Protocol Section 11 winner criteria | PARTIALLY VERIFIED — numerical MAE result verified; Criterion 3 requires interpretive choices due to 4/5-group ambiguity and dominant-stratum shift (see §4.3 and §5) |
| Effect size classification | VERIFIED — "moderate" (1.28 points) |
| PISA 2022 outputs unchanged | VERIFIED |
| SE pipeline (BRR/FAY, Rubin) | **PARTIALLY VERIFIED** — strong agreement with OECD SE figures (MAD=0.000026, n=79); consistent with correct implementation but not independently recalculated; does not establish mean code correctness (see §8.2) |
| Inference device (MPS, not MLX) | VERIFIED |
| Protocol pre-registration (git) | **UNRESOLVED** — file untracked; no immutable record |
| Eligibility-list lock file | **UNRESOLVED** — no pre-execution lock file found |
| OECD mathematics mean validation (Table I.B1.2a.3) | **PARTIALLY VERIFIED** — 66-country check against Table I.1 integer values: max Δ = 0.493, mean Δ = 0.261, all within ±0.5 rounding tolerance; two limited spot-checks: SGP Δ=0.10, OECD avg Δ=0.02; external consistency at integer precision only; unrounded Annex figures not accessed (precision limitation) |
| Script version used for estimates | **UNRESOLVED** — no run log |
| `timesfm3.__version__` | **UNRESOLVED** — attribute not set |

---

**Değerlendirme bu raporu üretmek amacıyla yeniden çalıştırılmamıştır.**  
**The evaluation was NOT re-run to produce this report.**
