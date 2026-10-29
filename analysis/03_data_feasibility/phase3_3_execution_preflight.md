# Phase 3.3 — Execution Preflight Check

**Date:** 2026-10-14  
**Script under review:** `scripts/step2_timesfm_zeroshot.py`  
**Proposed adaptation:** change `TEST_CYCLE = 2022` → `TEST_CYCLE = 2025`  
**Data source under review:** `outputs/stage5/unified_panel.csv`  
**This file does NOT overwrite any prior artifact.**

---

## Item 1 — Exact 66-Country Eligible Set

**Method:** Simulated the step2 script filter on `unified_panel.csv`  
(`program=="PISA"` ∧ `domain=="mathematics"` ∧ `PISA_mathematics` non-null, then `n_pre_2025 >= 3`)

**Result: CONFIRMED — n=66, exact set:**

| | | | | | | |
|--|--|--|--|--|--|--|
| ALB | ARE | ARG | AUS | AUT | BEL | BGR |
| BRA | CAN | CHE | CHL | COL | CRI | CZE |
| DEU | DNK | DOM | ESP | EST | FIN | FRA |
| GBR | GEO | GRC | HKG | HRV | HUN | IDN |
| IRL | ISL | ISR | ITA | JOR | JPN | KAZ |
| KOR | KSV | LTU | LUX | LVA | MAC | MDA |
| MEX | MKD | MLT | MNE | MYS | NLD | NOR |
| NZL | PER | POL | PRT | QAT | ROU | SGP |
| SRB | SVK | SVN | SWE | TAP | THA | TUR |
| URY | USA | VNM | | | | |

This set is identical to the set in `audit/pisa_2022_2025_country_eligibility.csv` (in_primary_2025=True).

**Notable:** LUX has no 2022 PISA entry in the panel. Cycles present for LUX: 2003, 2006, 2009, 2012, 2015, 2018, 2025 (n_pre_2025=6). The persistence baseline for LUX will use 2018, not 2022, as the last observed cycle. This is consistent with the protocol's missing-data rule and must be disclosed in the evaluation report.

---

## Item 2 — Six Quality-Flagged Countries

| Country | ISO3 | PISA 2025 mean (from panel) | In eligible-66? | Protocol treatment |
|---------|------|---------------------------|-----------------|-------------------|
| Albania | ALB | 432.82 | Yes | Included in primary n=66; reported in separate stratum; highlighted individually |
| Canada | CAN | 485.34 | Yes | Included in primary n=66; reported in separate stratum |
| Netherlands | NLD | 483.27 | Yes | Included in primary n=66; reported in separate stratum |
| Norway | NOR | 451.78 | Yes | Included in primary n=66; reported in separate stratum |
| New Zealand | NZL | 479.62 | Yes | Included in primary n=66; reported in separate stratum |
| United States | USA | 462.85 | Yes | Included in primary n=66; reported in separate stratum |

All six are confirmed present in `unified_panel.csv` with PISA 2025 mathematics values. Clean stratum (n=60) = eligible-66 minus these six. **CONFIRMED.**

---

## Item 3 — Common Country Set, Target Values, Error Definitions

### 3a. Persistence

Source: `unified_panel.csv`, column `PISA_mathematics`. For each eligible country, prediction = last non-missing pre-2025 value of `PISA_mathematics`. True score = `PISA_mathematics` at cycle 2025.  
The script reads from the same panel column for both context and target. No separate file is used.  
**CONSISTENT.**

### 3b. Linear Trend Baseline

**ISSUE IDENTIFIED — see Item 7 below.**

The linear trend baseline was computed for the 2022 experiment and is stored in `analysis/03_data_feasibility/audit/linear_trend_baseline_2022.csv`. That computation was done as an analysis artifact (not by a standalone production script). There is no `step2_linear_trend.py` or equivalent in `scripts/`. The protocol (Section 5.2) requires OLS regression of pre-2025 scores on cycle year with prediction at 2025.

**For the evaluation to include the linear trend baseline, a new script or inline computation must be written.** This is not a data problem; the underlying data exists in `unified_panel.csv`. It is a scripting gap.

### 3c. TimesFM

Source: same `unified_panel.csv`, same `PISA_mathematics` column. The script uses `context_scores = context_rows["PISA_mathematics"].tolist()` and `true_score = target_row["PISA_mathematics"].iloc[0]`. All three methods (when linear trend is implemented) will use the same source column and the same 66-country set.  
**CONSISTENT (pending linear trend script).**

### 3d. Error Definitions

The script (L123–125) stores:
```
"err_persistence": abs(true_score - persistence)
"err_timesfm":     abs(true_score - tfm_pred)
```

These are **absolute** errors, not signed. The protocol (Section 6) requires signed errors = predicted − actual (positive = overprediction). The signed errors can be computed post-hoc from the raw columns (`persistence − true_score`, `timesfm_zeroshot − true_score`) that the script writes to the output CSV. The stored `err_*` columns must NOT be used for signed-error analysis.

**Protocol compliance requires post-hoc signed error computation from raw columns. The script's stored errors are absolute and should be labelled AE in any report.**

---

## Item 4 — Minimum Historical Context Rule and Missing-Data Policy

| Rule | Script | Protocol | Status |
|------|--------|----------|--------|
| MIN_CONTEXT | L40: `MIN_CONTEXT = 3` | ≥3 pre-2025 cycles | **CONSISTENT** |
| Persistence missing-data | Uses `context_rows[-1]` = last non-missing cycle | Last non-missing pre-T cycle | **CONSISTENT** |
| Linear trend missing-data | N/A (script not yet written) | All available pre-2025 obs | Pending |
| Countries excluded for < MIN_CONTEXT | 23 of 89 (BRN, GTM, KHM, MAR, MNG, PHL, PRY, PSE, SLV, SAU, and 13 others with n_pre2025 < 3) | Excluded | **CONSISTENT** |

**LUX special case:** no 2022 cycle in panel; persistence will use 2018. Protocol allows this. Eligibility unaffected (n_pre2025=6).

---

## Item 5 — PISA 2025 Results Will Not Be Used to Tune or Select TimesFM Settings

**CONFIRMED.**

Evidence:
- `step2_timesfm_zeroshot.py` contains no `train()`, `fit()`, `optimizer`, `backward()`, `gradient`, or `epoch` calls (grep confirmed: zero matches).
- The script calls `forecaster.predict_batch(contexts, horizon=1, return_quantiles=False)` — pure inference.
- `step3_finetune_timesfm.py` is a completely separate file. It is not imported or invoked by step2.
- `ModelConfig` in step2 takes only `checkpoint_path`, `per_core_batch_size`, and `device`. No trainable parameters are set.
- The checkpoint used (`google/timesfm-3.0-pytorch`) is a fixed, pre-trained model. Running it with `TEST_CYCLE=2025` will not alter the model weights.
- The protocol was finalized (locked) before any TimesFM 2025 predictions exist. No PISA 2025 outcome data will influence model selection.

---

## Item 6 — Existing Outputs Will Not Be Overwritten

**BLOCKER — see Item 7.**

The script hardcodes `OUT_DIR = ROOT / "outputs" / "step2"`. If run with `TEST_CYCLE=2025`, it will write to:
- `outputs/step2/zeroshot_results.csv` ← **OVERWRITES** existing 2022 results
- `outputs/step2/metrics_summary.csv` ← **OVERWRITES** existing 2022 results
- `outputs/step2/zeroshot_scatter.png` ← **OVERWRITES** existing 2022 scatter plot

Current content of `outputs/step2/`:
```
zeroshot_results.csv  — 60 rows, PISA 2022 development results (Persistence MAE=14.48, TFM MAE=15.83)
metrics_summary.csv   — 2 rows: Persistence / TimesFM_zero-shot
zeroshot_scatter.png  — 2022 scatter visualization
```

**This is a hard constraint violation.** The evaluation cannot proceed without redirecting outputs to a new directory (e.g., `outputs/step2_2025/`).

---

## Item 7 — Exact Command, Configuration, Input Files, and Output Paths

### Current (2022) configuration
```
TEST_CYCLE  = 2022
PROGRAM     = "PISA"
DOMAIN      = "mathematics"
MIN_CONTEXT = 3
PANEL       = outputs/stage5/unified_panel.csv
OUT_DIR     = outputs/step2/
```

### Required changes for 2025 evaluation

| Parameter | Current value | Required value | Change type |
|-----------|--------------|----------------|-------------|
| `TEST_CYCLE` | 2022 | **2025** | One-line edit |
| `OUT_DIR` | `outputs/step2/` | **`outputs/step2_2025/`** | One-line edit (required to prevent overwrite) |
| Linear trend | Not in script | Must be added | New code block |
| Error storage | `abs()` | Protocol requires signed post-hoc | Analysis note (raw columns are sufficient) |

### Required changes (two mandatory, one advisory)

**Mandatory change 1 — Output directory (constraint: no overwrite)**
```python
# Line 33: change
OUT_DIR  = ROOT / "outputs" / "step2"
# to
OUT_DIR  = ROOT / "outputs" / "step2_2025"
```

**Mandatory change 2 — Test cycle**
```python
# Line 36: change
TEST_CYCLE  = 2022
# to
TEST_CYCLE  = 2025
```

**Mandatory addition 3 — Linear trend baseline**  
The protocol requires three baselines. There is currently no linear trend computation in step2. A new code block computing OLS on pre-2025 `PISA_mathematics` scores vs cycle year for each eligible country must be added or provided as a separate script. Without it, the evaluation is incomplete.

**Advisory — Signed error note**  
After the run, compute signed errors as:
```python
df["signed_err_persistence"] = df["persistence"] - df["true_score"]
df["signed_err_timesfm"]     = df["timesfm_zeroshot"] - df["true_score"]
```
The stored `err_*` columns (absolute values) must not be used for signed-error analysis.

### What the run would use

| Item | Value |
|------|-------|
| Input panel | `outputs/stage5/unified_panel.csv` (confirmed to have PISA 2025 mathematics, 89 countries, 0 duplicates after domain filter) |
| Target column | `PISA_mathematics` |
| Model | `google/timesfm-3.0-pytorch` (pre-trained, not fine-tuned) |
| Eligible countries | 66 (produced by MIN_CONTEXT filter on the panel) |
| Output directory | **Must be** `outputs/step2_2025/` |

---

## Item 8 — The Run Will Not Retrain or Fine-Tune TimesFM

**CONFIRMED.** Evidence cited in Item 5. No training code exists in `step2_timesfm_zeroshot.py`. The model is loaded with `TimesFM3Evaluator(config)` which uses inference mode only. The checkpoint is unmodified by the run.

---

## Item 9 — Official Country-Mean Validation Remains Incomplete

**CONFIRMED — explicitly stated.**

As documented in `phase3_3_data_provenance_status.md` and `phase3_3_pre_evaluation_check.md`:

- Direct comparison of project-computed PISA 2025 country mathematics means against OECD Results Volume I **has not been performed and cannot be performed** from this environment (HTTP 403; no locally stored official mean table found).
- The SE validation (MAD=0.000026, n=79) provides strong indirect evidence of pipeline correctness but **does not constitute point-estimate validation**.
- All evaluation reports must carry the statement: *"PISA 2025 country mathematics means are SE-validated against OECD Annex 14 but have not been independently confirmed against OECD Results Volume I."*
- This limitation qualifies but does not block the provisional evaluation.

---

## Item 10 — Existing Files Preserved; No Commit or Push

No production script has been modified during this preflight check. No output file has been written to `outputs/`. No git commit or push has been made. All files listed in `git status` at session start remain unchanged.

---

## Blockers Summary

| # | Blocker | Type | Required resolution |
|---|---------|------|---------------------|
| B1 | `OUT_DIR` hardcoded to `outputs/step2/` — would overwrite 2022 results | **Hard constraint violation** | Change `OUT_DIR` to `outputs/step2_2025/` before running |
| B2 | No linear trend production script exists | **Protocol incompleteness** | Write or add linear trend computation (OLS on `PISA_mathematics` vs cycle year, predict at 2025) |

| # | Advisory item | Type | Action |
|---|--------------|------|--------|
| A1 | Stored `err_*` columns are absolute, not signed | Interpretation risk | Compute signed errors post-hoc from raw columns |
| A2 | LUX persistence uses 2018, not 2022 | Disclosure required | Document in evaluation report |
| A3 | LOCO n_test=75 ≠ evaluation n=66 | Comparison caveat | Do not compare LOCO MAE directly to final MAE without adjustment |

---

## Decision

Both blockers are **resolvable with minimal, targeted script changes** (two one-line edits + one new code block for linear trend). They are not data problems, not architectural problems, and not protocol problems. They are scripting gaps that were identified precisely because this preflight check was run.

Executing the evaluation as-is (without resolving B1 and B2) would violate the overwrite constraint and produce an incomplete result. Resolving them requires **explicit user authorisation** to make those two targeted edits to `scripts/step2_timesfm_zeroshot.py` (or to an adapted copy) before executing.

---

## **BLOCKED BY UNRESOLVED TECHNICAL ISSUE**

**Reason:** Two issues must be resolved before execution:

1. **B1 (hard):** `OUT_DIR` in `step2_timesfm_zeroshot.py` must be changed from `outputs/step2/` to `outputs/step2_2025/` to prevent overwriting the 2022 development results.

2. **B2 (protocol):** The linear trend baseline has no production script. It must be added before the evaluation can be called complete under the three-baseline protocol.

**Both blockers are resolvable with targeted, minimal edits that require explicit user authorisation before implementation.**
