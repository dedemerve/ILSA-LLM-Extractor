# Phase 3.3 — Final Code Review

**Date:** 2026-10-15
**File reviewed:** `scripts/step2_timesfm_zeroshot.py` (read-only; no edits made)
**Preceding audit:** `phase3_3_final_static_safety_audit.md` (this file does not overwrite it)
**Forecasting pipeline NOT executed.**

Each check is classified as VERIFIED, PARTIALLY VERIFIED, or UNRESOLVED.

---

## 1. Common-Country Evaluation

### 1a. Common-country dataset construction

**Code (lines 240–246):**
```python
_common_cols = ["persistence", "linear_trend", "timesfm_zeroshot"]
df_common = df.dropna(subset=_common_cols).copy()
if len(df_common) == 0:
    raise RuntimeError(...)
```

**Checks:**

| Requirement | Code evidence | Classification |
|-------------|---------------|----------------|
| Only rows with valid persistence prediction | `"persistence"` in `dropna` subset | VERIFIED |
| Only rows with valid linear-trend prediction | `"linear_trend"` in `dropna` subset | VERIFIED |
| Only rows with valid TimesFM prediction | `"timesfm_zeroshot"` in `dropna` subset | VERIFIED |
| Only rows with non-missing actual target score | `true_score` is NOT in the `dropna` subset; non-null guaranteed upstream by `sub = sub.dropna(subset=[TARGET_COL])` at line 99 — all rows that reach `records[]` carry a non-null `true_score` by construction | PARTIALLY VERIFIED |

The `true_score` null-safety relies on the upstream `dropna` at line 99, not on an explicit check in the common-country construction. This is logically correct but the guarantee is implicit. A future edit that removes the upstream `dropna` or adds a new code path could silently violate it.

### 1b. MAE and mean signed error on identical country set

**Code (lines 252–263):**
```python
df_common["signed_err_persistence"] = df_common["persistence"]      - df_common["true_score"]
df_common["signed_err_linear"]      = df_common["linear_trend"]     - df_common["true_score"]
df_common["signed_err_timesfm"]     = df_common["timesfm_zeroshot"] - df_common["true_score"]

mae_persist_common = df_common["err_persistence"].mean()
mae_linear_common  = df_common["err_linear"].mean()
mae_tfm_common     = df_common["err_timesfm"].mean()
```

All six quantities operate on the same `df_common` DataFrame, which is a single filtered copy. All three methods therefore use exactly the same country IDs and exactly the same `true_score` values.

**Classification: VERIFIED**

### 1c. Method-specific prediction counts available separately

**Code (lines 233–237):**
```python
n_persist_valid = df["persistence"].notna().sum()
n_linear_valid  = df["linear_trend"].notna().sum()
n_tfm_valid     = df["timesfm_zeroshot"].notna().sum()
```
These are computed on the full `df` before `df_common` is constructed.

Countries excluded before records are built are tracked in `skipped_countries` (Bölüm 1, lines 109–137), with explicit per-country exclusion reasons.

Countries absent from `df_common` but present in `df` are identified at lines 248–250:
```python
_missing_from_common = sorted(set(df["country"]) - set(df_common["country"]))
```

**Classification: VERIFIED**

### 1d. Issue: common-country MAE not saved to metrics_summary.csv

**Finding:** The common-country MAE and mean signed error are printed to stdout (lines 265–269) but are NOT written to `metrics_summary.csv`. The saved file uses `mae_persist`, `mae_linear`, `mae_tfm` computed on the full `df` (lines 306–309, 312–314), with `n` set to `len(df)` (full-set count).

Consequence: downstream users reading `metrics_summary.csv` see full-set MAE with full-set `n`, not the common-country figures. If any country is absent from `df_common` (e.g., TimesFM fails to produce a prediction for one country), the saved and printed MAEs will differ without warning in the file itself.

**Classification: UNRESOLVED** — the common-country analysis exists correctly in memory and is printed, but is not persisted to the output CSV. The saved metrics file does not reflect the common-country comparison.

---

## 2. Linear-Trend Eligibility

### 2a. Fitted only on observations strictly before target cycle

**Code (line 112):**
```python
context_rows = cdata[cdata["cycle"] < TEST_CYCLE]
```
Strict `<` operator; `TEST_CYCLE` is excluded from context. These `context_rows` are passed to `_linear_trend_pred` as both `context_years` and `context_scores`.

**Classification: VERIFIED**

### 2b. At least three valid historical observations required

**Code (lines 118–123):**
```python
if len(context_rows) < MIN_CONTEXT:
    skipped_countries.append((
        country,
        f"yetersiz geçmiş: {len(context_rows)} gözlem (gerekli: {MIN_CONTEXT})"
    ))
    continue
```
`MIN_CONTEXT = 3` (line 41). Any country with fewer than 3 pre-target-cycle observations is excluded before `_linear_trend_pred` is called.

`_linear_trend_pred` has its own guard at line 84 (`if len(context_years) < 2`), but this is never triggered in practice because the calling loop enforces `≥ 3`. The function-level guard is a defensive fallback.

**Classification: VERIFIED**

### 2c. Actual target score not used during model fitting

**Code (`_linear_trend_pred`, lines 86–90):**
```python
cy = np.array(context_years, dtype=np.float64)
cs = np.array(context_scores, dtype=np.float64)
slope, intercept = np.polyfit(cy, cs, 1)
prediction = intercept + slope * float(target_year)
```
`numpy.polyfit` is called on `cy` and `cs` only. `target_year` is used solely as the x-value for prediction, not passed to `polyfit`. `context_years` and `context_scores` are sourced from `context_rows` (cycle < TEST_CYCLE); `true_score` from `target_row` is never passed to this function.

**Classification: VERIFIED**

### 2d. Non-missing historical observations guaranteed

All rows in `sub` have non-null `TARGET_COL` due to `sub = sub.dropna(subset=[TARGET_COL])` (line 99). `context_scores` is drawn directly from `sub` rows via `context_rows[TARGET_COL].tolist()`. No NaN scores can enter the OLS fit.

**Classification: VERIFIED**

---

## 3. Output Safety

### 3a. Fail-fast error before any existing file can be overwritten

**Code (lines 56–68):**
```python
_PLANNED_OUTPUTS = [
    OUT_DIR / "zeroshot_results.csv",
    OUT_DIR / "metrics_summary.csv",
    OUT_DIR / "zeroshot_scatter.png",
]
_existing_outputs = [str(p) for p in _PLANNED_OUTPUTS if p.exists()]
if _existing_outputs:
    raise FileExistsError(...)
```
This runs at module load time, before any data is read and before any output is written. A `FileExistsError` at this point halts the process entirely; no pandas or matplotlib write calls are reached.

**Classification: VERIFIED** for the primary use case.

**Caveat (partial write scenario):** `df.to_csv(OUT_DIR / "zeroshot_results.csv", ...)` is written at line 222, before the `df_common` empty-set check at line 242. If `df_common` is empty and raises `RuntimeError`, `zeroshot_results.csv` will exist on disk but `metrics_summary.csv` and the scatter plot will not. On a subsequent run attempt, the fail-fast guard at line 62 correctly detects `zeroshot_results.csv` and raises `FileExistsError`, preventing further writes. The partial-write state is therefore caught on retry, not silently overwritten.

**Classification: VERIFIED** for overwrite prevention; partial-write scenario is caught on retry.

### 3b. Output directory created before file-existence check

**Code (line 54):**
```python
OUT_DIR.mkdir(parents=True, exist_ok=True)
```
This line executes before the `_existing_outputs` check at lines 62–68.

**Behavior:** For `TEST_CYCLE=2025` (or any non-2022 cycle), `OUT_DIR = ROOT / "outputs" / "step2_2025/"` does not exist on a clean system. `mkdir` creates it as an empty directory, then the file-existence check runs and finds no files, and the script continues normally.

**Consequence:** A fresh run of the script for a new `TEST_CYCLE` leaves behind an empty `outputs/step2_<cycle>/` directory even if the script subsequently aborts before writing any files (e.g., due to a data error). The directory itself is not a data output. This behavior is noted and does not affect correctness.

**No change was made.** This is documented per the review instructions.

---

## 4. Static Validation

### 4a. Syntax check

```
python3 -m py_compile scripts/step2_timesfm_zeroshot.py → SYNTAX OK
```
**Classification: VERIFIED**

### 4b. PISA 2022 output files unchanged

SHA-256 prefix matches against values recorded in `phase3_3_script_change_audit.md`:

| File | SHA-256 (first 12 hex) | Status |
|------|------------------------|--------|
| `outputs/step2/zeroshot_results.csv` | `5b189e2b0b8c` | UNCHANGED |
| `outputs/step2/metrics_summary.csv` | `cdbffdac89df` | UNCHANGED |
| `outputs/step2/zeroshot_scatter.png` | `bacc69ca51af` | UNCHANGED |

**Classification: VERIFIED**

### 4c. Additional findings from static inspection

**Finding A — Stale print statement (line 103):**
```python
print(f"  2022 döngüsünde {len(test_countries)} ülke var.")
```
The string hardcodes "2022" instead of using `TEST_CYCLE`. When run with `TEST_CYCLE=2025`, this would print "2022 döngüsünde 89 ülke var." which is misleading. This is a cosmetic bug; it does not affect any computation.

**Classification: UNRESOLVED** (cosmetic; no impact on results)

**Finding B — metrics_summary.csv uses full-set n, not common-country n (carries forward from 1d):**
The `n` column in the saved metrics file is `len(df)` (full dataset). The common-country MAE uses a potentially smaller subset. If any country lacks a TimesFM prediction, these will diverge. Downstream tooling that reads `metrics_summary.csv` will not detect this.

**Classification: UNRESOLVED** (no impact on correctness of printed output; affects saved file only)

**Finding C — Country-wins comparison on full df, not df_common (lines 287–294):**
```python
tfm_wins = (df["err_timesfm"] < df["err_persistence"]).sum()
lin_wins_vs_persist = (df["err_linear"] < df["err_persistence"]).sum()
lin_wins_vs_tfm     = (df["err_linear"] < df["err_timesfm"]).sum()
```
These per-country win counts are computed on the full `df`, not `df_common`. If any country has a NaN `timesfm_zeroshot`, `df["err_timesfm"] < df["err_persistence"]` will evaluate to `False` (NaN comparisons are False in pandas), which silently understates TimesFM wins. This affects the printed text but not the saved MAE values.

**Classification: PARTIALLY VERIFIED** — win counts are computed on the full DataFrame and may be depressed if TimesFM fails for any country.

---

## 5. Summary of Findings

| # | Check | Classification | Impact |
|---|-------|----------------|--------|
| 1a | Common-country construction (persistence, linear, TimesFM, true_score) | PARTIALLY VERIFIED | `true_score` null-safety implicit; logically correct |
| 1b | MAE and mean signed error on identical country set | VERIFIED | — |
| 1c | Method-specific counts and missing-country reasons available separately | VERIFIED | — |
| 1d | Common-country MAE saved to metrics_summary.csv | UNRESOLVED | Common-country MAE printed but not persisted |
| 2a | Trend fitted on observations strictly before target cycle | VERIFIED | — |
| 2b | Minimum three valid historical observations required | VERIFIED | — |
| 2c | Target score not used in OLS fit | VERIFIED | — |
| 2d | Non-missing historical observations guaranteed | VERIFIED | — |
| 3a | Fail-fast before overwrite | VERIFIED | Partial-write caught on retry |
| 3b | Directory created before file-existence check | VERIFIED (documented) | Empty dir created on abort; not a data file |
| 4a | Syntax | VERIFIED | — |
| 4b | PISA 2022 outputs unchanged | VERIFIED | — |
| 4c-A | Stale "2022" in print statement | UNRESOLVED | Cosmetic only |
| 4c-B | metrics_summary.csv n = full-set, not common-country | UNRESOLVED | Saved file vs. stdout diverge if any method has NaN |
| 4c-C | Per-country win counts on full df (NaN-unsafe comparison) | PARTIALLY VERIFIED | Counts may be understated if TimesFM returns NaN for any country |

---

## 6. Readiness for Provisional PISA 2025 Evaluation

The script is **ready for an explicitly authorized provisional PISA 2025 evaluation**, subject to the following conditions and acknowledgements:

**Conditions that must hold at execution time:**
1. `TEST_CYCLE` must be changed to `2025` before running.
2. `outputs/step2_2025/` must not contain any existing output files (the fail-fast guard enforces this).
3. Execution requires explicit user authorisation (not granted in this review task).

**Findings that do not block execution:**
- Finding 1d (common-country MAE not saved): the correct MAE is computed and printed; the CSV gap is a reporting limitation, not an analytical error.
- Finding 4c-A (stale print string): cosmetic; no computational effect.
- Finding 4c-B (n in saved metrics): affects only the `n` column in `metrics_summary.csv`; the MAE values themselves are correct.
- Finding 4c-C (win counts): affects printed country-win tallies only; MAE and signed-error results are unaffected.

**Findings that introduce ambiguity if TimesFM fails for any country:**
- If `USE_TIMESFM = False` (TimesFM load fails), the `df_common` empty-set check at line 242 will raise `RuntimeError` because `timesfm_zeroshot` will be NaN for all rows and `df_common` will be empty. The script will halt cleanly with an explicit error message. This is the correct behaviour for a three-method evaluation when one method is unavailable.

**Overarching limitations (unchanged from prior documents):**
- PISA 2025 country means are SE-validated (MAD=0.000026, n=79) but not independently confirmed against OECD Results Volume I.
- Script provenance for existing estimates is incomplete (path bug in `build_country_estimates.py`).
- LOCO MAE figures (n=75) are not directly comparable to the final evaluation (n=66).

The script is **ready for execution pending explicit user authorisation**. The PISA 2025 evaluation was NOT started during this review.
