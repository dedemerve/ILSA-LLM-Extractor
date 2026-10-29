# Phase 3.4 — PISA 2025 TimesFM Output Provenance Reconciliation

**Date:** 2026-10-15  
**Scope:** Read-only reconciliation of existing PISA 2025 TimesFM outputs  
**Status:** Output arithmetic and panel linkage verified; execution provenance unresolved  
**Integrity:** Existing inputs and outputs were not modified.

---

## 1. Question

Earlier Phase 3.3 readiness artifacts say that no TimesFM PISA 2025 evaluation had been run, while `outputs/step2_2025/` currently contains prediction and metric files. This audit determines what can be confirmed from the current repository state.

## 2. Current Files and Hashes

| File | Modified (local time, Europe/Istanbul) | SHA-256 |
|---|---|---|
| `outputs/step2_2025/metrics_summary.csv` | 2026-10-15 07:43:09 | `cd065845f6b6a4ccd6c86d403142eb90c6a4e3edd07171e6d10ca219d62da01e` |
| `outputs/step2_2025/zeroshot_results.csv` | 2026-10-15 07:43:09 | `166c61b2c7a7b9482f148ff746221c7c16a0694ef18d00d95e7b24d841b48224` |
| `outputs/step2_2025/zeroshot_scatter.png` | 2026-10-15 07:43:09 | `c0bf5ae970385039609150918d32a9080a3e0d3c13815d1a0f6f3230116507c3` |
| `outputs/stage5/unified_panel.csv` | 2026-10-14 20:43:03 | `4df9ef31c198e3988aac351c4529e95af0b00e529068875f5f7906dd7f73f704` |
| Current `scripts/step2_timesfm_zeroshot.py` | 2026-10-15 07:43:13 | `a635af95f9dc1ca4633edfa369a319bfba9e5aa2eea4a67b9541fd87d51dd367` |

The three result files predate the current script file modification time by four seconds. This timestamp order alone does not establish which script contents were executed.

## 3. Checks Reproduced from Current Files

Using `outputs/stage5/unified_panel.csv` and the existing results CSV:

- The panel contains 89 PISA mathematics countries with a 2025 score.
- Applying the documented eligibility rule (2025 score present and at least three pre-2025 cycles) yields 66 countries.
- The results CSV contains exactly those 66 eligible countries, with no duplicates or omissions.
- All 66 `true_score` values match the corresponding 2025 mathematics values in the panel.
- Every row's `n_context` equals the number of years listed in `context_cycles_str`.
- All three prediction columns and error columns are populated for all 66 rows.
- Recomputed metrics from the raw predictions are:

| Method | Recomputed MAE | Recomputed mean signed error (prediction − actual) | n |
|---|---:|---:|---:|
| Persistence | 11.5836 | +6.2613 | 66 |
| Linear trend | 15.1879 | +12.1595 | 66 |
| TimesFM zero-shot | 12.8642 | +8.3664 | 66 |

The saved rounded MAEs (11.58, 15.19, 12.86) agree with the recomputation.

These checks establish consistency of the existing output rows with the current panel and arithmetic. They do not establish which executable code, model checkpoint, dependency versions, or runtime produced the TimesFM predictions.

## 4. Provenance Gaps

- The current `scripts/step2_timesfm_zeroshot.py` sets `TEST_CYCLE = 2022`; it is not a source snapshot configured for the 2025 run.
- The script is untracked in the current Git worktree, so no committed revision identifies its execution-time contents.
- No run-specific stdout/stderr, environment manifest, checkpoint revision, command record, or input manifest for this 2025 run was found in the repository search.
- The results CSV includes the 2025 `true_score`, as required for scoring. This means the file contains evaluation targets and must be treated as outcome-exposed in future model selection.
- Existing Phase 3.3 statements that say “no 2025 TimesFM output exists” or “the evaluation was not run” conflict with current repository state and need a versioned correction before manuscript use.

## 5. Interpretation and Decision

**Verified:** The stored 2025 file is a complete 66-country PISA mathematics comparison of persistence, linear trend, and TimesFM zero-shot; its targets match the current panel and its metrics recompute correctly.

**Not verified:** That the intended locked protocol was followed, the current source code produced the outputs, the expected TimesFM checkpoint was used, or the execution was authorized under the readiness decisions documented earlier.

Treat these files as **existing, arithmetically consistent, but execution-provenance-unresolved results**. Do not overwrite them. Do not use 2025 outcomes to tune or select the next model configuration. Because the 2025 outcome values and results have now been inspected, this dataset cannot serve as a blind confirmation set for the forthcoming model-development decisions. It may still be reported as a retrospective benchmark with the exposure and provenance caveats stated above.

## 6. Next Evidence Needed

1. Search the host's accessible execution records for a run command, temporary source copy, stdout/stderr, or notebook that produced the files.
2. If no such records exist, retain the result hashes above and mark execution provenance permanently unresolved; do not try to recreate provenance by rerunning over these outputs.
3. Freeze a new versioned protocol and create separate output paths for all future experiments.
4. Reserve a genuinely uninspected evaluation target for the final confirmatory comparison. Use PISA 2025 only as an exposed retrospective benchmark.
