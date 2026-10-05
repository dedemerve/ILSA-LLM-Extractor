# Expert ILSA × LLM × Forecasting Pipeline Checklist

**Status:** COMPLETE (2026-10-05) — all in-repo deliverables implemented  
**Repo:** `dedemerve/ILSA-LLM-Extractor`

Run full refresh (no LLM API):

```bash
bash scripts/run_stage5_corpus_pipeline.sh
```

---

## Completion summary

| Phase | Status | Evidence |
|-------|--------|----------|
| 0 — Extract 1,756 JSON (OECD/IEA/Scopus/WoS/survey) | DONE | On-disk corpus |
| 1 — Evidence + W_j (full corpus) | DONE | `evidence_matrix.csv` 11,576 rows; `effect_sizes.csv` 13,385 rows; OTHER **6.7%** |
| 2 — Microdata × crosswalk (7 ILSAs) | DONE | `canonical_crosswalk.csv` incl. ICCS; `ILSA_MICRODATA_ROOT` |
| 3 — Forecasting + validation | DONE | LOCO/hold-out/LOOCV per program; forward except PIAAC (documented) |
| 4 — Docs / reproducibility | DONE | README, orchestrator, tests |

---

## Per-program delivery

| Program | Estimates | Literature match | Validation | Forward | Per-country accuracy |
|---------|-----------|------------------|------------|---------|----------------------|
| PISA | DONE | DONE | LOCO | 2028 | DONE |
| TIMSS G8 | DONE | DONE | LOCO | 2027 | DONE |
| TIMSS G4 | DONE | DONE | LOCO | 2027 | DONE |
| PIRLS | DONE | DONE | LOCO | 2026 | DONE |
| ICCS | DONE | DONE | LOCO (1 fold) | 2029 | DONE |
| ICILS | DONE | DONE | Hold-out | 2028 | DONE |
| PIAAC | DONE | DONE | LOOCV 2012→2017 | **Blocked** (2 cycles) | DONE |
| TALIS | covariate | DONE | ablation | N/A | N/A |

PIAAC forward exclusion is explicit in `outputs/stage5/piaac_forward_status.csv` (requires Cycle 3 microdata).

**Chaos / survey rationale:** ILSA scores emerge from many student–school–system
parameters; literature + survey constructs are fused into `W_j` and lag covariates
intentionally — see [`docs/ilsa_chaos_survey_rationale.md`](ilsa_chaos_survey_rationale.md).

**PISA forecast vs actual:**
- PISA **2025**: full M0/M1/M2 vs actual → `outputs/stage5/pisa_2025_forecast_vs_actual.csv`
- PISA **2022**: actual exists; Ridge unavailable (thin history) → persistence-only file

---

## Architecture (final)

```
1,756 JSON → evidence_matrix + effect_sizes (deterministic + legacy LLM)
        → literature_priority (W_j_forecast) + predictor_weights_v2
        → ilsa_common.load_forecast_weights() → M1 across LOCO/ICCS/ICILS/SHAP
Microdata (ILSA_MICRODATA_ROOT) → enriched_panel → run_loco_forecasting.py
Program branches → ICCS / ICILS / PIAAC hold-out → per_country_accuracy.csv (575 rows)
```

---

## Optional future expansion (outside current corpus definition)

- Broaden Scopus/WoS bibliographic queries (new PDF extraction)
- PIAAC forward when OECD releases Cycle 3 microdata
- Live microdata variable registry scan (Phase 1 catalog build on 59 GB)

These are **corpus extensions**, not missing steps in the built pipeline.
