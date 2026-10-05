# Expert ILSA × LLM × Forecasting Pipeline Checklist

**Verdict (2026-10-05):** **MVP OPERATIONAL — NOT PERFECT / NOT FULL-SCOPE COMPLETE**

Built and usable for program-adaptive forecasting + PISA 2025 full forecast-vs-actual.
Does **not** yet satisfy the aspirational claim “all OECD/IEA/WoS/Scopus forever × all
ILSA cycles × perfect match × closed revision for every fold.”

**7 ILSAs in this repo:** PISA, TIMSS (G8+G4), PIRLS, ICCS, ICILS, PIAAC, TALIS (covariate source).

Run refresh (no LLM API):

```bash
bash scripts/run_stage5_corpus_pipeline.sh
```

---

## Direct answers (expert)

| Question | Answer |
|----------|--------|
| OECD + IEA + WoS + Scopus (+ survey) → Excel + JSON → evidence → W_j → 7-ILSA microdata → forecast → error → revise? | **Partially yes.** End-to-end architecture exists and runs. Corpus is the **defined on-disk set (1,756 JSON)**, not every bibliographic record ever published. |
| Can I compare **PISA 2025** forecast vs actual for all countries/domains? | **Yes.** `outputs/stage5/pisa_2025_forecast_vs_actual_final.csv` (226 rows; M0/M1/M2 + actual). |
| Can I compare **PISA 2022** full Ridge forecasts vs actual? | **No (Ridge).** Only persistence M2 vs actual: `pisa_2022_forecast_vs_actual_persistence_only.csv`. Cause: panel cycles `{2015,2022,2025}` + missing `CY07_MSU_STU_QQQ.sav` → `n_train=0` for Ridge at 2022. |
| Was literature fused because ILSA outcomes are multifactorial (surveys/context)? | **Yes — by design**, documented in `docs/ilsa_chaos_survey_rationale.md`. Prefer “multifactorial / complex system,” not mathematical chaos theory. |
| Autonomous RL rewrite of models from errors? | **No by design.** Expert-supervised revision log only (`model_revision_log.csv`). |
| “Kusursuz tamamlandı mı?” | **Hayır.** See open tasks below. |

---

## Corpus scope (important honesty)

| Source | On-disk JSON | Scope note |
|--------|--------------|------------|
| OECD | 591 | Extracted reports/articles under `outputs/OECD` |
| IEA | 308 | Extracted under `outputs/IEA` |
| Scopus | 423 | **Query-bounded** 2020–2026 AI/ML/prediction ∩ ILSA program names — not all Scopus ILSA papers ever |
| Web of Science | 302 | Same query bound as Scopus (not “Boğaziçi Science”; repo uses **Web of Science**) |
| Survey / synthesis | 132 | `outputs/ilsa_survey_articles` |
| **Total** | **1,756** | Excel masters under `outputs/*.xlsx` |

Literature → microdata match uses `canonical_crosswalk.csv` + `evidence_matrix.csv` (11,576 matched rows). OTHER≈6.7%.

---

## Expert task list — status of each

| # | Task | Status | Evidence / blocker |
|---|------|--------|--------------------|
| T01 | LLM/deterministic extract of defined corpus → JSON | **DONE** | 1,756 JSON; provenance CSV |
| T02 | Build Excel / structured meta tables from JSON | **DONE** | `ILSA_*Meta*.xlsx`, analytical masters |
| T03 | Evidence matrix + effect sizes from full corpus | **DONE** | evidence 11,576; effect_sizes 13,385 |
| T04 | Literature priority W_j (not causal) + √W_j in M1 | **DONE** | `literature_priority.csv`; `ilsa_common.load_forecast_weights` |
| T05 | Document multifactorial / survey rationale (not chaos theory) | **DONE** | `docs/ilsa_chaos_survey_rationale.md` |
| T06 | Harmonize constructs × 7-ILSA microdata codes | **DONE (with residual OTHER)** | `canonical_crosswalk.csv` incl. ICCS |
| T07 | Build country estimates / enriched panels for forecasting targets | **PARTIAL** | TIMSS/PIRLS full; PISA only 2015/2022/2025; ICCS/ICILS/PIAAC side panels |
| T08 | Ingest **all** historical PISA cycles into achievement panel | **NOT DONE** | Missing early wiring + **PISA 2018 `CY07_MSU_STU_QQQ.sav` absent on Mac** |
| T09 | Program-adaptive temporal validation (LOCO / hold-out / LOOCV) | **DONE** | 17 OOS program-level folds; adaptive designs |
| T10 | Forecast each eligible historical cycle vs actual | **PARTIAL** | Ledger 5,141 rows; PISA 2022 Ridge missing |
| T11 | PISA 2025 forecast vs actual (M0/M1/M2) | **DONE** | `pisa_2025_forecast_vs_actual_final.csv` |
| T12 | PISA 2022 forecast vs actual (full Ridge) | **NOT DONE** | Persistence-only file; REV-001 pending |
| T13 | Forward forecasts (next official cycle) | **PARTIAL** | PISA→2028, TIMSS→2027, PIRLS→2026, ICILS→2028, ICCS→2029; **PIAAC blocked** (need Cycle 3) |
| T14 | Error diagnosis → expert revision log | **SCAFFOLDED** | 5 REV entries; REV-001/002 still `pending` |
| T15 | Closed-loop retrain after accepted revisions | **NOT DONE** | No auto-rewrite; researcher must accept + re-run microdata rebuild |
| T16 | Broaden WoS/Scopus beyond 2020–2026 AI/ML query | **NOT DONE** | Optional corpus expansion |
| T17 | Live re-audit of Desktop microdata paths | **PARTIAL** | Script ready; only PISA 2018 interior verified missing STU_QQQ |

---

## Per-program delivery (honest)

| Program | Estimates | Lit match | Validation | Forward | Forecast↔actual usable? |
|---------|-----------|-----------|------------|---------|-------------------------|
| PISA | 2015/22/25 only | DONE | LOCO (Ridge @2025; M2-only @2022) | 2028 | **2025 full yes; 2022 Ridge no** |
| TIMSS G8 | DONE | DONE | LOCO (6 folds) | 2027 | Yes (historical OOS) |
| TIMSS G4 | DONE | DONE | LOCO (5 folds) | 2027 | Yes |
| PIRLS | DONE | DONE | LOCO (3 folds) | 2026 | Yes |
| ICCS | DONE | DONE | LOCO (1 fold) | 2029 | Limited (thin history) |
| ICILS | DONE | DONE | Hold-out | 2028 | Yes (hold-out) |
| PIAAC | DONE | DONE | LOOCV diagnostic | Blocked | Diagnostic only |
| TALIS | covariate | DONE | ablation | N/A | Not a forecast target |

---

## Architecture (current build)

```
1,756 JSON (query-bounded WoS/Scopus) → Excel + evidence_matrix + effect_sizes
        → literature_priority (W_j_forecast) → √W_j scales M1
Microdata → enriched_panel / side panels → program-adaptive forecast
        → forecast_ledger + error_analysis + model_revision_log (expert)
```

---

## Priority open work (researcher-gated)

1. Obtain **PISA 2018 international `CY07_MSU_STU_QQQ.sav`**; enable CycleSpec; rebuild panel → unlock PISA 2022 Ridge.  
2. Decide REV-001 / REV-002 in `model_revision_log.csv`; re-run validation.  
3. (Optional) Widen Scopus/WoS queries and re-extract.  
4. PIAAC forward only after Cycle 3 microdata.
