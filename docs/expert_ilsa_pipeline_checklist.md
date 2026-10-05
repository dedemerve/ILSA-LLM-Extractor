# Expert ILSA × LLM × Forecasting Pipeline Checklist

**Verdict (2026-10-05, updated):** **MVP OPERATIONAL + PISA panel history expanded**

PISA Ridge at 2022 is **unlocked** after ingesting World Bank published means for
2000–2018. Full PISA 2022 forecast-vs-actual (M0/M1/M2) is available.
Microdata BRR upgrade for 2018 (`CY07_MSU_STU_QQQ.sav`) remains preferred when the
OECD zip finishes downloading locally.

**7 ILSAs:** PISA, TIMSS (G8+G4), PIRLS, ICCS, ICILS, PIAAC, TALIS (covariate).

```bash
bash scripts/run_stage5_corpus_pipeline.sh
# Panel expansion only:
python scripts/ingest_pisa_published_means.py
python scripts/build_enriched_panel.py
python scripts/run_loco_forecasting.py
python scripts/build_forecast_feedback_layer.py
```

---

## Direct answers

| Question | Answer |
|----------|--------|
| End-to-end literature → microdata → forecast → revise? | **Yes (MVP).** Corpus = on-disk 1,756 JSON (query-bounded WoS/Scopus). |
| PISA **2025** forecast vs actual? | **Yes** — `pisa_2025_forecast_vs_actual_final.csv` |
| PISA **2022** Ridge forecast vs actual? | **Yes (now)** — `pisa_2022_forecast_vs_actual_final.csv` (204 rows; MAE_M0≈12.4) |
| Multifactorial / survey fusion into W_j? | **Yes** — `docs/ilsa_chaos_survey_rationale.md` |
| Kusursuz / full bibliographic universe? | **No** — WoS/Scopus still 2020–2026 AI/ML∩ILSA; PIAAC forward blocked; REV-002 pending |

---

## Expert task list — status

| # | Task | Status |
|---|------|--------|
| T01 | Extract defined corpus → JSON | **DONE** |
| T02 | Excel / structured meta | **DONE** |
| T03 | Evidence + effect sizes | **DONE** |
| T04 | W_j → M1 | **DONE** |
| T05 | Multifactorial / survey rationale | **DONE** |
| T06 | Crosswalk (OTHER residual) | **DONE** |
| T07 | Country panels for forecast targets | **DONE** (PISA expanded) |
| T08 | Historical PISA cycles in achievement panel | **DONE via PUBLISHED_WB** (microdata BRR for 2018 still upgrading) |
| T09 | Program-adaptive validation | **DONE** (23 OOS program folds) |
| T10 | Forecast each eligible cycle vs actual | **DONE** (ledger 10,492) |
| T11 | PISA 2025 forecast vs actual | **DONE** |
| T12 | PISA 2022 Ridge forecast vs actual | **DONE** |
| T13 | Forward forecasts | **PARTIAL** (PIAAC blocked) |
| T14 | Error → expert revision log | **PARTIAL** (REV-001 accepted; REV-002 pending) |
| T15 | Closed retrain after accepted revisions | **PARTIAL** (REV-001 applied + revalidated) |
| T16 | Broaden WoS/Scopus queries | **NOT DONE** (optional) |
| T17 | Mac microdata full audit | **PARTIAL** |

---

## Key files

- `outputs/stage5/pisa_2022_forecast_vs_actual_final.csv`
- `outputs/stage5/pisa_2025_forecast_vs_actual_final.csv`
- `outputs/stage5/forecast_ledger.csv`
- `outputs/stage5/model_revision_log.csv` (REV-001 accepted)
- `scripts/ingest_pisa_published_means.py`
- `scripts/ingest_pisa_2018_microdata.py` (when SAV present)
