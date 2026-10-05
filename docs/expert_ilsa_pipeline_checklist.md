# Expert ILSA × LLM × Forecasting Pipeline Checklist

**Verdict (2026-10-05, final gap-completion pass):** **MVP OPERATIONAL + PISA 2018 BRR upgrade done**

PISA Ridge at 2022 remains unlocked. OECD `SPSS_STU_QQQ.zip` downloaded; `CY07_MSU_STU_QQQ.sav`
ingested → PISA 2018 achievement + ESCS/HOMEPOS/BELONG now **BRR_FAY** (replaced PUBLISHED_WB).
Early cycles 2000–2012 stay PUBLISHED_WB (no cloud microdata). Upstream-blocked items documented only.

**7 ILSAs:** PISA, TIMSS (G8+G4), PIRLS, ICCS, ICILS, PIAAC, TALIS (covariate).

```bash
bash scripts/run_stage5_corpus_pipeline.sh
# PISA 2018 BRR upgrade (requires local SAV; gitignored):
export ILSA_MICRODATA_ROOT="/path/to/ILSA Datasets"
python scripts/ingest_pisa_2018_microdata.py
python scripts/build_enriched_panel.py
python scripts/run_loco_forecasting.py
python scripts/generate_icils_forward_predictions.py
python scripts/build_forecast_feedback_layer.py
python scripts/build_per_country_accuracy.py
```

---

## Direct answers

| Question | Answer |
|----------|--------|
| End-to-end literature → microdata → forecast → revise? | **Yes (MVP).** Corpus = on-disk 1,756 JSON (query-bounded WoS/Scopus). |
| PISA **2025** forecast vs actual? | **Yes** — `pisa_2025_forecast_vs_actual_final.csv` |
| PISA **2022** Ridge forecast vs actual? | **Yes** — `pisa_2022_forecast_vs_actual_final.csv` (210 rows; MAE_M0≈12.81 after 2018 BRR) |
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
| T07 | Country panels for forecast targets | **DONE** |
| T08 | Historical PISA cycles in achievement panel | **DONE** (PUBLISHED_WB 2000–2012; BRR 2015/2018/2022/2025) |
| T09 | Program-adaptive validation | **DONE** (23 OOS program folds) |
| T10 | Forecast each eligible cycle vs actual | **DONE** (ledger 10,543) |
| T11 | PISA 2025 forecast vs actual | **DONE** |
| T12 | PISA 2022 Ridge forecast vs actual | **DONE** |
| T13 | Forward forecasts | **PARTIAL** (PIAAC blocked upstream) |
| T14 | Error → expert revision log | **PARTIAL** (REV-001 accepted; REV-002 pending) |
| T15 | Closed retrain after accepted revisions | **PARTIAL** (REV-001 applied + revalidated) |
| T16 | Broaden WoS/Scopus queries | **NOT DONE** (optional; blocked as corpus expansion) |
| T17 | Mac microdata full audit | **PARTIAL** (cloud: PISA 2018 SAV done; ICCS SES / PIAAC C3 / TALIS 2024 still Mac/upstream) |

---

## Completable this pass vs blocked upstream

| Item | Status |
|------|--------|
| OECD PISA 2018 student QQQ → BRR + covariates | **DONE** (`ingest_pisa_2018_microdata.py`) |
| Rebuild enriched_panel + LOCO + ICILS forward + feedback + per-country | **DONE** |
| Fix unpadded `W_FSTURWT1..80` replicate detection | **DONE** (`get_rep_cols` in `build_country_estimates.py`) |
| ICCS 2022 SES / parental education | **BLOCKED** — no microdata on VM / Mac Desktop |
| PIAAC Cycle 3 | **BLOCKED_UPSTREAM** — not published |
| TALIS 2024 country files | **BLOCKED_UPSTREAM** — not published |
| Full WoS/Scopus bibliographic expansion | **OPEN_OPTIONAL** — query redesign, not microdata |

---

## Key files

- `outputs/stage5/pisa_2022_forecast_vs_actual_final.csv` (MAE_M0≈12.81, MAE_M1≈12.81, MAE_M2≈12.27)
- `outputs/stage5/pisa_2025_forecast_vs_actual_final.csv`
- `outputs/stage5/forecast_ledger.csv`
- `outputs/stage5/model_revision_log.csv` (REV-001 accepted; 2018 BRR noted)
- `outputs/stage5/microdata_missing_inventory.csv`
- `outputs/stage5/per_country_accuracy.csv`
- `scripts/ingest_pisa_published_means.py`
- `scripts/ingest_pisa_2018_microdata.py`
- `scripts/build_country_estimates.py` (`get_rep_cols` unpadded/padded)
