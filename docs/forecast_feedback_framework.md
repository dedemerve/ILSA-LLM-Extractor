# Forecast → Actual → Error → Expert-Supervised Revision Framework

**Status:** Implemented as auditable artifacts (not autonomous RL)  
**Primary script:** `scripts/build_forecast_feedback_layer.py`  
**Outputs:** `outputs/stage5/forecast_ledger.csv`, `forecast_error_analysis.csv`, `model_revision_log.csv`, `final_pipeline_audit.csv`

---

## 1. Scientific chain

```
Literature (OECD + IEA + Scopus + WoS + survey)
    → LLM / deterministic evidence extraction
    → canonical constructs
    → W_j literature-priority vector  (NOT an effect size; NOT causal importance)
    → ILSA microdata (7 programs)
    → country × program × cycle × domain panel
    → M0 / M1 / M2 (/ M3) forecasts
    → actual ILSA outcomes
    → forecast errors
    → error diagnosis
    → expert-supervised revision recommendations
    → next forecast (after researcher approval)
```

This is **not** reinforcement learning. The pipeline may recommend revisions; the researcher accepts or rejects them. Accepted changes must be versioned (`model_revision_log.csv`).

---

## 2. Why multiple ILSAs and survey/contextual variables

Student achievement is **multifactorial and multidimensional** (not a proven mathematical “chaotic dynamical system”). Integrating PISA, TIMSS, PIRLS, ICCS, ICILS, PIAAC, and TALIS (covariate source) allows:

- literature constructs (SES, home resources, teacher efficacy, belonging, ICT, climate, …)
- to map to ILSA-specific operationalizations
- and enter forecasting as lag covariates scaled by √W_j in M1

Every candidate predictor still must pass construct validity, cycle comparability, country aggregability, no leakage, and literature linkage checks. Constructs are **not** forced into every model.

See also: `docs/ilsa_chaos_survey_rationale.md` (terminology: complex / multifactorial — not chaos theory).

---

## 3. Models

| Model | Definition |
|-------|------------|
| **M0** | Ridge, no literature weighting |
| **M1** | Ridge with feature scaling \(X_j^{(M1)} = X_j \sqrt{W_j}\) |
| **M2** | Persistence baseline |
| **M3** | AR(1) / hierarchical variant where implemented |

**ΔMAE convention (global):** \(\Delta\mathrm{MAE} = \mathrm{MAE}_{M1} - \mathrm{MAE}_{M0}\)  
(\(\Delta < 0\) ⇒ M1 better; \(\Delta > 0\) ⇒ M0 better)

W_j is **literature priority**, not effect magnitude and not causal importance.

---

## 4. Program-adaptive validation (verified)

| Program | Design | Verified OOS folds / notes |
|---------|--------|----------------------------|
| PISA | Expanding-window LOCO | **1** Ridge fold (test=2025); 2022 = M2-only |
| TIMSS G8 | Expanding-window LOCO | **6** Ridge folds |
| TIMSS G4 | Expanding-window LOCO | **5** Ridge folds |
| PIRLS | Expanding-window LOCO | **3** Ridge folds |
| ICCS | Expanding-window LOCO | **1** fold (exploratory) |
| ICILS | Pooled hold-out | **1** hold-out (not LOCO) |
| PIAAC | Single-transition LOOCV | **Diagnostic only** — not comparable to LOCO MAE |
| TALIS | Covariate source | Not a forecasting target |

**Total program-level OOS temporal folds (excl. PIAAC diagnostic): 17**  
(PISA1 + TIMSS6 + TIMSS_G45 + PIRLS3 + ICCS1 + ICILS1)

Not every historical cycle is independently forecastable: eligibility depends on preceding-cycle availability, country overlap, predictor availability, and program structure.

---

## 5. Forecast ledger

`outputs/stage5/forecast_ledger.csv` stores country-level:

forecast vs actual vs error for each eligible historical target, with:

- `training_cycles` (only cycles **before** target)
- `is_out_of_sample` / `is_diagnostic`
- model / panel / W_j versions

Filter example: `program==PISA & target_cycle==2025 & domain==mathematics`.

---

## 6. PISA 2022 vs 2025

| Target | Actual | M0/M1 Ridge | M2 | Artifact |
|--------|--------|-------------|-----|----------|
| **2025** | Yes | Yes | Yes | `pisa_2025_forecast_vs_actual_final.csv` |
| **2022** | Yes | **No** (n_train=0) | Yes | persistence-only CSV; ledger has M2 only |

PISA 2022 Ridge is **not fabricated**. Enabling it requires expanding the PISA panel history (see `model_revision_log.csv` REV-001).

---

## 7. Error diagnosis → revision

`forecast_error_analysis.csv`: country / fold / M0-vs-M1 diagnostics.  
`model_revision_log.csv`: researcher-controlled change proposals (pending/accepted).

Workflow after an actual cycle is released:

1. Retrieve forecasts from ledger  
2. Attach actuals → errors  
3. Diagnose (bias, coverage, design limits)  
4. Log revision recommendation  
5. Researcher accepts/rejects  
6. If accepted: update specification, re-run affected folds, bump model version  
7. Produce next forward forecast with explicit `forecast_origin`, versions, and validation basis  

---

## 8. Forward vs historical vs diagnostic

| Label | Meaning |
|-------|---------|
| HISTORICAL OOS | Train on past; score against known actual |
| FORWARD | Next unobserved cycle (e.g. PISA→2028) |
| IN-SAMPLE DIAGNOSTIC | PIAAC LOOCV — not LOCO-comparable |

Do not mix these in the same MAE league table.

---

## 9. What this framework does *not* claim

- Causality from lagged predictors or from W_j  
- That M1 always beats M0  
- That all seven ILSAs share one validation design  
- That every cycle is forecastable  
- That the education system is a proven chaotic dynamical system  
- That the model autonomously rewrites itself from errors  
