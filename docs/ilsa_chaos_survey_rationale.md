# Design rationale: why literature + surveys enter ILSA forecasting

ILSA outcomes (PISA, TIMSS, PIRLS, ICCS, ICILS, PIAAC) are produced in a
**high-dimensional, partially observed student–school–system**. Scores are
not driven by a single covariate; they reflect SES, home resources, belonging,
ICT access, teacher practices (TALIS), instructional climate, and many other
constructs that appear unevenly across cycles and programs.

## Terminology (important)

Do **not** describe this as a mathematically proven *chaotic dynamical system*.

Prefer:

- complex educational system
- multidimensional / multifactorial outcome structure
- heterogeneous educational contexts
- dynamically evolving educational system

Informal “chaos” only means many interacting factors — not chaos theory.

## Why four literature sources are fused

| Source | Role |
|--------|------|
| OECD / IEA official docs | Construct inventory, frameworks, official variable meaning |
| Scopus / WoS | Empirical effect magnitudes and predictor priority under AI/assessment research |
| Survey / synthesis papers | Cross-program constructs that single-cycle microdata alone under-identify |

Without literature-informed weights, Ridge would treat every lag covariate as
equally important. With `W_j_forecast`, M1 scales features by √W_j so that
constructs repeatedly evidenced in the corpus (e.g. SES, home resources,
teacher efficacy) influence forecasts more than sparsely measured noise.

W_j is **literature priority**, not an effect size and not causal importance.

## How this is implemented (not optional decoration)

1. **Extract** confounders / top predictors from all 1,756 JSON records  
2. **Map** free text + ILSA codes → canonical constructs (`canonical_predictor_map.py`)  
3. **Crosswalk** constructs → microdata codes (`canonical_crosswalk.csv`, incl. TALIS)  
4. **Lag covariates** on the enriched panel (`lag_ESCS`, `lag_BELONG`, TALIS lags, …)  
5. **M0 vs M1**: unweighted Ridge vs literature-scaled Ridge  
6. **Validate** temporally (LOCO / hold-out / LOOCV), then inspect country errors  
7. **Ledger + diagnosis + expert revision** — see `docs/forecast_feedback_framework.md`

Every candidate predictor must still pass construct validity, measurement
comparability, country aggregability, cycle/temporal availability, no leakage,
and literature linkage. Constructs are not forced into every model.

## What “revise and rebuild” means here

Each eligible historical fold is an out-of-sample check against **actual** cycle scores.
Errors feed **expert-supervised** design revision (feature set, weights, cycle calendar)
via `model_revision_log.csv`. This is not an autonomous RL loop.

## PISA 2022 vs 2025 comparison (practical)

| Cycle | Actual scores | Ridge M0/M1 forecast | Persistence M2 |
|-------|---------------|----------------------|----------------|
| **2025** | Yes (`y_true`) | Yes — `pisa_2025_forecast_vs_actual_final.csv` | Yes |
| **2022** | Yes | **No** (n_train=0 under current panel) | Yes only |

Files:
- `outputs/stage5/pisa_2025_forecast_vs_actual_final.csv`
- `outputs/stage5/pisa_2022_forecast_vs_actual_persistence_only.csv`
- `outputs/stage5/forecast_ledger.csv`
