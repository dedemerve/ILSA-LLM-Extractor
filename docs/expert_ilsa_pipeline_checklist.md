# Expert ILSA × LLM × Forecasting Pipeline Checklist

**Document role:** ILSA / AI / assessment–evaluation expert task inventory  
**Date:** 2026-10-05  
**Repo:** `dedemerve/ILSA-LLM-Extractor`  
**Status key:** `DONE` | `PARTIAL` | `NOT DONE` | `BLOCKED` (needs microdata/API outside this VM)

This checklist answers: *Did we fully build the system that extracts OECD/IEA/WoS/Scopus literature with LLMs, matches every study to the seven ILSAs’ microdata cycles, and forecasts from that joint evidence?*

---

## Target architecture (updated recommendation)

```
A. Literature corpus (OECD + IEA + Scopus + WoS + survey)
        │ LLM extraction → ILSAArticleMetadata JSON
        ▼
B. Evidence layer (effect sizes + construct frequency + W_j)
        │ canonical crosswalk (literature construct ↔ ILSA code)
        ▼
C. Microdata layer (7 ILSAs × all released cycles)
        │ PV + replicate-weight country estimates + lag covariates
        ▼
D. Enriched panel (Y_t, Y_{t-1}, X_{t-1})  — single dual-source X matrix
        │
        ▼
E. Validation (program-appropriate: LOCO / hold-out / in-sample diagnostic)
        │
        ▼
F. Forward forecasts + per-country accuracy + SHAP / sensitivity
```

**Design invariants**

1. Official reports (OECD/IEA) inform *construct inventory*; peer literature (Scopus/WoS/survey) informs *effect magnitudes and W_j*.
2. Numeric truth comes from microdata (PV/replicate weights), never from the LLM.
3. One official cycle calendar drives `predicted_cycle` (no mean-gap heuristic).
4. One weight system (`W_j_forecast`) and one X-matrix builder feed published LOCO/forward results.
5. Validation design must be honest: LOCO ≠ hold-out ≠ in-sample.

---

## Phase 0 — Corpus & extraction

| ID | Task | Status | Evidence / note |
|----|------|--------|-----------------|
| 0.1 | PDF → JSON extraction pipeline (schema, prompt, resume) | DONE | `src/extractors/`, `prompts/`, `ilsa_pipeline/` |
| 0.2 | OECD official documents extracted | DONE | 591 JSON under `outputs/OECD/` |
| 0.3 | IEA official documents extracted | DONE | 308 JSON under `outputs/IEA/` |
| 0.4 | Scopus bibliographic corpus extracted | PARTIAL | 423 JSON; **AI/ML × 2020–2026 query only** — not all ILSA literature |
| 0.5 | Web of Science corpus extracted | PARTIAL | 302 JSON; same query window as Scopus |
| 0.6 | Survey / synthesis papers extracted | DONE | 132 JSON under `outputs/ilsa_survey_articles/` |
| 0.7 | Deduplicated master study table | DONE | HF dataset / Excel master (~1,266 unique studies) |
| 0.8 | Full-corpus provenance reconciled (JSON counts) | PARTIAL | On-disk **1,756** JSON; older summary still reported 725 vs 867 |
| 0.9 | Anti-hallucination / schema validation tests | DONE | `tests/` + resanitize pipeline |

**Verdict Phase 0:** Extraction engine is production-grade. Bibliographic recall is **not** “all ILSA literature”; it is a targeted AI/assessment slice plus official docs.

---

## Phase 1 — Literature → evidence → weights

| ID | Task | Status | Evidence / note |
|----|------|--------|-----------------|
| 1.1 | Second-pass effect-size extraction | DONE | `effect_sizes.csv` = **3,083 rows / 577 papers** |
| 1.2 | Include OECD/IEA in effect-size / W_j path | NOT DONE | `extract_effect_sizes.py` sources = survey+Scopus+WoS only |
| 1.3 | Evidence matrix (construct matching) | PARTIAL | Survey-only input; 530 matched / 1,121 full |
| 1.4 | Canonical construct taxonomy | PARTIAL | Multiple parallel taxonomies (RAG vs forecasting vs effect-size) |
| 1.5 | Literature priority weights W_j | DONE | `literature_priority.csv`, `predictor_weights_v2.csv` |
| 1.6 | Unified single W_j used everywhere | NOT DONE | Main LOCO uses v2 `w_norm`; SHAP/ρ uses `W_j_forecast`; ICILS often falls back to frequency |
| 1.7 | Map free-text predictors to canonical codes | PARTIAL | **66.9%** of effect rows are `OTHER` |

**Verdict Phase 1:** Forecasting weights rest on **577 papers**, not the full 1,756 JSON tree. Matching quality is the largest scientific weak point.

---

## Phase 2 — Microdata × literature matching

| ID | Task | Status | Evidence / note |
|----|------|--------|-----------------|
| 2.1 | Country estimates (PV + replicates) for PISA | PARTIAL | Cycles 2015/2022/2025 present; earlier PISA history thin |
| 2.2 | TIMSS G8 estimates (1995–2023) | DONE | In `country_estimates` / enriched panel |
| 2.3 | TIMSS G4 estimates (1995–2023) | DONE | Separate program key `TIMSS_G4` |
| 2.4 | PIRLS estimates (2001–2021) | DONE | |
| 2.5 | ICCS estimates + SES/parental-edu covariates | DONE | `iccs_estimates.csv`, `iccs_enriched_panel.csv` |
| 2.6 | ICILS estimates + covariates | DONE | `icils_piaac_estimates.csv`, `icils_enriched_panel.csv` |
| 2.7 | PIAAC estimates (literacy/numeracy) | DONE | 2012 & 2017 |
| 2.8 | TALIS as cross-program covariate source | DONE | Ablation documented |
| 2.9 | Canonical crosswalk for all 7 programs | PARTIAL | `canonical_crosswalk.csv` covers PISA/TIMSS/PIRLS/TALIS/ICILS/PIAAC; **ICCS rows missing** |
| 2.10 | Live official variable registry from microdata scan | NOT DONE | Phase-0 template only (`PHASE0_TEMPLATE`) |
| 2.11 | Reproducible microdata paths in this environment | BLOCKED | Scripts hardcode `/Users/mrved/Desktop/ILSA Datasets/...` |

**Verdict Phase 2:** Matching exists for the four core student assessments + ICILS/PIAAC constructs; ICCS crosswalk and live registry remain open.

---

## Phase 3 — Forecasting & validation (7 ILSAs)

| ID | Task | Status | Evidence / note |
|----|------|--------|-----------------|
| 3.1 | Enriched panel (lag scores + lag covariates) | DONE | `enriched_panel.csv` (+ v2 long form) |
| 3.2 | Single dual-source X matrix for all published metrics | PARTIAL | Wide path (published) vs long v2 path (hierarchy/SHAP) can diverge |
| 3.3 | PISA LOCO (M0/M1) | PARTIAL | **1 Ridge fold (test=2025)**; 2022 is M2-only |
| 3.4 | TIMSS G8 LOCO | DONE | 6 Ridge folds |
| 3.5 | TIMSS G4 LOCO | DONE | 5 Ridge folds |
| 3.6 | PIRLS LOCO | DONE | 3 Ridge folds |
| 3.7 | ICCS LOCO | PARTIAL | **1 fold (2022)** — exploratory |
| 3.8 | ICILS out-of-sample evaluation | DONE (hold-out) | LOCO infeasible (n_train=3); pooled hold-out MAE≈12.5 |
| 3.9 | PIAAC out-of-sample forecasting | NOT DONE | Only in-sample diagnostic (2 cycles) |
| 3.10 | Forward predictions — PISA→2028 | DONE | CSV corrected; **code calendar now fixed** |
| 3.11 | Forward — TIMSS/TIMSS_G4→2027, PIRLS→2026 | DONE | |
| 3.12 | Forward — ICCS→2029 | DONE (CSV) | Script previously hardcoded 2027 — **fixed to 2029** |
| 3.13 | Forward — ICILS→2028 | DONE | |
| 3.14 | Forward — PIAAC | NOT DONE | Structurally under-identified with 2 cycles |
| 3.15 | Per-country accuracy (5 programs) | PARTIAL | PISA/PIRLS/TIMSS/TIMSS_G4/ICCS; **ICILS missing until hold-out preds regenerated** |
| 3.16 | SHAP + sensitivity / permutation / ρ-align | DONE | Stage5 artifacts present |
| 3.17 | Honest validation design summary | PARTIAL | File existed but PISA fold notes were stale; corpus counts drifted |

**Verdict Phase 3:** Core forecasting MVP is complete for **six** programs (PIAAC diagnostic-only). Not every cell is “kusursuz LOCO”.

---

## Phase 4 — Documentation, reproducibility, governance

| ID | Task | Status | Evidence / note |
|----|------|--------|-----------------|
| 4.1 | README documents Stage 5 as source of truth | PARTIAL → fixed in this PR | Was Stage-4-only |
| 4.2 | Regenerable `per_country_accuracy.csv` | DONE (script) | `scripts/build_per_country_accuracy.py` |
| 4.3 | Official `predicted_cycle` calendar in code | DONE (this PR) | Intervals + ICCS 2029 override |
| 4.4 | Commit attribution = Merve only | DONE (policy) | No Claude Co-Authored-By |
| 4.5 | Environment-portable microdata config | NOT DONE | Needs path config / env var |

---

## Overall expert verdict

| Question | Answer |
|----------|--------|
| Is the end-to-end system *built*? | **Yes — MVP complete** for literature-informed country-level forecasting on PISA, TIMSS G8/G4, PIRLS, ICCS, ICILS. |
| Was every task done *perfectly*? | **No.** Bibliographic completeness, OECD/IEA→W_j inclusion, construct mapping (67% OTHER), unified W_j/X matrix, PIAAC OOS, and ICILS per-country accuracy remain open. |
| Is claiming “all OECD/IEA/WoS/Scopus sources × all cycles × perfect matching” accurate? | **No — overclaim.** Accurate claim: targeted corpus (1,756 JSON; 577 papers in W_j), seven-program microdata estimates, program-appropriate validation, forward forecasts for six programs. |

---

## Priority backlog (recommended next build order)

1. **Unify W_j** — make `W_j_forecast` the sole scaler for M1 across LOCO, ICCS, ICILS, SHAP.
2. **Reduce OTHER rate** — expand canonical predictor dictionary; re-run effect-size mapping without new PDFs first.
3. **Add ICCS to `canonical_crosswalk.csv`** and wire parental-edu/SES codes explicitly.
4. **Regenerate ICILS hold-out predictions** → append ICILS to `per_country_accuracy.csv`.
5. **Reconcile corpus counts** in `validation_design_and_corpus_summary.csv` to on-disk 1,756.
6. **Portable microdata root** via env var (e.g. `ILSA_MICRODATA_ROOT`).
7. **Optional expansion** — broaden Scopus/WoS queries beyond 2020–2026 AI/ML if the claim is full ILSA literature coverage.
8. **Do not** treat PIAAC MAE as cross-program comparable until a third cycle exists.
