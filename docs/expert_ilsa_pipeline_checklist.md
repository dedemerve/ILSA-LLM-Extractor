# Expert ILSA × LLM × Forecasting Pipeline Checklist

**Document role:** ILSA / AI / assessment–evaluation expert task inventory  
**Date:** 2026-10-05 (updated after hardening pass)  
**Repo:** `dedemerve/ILSA-LLM-Extractor`  
**Status key:** `DONE` | `PARTIAL` | `NOT DONE` | `BLOCKED`

---

## Target architecture

```
A. Literature corpus (OECD + IEA + Scopus + WoS + survey)
        │ LLM extraction → ILSAArticleMetadata JSON
        ▼
B. Evidence layer (effect sizes + construct frequency + W_j_forecast)
        │ canonical crosswalk (literature construct ↔ ILSA code)
        ▼
C. Microdata layer (7 ILSAs × released cycles) via ILSA_MICRODATA_ROOT
        │ PV + replicate-weight country estimates + lag covariates
        ▼
D. Enriched panel (Y_t, Y_{t-1}, X_{t-1})
        │
        ▼
E. Validation (LOCO / hold-out / in-sample diagnostic)
        │ unified load_forecast_weights() → M1 √W_j
        ▼
F. Forward forecasts + per_country_accuracy + SHAP / sensitivity
```

---

## Phase 0 — Corpus & extraction

| ID | Task | Status | Evidence |
|----|------|--------|----------|
| 0.1 | PDF → JSON extraction pipeline | DONE | `src/extractors/`, `prompts/` |
| 0.2 | OECD documents extracted | DONE | 591 JSON |
| 0.3 | IEA documents extracted | DONE | 308 JSON |
| 0.4 | Scopus corpus | PARTIAL | 423 JSON; 2020–2026 AI/ML slice |
| 0.5 | WoS corpus | PARTIAL | 302 JSON; same slice |
| 0.6 | Survey papers | DONE | 132 JSON |
| 0.7 | Deduplicated master table | DONE | ~1,266 unique |
| 0.8 | Corpus provenance reconciled | DONE | Summary uses on-disk **1,756** |
| 0.9 | Schema / anti-hallucination tests | DONE | `tests/` |

---

## Phase 1 — Evidence → weights

| ID | Task | Status | Evidence |
|----|------|--------|----------|
| 1.1 | Effect-size extraction | DONE | 3,083 rows / 577 papers |
| 1.2 | OECD/IEA in effect-size path | NOT DONE | Still survey+Scopus+WoS only |
| 1.3 | Evidence matrix | PARTIAL | Survey-only input |
| 1.4 | Canonical taxonomies | PARTIAL | Parallel RAG vs forecast maps remain |
| 1.5 | Literature priority W_j | DONE | `literature_priority.csv` |
| 1.6 | Unified W_j everywhere | DONE | `scripts/ilsa_common.load_forecast_weights()` wired into LOCO / ICCS / ICILS / ICILS-forward |
| 1.7 | Map free-text → canonical | PARTIAL | OTHER **66.9% → 50.6%** after dictionary expansion (still high) |

---

## Phase 2 — Microdata matching

| ID | Task | Status | Evidence |
|----|------|--------|----------|
| 2.1–2.8 | Estimates for 7 ILSAs + TALIS covariates | DONE | Stage4/5 panels |
| 2.9 | Canonical crosswalk all 7 programs | DONE | ICCS **7 rows** added; 45 total |
| 2.10 | Live variable registry from scan | NOT DONE | Phase-0 template |
| 2.11 | Portable microdata root | DONE | `ILSA_MICRODATA_ROOT` via `ilsa_common.microdata_root()` |

---

## Phase 3 — Forecasting & validation

| ID | Task | Status | Evidence |
|----|------|--------|----------|
| 3.1 | Enriched panels | DONE | |
| 3.2 | Dual-source X matrix unification | PARTIAL | Published path unified on W_j; hierarchy/SHAP long panel still parallel |
| 3.3–3.6 | PISA/TIMSS/TIMSS_G4/PIRLS LOCO | DONE | |
| 3.7 | ICCS LOCO | PARTIAL | 1 fold exploratory |
| 3.8 | ICILS hold-out | DONE | MAE_M0=12.45, **MAE_M1=11.10** (Δ=-1.35 with unified W_j) |
| 3.9 | PIAAC OOS | NOT DONE | In-sample only |
| 3.10–3.13 | Forward cycles | DONE | Official calendar in code |
| 3.14 | PIAAC forward | NOT DONE | |
| 3.15 | Per-country accuracy | DONE | **545 rows**: PISA/PIRLS/TIMSS/TIMSS_G4/ICCS/ICILS |
| 3.16 | SHAP / sensitivity | DONE | |
| 3.17 | Validation design honesty | DONE | Summaries updated |

---

## Phase 4 — Docs / governance

| ID | Task | Status |
|----|------|--------|
| 4.1 | README Stage 5 | DONE |
| 4.2 | Regenerable per_country_accuracy | DONE |
| 4.3 | Official predicted_cycle calendar | DONE |
| 4.4 | Author = Merve only | DONE (policy) |
| 4.5 | Portable microdata config | DONE |

---

## Overall expert verdict

| Question | Answer |
|----------|--------|
| End-to-end MVP built? | **Yes** — 6 programs forecast + ICILS/ICCS country accuracy |
| Every ideal task perfect? | **No** — bibliographic completeness, OECD/IEA→effect path, ~50% OTHER, PIAAC OOS, dual X-panel remaining |
| Honest claim? | Targeted corpus (1,756 JSON; 577 W_j papers) + seven-program microdata estimates + program-appropriate validation |

## Remaining backlog (priority)

1. Further OTHER reduction / optional OECD+IEA effect pass  
2. Collapse hierarchy/SHAP long panel into published X path  
3. Broaden Scopus/WoS if full-literature claim is required  
4. PIAAC third cycle when released → true OOS  
