# Phase 0 Methodological Specification

**Project:** ILSA Literature Extraction, Verification, and Predictive Matrix  
**Document version:** 1.0.0-phase0  
**Status:** Steering committee approved — pre-implementation blueprint  
**Scope:** Metadata layers, verification engine specification, Excel expansion, agent architecture  
**Explicit exclusion:** No execution on 59 GB raw microdata in Phase 0  

**Companion documents:**

- [phase0_clean_excel_sheets_5_7_schema.md](./phase0_clean_excel_sheets_5_7_schema.md) — Sheet 5–7 column and validation reference  
- `outputs/ilsa_official_variable_registry.json` — Layer 2 pilot template (PISA 2012 ESCS / PV Math)  
- `outputs/ilsa_microdata_catalog/schema_manifest.json` — Layer 1 storage manifest  

---

## Executive Summary

Phase 0 elevates the project from a passive literature meta-synthesis repository (`ILSA_Meta_Analysis_Dataset_CLEAN.xlsx`, Sheets 0–4) to a **verification-capable and prediction-planning control system** without loading 59 GB of raw SPSS microdata. Three deterministic metadata layers and three append-only Excel sheets form the infrastructure:

| Layer | Artifact | Function |
|-------|----------|----------|
| **Layer 1** | `outputs/ilsa_microdata_catalog/` | Index `.sav` files and variables (metadata-only) |
| **Layer 2** | `outputs/ilsa_official_variable_registry.json` | Bridge literature labels → official SPSS codes |
| **Layer 3** | `scripts/verify_claim.py` (specification) | Deterministic replication engine |
| **Excel 5–7** | Append-only sheets | Claims, results, prediction inventory |

**Design invariant:** All numeric truth flows from deterministic computation; LLMs may extract or narrate claims but never generate statistics.

---

## Section 1 — Detailed Component Specifications & Structural Schemas

### 1.1 Design Principles

1. **Determinism over inference** — Indexing, bridging, and verification are rule-based and auditable.  
2. **Separation of concerns** — Literature (Sheets 0–4), claims (5), results (6), prediction inventory (7) remain distinct.  
3. **Index, don’t ingest** — Microdata is referenced by path until a bounded verification job loads a subset.  
4. **Psychometric fidelity** — PVs, sampling weights, and replicate weights are first-class schema entities.

### 1.2 Layer 1 — Microdata Catalog

#### Storage layout

```
outputs/ilsa_microdata_catalog/
├── parquet/
│   ├── catalog_files.parquet
│   ├── catalog_variables.parquet
│   └── catalog_countries.parquet
├── duckdb/
│   └── ilsa_microdata_catalog.duckdb
├── schema_manifest.json
└── manifest.json                    # build provenance (Phase 1+)
```

#### Entity-relationship model

```
catalog_files (1) ──< catalog_variables (N)
        │
        └──< catalog_countries (N)
```

---

#### Table: `catalog_files`

| Column | Data Type | Nullable | Description | Constraints / Allowed Values |
|--------|-----------|----------|-------------|------------------------------|
| `file_id` | `STRING` | NO | Stable surrogate key | SHA-256 prefix + ISA, e.g. `PISA_a3f9c2e1b8d04f12` |
| `isa` | `STRING` | NO | Assessment program | `PISA`, `TIMSS`, `PIRLS`, `ICCS`, `ICILS`, `PIAAC`, `TALIS` |
| `cycle` | `INT16` | NO | Administration year | e.g. `2012`, `2019`, `2022` |
| `cycle_label` | `STRING` | YES | Official cycle name | e.g. `Cycle 6` |
| `grade` | `STRING` | YES | Target grade / age | `4`, `8`, `9`, `15`, `Adult`, `NA` |
| `file_role` | `STRING` | NO | File type | `student`, `school`, `teacher`, `parent`, `principal`, `cognitive`, `other` |
| `file_path` | `STRING` | NO | Absolute path to `.sav` | Must exist at build time |
| `file_name` | `STRING` | NO | Basename | e.g. `CY6_MS_CMB_STU_QTM.sav` |
| `file_mask` | `STRING` | NO | Pattern for registry matching | e.g. `CY*_MS_CMB_STU_*` |
| `file_size_bytes` | `INT64` | NO | File size | ≥ 0 |
| `n_rows` | `INT32` | NO | Case count | ≥ 0 |
| `n_vars` | `INT16` | NO | Variable count | ≥ 0 |
| `spss_creation_date` | `TIMESTAMP` | YES | SPSS header date | ISO-8601 |
| `catalog_build_version` | `STRING` | NO | Pipeline semver | e.g. `1.0.0` |
| `catalog_built_at` | `TIMESTAMP` | NO | Build timestamp | UTC |
| `metadata_only_scan` | `BOOL` | NO | No full data load | Always `true` |
| `has_replicate_weights` | `BOOL` | NO | Replicate weights present | Derived from variable scan |
| `primary_weight_var` | `STRING` | YES | Main weight variable | e.g. `W_FSTUWT`, `TOTWGT` |
| `country_var_candidate` | `STRING` | YES | Country ID field | e.g. `CNT`, `IDCNTRY` |
| `notes` | `STRING` | YES | Parser warnings | Free text |

**Primary key:** `file_id`  
**Unique:** (`file_path`)  
**Indexes:** (`isa`, `cycle`, `file_role`), (`file_mask`)

---

#### Table: `catalog_variables`

| Column | Data Type | Nullable | Description | Constraints |
|--------|-----------|----------|-------------|-------------|
| `variable_id` | `STRING` | NO | Surrogate key | `{file_id}::{variable_name}` |
| `file_id` | `STRING` | NO | FK → `catalog_files` | Must exist |
| `variable_name` | `STRING` | NO | SPSS name | Uppercase ILSA convention |
| `variable_label` | `STRING` | YES | Question / value label | Truncated at 512 chars |
| `var_type` | `STRING` | NO | Storage type | `numeric`, `string`, `date`, `unknown` |
| `measurement_level` | `STRING` | YES | Measurement level | `nominal`, `ordinal`, `scale`, `unknown` |
| `is_weight` | `BOOL` | NO | Sampling weight flag | Rule-based |
| `is_replicate_weight` | `BOOL` | NO | Jackknife / BRR replicate | e.g. `W_FSTR1`…`W_FSTR80` |
| `is_pv` | `BOOL` | NO | Plausible value draw | Pattern per ISA |
| `is_country_code` | `BOOL` | NO | Country identifier | `CNT`, `IDCNTRY`, etc. |
| `is_achievement_scale` | `BOOL` | NO | Non-PV scaled score | IRT/WLE scales |
| `pv_domain` | `STRING` | YES | PV domain | `math`, `reading`, `science`, `civic`, `other` |
| `pv_draw_index` | `INT8` | YES | Draw index | 1–10 (PISA), 1–5 (TIMSS), etc. |
| `missingness_pct` | `FLOAT32` | YES | Missing rate estimate | Phase 1+ optional |
| `value_labels_hash` | `STRING` | YES | Label hash | Audit |
| `detection_rule_id` | `STRING` | NO | Detection rule | e.g. `RULE_PV_PISA_V1` |

**Primary key:** `variable_id`  
**Indexes:** (`file_id`, `is_pv`), (`file_id`, `is_weight`), (`variable_name`)

---

#### Table: `catalog_countries`

| Column | Data Type | Nullable | Description | Constraints |
|--------|-----------|----------|-------------|-------------|
| `country_row_id` | `STRING` | NO | Surrogate key | `{file_id}::{country_code}` |
| `file_id` | `STRING` | NO | FK → `catalog_files` | Must exist |
| `country_code` | `STRING` | NO | Country code | ISO3 or ISA-specific |
| `country_code_system` | `STRING` | NO | Code system | `ISO3`, `PISA_CNT`, `TIMSS_IDCNTRY`, `OTHER` |
| `country_label` | `STRING` | YES | Display name | e.g. `Turkey` |
| `n_unweighted` | `INT32` | YES | Cases in file | ≥ 0 |
| `is_analyzable` | `BOOL` | NO | Meets minimum *n* | Committee-defined threshold |

**Primary key:** `country_row_id`  
**Purpose:** Satisfies `countries_available` without denormalizing variable rows.

---

#### Boolean detection rules (deterministic)

| Flag | Rule ID | Logic |
|------|---------|-------|
| `is_weight` | `RULE_WGT_01` | Name ∈ registry OR `^(W_FSTUWT\|TOTWGT\|SENWGT\|HOUWGT\|STUDWT\|SCHWGT)$` |
| `is_replicate_weight` | `RULE_WGT_02` | `^W_FSTR[0-9]+$`, BRR/jackknife patterns per ISA manual |
| `is_pv` | `RULE_PV_01` | PISA: `^PV[0-9]+(MATH\|READ\|SCIE\|CPS\|CT)$`; TIMSS: cycle-specific extensions |
| `is_country_code` | `RULE_GEO_01` | `{CNT, IDCNTRY, COUNTRY, ISO}` per ISA |

---

#### DuckDB views (read-only)

```sql
CREATE VIEW v_catalog_file_summary AS
SELECT
    f.file_id, f.isa, f.cycle, f.grade, f.file_role,
    f.file_path, f.n_rows, f.n_vars,
    f.primary_weight_var, f.country_var_candidate,
    COUNT(DISTINCT c.country_code) AS n_countries_available,
    SUM(CASE WHEN v.is_pv THEN 1 ELSE 0 END) AS n_pv_vars,
    SUM(CASE WHEN v.is_weight THEN 1 ELSE 0 END) AS n_weight_vars
FROM catalog_files f
LEFT JOIN catalog_variables v ON f.file_id = v.file_id
LEFT JOIN catalog_countries c ON f.file_id = c.file_id
GROUP BY f.file_id, f.isa, f.cycle, f.grade, f.file_role,
         f.file_path, f.n_rows, f.n_vars,
         f.primary_weight_var, f.country_var_candidate;
```

---

### 1.3 Layer 2 — Official Variable Registry

**Artifact:** `outputs/ilsa_official_variable_registry.json`

**Purpose:** Deterministic bridge between free-text literature labels, canonical taxonomy (`canonical_variable`), and official microdata fields.

**Entry requirements:** (a) codebook citation, (b) catalog confirmation (Phase 1+), or (c) approved heuristic rule ID.

**Pilot triad (Phase 0 template):** PISA 2012 — `ESCS` (predictor) → `PV1MATH`…`PV10MATH` (outcome PV set) — see committed JSON template.

#### `VariableEntry` schema (summary)

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `registry_key` | string | YES | e.g. `PISA_ESCS` |
| `canonical_variable` | string | YES | Links to taxonomy, e.g. `SES` |
| `isa` | string[] | YES | Applicable programs |
| `cycles` | int[] | YES | Applicable cycles |
| `official_codes` | object[] | YES | Per-cycle SPSS names |
| `aliases` | string[] | NO | Literature synonyms |
| `role` | enum | YES | `predictor`, `outcome_pv`, `weight`, … |
| `file_mask` | string[] | YES | e.g. `CY*_STU_*` |
| `confidence_tier` | enum | YES | `A_codebook`, `B_catalog`, `C_heuristic` |

#### `PlausibleValueSetEntry` — Rubin constraints

| Field | Required | Description |
|-------|----------|-------------|
| `pooling_rule` | YES | Default: `rubin_rules` |
| `n_draws` | YES | e.g. `10` for PISA 2012 math |
| `rubin_constraints.never_pool_across_domains` | YES | Must be `true` |
| `rubin_constraints.never_average_without_variance_adjustment` | YES | Must be `true` |

---

## Section 2 — CLEAN Excel Expansion (Sheets 5, 6, 7)

Sheets 0–4 and `_FilterLists` semantics for literature layers are **unchanged**. Sheets 5–7 are append-only operational layers.

**Full column headers, dropdown bindings, and pilot sample rows:**  
→ [phase0_clean_excel_sheets_5_7_schema.md](./phase0_clean_excel_sheets_5_7_schema.md)

**Initializer command:**

```bash
python scripts/init_phase0_verification_sheets.py
```

### Join discipline

| Sheet | Primary key | Foreign keys |
|-------|-------------|--------------|
| `5_Verification_Claims` | `claim_id` | `file_name` → Sheet 1 |
| `6_Verification_Results` | `result_id` | `claim_id` → Sheet 5 |
| `7_Prediction_Inventory` | `inventory_id` | `canonical_variable` → `Canonical_View` |

### `catalog_match_confidence` rubric

| Tier | Criteria |
|------|----------|
| **High** | Unique `file_id`; all variables in catalog; countries in `catalog_countries` |
| **Medium** | File resolved; ≥1 variable via alias; or partial country subset |
| **Low** | Ambiguous cycle/ISA; variable not in catalog; manual adjudication required |

---

## Section 3 — Deterministic Verification Engine (`verify_claim.py`)

### 3.1 Module architecture (Phase 1 implementation gate)

```
scripts/verify_claim.py
├── cli.py
├── claim_loader.py          # Sheet 5 / parquet
├── catalog_resolver.py      # DuckDB → file_id
├── registry_resolver.py     # Layer 2 JSON
├── data_loader.py           # Bounded pyreadstat load
├── weight_engine.py
├── pv_engine.py             # Rubin pooling
├── check_descriptive.py
├── check_correlation.py
├── result_writer.py         # Sheet 6 + evidence bundle
└── manifest.py              # Reproducibility hash
```

**Invariant:** No generative API imports.

### 3.2 Main algorithm (pseudocode)

```
FUNCTION verify_claim(claim_id, check_types[], config):

    claim ← LOAD FROM Sheet5 WHERE claim_id = claim_id
    ASSERT claim.verification_eligible = TRUE

    file ← RESOLVE via DuckDB(isa, cycle, file_role, file_mask)
    vars ← CROSSCHECK registry + catalog_variables

    cols ← outcome PV draws + predictors + weight + country
    df ← READ_SAV(file_path, columns=cols)
    df ← FILTER country IN claim.countries_iso3
    w ← df[weight_var]

    FOR check IN check_types:
        result ← RUN_CHECK(check, df, w, claim, config)
        APPEND TO Sheet6 AND evidence_path

    RETURN evidence_bundle
```

### 3.3 Rubin’s Rules (1987) — Plausible Value Pooling

For $m$ plausible values, let $\hat{\theta}_j$ denote the weighted statistic on draw $j$ (e.g. Pearson $r$ between ESCS and `PVjMATH`), with associated within-imputation variance $U_j$.

**Within-imputation variance (mean across draws):**

$$\bar{U} = \frac{1}{m} \sum_{j=1}^{m} U_j$$

**Between-imputation variance:**

$$B = \frac{1}{m - 1} \sum_{j=1}^{m} \left( \hat{\theta}_j - \bar{\theta} \right)^2, \quad \bar{\theta} = \frac{1}{m} \sum_{j=1}^{m} \hat{\theta}_j$$

**Total variance (Rubin, 1987):**

$$T = \bar{U} + \left(1 + \frac{1}{m}\right) B$$

**Standard error:** $\text{SE}(\bar{\theta}) = \sqrt{T}$

**Engine outputs:** `replicated_value` = $\bar{\theta}$; `replicated_se` = $\sqrt{T}$; `pv_rule_applied` = `rubin_rules`.

**Directional classification (pilot):**

$$\text{direction} = \begin{cases} \text{Positive} & \text{if } \bar{\theta} > \delta_{+} \\ \text{Negative} & \text{if } \bar{\theta} < \delta_{-} \\ \text{Null} & \text{otherwise} \end{cases}$$

with committee-defined practical thresholds $\delta_{+}, \delta_{-}$ (default: $\pm 0.05$ for correlation).

### 3.4 Pilot check — PISA 2012 Turkey ESCS × Mathematics PV

1. Load claim `CLM_PISA_2012_000001`.  
2. Resolve student file via `CY6_*STU*` mask.  
3. Filter `CNT` → Turkey (`TUR`).  
4. Apply `W_FSTUWT`.  
5. Compute Rubin-pooled weighted correlation: ESCS vs. `PV1MATH`…`PV10MATH`.  
6. Compare `replicated_direction` to `paper_reported_direction`.  
7. Write `match_status` ∈ {`Full_Match`, `Partial_Match`, `Direction_Mismatch`, …}.

---

## Section 4 — Agentic RAG Integration Architecture

### 4.1 Collection inventory

| Collection | Source | Document unit |
|------------|--------|---------------|
| `ilsa_literature_synthesis` | `final_knowledge_synthesis_v4.csv` | Aggregated trends |
| `ilsa_verification_claims` | Sheet 5 | One claim |
| `ilsa_microdata_catalog` | Layer 1 Parquet | Variable/file metadata |
| `ilsa_verification_results` | Sheet 6 | One check result |
| `ilsa_codebook_registry` | Layer 2 JSON | Variable definitions |

### 4.2 Orchestration workflow (strict layers)

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ LAYER 0 — USER QUERY (natural language)                                      │
│ "Is ESCS–mathematics achievement in PISA 2012 Turkey positive in raw data?"  │
└──────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ LAYER 1 — DETERMINISTIC SLOT PARSER (regex + enum; NOT generative inference) │
│ intent=VERIFY_CLAIM │ isa=PISA │ cycle=2012 │ country=TUR                    │
│ predictor=ESCS │ outcome=Math │ expected_direction=Positive                  │
└──────────────────────────────────────────────────────────────────────────────┘
                    ┌─────────────────┴─────────────────┐
                    ▼                                       ▼
┌─────────────────────────────┐         ┌────────────────────────────────────┐
│ LAYER 2A — LITERATURE RAG    │         │ LAYER 2B — STRUCTURED CLAIM LOOKUP │
│ Vector: synthesis + claims   │         │ DuckDB/Parquet on Sheet 5          │
│ Filter: empirical_finding    │         │ → claim_id = CLM_PISA_2012_000001  │
└─────────────────────────────┘         └────────────────────────────────────┘
                    │                                       │
                    └─────────────────┬─────────────────┘
                                      ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ LAYER 3 — CLAIM MATERIALIZATION                                              │
│ paper_claim │ catalog_match_confidence │ official_outcome_vars │ pv_rule     │
└──────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ LAYER 4 — TOOL INVOCATION (EXTERNAL SUBPROCESS — ZERO LLM)                   │
│ python scripts/verify_claim.py --claim-id CLM_PISA_2012_000001               │
│                  --checks correlation_direction                               │
└──────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ LAYER 5 — RESULT INGESTION (deterministic read)                              │
│ Sheet 6 / evidence_path/summary.json → replicated_value, match_status        │
└──────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ LAYER 6 — NARRATION (LLM permitted — prose ONLY; numbers from Layer 5)       │
│ [Literature Claim] vs [Raw Data Result] vs [Alignment Status] + caveats      │
│ HARD RULE: No replicated statistic without result_id + evidence_path         │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 4.3 Orchestration accountability matrix

| Step | Actor | Stochastic? | Produces numbers? |
|------|-------|-------------|-------------------|
| Slot parser | Rule engine | No | No |
| Literature RAG | Embeddings + retrieval | No | No |
| Claim lookup | SQL/Parquet | No | No |
| `verify_claim.py` | Python + pyreadstat | No | **Yes** |
| Result reader | JSON parser | No | No |
| LLM narrator | Generative model | Yes (prose only) | **No** |

---

## Section 5 — “What Not to Do” Manifesto

> [!WARNING]
> **Prohibition 1 — Fine-tuning an LLM on 59 GB raw microdata or letting an LLM generate statistics**
>
> Generative models optimize for plausible text, not correct jackknife variance, Rubin pooling, or replicate-weight structures. Embedding microrecords in model weights destroys auditability required for thesis defense. Plausible Values encode imputation uncertainty; collapsing $\bar{U}$ and $B$ into a single LLM-generated coefficient is psychometrically invalid equivalent to ignoring imputation variance (Rubin, 1987). **Approved path:** LLM narrates evidence bundles; all numbers originate from `verify_claim.py` with hashed manifests.

> [!WARNING]
> **Prohibition 2 — Injecting raw microdata rows into CLEAN Excel**
>
> ~59 GB spans millions of student records; Excel’s row limit ($\approx 1.05 \times 10^6$) is insufficient by orders of magnitude. CLEAN Excel is a **control plane**, not a **data plane**. Merging microdata destroys relational integrity with Sheets 1–4 and makes git-based reproducibility impossible. **Approved path:** Layer 1 catalog references (`file_id`, `variable_id`); bounded loads at verification runtime only.

> [!WARNING]
> **Prohibition 3 — Single global prediction model across ISAs and domains**
>
> PISA mathematics proficiency, TIMSS content domains, PIRLS reading purposes, and ICCS civic knowledge are not interchangeable latent scales. Sampling designs, weight structures, and PV counts differ by ISA and cycle. A unified model confounds cycle effects with construct drift and violates measurement invariance assumptions without explicit equating. **Approved path:** Domain × ISA × cycle pipelines documented in Sheet 7 with mandatory `measurement_invariance_note`.

> [!WARNING]
> **Additional boundary — LLM variable resolution at verification time**
>
> Allowing a generative model to select SPSS variable names during replication invites wrong-variable confirmation bias. **Approved path:** Layer 2 registry + Layer 1 catalog cross-check with logged `bridge_provenance`.

> [!WARNING]
> **Additional boundary — Single-PV correlation without documentation**
>
> Using one PV draw without Rubin-adjusted variance produces anti-conservative inference. **Approved path:** Default engine policy `pv_rule = rubin_rules` unless `pv_draw_used_by_paper` is explicitly documented in Sheet 5.

---

## Implementation Gate

| Phase | Deliverable | Prerequisite |
|-------|-------------|--------------|
| **0 (current)** | This specification + Excel Sheets 5–7 schema + registry template | Committee approval ✓ |
| **1** | `build_microdata_catalog.py` metadata scan | Approved registry pilot entries |
| **1** | `verify_claim.py` pilot run (PISA 2012 TUR) | Catalog populated for CY6 student file |
| **2** | Feature store + domain predictors | Sheet 7 inventory prioritized |

**Formal statement for methodology appendix:**

$$\text{Literature Claim} \xrightarrow{\text{Layer 2}} \text{Official Variables} \xrightarrow{\text{Layer 1}} \text{Microdata Location} \xrightarrow{\text{Layer 3}} \text{Replicated Statistic} \xrightarrow{\text{Sheet 6}} \text{Auditable Evidence}$$

---

## References

- Rubin, D. B. (1987). *Multiple Imputation for Nonresponse in Surveys*. Wiley.  
- OECD (2014). *PISA 2012 Technical Report*. OECD Publishing.  
- OECD. *PISA Data Analysis Manual* — Plausible Values and survey weights.

---

*Document maintained in repository root: `docs/phase0_methodological_specification.md`*  
*Last updated: 2026-05-31*
