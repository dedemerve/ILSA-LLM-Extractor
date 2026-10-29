# Pre-Fine-Tuning Repository Inventory

**Date:** 2026-10-05  
**Branch:** experiment/ilsa-domain-finetuning  
**Baseline commit:** a082e16 (fix/stage5-dual-source-build-xy)  
**Experiment commit:** 40de9da  

---

## 1. Branch and Commit Verification

```
Branch:  experiment/ilsa-domain-finetuning
Commits: 40de9da (audit artifacts)
         a082e16 (frozen baseline — docs: provenance audit)
         36e2f21 (frozen baseline — fix: ESCS contamination + unified panel)
```

All Stage 5 outputs were produced on or before commit a082e16 and are treated as immutable.

---

## 2. Frozen Baseline Artifacts

### 2.1 Extraction schema (FROZEN — do not modify)

| File | Role |
|------|------|
| `src/schemas/models.py` | Pydantic v2 schema: ILSAArticleMetadata, DataBlock, StructuredFinding, Confounder |
| `src/schemas/findings_validation.py` | Validation helpers, coercion functions |
| `src/extractors/gpt_extractor.py` | OpenAI API extraction driver |
| `src/extractors/pdf_processor.py` | PDF → raw text via PyMuPDF |
| `ilsa_pipeline/extractors/gpt_extractor.py` | Pipeline copy of extractor |

### 2.2 Evidence construction (FROZEN — do not modify)

| File | Role |
|------|------|
| `scripts/build_evidence_matrix_v2.py` | Core evidence matrix builder |
| `scripts/expand_evidence_matrix.py` | Expander + RAW_TO_CANONICAL vocabulary |
| `scripts/build_canonical_taxonomy.py` | Canonical construct taxonomy builder |

The `RAW_TO_CANONICAL` dict in `expand_evidence_matrix.py` is the frozen canonical vocabulary.

### 2.3 Weight construction (FROZEN — do not modify)

| File | Role |
|------|------|
| `scripts/compute_lit_weight_sensitivity.py` | W_j / W_{j,p,d} LOSO stability |

W formula: `w_norm_v3 = (lit_component + w_norm_v2) / 2`  
where `lit_component = log(1+freq) / max_log_freq` (frequency)  
and `w_norm_v2` = program × domain diversity (normalized).

### 2.4 Forecasting (FROZEN — do not modify)

| File | Role |
|------|------|
| `scripts/compare_m0_m1_m1pd.py` | M0/M1/M1_pd/M2 LOCO comparison |
| `scripts/validate_m1pd.py` | Paired permutation test H1 |
| `scripts/build_unified_panel.py` | Country panel construction |
| `scripts/compute_permutation_tests.py` | Permutation tests |

### 2.5 Stage 5 output artifacts (FROZEN — read-only)

| File | Role | Status |
|------|------|--------|
| `outputs/stage5/evidence_matrix_full_v2.csv` | 6,799 evidence rows | FROZEN |
| `outputs/stage5/predictor_weights_v3.csv` | W_j 19 constructs | FROZEN |
| `outputs/stage5/predictor_weights_v3_pd.csv` | W_{j,p,d} 97 cells | FROZEN |
| `outputs/stage5/canonical_crosswalk.csv` | 38 canonical variable mappings | FROZEN |
| `outputs/stage5/unified_panel.csv` | Country temporal panel | FROZEN |
| `outputs/stage5/final_model_comparison.csv` | MAE results M0/M1/M1_pd/M2 | FROZEN |
| `outputs/stage5/provenance_audit.md` | Authoritative provenance | FROZEN |
| `outputs/stage5/weight_loso_stability.csv` | LOSO stability 32 groups | FROZEN |
| `outputs/stage5/corpus_provenance_audit.csv` | Source DB record counts | FROZEN |
| `outputs/stage5/evidence_provenance_audit.csv` | Construct evidence summary | FROZEN |

---

## 3. Raw JSON Corpora (FROZEN — read-only inputs)

| Source | N JSON | Role |
|--------|-------:|------|
| `outputs/ilsa_survey_articles/json/` | 142 (132 unique) | Targeted ILSA ML survey corpus |
| `outputs/Web of Science/` | 302 | WoS extraction outputs |
| `outputs/Scopus/` | 423 | Scopus extraction outputs |
| `outputs/OECD/` | 591 | OECD report extraction outputs |
| `outputs/reports/json/` | 5 | OECD/IEA official report JSONs |

**None of these files may be modified.** Fine-tuning reads from them but writes only to `outputs/finetuning/`.

---

## 4. Source Text Availability Assessment

| Location | Files | Role |
|----------|------:|------|
| `data/*.pdf` | 10 | Source PDFs (articles 1–10 of ilsa_survey) |
| `outputs/finetuning/source_recovery/texts/*.txt` | 10 | Extracted text (pdftotext, ~912 KB total) |

**10 of 132 ilsa_survey JSONs have source text.** 122 are missing. No source text exists for WoS/Scopus/OECD JSONs.

---

## 5. Fine-Tuning Experiment Namespace (ADDITIVE — safe to modify)

All new experimental artifacts reside under `outputs/finetuning/` and `scripts/finetuning/`.

### 5.1 Already created (commit 40de9da + current session)

```
outputs/finetuning/
├── audit/
│   ├── pre_finetuning_baseline_inventory.csv   (commit 40de9da)
│   ├── pre_finetuning_baseline_report.md       (commit 40de9da)
│   ├── gold_standard_quality_report.md         (commit 40de9da)
│   ├── split_integrity.csv                     (commit 40de9da)
│   ├── training_configuration.json             (commit 40de9da)
│   ├── experiment_manifest.json                (commit 40de9da)
│   ├── pre_finetuning_inventory.csv            (this session)
│   └── pre_finetuning_inventory.md             (this session)
├── source_recovery/
│   ├── source_inventory.csv                    (this session)
│   └── texts/[10 .txt files]                   (this session)
└── data/
    ├── normalized_labels.csv                   (this session)
    └── gold_standard.csv                       (this session)

scripts/finetuning/
├── __init__.py                                 (this session)
└── normalize_training_labels.py                (this session)
```

### 5.2 Pending

```
outputs/finetuning/data/
    train.jsonl             — BLOCKED: only 9 silver-verified candidates
    validation.jsonl        — BLOCKED
    test.jsonl              — BLOCKED

outputs/finetuning/baseline/    — pending model execution
outputs/finetuning/fine_tuned/  — pending model training
outputs/finetuning/comparison/  — pending model comparison
outputs/finetuning/FINETUNING_FINAL_REPORT.md — pending completion
```

---

## 6. Frozen Baseline Results (Reference — do not recompute to replace)

| Model | MAE (full panel, 45 folds) | MAE (balanced, 18 folds) |
|-------|---------------------------:|-------------------------:|
| M2 Persistence | 40.19 | 23.98 |
| M0 Ridge | 17.08 | — |
| M1 Global W_j | 17.02 | — |
| **M1_pd W_{j,p,d}** | **16.94** | **12.97** |

H1: mean ΔMAE = +0.077, 95% CI [0.003, 0.151], p = 0.017*, wins/ties/losses = 28/4/13

---

## 7. Silver-Standard Candidate Set (NOT gold-verified)

**Terminology:** The 9 records identified as `silver_verified_candidate` in `outputs/finetuning/data/gold_standard.csv` are:
- source text available (PDF + extracted text)
- mapped to canonical evidence matrix (DOI match)
- pass automated schema checks (Tier-1/2 fields)
- have non-empty `top_predictors`

**They are NOT human-verified.** Human verification has not been performed. They must be called "automatically validated silver-standard candidates" until manual annotation is complete.

| Study | Programs | N findings |
|-------|----------|----------:|
| 1. Lee & Lee (2025) | TIMSS | 1 |
| 10. Aydoğan & Tat (2025) | PISA | 1 |
| 3. Zhou et al. (2024) | PISA | 1 |
| 4. Zheng et al. (2023) | PISA | 1 |
| 5. Song & Cutumisu (2024) | TIMSS | 1 |
| 6. Elouafi (2025) | TIMSS | 1 |
| 7. Alas & Tezer (2024) | PISA | 1 |
| 8. Acıslı-Celik & Yesilkanat (2023) | PISA | 1 |
| 9. Alkan et al. (2025) | PISA | 3 |

**Programs represented:** PISA (7/9), TIMSS (3/9). PIRLS, TALIS, ICILS, ICCS, PIAAC: 0.  
**Critical gap:** Non-PISA programs are absent from the silver-verified set.

---

## 8. Primary Blockers Before Training

1. **Source text gap:** 122/132 source PDFs missing → 9 silver-verified candidates is insufficient for reliable fine-tuning
2. **PISA dominance:** All candidates are PISA or TIMSS only — no PIRLS/TALIS/ICILS/ICCS/PIAAC
3. **No human annotation:** Zero records are human-verified; all labels are LLM-generated
4. **Schema drift normalization:** research_design_type values require human judgment for 100% of records
5. **Compute/API access:** Base model for fine-tuning not yet selected

---

*Generated: 2026-10-05 | Branch: experiment/ilsa-domain-finetuning*
