# ILSA Domain Fine-Tuning Experiment — Final Report

**Status: PHASE 1 COMPLETE — Phases 2–4 Pending**  
**Date:** 2026-10-05  
**Branch:** experiment/ilsa-domain-finetuning  
**Baseline commit:** a082e16 | Experiment commit: see git log  

---

## 1. Executive Summary

This report covers Phase 1 (repository audit + source recovery) of the controlled domain-specific fine-tuning experiment for the ILSA evidence-extraction LLM layer. Phases 2–4 (annotation, training, evaluation) are blocked pending source text availability.

**Current state:**
- Frozen baseline documented and verified
- 10 source texts recovered (of 132 candidates)
- 9 automatically validated silver-standard candidates identified
- Schema drift characterized and normalization implemented
- All integrity checks pass

**Primary blocker:** Only 10/132 source PDFs are present in `data/`. Supervised fine-tuning requires source text + structured label pairs. With N=9 viable pairs, no standard fine-tuning method is feasible without additional PDF acquisition.

---

## 2. Frozen Baseline

The following numbers are the authoritative thesis reference and must not be altered.

| Model | MAE (full, 45 folds) | MAE (balanced, 18 folds) |
|-------|---------------------:|-------------------------:|
| M2 Persistence | 40.19 | 23.98 |
| M0 Ridge (no lit) | 17.08 | — |
| M1 Global W_j | 17.02 | — |
| **M1_pd W_{j,p,d}** | **16.94** | **12.97** |

H1 test: mean ΔMAE = +0.077, 95% CI [0.003, 0.151], p=0.017*, wins/ties/losses = 28/4/13

Literature prior formula: `w_norm_v3 = (lit_component + w_norm_v2) / 2`  
W_j stability: LOSO CV < 25% for 32/32 groups, mean CV ≈ 6.6%  
External stability W_v2 vs W_v3: Pearson r = 0.9933

---

## 3. Corpus Provenance

Authoritative provenance (from `outputs/stage5/provenance_audit.md`):

```
N_json_total         = 1,458
N_ilsa_relevant      = 1,349
N_evidence_bearing   = 1,144
N_unique_any         = 820
N_canonical_mapped   = 620   ← enters W_{j,p,d}
```

Note: 577 is a historical intermediate (WoS+Scopus pre-OECD) — not the final count.

---

## 4. Source Recovery

**Method:** Filename-prefix matching (normalized Unicode) between `data/*.pdf` and `outputs/ilsa_survey_articles/json/*.json`, followed by pdftotext extraction.

| Status | N |
|--------|--:|
| Recovered (PDF + text extracted) | 10 |
| Missing (no PDF in repository) | 122 |
| Invalid | 0 |
| **Total** | **132** |

Source texts stored in `outputs/finetuning/source_recovery/texts/` (total ~912 KB, 10 files).

**Expanded search:** WoS/Scopus/OECD JSON files do not contain raw text and no source PDFs were found for these sources in the repository.

---

## 5. Silver-to-Gold Construction

### Terminology

| Label | Meaning |
|-------|---------|
| `gold_verified` | Human-annotated against source text — NOT YET PERFORMED |
| `silver_verified_candidate` | Source available + canonical match + auto-clean Tier-1/2 |
| `silver_only` | Canonical match but source unavailable or schema issue |
| `excluded` | No findings or not in canonical evidence |

### Distribution (N=132)

| Status | N |
|--------|--:|
| silver_verified_candidate | 9 |
| silver_only | 103 |
| excluded | 20 |
| **gold_verified** | **0** |

---

## 6. Annotation Quality

**Human annotation status: NOT YET CONDUCTED.**

A protocol has been established in `outputs/finetuning/audit/human_annotation_reliability.md` covering:
- Field-level codebook for all Tier-1/2/3 fields
- Sampling strategy (stratified by program, design, predictor category)
- Single-annotator intra-rater reliability protocol (washout ≥ 7 days)
- Adjudication rules for ambiguous cases

No κ statistics can be reported until annotation is performed.

---

## 7. Train/Validation/Test Split

**Status: BLOCKED.**

With N=9 silver-verified candidates, no 70/15/15 split is viable. Minimum requirement: ~50–100 source-text+label pairs. Split will be constructed after source recovery is completed.

Design (when applicable):
- Split at article/DOI level
- No DOI overlap across splits
- Group-stratified to prevent PISA dominance in test set
- Files: `train.jsonl`, `validation.jsonl`, `test.jsonl`

---

## 8. Fine-Tuning Configuration

**Status: PENDING — awaiting source text expansion.**

Template recorded in `outputs/finetuning/audit/training_configuration.json`.

Key decisions pending:
- Base model selection (OpenAI fine-tuning API vs open-source LoRA/QLoRA)
- Minimum viable training set size
- Whether to proceed with 9 examples as few-shot evaluation only

---

## 9–10. Baseline / Fine-Tuned Extraction Performance

**Status: NOT YET RUN.** Requires held-out test set and (for fine-tuned) trained model.

Metrics to be reported per Section 21 of the protocol:
- Field-level exact match, precision, recall, F1 for all Tier-1 fields
- JSON schema validity rate
- Canonical mapping accuracy
- Unsupported inference rate
- Source-grounding rates

---

## 11. Source-Grounding Performance

**Status: NOT YET RUN.**

For each extracted claim: directly supported / valid normalization / unsupported / contradicted.

---

## 12. Canonical Mapping Performance

**Status: NOT YET RUN.**

Attention items from protocol: ESCS→SES_COMPOSITE, BSDGEDUP→PARENTAL_EDUCATION (inverted), ITSEX≠GENDER_GAP, ICT variable disambiguation.

---

## 13–14. W_j and W_{j,p,d} Stability

**Status: NOT YET RUN.** Requires fine-tuned evidence matrix.

Baseline stability for reference:
- LOSO: 32/32 groups CV < 25%, mean CV = 6.6%
- External: Pearson r = 0.9933 (W_v2 vs W_v3)

---

## 15. LOCO Stability

**Status: NOT YET RUN.**

---

## 16. Forecasting Comparison

**Status: NOT YET RUN.**

Models to compare:
- M2, M0, M1_global_baseline, M1_pd_baseline (frozen results above)
- M1_global_FT, M1_pd_FT (pending fine-tuned evidence)

---

## 17–18. Balanced Panel / Ablation

**Status: NOT YET RUN.**

---

## 19. Error Analysis

**Status: NOT YET RUN.**

Error taxonomy defined in protocol (Section 35): wrong program, cycle, target, predictor, canonical mapping, direction, effect type, unsupported inference, missing evidence, hallucinated methodology, PV/weight misclassification.

---

## 20. Cross-Program Generalization

**Status: NOT ESTIMABLE.**

Silver-verified candidates: PISA = 77.8%, TIMSS = 33.3%, all other programs = 0%.  
Cross-program generalization cannot be assessed from current source text corpus.

---

## 21. Limitations

1. **Source text availability:** 122/132 PDF sources missing. Fundamental bottleneck for supervised fine-tuning.
2. **PISA dominance:** Any fine-tuned model trained on current corpus will be heavily PISA-biased. Cross-program claims require additional data.
3. **No human annotation:** All 9 silver candidates are LLM-generated labels without human verification.
4. **research_design_type drift:** All 132 records have non-standard research_design_type values. The field requires systematic normalization with human judgment.
5. **Single-program source constraint:** No PIRLS, TALIS, ICILS, ICCS, or PIAAC source texts available.
6. **N=9 is insufficient:** Standard SFT requires ≥50 examples for meaningful generalization.

---

## 22. Scientific Interpretation

At this stage, no empirical claims about fine-tuning can be made.

The four required claims (protocol Section 44) each require data not yet available:
- **CLAIM 1** (extraction improvement): requires held-out test evaluation
- **CLAIM 2** (prior quality improvement): requires fine-tuned evidence matrix
- **CLAIM 3** (forecasting improvement): requires full downstream evaluation
- **CLAIM 4** (cross-program generalization): requires non-PISA source texts

---

## 23. Decision on Fine-Tuning Adoption

**DEFERRED — insufficient evidence.**

Current verdict: fine-tuning experiment cannot proceed to training phase without resolving the source text bottleneck.

Recommended immediate action: Contact original authors (articles 11–132) or access institutional repositories to obtain PDFs for ≥40 additional studies, prioritizing non-PISA programs.

---

## 24. Reproducibility Information

| Artifact | Value |
|----------|-------|
| Git branch | experiment/ilsa-domain-finetuning |
| Baseline commit | a082e16 |
| Audit commits | 40de9da, [current] |
| Corpus version | v1.0-2026-10-05 |
| Source text version | 10-pdfs-pdftotext-2026-10-05 |
| Schema version | src/schemas/models.py @ a082e16 |
| Canonical vocabulary | expand_evidence_matrix.py RAW_TO_CANONICAL @ a082e16 |
| W formula version | predictor_weights_v3.csv w_norm_v3 @ 36e2f21 |
| Gold standard version | v2-2026-10-05 (silver_verified_candidate only) |
| Normalization script | scripts/finetuning/normalize_training_labels.py |
| Normalization output | outputs/finetuning/data/normalized_labels.csv |
| Source inventory | outputs/finetuning/source_recovery/source_inventory.csv |
| Integrity check | outputs/finetuning/audit/final_integrity_check.csv |

---

*This document will be updated as each phase completes.*
