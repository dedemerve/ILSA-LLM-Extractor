# Phase 3.11 — ILSA Algorithm Frequency Audit

**Source:** `analysis/02_prediction_taxonomy/study_classification.csv`  
**Records:** 1266 unique study records  
**Source SHA-256:** `a0e0fac5b7cb80b2ab7a9727b1da183bb1712234cf34d66ef38442eba8914402`  
**Output table:** `analysis/03_data_feasibility/audit/ilsa_ml_method_frequency_v1.csv`

> **Scope update (2026-10-15):** The counts below are still valid for the frozen 1,266-record classification file and for general predictive studies. They are not frequencies among genuine future-cycle forecasts. A separately verified 2026 PISA 2022 country-mean trend forecast (linear/quadratic regression; DOI 10.3389/feduc.2026.1734988) is absent from the file. Keep the locked counts unchanged pending recovery of the original search criteria; update corpus membership and frequency denominators only in a separately versioned release if the paper meets those criteria. Overall literature coverage is unresolved. Details: `phase3_14_cross_cycle_review_and_scope_correction.md`.

## Primary algorithm field

`ml_primary` contains a coded primary method for 167/1,266 records. Among 166 studies classified as predictive, 147 have a coded primary method. In this predictive, algorithm-coded subset the exact primary labels are:

| Primary method | Studies | Share of 147 coded predictive studies | Share of all 166 predictive studies |
|---|---:|---:|---:|
| Random Forest | 42 | 28.6% | 25.3% |
| XGBoost | 17 | 11.6% | 10.2% |
| Neural Network | 16 | 10.9% | 9.6% |
| SVM | 14 | 9.5% | 8.4% |

RF and XGBoost are the two most frequent exact `ml_primary` labels among predictive records. Counts over all records with a primary algorithm are 44 RF and 18 XGBoost (out of 167 coded records). These are literature-motivated general predictive baselines, not evidence that they dominate genuine future-cycle ILSA forecasting.

## Country-level scope check

For primary task code D (country-level prediction), the algorithm-coded subset contains 17 records and its leading primary methods are: SVM 4, Random Forest 3, and XGBoost 2. In that narrower scope, RF and XGBoost are not the two most frequent methods. The original 28 cross-cycle flags have since been source-aware reviewed; the independent external candidate confirms that this coded corpus is not demonstrably exhaustive. Therefore no algorithm frequency ranking can be claimed specifically for genuine ILSA forecasting from this file.

## Any-method field and coding limits

The companion `ml_all_techniques` table counts each technique at most once per study and normalizes only transparent Random Forest/XGBoost label variants. It is an any-method count, not the primary-method count; multi-method studies contribute to several method totals. The exact raw primary field remains unaltered. The table preserves the two fields as separate analyses so these denominators are not mixed.

The study labels are the output of the existing automated classification pipeline. This audit verifies frequencies in that coded file; it does not independently re-read source documents or establish inter-rater reliability. Before manuscript claims, review ambiguous algorithms and the 28 unresolved cross-cycle studies at source.

## Machine-readable frequency table

The CSV contains every method count for four primary-method scopes and four any-technique scopes, including the denominator, study share, and counting rule. The temporal-status codes in the input remain:

| Temporal-forecasting status | Study count |
|---|---:|
| not_applicable | 545 |
| no_temporal_prediction | 538 |
| within_cycle_prediction | 155 |
| cross_cycle_prediction_not_fully_verified | 28 |
