# PIAAC Gold Eligibility Summary

**Date:** 2026-10-05
**Branch:** experiment/ilsa-domain-finetuning

## Overview

| Metric | Count |
|--------|-------|
| Total PIAAC records in silver pool | 52 |
| Journal articles | 20 |
| Reports | 32 |
| Empirical papers (journal/conference) | 20 |
| Descriptive reports | 32 |
| Source text available | 52 |
| Eligible gold candidates (primary_candidate) | 18 |
| Excluded candidates | 34 |

## Exclusion Reasons

- **Descriptive OECD reports** (32): PIAAC country notes and summary reports lack empirical ML analysis. Excluded per v1.0 codebook rule (OECD country notes → excluded).
- **Missing source text** (0): Cannot annotate without source text.
- **No evidence findings** (2): Empirical papers with no extractable ML findings.

## Note

PIAAC is currently unrepresented in the gold set (0 out of 12 studies). Prioritizing PIAAC journal articles for the next annotation batch is critical for program coverage.
