# Gold Sampling Coverage Analysis

**Date:** 2026-10-05

## Overview

| Level | Total | PISA | TALIS | PIAAC | TIMSS | PIRLS | ICCS | ICILS |
|-------|-------|------|-------|-------|-------|-------|------|-------|
| Full corpus | 1,266 | 358 | 69 | 59 | 56 | 31 | 15 | 18 |
| Source-text available | 577 | 337 | 69 | 55 | 55 | 29 | 14 | 17 |
| Silver verified 526 | 526 | 183 | 65 | 49 | 23 | 13 | 8 | 9 |
| Gold 12 | 12 | 3 | 3 | 0 | 1 | 2 | 1 | 2 |

## Key Findings

- **PIAAC gap**: 0/12 gold studies represent PIAAC despite 59 corpus records (4.7%). Critical priority.
- **PISA overrepresentation**: 3/12 gold studies (25%) vs 28% corpus share — roughly proportional.
- **TALIS**: 3/12 gold (25%) vs 5.5% corpus — intentionally overrepresented for fine-tuning diversity.
- **ICCS, ICILS**: 1 and 2 gold studies respectively — small but present; need more coverage.
- **TIMSS, PIRLS**: 1 and 2 gold studies; underrepresented relative to corpus share.

## Recommendation

Next annotation batch should prioritize:
1. PIAAC journal articles (0 gold; 49 silver_verified candidates)
2. TIMSS empirical papers (1 gold; 19+ silver candidates)  
3. PIRLS empirical papers (2 gold; 13+ silver candidates)
