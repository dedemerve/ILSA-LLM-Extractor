# Program Coverage Profile

**Date:** 2026-10-05  
**Phase 3 corpus analysis**

---

## 1. Full Corpus vs Source-Text vs Silver vs Gold

| Program | Full Corpus (1,266) | Source Text (577) | Silver Verified (341) | Gold Annotated (12) |
|---------|-------------------:|------------------:|----------------------:|--------------------:|
| PISA | 358 (35.9%) | 337 (58.4%) | 183 (53.7%) | 3 (25.0%) |
| TALIS | 69 (6.9%) | 69 (12.0%) | 65 (19.1%) | 3 (25.0%) |
| PIAAC | 59 (5.9%) | 55 (9.5%) | 49 (14.4%) | 0 (0.0%) |
| TIMSS | 56 (5.6%) | 55 (9.5%) | 23 (6.7%) | 1 (8.3%) |
| PIRLS | 31 (3.1%) | 29 (5.0%) | 13 (3.8%) | 2 (16.7%) |
| ICCS | 15 (1.5%) | 14 (2.4%) | 8 (2.3%) | 1 (8.3%) |
| ICILS | 18 (1.8%) | 17 (2.9%) | 9 (2.6%) | 2 (16.7%) |
| PIAAC | 59 | 55 | 49 | 0 |

Note: full corpus has 995 non-UNKNOWN DOI records in evidence matrix; 271 have UNKNOWN DOI (mostly IEA reports without DOI).

---

## 2. Key Observations

### PISA overrepresentation
- Full corpus: 36% → Source text: 58% → Silver: 54% → Gold pilot: 25%
- PISA studies are disproportionately ML prediction papers and journals
- Gold construction actively reduced PISA to 25% (deliberate design)

### PIAAC collapse
- 59 PIAAC records in full corpus → 49 silver candidates → 0 gold verified
- Pilot revealed most OECD/PIAAC records are DESCRIPTIVE country notes, NOT ML prediction studies
- The canonical mapping "See outcome_summary (descriptive associations not coded as predictors)" → ICT_HOME_ACCESS is a systematic LLM heuristic artifact
- **Critical finding:** PIAAC silver candidates are largely NOT fine-tuning-viable for ML extraction

### TALIS adequate
- 65 silver candidates → 3 gold verified in pilot
- TALIS records are ML/prediction studies but have systematic canonical mapping issue: teacher-level ICT constructs mapped to ICT_HOME_ACCESS (student-level construct)

### TIMSS/PIRLS/ICCS/ICILS underrepresented
- These programs need priority acquisition of additional source texts
- Pilot confirmed they are genuine ML prediction studies suitable for fine-tuning

---

## 3. Revised Program Targets for Gold Set

Based on pilot findings, revised gold targets:

| Program | Pilot verified | Recommended gold target | Notes |
|---------|---------------:|------------------------:|-------|
| PISA | 3/5 verified, 0 excluded | 15–20 | Abundant but need quality filter |
| TALIS | 3/3 verified | 10–12 | ICT canonical issue; verify each |
| TIMSS | 1/3 (2 excluded as macro/data papers) | 8–10 | Filter to individual ML studies |
| PIRLS | 2/2 verified | 6–8 | Genuinely scarce; use manual review set |
| ICILS | 2/2 verified | 6–8 | Good quality |
| ICCS | 1/2 (1 ambiguous) | 4–6 | Check for measurement invariance studies |
| PIAAC | 0/3 (all excluded as descriptive) | 2–4 | Focus on PIAAC journal articles, not OECD reports |

Total recommended gold: **51–68 studies** (justified by program coverage requirements)

---

## 4. PIAAC Recovery Strategy

The 49 PIAAC silver candidates are mostly OECD publications. A scan of their publication types:
- Most are OECD working papers or country notes → descriptive, not ML
- **Action required:** Filter PIAAC silver candidates to journal articles only (likely ~5–10 records remain)

---

*Generated: 2026-10-05 | Phase 3 | Branch: experiment/ilsa-domain-finetuning*
