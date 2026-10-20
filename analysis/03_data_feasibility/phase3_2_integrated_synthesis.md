# Phase 3.2 — Integrated Synthesis (Phase G)

**Date:** 2026-10-14  
**Protocol:** Phase 3.2 Full Specification, Phase G  
**Status:** COMPLETE  
**This file does NOT overwrite any prior artifact.**

---

## G.1 Synthesis Table

| Research question | Literature evidence | Experimental result | Integration | Remaining uncertainty |
|-------------------|--------------------|--------------------|-------------|----------------------|
| What types of studies dominate the corpus? | 1,100/1,266 (87%) non-predictive; 166 (13%) predictive; 17 (1.3%) country-level prediction | N/A | The literature is dominated by explanatory analyses of student-level factors | Low uncertainty |
| How many studies genuinely forecast future ILSA cycles? | 0 confirmed; 28 ambiguous | N/A | No external benchmark exists for country-level ILSA future-cycle forecasting | 28 studies unresolved |
| Are performance metrics comparable across studies? | No — incompatible metrics, targets, units | N/A | Quantitative meta-analysis not justified | High heterogeneity |
| Do simple baselines beat complex models for short country-level series? | No direct evidence; general forecasting theory supports this | Persistence (14.48) beats TimesFM (15.83) and linear trend (16.98) in 2022 | Literature gap confirmed by experiment | Single year, single program |
| Does context length predict error? | No direct evidence | r(n_context, AE_TFM) = −0.114, ns | H1 not supported | Low power (n=60, range=3–6) |
| Does historical trend slope predict TFM error? | No direct evidence | r=+0.496, p<0.001 | COVID-19 trend disruption is a key confound; literature provides no prior | Observational only |
| Is positive prediction bias expected? | No direct evidence | Mean signed error ≈ MAE for both methods (nearly all overpredictions) | Consistent with COVID learning loss — structurally unpredictable | Cannot disentangle COVID from general trend regression |
| Is PISA 2025 ready for final evaluation? | N/A | CONDITIONALLY READY (Phase 3.1.1) | SE validation confirms pipeline; direct mean comparison unresolved | OECD Results Volume I inaccessible |

---

## G.2 Evidence Chains

### Chain 1: State of the literature

```
1,266 corpus records
  → Only 166 predictive (13%)
  → Only 17 country-level prediction (1.3%)
  → 0 confirmed genuine future-cycle forecasting
  → 28 ambiguous (require source inspection)
  → FINDING: The 2022 PISA experiment fills a genuine methodological gap in the literature.
    No external performance benchmark can be drawn from the corpus.
```

### Chain 2: Performance ranking on PISA 2022

```
PISA math 2022, n=60 countries, ≥3 pre-2022 cycles:
  Persistence MAE=14.48 < TimesFM MAE=15.83 < Linear trend MAE=16.98
  → Both TFM and linear trend underperform a naive persistence baseline
  → TimesFM outperforms only for n_context=6 (n=36), not for shorter series
  → FINDING: Persistence is currently the best single-method baseline.
    TimesFM has no aggregate advantage in the 2022 development sample.
```

### Chain 3: COVID-19 confound

```
2022 PISA math:
  All 60 countries over-predicted (positive bias = MAE for both methods)
  Trend slope correlated with TFM error (r=0.50, p<0.001)
  Albania: largest error (68–93 pts); 2022 anomaly confirmed by 2025 recovery
  → FINDING: The 2022 evaluation is heavily influenced by COVID-19 learning loss.
    This is a structurally unpredictable shock that inflates error for all methods.
    Conclusions about model capability are confounded by the COVID disruption.
```

### Chain 4: PISA 2025 readiness

```
Phase 3.1.1 validation:
  SE pipeline matches Annex 14 official values (MAD=0.000026, n=79)
  Direct mean comparison: BLOCKED (OECD Results Volume HTTP 403)
  Albania: official quality flag confirmed (Annex 14 footnote 1)
  → FINDING: PISA 2025 is CONDITIONALLY READY as held-out test.
    Quality flag countries (ALB, CAN, NLD, NZL, NOR, USA) should be reported separately.
```

---

## G.3 Answers to the Required Research Questions

**1. What kinds of predictive tasks dominate the corpus?**  
Student-level within-cycle prediction (code A, n=96) is the most common predictive task. Country-level prediction (code D, n=17) is marginal. 87% of all studies are non-predictive. [VERIFIED]

**2. How many studies genuinely forecast future ILSA cycles?**  
**0 confirmed.** 28 studies are ambiguous and require source inspection. The closest candidate (Braun et al. 2025) was confirmed as retrospective at source. [VERIFIED]

**3. How many studies provide comparable predictive-performance evidence?**  
**0 directly comparable** to the country-level PISA future-cycle experiment. The 166 predictive studies use different metrics, units, and targets. Quantitative synthesis is not justified. [VERIFIED]

**4. Which methodological differences are documented in the corpus?**  
Documented: model family, prediction target level, ILSA program, temporal design classification, sampling weight usage. Not coded: train/test split design, baseline comparison, leakage risk, numeric performance values. [VERIFIED / UNRESOLVED for uncoded fields]

**5. Which proposed explanations for performance variation are testable?**  
Testable now: H3 (simple baselines competitive), H4 (trend slope → error), H5 (positive bias), H6 (context-length stratification). Not testable: H2 (random split inflation), H7 (split design comparison). [TESTED for H3/H4/H5/H6]

**6. What does the literature imply for country-level forecasting?**  
The literature provides **no direct evidence** for country-level ILSA future-cycle forecasting performance. The gap is the motivation for this project. The general forecasting literature suggests simple baselines are hard to beat for short, noisy series. [INFERRED from gap]

**7. What does the existing TimesFM experiment actually establish?**  
On a single development sample (PISA 2022 math, n=60, pre-2022 context), persistence outperforms TimesFM zero-shot by ~1.35 MAE points. Both methods show systematic positive bias from COVID-19 disruption. TimesFM has advantages only when 6 historical observations are available. The experiment is a development/diagnostic result — not the final evaluation. [VERIFIED by direct reproduction]

**8. What explains the difference between TimesFM and persistence?**  
Partially explained: (a) COVID-19 disruption caused a trend reversal that TimesFM — which extrapolates patterns — could not anticipate, while persistence anchors to the last value and is thereby less sensitive to trend extrapolation errors. Evidence: r(trend_slope, AE_TFM)=0.50, p<0.001; r(trend_slope, AE_Persistence)=0.12, ns. (b) Short context series (3–5 observations) appear insufficient for TimesFM to identify useful patterns. What is not established: the underlying mechanism in TimesFM's architecture; whether a different model configuration would perform differently. [PARTIALLY EXPLAINED]

**9. What additional evidence is needed?**  
- PISA 2025 evaluation (held-out test; not yet evaluated)
- Pre-COVID evaluation year (e.g., PISA 2018 as target, using 2015 context) to separate COVID confound from model capability
- TIMSS and PIRLS evaluation to assess generalization
- Direct mean comparison against OECD Results Volume I for PISA 2025 (currently blocked)
- Source inspection of 28 ambiguous cross-cycle studies

**10. Is PISA 2025 ready for a defensible final evaluation?**  
**CONDITIONALLY READY.** SE validation passes. Direct mean comparison blocked. Albania has official quality flag. Path fix needed before re-run. [CONDITIONALLY VERIFIED]

**11. What should be done next, and in what order?**  
Priority order:
1. **Fix PISA 2025 SAV path in production script** (B-C), verify estimates unchanged after re-run.
2. **Evaluate PISA 2025** as held-out test using identical protocol — after confirming no further use of 2025 outcomes has occurred.
3. **Run pre-COVID baseline test** (PISA 2018 as target) to disentangle COVID confound from model capability.
4. **Source-inspect the 28 ambiguous cross-cycle studies** to update the "0 genuine forecasting studies" claim.
5. **Extend to TIMSS and PIRLS** to test generalization of persistence-beats-TFM finding.

---

## G.4 What Is Not Known

| Unknown | Why unresolved |
|---------|---------------|
| True count of genuine future-cycle forecasting studies | 28 unresolved; require source access |
| Numeric performance benchmarks from literature | Not coded in free-text field |
| TimesFM performance on PISA 2025 | Not yet evaluated (held-out) |
| Whether persistence advantage persists in non-COVID years | 2018 evaluation not run |
| Why TimesFM underperforms with n_context=3–4 | Architectural explanation not available |
| Official PISA 2025 mean scores | OECD Results Volume I inaccessible (HTTP 403) |

---

## G.5 Integrity Confirmations

- **No existing corpus or gold-standard artifact was modified.** ✓
- **No train/validation/test split was changed.** ✓
- **No model was trained or fine-tuned.** ✓
- **No existing experiment output was overwritten.** ✓
- **No commit or push was made.** ✓
- **Phase 3.1.1 addendum kept separate.** ✓
