# Annotation Codebook Changelog

---

## v1.0 — 2026-10-05

**Status:** Initial release  
**Based on:** Pilot annotation of 18 records across 7 programs  

### New rules established

| Rule | Affected fields | Reason |
|------|----------------|---------|
| ICT_HOME_ACCESS does NOT apply to teacher-level ICT in TALIS | canonical_construct | Pilot: 10+15959+RPGE+EN.pdf, Patterns..., ICILS teachers |
| direction=0 in evidence matrix = not_reported unless source confirms null | direction_canonical | Pilot: systematic ambiguity in direction encoding |
| OECD country notes → excluded (descriptive reports) | gold_status | Pilot: 09034ae3, 5jlpq7tg3hxs, 5kmhds6czqzq all excluded |
| "Primary study outcome (heuristic migration)" = LLM placeholder → must find real outcome | target_variable | Systematic LLM failure mode across multiple studies |
| Measurement invariance studies (CFA/MGCFA) → excluded OR descriptive | gold_status, effect_type | Pilot: Are-life-goals-invariant... ambiguous |
| Macroeconomic aggregate studies → excluded | gold_status | Pilot: Test scores and economic growth excluded |
| Data harmonization papers → excluded | gold_status | Pilot: data-10-00130 excluded |
| rubin_rules → all_pvs (HIGH confidence) — does not block silver | plausible_values_handling | Normalization script validation |
| irt_theta / wle → code as single_pv with flag | plausible_values_handling | Neither is the standard PV procedure |
| direction=not_reported (not +/-/0) when NaN in evidence matrix | direction_canonical | Explicit rule to avoid misinterpretation |

### Fields added to gold schema (not in v0)
- ambiguity_flag (multiple values with | separator)
- adjudication_note
- source_supported_claim

### Fields modified
- effect_type: added "descriptive" as explicit value (was not in original protocol)
- plausible_values_handling: irt_theta and wle guidance added

### Records affected
All 18 pilot records. Future annotations inherit these rules from v1.0.

---

## Planned v1.1 changes (pending)

- Extend canonical construct vocabulary to include UNMATCHED_TEACHER_ICT as a distinct value for TALIS teacher ICT variables
- Add explicit coding rule for PIAAC "literacy/numeracy proficiency" outcomes
- Clarify SELF_REGULATION vs SELF_CONCEPT distinction

---

*Document initialized: 2026-10-05*
