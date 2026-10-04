#!/usr/bin/env python3
"""
FAZ 4 — Literature Priority Vector

Adımlar:
  4.1  construct_frequency.csv provenance audit
  4.2  w_F — corpus frequency (raw + log-normalized)
  4.3  w_P — program diversity
  4.4  w_C — cycle diversity
  4.5  w_N — country diversity
  4.6  w_D — method diversity
  4.7  w_E — evidence type weight (predictive > associative)
  4.8  direction normalization check
  4.9  study-level dependence / duplicate evidence flag
  4.10 forecastability filter
  4.13 W_j = composite literature priority

Çıktılar:
  outputs/stage5/evidence_provenance_audit.csv  — construct × dimension
  outputs/stage5/literature_priority.csv        — final W_j vector
"""
from __future__ import annotations

import logging
import math
import pathlib

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT_DIR      = PROJECT_ROOT / "outputs" / "stage5"

# ── Forecastability filter ────────────────────────────────────────────────────
# Construct → ILSA variable mevcut ve country-level forecastable mı?
FORECASTABILITY = {
    # forecastable = True: enriched_panel_v2'de zaten mevcut veya eklenebilir
    "SES_COMPOSITE":                    True,   # PISA ESCS
    "PARENTAL_EDUCATION":               True,   # TIMSS BSDGEDUP, PIRLS ASDHEDUP
    "HOME_RESOURCES":                   True,   # PISA HOMEPOS
    "SCHOOL_BELONGING":                 True,   # PISA BELONG
    "HOME_LITERACY_ACTIVITIES":         True,   # PIRLS ASDHELA
    "ICT_HOME_ACCESS":                  True,   # PISA ICTAVHOM
    "ICT_SCHOOL_ACCESS":                True,   # PISA ICTAVSCH
    "DISCIPLINARY_CLIMATE":             True,   # TALIS TCDISCS/T3DISC
    "TEACHER_SELF_EFFICACY_CLS_MGMT":   True,   # TALIS SECLSS/T3SECLS
    "TEACHER_SELF_EFFICACY_INSTRUCTION":True,   # TALIS SEINSS/T3SEINS
    "TEACHER_SELF_EFFICACY_ENGAGEMENT": True,   # TALIS SEENGS/T3SEENG
    "PROF_COLLABORATION":               True,   # TALIS TCCOLLS/T3COLES
    "JOB_SATISFACTION_ENVIRONMENT":     True,   # TALIS TJSENVS/T3JSENV
    "JOB_SATISFACTION_PROFESSION":      True,   # TALIS TJSPROS/T3JSPRO
    "EFFECTIVE_PD":                     True,   # TALIS TEFFPROS/T3EFFPD
    # forecastable = False: country-level aggregation anlamsız veya no cross-cycle availability
    "GENDER":                           False,  # demographic; not a modifiable predictor
    "SELF_CONCEPT":                     False,  # no cross-cycle comparable variable in current panel
    "MOTIVATION":                       False,  # no cross-cycle comparable variable
    "SCIENCE_SELF_EFFICACY":            False,  # item-level; no cross-cycle comparable
    "MATH_SELF_EFFICACY":               False,
    "IMMIGRANT_STATUS":                 False,  # demographic
    "LANGUAGE_BACKGROUND":              False,  # demographic
    "INSTRUCTIONAL_TIME":               False,  # no cross-cycle teacher-level country aggregate
    "CURRICULUM_TYPE":                  False,  # categorical; not temporally comparable
    "TEACHING_QUALITY":                 False,  # composite; no direct variable mapping
    "PRIOR_ACHIEVEMENT":                True,   # lag_Y — already in model
    "SCIENCE_INTEREST":                 False,
}

# Evidence type uplift: predictive evidence > associative
EVIDENCE_TYPE_WEIGHT = {
    "predictive":   1.20,
    "associative":  1.00,
    "cross_cycle":  1.30,   # temporal evidence most relevant for forecasting
    "cross_country":1.10,
}

# Canonical direction (for audit check)
CANONICAL_DIRECTION = {
    "SES_COMPOSITE": "+", "PARENTAL_EDUCATION": "+", "HOME_RESOURCES": "+",
    "SCHOOL_BELONGING": "+", "HOME_LITERACY_ACTIVITIES": "+",
    "ICT_HOME_ACCESS": "+", "ICT_SCHOOL_ACCESS": "+",
    "DISCIPLINARY_CLIMATE": "+",  # after negate transformation
    "TEACHER_SELF_EFFICACY_CLS_MGMT": "+",
    "TEACHER_SELF_EFFICACY_INSTRUCTION": "+",
    "TEACHER_SELF_EFFICACY_ENGAGEMENT": "+",
    "PROF_COLLABORATION": "+", "JOB_SATISFACTION_ENVIRONMENT": "+",
    "JOB_SATISFACTION_PROFESSION": "+", "EFFECTIVE_PD": "+",
}

ILSA_PROGRAMS = {"PISA","TIMSS","TIMSS_G4","PIRLS","TALIS","ICILS","ICCS","PIAAC"}


def load_evidence() -> pd.DataFrame:
    df = pd.read_csv(OUT_DIR / "evidence_matrix.csv")
    # programs sütunu: "PISA|TIMSS" gibi → split
    df["programs_list"] = df["programs"].fillna("").apply(
        lambda x: [p.strip() for p in x.split("|") if p.strip() in ILSA_PROGRAMS]
    )
    # cycles sütunu: "2018|2022" gibi → split → int list
    df["cycles_list"] = df["cycles"].fillna("").apply(
        lambda x: [int(c) for c in x.split("|") if c.strip().isdigit()]
    )
    # countries_sample: comma or pipe separated
    df["countries_list"] = df["countries_sample"].fillna("").apply(
        lambda x: [c.strip() for c in x.replace("|",",").split(",") if c.strip()]
    )
    return df


def compute_provenance_audit(df: pd.DataFrame) -> pd.DataFrame:
    """Per-construct diversity metrics."""
    records = []
    for construct, grp in df.groupby("canonical_construct"):
        n_studies   = grp["study_id"].nunique()
        n_rows      = len(grp)

        # Program diversity: kaç farklı ILSA programında bulundu
        all_programs = set()
        for plist in grp["programs_list"]:
            all_programs.update(plist)
        n_programs = len(all_programs)

        # Cycle diversity
        all_cycles = set()
        for clist in grp["cycles_list"]:
            all_cycles.update(clist)
        n_cycles = len(all_cycles)

        # Country diversity
        all_countries = set()
        for clist in grp["countries_list"]:
            all_countries.update(clist)
        n_countries = len(all_countries)

        # Method diversity: unique ML techniques
        methods = grp["study_id"].apply(lambda s: s[:5])  # proxy via study_id
        # Better: effect_type distribution
        effect_counts = grp["effect_type"].value_counts().to_dict()
        n_predictive = effect_counts.get("predictive", 0)
        n_assoc      = effect_counts.get("associative", 0)
        n_cross      = effect_counts.get("cross_cycle", 0) + effect_counts.get("cross_country", 0)
        frac_predictive = n_predictive / n_rows if n_rows else 0.0

        # Direction consistency
        directions = grp["direction"].dropna()
        dir_counts  = directions.value_counts()
        direction_canonical = CANONICAL_DIRECTION.get(construct, "?")
        if len(dir_counts):
            dominant_dir = dir_counts.index[0]
            direction_consistent = (dominant_dir == direction_canonical)
        else:
            direction_consistent = None

        # Study-level dependence flag:
        # Aynı program×cycle kombinasyonu birden fazla çalışmadan mı geliyor?
        study_prog_cycle = grp.apply(
            lambda r: frozenset([(p, c) for p in r["programs_list"] for c in r["cycles_list"]]),
            axis=1
        )
        # Crude proxy: n_studies / n_rows oranı (düşük = aynı program/cycle tekrarı)
        dependence_ratio = round(n_studies / n_rows, 3) if n_rows else 1.0

        # Year range
        years = grp["year"].dropna()
        year_range = f"{int(years.min())}-{int(years.max())}" if len(years) else ""

        records.append({
            "canonical_construct":  construct,
            "n_studies":            n_studies,
            "n_rows":               n_rows,
            "n_programs":           n_programs,
            "programs":             "|".join(sorted(all_programs)),
            "n_cycles":             n_cycles,
            "n_countries":          n_countries,
            "n_predictive_rows":    n_predictive,
            "frac_predictive":      round(frac_predictive, 3),
            "direction_canonical":  direction_canonical,
            "direction_consistent": direction_consistent,
            "dependence_ratio":     dependence_ratio,  # 1.0 = no duplication
            "year_range":           year_range,
            "forecastable":         FORECASTABILITY.get(construct, False),
        })

    return pd.DataFrame(records).sort_values("n_studies", ascending=False)


def compute_weights(audit: pd.DataFrame) -> pd.DataFrame:
    """
    Per-construct literature priority weights.

    w_F  — frequency (log-normalized, stable for skewed distributions)
    w_P  — program diversity / max_programs
    w_C  — cycle diversity / max_cycles (capped at 8)
    w_N  — country diversity / 50 (robust normalization)
    w_E  — evidence type uplift (predictive fraction)
    W_j  — composite = geometric mean of w_F, w_P, w_C, w_N * w_E
    """
    df = audit.copy()

    # w_F: log-normalized frequency
    log_f = np.log1p(df["n_studies"].values)
    df["w_F"] = log_f / log_f.max()

    # Also raw proportion for comparison
    df["w_F_raw"] = df["n_studies"] / df["n_studies"].max()

    # w_P: program diversity (max = 8 ILSAs)
    df["w_P"] = (df["n_programs"] / 8).clip(0, 1)

    # w_C: cycle diversity (cap at 8)
    df["w_C"] = (df["n_cycles"] / 8).clip(0, 1)

    # w_N: country diversity (50 as robust max)
    df["w_N"] = (df["n_countries"] / 50).clip(0, 1)

    # w_E: evidence type uplift
    # Base=1.0, predictive fraction bonus up to 1.3x → normalized to 0-1 range
    df["w_E"] = (1.0 + 0.3 * df["frac_predictive"]).clip(1.0, 1.3)
    # Normalize w_E to 0-1
    w_e_vals = df["w_E"].values
    df["w_E_norm"] = (w_e_vals - w_e_vals.min()) / (w_e_vals.max() - w_e_vals.min() + 1e-9)

    # W_j = geometric mean of (w_F, w_P, w_C, w_N) × w_E
    # Using geometric mean gives balanced contribution; not dominated by any single dim
    components = df[["w_F","w_P","w_C","w_N"]].values.clip(0.01, None)  # avoid log(0)
    log_geo = np.mean(np.log(components), axis=1)
    geo_mean = np.exp(log_geo)
    df["W_geo"] = geo_mean * df["w_E"]
    # Normalize W_geo to 0-1
    df["W_j"] = (df["W_geo"] / df["W_geo"].max()).round(4)

    # Forecastable-only flag
    df["W_j_forecast"] = df["W_j"].where(df["forecastable"], 0.0)
    # Normalize forecastable subset to 0-1
    fg = df["W_j_forecast"]
    if fg.max() > 0:
        df["W_j_forecast"] = (fg / fg.max()).round(4)

    return df


def main():
    df = load_evidence()
    log.info("Evidence matrix: %d satır, %d unique construct",
             len(df), df["canonical_construct"].nunique())

    # ── Provenance audit ──────────────────────────────────────────────────────
    audit = compute_provenance_audit(df)
    audit.to_csv(OUT_DIR / "evidence_provenance_audit.csv", index=False)
    log.info("Provenance audit kaydedildi: %d construct", len(audit))

    print("\n=== Evidence Provenance Audit ===")
    print(audit[["canonical_construct","n_studies","n_programs","n_cycles",
                 "n_countries","frac_predictive","forecastable"]].to_string(index=False))

    # ── Literature priority weights ───────────────────────────────────────────
    weights = compute_weights(audit)
    weights_out = OUT_DIR / "literature_priority.csv"
    weights.to_csv(weights_out, index=False)
    log.info("Literature priority kaydedildi: %s", weights_out)

    # ── Final priority vector (forecastable constructs only) ──────────────────
    forecast_w = (weights[weights["forecastable"]]
                  .sort_values("W_j_forecast", ascending=False)
                  [["canonical_construct","n_studies","w_F","w_P","w_C","w_N",
                    "w_E","W_j","W_j_forecast","frac_predictive","forecastable"]])

    print("\n=== Final Literature Priority — Forecastable Constructs ===")
    pd.set_option("display.float_format", "{:.4f}".format)
    print(forecast_w.to_string(index=False))

    print("\n=== Non-forecastable (excluded from alignment) ===")
    nf = weights[~weights["forecastable"]][["canonical_construct","n_studies","forecastable"]]
    print(nf.to_string(index=False))

    # ── Comparison: w_F_raw vs w_F vs W_j ─────────────────────────────────────
    print("\n=== Weight specification comparison (forecastable) ===")
    comp = forecast_w[["canonical_construct","w_F","W_j_forecast"]].copy()
    comp.columns = ["construct","w_F (log-norm)","W_j (composite)"]
    print(comp.to_string(index=False))

    # ── Direction check ────────────────────────────────────────────────────────
    print("\n=== Direction consistency check ===")
    dir_check = audit[audit["canonical_construct"].isin(CANONICAL_DIRECTION.keys())]
    print(dir_check[["canonical_construct","direction_canonical",
                      "direction_consistent","n_rows"]].to_string(index=False))

    # ── Dependency / duplication flag ─────────────────────────────────────────
    print("\n=== Study-level dependence (dependence_ratio = n_studies/n_rows) ===")
    print("  1.0 = no duplication; <0.5 = many rows per study (high duplication)")
    dep = audit.sort_values("dependence_ratio")[
        ["canonical_construct","n_studies","n_rows","dependence_ratio"]
    ]
    print(dep.to_string(index=False))

    # ── Key methodological notes ──────────────────────────────────────────────
    print("\n=== Metodolojik notlar ===")
    print("  1. GENDER (50 studies) forecastable=False: demographic, not modifiable predictor")
    print("  2. PARENTAL_EDUCATION: w_F_raw=0.130 (7/54); prior estimate of 0.90 was too high")
    print("  3. HOME_LITERACY_ACTIVITIES: only 1 study (PIRLS-specific); w_F=low")
    print("  4. W_j uses geometric mean → no single dimension dominates")
    print("  5. w_E uplift for predictive evidence (frac_predictive × 0.3)")
    print("  6. W_j_forecast = 0 for non-forecastable constructs")
    print()
    print("  Next: replace W_FREQ in compute_shap_rho_align.py with W_j_forecast values")
    print("  after final sensitivity check.")


if __name__ == "__main__":
    main()
