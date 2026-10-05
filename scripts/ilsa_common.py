#!/usr/bin/env python3
"""Shared ILSA forecasting helpers: microdata root + unified W_j loader."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STAGE5 = PROJECT_ROOT / "outputs" / "stage5"


def microdata_root() -> Path:
    """Portable microdata root.

    Priority:
      1. ILSA_MICRODATA_ROOT env var
      2. Legacy Desktop path (Merve local)
    """
    env = os.environ.get("ILSA_MICRODATA_ROOT", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return Path.home() / "Desktop" / "ILSA Datasets"


# literature_priority construct → feature keys used by LOCO / hold-out scripts
CONSTRUCT_TO_FEATURE_ALIASES: dict[str, list[str]] = {
    "SES_COMPOSITE": [
        "SES_COMPOSITE", "ESCS", "NAT_SES_INDEX", "ISESCS", "S_NISB", "NISB",
    ],
    "HOME_RESOURCES": ["HOME_RESOURCES", "HOMEPOS"],
    "SCHOOL_BELONGING": ["SCHOOL_BELONGING", "BELONGING", "BELONG"],
    "PARENTAL_EDUCATION": [
        "PARENTAL_EDUCATION", "PARENTAL_EDU", "PARENTAL_EDUCATION_ISCED",
        "BSDGEDUP", "ASDHEDUP", "HISCED", "S_HISCED", "PARED",
    ],
    "HOME_LITERACY_ACTIVITIES": [
        "HOME_LITERACY_ACTIVITIES", "HOME_LITERACY_INDEX", "ASDHELA", "S_HOMLIT",
    ],
    "ICT_HOME_ACCESS": ["ICT_HOME_ACCESS", "ICT_ACCESS", "ICT_INDEX", "ICTAVHOM"],
    "ICT_SCHOOL_ACCESS": ["ICT_SCHOOL_ACCESS", "ICTAVSCH"],
    "TEACHER_SELF_EFFICACY_CLS_MGMT": [
        "TEACHER_SELF_EFFICACY_CLS_MGMT", "TEACHER_QUALITY", "SECLSS", "T3SECLS",
    ],
    "DISCIPLINARY_CLIMATE": ["DISCIPLINARY_CLIMATE", "DISCLIMA", "TCDISCS", "T3DISC"],
    "EFFECTIVE_PD": ["EFFECTIVE_PD", "EFFPD", "TEFFPROS", "T3EFFPD"],
    "JOB_SATISFACTION_PROFESSION": [
        "JOB_SATISFACTION_PROFESSION", "JOB_SAT_PROF", "TJSPROS", "T3JSPRO",
    ],
    "PRIOR_ACHIEVEMENT": ["PRIOR_ACHIEVEMENT", "LAG_SCORE"],
    "GENDER": ["GENDER", "GENDER_GAP"],
    "SELF_EFFICACY": ["SELF_EFFICACY", "SELF_CONCEPT"],
    "MOTIVATION": ["MOTIVATION"],
    "ANXIETY": ["ANXIETY"],
    "INSTRUCTIONAL_TIME": ["INSTRUCTIONAL_TIME", "INSTR_TIME"],
    "SCHOOL_RESOURCES": ["SCHOOL_RESOURCES", "SCHOOL_RES"],
    "GDP_EXPENDITURE": ["GDP_EXPENDITURE", "GDP_PC"],
}


def load_forecast_weights() -> dict[str, float]:
    """Unified W_j for M1 √W scaling.

    Primary: literature_priority.csv → W_j_forecast (0 for non-forecastable).
    Fallback: predictor_weights_v2.csv → w_norm.
    Returns a flat dict keyed by construct names AND feature aliases.
    """
    lit_path = STAGE5 / "literature_priority.csv"
    v2_path = STAGE5 / "predictor_weights_v2.csv"
    weights: dict[str, float] = {}

    if lit_path.exists():
        lit = pd.read_csv(lit_path)
        if "W_j_forecast" in lit.columns and "canonical_construct" in lit.columns:
            for _, row in lit.iterrows():
                construct = str(row["canonical_construct"])
                w = float(row["W_j_forecast"]) if pd.notna(row["W_j_forecast"]) else 0.0
                # Non-forecastable constructs stay at 0 in W_j_forecast; for Ridge
                # scaling treat missing evidence as neutral 1.0 only when truly absent.
                aliases = CONSTRUCT_TO_FEATURE_ALIASES.get(construct, [construct])
                for alias in aliases:
                    weights[alias] = w if w > 0 else weights.get(alias, 1.0)
            # Re-apply: if W_j_forecast==0 explicitly (non-forecastable), keep 1.0
            # so M1 does not zero-out features that still enter the X matrix.
            for _, row in lit.iterrows():
                construct = str(row["canonical_construct"])
                w = float(row["W_j_forecast"]) if pd.notna(row["W_j_forecast"]) else 0.0
                if w <= 0:
                    continue
                for alias in CONSTRUCT_TO_FEATURE_ALIASES.get(construct, [construct]):
                    weights[alias] = w

    if v2_path.exists():
        v2 = pd.read_csv(v2_path)
        key_col = "feature_name" if "feature_name" in v2.columns else "variable"
        for _, row in v2.iterrows():
            feat = str(row[key_col])
            w = float(row["w_norm"]) if pd.notna(row.get("w_norm")) else 1.0
            weights.setdefault(feat, w)
            if "predictor_canonical" in v2.columns and pd.notna(row["predictor_canonical"]):
                weights.setdefault(str(row["predictor_canonical"]), w)

    return weights
