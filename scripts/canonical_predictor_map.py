#!/usr/bin/env python3
"""Unified literature predictor → canonical construct mapping (Stage 5)."""

from __future__ import annotations

import re

# Official ILSA microdata codes → forecasting canonical construct
ILSA_CODE_TO_CANONICAL: dict[str, str] = {
    "ESCS": "SES_COMPOSITE",
    "HOMEPOS": "HOME_RESOURCES",
    "HEDRES": "HOME_RESOURCES",
    "CULTPOSS": "HOME_RESOURCES",
    "WEALTH": "SES_COMPOSITE",
    "HISEI": "SES_COMPOSITE",
    "HISCED": "PARENTAL_EDUCATION",
    "S_HISCED": "PARENTAL_EDUCATION",
    "MISCED": "PARENTAL_EDUCATION",
    "FISCED": "PARENTAL_EDUCATION",
    "PARED": "PARENTAL_EDUCATION",
    "BSDGEDUP": "PARENTAL_EDUCATION",
    "BSDG07": "PARENTAL_EDUCATION",
    "BSDG08": "PARENTAL_EDUCATION",
    "ASDHEDUP": "PARENTAL_EDUCATION",
    "ASDHELA": "HOME_LITERACY_ACTIVITIES",
    "ASDHELB": "HOME_LITERACY_ACTIVITIES",
    "S_HOMLIT": "HOME_LITERACY_ACTIVITIES",
    "BELONG": "SCHOOL_BELONGING",
    "ICTAVHOM": "ICT_HOME_ACCESS",
    "ICTAVSCH": "ICT_SCHOOL_ACCESS",
    "NISB": "SES_COMPOSITE",
    "S_NISB": "SES_COMPOSITE",
    "S_ECOB": "SES_COMPOSITE",
    "ISESCS": "SES_COMPOSITE",
    "ITSEX": "GENDER",
    "IMMIG": "IMMIGRANT_STATUS",
    "LANGN": "LANGUAGE_BACKGROUND",
    "MATHEFF": "SELF_EFFICACY",
    "SCIEEFF": "SELF_EFFICACY",
    "ANXMAT": "ANXIETY",
    "MOTIV": "MOTIVATION",
    "JOYREAD": "MOTIVATION",
    "LMINS": "INSTRUCTIONAL_TIME",
    "SECLSS": "TEACHER_SELF_EFFICACY_CLS_MGMT",
    "T3SECLS": "TEACHER_SELF_EFFICACY_CLS_MGMT",
    "SEINSS": "TEACHER_SELF_EFFICACY_INSTRUCTION",
    "T3SEINS": "TEACHER_SELF_EFFICACY_INSTRUCTION",
    "SEENGS": "TEACHER_SELF_EFFICACY_ENGAGEMENT",
    "T3SEENG": "TEACHER_SELF_EFFICACY_ENGAGEMENT",
    "TCDISCS": "DISCIPLINARY_CLIMATE",
    "T3DISC": "DISCIPLINARY_CLIMATE",
    "TCCOLLS": "PROF_COLLABORATION",
    "T3COLES": "PROF_COLLABORATION",
    "TEFFPROS": "EFFECTIVE_PD",
    "T3EFFPD": "EFFECTIVE_PD",
    "TJSENVS": "JOB_SATISFACTION_ENVIRONMENT",
    "T3JSENV": "JOB_SATISFACTION_ENVIRONMENT",
    "TJSPROS": "JOB_SATISFACTION_PROFESSION",
    "T3JSPRO": "JOB_SATISFACTION_PROFESSION",
}

SCHEMA_CATEGORY_TO_CANONICAL: dict[str, str] = {
    "socioeconomic": "SES_COMPOSITE",
    "parent_home": "HOME_RESOURCES",
    "demographic": "GENDER",
    "student_attitude": "MOTIVATION",
    "student_behavior": "INSTRUCTIONAL_TIME",
    "teacher": "TEACHER_QUALITY",
    "school": "SCHOOL_RESOURCES",
    "ict": "ICT_ACCESS",
    "curriculum": "INSTRUCTIONAL_TIME",
    "prior_achievement": "PRIOR_ACHIEVEMENT",
    "peer_effects": "SCHOOL_CLIMATE",
    "system_level": "GDP_EXPENDITURE",
    "process_data": "PRIOR_ACHIEVEMENT",
}

# Longest keyword first (substring match)
KEYWORD_TO_CANONICAL: list[tuple[str, str]] = sorted(
    [
        ("socio-economic status", "SES_COMPOSITE"),
        ("socioeconomic status", "SES_COMPOSITE"),
        ("economic, social and cultural status", "SES_COMPOSITE"),
        ("economic social and cultural", "SES_COMPOSITE"),
        ("home educational resources", "HOME_RESOURCES"),
        ("home literacy activities", "HOME_LITERACY_ACTIVITIES"),
        ("home literacy", "HOME_LITERACY_ACTIVITIES"),
        ("literacy activities", "HOME_LITERACY_ACTIVITIES"),
        ("books at home", "HOME_RESOURCES"),
        ("parental education", "PARENTAL_EDUCATION"),
        ("parents' education", "PARENTAL_EDUCATION"),
        ("parent education", "PARENTAL_EDUCATION"),
        ("mother's education", "PARENTAL_EDUCATION"),
        ("father's education", "PARENTAL_EDUCATION"),
        ("educational attainment", "PARENTAL_EDUCATION"),
        ("education level", "PARENTAL_EDUCATION"),
        ("school belonging", "SCHOOL_BELONGING"),
        ("sense of belonging", "SCHOOL_BELONGING"),
        ("disciplinary climate", "DISCIPLINARY_CLIMATE"),
        ("classroom climate", "SCHOOL_CLIMATE"),
        ("school climate", "SCHOOL_CLIMATE"),
        ("teacher self-efficacy", "TEACHER_SELF_EFFICACY_CLS_MGMT"),
        ("teacher collaboration", "PROF_COLLABORATION"),
        ("professional collaboration", "PROF_COLLABORATION"),
        ("professional development", "EFFECTIVE_PD"),
        ("job satisfaction", "JOB_SATISFACTION_PROFESSION"),
        ("teacher support", "TEACHER_QUALITY"),
        ("teacher quality", "TEACHER_QUALITY"),
        ("teacher enthusiasm", "TEACHER_QUALITY"),
        ("teacher feedback", "TEACHER_QUALITY"),
        ("instructional quality", "TEACHER_QUALITY"),
        ("teacher-directed instruction", "TEACHER_QUALITY"),
        ("student-teacher ratio", "SCHOOL_RESOURCES"),
        ("class size", "SCHOOL_RESOURCES"),
        ("school type", "SCHOOL_RESOURCES"),
        ("private tutoring", "SCHOOL_RESOURCES"),
        ("internet access", "ICT_ACCESS"),
        ("computer access", "ICT_ACCESS"),
        ("ict access", "ICT_ACCESS"),
        ("computer literacy", "ICT_ACCESS"),
        ("immigrant background", "IMMIGRANT_STATUS"),
        ("immigration status", "IMMIGRANT_STATUS"),
        ("language background", "LANGUAGE_BACKGROUND"),
        ("language at home", "LANGUAGE_BACKGROUND"),
        ("prior achievement", "PRIOR_ACHIEVEMENT"),
        ("previous score", "PRIOR_ACHIEVEMENT"),
        ("grade repetition", "PRIOR_ACHIEVEMENT"),
        ("performance level", "PRIOR_ACHIEVEMENT"),
        ("reading enjoyment", "MOTIVATION"),
        ("math anxiety", "ANXIETY"),
        ("test anxiety", "ANXIETY"),
        ("student behavior stress", "ANXIETY"),
        ("self-efficacy", "SELF_EFFICACY"),
        ("self concept", "SELF_EFFICACY"),
        ("self-concept", "SELF_EFFICACY"),
        ("intrinsic motivation", "MOTIVATION"),
        ("instructional time", "INSTRUCTIONAL_TIME"),
        ("learning time", "INSTRUCTIONAL_TIME"),
        ("time on task", "INSTRUCTIONAL_TIME"),
        ("weekly math learning time", "INSTRUCTIONAL_TIME"),
        ("response time", "INSTRUCTIONAL_TIME"),
        ("education spending", "GDP_EXPENDITURE"),
        ("civic knowledge", "CIVIC_KNOWLEDGE"),
        ("computer and information literacy", "ICT_ACCESS"),
        ("independent reading", "MOTIVATION"),
        ("curriculum", "CURRICULUM_TYPE"),
        ("teaching time", "INSTRUCTIONAL_TIME"),
        ("homework", "INSTRUCTIONAL_TIME"),
        ("escs", "SES_COMPOSITE"),
        ("homepos", "HOME_RESOURCES"),
        ("belonging", "SCHOOL_BELONGING"),
        ("motivation", "MOTIVATION"),
        ("anxiety", "ANXIETY"),
        ("gender", "GENDER"),
        ("female", "GENDER"),
        ("male", "GENDER"),
        ("age", "AGE_GRADE"),
        ("grade level", "AGE_GRADE"),
        ("gdp", "GDP_EXPENDITURE"),
        ("achievement", "PRIOR_ACHIEVEMENT"),
        ("mathematics score", "PRIOR_ACHIEVEMENT"),
        ("reading score", "PRIOR_ACHIEVEMENT"),
        ("science score", "PRIOR_ACHIEVEMENT"),
        ("score", "PRIOR_ACHIEVEMENT"),
        ("teacher", "TEACHER_QUALITY"),
        ("school", "SCHOOL_RESOURCES"),
        ("student", "MOTIVATION"),
        ("reading", "MOTIVATION"),
        ("science", "MOTIVATION"),
        ("math", "PRIOR_ACHIEVEMENT"),
        ("mathematics", "PRIOR_ACHIEVEMENT"),
        ("literacy", "MOTIVATION"),
        ("ict", "ICT_ACCESS"),
        ("digital", "ICT_ACCESS"),
        ("computer", "ICT_ACCESS"),
        ("internet", "ICT_ACCESS"),
        ("support", "TEACHER_QUALITY"),
        ("collaboration", "PROF_COLLABORATION"),
        ("enjoyment", "MOTIVATION"),
        ("interest", "MOTIVATION"),
        ("ses", "SES_COMPOSITE"),
        ("home", "HOME_RESOURCES"),
    ],
    key=lambda x: len(x[0]),
    reverse=True,
)

# Legacy aliases used in effect_sizes / v2 weights
CANONICAL_ALIASES: dict[str, str] = {
    "BELONGING": "SCHOOL_BELONGING",
    "SCHOOL_CLIMATE": "DISCIPLINARY_CLIMATE",
    "ICT_ACCESS": "ICT_HOME_ACCESS",
    "HOME_LITERACY_INDEX": "HOME_LITERACY_ACTIVITIES",
    "NAT_SES_INDEX": "SES_COMPOSITE",
}


def normalize_canonical(label: str) -> str:
    if not label:
        return "OTHER"
    lab = label.strip().upper()
    return CANONICAL_ALIASES.get(lab, lab)


def canonical_map(
    predictor_text: str,
    *,
    variable_code: str | None = None,
    category: str | None = None,
) -> str:
    """Map free text / code / schema category → canonical construct."""
    if variable_code:
        code = re.sub(r"[^A-Za-z0-9_]", "", str(variable_code)).upper()
        if code in ILSA_CODE_TO_CANONICAL:
            return normalize_canonical(ILSA_CODE_TO_CANONICAL[code])
        # PV prefixes
        if code.startswith("PV") and "CIV" in code:
            return "CIVIC_KNOWLEDGE"
        if code.startswith("PV") and ("CIL" in code or "CT" in code):
            return "ICT_ACCESS"

    if category:
        cat = str(category).strip().lower()
        if cat in SCHEMA_CATEGORY_TO_CANONICAL:
            base = SCHEMA_CATEGORY_TO_CANONICAL[cat]
            # Refine attitude/behavior with predictor text
            if cat in ("student_attitude", "student_behavior", "demographic", "teacher", "school"):
                sub = canonical_map(predictor_text)
                if sub != "OTHER":
                    return sub
            return normalize_canonical(base)

    text = (predictor_text or "").strip()
    if not text:
        return "OTHER"

    # Token that looks like an ILSA code
    code_match = re.search(r"\b([A-Z][A-Z0-9_]{2,})\b", text.upper())
    if code_match:
        code = code_match.group(1)
        if code in ILSA_CODE_TO_CANONICAL:
            return normalize_canonical(ILSA_CODE_TO_CANONICAL[code])

    tl = text.lower()
    for kw, construct in KEYWORD_TO_CANONICAL:
        if kw in tl:
            return normalize_canonical(construct)

    return "OTHER"
