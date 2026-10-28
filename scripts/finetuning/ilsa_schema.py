"""
Shared extraction schema for the ILSA evidence-extraction experiments (v2).

Single source of truth for:
  - controlled vocabularies (codebook v1.3 constructs, direction, effect type, PV handling)
  - normalization of gold v1.4.1 labels and model outputs onto those vocabularies
  - the Pydantic model used for validation (instructor / post-hoc)
  - the JSON schema used for constrained decoding (vLLM structured outputs)

Every other v2 script imports from here so that training targets, decoding
constraints and the evaluator can never drift apart.
"""

from __future__ import annotations

import json
import re
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

# Controlled vocabularies

CONSTRUCTS: tuple[str, ...] = (
    "AGE_GROUP", "ANXIETY", "DISCIPLINARY_CLIMATE", "EDUCATIONAL_ATTAINMENT",
    "GENDER", "HOMEWORK_TIME", "HOME_LITERACY_ACTIVITIES", "ICT_HOME_ACCESS",
    "IMMIGRANT_STATUS", "INSTRUCTION_TIME", "JOB_TRAINING", "LITERACY_OUTCOME",
    "MOTIVATION", "NUMERACY_OUTCOME", "PARENTAL_EDUCATION", "PRIVATE_TUTORING",
    "PROCESS_DATA_BEHAVIOR", "SCHOOL_BELONGING", "SCHOOL_ICT_INFRASTRUCTURE",
    "SCHOOL_LEADERSHIP", "SCHOOL_LOCATION", "SELF_CONCEPT", "SES_COMPOSITE",
    "TEACHER_EXPERIENCE", "TEACHER_ICT_USE", "TEACHER_PD",
    "TEACHER_SELF_EFFICACY_CLS_MGMT", "TEACHING_QUALITY", "WORKPLACE_AUTONOMY",
    "UNMATCHED",
)
DIRECTIONS: tuple[str, ...] = ("positive", "negative", "mixed", "not_reported")
EFFECT_TYPES: tuple[str, ...] = ("E_assoc", "E_predictive", "E_cross_cycle", "E_cross_country")
PV_HANDLING: tuple[str, ...] = (
    "rubin_rules", "average_pv", "single_pv", "all_pvs", "not_applicable", "not_reported",
)
OUTCOME_CLASSES: tuple[str, ...] = ("mathematics", "reading", "science", "other")

# Construct frequency strata from EVALUATION_PROTOCOL_v1.md section 4.4
STRATA: dict[str, str] = {
    **{c: "HIGH" for c in (
        "SES_COMPOSITE", "GENDER", "SELF_CONCEPT", "ICT_HOME_ACCESS",
        "DISCIPLINARY_CLIMATE", "IMMIGRANT_STATUS", "SCHOOL_ICT_INFRASTRUCTURE")},
    **{c: "MID" for c in (
        "EDUCATIONAL_ATTAINMENT", "TEACHER_ICT_USE", "MOTIVATION", "ANXIETY",
        "NUMERACY_OUTCOME", "PROCESS_DATA_BEHAVIOR")},
    **{c: "LOW" for c in (
        "TEACHER_SELF_EFFICACY_CLS_MGMT", "INSTRUCTION_TIME", "TEACHING_QUALITY",
        "SCHOOL_BELONGING", "PARENTAL_EDUCATION", "HOME_LITERACY_ACTIVITIES",
        "SCHOOL_LOCATION", "PRIVATE_TUTORING", "TEACHER_EXPERIENCE", "TEACHER_PD",
        "JOB_TRAINING", "AGE_GROUP", "LITERACY_OUTCOME")},
    **{c: "SINGLE" for c in ("HOMEWORK_TIME", "SCHOOL_LEADERSHIP", "WORKPLACE_AUTONOMY")},
    "UNMATCHED": "UNMATCHED",
}

# Length caps used both in the JSON schema and in Pydantic validation.
# They bound runaway strings inside an otherwise schema-valid object.
MAX_LEN_PREDICTOR = 200
MAX_LEN_OUTCOME = 200
MAX_LEN_EFFECT_SIZE = 120
MAX_LEN_EVIDENCE = 600
MAX_LEN_ILSA_VARIABLE = 60

# Fixed key order for serialized targets (deterministic training targets).
FIELD_ORDER: tuple[str, ...] = (
    "predictor", "ilsa_variable", "canonical_construct", "outcome", "effect_type",
    "direction", "effect_size", "plausible_values_handling", "student_weights_used",
    "evidence_text",
)
# Chunk-level extraction omits the two study-level methods fields: in gold v1.4.1 they are
# constant within every study, so they are extracted once per study by a methods pass and
# attached to every pair at merge time.
STUDY_LEVEL_FIELDS: tuple[str, ...] = ("plausible_values_handling", "student_weights_used")
CHUNK_FIELDS: tuple[str, ...] = tuple(f for f in FIELD_ORDER if f not in STUDY_LEVEL_FIELDS)


# Normalization

def _clean(value: Any) -> str:
    if value is None:
        return ""
    s = str(value).strip()
    return "" if s.lower() in ("nan", "none", "null") else s


def norm_construct(value: Any) -> str:
    """Map any construct label onto the codebook; out-of-vocabulary labels become UNMATCHED."""
    s = _clean(value).upper().replace(" ", "_")
    if s in CONSTRUCTS:
        return s
    return "UNMATCHED"


def norm_direction(value: Any) -> str:
    s = _clean(value).lower()
    if s in ("1", "1.0", "+", "positive", "pos"):
        return "positive"
    if s in ("-1", "-1.0", "-", "negative", "neg"):
        return "negative"
    if s == "mixed":
        return "mixed"
    # "0", "", "not_reported", "null" and anything unrecognized
    return "not_reported"


def norm_effect_type(value: Any) -> str:
    s = _clean(value).lower()
    if s in ("e_predictive", "predictive"):
        return "E_predictive"
    if s in ("e_cross_cycle", "cross_cycle"):
        return "E_cross_cycle"
    if s in ("e_cross_country", "cross_country"):
        return "E_cross_country"
    # "E_assoc", "associative", "unspecified" -> association is the codebook default
    return "E_assoc"


def norm_pv(value: Any) -> str:
    s = _clean(value).lower()
    for v in PV_HANDLING:
        if s.startswith(v):
            return v
    if s.startswith("all_pv"):
        return "all_pvs"
    return "not_reported"


def norm_weights(value: Any) -> Optional[bool]:
    s = _clean(value).lower()
    if s in ("true", "1", "yes"):
        return True
    if s in ("false", "0", "no"):
        return False
    return None


_OUTCOME_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("mathematics", re.compile(r"\bmath|numeracy|\bpvmath|pv\dmath", re.I)),
    ("science", re.compile(r"scien", re.I)),
    ("reading", re.compile(r"\bread|literacy\b(?!.*computer)|pv\dread", re.I)),
)


def outcome_class(value: Any) -> str:
    """Coarse outcome class used in the strict matching key."""
    s = _clean(value)
    for label, pat in _OUTCOME_PATTERNS:
        if pat.search(s):
            return label
    return "other"


def normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", _clean(value)).strip().lower()


def normalize_gold_row(row: dict[str, Any]) -> dict[str, Any]:
    """Project one gold v1.4.1 CSV row onto the extraction schema."""
    return {
        "record_id": _clean(row.get("record_id")),
        "study_id": _clean(row.get("study_id")),
        "predictor": _clean(row.get("predictor")),
        "canonical_construct": norm_construct(row.get("canonical_construct")),
        "canonical_construct_raw": _clean(row.get("canonical_construct")),
        "outcome": _clean(row.get("target_variable")),
        "outcome_class": outcome_class(row.get("target_variable")),
        "effect_type": norm_effect_type(row.get("effect_type")),
        "direction": norm_direction(row.get("direction_canonical")),
        "plausible_values_handling": norm_pv(row.get("plausible_values_handling")),
        "student_weights_used": norm_weights(row.get("student_weights_used")),
        "program": _clean(row.get("program")),
        "cycle": _clean(row.get("cycle")),
    }


def normalize_pred(pair: dict[str, Any]) -> dict[str, Any]:
    """Project one model-produced pair onto the extraction schema (tolerant of drift)."""
    return {
        "predictor": _clean(pair.get("predictor")),
        "canonical_construct": norm_construct(pair.get("canonical_construct")),
        "canonical_construct_raw": _clean(pair.get("canonical_construct")),
        "outcome": _clean(pair.get("outcome")),
        "outcome_class": outcome_class(pair.get("outcome")),
        "effect_type": norm_effect_type(pair.get("effect_type")),
        "direction": norm_direction(pair.get("direction")),
        "plausible_values_handling": norm_pv(pair.get("plausible_values_handling")),
        "student_weights_used": norm_weights(pair.get("student_weights_used")),
        "effect_size": _clean(pair.get("effect_size")),
        "evidence_text": _clean(pair.get("evidence_text")),
    }


# Pydantic models (validation layer)

class EvidencePair(BaseModel):
    model_config = ConfigDict(extra="forbid")

    predictor: str = Field(min_length=1, max_length=MAX_LEN_PREDICTOR)
    ilsa_variable: Optional[str] = Field(default=None, max_length=MAX_LEN_ILSA_VARIABLE)
    canonical_construct: Literal[CONSTRUCTS]  # type: ignore[valid-type]
    outcome: str = Field(min_length=1, max_length=MAX_LEN_OUTCOME)
    effect_type: Literal[EFFECT_TYPES]  # type: ignore[valid-type]
    direction: Literal[DIRECTIONS]  # type: ignore[valid-type]
    effect_size: Optional[str] = Field(default=None, max_length=MAX_LEN_EFFECT_SIZE)
    plausible_values_handling: Literal[PV_HANDLING]  # type: ignore[valid-type]
    student_weights_used: Optional[bool] = None
    evidence_text: str = Field(min_length=1, max_length=MAX_LEN_EVIDENCE)

    @field_validator("effect_size", mode="before")
    @classmethod
    def _effect_size_to_str(cls, v: Any) -> Optional[str]:
        return None if v is None else str(v)


class MethodsOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    plausible_values_handling: Literal[PV_HANDLING]  # type: ignore[valid-type]
    student_weights_used: Optional[bool] = None
    evidence_text: Optional[str] = Field(default=None, max_length=MAX_LEN_EVIDENCE)


class ExtractionOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[EvidencePair]


def json_schema(max_items: int = 8, fields: tuple[str, ...] = FIELD_ORDER) -> dict[str, Any]:
    """JSON schema for constrained decoding.

    A top-level object wrapper is used instead of a bare array: after the closing
    brace the grammar only admits EOS, and `maxItems` forces the array to close.
    `uniqueItems` is deliberately absent (not enforced by the grammar backends);
    duplicates are removed in post-processing instead.
    """
    nullable_str = lambda n: {"anyOf": [{"type": "string", "maxLength": n}, {"type": "null"}]}  # noqa: E731
    properties = {
        "predictor": {"type": "string", "minLength": 1, "maxLength": MAX_LEN_PREDICTOR},
        "ilsa_variable": nullable_str(MAX_LEN_ILSA_VARIABLE),
        "canonical_construct": {"type": "string", "enum": list(CONSTRUCTS)},
        "outcome": {"type": "string", "minLength": 1, "maxLength": MAX_LEN_OUTCOME},
        "effect_type": {"type": "string", "enum": list(EFFECT_TYPES)},
        "direction": {"type": "string", "enum": list(DIRECTIONS)},
        "effect_size": nullable_str(MAX_LEN_EFFECT_SIZE),
        "plausible_values_handling": {"type": "string", "enum": list(PV_HANDLING)},
        "student_weights_used": {"anyOf": [{"type": "boolean"}, {"type": "null"}]},
        "evidence_text": {"type": "string", "minLength": 1, "maxLength": MAX_LEN_EVIDENCE},
    }
    item = {
        "type": "object",
        "additionalProperties": False,
        "required": list(fields),
        "properties": {k: properties[k] for k in fields},
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["items"],
        "properties": {"items": {"type": "array", "maxItems": max_items, "items": item}},
    }


def methods_json_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["plausible_values_handling", "student_weights_used", "evidence_text"],
        "properties": {
            "plausible_values_handling": {"type": "string", "enum": list(PV_HANDLING)},
            "student_weights_used": {"anyOf": [{"type": "boolean"}, {"type": "null"}]},
            "evidence_text": {"anyOf": [{"type": "string", "maxLength": MAX_LEN_EVIDENCE}, {"type": "null"}]},
        },
    }


def serialize_target(pairs: list[dict[str, Any]], fields: tuple[str, ...] = FIELD_ORDER) -> str:
    """Deterministic serialization of a training target: fixed key order, compact separators."""
    ordered = [{k: p.get(k) for k in fields} for p in pairs]
    return json.dumps({"items": ordered}, ensure_ascii=False, separators=(",", ":"))


def parse_output(raw: str) -> tuple[list[dict[str, Any]], str]:
    """Parse raw model text. Returns (pairs, status) with status in
    {"ok", "repaired", "truncated", "empty"}. Accepts both {"items": [...]} and bare arrays."""
    text = re.sub(r"^```(?:json)?\s*|```\s*$", "", (raw or "").strip(), flags=re.MULTILINE).strip()
    if not text:
        return [], "empty"
    for candidate in (text, re.sub(r",\s*([}\]])", r"\1", text)):
        try:
            obj = json.loads(candidate)
            items = obj.get("items", []) if isinstance(obj, dict) else obj
            if isinstance(items, list):
                return [p for p in items if isinstance(p, dict)], ("ok" if candidate is text else "repaired")
        except json.JSONDecodeError:
            pass
    # Salvage complete objects from a truncated array (never counted as a clean parse)
    salvaged = []
    decoder = json.JSONDecoder()
    start = text.find("{", 1 if text.startswith("{") else 0)
    while start != -1:
        try:
            obj, end = decoder.raw_decode(text, start)
            if isinstance(obj, dict) and "predictor" in obj:
                salvaged.append(obj)
            start = text.find("{", end)
        except json.JSONDecodeError:
            start = text.find("{", start + 1)
    return salvaged, "truncated"
