import json
import logging
import os
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from openai import OpenAI, APIError, RateLimitError, APITimeoutError
from pydantic import ValidationError

from src.schemas import ILSAArticleMetadata

if TYPE_CHECKING:
    from src.extractors.pdf_processor import ProcessedPDF

logger = logging.getLogger(__name__)

MODEL_NAME = "gpt-5.4-nano"
PRICE_INPUT_PER_1M = 2.50
PRICE_OUTPUT_PER_1M = 10.00

SYSTEM_PROMPT = """You are an expert research analyst specializing in International Large-Scale Assessments (ILSA: PISA, TIMSS, PIRLS, TALIS, ICILS, ICCS, PIAAC) and the use of machine learning on educational survey data.

Your task is to produce one JSON object that matches the ILSAArticleMetadata schema exactly.

COVERAGE (use the article text plus the user-message interpretation guidance):
1) metadata: bibliographic fields (title, authors, year, doi, venue, publication_type, open_access, source_category, file_name).
2) data.survey_design: whether student/replicate weights are used and any named weight variable.
3) data.plausible_values_handling and data.missing_data_handling: map the manuscript to the allowed enum literals.
4) data.sample_details: total_students and per-country counts when reported.
5) data.ml_techniques: primary model if clear, and all named ML / statistical-learning algorithms used for modeling (not mere preprocessing).
6) data.confounders_identified: covariates explicitly mentioned as controls or predictors in the model.
7) data.outcome_summary: 2-4 sentences on findings and model performance, grounded in the text.
8) data.research_design_type: predictive, causal_observational, causal_experimental, or exploratory when the paper supports it.

CORE RULES:
- Ground every filled value in the supplied article text (or in a clearly labeled EXTRACTED_TITLE_HINT in the user message). Do not invent DOIs, author lists, sample sizes, country codes, or weight variable names that never appear.
- Prefer the closest allowed enum or a concise string when the paper gives partial but directional evidence. Reserve null for fields where the manuscript truly offers no usable signal.
- Use canonical algorithm names when the paper uses synonyms (e.g. "random forests" -> Random Forest).
- Empty list [] is allowed for data.confounders_identified and data.sample_details.countries when nothing is stated.

OUTPUT: Return a single JSON object with exactly these top-level keys: metadata, data.

metadata fields only: file_name, title, authors, year, doi, venue, publication_type, open_access, source_category.

data fields only: survey_design, plausible_values_handling, missing_data_handling, sample_details, ml_techniques, confounders_identified, outcome_summary, research_design_type.

data.survey_design fields only: student_weights_used, replicate_weights_used, weight_variable_name.

data.sample_details fields only: total_students, countries (each country: country_code, n_students).

data.ml_techniques fields only: primary, all_techniques.

Do not emit any other top-level or nested keys.

"""


@dataclass
class ExtractionResult:
    file_name: str
    success: bool
    extraction: ILSAArticleMetadata | None
    input_tokens: int
    output_tokens: int
    cost_usd: float
    duration_seconds: float
    error: str | None = None


class GPTExtractor:
    def __init__(
        self,
        api_key: str = None,
        model: str = MODEL_NAME,
        max_retries: int = 4,
        base_delay: float = 2.0,
    ):
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.max_retries = max_retries
        self.base_delay = base_delay

    def _build_user_message(self, processed: "ProcessedPDF") -> list[dict]:
        sections_label = ", ".join(processed.sections.keys()) or "none"
        title_hint = ""
        if processed.metadata.get("extracted_title"):
            title_hint = (
                f"\nEXTRACTED_TITLE_HINT (use only if the body never states a title; "
                f"still do not contradict the PDF): "
                f"{processed.metadata['extracted_title']}\n"
            )

        document_text = (
            f"FILE: {processed.file_name}\n"
            f"SOURCE: {processed.source_database}\n"
            f"SECTIONS_DETECTED: {sections_label}\n"
            f"{title_hint}"
            f"--- BEGIN ARTICLE TEXT ---\n\n"
            f"{processed.extraction_text}\n\n"
            f"--- END ARTICLE TEXT ---\n\n"
            "Apply the system prompt schema to the article above. "
            "Return valid JSON only — no markdown fences, no preamble."
        )

        interpretation_text = (
            "ADDITIONAL INTERPRETATION (same article as previous block; "
            "reduces over-use of null without allowing fabrication):\n\n"
            "1) Evidence ladder — prefer filling fields in this order:\n"
            "   (A) Explicit statements in the manuscript.\n"
            "   (B) Single reasonable reading: the paper describes a procedure, "
            "estimator, or data product so concretely that only one schema value fits "
            "(map to the closest allowed enum or string).\n"
            "   (C) If the paper is silent on a dimension, use null for that field, "
            "or not_reported / not_applicable for the PV and missing-data enums only "
            "when the silence is genuine (no methods clue at all).\n\n"
            "2) Methodology enums (data.plausible_values_handling, "
            "data.missing_data_handling): do not default to null or not_reported "
            "out of caution when the abstract, methods, or results clearly mentions "
            "deletion, imputation, complete cases, MICE, Rubin's rules, PV averaging, "
            "or a single PV draw — map to the closest literal.\n\n"
            "3) Anti-hallucination — never invent: exact N, country list entries, "
            "DOIs, author strings, weight variable names, or algorithm names that "
            "never appear. Booleans (e.g. student_weights_used) require at least a "
            "clear discussion of sampling weights, representativeness, or a named "
            "weight column; otherwise null.\n\n"
            "4) data.ml_techniques.all_techniques: include every modeling algorithm "
            "named in the study (including baselines). data.ml_techniques.primary: "
            "the main or best-performing model if stated; else the model emphasized "
            "in the abstract; else null with a non-empty all_techniques when possible."
            "\n\n"
            "5) data.outcome_summary: 2-4 sentences summarizing reported findings "
            "and performance metrics only from the text — no external facts or "
            "speculative policy.\n\n"
            "6) data.confounders_identified: list variable names or short phrases "
            "the paper says were controlled or entered as predictors; [] if none named."
            "\n"
        )

        return [
            {"type": "text", "text": document_text},
            {"type": "text", "text": interpretation_text},
        ]

    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        return (
            input_tokens * PRICE_INPUT_PER_1M / 1_000_000
            + output_tokens * PRICE_OUTPUT_PER_1M / 1_000_000
        )

    @staticmethod
    def _sanitize(parsed_data: dict) -> dict:
        """
        Post-process model output to fix known failure modes before Pydantic validation.
        Modifies parsed_data in place and returns it.
        """
        INVALID_STR = {"not_reported", "not_applicable", "N/A", "n/a", "unknown", ""}

        def _normalize_literal(value, field_name, allowed, default):
            if isinstance(value, str):
                return value
            if isinstance(value, dict):
                text = " ".join(
                    str(v) for v in value.values() if isinstance(v, str)
                ).lower()
                if field_name == "plausible_values_handling":
                    if "plausible" in text or "pv" in text:
                        return "not_reported"
                    if "no" in text or "not" in text and "pv" not in text:
                        return "not_applicable"
                if field_name == "missing_data_handling":
                    if "imput" in text or "mice" in text or "miss" in text:
                        return "multiple_imputation"
                    if "listwise" in text or "complete case" in text or "complete-case" in text:
                        return "listwise_deletion"
                    return "not_reported"
            return default

        DATA_KEYS = (
            "survey_design",
            "plausible_values_handling",
            "missing_data_handling",
            "sample_details",
            "ml_techniques",
            "confounders_identified",
            "outcome_summary",
            "research_design_type",
        )

        if not isinstance(parsed_data.get("data"), dict):
            parsed_data["data"] = {}
        data = parsed_data["data"]

        # Legacy flat JSON → nest under data
        for k in DATA_KEYS:
            if k in parsed_data:
                if k not in data:
                    data[k] = parsed_data.pop(k)
                else:
                    parsed_data.pop(k, None)

        for key in list(parsed_data.keys()):
            if key not in ("metadata", "data"):
                parsed_data.pop(key, None)

        for key in list(data.keys()):
            if key not in DATA_KEYS:
                data.pop(key, None)

        ml = data.get("ml_techniques")
        if isinstance(ml, dict):
            for legacy in ("feature_selection", "baseline_model", "xai_method"):
                ml.pop(legacy, None)
            primary = ml.get("primary")
            if primary is None or (isinstance(primary, str) and primary in INVALID_STR):
                ml["primary"] = None
            elif isinstance(primary, list):
                ml["primary"] = None
            if isinstance(ml.get("all_techniques"), list):
                ml["all_techniques"] = [
                    t for t in ml["all_techniques"]
                    if isinstance(t, str) and t not in INVALID_STR
                ]
            elif isinstance(ml.get("all_techniques"), str):
                ml["all_techniques"] = [ml["all_techniques"]]
            elif ml.get("all_techniques") is None:
                ml["all_techniques"] = []

        sd = data.get("sample_details")
        if isinstance(sd, dict):
            countries = sd.get("countries")
            if isinstance(countries, list):
                cleaned = []
                for c in countries:
                    if not isinstance(c, dict):
                        continue
                    code = c.get("country_code")
                    if not code or not isinstance(code, str):
                        continue
                    n = c.get("n_students")
                    if not isinstance(n, int):
                        c["n_students"] = None
                    cleaned.append(c)
                sd["countries"] = cleaned

        meta = parsed_data.get("metadata")
        if isinstance(meta, dict):
            for legacy in (
                "extraction_timestamp",
                "extraction_cost_usd",
                "prompt_tokens",
                "completion_tokens",
            ):
                meta.pop(legacy, None)
            for field in ("doi", "venue", "title"):
                if meta.get(field) in INVALID_STR:
                    meta[field] = None
            if not isinstance(meta.get("authors"), list):
                meta["authors"] = []

        data["plausible_values_handling"] = _normalize_literal(
            data.get("plausible_values_handling"),
            "plausible_values_handling",
            {
                "rubin_rules", "single_pv", "average_pv", "mitml",
                "not_applicable", "not_reported"
            },
            "not_reported",
        )
        data["missing_data_handling"] = _normalize_literal(
            data.get("missing_data_handling"),
            "missing_data_handling",
            {
                "listwise_deletion", "pairwise_deletion", "mean_imputation",
                "multiple_imputation", "not_reported"
            },
            "not_reported",
        )

        outcome = data.get("outcome_summary")
        if isinstance(outcome, dict):
            if isinstance(outcome.get("summary"), str):
                data["outcome_summary"] = outcome["summary"]
            else:
                data["outcome_summary"] = " ".join(
                    str(v) for v in outcome.values() if isinstance(v, str)
                )

        conf = data.get("confounders_identified")
        if not isinstance(conf, list):
            data["confounders_identified"] = []
        else:
            data["confounders_identified"] = [
                c for c in conf if isinstance(c, str) and c not in INVALID_STR
            ]

        return parsed_data

    def extract(self, processed: "ProcessedPDF") -> ExtractionResult:
        first_error = processed.parse_errors[0] if processed.parse_errors else None
        if not processed.extraction_text:
            return ExtractionResult(
                file_name=processed.file_name,
                success=False,
                extraction=None,
                input_tokens=0,
                output_tokens=0,
                cost_usd=0.0,
                duration_seconds=0.0,
                error=first_error or "Empty extracted text",
            )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": self._build_user_message(processed)},
        ]

        last_error = None
        for attempt in range(self.max_retries):
            start = time.perf_counter()
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.0,
                    response_format={"type": "json_object"},
                )
                duration = time.perf_counter() - start
                content = response.choices[0].message.content
                parsed_data = json.loads(content) if content else None

                if parsed_data is None:
                    last_error = "Model returned empty response"
                    break

                # Always inject correct file_name
                if isinstance(parsed_data.get("metadata"), dict):
                    parsed_data["metadata"]["file_name"] = processed.file_name

                # Sanitize before Pydantic validation
                parsed_data = self._sanitize(parsed_data)

                try:
                    extraction = ILSAArticleMetadata.model_validate(parsed_data)
                except ValidationError as e:
                    last_error = f"Schema validation failed: {e}"
                    logger.warning(
                        f"Validation error on {processed.file_name} "
                        f"attempt {attempt + 1}: {e}"
                    )
                    continue

                usage = response.usage
                cost = self._calculate_cost(usage.prompt_tokens, usage.completion_tokens)
                return ExtractionResult(
                    file_name=processed.file_name,
                    success=True,
                    extraction=extraction,
                    input_tokens=usage.prompt_tokens,
                    output_tokens=usage.completion_tokens,
                    cost_usd=cost,
                    duration_seconds=duration,
                )

            except RateLimitError as e:
                wait = self.base_delay * (2 ** attempt)
                logger.warning(
                    f"Rate limit on {processed.file_name}, "
                    f"retry {attempt + 1}/{self.max_retries} in {wait:.1f}s"
                )
                time.sleep(wait)
                last_error = f"Rate limit: {e}"

            except APITimeoutError as e:
                wait = self.base_delay * (2 ** attempt)
                logger.warning(
                    f"Timeout on {processed.file_name}, retry {attempt + 1} in {wait:.1f}s"
                )
                time.sleep(wait)
                last_error = f"Timeout: {e}"

            except APIError as e:
                last_error = f"API error: {e}"
                logger.error(f"API error on {processed.file_name}: {e}")
                break

            except Exception as e:
                last_error = f"Unexpected: {e}"
                logger.exception(f"Unexpected error on {processed.file_name}")
                break

        return ExtractionResult(
            file_name=processed.file_name,
            success=False,
            extraction=None,
            input_tokens=0,
            output_tokens=0,
            cost_usd=0.0,
            duration_seconds=0.0,
            error=last_error,
        )
