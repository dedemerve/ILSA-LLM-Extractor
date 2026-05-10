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

SYSTEM_PROMPT = """You are an expert research analyst specializing in International Large-Scale Assessments (ILSA: PISA, TIMSS, PIRLS, TALIS, ICILS, ICCS, PIAAC) and the application of Artificial Intelligence and Machine Learning to educational data.

Your task is to extract a structured information sheet from an academic article, focusing on:
1. Bibliographic identification (title, authors, year, DOI/URL)
2. Which ILSA dataset, cycle, countries, and target population were studied
3. Methodological rigor: sampling weights, missing data handling, multilevel hierarchical structure
4. ML algorithms used, performance metrics, and Explainable AI (XAI) methods
5. Substantive findings about education dynamics
6. Concrete policy recommendations for decision-makers, especially around equity

EXTRACTION RULES:
- Extract ONLY information explicitly stated in the article. Use null for any field where the article is silent.
- For 'is_genuine_ilsa_ai_study': set False if the article merely cites ILSA in passing or uses no ML/AI methods.
- Use canonical algorithm names (Random Forest, XGBoost, Logistic Regression).
- Policy recommendations must be CONCRETE and ACTIONABLE: specify WHO should do WHAT.
- For 'extraction_confidence': use 'low' if the PDF text is OCR-noisy or critical methodological details are missing.

OUTPUT: Return a single JSON object conforming exactly to the ILSAArticleMetadata schema."""


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
            title_hint = f"\nEXTRACTED_TITLE: {processed.metadata['extracted_title']}\n"
        return [
            {
                "type": "text",
                "text": (
                    f"FILE: {processed.file_name}\n"
                    f"SOURCE: {processed.source_database}\n"
                    f"SECTIONS DETECTED: {sections_label}\n"
                    f"{title_hint}"
                    f"--- BEGIN ARTICLE TEXT ---\n\n"
                    f"{processed.extraction_text}\n\n"
                    f"--- END ARTICLE TEXT ---\n\n"
                    "Follow the five steps in the system prompt. "
                    "Return valid JSON only — no markdown fences, no preamble."
                ),
            }
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

        # ml_techniques cleanup
        ml = parsed_data.get("ml_techniques")
        if isinstance(ml, dict):
            # primary must be a real string or null
            if ml.get("primary") in INVALID_STR or ml.get("primary") is None:
                ml["primary"] = None
            # all_techniques: remove sentinel strings
            if isinstance(ml.get("all_techniques"), list):
                ml["all_techniques"] = [
                    t for t in ml["all_techniques"]
                    if isinstance(t, str) and t not in INVALID_STR
                ]
            # optional string fields
            for field in ("feature_selection", "baseline_model", "xai_method"):
                if ml.get(field) in INVALID_STR:
                    ml[field] = None

        # sample_details.countries cleanup
        sd = parsed_data.get("sample_details")
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
                    # n_students must be int or null
                    n = c.get("n_students")
                    if not isinstance(n, int):
                        c["n_students"] = None
                    cleaned.append(c)
                sd["countries"] = cleaned

        # metadata optional string fields
        meta = parsed_data.get("metadata")
        if isinstance(meta, dict):
            for field in ("doi", "venue", "title"):
                if meta.get(field) in INVALID_STR:
                    meta[field] = None
            # authors must be a list
            if not isinstance(meta.get("authors"), list):
                meta["authors"] = []

        # confounders must be a list of strings
        conf = parsed_data.get("confounders_identified")
        if not isinstance(conf, list):
            parsed_data["confounders_identified"] = []
        else:
            parsed_data["confounders_identified"] = [
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
