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

COUNTRY_NAME_TO_ISO = {
    "turkey": "TUR", "usa": "USA", "united states": "USA",
    "germany": "DEU", "deutschland": "DEU", "france": "FRA",
    "japan": "JPN", "korea": "KOR", "south korea": "KOR",
    "china": "CHN", "brazil": "BRA", "finland": "FIN",
    "singapore": "SGP", "australia": "AUS", "canada": "CAN",
    "uk": "GBR", "united kingdom": "GBR", "england": "GBR",
    "spain": "ESP", "italy": "ITA", "netherlands": "NLD",
    "sweden": "SWE", "norway": "NOR", "denmark": "DNK",
    "israel": "ISR", "new zealand": "NZL", "ireland": "IRL",
    "austria": "AUT", "belgium": "BEL", "switzerland": "CHE",
    "portugal": "PRT", "poland": "POL", "czech republic": "CZE",
    "hungary": "HUN", "greece": "GRC", "romania": "ROU",
    "russia": "RUS", "thailand": "THA", "indonesia": "IDN",
    "malaysia": "MYS", "chile": "CHL", "mexico": "MEX",
    "colombia": "COL", "argentina": "ARG", "india": "IND",
    "south africa": "ZAF", "taiwan": "TWN", "hong kong": "HKG",
    "macao": "MAC", "macau": "MAC", "estonia": "EST",
    "latvia": "LVA", "lithuania": "LTU", "slovakia": "SVK",
    "slovenia": "SVN", "croatia": "HRV", "serbia": "SRB",
    "bulgaria": "BGR", "cyprus": "CYP", "malta": "MLT",
    "luxembourg": "LUX", "iceland": "ISL", "qatar": "QAT",
    "uae": "ARE", "saudi arabia": "SAU", "jordan": "JOR",
    "iran": "IRN", "egypt": "EGY", "morocco": "MAR",
    "tunisia": "TUN", "ghana": "GHA", "kenya": "KEN",
    "nigeria": "NGA", "pakistan": "PAK", "vietnam": "VNM",
    "philippines": "PHL", "peru": "PER", "uruguay": "URY",
    "costa rica": "CRI", "panama": "PAN",
}

MODEL_NAME = "gpt-5.4-nano"
PRICE_INPUT_PER_1M = 2.50
PRICE_OUTPUT_PER_1M = 10.00

SYSTEM_PROMPT = """You are an expert research analyst specializing in International \
Large-Scale Assessments (ILSA: PISA, TIMSS, PIRLS, TALIS, ICILS, ICCS, PIAAC) \
and the intersection of Machine Learning in educational data mining.

Your task is to extract a highly detailed, structured metadata and methodological \
sheet from an academic article. You must rigorously map academic jargon to the \
strict schema provided, and MINIMIZE null values through deep semantic search \
and expert domain inference.

═══════════════════════════════════════════════════════════════
CRITICAL EXTRACTION & INFERENCE RULES
═══════════════════════════════════════════════════════════════

1) STRICT ENUMERATIONS & CATEGORIES:
   - publication_type MUST be exactly one of:
     ['journal', 'conference', 'book_chapter', 'preprint', 'report', 'thesis'].
   - source_category MUST be exactly one of:
     ['technical_report', 'review_article', 'methodology_paper', 'peer_reviewed_research'].
   - research_design_type MUST be exactly one of:
     ['predictive', 'causal_observational', 'causal_experimental', 'exploratory'].
     Mapping: prediction/classification/regression → "predictive";
     causal forests, propensity scores, diff-in-diff, IV → "causal_observational";
     randomized experiment → "causal_experimental";
     clustering, topic modeling, EDA, data description → "exploratory".
   - plausible_values_handling MUST be exactly one of:
     ['rubin_rules', 'single_pv', 'average_pv', 'mitml', 'not_applicable', 'not_reported'].
     Synonym table:
       "Rubin's rules" / "Rubin combining rules" / "combined PV estimates" /
       "pooled across PVs"                                        → rubin_rules
       "first plausible value" / "PV1 only" / "single PV draw" /
       "one PV per student" / "separate analyses per PV"          → single_pv
       "averaged plausible values" / "mean of PVs" / "PV average" /
       "all five PVs averaged"                                    → average_pv
       "mitml" / "Mplus complex survey" / "multilevel MI"         → mitml
       TALIS/PIAAC without PVs, or DV is Likert/direct measure   → not_applicable
     ILSA domain default: PISA/TIMSS/PIRLS always ship PVs for achievement scores.
     If the paper models achievement and never mentions PV handling → average_pv.
   - missing_data_handling MUST be exactly one of:
     ['listwise_deletion', 'pairwise_deletion', 'mean_imputation',
      'multiple_imputation', 'not_reported'].
     Synonym table:
       "listwise deletion" / "complete case" / "excluded incomplete" → listwise_deletion
       "pairwise deletion" / "available case analysis"              → pairwise_deletion
       "mean substitution" / "mean replacement" / "imputed with mean" → mean_imputation
       "MICE" / "MI" / "missForest" / "FIML" / "EM algorithm" /
       "chained equations" / "hot-deck" / any ML-based imputation   → multiple_imputation

2) COUNTRY CODES (ISO 3166-1 alpha-3):
   - country_code MUST always be a 3-letter UPPERCASE ISO code (e.g. 'TUR', 'USA', \
     'DEU', 'GBR', 'FRA', 'JPN', 'KOR', 'CHN', 'BRA', 'FIN', 'SGP', 'AUS').
   - NEVER write full country names, 2-letter codes, or non-standard abbreviations.

3) ML vs. TRADITIONAL STATISTICS (critical for ml_techniques):
   - For 'all_techniques' and 'primary', extract ONLY Machine Learning / \
     predictive-modeling algorithms: XGBoost, Random Forest, SVM, Neural Network, \
     Decision Tree, LASSO, Ridge, Elastic Net, k-NN, Gradient Boosting, \
     Logistic Regression (when used for classification), Naive Bayes, LightGBM, \
     CatBoost, AdaBoost, Bagging, ANFIS, etc.
   - DO NOT include: PCA, factor analysis, t-tests, ANOVA, chi-square, basic \
     correlations, descriptive statistics, EFA/CFA, SEM, HLM/mixed-effects \
     (unless explicitly used as an ML baseline), or ESCS index computations.
   - Algorithm name mapping (use canonical short names):
     "gradient boosted trees" / "GBT" / "GBM"        → Gradient Boosting
     "random forests" / "RF"                          → Random Forest
     "ANN" / "MLP" / "deep learning"                  → Neural Network
     "SVM" / "SVC" / "SVR"                            → SVM
     "lasso" / "L1 regression"                        → LASSO
     "ridge" / "L2 regression"                        → Ridge Regression
     "elastic net" / "L1+L2"                          → Elastic Net
     "ANFIS" / "neuro-fuzzy"                          → ANFIS
     "bagging" / "bootstrap aggregation"              → Bagging

4) WEIGHTING & REPLICATE DESIGN LOGIC:
   - student_weights_used: set true if "student weights", "sampling weights", \
     "W_FSTUWT", "TOTWGT", "SCHWGT", "HOUWGT", "final weight", "senate weight", \
     "analysis weight", "probability weight", "design weight", or "weighted \
     estimation" appear anywhere.
   - replicate_weights_used: set true if "BRR", "balanced repeated replication", \
     "Fay's method", "jackknife", "replicate weights", "JK2", or "JRR" appear.
   - weight_variable_name: exact variable name string if mentioned (e.g. 'W_FSTUWT').
   - ILSA domain default: if a study uses ILSA micro-data and never discusses \
     weights → student_weights_used = false (omission = likely unweighted).
   - weight_fields_interpretation: FILL ONLY IF student_weights_used, \
     replicate_weights_used, AND weight_variable_name are ALL null. Write 3-4 \
     analytical sentences: what the manuscript says about sample design, why \
     weights might be missing (e.g. small convenience sample, secondary analysis \
     without original weights), and what exact wording would be needed to extract them. \
     IF ANY weight field is non-null, this field MUST BE null.

5) NULL FIELDS INTERPRETATION (THE FALLBACK):
   - null_fields_interpretation: trigger ONLY if the overall extraction is \
     extremely sparse — e.g. missing sample sizes, missing ML algorithms, missing \
     PV handling, multiple metadata fields null. Write a structured diagnostic note \
     (plain text) explaining WHY the paper lacks data (e.g. "This is a theoretical \
     review paper, hence no sample size or ML models are evaluated." or "The \
     manuscript is a meta-analysis without original ILSA micro-data analysis.").
   - If the record is reasonably dense (most fields filled), this MUST BE null.

6) EXHAUSTIVE DATA & METHODOLOGY SEARCH (NO LAZY EXTRACTIONS):
   - DataBlock and SurveyDesign are the MOST CRITICAL sections. You must \
     aggressively scan "Methodology", "Data", "Measures", "Analytical Strategy", \
     "Sample", "Participants", AND footnotes, table notes, and appendices.
   - EXTENDED WEIGHT SYNONYMS — also look for: "senate weights", "house weights", \
     "overall weights", "SENWGT", "MATWGT", "SCIWGT", "REAWGT", variables starting \
     with "W_" or ending in "WGT". For replicate weights also: "JRR", "jackknife \
     repeated replication", "Taylor series linearization".
   - INFERRING COMPLEX DESIGN — if the authors mention adjusting for "complex \
     survey design", "stratification", "clustering", or "multilevel weighting", \
     you MUST infer student_weights_used = true.
   - STRICT FAIL-SAFE ENFORCEMENT — if BOTH student_weights_used and \
     replicate_weights_used end up as false or null, AND weight_variable_name \
     is null, you ABSOLUTELY MUST fill weight_fields_interpretation with 3-4 \
     sentences explaining: (a) what the methodology section says about the data, \
     (b) why weight information is missing (e.g. "The authors focused solely on \
     the ML architecture without detailing data preparation or ILSA weighting"), \
     (c) what explicit wording or variable names would be needed to confirm \
     weights were used.
   - *** FATAL ERROR ***: returning false/null for all weight fields AND leaving \
     weight_fields_interpretation as null is a schema violation. You must always \
     provide either evidence of weighting OR an explanation of its absence.

7) AGGRESSIVE SAMPLE & COUNTRY EXTRACTION (sample_details):
   - total_students: NEVER default to null without an exhaustive search. Scan \
     "Method", "Participants", "Data", "Data Cleaning", and "Results" sections \
     for keywords: "N =", "n =", "final sample", "consisted of", "analytic \
     sample", "valid responses", "after removing", "after exclusion", \
     "remaining students", "total of". Check tables and figure captions too.
   - countries: identify ALL countries or economies analyzed. If the abstract \
     says "using PISA data from the USA", extract country_code = "USA". If a \
     table lists multiple countries, extract ALL of them with ISO 3166-1 alpha-3 \
     codes. Do not leave the list empty if the data source inherently implies \
     a country (e.g. "TIMSS 2019 data from Morocco" → [{"country_code":"MAR"}]).

8) LOGICAL DEDUCTION FOR ML TECHNIQUES (ml_techniques):
   - primary: DO NOT leave primary null if all_techniques is populated!
     a) If ONLY ONE algorithm is in all_techniques (e.g. ["LASSO"]), that \
        algorithm IS inherently the primary model — copy it to primary.
     b) If MULTIPLE algorithms are listed, scan "Results", "Abstract", or \
        "Conclusion" for: "performed best", "achieved the highest accuracy", \
        "outperformed", "best-performing model", "highest R²/AUC/F1". Assign \
        that winning model to primary.
     c) If the paper genuinely compares models without declaring a winner, \
        pick the one highlighted in the abstract or conclusion.
   - *** FATAL ERROR ***: primary left null while all_techniques has values \
     is a schema violation.

9) ENFORCING THE NULL INTERPRETATION FALLBACK:
   - If after exhaustive search total_students is still null, OR primary is \
     null while all_techniques is empty, you MUST trigger null_fields_interpretation. \
     Write 2-3 sentences diagnosing the omission (e.g. "The study is a scoping \
     review without an empirical sample" or "The authors listed LASSO and Random \
     Forest but did not report which model achieved the best metric.").
   - This rule complements Rule 5 — both may apply simultaneously.

10) ANTI-HALLUCINATION:
   - Never INVENT: DOIs, exact N, country codes, author names, weight variable \
     names, or algorithm names not present in the text.
   - Inference is allowed ONLY for categorical/boolean/enum fields where ILSA \
     domain knowledge provides a clear default (rules 1 and 4 above).
   - Numeric fields (total_students, n_students, year) MUST come from the text.

═══════════════════════════════════════════════════════════════
OUTPUT SCHEMA
═══════════════════════════════════════════════════════════════

Return a single JSON with exactly two top-level keys: metadata, data.

metadata fields: file_name, title, authors, year, doi, venue, publication_type,
  open_access, source_category.

data fields: survey_design, plausible_values_handling, missing_data_handling,
  sample_details, ml_techniques, confounders_identified, outcome_summary,
  research_design_type, null_fields_interpretation.

data.survey_design: student_weights_used, replicate_weights_used,
  weight_variable_name, weight_fields_interpretation.

data.sample_details: total_students, countries (each: country_code, n_students).

data.ml_techniques: primary, all_techniques.

Do not emit any other top-level or nested keys.
No markdown fences, no preamble — valid JSON only.

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
    _use_structured: bool | None = None

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

    @staticmethod
    def _post_process_model(extraction: ILSAArticleMetadata) -> None:
        """Post-process a structured-output extraction in place."""
        for c in extraction.data.sample_details.countries:
            code = c.country_code.strip()
            if len(code) != 3 or not code.isalpha():
                mapped = COUNTRY_NAME_TO_ISO.get(code.lower())
                if mapped:
                    c.country_code = mapped
            c.country_code = c.country_code.upper()

        ml = extraction.data.ml_techniques
        if ml.primary is None and len(ml.all_techniques) == 1:
            ml.primary = ml.all_techniques[0]

        sd = extraction.data.survey_design
        has_positive = (
            sd.student_weights_used is True
            or sd.replicate_weights_used is True
            or (sd.weight_variable_name and sd.weight_variable_name.strip())
        )
        if has_positive:
            sd.weight_fields_interpretation = None

    def _build_user_message(self, processed: "ProcessedPDF") -> list[dict]:
        sections_label = ", ".join(processed.sections.keys()) or "none"
        title_hint = ""
        if processed.metadata.get("extracted_title"):
            title_hint = (
                f"\nEXTRACTED_TITLE_HINT (use only if the body never states "
                f"a title; still do not contradict the PDF): "
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
            "Extract the structured JSON from the article above using the "
            "system prompt rules. Return valid JSON only."
        )

        interpretation_text = (
            "EXPERT INFERENCE CHECKLIST (same article; apply BEFORE finalising JSON):\n\n"

            "A) STRICT ENUMS — publication_type, source_category, research_design_type, "
            "plausible_values_handling, missing_data_handling must each be EXACTLY one "
            "of the allowed values listed in the system prompt. Use the synonym tables "
            "to map academic jargon. Never write free-text descriptions or new slugs. "
            "Examples: 'FIML' → multiple_imputation; 'complete cases' → listwise_deletion; "
            "'averaged across five PVs' → average_pv; 'PV1' → single_pv.\n\n"

            "B) COUNTRY CODES — every country_code must be ISO 3166-1 alpha-3 "
            "(3 uppercase letters). Never write full names or 2-letter codes.\n\n"

            "C) ML TECHNIQUES ONLY — all_techniques and primary must contain ONLY "
            "Machine Learning / predictive-modeling algorithms. DO NOT include: "
            "PCA, factor analysis, t-tests, ANOVA, chi-square, correlations, "
            "descriptive statistics, EFA/CFA, SEM, HLM (unless ML baseline), "
            "or ESCS computations. Set primary to the best-performing model; "
            "if ambiguous pick the one highlighted in the abstract.\n\n"

            "D) SURVEY WEIGHTS (CRITICAL — system rules 4 + 6): "
            "Aggressively scan methodology, data, footnotes, and table notes for "
            "weight terms (W_FSTUWT, TOTWGT, senate/house weights, BRR, jackknife, "
            "complex survey design, stratification, clustering). "
            "If found → set student_weights_used/replicate_weights_used = true. "
            "If ILSA micro-data is used but NO weight evidence exists → set false. "
            "*** FAIL-SAFE ***: when both student_weights_used and replicate_weights_used "
            "are false or null AND weight_variable_name is null, you MUST fill "
            "weight_fields_interpretation with 3-4 sentences explaining WHY weighting "
            "information is absent. Leaving all weight fields as false/null AND "
            "weight_fields_interpretation as null is a FATAL ERROR. "
            "ONLY set weight_fields_interpretation to null when student_weights_used=true "
            "OR replicate_weights_used=true (i.e. positive evidence of weighting exists).\n\n"

            "E) SAMPLE DETAILS (system rule 7) — exhaustively search Method, "
            "Participants, Data, Data Cleaning, and Results for total N. Look for "
            "'N =', 'final sample', 'analytic sample', 'valid responses', 'after "
            "removing/exclusion'. Check tables and figure captions. For countries, "
            "extract ALL ISO alpha-3 codes; never leave countries empty if the "
            "data source implies a country.\n\n"

            "F) ML PRIMARY (system rule 8) — *** FATAL ERROR *** to leave primary "
            "null while all_techniques has values. If only ONE algorithm is listed, "
            "it IS the primary. If multiple, scan Results/Abstract/Conclusion for "
            "'performed best', 'highest accuracy/R²/AUC', 'outperformed'. If truly "
            "ambiguous pick the one highlighted in the abstract.\n\n"

            "G) CONFOUNDERS — list variable names or short phrases controlled for "
            "as predictors (SES, gender, parental education, etc.). [] only if "
            "the paper truly names none.\n\n"

            "H) outcome_summary — 4-5 sentences of findings and performance metrics "
            "ONLY from the text. Do NOT put null-field commentary here.\n\n"

            "I) null_fields_interpretation — trigger if total_students is still "
            "null, or primary is null while all_techniques is empty, or extraction "
            "is extremely sparse. Write a diagnostic note explaining WHY. "
            "If the record is reasonably dense, this MUST be null.\n\n"

            "J) ANTI-HALLUCINATION — never invent DOIs, exact N, country codes, "
            "weight variable names, or algorithm names absent from the text. "
            "Inference applies ONLY to categorical/boolean/enum fields.\n"
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
    def _coerce_pv_literal(value) -> str:
        """Map free-text / invalid PV labels to schema literals."""
        allowed = frozenset({
            "rubin_rules", "single_pv", "average_pv", "mitml",
            "not_applicable", "not_reported",
        })
        if value in allowed:
            return value
        if not isinstance(value, str):
            return "not_reported"
        t = value.lower().replace("-", "_").replace(" ", "_")
        if "rubin" in t or "combined_estimates" in t:
            return "rubin_rules"
        if "mitml" in t or "mplus" in t:
            return "mitml"
        if "not_applicable" in t or "no_pv" in t or "no_pvs" in t:
            return "not_applicable"
        if (
            "first_plausible" in t
            or "single_pv" in t
            or "pv1_only" in t
            or "pv1" == t
            or "separate" in t and "plausible" in t
            or "per_pv" in t
            or "per_plausible" in t
            or "one_pv" in t
            or ("target" in t and "indicator" in t)
            or ("binary" in t and ("pv" in t or "plausible" in t))
        ):
            return "single_pv"
        if (
            "average" in t and "pv" in t
            or "all_plausible" in t
            or "across_pv" in t
            or "across_pvs" in t
            or "mean_pv" in t
        ):
            return "average_pv"
        if "plausible" in t or "_pv" in t or "pv_" in t:
            return "not_reported"
        return "not_reported"

    @staticmethod
    def _coerce_md_literal(value) -> str:
        """Map free-text / invalid missing-data labels to schema literals."""
        allowed = frozenset({
            "listwise_deletion", "pairwise_deletion", "mean_imputation",
            "multiple_imputation", "not_reported",
        })
        if value in allowed:
            return value
        if not isinstance(value, str):
            return "not_reported"
        t = value.lower().replace("-", "_").replace(" ", "_")
        if len(t) > 120 or "the_manuscript" in t or "the_paper" in t or "the_dataset" in t:
            return "not_reported"
        if "no_missing" in t or "without_missing" in t or "no_missing_data" in t:
            return "not_reported"
        if "pairwise" in t:
            return "pairwise_deletion"
        if "listwise" in t or "complete_case" in t or "complete case" in t:
            return "listwise_deletion"
        if "listwise" in t or "exclusion" in t and "missing" in t:
            return "listwise_deletion"
        if ("mean" in t and "imput" in t) or ("mean" in t and "substitut" in t) or ("mean" in t and "replac" in t):
            return "mean_imputation"
        if (
            "imput" in t
            or "mice" in t
            or "missforest" in t
            or "miss_forest" in t
            or "rf_based" in t
            or "fiml" in t
            or "full_information" in t
            or "maximum_likelihood" in t
            or "em_algorithm" in t
            or "hot_deck" in t
            or "hot deck" in t
            or "chained_equations" in t
            or "machine_learning" in t and "missing" in t
            or t == "imputation"
        ):
            return "multiple_imputation"
        return "not_reported"

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
            "null_fields_interpretation",
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

            if ml["primary"] is None and len(ml["all_techniques"]) == 1:
                ml["primary"] = ml["all_techniques"][0]

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
                    code = code.strip()
                    if len(code) != 3 or not code.isalpha():
                        mapped = COUNTRY_NAME_TO_ISO.get(code.lower())
                        if mapped:
                            code = mapped
                        else:
                            continue
                    c["country_code"] = code.upper()
                    n = c.get("n_students")
                    if not isinstance(n, int):
                        c["n_students"] = None
                    cleaned.append(c)
                sd["countries"] = cleaned

        sdw = data.get("survey_design")
        if isinstance(sdw, dict):
            wfi = sdw.get("weight_fields_interpretation")
            if isinstance(wfi, str) and wfi.strip() in INVALID_STR:
                sdw["weight_fields_interpretation"] = None
            wn = sdw.get("weight_variable_name")
            if isinstance(wn, str) and wn in INVALID_STR:
                sdw["weight_variable_name"] = None
                wn = None

            has_positive_weight_evidence = (
                sdw.get("student_weights_used") is True
                or sdw.get("replicate_weights_used") is True
                or (isinstance(wn, str) and wn.strip())
            )
            if has_positive_weight_evidence:
                sdw["weight_fields_interpretation"] = None

        nfi = data.get("null_fields_interpretation")
        if isinstance(nfi, str) and nfi.strip() in INVALID_STR:
            data["null_fields_interpretation"] = None

        VALID_PUB_TYPES = frozenset({
            "journal", "conference", "book_chapter", "preprint", "report", "thesis",
        })
        VALID_SOURCE_CATS = frozenset({
            "technical_report", "review_article", "methodology_paper",
            "peer_reviewed_research",
        })
        VALID_DESIGN_TYPES = frozenset({
            "predictive", "causal_observational", "causal_experimental", "exploratory",
        })

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
            pt = meta.get("publication_type")
            if isinstance(pt, str) and pt not in VALID_PUB_TYPES:
                normed = pt.lower().replace("-", "_").replace(" ", "_")
                if normed in VALID_PUB_TYPES:
                    meta["publication_type"] = normed
                else:
                    matched = None
                    for v in VALID_PUB_TYPES:
                        if v in normed or normed.startswith(v):
                            matched = v
                            break
                    meta["publication_type"] = matched

            sc = meta.get("source_category")
            if isinstance(sc, str) and sc not in VALID_SOURCE_CATS:
                normed = sc.lower().replace("-", "_").replace(" ", "_")
                if normed in VALID_SOURCE_CATS:
                    meta["source_category"] = normed
                else:
                    matched = None
                    for v in VALID_SOURCE_CATS:
                        if v in normed or normed.startswith(v):
                            matched = v
                            break
                    meta["source_category"] = matched

        rdt = data.get("research_design_type")
        if isinstance(rdt, str) and rdt not in VALID_DESIGN_TYPES:
            normed = rdt.lower().replace("-", "_").replace(" ", "_")
            if normed in VALID_DESIGN_TYPES:
                data["research_design_type"] = normed
            else:
                matched = None
                for v in VALID_DESIGN_TYPES:
                    if v in normed or normed.startswith(v):
                        matched = v
                        break
                data["research_design_type"] = matched

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

        pv_allowed = frozenset({
            "rubin_rules", "single_pv", "average_pv", "mitml",
            "not_applicable", "not_reported",
        })
        md_allowed = frozenset({
            "listwise_deletion", "pairwise_deletion", "mean_imputation",
            "multiple_imputation", "not_reported",
        })

        pv_raw = data.get("plausible_values_handling")
        if pv_raw not in pv_allowed:
            data["plausible_values_handling"] = GPTExtractor._coerce_pv_literal(pv_raw)

        md_raw = data.get("missing_data_handling")
        if md_raw not in md_allowed:
            data["missing_data_handling"] = GPTExtractor._coerce_md_literal(md_raw)

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
                # ── Path A: Structured Outputs (token-level schema enforcement) ──
                if GPTExtractor._use_structured is not False:
                    try:
                        response = self.client.beta.chat.completions.parse(
                            model=self.model,
                            messages=messages,
                            temperature=0.0,
                            response_format=ILSAArticleMetadata,
                        )
                        duration = time.perf_counter() - start
                        extraction = response.choices[0].message.parsed
                        if extraction is not None:
                            GPTExtractor._use_structured = True
                            extraction.metadata.file_name = processed.file_name
                            self._post_process_model(extraction)
                            usage = response.usage
                            cost = self._calculate_cost(
                                usage.prompt_tokens, usage.completion_tokens
                            )
                            return ExtractionResult(
                                file_name=processed.file_name,
                                success=True,
                                extraction=extraction,
                                input_tokens=usage.prompt_tokens,
                                output_tokens=usage.completion_tokens,
                                cost_usd=cost,
                                duration_seconds=duration,
                            )
                        last_error = "Model refused structured output"
                        continue
                    except (AttributeError, TypeError):
                        GPTExtractor._use_structured = False
                        logger.info(
                            "Structured outputs unavailable in SDK, "
                            "falling back to JSON mode"
                        )
                    except APIError as struct_err:
                        if GPTExtractor._use_structured is None:
                            GPTExtractor._use_structured = False
                            logger.info(
                                "Model does not support structured outputs "
                                f"({struct_err}), falling back to JSON mode"
                            )
                        else:
                            raise

                # ── Path B: JSON mode with _sanitize + manual validation ──
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

                if isinstance(parsed_data.get("metadata"), dict):
                    parsed_data["metadata"]["file_name"] = processed.file_name

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
                cost = self._calculate_cost(
                    usage.prompt_tokens, usage.completion_tokens
                )
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
                    f"Timeout on {processed.file_name}, "
                    f"retry {attempt + 1} in {wait:.1f}s"
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
