"""
REFACTORED ILSA Paper Extractor (v2.0)
======================================
Nano-LLM optimized extraction with anti-hallucination constraints, explainability layers, and context optimization.

Key Features:
1. Anti-hallucination system prompt with explicit "do not infer" rules
2. Strict type handling via Optional fields to prevent validation crashes
3. Explainability layer: every extraction field has a corresponding _quote field
4. Document context optimization: limits to first 15 pages to prevent context degradation
5. Comprehensive error handling and graceful degradation

Usage:
    extractor = ILSAPaperExtractor(model="gemini-2.5-flash")
    result = extractor.extract_from_pdf(pdf_text, file_name="example.pdf")
    print(result.paper.reasoning_scratchpad)
"""

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

# For Gemini API
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("Warning: google-generativeai not installed. Install with: pip install google-generativeai")

# For OpenAI API (fallback)
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from pydantic import ValidationError

from src.schemas.article_schema_refactored import PaperExtraction, ExtractionResponse

logger = logging.getLogger(__name__)


# ============================================================================
# SYSTEM PROMPT: ANTI-HALLUCINATION & STRICT RULES
# ============================================================================

SYSTEM_PROMPT_V2 = """You are a specialist metadata extractor for academic literature on \
International Large-Scale Assessments (ILSA: PISA, TIMSS, PIRLS, TALIS, ICILS, ICCS, PIAAC) \
and machine learning / statistical modeling.

YOUR TASK:
Extract structured metadata from a single PDF document using the provided JSON schema.

================================================================================
CRITICAL RULES (READ FIRST):
================================================================================

RULE 1: DO NOT INFER OR HALLUCINATE
  - If a piece of information is NOT explicitly stated in the document, return None (null in JSON).
  - Do NOT guess, infer, or use default values.
  - If you cannot find a field, set it to None.
  - Return "not_reported" ONLY in Literal fields where it is an option.

RULE 2: EXPLAINABILITY VIA QUOTES
  - Every major extraction field (e.g., research_design_type, missing_data_handling) 
    has a corresponding _quote field.
  - Populate the _quote field with the EXACT VERBATIM SENTENCE(S) from the PDF 
    that justify your extraction.
  - If no quote can be found, set both the value and the _quote to None.
  - This allows downstream validation and error detection.

RULE 3: THINK FIRST, EXTRACT SECOND
  - ALWAYS start by populating "reasoning_scratchpad" with 1-2 sentences summarizing 
    the paper's methodology.
  - Example: "This paper uses PISA 2018 microdata with random forest to predict student achievement, 
    applying Rubin's rules for plausible values."
  - Do this BEFORE filling out other fields.

RULE 4: DOCUMENT VALIDITY CHECK
  - If the document appears to be unreadable OCR garbage, multiple unrelated papers, 
    or entirely off-topic, set is_valid_document = False and leave all extraction fields as None.
  - If the document is valid but has ambiguous or conflicting information, 
    set needs_human_review = True and populate what you can with confidence.

RULE 5: OPTIONAL FIELD HANDLING
  - All extraction fields use Optional[T] (can be null).
  - Never force a value to fit a non-optional type.
  - Return None instead of empty strings, empty lists, or placeholder values.

================================================================================
FIELD-BY-FIELD EXTRACTION DEFINITIONS:
================================================================================

BIBLIOGRAPHIC FIELDS (title, authors, publication_year, doi, venue, publication_type):
  - Extract exactly as they appear in the document.
  - Return None if not found.
  - publication_type: must match one of: journal, conference, report, book_chapter, thesis, preprint, other, not_reported.

ILSA CONTEXT (assessment_used, target_countries, total_sample_size):
  - assessment_used: Identify which ILSA program is used. If multiple, list as "multiple". 
    If none, "none". Values: PISA, TIMSS, PIRLS, TALIS, ICILS, ICCS, PIAAC, multiple, none, not_reported.
  - target_countries: List ISO 3166-1 alpha-3 codes (e.g., USA, GBR, JPN, DEU, FRA).
  - total_sample_size: Final analytic N (number of students/individuals). Return None if not stated.
  - ALWAYS include _quote fields for each of these.

RESEARCH DESIGN TYPE:
  This is CRITICAL. Classify based on the paper's explicit statements:
  
  "exploratory": Primarily descriptive analysis, EDA, correlation studies, factor analysis.
    Keywords: "explored", "descriptive", "examined the relationship", "associations".
  
  "predictive": Machine learning model built to forecast outcomes. No causal claims.
    Keywords: "predict", "forecasting", "classification model", "regression model", "machine learning".
  
  "causal_observational": Uses observational data with causal inference techniques 
    to estimate treatment effects. Includes propensity score matching, instrumental variables, 
    difference-in-differences, matching.
    Keywords: "causal", "treatment effect", "propensity score", "instrumental variable", "DID".
  
  "causal_experimental": Randomized controlled trial or experimental manipulation.
    Keywords: "randomized", "RCT", "randomly assigned", "experimental design".
  
  "not_reported": If the design is genuinely unclear or not stated.
  
  DO NOT INFER THE DESIGN. If the paper uses ML but doesn't use causal language → "predictive".
  Always include research_design_type_quote with the justifying excerpt.

MISSING DATA HANDLING:
  Describes how the paper handles missing values in its analytic sample.
  
  "multiple_imputation": MICE, Amelia, chained equations, multivariate normal imputation.
    Keywords: "multiple imputation", "MICE", "MI", "imputed datasets", "chained equations".
  
  "listwise_deletion": Only complete cases used. Rows with any missing value are dropped.
    Keywords: "complete cases", "listwise deletion", "casewise deletion", "excluded students with missing".
  
  "FIML": Full Information Maximum Likelihood (structural equation modeling).
    Keywords: "FIML", "full information maximum likelihood".
  
  "LASSO_imputation": LASSO/regularization-based imputation.
  
  "mean_imputation": Missing values replaced with variable mean.
    Keywords: "mean imputation", "replaced with the mean".
  
  "pairwise_deletion": Uses all available data per analysis.
    Keywords: "pairwise deletion", "available case analysis".
  
  "not_reported": Paper does not describe missing data handling.
    Use this if the paper is silent on how missing data was handled.
  
  CRITICAL: Do NOT infer. If missing data handling is not mentioned, return "not_reported".

PLAUSIBLE VALUES HANDLING:
  ILSA achievement scores are reported as 5-10 plausible values (PVs).
  This is SPECIFIC to PISA, TIMSS, and related assessments.
  
  "rubins_rules": Rubin's combining rules applied across all PVs. Separate analysis per PV, then pooled.
    Keywords: "Rubin's rules", "combining rules", "pooled across PVs", "each PV analyzed separately".
  
  "single_pv": Only one PV used (e.g., PV1).
    Keywords: "first plausible value", "PV1", "single PV".
  
  "average_pv": Mean of all PVs used as a point estimate.
    Keywords: "average of plausible values", "mean PV".
  
  "mitml": Multilevel multiple imputation for PVs using mitml package.
    Keywords: "mitml", "multilevel multiple imputation".
  
  "not_applicable": Study does NOT use ILSA achievement scores. Examples:
    - Country-level analysis (one row per country, no microdata)
    - TALIS teacher survey (no student achievement)
    - Process data / log files only
    - Outcome is attitudes, creative thinking index, or non-achievement score
    - Methodology or review paper (no primary analysis)
  
  "not_reported": Study uses individual-level achievement data but never describes PV handling.
    CRITICAL DISTINCTION: If paper uses PISA/TIMSS achievement scores but is silent on PVs → "not_reported".
    Only use "not_applicable" if the outcome is genuinely non-achievement.

STUDENT WEIGHTS:
  Does the paper apply sampling weights in its statistical analysis?
  
  true: Paper explicitly applies sampling/student weights.
    Keywords: "weighted analysis", "W_FSTUWT", "TOTWGT", "final student weight", "sampling weight", "design weight".
  
  false: Paper explicitly states weights were NOT used, OR uses only country-level aggregates.
  
  None: Paper uses individual microdata but never mentions weights (missing information).
  
  CRITICAL: Only return true if explicitly stated. Do NOT infer from "we used ILSA data".

ML TECHNIQUES:
  ml_techniques_primary: The single most prominent ML/statistical algorithm.
    Examples: "Random Forest", "XGBoost", "LSTM", "Logistic Regression", "Multilevel Linear Model".
    Return None if no algorithm is clearly stated. Do NOT infer.
  
  ml_techniques_all: List of ALL algorithms/statistical methods used.
    Include every model mentioned, even secondary ones.

CONFOUNDERS:
  List of confounding variables explicitly acknowledged or controlled for.
  Examples: ["socioeconomic status", "school funding", "teacher experience"].
  CRITICAL: Only list explicitly named confounders. Do NOT infer from context.
  Return None if confounders are not discussed.

QUALITY FLAGS:
  is_valid_document: False if OCR garbage, multiple papers, or entirely off-topic.
  needs_human_review: True if methodology is ambiguous, contradictory, or key fields 
    cannot be determined with confidence.

================================================================================
OUTPUT FORMAT:
================================================================================
Return ONLY valid JSON (no markdown, no code fences, no explanatory text).
Follow this structure:

{
  "paper": {
    "reasoning_scratchpad": "1-2 sentence summary of paper methodology",
    "is_valid_document": true,
    "needs_human_review": false,
    "file_name": "exact_filename.pdf",
    "title": "Full Title" or null,
    "authors": ["Author One", "Author Two"] or null,
    "publication_year": 2024 or null,
    "doi": "10.1234/example" or null,
    "venue": "Journal Name" or null,
    "publication_type": "journal" or null,
    "assessment_used": "PISA" or "multiple" or "none" or null,
    "assessment_used_quote": "verbatim excerpt" or null,
    "target_countries": ["USA", "GBR", "JPN"] or null,
    "target_countries_quote": "verbatim excerpt" or null,
    "total_sample_size": 12450 or null,
    "total_sample_size_quote": "N = 12,450 students" or null,
    "research_design_type": "exploratory" | "predictive" | "causal_observational" | "causal_experimental" | "not_reported" or null,
    "research_design_type_quote": "verbatim excerpt justifying design type" or null,
    "missing_data_handling": "multiple_imputation" | "listwise_deletion" | ... | "not_reported" or null,
    "missing_data_handling_quote": "verbatim excerpt" or null,
    "plausible_values_handling": "rubins_rules" | "single_pv" | ... | "not_reported" or null,
    "plausible_values_handling_quote": "verbatim excerpt" or null,
    "student_weights_used": true | false or null,
    "student_weights_quote": "verbatim excerpt" or null,
    "ml_techniques_primary": "Random Forest" or "XGBoost" or null,
    "ml_techniques_quote": "verbatim excerpt" or null,
    "ml_techniques_all": ["Random Forest", "SHAP", "Logistic Regression"] or null,
    "confounders_identified": ["variable1", "variable2"] or null,
    "confounders_quote": "verbatim excerpt" or null,
    "primary_outcome": "Student math achievement" or null,
    "key_finding_summary": "1-3 sentence summary of main finding" or null,
    "source_category": "peer_reviewed_research" or null,
    "extraction_quality_flags": ["flag1", "flag2"] or []
  }
}

================================================================================
FINAL CHECKLIST:
================================================================================
Before returning JSON:
1. ✓ reasoning_scratchpad is populated with 1-2 sentence summary
2. ✓ is_valid_document is set (check for OCR garbage, multiple papers)
3. ✓ needs_human_review is set (check for ambiguity, conflicting info)
4. ✓ All _quote fields have verbatim excerpts or None (no made-up quotes)
5. ✓ Optional fields are null, not empty strings or empty lists
6. ✓ Literal fields match allowed enum values exactly
7. ✓ No hallucinated data; only values explicitly found in the document
8. ✓ JSON is valid and parseable
"""


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def truncate_pdf_text_to_pages(pdf_text: str, max_pages: int = 15) -> str:
    """
    Truncate PDF text to the first N pages.
    
    Nano LLMs suffer from context degradation when processing massive documents.
    The first 15 pages typically contain: Abstract, Introduction, Methods, and partial Results.
    This omits lengthy reference lists, appendices, and supplementary materials.
    
    Args:
        pdf_text: Full extracted PDF text (typically from PyMuPDF or PyPDFLoader)
        max_pages: Maximum number of pages to retain (default 15)
    
    Returns:
        Truncated text containing approximately the first max_pages pages
    """
    # Split by common page break markers
    # PyMuPDF typically uses "\n\n" or "===" or form feed characters
    page_splits = re.split(r'\n\s*\n|\f|===', pdf_text, maxsplit=max_pages)
    
    truncated = '\n\n'.join(page_splits[:max_pages])
    
    # Sanity check: if result is suspiciously small, include more sections
    if len(truncated) < 2000:  # Less than 2000 chars seems too small
        truncated = '\n\n'.join(page_splits[:min(max_pages + 5, len(page_splits))])
    
    return truncated


def extract_json_from_response(response_text: str) -> Optional[dict]:
    """
    Extract JSON from LLM response, handling markdown code fences and partial responses.
    
    Args:
        response_text: Raw text response from the LLM
    
    Returns:
        Parsed JSON dict, or None if extraction fails
    """
    # Try to extract JSON from markdown code fences
    json_match = re.search(r'```(?:json)?\s*([\s\S]*?)```', response_text)
    if json_match:
        response_text = json_match.group(1)
    
    # Try to find JSON object directly
    json_match = re.search(r'\{[\s\S]*\}', response_text)
    if json_match:
        response_text = json_match.group(0)
    
    try:
        return json.loads(response_text)
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse JSON from response: {e}")
        return None


# ============================================================================
# MAIN EXTRACTOR CLASS
# ============================================================================

@dataclass
class ExtractionResult:
    """Dataclass holding extraction result and metadata."""
    success: bool
    paper: Optional[PaperExtraction] = None
    response: Optional[ExtractionResponse] = None
    error: Optional[str] = None
    raw_response: Optional[str] = None
    processing_notes: Optional[str] = None


class ILSAPaperExtractor:
    """
    Nano-LLM optimized extractor for ILSA academic papers.
    
    Supports both Gemini (primary) and OpenAI (fallback) models.
    Implements anti-hallucination constraints and explainability layer.
    """
    
    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
    ):
        """
        Initialize the extractor.
        
        Args:
            model: LLM model name (e.g., "gemini-2.5-flash", "gpt-4o-mini")
            api_key: API key for the model. If None, uses environment variables.
            temperature: LLM temperature (0.0 for deterministic, best for extraction)
        """
        self.model = model
        self.temperature = temperature
        self.api_key = api_key
        
        # Determine which API to use
        if "gemini" in model.lower():
            if not GEMINI_AVAILABLE:
                raise RuntimeError("Gemini API not installed. Install with: pip install google-generativeai")
            self.api_type = "gemini"
            if api_key:
                genai.configure(api_key=api_key)
        elif "gpt" in model.lower():
            if not OPENAI_AVAILABLE:
                raise RuntimeError("OpenAI API not installed. Install with: pip install openai")
            self.api_type = "openai"
            self.client = OpenAI(api_key=api_key) if api_key else OpenAI()
        else:
            raise ValueError(f"Unsupported model: {model}. Use 'gemini-*' or 'gpt-*'")
        
        logger.info(f"Initialized ILSAPaperExtractor with model: {model}")
    
    def extract_from_pdf(
        self,
        pdf_text: str,
        file_name: str,
        max_context_pages: int = 15,
    ) -> ExtractionResult:
        """
        Extract metadata from a PDF's text content.
        
        Args:
            pdf_text: Full text extracted from the PDF
            file_name: Original filename of the PDF
            max_context_pages: Maximum pages to include (default 15 for nano LLMs)
        
        Returns:
            ExtractionResult with parsed PaperExtraction or error details
        """
        processing_notes = []
        
        # Step 1: Validate input
        if not pdf_text or len(pdf_text.strip()) < 100:
            return ExtractionResult(
                success=False,
                error="PDF text is empty or too short",
                processing_notes="PDF produced insufficient text for extraction"
            )
        
        # Step 2: Truncate to first N pages (context optimization for nano models)
        original_length = len(pdf_text)
        pdf_text_truncated = truncate_pdf_text_to_pages(pdf_text, max_pages=max_context_pages)
        truncated_length = len(pdf_text_truncated)
        
        if truncated_length < original_length:
            processing_notes.append(
                f"Context truncated from {original_length} to {truncated_length} chars "
                f"(first {max_context_pages} pages)"
            )
        
        # Step 3: Build the extraction prompt
        user_prompt = f"""Extract metadata from this academic paper PDF.

FILE_NAME: {file_name}

PDF TEXT (first {max_context_pages} pages):
---
{pdf_text_truncated}
---

Return ONLY valid JSON matching the schema. No explanations, no markdown code fences.
Start with reasoning_scratchpad and proceed field by field.
If a field is not explicitly stated in the text, set it to null.
Never infer, guess, or hallucinate data.
Always include _quote fields with verbatim excerpts when available.
"""
        
        # Step 4: Call the LLM
        try:
            if self.api_type == "gemini":
                raw_response = self._call_gemini_api(user_prompt)
            else:
                raw_response = self._call_openai_api(user_prompt)
        except Exception as e:
            return ExtractionResult(
                success=False,
                error=f"LLM API call failed: {str(e)}",
                processing_notes="; ".join(processing_notes) if processing_notes else None
            )
        
        # Step 5: Parse the JSON response
        response_json = extract_json_from_response(raw_response)
        if not response_json:
            return ExtractionResult(
                success=False,
                error="Failed to parse JSON from LLM response",
                raw_response=raw_response[:500],  # Store first 500 chars for debugging
                processing_notes="; ".join(processing_notes) if processing_notes else None
            )
        
        # Step 6: Validate against Pydantic schema (with graceful error handling)
        try:
            paper_json = response_json.get("paper", response_json)
            # Add file_name if not present
            if "file_name" not in paper_json:
                paper_json["file_name"] = file_name
            
            paper = PaperExtraction(**paper_json)
            
            # Create full response
            extraction_response = ExtractionResponse(
                paper=paper,
                extraction_timestamp=datetime.utcnow().isoformat() + "Z",
                model_used=self.model,
                processing_notes="; ".join(processing_notes) if processing_notes else None
            )
            
            return ExtractionResult(
                success=True,
                paper=paper,
                response=extraction_response,
                raw_response=raw_response,
                processing_notes="; ".join(processing_notes) if processing_notes else None
            )
        
        except ValidationError as e:
            # Graceful degradation: extract what we can, flag for human review
            error_msg = f"Pydantic validation failed: {str(e)}"
            logger.warning(f"Validation error for {file_name}: {error_msg}")
            
            # Try to extract at least the reasoning_scratchpad and flags
            try:
                paper_data = response_json.get("paper", response_json)
                paper_data["file_name"] = file_name
                paper_data["needs_human_review"] = True  # Flag for manual review due to validation errors
                paper = PaperExtraction(**paper_data)
                return ExtractionResult(
                    success=True,
                    paper=paper,
                    processing_notes=f"Partial extraction with validation errors: {error_msg}; "
                                    + ("; ".join(processing_notes) if processing_notes else "")
                )
            except Exception:
                return ExtractionResult(
                    success=False,
                    error=error_msg,
                    raw_response=raw_response[:500],
                    processing_notes="; ".join(processing_notes) if processing_notes else None
                )
    
    def _call_gemini_api(self, user_prompt: str) -> str:
        """Call Gemini API and return raw response text."""
        model = genai.GenerativeModel(self.model)
        response = model.generate_content(
            [
                {"role": "user", "parts": [{"text": SYSTEM_PROMPT_V2}]},
                {"role": "user", "parts": [{"text": user_prompt}]}
            ],
            generation_config=genai.types.GenerationConfig(
                temperature=self.temperature,
                max_output_tokens=2048,
            ),
        )
        return response.text
    
    def _call_openai_api(self, user_prompt: str) -> str:
        """Call OpenAI API and return raw response text."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_V2},
                {"role": "user", "content": user_prompt}
            ],
            temperature=self.temperature,
            max_tokens=2048,
        )
        return response.choices[0].message.content


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    # Example usage
    import os
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    
    # Initialize extractor (uses GOOGLE_API_KEY or OPENAI_API_KEY from environment)
    extractor = ILSAPaperExtractor(model="gemini-2.5-flash")
    
    # Example: read a PDF and extract
    # pdf_text = open("example.pdf", "rb").read()  # In practice, use PyMuPDF or PyPDFLoader
    # result = extractor.extract_from_pdf(pdf_text, file_name="example.pdf")
    
    # if result.success:
    #     print(f"✓ Extraction successful")
    #     print(f"Reasoning: {result.paper.reasoning_scratchpad}")
    #     print(f"Research Design: {result.paper.research_design_type}")
    # else:
    #     print(f"✗ Extraction failed: {result.error}")
    #     if result.raw_response:
    #         print(f"Raw LLM response (first 500 chars):\n{result.raw_response}")
