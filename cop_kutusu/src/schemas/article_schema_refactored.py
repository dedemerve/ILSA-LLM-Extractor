"""
REFACTORED ILSA Article Extraction Schema (v2.0)
===============================================
Implements strict Optional typing, quote-based explainability, and anti-hallucination constraints.

Key Changes:
- All extraction fields use Optional[T] to prevent validation crashes
- Every major field has a corresponding "_quote" field to provide verbatim evidence
- Added reasoning_scratchpad, is_valid_document, and needs_human_review for quality control
- Designed for nano LLM (Gemini 2.5-Flash, GPT-4o-mini) reliability
"""

from __future__ import annotations

from typing import Any, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class PaperExtraction(BaseModel):
    """
    Top-level extraction record for a single academic paper on ILSA and machine learning.
    
    This is a simplified, nano-LLM-friendly schema prioritized for:
    1. Strict Optional typing to prevent crashes on missing data
    2. Quote fields for explainability and error detection
    3. Quality flags (reasoning_scratchpad, needs_human_review, is_valid_document)
    4. Essential ILSA methodology tracking
    """

    model_config = ConfigDict(extra="forbid", json_schema_extra={
        "description": "Refactored schema for robust nano-LLM extraction with explainability"
    })

    # =========================================================================
    # SECTION 1: THINKING & QUALITY CONTROL (PROCESSED FIRST BY LLM)
    # =========================================================================
    
    reasoning_scratchpad: str = Field(
        description=(
            "LLM thinking block: 1-2 sentence summary of the paper's research design, "
            "primary outcome, and methodology BEFORE extracting structured fields. "
            "Must be populated FIRST in the JSON response. "
            "Example: 'This paper uses PISA 2018 microdata with random forest to predict "
            "student achievement, applying Rubin rules for plausible values.'"
        )
    )
    
    is_valid_document: bool = Field(
        default=True,
        description=(
            "False if the document is unreadable OCR garbage, entirely off-topic, "
            "or appears to contain multiple unrelated papers. "
            "If False, all extraction fields should be set to None. "
            "True if the document appears to be a coherent academic paper on education/ILSA."
        )
    )
    
    needs_human_review: bool = Field(
        default=False,
        description=(
            "True if: (1) multiple papers appear mixed in the text, (2) the PDF is heavily OCR-corrupted, "
            "(3) methodology is ambiguous or contradictory, or (4) you cannot confidently determine "
            "key fields (e.g., research_design_type, missing_data_handling). "
            "Set this to True to flag for manual inspection without returning null values."
        )
    )

    # =========================================================================
    # SECTION 2: BIBLIOGRAPHIC METADATA
    # =========================================================================
    
    file_name: str = Field(
        description="Original filename of the PDF as provided by the document loader"
    )
    
    title: Optional[str] = Field(
        default=None,
        description="Full title of the paper as printed in the document"
    )
    
    authors: Optional[List[str]] = Field(
        default=None,
        description="List of author names; null if no named authors found"
    )
    
    publication_year: Optional[int] = Field(
        default=None,
        description="Four-digit publication year; null if not determinable"
    )
    
    doi: Optional[str] = Field(
        default=None,
        description="DOI without URL prefix (e.g., '10.1016/j.foo.2020.01.001'); null if unavailable"
    )
    
    venue: Optional[str] = Field(
        default=None,
        description="Journal name, conference, or publisher; null if not stated"
    )
    
    publication_type: Optional[Literal[
        "journal", "conference", "report", "book_chapter", "thesis", "preprint", "other", "not_reported"
    ]] = Field(
        default=None,
        description="Type of publication; null if cannot be determined"
    )

    # =========================================================================
    # SECTION 3: ILSA CONTEXT (REQUIRED FIELDS)
    # =========================================================================
    
    assessment_used: Optional[Literal[
        "PISA", "TIMSS", "PIRLS", "TALIS", "ICILS", "ICCS", "PIAAC", "multiple", "none", "not_reported"
    ]] = Field(
        default=None,
        description="Primary ILSA assessment used; 'multiple' if >1; 'none' if no ILSA data"
    )
    
    assessment_used_quote: Optional[str] = Field(
        default=None,
        description="Verbatim sentence(s) from the PDF naming the ILSA assessment(s)"
    )
    
    target_countries: Optional[List[str]] = Field(
        default=None,
        description="ISO 3166-1 alpha-3 country codes (e.g., ['USA', 'GBR', 'JPN']); null if not specified"
    )
    
    target_countries_quote: Optional[str] = Field(
        default=None,
        description="Verbatim excerpt listing the countries analyzed"
    )
    
    total_sample_size: Optional[int] = Field(
        default=None,
        description="Final analytic sample size (number of students/individuals); null if not reported"
    )
    
    total_sample_size_quote: Optional[str] = Field(
        default=None,
        description="Verbatim excerpt stating the final sample size (e.g., 'N = 12,450 students')"
    )

    # =========================================================================
    # SECTION 4: RESEARCH DESIGN & METHODOLOGY (CORE ILSA TRACKING)
    # =========================================================================
    
    research_design_type: Optional[Literal[
        "exploratory", "predictive", "causal_observational", "causal_experimental", "not_reported"
    ]] = Field(
        default=None,
        description=(
            "Type of research design. "
            "'exploratory': primarily descriptive analysis, EDA, correlation studies. "
            "'predictive': machine learning model to predict outcomes, no causal claims. "
            "'causal_observational': uses observational data with causal inference techniques "
            "(propensity score matching, IV, DID) to estimate treatment effects. "
            "'causal_experimental': randomized controlled trial or experimental manipulation. "
            "'not_reported': design type not stated. "
            "MUST NOT infer; if unclear, return 'not_reported'."
        )
    )
    
    research_design_type_quote: Optional[str] = Field(
        default=None,
        description=(
            "Verbatim excerpt justifying the research_design_type classification. "
            "Example: 'We use a multi-level logistic regression to predict student achievement' "
            "→ predictive design. Return None if no explicit methodological statement found."
        )
    )
    
    missing_data_handling: Optional[Literal[
        "multiple_imputation", "listwise_deletion", "FIML", "LASSO_imputation", 
        "mean_imputation", "pairwise_deletion", "not_reported"
    ]] = Field(
        default=None,
        description=(
            "How the paper handles missing data in its analytic sample. "
            "'multiple_imputation': MICE, Amelia, chained equations, multivariate normal imputation. "
            "'listwise_deletion': only complete cases used. "
            "'FIML': Full Information Maximum Likelihood (structural equation modeling). "
            "'LASSO_imputation': LASSO/regularization-based imputation. "
            "'mean_imputation': replaced with variable mean. "
            "'pairwise_deletion': uses all available data per analysis. "
            "'not_reported': paper does not describe missing data handling. "
            "CRITICAL: Do NOT infer. If missing data is not mentioned, return 'not_reported'."
        )
    )
    
    missing_data_handling_quote: Optional[str] = Field(
        default=None,
        description=(
            "Verbatim excerpt describing the missing data strategy. "
            "Example: 'Missing values were imputed using multiple imputation by chained equations' "
            "or 'We retained only students with complete data across all variables.' "
            "Return None if no explicit statement found."
        )
    )
    
    plausible_values_handling: Optional[Literal[
        "rubins_rules", "single_pv", "average_pv", "mitml", "not_applicable", "not_reported"
    ]] = Field(
        default=None,
        description=(
            "How plausible values (PVs) from ILSA achievement scores are handled. "
            "'rubins_rules': Rubin's combining rules applied across all PVs; separate analysis per PV then pooled. "
            "'single_pv': only one PV used (e.g., PV1). "
            "'average_pv': mean of all PVs used as a point estimate. "
            "'mitml': multilevel multiple imputation framework for PVs. "
            "'not_applicable': study does NOT use ILSA achievement scores (e.g., country-level analysis, "
            "TALIS teacher survey, process data only). "
            "'not_reported': study uses achievement data but does not describe PV handling. "
            "CRITICAL RULE: If paper uses PISA/TIMSS achievement scores but never mentions PVs → 'not_reported', "
            "NOT 'not_applicable'. Only use 'not_applicable' if the outcome is non-achievement "
            "(e.g., attitudes, teaching quality, teacher beliefs)."
        )
    )
    
    plausible_values_handling_quote: Optional[str] = Field(
        default=None,
        description=(
            "Verbatim excerpt describing plausible value handling. "
            "Example: 'We applied Rubin's rules and analyzed each of the 10 plausible values separately' "
            "or 'We used the mean of all plausible values as the outcome variable.' "
            "Return None if no PV mention found (in which case plausible_values_handling = 'not_reported')."
        )
    )

    # =========================================================================
    # SECTION 5: SURVEY DESIGN & WEIGHTING
    # =========================================================================
    
    student_weights_used: Optional[bool] = Field(
        default=None,
        description=(
            "True if the paper explicitly applies sampling/student weights in statistical analysis. "
            "False if weights were explicitly NOT used or study uses only country-level aggregates. "
            "None if using individual microdata but weights are not mentioned. "
            "Keywords: 'weighted analysis', 'W_FSTUWT', 'final student weight', 'sampling weight', "
            "'design weight', 'survey weight'. "
            "Return True ONLY if explicitly stated; do not infer."
        )
    )
    
    student_weights_quote: Optional[str] = Field(
        default=None,
        description=(
            "Verbatim excerpt confirming or denying the use of student weights. "
            "Example: 'We applied the final student weight (W_FSTUWT) in all models' "
            "or 'Weights were not applied due to data aggregation.' "
            "Return None if weights are not mentioned."
        )
    )

    # =========================================================================
    # SECTION 6: MACHINE LEARNING & ANALYTICAL TECHNIQUES
    # =========================================================================
    
    ml_techniques_primary: Optional[str] = Field(
        default=None,
        description=(
            "Single most prominent ML/statistical algorithm used for prediction or modeling. "
            "Examples: 'Random Forest', 'XGBoost', 'LSTM', 'Logistic Regression', "
            "'Multilevel Linear Model', 'Structural Equation Model', 'BERT', 'ANFIS'. "
            "For descriptive studies with no modeling, return None. "
            "Do NOT infer; if no algorithm is clearly stated, return None (not 'not_reported')."
        )
    )
    
    ml_techniques_quote: Optional[str] = Field(
        default=None,
        description=(
            "Verbatim excerpt naming the primary ML technique. "
            "Example: 'We employed a gradient boosting ensemble (XGBoost)' "
            "or 'A multi-layer perceptron neural network with LSTM cells.' "
            "Return None if no primary technique is identified."
        )
    )
    
    ml_techniques_all: Optional[List[str]] = Field(
        default=None,
        description=(
            "List of ALL algorithms/statistical methods used, even if secondary. "
            "Examples: ['XGBoost', 'SHAP analysis', 'Logistic Regression baseline']. "
            "Return None if no algorithms mentioned; empty list [] if field is present but no techniques listed."
        )
    )

    # =========================================================================
    # SECTION 7: CONFOUNDERS & STUDY CHARACTERISTICS
    # =========================================================================
    
    confounders_identified: Optional[List[str]] = Field(
        default=None,
        description=(
            "List of confounding variables explicitly acknowledged or controlled for "
            "in the paper (e.g., ['socioeconomic status', 'school funding', 'teacher experience']). "
            "Return None if confounders are not discussed. "
            "CRITICAL: Only list explicitly named confounders, do NOT infer from context."
        )
    )
    
    confounders_quote: Optional[str] = Field(
        default=None,
        description=(
            "Verbatim excerpt listing or defining confounders. "
            "Example: 'We controlled for socioeconomic status, parental education, and prior achievement.' "
            "Return None if confounders are not named."
        )
    )

    # =========================================================================
    # SECTION 8: OUTCOME & KEY FINDINGS SUMMARY
    # =========================================================================
    
    primary_outcome: Optional[str] = Field(
        default=None,
        description=(
            "The main outcome variable being predicted or analyzed. "
            "Examples: 'Student math achievement', 'Graduation rate', 'Teacher effectiveness score'. "
            "Return None if outcome is not clearly defined."
        )
    )
    
    key_finding_summary: Optional[str] = Field(
        default=None,
        description=(
            "1-3 sentence summary of the main finding or result. "
            "Example: 'Random Forest achieved 78% accuracy in predicting low-performing students, "
            "outperforming logistic regression by 5%.' "
            "Return None if results are not available or unclear."
        )
    )

    # =========================================================================
    # SECTION 9: DATA SOURCE & PROCESSING METADATA
    # =========================================================================
    
    source_category: Optional[Literal[
        "peer_reviewed_research", "technical_report", "review_article", "methodology_paper", "other", "not_reported"
    ]] = Field(
        default=None,
        description=(
            "Document type. "
            "'peer_reviewed_research': empirical study in a journal or conference. "
            "'technical_report': OECD, IEA, or institutional report. "
            "'review_article': systematic review, meta-analysis, or narrative review. "
            "'methodology_paper': dataset description or methodological framework. "
            "'other': thesis, preprint, or other. "
            "'not_reported': cannot determine."
        )
    )
    
    extraction_quality_flags: Optional[List[str]] = Field(
        default=None,
        description=(
            "Flags indicating potential data quality issues. "
            "Possible values: ['low_confidence_no_methods_section', 'abstract_only', "
            "'ocr_quality_poor', 'multiple_papers_detected', 'conflicting_info', 'document_truncated']. "
            "Return empty list [] if no flags; None if not assessed."
        )
    )


class ExtractionResponse(BaseModel):
    """
    Wrapper for the full extraction response from the LLM.
    Includes the extracted paper data and metadata about the extraction process.
    """
    
    model_config = ConfigDict(extra="forbid")
    
    paper: PaperExtraction = Field(
        description="The extracted paper metadata and methodology"
    )
    
    extraction_timestamp: Optional[str] = Field(
        default=None,
        description="ISO 8601 timestamp of when extraction occurred"
    )
    
    model_used: Optional[str] = Field(
        default=None,
        description="Name of the LLM model that performed extraction (e.g., 'gemini-2.5-flash')"
    )
    
    processing_notes: Optional[str] = Field(
        default=None,
        description="Any additional notes from the extraction process (e.g., context truncation, parsing errors)"
    )
