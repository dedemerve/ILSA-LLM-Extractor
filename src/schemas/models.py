from typing import List, Optional, Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator


class Metadata(BaseModel):
    """Bibliographic and extraction provenance fields."""

    model_config = ConfigDict(extra="forbid")

    file_name: str = Field(
        description="Source PDF filename; primary key across all tables."
    )
    title: Optional[str] = Field(
        default=None,
        description="Full article title as it appears in the document."
    )
    authors: Optional[List[str]] = Field(
        default=None,
        description="Ordered list of author full names."
    )
    year: Optional[int] = Field(
        default=None,
        description="Four-digit publication year."
    )
    doi: Optional[str] = Field(
        default=None,
        description="DOI without URL prefix (e.g. '10.1016/j.foo.2020.01.001')."
    )
    venue: Optional[str] = Field(
        default=None,
        description="Journal, conference, or repository name."
    )
    publication_type: Optional[str] = Field(
        default=None,
        description="One of: 'journal', 'conference', 'book_chapter', 'preprint', 'report', 'thesis'."
    )
    open_access: Optional[bool] = Field(
        default=None,
        description="True if freely accessible without paywall; null if unknown."
    )
    source_category: Optional[str] = Field(
        default=None,
        description="One of: 'technical_report', 'review_article', 'methodology_paper', 'peer_reviewed_research'."
    )
    extraction_timestamp: Optional[str] = Field(
        default=None,
        description="ISO-8601 timestamp of when this record was extracted."
    )
    extraction_cost_usd: Optional[float] = Field(
        default=None,
        description="OpenAI API cost for this extraction in USD."
    )
    prompt_tokens: Optional[int] = Field(
        default=None,
        description="Input token count for this extraction."
    )
    completion_tokens: Optional[int] = Field(
        default=None,
        description="Output token count for this extraction."
    )

    @field_validator("doi", mode="before")
    @classmethod
    def strip_doi_prefix(cls, v: Optional[str]) -> Optional[str]:
        if isinstance(v, str):
            for prefix in (
                "https://doi.org/",
                "http://doi.org/",
                "https://dx.doi.org/",
                "http://dx.doi.org/",
            ):
                if v.startswith(prefix):
                    return v[len(prefix):]
        return v


class SurveyDesign(BaseModel):
    """Survey design and weighting methodology."""

    model_config = ConfigDict(extra="forbid")

    student_weights_used: Optional[bool] = Field(
        default=None,
        description="True if student/sampling weights (e.g. W_FSTUWT) were applied."
    )
    replicate_weights_used: Optional[bool] = Field(
        default=None,
        description="True if replicate weights (BRR, Fay) or jackknife were used."
    )
    weight_variable_name: Optional[str] = Field(
        default=None,
        description="Name of weight variable if mentioned (e.g. 'W_FSTUWT', 'TOTWGT')."
    )


class CountrySample(BaseModel):
    """Sample size by country."""

    model_config = ConfigDict(extra="forbid")

    country_code: str = Field(
        description="ISO 3166-1 alpha-3 country code (e.g. 'ESP', 'USA')."
    )
    n_students: Optional[int] = Field(
        default=None,
        description="Number of students from this country in the analytic sample."
    )


class SampleDetails(BaseModel):
    """Detailed sample composition."""

    model_config = ConfigDict(extra="forbid")

    total_students: Optional[int] = Field(
        default=None,
        description="Total number of students in the analytic sample."
    )
    countries: List[CountrySample] = Field(
        default_factory=list,
        description="Breakdown of students by country."
    )


class MLTechniques(BaseModel):
    """ML algorithms and methodological components."""

    model_config = ConfigDict(extra="forbid")

    primary: Optional[str] = Field(
        default=None,
        description="Primary/best-performing ML algorithm (e.g. 'XGBoost', 'Random Forest')."
    )
    all_techniques: List[str] = Field(
        description="All ML algorithms evaluated (NOT preprocessing/stats methods)."
    )
    feature_selection: Optional[str] = Field(
        default=None,
        description="Feature selection method (e.g. 'LASSO', 'Elastic Net', 'RFE')."
    )
    baseline_model: Optional[str] = Field(
        default=None,
        description="Baseline/comparison model (e.g. 'Linear Regression')."
    )
    xai_method: Optional[str] = Field(
        default=None,
        description="Explainability method (e.g. 'SHAP', 'LIME', 'Permutation Importance')."
    )


class ILSAArticleMetadata(BaseModel):
    """Top-level extraction record for ILSA ML papers (v4.1 schema)."""

    model_config = ConfigDict(extra="forbid")

    metadata: Metadata = Field(
        description="Bibliographic and extraction provenance fields."
    )
    survey_design: SurveyDesign = Field(
        description="Survey weighting and replicate design methodology."
    )
    plausible_values_handling: Literal[
        "rubin_rules", "single_pv", "average_pv",
        "mitml", "not_applicable", "not_reported"
    ] = Field(
        description="How plausible values (PVs) were handled in analysis."
    )
    missing_data_handling: Literal[
        "listwise_deletion", "pairwise_deletion",
        "mean_imputation", "multiple_imputation", "not_reported"
    ] = Field(
        description="How missing data was addressed."
    )
    sample_details: SampleDetails = Field(
        description="Total sample size and breakdown by country."
    )
    ml_techniques: MLTechniques = Field(
        description="ML algorithms and methodological components."
    )
    confounders_identified: List[str] = Field(
        default_factory=list,
        description="Sociodemographic variables explicitly controlled for."
    )
    outcome_summary: str = Field(
        description="2-4 sentence summary of key findings and model performance."
    )
    research_design_type: Optional[str] = Field(
        default=None,
        description="One of: 'predictive', 'causal_observational', 'causal_experimental', 'exploratory'."
    )
