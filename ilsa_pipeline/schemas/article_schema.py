from pydantic import BaseModel, Field
from typing import Optional, List


class ILSAContext(BaseModel):
    primary_ilsa: Optional[str] = Field(
        default=None,
        description="Primary ILSA dataset (e.g. 'PISA', 'TIMSS', 'PIRLS', 'TALIS', 'ICILS', 'ICCS', 'PIAAC')."
    )
    primary_year: Optional[int] = Field(
        default=None,
        description="Assessment cycle year of the primary dataset (e.g. 2018)."
    )
    all_ilsas: List[str] = Field(
        default_factory=list,
        description="All ILSA datasets used in the study."
    )
    all_years: List[int] = Field(
        default_factory=list,
        description="All assessment years used."
    )
    is_longitudinal: Optional[bool] = Field(
        default=None,
        description="True if tracking same individuals OR comparing multiple cycles."
    )
    longitudinal_waves: Optional[List[str]] = Field(
        default=None,
        description="List of longitudinal waves (e.g. ['PISA 2015', 'PISA 2018'])."
    )


class MLTechniques(BaseModel):
    primary: Optional[str] = Field(
        default=None,
        description="Primary ML algorithm used (e.g. 'XGBoost')."
    )
    all_techniques: List[str] = Field(
        default_factory=list,
        description="All ML algorithms compared or mentioned."
    )
    feature_selection: Optional[str] = Field(
        default=None,
        description="Feature selection method if used (e.g. 'LASSO', 'Elastic Net', 'RFE')."
    )
    baseline_model: Optional[str] = Field(
        default=None,
        description="Baseline model for comparison (e.g. 'Linear Regression')."
    )
    xai_method: Optional[str] = Field(
        default=None,
        description="XAI method used (e.g. 'SHAP', 'LIME', 'Permutation Importance')."
    )


class Metadata(BaseModel):
    file_name: str = Field(
        description="Source PDF filename, used as the primary key across all tables."
    )
    title: str = Field(
        description="Full title of the article exactly as it appears in the document."
    )
    authors: List[str] = Field(
        default_factory=list,
        description="Ordered list of author full names as they appear on the paper."
    )
    year: Optional[int] = Field(
        default=None,
        description="Publication year as a four-digit integer (e.g. 2023)."
    )
    doi: Optional[str] = Field(
        default=None,
        description="DOI string or URL. Include the full URL if only a URL is present."
    )
    venue: Optional[str] = Field(
        default=None,
        description="Journal name, conference name, or repository (e.g. 'Computers & Education', 'NeurIPS', 'arXiv')."
    )
    publication_type: Optional[str] = Field(
        default=None,
        description="One of: 'journal', 'conference', 'preprint', 'thesis'."
    )
    open_access: Optional[bool] = Field(
        default=None,
        description="True if the paper is openly available without a paywall."
    )
    source_category: Optional[str] = Field(
        default=None,
        description="One of: 'technical_report', 'review_article', 'methodology_paper', 'peer_reviewed_research'."
    )


class ILSAArticleMetadata(BaseModel):
    metadata: Metadata = Field(
        description="Bibliographic identification fields."
    )
    ilsa_context: Optional[ILSAContext] = Field(
        default=None,
        description="ILSA dataset context."
    )
    sample_size: Optional[int] = Field(
        default=None,
        description="Total number of observations used in the analysis."
    )
    research_design_type: Optional[str] = Field(
        default=None,
        description="One of: 'predictive', 'causal_observational', 'causal_experimental', 'exploratory'."
    )
    ml_techniques: Optional[MLTechniques] = Field(
        default=None,
        description="ML techniques and methodology."
    )
    survey_weights_used: Optional[str] = Field(
        default=None,
        description="'true', 'false', or 'not_reported'."
    )
    confounders_identified: List[str] = Field(
        default_factory=list,
        description="Variables explicitly controlled for as covariates."
    )
    outcome_summary: Optional[str] = Field(
        default=None,
        description="2-4 sentence summary of main findings."
    )
