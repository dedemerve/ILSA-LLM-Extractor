from typing import List, Optional, Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator


class MetadataBlock(BaseModel):
    """Bibliographic fields (no extraction provenance in JSON schema)."""

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
    publication_type: Optional[Literal[
        "journal", "conference", "book_chapter", "preprint", "report", "thesis"
    ]] = Field(
        default=None,
        description="Strict publication type categorization."
    )
    open_access: Optional[bool] = Field(
        default=None,
        description="True if freely accessible without paywall; null if unknown."
    )
    source_category: Optional[Literal[
        "technical_report", "review_article", "methodology_paper",
        "peer_reviewed_research"
    ]] = Field(
        default=None,
        description="Strict research type categorization."
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
    weight_fields_interpretation: str = Field(
        description=(
            "ALWAYS REQUIRED. Write 3-4 sentences detailing the data preparation, "
            "sample selection, and weighting strategy. Explain which dataset was used, "
            "how the data was cleaned or filtered, whether complex survey weights were "
            "applied (and which variable, e.g. W_FSTUWT), and if weights were ignored, "
            "explicitly state that and explain why (e.g. ML algorithms lack native "
            "weight support). This field must never be null."
        ),
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


class Confounder(BaseModel):
    """A single independent variable / predictor / feature used in the model.

    EXTRACTION RULES — enforced at the schema level:
    • ONE variable per object. NEVER combine multiple variables into one entry.
    • Extract EVERY variable mentioned in the study — missing one is a failure.
    • Do NOT invent ILSA codes — use 'N/A' when no official code is stated.
    """

    model_config = ConfigDict(extra="forbid")

    variable_code: str = Field(
        description=(
            "The official ILSA alphanumeric code exactly as written in the paper "
            "(e.g. 'ESCS', 'ST004Q01TA', 'BSBG11A', 'HOMEPOS', 'W_FSTUWT'). "
            "If the paper does NOT explicitly mention a standard code for this "
            "variable, set to 'N/A'. Do NOT invent or guess codes."
        ),
    )
    variable_name: str = Field(
        description=(
            "A concise, standardised English label (max 8 words). Use consistent "
            "naming across papers: 'Gender', 'Socioeconomic status (ESCS)', "
            "'Home possessions', 'Math self-efficacy', 'School type', "
            "'Parental education (mother)', 'ICT resources', etc. Remove "
            "academic jargon and long-winded descriptions."
        ),
    )
    category: Literal[
        "socioeconomic",
        "demographic",
        "student_attitude",
        "student_behavior",
        "teacher",
        "school",
        "ict",
        "curriculum",
        "parent_home",
        "process_data",
        "prior_achievement",
        "peer_effects",
        "system_level",
        "other",
    ] = Field(
        description=(
            "The domain category that BEST fits the variable. Favor specific "
            "categories over 'other'. Mapping guide: "
            "socioeconomic → ESCS, HOMEPOS, WEALTH, HISEI, parental education, "
            "books at home, cultural possessions, family resources; "
            "demographic → gender, age, immigration/migrant status, language at home, grade; "
            "student_attitude → self-efficacy, motivation, anxiety, enjoyment, "
            "belonging, self-concept, interest, value beliefs, intrinsic motivation; "
            "student_behavior → study time, homework time/frequency, absenteeism, "
            "learning strategies, reading habits, metacognition; "
            "teacher → qualifications, experience, professional development, "
            "teaching practices, job satisfaction, instructional strategies; "
            "school → school type (public/private), resources, class size, climate, "
            "safety, autonomy, leadership, location (urban/rural); "
            "ict → ICT resources (ICTRES), computer use, digital access, "
            "technology integration in lessons, internet availability; "
            "curriculum → curriculum type, instructional time (SMINS/TMINS), "
            "content coverage, assessment practices; "
            "parent_home → parental involvement/support (EMOSUPS), home environment, "
            "family structure, homework supervision; "
            "process_data → response time, action counts, time-to-first-action, "
            "VOTAT scores, action sequences, number of visits; "
            "prior_achievement → previous test scores, prior-year grades, "
            "achievement in other domains (e.g. reading score used as predictor "
            "for math), WLE/PV scores used as control variables; "
            "peer_effects → classroom disciplinary climate, peer bullying, "
            "class-average achievement, classroom composition; "
            "system_level → country-level GDP, education expenditure, tracking age, "
            "national policy variables, GINI coefficient, teacher-student ratio "
            "at system level; "
            "other → ONLY as a last resort when the variable absolutely does not "
            "fit any of the above categories."
        ),
    )


class MLTechniques(BaseModel):
    """ML algorithms and methodological components."""

    model_config = ConfigDict(extra="forbid")

    primary: Optional[str] = Field(
        default=None,
        description=(
            "Primary/best-performing ML algorithm (e.g. 'XGBoost', 'Random Forest'). "
            "MUST NOT be null if all_techniques has values — deduce the best model "
            "from results, or copy the only technique if just one is listed."
        ),
    )
    all_techniques: List[str] = Field(
        description="All ML algorithms evaluated (NOT preprocessing/stats methods)."
    )


class DataBlock(BaseModel):
    """Methodological and analytic extraction fields."""

    model_config = ConfigDict(extra="forbid")

    survey_design: SurveyDesign = Field(
        description="Survey weighting and replicate design methodology."
    )
    plausible_values_handling: Literal[
        "rubin_rules", "single_pv", "average_pv", "all_pv",
        "mitml", "wle", "irt_theta",
        "not_applicable", "not_reported"
    ] = Field(
        description="How plausible values (PVs) were handled in analysis."
    )
    missing_data_handling: Literal[
        "listwise_deletion", "pairwise_deletion",
        "mean_imputation", "single_imputation", "knn_imputation",
        "multiple_imputation", "not_reported"
    ] = Field(
        description=(
            "How missing data was addressed. Map missForest/RF-based imputation "
            "to single_imputation; kNN imputation to knn_imputation."
        ),
    )
    handling_not_reported_explanation: Optional[str] = Field(
        default=None,
        description=(
            "REQUIRED IF plausible_values_handling OR missing_data_handling is "
            "'not_reported' or 'not_applicable'. Write 2-3 sentences as a critical "
            "peer-reviewer diagnosing WHY the information is missing. Is it a "
            "reporting gap (authors failed to document their strategy), or is it "
            "the study's nature (e.g., only Likert-scale responses, no cognitive "
            "PVs)? Must be null when both PV and missing data handling are explicitly "
            "reported."
        ),
    )
    sample_details: SampleDetails = Field(
        description="Total sample size and breakdown by country."
    )
    ml_techniques: MLTechniques = Field(
        description="ML algorithms and methodological components."
    )
    confounders_identified: List[Confounder] = Field(
        default_factory=list,
        description=(
            "An EXHAUSTIVE list of ALL independent variables, predictors, features, "
            "and control variables used in the study. Each variable MUST be a "
            "separate Confounder object — NEVER combine multiple variables into one "
            "entry. If the study uses 20 features, output 20 objects. Missing a "
            "variable is a critical extraction failure. DO NOT leave empty if the "
            "study uses input features — scan methodology, variables, measures, "
            "features, and results sections exhaustively."
        ),
    )
    outcome_summary: str = Field(
        description=(
            "4-5 sentence summary (max ~120 words) of key findings and model "
            "performance, grounded only in the article text. Focus on empirical "
            "metrics, model comparisons, and policy-relevant conclusions."
        )
    )
    research_design_type: Optional[Literal[
        "predictive", "causal_observational", "causal_experimental", "exploratory"
    ]] = Field(
        default=None,
        description="Strict research design categorization."
    )
    null_fields_interpretation: Optional[str] = Field(
        default=None,
        description=(
            "REQUIRED when total_students is null, or primary ML model is null while "
            "all_techniques is empty, or the extraction is extremely sparse. 2-3 "
            "sentences diagnosing the omission (e.g. theoretical review with no "
            "empirical data, or authors listed models but never reported which "
            "performed best). Must be null when the record is reasonably dense."
        ),
    )


class ILSAArticleMetadata(BaseModel):
    """Top-level extraction record for ILSA ML papers (nested metadata + data)."""

    model_config = ConfigDict(extra="forbid")

    metadata: MetadataBlock = Field(
        description="Bibliographic identification fields."
    )
    data: DataBlock = Field(
        description="Survey design, sample, ML, and outcome fields."
    )
