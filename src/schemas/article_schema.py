from __future__ import annotations

import copy
import json
from typing import Any, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class BibliographicInfo(BaseModel):
    """Bibliographic metadata extracted from the article header and reference section."""

    model_config = ConfigDict(extra="forbid")

    title: str = Field(
        description="Full title of the article exactly as it appears in the source"
    )
    authors: List[str] = Field(
        description="Author full names; prefer 'Last, First' format"
    )
    publication_year: Optional[int] = Field(
        description="Four-digit publication year; null if not determinable"
    )
    doi: Optional[str] = Field(
        description="DOI without URL prefix (e.g. '10.1016/j.foo.2020.01.001'); null if unavailable"
    )
    url: Optional[str] = Field(
        description="Direct URL to the article; null if unavailable"
    )
    venue: Optional[str] = Field(
        description="Full name of the journal, conference, or book; null if not stated"
    )
    venue_type: Literal[
        "journal", "conference", "book_chapter", "preprint", "report", "thesis", "not_specified"
    ] = Field(description="Classification of the publication venue type")
    open_access: Optional[bool] = Field(
        description="True if freely accessible, False if paywalled, null if unknown"
    )

    @field_validator("doi", mode="before")
    @classmethod
    def normalize_doi(cls, v: Optional[str]) -> Optional[str]:
        """Strip common DOI URL prefixes to store a bare DOI string."""
        if isinstance(v, str):
            for prefix in (
                "https://doi.org/",
                "http://doi.org/",
                "http://dx.doi.org/",
                "https://dx.doi.org/",
            ):
                if v.startswith(prefix):
                    return v[len(prefix):]
        return v


class ILSAContext(BaseModel):
    """Details about which ILSA assessments, cycles, domains, and populations were studied."""

    model_config = ConfigDict(extra="forbid")

    assessments_used: List[
        Literal["PISA", "TIMSS", "PIRLS", "TALIS", "ICILS", "ICCS", "PIAAC", "other", "not_specified"]
    ] = Field(description="ILSA programs used in the study; include all that apply")
    assessment_cycles: List[str] = Field(
        description="Specific cycles referenced (e.g. 'PISA 2018', 'TIMSS 2019'); empty list if not specified"
    )
    target_domain: List[
        Literal[
            "mathematics",
            "reading",
            "science",
            "ICT_literacy",
            "civic_education",
            "collaborative_problem_solving",
            "creative_thinking",
            "financial_literacy",
            "global_competence",
            "teacher_practices",
            "adult_skills",
            "other",
            "not_specified",
        ]
    ] = Field(description="Cognitive or content domains analyzed in the study; include all that apply")
    countries_analyzed: List[str] = Field(
        description="Countries or economies in the analysis; ISO 3166-1 alpha-3 codes preferred"
    )
    country_count: Optional[int] = Field(
        description="Total number of countries/economies included; null if not reported"
    )
    target_population: Literal[
        "students_grade_4",
        "students_grade_8",
        "students_age_15",
        "students_other",
        "teachers",
        "school_principals",
        "adults",
        "mixed",
        "not_specified",
    ] = Field(description="Primary population targeted by the assessment(s) used")
    sample_size: Optional[int] = Field(
        description="Total individual observations in the analytic sample; null if not reported"
    )


class MethodologicalHandling(BaseModel):
    """How the study handled the complex sampling and measurement design of ILSA data."""

    model_config = ConfigDict(extra="forbid")

    weights_applied: Literal[
        "full_replicate_weights",
        "final_student_weight_only",
        "school_weights",
        "teacher_weights",
        "no_weights",
        "weights_mentioned_but_unclear",
        "not_applicable",
        "not_specified",
        "other",
    ] = Field(description="Sampling weight strategy used in the statistical analysis")
    plausible_values_handling: Literal[
        "all_PVs_with_Rubins_rules",
        "all_PVs_simple_average",
        "single_PV_used",
        "PVs_not_used",
        "not_applicable",
        "not_specified",
    ] = Field(description="How proficiency plausible values were handled")
    missing_data_strategy: List[
        Literal[
            "listwise_deletion",
            "pairwise_deletion",
            "mean_imputation",
            "multiple_imputation",
            "model_based_imputation",
            "indicator_variables",
            "complete_case_analysis",
            "not_addressed",
            "not_specified",
            "other",
        ]
    ] = Field(description="Missing data strategies used; include all that apply")
    multilevel_structure: Literal[
        "acknowledged_modeled",
        "acknowledged_not_modeled",
        "not_acknowledged",
        "not_applicable",
        "not_specified",
    ] = Field(
        description=(
            "Whether the hierarchical student-school-country structure was recognized and addressed"
        )
    )
    multilevel_handled_explicitly: bool = Field(
        description="True if a multilevel or hierarchical model (e.g. HLM, mixed-effects) was used"
    )


class MLMethodology(BaseModel):
    """Machine learning methods, architectures, and validation strategies used in the study."""

    model_config = ConfigDict(extra="forbid")

    task_type: List[
        Literal[
            "classification",
            "regression",
            "clustering",
            "anomaly_detection",
            "time_series_forecasting",
            "natural_language_processing",
            "recommendation",
            "feature_selection_only",
            "descriptive_only",
            "not_specified",
            "other",
        ]
    ] = Field(description="ML task types performed in the study; include all that apply")
    algorithms_used: List[str] = Field(
        description=(
            "Names of all ML/statistical algorithms used "
            "(e.g. 'Random Forest', 'XGBoost', 'BERT')"
        )
    )
    deep_learning_used: bool = Field(
        description="True if any deep learning architecture was employed in the study"
    )
    deep_learning_architectures: List[
        Literal[
            "MLP", "CNN", "RNN", "LSTM", "GRU", "Transformer", "BERT",
            "GAN", "Autoencoder", "GNN", "not_applicable", "other",
        ]
    ] = Field(
        description=(
            "Deep learning architectures used; set to ['not_applicable'] if deep_learning_used is false"
        )
    )
    feature_engineering: List[str] = Field(
        description=(
            "Feature engineering steps described (e.g. 'PCA', 'log transformation of SES'); "
            "empty list if none reported"
        )
    )
    train_test_strategy: Literal[
        "holdout",
        "k_fold_CV",
        "stratified_k_fold",
        "leave_one_out",
        "time_series_split",
        "country_stratified_split",
        "no_validation",
        "not_specified",
        "other",
    ] = Field(description="Primary strategy used to evaluate generalization performance")
    hyperparameter_tuning: Literal[
        "grid_search",
        "random_search",
        "bayesian_optimization",
        "manual",
        "default_parameters",
        "not_specified",
        "other",
    ] = Field(description="Hyperparameter optimization approach used")

    @model_validator(mode="after")
    def normalize_deep_learning_fields(self) -> MLMethodology:
        """Coerce deep_learning_architectures to ['not_applicable'] when deep_learning_used is False."""
        if not self.deep_learning_used:
            non_na = [a for a in self.deep_learning_architectures if a != "not_applicable"]
            if non_na:
                self.deep_learning_architectures = ["not_applicable"]
        return self


class PerformanceMetrics(BaseModel):
    """Quantitative evaluation metrics reported for the best or primary model."""

    model_config = ConfigDict(extra="forbid")

    classification_metrics: List[
        Literal[
            "accuracy", "precision", "recall", "F1", "AUC_ROC", "AUC_PR",
            "balanced_accuracy", "kappa", "MCC", "confusion_matrix",
            "not_applicable", "other",
        ]
    ] = Field(
        description=(
            "Classification metrics reported; set to ['not_applicable'] if not a classification task"
        )
    )
    regression_metrics: List[
        Literal[
            "RMSE", "MAE", "MAPE", "R_squared", "adjusted_R_squared",
            "MSE", "explained_variance", "not_applicable", "other",
        ]
    ] = Field(
        description=(
            "Regression metrics reported; set to ['not_applicable'] if not a regression task"
        )
    )
    best_model: Optional[str] = Field(
        description="Name of the best-performing model (e.g. 'XGBoost'); null if not identified"
    )
    best_score: Optional[float] = Field(
        description="Primary metric value for the best model; null if not reported"
    )
    benchmark_comparison: bool = Field(
        description="True if the study compared ML results against a non-ML baseline or prior benchmark"
    )


class XAIUsage(BaseModel):
    """Explainability and interpretability methods applied to the ML model outputs."""

    model_config = ConfigDict(extra="forbid")

    xai_used: bool = Field(
        description="True if any explainability or interpretability technique was applied"
    )
    xai_methods: List[
        Literal[
            "SHAP", "LIME", "PDP", "ICE", "permutation_importance", "feature_importance",
            "saliency_maps", "attention_visualization", "counterfactual_explanations",
            "rule_extraction", "not_applicable", "other",
        ]
    ] = Field(
        description="XAI methods used; set to ['not_applicable'] if xai_used is false"
    )
    global_vs_local: Literal[
        "global_only", "local_only", "both", "not_applicable", "not_specified"
    ] = Field(
        description="Scope of explanations provided; 'not_applicable' if xai_used is false"
    )
    top_features_identified: List[str] = Field(
        description=(
            "Most important features named by XAI analysis; "
            "empty list if xai_used is false or none reported"
        )
    )

    @model_validator(mode="after")
    def normalize_xai_fields(self) -> XAIUsage:
        """Coerce XAI-related fields to not_applicable / empty when xai_used is False."""
        if not self.xai_used:
            if not self.xai_methods or any(m != "not_applicable" for m in self.xai_methods):
                self.xai_methods = ["not_applicable"]
            if self.global_vs_local != "not_applicable":
                self.global_vs_local = "not_applicable"
            self.top_features_identified = []
        return self


class FindingsAndPolicy(BaseModel):
    """Core findings, equity dimensions addressed, and policy implications of the study."""

    model_config = ConfigDict(extra="forbid")

    key_findings: List[str] = Field(
        description="Key empirical findings; maximum 3 items, each stated in 1-2 sentences"
    )
    inequality_dimensions_addressed: List[
        Literal[
            "socioeconomic_status", "gender", "immigration_background", "language",
            "rural_urban", "school_type", "ethnicity", "disability",
            "not_addressed", "other",
        ]
    ] = Field(
        description="Equity dimensions explicitly analyzed; set to ['not_addressed'] if none"
    )
    policy_recommendations: List[str] = Field(
        description="Explicit policy recommendations by the authors; maximum 3 items, each 1-2 sentences"
    )
    target_decision_makers: List[
        Literal[
            "national_policymakers", "school_administrators", "teachers", "researchers",
            "international_organizations", "parents_students", "not_specified", "other",
        ]
    ] = Field(
        description="Intended audiences for the policy recommendations; include all that apply"
    )
    limitations_acknowledged: List[str] = Field(
        description="Limitations explicitly stated by the authors; maximum 3 items, each 1-2 sentences"
    )


class ArticleExtraction(BaseModel):
    """
    Top-level extraction record for a single ILSA-AI academic article.

    Combines bibliographic, contextual, methodological, and findings metadata
    into a single structured record for downstream analysis and forecasting pipelines.
    Designed for use with GPT-4o Structured Outputs API.
    """

    model_config = ConfigDict(extra="forbid")

    bibliographic_info: BibliographicInfo = Field(
        description="Bibliographic metadata: title, authors, venue, DOI"
    )
    ilsa_context: ILSAContext = Field(
        description="Which assessments, cycles, domains, and populations were studied"
    )
    methodological_handling: MethodologicalHandling = Field(
        description="How the study handled ILSA-specific design features: weights, PVs, multilevel structure"
    )
    ml_methodology: MLMethodology = Field(
        description="ML algorithms, architectures, and validation strategy used"
    )
    performance_metrics: PerformanceMetrics = Field(
        description="Quantitative evaluation results reported in the study"
    )
    xai_usage: XAIUsage = Field(
        description="Explainability techniques applied and their key findings"
    )
    findings_and_policy: FindingsAndPolicy = Field(
        description="Key results, equity dimensions analyzed, and policy implications"
    )
    is_genuine_ilsa_ai_study: bool = Field(
        description=(
            "True if the paper genuinely applies ML/AI to ILSA data; "
            "False if it is a false positive from keyword search"
        )
    )
    extraction_confidence: Literal["high", "medium", "low"] = Field(
        description=(
            "Confidence in extraction accuracy: "
            "high (clear full-text), medium (some ambiguity), low (abstract only or highly ambiguous)"
        )
    )
    source_database: Literal["wos", "scopus", "oecd", "iea"] = Field(
        description="Database from which the source PDF was obtained"
    )
    source_pdf_filename: str = Field(
        description="Filename of the source PDF as stored in data/raw_pdfs/"
    )


def resolve_schema_refs(schema: dict) -> dict:
    """
    Inline all $ref references in a JSON schema dict.

    Replaces every {"$ref": "#/$defs/X"} occurrence with the full inline
    definition of X from the $defs section, then removes the $defs key.
    Required for API endpoints that do not accept $ref notation (e.g. GPT-4o
    Structured Outputs strict mode).
    """
    schema = copy.deepcopy(schema)
    defs = schema.pop("$defs", {})

    def _resolve(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                ref_name = node["$ref"].split("/")[-1]
                return _resolve(copy.deepcopy(defs[ref_name]))
            return {k: _resolve(v) for k, v in node.items()}
        if isinstance(node, list):
            return [_resolve(item) for item in node]
        return node

    return _resolve(schema)


def _check_additional_properties(node: Any, path: str = "root") -> None:
    """
    Recursively assert that every object-type node has additionalProperties: false.

    Raises AssertionError with the offending JSON path if any object node is missing
    the constraint required by GPT-4o Structured Outputs.
    """
    if isinstance(node, dict):
        if node.get("type") == "object":
            assert node.get("additionalProperties") is False, (
                f"Missing 'additionalProperties: false' at path: {path}"
            )
        for key, value in node.items():
            _check_additional_properties(value, f"{path}.{key}")
    elif isinstance(node, list):
        for i, item in enumerate(node):
            _check_additional_properties(item, f"{path}[{i}]")


if __name__ == "__main__":
    mock_article = ArticleExtraction(
        bibliographic_info=BibliographicInfo(
            title="Predicting PISA Mathematics Achievement Using Gradient Boosting: A Cross-National Study",
            authors=["Smith, John A.", "Okonkwo, Emeka B.", "Chen, Wei"],
            publication_year=2022,
            doi="10.1016/j.compedu.2022.104512",
            url="https://www.sciencedirect.com/science/article/pii/S0360131522000512",
            venue="Computers and Education",
            venue_type="journal",
            open_access=False,
        ),
        ilsa_context=ILSAContext(
            assessments_used=["PISA"],
            assessment_cycles=["PISA 2018"],
            target_domain=["mathematics"],
            countries_analyzed=["TUR", "DEU", "FIN", "KOR", "BRA"],
            country_count=5,
            target_population="students_age_15",
            sample_size=42350,
        ),
        methodological_handling=MethodologicalHandling(
            weights_applied="full_replicate_weights",
            plausible_values_handling="all_PVs_with_Rubins_rules",
            missing_data_strategy=["multiple_imputation"],
            multilevel_structure="acknowledged_modeled",
            multilevel_handled_explicitly=True,
        ),
        ml_methodology=MLMethodology(
            task_type=["regression"],
            algorithms_used=["Random Forest", "XGBoost", "Ridge Regression"],
            deep_learning_used=False,
            deep_learning_architectures=["not_applicable"],
            feature_engineering=[
                "log transformation of parental income",
                "one-hot encoding of school type",
            ],
            train_test_strategy="k_fold_CV",
            hyperparameter_tuning="grid_search",
        ),
        performance_metrics=PerformanceMetrics(
            classification_metrics=["not_applicable"],
            regression_metrics=["RMSE", "R_squared"],
            best_model="XGBoost",
            best_score=0.74,
            benchmark_comparison=True,
        ),
        xai_usage=XAIUsage(
            xai_used=True,
            xai_methods=["SHAP", "permutation_importance"],
            global_vs_local="both",
            top_features_identified=[
                "ESCS (economic, social, cultural status)",
                "reading score",
                "school resources index",
            ],
        ),
        findings_and_policy=FindingsAndPolicy(
            key_findings=[
                "XGBoost outperformed all baseline models with R-squared of 0.74 on held-out test sets.",
                "Socioeconomic status was the dominant predictor across all five countries, explaining over 35% of variance.",
            ],
            inequality_dimensions_addressed=["socioeconomic_status", "immigration_background"],
            policy_recommendations=[
                "Targeted resource allocation to schools with high proportions of low-SES students is supported by feature importance findings.",
                "Early identification systems using the top SHAP features could flag at-risk students before age 15.",
            ],
            target_decision_makers=["national_policymakers", "school_administrators"],
            limitations_acknowledged=[
                "Cross-sectional design prevents causal inference about the identified predictors.",
                "PISA 2018 data does not capture pandemic-era disruptions to education systems.",
            ],
        ),
        is_genuine_ilsa_ai_study=True,
        extraction_confidence="high",
        source_database="scopus",
        source_pdf_filename="scopus_smith2022_pisa_ml.pdf",
    )

    print("Mock ArticleExtraction (JSON):")
    print(mock_article.model_dump_json(indent=2))

    raw_schema = ArticleExtraction.model_json_schema()
    print("\nRaw Pydantic JSON Schema (contains $defs and $ref):")
    print(json.dumps(raw_schema, indent=2))

    inline_schema = resolve_schema_refs(raw_schema)
    print("\nResolved Schema (all $ref inlined, ready for GPT-4o Structured Outputs):")
    print(json.dumps(inline_schema, indent=2))

    _check_additional_properties(inline_schema)
    assert "$defs" not in inline_schema, "$defs key still present after resolution"
    assert "$ref" not in json.dumps(inline_schema), "$ref still present in resolved schema"
    print("\nAll checks passed: additionalProperties:false on every object node, no $ref remaining.")
