import sys
sys.path.insert(0, '/Users/mrved/Desktop/ILSA_LLMs')

# Import and rebuild in correct order
from src.schemas import Metadata, CoreData, ILSAArticleMetadata

# Force rebuild after all classes are loaded
ILSAArticleMetadata.model_rebuild()

test_data = {
    "metadata": {
        "file_name": "test_paper_002.pdf",
        "title": "Random Forest for TIMSS",
        "authors": "Doe, M.; Smith, J.",
        "year": 2022,
        "publication_type": "journal"
    },
    "core_data": {
        "ilsa_type": "TIMSS",
        "ilsa_year": 2019,
        "sample_size": 3000,
        "research_design_type": "causal_observational",
        "ml_technique_primary": "Random Forest",
        "survey_weights_used": False,
        "confounders_identified": ["SES"],
        "outcome_summary": "School resources showed no effect after controlling for SES."
    }
}

article = ILSAArticleMetadata(**test_data)
print(article.model_dump_json(indent=2))
print("\n✅ Pydantic validation passed!")
