"""
Red Team Test: Feed adversarial text with 4 deliberate methodological traps
to verify the extraction pipeline catches them.

Traps:
  1. Flat XGBoost on nested 150-school ILSA data, no weights (Rule 18)
  2. 98% accuracy on 2% minority class without SMOTE/F1 (Rule 20)
  3. Process data lazily aggregated to total time + total clicks (Rule 19)
  4. SHAP values claimed to "prove" causality (Rule 17)
"""

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from src.extractors.gpt_extractor import GPTExtractor

ADVERSARIAL_TEXT = """\
Title: Predicting the Unpredictable: Machine Learning in PISA 2022

Authors: Jane Doe, John Smith

Journal: Journal of Educational Data Mining, 2024

DOI: 10.1234/jedm.2024.fake

Abstract

This study examined PISA 2022 creative thinking process data to predict
academically resilient students using machine learning. We analyzed log files
from 10,000 students across 150 schools in Turkey (TUR). An XGBoost model
achieved 98% accuracy. SHAP analysis revealed that the number of books at home
is the strongest causal factor of creative thinking performance.

1. Introduction

International Large-Scale Assessments such as PISA provide rich process data
from computer-based assessments. This study leverages PISA 2022 creative
thinking data to identify the top 2% of highly resilient students.

2. Method

2.1 Participants
The sample comprised 10,000 15-year-old students from 150 schools in Turkey
who participated in PISA 2022.

2.2 Variables
The dependent variable was a binary indicator of resilience (top 2% vs. rest).
Plausible values were averaged to create a single performance score for each
student. Independent variables included ESCS, gender, number of books at home,
immigration status, and parental education.

2.3 Data Processing
To process the log files, we aggregated the data into simple counts, using only
the total time spent and total number of mouse clicks per student. No advanced
sequence mining or time standardization was applied.

2.4 Missing Data
Records with any missing values were removed (listwise deletion).

2.5 Machine Learning Model
We trained a standard XGBoost algorithm on the student-level data. We did not
use any sampling weights or multilevel adjustments because XGBoost is robust to
data structures. Feature selection, standardization, and SMOTE were applied to
the full dataset before the 80/20 train-test split.

2.6 Evaluation
Model performance was evaluated using overall accuracy on the held-out test set.

3. Results

The XGBoost model was highly successful, achieving an overall accuracy of 98%
on the test set. Furthermore, we used SHAP values to explain the model. The
SHAP analysis proves that having more books at home directly causes a higher
level of creative thinking.

4. Discussion

Our findings demonstrate that machine learning can effectively predict student
resilience from PISA process data. The causal link between books at home and
creative thinking has important policy implications.

5. Conclusion

XGBoost is a powerful and sufficient tool for analyzing ILSA data. Accuracy of
98% confirms the validity of our approach.

References
[1] OECD (2023). PISA 2022 Technical Report.
"""


@dataclass
class MockProcessedPDF:
    file_path: Path = Path("red_team_adversarial.pdf")
    file_name: str = "red_team_adversarial.pdf"
    source_database: str = "manual_test"
    total_pages: int = 8
    raw_text: str = ADVERSARIAL_TEXT
    sections: dict = field(default_factory=lambda: {
        "abstract": "abstract",
        "methods": "method",
        "results": "results",
        "discussion": "discussion",
        "conclusion": "conclusion",
    })
    used_smart_sections: bool = False
    extraction_text: str = ADVERSARIAL_TEXT
    estimated_tokens: int = 800
    parse_errors: list = field(default_factory=list)
    metadata: dict = field(default_factory=lambda: {
        "extracted_title": "Predicting the Unpredictable: Machine Learning in PISA 2022"
    })


TRAP_KEYWORDS = {
    "hierarchy / i.i.d.": [
        "nested", "hierarchi", "i.i.d", "iid", "clustered",
        "multilevel", "flat", "student-level", "survey weight",
    ],
    "class imbalance": [
        "imbalanc", "misleading", "skew", "2%", "class weight",
        "smote", "f1", "kappa", "precision-recall",
    ],
    "process data laziness": [
        "aggregat", "frequency count", "total click", "total time",
        "lazy", "sequence", "n-gram", "chronolog",
    ],
    "XAI causality overstatement": [
        "causal", "overstat", "correlat", "cross-sectional",
        "feature importance", "not causal", "does not prove",
        "erroneously", "predictive",
    ],
}


def evaluate_traps(json_output: dict) -> dict[str, dict]:
    """Check whether the LLM caught each trap in the output text."""
    full_text = json.dumps(json_output, ensure_ascii=False).lower()

    results = {}
    for trap_name, keywords in TRAP_KEYWORDS.items():
        hits = [kw for kw in keywords if kw.lower() in full_text]
        results[trap_name] = {
            "caught": len(hits) >= 2,
            "keyword_hits": hits,
            "hit_count": len(hits),
        }
    return results


def main():
    print("=" * 70)
    print("RED TEAM TEST — Adversarial ILSA Paper Extraction")
    print("=" * 70)

    extractor = GPTExtractor()
    mock_pdf = MockProcessedPDF()

    print(f"\nModel: {extractor.model}")
    print("Sending adversarial text to extractor...\n")

    result = extractor.extract(mock_pdf)

    print(f"Success: {result.success}")
    print(f"Duration: {result.duration_seconds:.1f}s")
    print(f"Tokens: {result.input_tokens} in / {result.output_tokens} out")
    print(f"Cost: ${result.cost_usd:.4f}")

    if not result.success:
        print(f"\nERROR: Extraction failed — {result.error}")
        return

    output = result.extraction.model_dump(mode="json")

    print("\n" + "=" * 70)
    print("FULL JSON OUTPUT")
    print("=" * 70)
    print(json.dumps(output, indent=2, ensure_ascii=False))

    print("\n" + "=" * 70)
    print("CRITICAL FIELDS FOR TRAP DETECTION")
    print("=" * 70)

    data = output.get("data", {})
    for field_name in [
        "outcome_summary",
        "null_fields_interpretation",
    ]:
        val = data.get(field_name, "")
        print(f"\n--- {field_name} ---")
        print(val if val else "(empty/null)")

    sd = data.get("survey_design", {})
    print(f"\n--- student_weights_used ---")
    print(sd.get("student_weights_used"))
    print(f"\n--- weight_fields_interpretation ---")
    print(sd.get("weight_fields_interpretation", "(null)"))

    print(f"\n--- research_design_type ---")
    print(data.get("research_design_type"))

    print(f"\n--- plausible_values_handling ---")
    print(data.get("plausible_values_handling"))

    print(f"\n--- missing_data_handling ---")
    print(data.get("missing_data_handling"))

    ml = data.get("ml_techniques", {})
    print(f"\n--- ml_techniques ---")
    print(f"  primary: {ml.get('primary')}")
    print(f"  all: {ml.get('all_techniques')}")

    print("\n" + "=" * 70)
    print("TRAP EVALUATION")
    print("=" * 70)

    trap_results = evaluate_traps(output)
    all_caught = True
    for trap_name, info in trap_results.items():
        status = "CAUGHT" if info["caught"] else "MISSED"
        if not info["caught"]:
            all_caught = False
        print(f"\n  [{status}] {trap_name}")
        print(f"    Keywords found ({info['hit_count']}): {info['keyword_hits']}")

    print("\n" + "=" * 70)
    if all_caught:
        print("VERDICT: ALL 4 TRAPS CAUGHT — Academic Reviewer mode is active!")
    else:
        missed = [k for k, v in trap_results.items() if not v["caught"]]
        print(f"VERDICT: {len(missed)} TRAP(S) MISSED — {', '.join(missed)}")
        print("Prompt refinement may be needed for the missed traps.")
    print("=" * 70)


if __name__ == "__main__":
    main()
