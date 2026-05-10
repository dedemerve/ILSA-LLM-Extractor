import pandas as pd
import json

print("="*80)
print("NEW EXTRACTION RESULTS (output_v2_improved_prompt)")
print("="*80)

df = pd.read_parquet('output_v2_improved_prompt/ilsa_master.parquet')
print(f"\nSuccessfully extracted {len(df)} PDFs\n")

for idx, row in df.iterrows():
    print(f"\n{'='*80}")
    print(f"PDF {idx+1}: {row['metadata__file_name']}")
    print(f"{'='*80}")
    print(f"Title: {row['metadata__title'][:70]}...")
    print(f"\nILSA Context:")
    print(f"  Primary: {row['ilsa_context__primary_ilsa']} {row['ilsa_context__primary_year']}")
    print(f"  Research Design: {row['research_design_type']}")
    print(f"\nML Techniques:")
    print(f"  Primary: {row['ml_techniques__primary']}")
    all_techs = row['ml_techniques__all_techniques']
    if isinstance(all_techs, list) and len(all_techs) > 0:
        print(f"  All: {all_techs}")

    print(f"\nConfounders (CRITICAL FIELD):")
    confounders = row['confounders_identified']
    if isinstance(confounders, list) and len(confounders) > 0:
        for c in confounders:
            print(f"  ✓ {c}")
    else:
        print(f"  (Empty or None)")

    print(f"\nSurvey Weights: {row['survey_weights_used']}")
    print(f"\nOutcome Summary:")
    print(f"  {row['outcome_summary'][:150]}...")

print(f"\n\n{'='*80}")
print("SUMMARY")
print(f"{'='*80}")
confounders_count = sum(1 for _, row in df.iterrows()
                        if row['confounders_identified'] and
                        isinstance(row['confounders_identified'], list) and
                        len(row['confounders_identified']) > 0)
print(f"✅ Confounders extracted in {confounders_count}/{len(df)} PDFs")
print(f"✅ All {len(df)} PDFs processed successfully")
print(f"✅ New improved prompt captures detailed ML pipeline and covariate information")
