import pandas as pd

df = pd.read_parquet('output_test_run/ilsa_master.parquet')
pd.set_option('display.max_colwidth', None)

for idx, row in df.iterrows():
    print(f"\n{'='*80}")
    print(f"PDF {idx+1}: {row['file_name']}")
    print(f"{'='*80}")
    
    print("\nMETADATA:")
    print(f"  Title: {row['metadata__title']}")
    print(f"  Authors: {row['metadata__authors']}")
    print(f"  Year: {row['metadata__year']}")
    print(f"  DOI: {row['metadata__doi']}")
    print(f"  Venue: {row['metadata__venue']}")
    print(f"  Publication Type: {row['metadata__publication_type']}")
    print(f"  Open Access: {row['metadata__open_access']}")
    
    print("\nCORE DATA:")
    print(f"  ILSA Type: {row['core_data__ilsa_type']}")
    print(f"  ILSA Year: {row['core_data__ilsa_year']}")
    print(f"  Sample Size: {row['core_data__sample_size']}")
    print(f"  Research Design: {row['core_data__research_design_type']}")
    print(f"  ML Technique: {row['core_data__ml_technique_primary']}")
    print(f"  Survey Weights Used: {row['core_data__survey_weights_used']}")
    print(f"  Confounders: {row['core_data__confounders_identified']}")
    
    print(f"\nOUTCOME SUMMARY:")
    print(f"  {row['core_data__outcome_summary']}")
    
    print(f"\nEXTRACTION INFO:")
    print(f"  Input Tokens: {row['input_tokens']}")
    print(f"  Output Tokens: {row['output_tokens']}")
    print(f"  Cost: ${row['cost_usd']:.4f}")
