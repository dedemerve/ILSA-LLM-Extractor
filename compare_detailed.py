import pandas as pd

print("="*80)
print("COMPARISON: OLD vs NEW EXTRACTION")
print("="*80)

# Old results
df_old = pd.read_parquet('output_test_run/ilsa_master.parquet')
print("\n" + "="*80)
print("OLD EXTRACTION (output_test_run)")
print("="*80)
for idx, row in df_old.iterrows():
    print(f"\nPDF {idx+1}: {row['file_name']}")
    print(f"  Confounders: {row['core_data__confounders_identified']}")
    print(f"  ML Primary: {row['core_data__ml_technique_primary']}")

# New results
df_new = pd.read_parquet('output_v2_improved_prompt/ilsa_master.parquet')
print("\n" + "="*80)
print("NEW EXTRACTION (output_v2_improved_prompt)")
print("="*80)
for idx, row in df_new.iterrows():
    print(f"\nPDF {idx+1}: {row['file_name']}")
    print(f"  Confounders: {row.get('confounders_identified', 'N/A')}")
    print(f"  ML Techniques: {row.get('ml_techniques', 'N/A')}")
    print(f"  Survey Weights: {row.get('survey_weights_used', 'N/A')}")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
old_empty = sum(1 for _, row in df_old.iterrows() if not row['core_data__confounders_identified'])
print(f"Old: {old_empty}/3 PDFs had empty confounders")
print(f"New: Checking new format...")

# Check if new format has data
if 'confounders_identified' in df_new.columns:
    new_filled = sum(1 for _, row in df_new.iterrows() if row.get('confounders_identified'))
    print(f"New: {new_filled}/3 PDFs have confounders populated")
else:
    print("Note: New format uses different column names")
    print("\nAll columns in new data:")
    print(df_new.columns.tolist())
