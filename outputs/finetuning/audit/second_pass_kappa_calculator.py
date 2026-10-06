#!/usr/bin/env python3
"""
Second-Pass Intra-Rater Reliability Calculator
Run after Merve fills in the *_second_pass columns in second_pass_reliability_form.csv
Usage: python3 second_pass_kappa_calculator.py
"""
import pandas as pd
import numpy as np
import json
from pathlib import Path

TIER1_FIELDS = [
    'program', 'cycle', 'target_variable', 'predictor',
    'canonical_construct', 'direction', 'effect_type', 'pv'
]

# Cohen's kappa — pairwise, nominal categories
def cohen_kappa(y1, y2):
    y1, y2 = list(y1), list(y2)
    n = len(y1)
    if n == 0:
        return np.nan
    observed_agreement = sum(a == b for a, b in zip(y1, y2)) / n
    cats = list(set(y1 + y2))
    p1 = {c: y1.count(c) / n for c in cats}
    p2 = {c: y2.count(c) / n for c in cats}
    expected = sum(p1.get(c, 0) * p2.get(c, 0) for c in cats)
    if expected == 1.0:
        return 1.0
    return (observed_agreement - expected) / (1 - expected)

def run():
    form_path = Path(__file__).parent / 'second_pass_reliability_form.csv'
    df = pd.read_csv(form_path)

    # Verify second-pass columns are filled
    unfilled = df[df['program_second_pass'].isna() | (df['program_second_pass'] == '')]
    if len(unfilled) > 0:
        print(f"WARNING: {len(unfilled)} rows still have empty program_second_pass.")
        print("Fill all *_second_pass columns before running this script.")
        return

    results = {}
    print(f"\n{'Field':<30} {'Raw Agree':>10} {'Kappa':>8} {'N':>5} {'Interpretation'}")
    print("-" * 70)
    for field in TIER1_FIELDS:
        fp = f'{field}_first_pass'
        sp = f'{field}_second_pass'
        if fp not in df.columns or sp not in df.columns:
            continue
        valid = df[[fp, sp]].dropna()
        valid = valid[valid[fp] != '']
        valid = valid[valid[sp] != '']
        n = len(valid)
        if n == 0:
            continue
        ra = (valid[fp] == valid[sp]).mean()
        kappa = cohen_kappa(valid[fp], valid[sp])
        interp = (
            'Almost perfect' if kappa >= 0.81 else
            'Substantial' if kappa >= 0.61 else
            'Moderate' if kappa >= 0.41 else
            'Fair' if kappa >= 0.21 else
            'Slight' if kappa >= 0.01 else 'Poor'
        )
        results[field] = {'n': n, 'raw_agreement': round(ra, 3), 'kappa': round(kappa, 3), 'interpretation': interp}
        status = "PASS" if kappa >= 0.61 else ("WARN" if kappa >= 0.41 else "FAIL")
        print(f"{field:<30} {ra:>9.1%}  {kappa:>7.3f}  {n:>4}  {interp} [{status}]")

    print("\n")
    macro_kappa = np.mean([r['kappa'] for r in results.values()])
    print(f"Macro-average kappa: {macro_kappa:.3f}")
    threshold_met = macro_kappa >= 0.61
    print(f"Threshold (≥0.61): {'PASS' if threshold_met else 'FAIL'}")
    print(f"Fine-tuning cleared: {'YES' if threshold_met else 'NO — adjudication required'}")

    # Save results
    out = {
        'test_date': pd.Timestamp.now().isoformat()[:10],
        'n_studies_retested': int(df['study_id'].nunique()),
        'n_rows': int(len(df)),
        'per_field': results,
        'macro_kappa': round(macro_kappa, 3),
        'threshold_met': threshold_met,
        'fine_tuning_cleared': threshold_met,
    }
    out_path = Path(__file__).parent / 'intra_rater_reliability_v4.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved → {out_path}")

if __name__ == '__main__':
    run()
