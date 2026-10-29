#!/usr/bin/env python3
"""
PIAAC Cycle 2 yılı düzeltmesi: unified_panel.csv içinde 2017 → 2022.

CHANGELOG v1.1.0'da belgelenmiş düzeltme:
  icils_piaac_estimates.csv ve ilgili çıktılarda 2017→2022 uygulandı.
  Ancak unified_panel.csv yeniden üretilmediği için hâlâ 2017 gösteriyor.

Çalıştırma:
  python3 scripts/fix_piaac_cycle_year.py
"""
from pathlib import Path
import pandas as pd

ROOT   = Path(__file__).resolve().parents[1]
TARGET = ROOT / "outputs" / "stage5" / "unified_panel.csv"

panel = pd.read_csv(TARGET, dtype={"country_iso3": str})

piaac_2017 = (panel["program"] == "PIAAC") & (panel["cycle"] == 2017)
n = int(piaac_2017.sum())
if n == 0:
    print("unified_panel.csv: PIAAC 2017 satırı yok — zaten düzeltilmiş.")
else:
    panel.loc[piaac_2017, "cycle"] = 2022
    panel.to_csv(TARGET, index=False)
    print(f"Düzeltildi: {n} satırda PIAAC cycle 2017 → 2022")
    print(f"PIAAC döngüleri artık: {sorted(panel[panel['program']=='PIAAC']['cycle'].unique())}")
