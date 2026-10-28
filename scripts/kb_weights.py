#!/usr/bin/env python3
"""
Strict knowledge-base weight lookup (v3).

One key namespace: ILSA variable name (without the "lag_" prefix) ->
canonical_construct (kb_feature_map_v3.csv) -> w (predictor_weights_v3.csv).

There is no silent fallback. An unmapped covariate, an unknown construct or
an undefined weight raises KeyError. Score/history features are neutral
(w=1.0) only when the caller declares them as such.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS_CSV = ROOT / "outputs" / "stage5" / "predictor_weights_v3.csv"
FEATURE_MAP_CSV = ROOT / "outputs" / "stage5" / "kb_feature_map_v3.csv"


class KBWeights:
    def __init__(self, weights_csv: Path = WEIGHTS_CSV, feature_map_csv: Path = FEATURE_MAP_CSV):
        w = pd.read_csv(weights_csv)
        if w["canonical_construct"].duplicated().any():
            raise ValueError("duplicate constructs in weight table")
        self.weights: dict[str, float] = dict(zip(w["canonical_construct"], w["w"]))

        fmap = pd.read_csv(feature_map_csv, dtype=str, keep_default_na=False)
        if fmap["ilsa_variable"].duplicated().any():
            raise ValueError("duplicate ILSA variables in feature map")
        self.feature_map = fmap.set_index("ilsa_variable")
        unknown = set(fmap.loc[fmap["status"] == "weighted", "canonical_construct"]) - set(self.weights)
        if unknown:
            raise KeyError(f"feature map references constructs missing from weights: {sorted(unknown)}")

    @staticmethod
    def _raw(feature: str) -> str:
        return feature[4:] if feature.startswith("lag_") else feature

    def is_excluded(self, feature: str) -> bool:
        return self.feature_map.loc[self._lookup_row(feature), "status"] == "excluded"

    def excluded_source_cycles(self, feature: str) -> list[str]:
        """Return entries like 'TALIS:2018' whose values must be masked to NaN."""
        value = self.feature_map.loc[self._lookup_row(feature), "excluded_source_cycles"]
        return [v for v in value.split("|") if v]

    def _lookup_row(self, feature: str) -> str:
        raw = self._raw(feature)
        if raw not in self.feature_map.index:
            raise KeyError(f"unmapped covariate: {feature}")
        return raw

    def weight(self, feature: str) -> float:
        row = self._lookup_row(feature)
        if self.feature_map.loc[row, "status"] != "weighted":
            raise KeyError(f"covariate is explicitly excluded, not weighted: {feature}")
        w = float(self.weights[self.feature_map.loc[row, "canonical_construct"]])
        if not np.isfinite(w) or w <= 0:
            raise KeyError(f"non-positive or undefined weight for {feature}: {w}")
        return w

    def vector(self, features: list[str], neutral: set[str]) -> np.ndarray:
        """Weights for a feature list; features in `neutral` (score history) get 1.0."""
        return np.array([1.0 if f in neutral else self.weight(f) for f in features])
