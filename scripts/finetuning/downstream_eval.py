"""
Downstream (extrinsic) evaluation: extraction -> W_j -> LOCO forecasting (action plan step 8).

Conditions are named weight sets built from evidence pairs:
  oracle   gold v1.4.1 pairs of the evaluated split (the extraction ceiling)
  <arm>    pairs predicted by an extraction arm (any predictions JSON from extract_v2.py)

Weights are rebuilt with the unchanged Stage 5 formula (compute_predictor_weights_v2.compute_weights,
w = F * D * C * E, min-max normalized) after mapping codebook v1.3 constructs onto the Stage 5
predictor vocabulary (CROSSWALK below). Mode `augment` (default) adds the split's evidence to the
Stage 5 literature corpus, i.e. the marginal effect that the thesis pipeline would see; mode
`standalone` builds W from the split's evidence alone.

Reported
  delta_W      L1 distance and Spearman rho of each condition's W to the oracle W, and
               dominant-direction agreement
  forecasting  M1 (literature-weighted Ridge) re-run per condition with the unchanged LOCO code,
               written to a scratch directory (Stage 5 outputs are never touched); per-country
               absolute-error differences tested with a paired bootstrap CI, Wilcoxon signed-rank
               and a TOST equivalence test against --margin PISA points
  noise        dose-response of MAE to synthetic extraction error injected into the oracle
               evidence (direction flips, pair drops, construct swaps), --reps replicates per level

The cross-sectional Diebold-Mariano statistic in run_loco_forecasting.py applies a lag-1
Newey-West term across countries, which have no time order; it is replaced here by the paired
per-country tests above.

Usage
  python3 scripts/finetuning/downstream_eval.py --split val \
      --condition base=outputs/finetuning/v2/predictions/base_qwen3_val.json \
      --condition lora=outputs/finetuning/v2/predictions/qwen3_lora_e3_val.json \
      --noise --reps 100 --out outputs/finetuning/v2/downstream/val.json
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import logging
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "scripts"))

from eval_extraction import gold_by_study, load_gold, load_predictions  # noqa: E402
from ilsa_schema import CONSTRUCTS  # noqa: E402

STAGE5_EFFECTS = ROOT / "outputs/stage5/effect_sizes.csv"
SEED = 2026

# codebook v1.3 construct -> (Stage 5 predictor_canonical, W feature key, match quality)
CROSSWALK: dict[str, tuple[str, str, str]] = {
    "SES_COMPOSITE": ("SES_COMPOSITE", "ESCS", "exact"),
    "SCHOOL_BELONGING": ("BELONGING", "BELONG", "exact"),
    "PARENTAL_EDUCATION": ("PARENTAL_EDUCATION", "PARENTAL_EDU", "exact"),
    "DISCIPLINARY_CLIMATE": ("SCHOOL_CLIMATE", "DISCLIMA", "exact"),
    "TEACHER_SELF_EFFICACY_CLS_MGMT": ("TEACHER_QUALITY", "TEACHER_QUALITY", "exact"),
    "TEACHER_PD": ("EFFECTIVE_PD", "EFFPD", "exact"),
    "ANXIETY": ("ANXIETY", "ANXIETY", "exact"),
    "MOTIVATION": ("MOTIVATION", "MOTIVATION", "exact"),
    "GENDER": ("GENDER", "GENDER_GAP", "exact"),
    "INSTRUCTION_TIME": ("INSTRUCTIONAL_TIME", "INSTR_TIME", "exact"),
    "ICT_HOME_ACCESS": ("ICT_ACCESS", "ICT_INDEX", "exact"),
    "SELF_CONCEPT": ("SELF_EFFICACY", "SELF_EFFICACY", "approximate: self-concept vs self-efficacy"),
    "TEACHING_QUALITY": ("TEACHER_QUALITY", "TEACHER_QUALITY", "approximate: instructional quality vs teacher quality"),
    "SCHOOL_ICT_INFRASTRUCTURE": ("SCHOOL_RESOURCES", "SCHOOL_RES", "approximate: ICT subset of school resources"),
}
FEATURE_OF = {pc: feat for pc, feat, _ in CROSSWALK.values()}


def crosswalk_table() -> pd.DataFrame:
    rows = [{"codebook_construct": c, "stage5_predictor_canonical": pc, "w_feature": f, "match": m}
            for c, (pc, f, m) in CROSSWALK.items()]
    rows += [{"codebook_construct": c, "stage5_predictor_canonical": None, "w_feature": None,
              "match": "none: no Stage 5 weight"} for c in CONSTRUCTS if c not in CROSSWALK]
    return pd.DataFrame(rows)


# Evidence -> Stage 5 effect-size rows

_NUM = re.compile(r"-?\d*\.\d+|-?\d+")


def _effect_value(effect_size: Any) -> float:
    for m in _NUM.findall(str(effect_size or "")):
        v = float(m)
        if -1.0 <= v <= 1.0 and v != 0.0:
            return v
    return float("nan")


def evidence_rows(pairs_by_study: dict[str, list[dict]], meta: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for sid, pairs in pairs_by_study.items():
        for p in pairs:
            mapped = CROSSWALK.get(p["canonical_construct"])
            if not mapped:
                continue
            pv = p.get("plausible_values_handling")
            rows.append({
                "paper_id": sid, "program": meta[sid]["program"],
                "domain": p.get("outcome_class", "other"),
                "predictor_canonical": mapped[0],
                "effect_type": "coef", "effect_value": _effect_value(p.get("effect_size")),
                "direction": p["direction"], "outcome": "achievement score",
                "pv_correct": True if pv in ("rubin_rules", "all_pvs", "average_pv")
                else (False if pv == "single_pv" else None),
                "weight_correct": p.get("student_weights_used"),
                "quality_flag": "medium",
            })
    return pd.DataFrame(rows)


def weights_from_rows(rows: pd.DataFrame, mode: str) -> dict[str, float]:
    from compute_predictor_weights_v2 import compute_weights

    frame = rows
    if mode == "augment":
        base = pd.read_csv(STAGE5_EFFECTS)
        frame = pd.concat([base, rows], ignore_index=True) if len(rows) else base
    if frame.empty:
        return {}
    w = compute_weights(frame.copy())
    w["feature_name"] = [FEATURE_OF.get(pc, fn) for pc, fn in zip(w["predictor_canonical"], w["feature_name"])]
    direction = dict(zip(w["feature_name"], w["dominant_direction"]))
    weights = dict(zip(w["feature_name"], w["w_norm"].astype(float)))
    weights["_direction"] = direction  # type: ignore[assignment]
    return weights


def delta_w(cond: dict[str, Any], oracle: dict[str, Any]) -> dict[str, Any]:
    keys = sorted((set(cond) | set(oracle)) - {"_direction"})
    a = np.array([cond.get(k, 0.0) for k in keys])
    b = np.array([oracle.get(k, 0.0) for k in keys])
    rho = stats.spearmanr(a, b).statistic if len(keys) > 2 else float("nan")
    da, db = cond.get("_direction", {}), oracle.get("_direction", {})
    shared = [k for k in keys if k in da and k in db]
    return {"n_features": len(keys), "l1_to_oracle": round(float(np.abs(a - b).sum()), 4),
            "spearman_to_oracle": round(float(rho), 4),
            "direction_agreement": round(sum(da[k] == db[k] for k in shared) / len(shared), 4) if shared else None}


# Forecasting

def run_forecast(weights: dict[str, Any]) -> pd.DataFrame:
    import run_loco_forecasting as rlf

    w = {k: v for k, v in weights.items() if k != "_direction"}
    original_load = rlf.load_data
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        saved = (rlf.OUT_RESULTS, rlf.OUT_PREDS, rlf.OUT_SHAP)
        rlf.OUT_RESULTS, rlf.OUT_PREDS, rlf.OUT_SHAP = tmp_path / "r.csv", tmp_path / "p.csv", tmp_path / "s.csv"
        rlf.load_data = lambda: (original_load()[0], w)
        logging.disable(logging.WARNING)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                _, preds = rlf.run_loco()
        finally:
            rlf.load_data = original_load
            rlf.OUT_RESULTS, rlf.OUT_PREDS, rlf.OUT_SHAP = saved
            logging.disable(logging.NOTSET)
    return preds


def fold_errors(preds: pd.DataFrame, program: str, cycle: int) -> pd.Series:
    p = preds[(preds["program"] == program) & (preds["cycle"] == cycle)].dropna(subset=["y_M1"])
    return (p["y_true"] - p["y_M1"]).abs().set_axis(p["domain"] + ":" + p["country_iso3"])


def paired_test(err_a: pd.Series, err_b: pd.Series, margin: float, rng: np.random.Generator) -> dict[str, Any]:
    """B minus A on shared country-domain units: d < 0 means B forecasts better."""
    idx = err_a.index.intersection(err_b.index)
    d = (err_b[idx] - err_a[idx]).to_numpy()
    if len(d) < 3:
        return {"n": int(len(d))}
    boot = d[rng.integers(0, len(d), size=(10_000, len(d)))].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    nonzero = d[d != 0]
    wil = stats.wilcoxon(nonzero).pvalue if len(nonzero) >= 3 else float("nan")
    se = d.std(ddof=1) / np.sqrt(len(d))
    if se > 0:
        p_lower = 1 - stats.t.cdf((d.mean() + margin) / se, len(d) - 1)
        p_upper = stats.t.cdf((d.mean() - margin) / se, len(d) - 1)
        tost = float(max(p_lower, p_upper))
    else:
        tost = 0.0 if abs(d.mean()) < margin else 1.0
    return {"n": int(len(d)), "mae_a": round(float(err_a[idx].mean()), 4), "mae_b": round(float(err_b[idx].mean()), 4),
            "mean_delta": round(float(d.mean()), 4), "ci95_bootstrap": [round(float(lo), 4), round(float(hi), 4)],
            "wilcoxon_p": round(float(wil), 4), "tost_margin": margin, "tost_p": round(tost, 4),
            "equivalent_at_margin": bool(tost < 0.05)}


# Noise injection

def perturb(pairs_by_study: dict[str, list[dict]], flip: float, drop: float, swap: float,
            rng: np.random.Generator) -> dict[str, list[dict]]:
    flip_map = {"positive": "negative", "negative": "positive", "mixed": "mixed", "not_reported": "not_reported"}
    mapped = list(CROSSWALK)
    out = {}
    for sid, pairs in pairs_by_study.items():
        new = []
        for p in pairs:
            if rng.random() < drop:
                continue
            q = dict(p)
            if rng.random() < flip:
                q["direction"] = flip_map[q["direction"]]
            if rng.random() < swap:
                q["canonical_construct"] = mapped[rng.integers(len(mapped))]
            new.append(q)
        out[sid] = new
    return out


def noise_curve(oracle_pairs: dict, meta: dict, mode: str, program: str, cycle: int,
                reps: int, rng: np.random.Generator) -> list[dict[str, Any]]:
    levels = [0.0, 0.1, 0.25, 0.5, 1.0]
    rows = []
    for kind in ("flip", "drop", "swap"):
        for level in levels:
            maes = []
            for _ in range(1 if level == 0.0 else reps):
                kw = {"flip": 0.0, "drop": 0.0, "swap": 0.0, kind: level}
                w = weights_from_rows(evidence_rows(perturb(oracle_pairs, rng=rng, **kw), meta), mode)
                maes.append(float(fold_errors(run_forecast(w), program, cycle).mean()))
            rows.append({"perturbation": kind, "level": level, "reps": len(maes),
                         "mae_mean": round(float(np.mean(maes)), 4),
                         "mae_p2_5": round(float(np.percentile(maes, 2.5)), 4),
                         "mae_p97_5": round(float(np.percentile(maes, 97.5)), 4)})
            print(f"  noise {kind}={level}: MAE {rows[-1]['mae_mean']}")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="val", choices=["train", "val", "test", "all"])
    ap.add_argument("--condition", action="append", default=[], metavar="NAME=PREDICTIONS.json")
    ap.add_argument("--mode", default="augment", choices=["augment", "standalone"])
    ap.add_argument("--program", default="PISA")
    ap.add_argument("--cycle", type=int, default=2022)
    ap.add_argument("--margin", type=float, default=1.0, help="TOST equivalence margin in score points")
    ap.add_argument("--noise", action="store_true")
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    rng = np.random.default_rng(SEED)

    gold = load_gold(args.split)
    meta = {sid: {"program": str(g["program"].iloc[0])} for sid, g in gold.groupby("study_id")}
    oracle_pairs = gold_by_study(gold)
    conditions: dict[str, dict[str, list[dict]]] = {"oracle": oracle_pairs}
    for spec in args.condition:
        name, path = spec.split("=", 1)
        preds = load_predictions(Path(path))
        conditions[name] = {sid: preds.get(sid, {"pairs": []})["pairs"] for sid in oracle_pairs}

    weights = {name: weights_from_rows(evidence_rows(pairs, meta), args.mode) for name, pairs in conditions.items()}
    weights["stage5_frozen"] = weights_from_rows(pd.DataFrame(), "augment")
    errors = {name: fold_errors(run_forecast(w), args.program, args.cycle) for name, w in weights.items()}

    names = list(weights)
    report: dict[str, Any] = {
        "_split": args.split, "_mode": args.mode, "_fold": f"{args.program} {args.cycle}",
        "crosswalk": crosswalk_table().to_dict(orient="records"),
        "evidence_rows_mapped": {n: int(len(evidence_rows(p, meta))) for n, p in conditions.items()},
        "weights": {n: {k: (round(v, 4) if not isinstance(v, dict) else v) for k, v in w.items()} for n, w in weights.items()},
        "delta_w": {n: delta_w(weights[n], weights["oracle"]) for n in names if n != "oracle"},
        "mae": {n: round(float(e.mean()), 4) for n, e in errors.items()},
        "paired_tests": {f"{b}_vs_{a}": paired_test(errors[a], errors[b], args.margin, rng)
                         for i, a in enumerate(names) for b in names[i + 1:]},
    }
    if args.noise:
        report["noise_dose_response"] = noise_curve(oracle_pairs, meta, args.mode, args.program, args.cycle, args.reps, rng)
    text = json.dumps(report, indent=2, ensure_ascii=False, default=str)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("evidence_rows_mapped", "delta_w", "mae")}, indent=2))
    for k, v in report["paired_tests"].items():
        print(k, v)


if __name__ == "__main__":
    main()
