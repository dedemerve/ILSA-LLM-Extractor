#!/usr/bin/env python3
"""
RQ2 redesign (Phase 3): does the content of the knowledge base matter?

Runs on exactly the complete folds and test rows of run_unified_backtest.py
(imported, not copied) and evaluates, with the same time-ordered Ridge:
  M0A, M0, M1 (KB v3 sqrt(w) scaling)
  Uniform_c        c in {0.75, 0.5, 0.25}
  Permutation null 200 random assignments of the 12 positive KB weights
  DataDriven_block covariate-block multiplier chosen with alpha by inner CV
  DataDriven_adaptive  w_j = |beta_j| / max|beta| from M0 on the training data
  KB_top_k         top-k covariates by KB weight (unweighted), k in {1,2,3}
  Random_top_k     200 random covariate orderings, same top-k rule

The decision rule is pre-registered in analysis/remediation/03_preregistration.md;
the script refuses to run if that file changed (sha256 check).
Conditions needing confidence intervals are evaluated in Phase 4.

Outputs (refuses to overwrite): outputs/rq2_v3/ by default.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_unified_backtest as U  # noqa: E402
from kb_weights import KBWeights  # noqa: E402

PREREG = ROOT / "analysis" / "remediation" / "03_preregistration.md"
PREREG_SHA256 = "1713cbb6276797b8dd750d1b53ac2666677f388ef6d83df11e0073eaafa4fdb8"
HARNESS_PRED = ROOT / "outputs" / "unified_backtest_v1" / "predictions.csv"
DEFAULT_OUT = ROOT / "outputs" / "rq2_v3"

SEED = 20261027
N_DRAWS = 200
UNIFORM = [0.75, 0.5, 0.25]
BLOCK_GRID = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
TOP_K = [1, 2, 3]
SF = U.SCORE_FEATURES


# ---------------------------------------------------------------------------
# Folds identical to the harness
# ---------------------------------------------------------------------------

def harness_folds(ser: U.Series) -> list[dict]:
    folds = []
    for t in ser.cycles:
        if ser.prev_cycle(t) is None:
            continue
        test, _ = ser.rows(t)
        if test.empty or not any(s < t for s in ser.cycles):
            continue
        train = pd.concat([ser.rows(s)[0] for s in ser.cycles if s < t], ignore_index=True)
        if len(train) < U.MIN_TRAIN:
            continue
        cov_used = []
        for v in ser.cov_vars:
            col = train[v].to_numpy(dtype=float) if v in train else np.array([np.nan])
            if np.isfinite(col).any() and np.nanstd(col) > 1e-9:
                cov_used.append(v)
        folds.append({"t": t, "train": train, "test": test, "cov": cov_used})
    return folds


# ---------------------------------------------------------------------------
# Ridge helpers (same preprocessing as the harness)
# ---------------------------------------------------------------------------

def fit_fixed(tr: pd.DataFrame, te: pd.DataFrame, cols: list[str], scale: np.ndarray,
              alpha: float) -> tuple[np.ndarray, Ridge]:
    xtr, xte = U._prep(tr, te, cols)
    sc = StandardScaler().fit(xtr)
    m = Ridge(alpha=alpha).fit(sc.transform(xtr) * scale, tr["y"].to_numpy())
    return m.predict(sc.transform(xte) * scale), m


def inner_split(train: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    cycles = sorted(train["cycle"].unique())
    tr, va = train[train["cycle"] < cycles[-1]], train[train["cycle"] == cycles[-1]]
    if len(cycles) >= 2 and len(tr) >= U.MIN_TRAIN and len(va) >= 3:
        return tr, va
    return None


def weighted(fold: dict, wmap: dict[str, float]) -> np.ndarray:
    cols = SF + fold["cov"]
    scale = np.sqrt(np.array([1.0] * len(SF) + [wmap[v] for v in fold["cov"]]))
    return U.fit_ridge(fold["train"], fold["test"], cols, scale)[0]


def subset(fold: dict, keep: list[str]) -> np.ndarray:
    cols = SF + keep
    return U.fit_ridge(fold["train"], fold["test"], cols, np.ones(len(cols)))[0]


def block(fold: dict) -> tuple[np.ndarray, float, float, str]:
    cols = SF + fold["cov"]
    split = inner_split(fold["train"])
    if split is None:
        c, alpha, mode = 1.0, U.FALLBACK_ALPHA, "fallback_fixed"
    else:
        best = None
        for c_ in BLOCK_GRID:
            sc_ = np.array([1.0] * len(SF) + [c_] * len(fold["cov"]))
            for a_ in U.ALPHA_GRID:
                p, _ = fit_fixed(split[0], split[1], cols, sc_, a_)
                mae = float(np.mean(np.abs(p - split[1]["y"].to_numpy())))
                if best is None or mae < best[0]:
                    best = (mae, c_, a_)
        _, c, alpha = best
        mode = "inner_time_cv"
    scale = np.array([1.0] * len(SF) + [c] * len(fold["cov"]))
    return fit_fixed(fold["train"], fold["test"], cols, scale, alpha)[0], c, alpha, mode


def adaptive(fold: dict) -> tuple[np.ndarray, dict]:
    cols = SF + fold["cov"]
    ones = np.ones(len(cols))
    _, alpha0, _ = U.fit_ridge(fold["train"], fold["test"], cols, ones)
    _, m = fit_fixed(fold["train"], fold["test"], cols, ones, alpha0)
    beta = np.abs(m.coef_[len(SF):])
    w = beta / beta.max() if beta.size and beta.max() > 0 else np.ones_like(beta)
    w = np.clip(w, 0.01, None)
    scale = np.sqrt(np.concatenate([np.ones(len(SF)), w]))
    return U.fit_ridge(fold["train"], fold["test"], cols, scale)[0], dict(zip(fold["cov"], np.round(w, 4)))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def mae_of(rows: pd.DataFrame, pred: np.ndarray, excl2025: bool) -> float:
    mask = ~rows["test_cycle"].isin(U.PRIMARY_EXCLUDED_TEST_CYCLES).to_numpy() if excl2025 else np.ones(len(rows), bool)
    return float(np.mean(np.abs(pred[mask] - rows["y_true"].to_numpy()[mask])))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    out_dir = args.out if args.out.is_absolute() else ROOT / args.out
    if out_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {out_dir}")
    if U.sha256(PREREG) != PREREG_SHA256:
        raise RuntimeError("Pre-registration file changed since it was locked; refusing to run.")

    kb = KBWeights()
    est, _ = U.load_estimates()
    cov, _ = U.load_covariates(est, kb)
    kb_pos = np.array(sorted(w for w in kb.weights.values() if w > 0))
    harness = pd.read_csv(HARNESS_PRED)

    pred_rows, contrasts, perm_rows, sel_rows, block_log, adapt_log = [], [], [], [], [], []
    for i, (prog, dom) in enumerate(U.MAIN):
        ser = U.Series(prog, dom, est, cov)
        folds = harness_folds(ser)
        rows = pd.concat([f["test"][["country_iso3", "y"]].assign(test_cycle=f["t"]) for f in folds],
                         ignore_index=True).rename(columns={"y": "y_true"})
        rng = np.random.default_rng(SEED + i)
        cat = lambda fn: np.concatenate([fn(f) for f in folds])  # noqa: E731

        preds: dict[str, np.ndarray] = {}
        preds["Ridge_M0A"] = cat(lambda f: U.fit_ridge(f["train"], f["test"], SF, np.ones(len(SF)))[0])
        preds["Ridge_M0"] = cat(lambda f: weighted(f, {v: 1.0 for v in f["cov"]}))
        preds["Ridge_M1"] = cat(lambda f: weighted(f, {v: kb.weight(v) for v in f["cov"]}))
        for c in UNIFORM:
            preds[f"Uniform_{c}"] = cat(lambda f, c=c: weighted(f, {v: c for v in f["cov"]}))
        parts = []
        for f in folds:
            p, c, a, mode = block(f)
            parts.append(p)
            block_log.append({"program": prog, "domain": dom, "test_cycle": f["t"], "c": c, "alpha": a, "mode": mode})
        preds["DataDriven_block"] = np.concatenate(parts)
        parts = []
        for f in folds:
            p, w = adaptive(f)
            parts.append(p)
            adapt_log.append({"program": prog, "domain": dom, "test_cycle": f["t"], "weights": json.dumps(w, sort_keys=True)})
        preds["DataDriven_adaptive"] = np.concatenate(parts)

        # KB-guided selection: only when the program has more covariates than k.
        kb_rank = sorted(ser.cov_vars, key=lambda v: (-kb.weight(v), v))
        ks = [k for k in TOP_K if len(ser.cov_vars) > k]
        for k in ks:
            preds[f"KB_top_{k}"] = cat(lambda f, k=k: subset(f, [v for v in kb_rank if v in f["cov"]][:k]))

        # Harness reproduction check
        hp = harness[(harness["program"] == prog) & (harness["domain"] == dom) & harness["fold_complete"]]
        for m in ("Ridge_M0A", "Ridge_M0", "Ridge_M1"):
            ref = hp[hp["model"] == m].sort_values(["test_cycle", "country_iso3"])
            mine = rows.assign(p=preds[m]).sort_values(["test_cycle", "country_iso3"])
            if len(ref) != len(mine) or np.abs(ref["y_pred"].to_numpy() - mine["p"].to_numpy()).max() > 1e-5:
                raise AssertionError(f"{prog} {dom} {m} does not reproduce the harness")

        for m, p in preds.items():
            pred_rows.append(rows.assign(program=prog, domain=dom, model=m, y_pred=p))

        # Permutation null for KB weight content
        for d in range(N_DRAWS):
            assign = dict(zip(ser.cov_vars, rng.choice(kb_pos, size=len(ser.cov_vars), replace=False)))
            p = cat(lambda f: weighted(f, {v: assign[v] for v in f["cov"]}))
            perm_rows.append({"program": prog, "domain": dom, "draw": d,
                              "MAE_primary": mae_of(rows, p, True), "MAE_incl2025": mae_of(rows, p, False),
                              "assignment": json.dumps({k: round(float(v), 4) for k, v in assign.items()}, sort_keys=True)})

        # Random top-k null
        for d in range(N_DRAWS):
            order = list(rng.permutation(ser.cov_vars))
            for k in ks:
                p = cat(lambda f: subset(f, [v for v in order if v in f["cov"]][:k]))
                sel_rows.append({"program": prog, "domain": dom, "k": k, "draw": d,
                                 "MAE_primary": mae_of(rows, p, True), "MAE_incl2025": mae_of(rows, p, False)})

        for excl, label in ((True, "primary_excl2025"), (False, "sensitivity_incl2025")):
            r = {"analysis": label, "program": prog, "domain": dom,
                 "n": int((~rows["test_cycle"].isin(U.PRIMARY_EXCLUDED_TEST_CYCLES)).sum()) if excl else len(rows),
                 "n_covariates_program": len(ser.cov_vars)}
            for m, p in preds.items():
                r[f"MAE_{m}"] = mae_of(rows, p, excl)
            col = "MAE_primary" if excl else "MAE_incl2025"
            null = np.array([x[col] for x in perm_rows if x["program"] == prog and x["domain"] == dom])
            r["perm_p05"] = float(np.percentile(null, 5))
            r["perm_median"] = float(np.median(null))
            r["M1_share_of_perms_better_or_equal"] = float(np.mean(null <= r["MAE_Ridge_M1"]))
            r["dMAE_M0A_minus_M1"] = r["MAE_Ridge_M0A"] - r["MAE_Ridge_M1"]
            r["dMAE_M0_minus_M1"] = r["MAE_Ridge_M0"] - r["MAE_Ridge_M1"]
            placebo = {m: r[f"MAE_{m}"] for m in preds if m.startswith(("Uniform", "DataDriven"))}
            r["best_placebo"] = min(placebo, key=placebo.get)
            r["best_placebo_minus_M1"] = placebo[r["best_placebo"]] - r["MAE_Ridge_M1"]
            r["rule_c1_point_gain_ge_1"] = r["dMAE_M0A_minus_M1"] >= 1.0
            r["rule_c2_beats_perm_p05"] = r["MAE_Ridge_M1"] < r["perm_p05"]
            r["rule_c3_beats_M0"] = r["dMAE_M0_minus_M1"] > 0
            for k in ks:
                nul = np.array([x[col] for x in sel_rows if x["program"] == prog and x["domain"] == dom and x["k"] == k])
                r[f"KB_top_{k}_random_p05"] = float(np.percentile(nul, 5))
                r[f"KB_top_{k}_share_random_better_or_equal"] = float(np.mean(nul <= r[f"MAE_KB_top_{k}"]))
            contrasts.append(r)
        print(f"done {prog} {dom}: folds={len(folds)} rows={len(rows)}", flush=True)

    out_dir.mkdir(parents=True)
    fmt = "%.6f"
    pred = pd.concat(pred_rows, ignore_index=True)[["program", "domain", "test_cycle", "country_iso3", "model", "y_true", "y_pred"]]
    pred.to_csv(out_dir / "predictions.csv", index=False, float_format=fmt)
    con = pd.DataFrame(contrasts)
    con[con["analysis"] == "primary_excl2025"].to_csv(out_dir / "contrasts_primary_excl2025.csv", index=False, float_format=fmt)
    con[con["analysis"] == "sensitivity_incl2025"].to_csv(out_dir / "contrasts_sensitivity_incl2025.csv", index=False, float_format=fmt)
    pd.DataFrame(perm_rows).to_csv(out_dir / "permutation_null.csv", index=False, float_format=fmt)
    pd.DataFrame(sel_rows).to_csv(out_dir / "random_topk_null.csv", index=False, float_format=fmt)
    pd.DataFrame(block_log).to_csv(out_dir / "datadriven_block_log.csv", index=False)
    pd.DataFrame(adapt_log).to_csv(out_dir / "datadriven_adaptive_weights.csv", index=False)

    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "git_head": git,
        "preregistration": str(PREREG.relative_to(ROOT)), "preregistration_sha256": PREREG_SHA256,
        "harness": "scripts/run_unified_backtest.py (folds, preprocessing, alpha selection imported)",
        "harness_predictions_sha256": U.sha256(HARNESS_PRED),
        "seed": SEED, "n_draws": N_DRAWS, "uniform": UNIFORM, "block_grid": BLOCK_GRID, "top_k": TOP_K,
        "kb_positive_weights": [float(x) for x in kb_pos],
        "harness_reproduction_check": "Ridge_M0A/M0/M1 predictions match harness within 1e-5",
        "note": "rule condition 1 needs the Phase 4 cluster-bootstrap CI; only the point-gain part is evaluated here",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Saved to {out_dir}/")


if __name__ == "__main__":
    main()
