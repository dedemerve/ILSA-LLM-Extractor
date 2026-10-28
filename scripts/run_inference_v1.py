#!/usr/bin/env python3
"""
Phase 4 inference for RQ1 and RQ2 on the unified backtest.

Paired per-row loss difference d_i = |e_ref,i| - |e_model,i| (positive = model
better), Delta = mean(d). Uncertainty by weighted bootstrap of rows:
  country  : resample countries within a program-domain (all their folds) - primary
  cycle    : resample test cycles (3-6 clusters; unstable, sensitivity only)
  twoway   : pigeonhole bootstrap, countries and cycles resampled independently,
             row weight = product of the two multiplicities (Owen 2007)
Tests (country scheme):
  superiority  one-sided p for H1: Delta > 0, null-shifted bootstrap; Holm over program-domains
  difference   two-sided p (RQ1); Holm over program-domains per model
  TOST         margin +-1 point: equivalent if the 90% CI lies inside (-1, 1); bootstrap TOST p
               = max of the two one-sided p values; Holm over program-domains

RQ2 contrasts (pre-registered, analysis/remediation/03_preregistration.md):
  M1 vs M0A and M1 vs M0. Condition 1 = Delta(M0A - M1) >= 1 and CI lower > 0 and Holm p < .05.
RQ1 contrasts: every model vs Persistence; TimesFM 2.5 zero-shot on its subset.
Context: median SE of country means and a sampling-noise floor for persistence,
E|e| = sqrt(2/pi) * sqrt(SE_t^2 + SE_(t-1)^2), which ignores link error.

Primary analysis excludes test cycle 2025; sensitivity includes it.
Outputs (refuses to overwrite): outputs/inference_v1/
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_unified_backtest as U  # noqa: E402

HARNESS = ROOT / "outputs" / "unified_backtest_v1" / "predictions.csv"
RQ2 = ROOT / "outputs" / "rq2_v3" / "predictions.csv"
RQ2_CONTRASTS = ROOT / "outputs" / "rq2_v3" / "contrasts_primary_excl2025.csv"
PREREG = ROOT / "analysis" / "remediation" / "03_preregistration.md"
PREREG_SHA256 = "1713cbb6276797b8dd750d1b53ac2666677f388ef6d83df11e0073eaafa4fdb8"
DEFAULT_OUT = ROOT / "outputs" / "inference_v1"

SEED = 20261027
B = 5000
MARGIN = 1.0
KEY = ["program", "domain", "test_cycle", "country_iso3"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def holm(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj


def weights(rows: pd.DataFrame, scheme: str, rng: np.random.Generator) -> np.ndarray:
    """(B, n) row weights for one program-domain."""
    def cluster_w(labels: np.ndarray) -> np.ndarray:
        uniq, inv = np.unique(labels, return_inverse=True)
        counts = rng.multinomial(len(uniq), np.full(len(uniq), 1 / len(uniq)), size=B)
        return counts[:, inv].astype(float)
    if scheme == "country":
        return cluster_w(rows["country_iso3"].to_numpy())
    if scheme == "cycle":
        return cluster_w(rows["test_cycle"].to_numpy())
    return cluster_w(rows["country_iso3"].to_numpy()) * cluster_w(rows["test_cycle"].to_numpy())


def boot_stats(d: np.ndarray, W: np.ndarray) -> dict:
    est = float(d.mean())
    tot = W.sum(axis=1)
    ok = tot > 0
    star = (W[ok] @ d) / tot[ok]
    lo95, hi95 = np.percentile(star, [2.5, 97.5])
    lo90, hi90 = np.percentile(star, [5, 95])
    cen = star - est  # null-shifted distribution
    p_sup = (1 + np.sum(cen >= est)) / (len(star) + 1)                 # H1: Delta > 0
    p_two = (1 + np.sum(np.abs(cen) >= abs(est))) / (len(star) + 1)
    p_lo = (1 + np.sum(cen >= est + MARGIN)) / (len(star) + 1)         # H1: Delta > -margin
    p_hi = (1 + np.sum(cen <= est - MARGIN)) / (len(star) + 1)         # H1: Delta < +margin
    return {"delta": est, "ci95_lo": lo95, "ci95_hi": hi95, "ci90_lo": lo90, "ci90_hi": hi90,
            "p_superiority": p_sup, "p_two_sided": p_two, "p_tost": max(p_lo, p_hi),
            "equivalent_90ci": bool(lo90 > -MARGIN and hi90 < MARGIN)}


def contrasts(pred: pd.DataFrame, pairs: list[tuple[str, str]], analysis: str, family: str) -> pd.DataFrame:
    out = []
    for (prog, dom), g in pred.groupby(["program", "domain"], sort=True):
        wide = g.pivot_table(index=KEY, columns="model", values="abs_err").reset_index()
        rows = wide[KEY]
        for scheme in ("country", "cycle", "twoway"):
            tag = hashlib.sha256(f"{prog}|{dom}|{scheme}|{analysis}|{family}".encode()).hexdigest()
            rng = np.random.default_rng([SEED, int(tag[:8], 16)])
            W = weights(rows, scheme, rng)
            for ref, model in pairs:
                if ref not in wide or model not in wide:
                    continue
                sub = wide[[ref, model]].notna().all(axis=1).to_numpy()
                d = (wide[ref] - wide[model]).to_numpy()[sub]
                s = boot_stats(d, W[:, sub])
                out.append({"analysis": analysis, "family": family, "scheme": scheme, "program": prog,
                            "domain": dom, "reference": ref, "model": model, "n": int(sub.sum()),
                            "n_countries": int(rows.loc[sub, "country_iso3"].nunique()),
                            "n_cycles": int(rows.loc[sub, "test_cycle"].nunique()),
                            "MAE_reference": float(wide.loc[sub, ref].mean()),
                            "MAE_model": float(wide.loc[sub, model].mean()), **s})
    df = pd.DataFrame(out)
    for scheme, g in df.groupby("scheme"):
        for (ref, model), gg in g.groupby(["reference", "model"]):
            idx = gg.index
            df.loc[idx, "p_superiority_holm"] = holm(gg["p_superiority"].to_numpy())
            df.loc[idx, "p_two_sided_holm"] = holm(gg["p_two_sided"].to_numpy())
            df.loc[idx, "p_tost_holm"] = holm(gg["p_tost"].to_numpy())
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    out_dir = args.out if args.out.is_absolute() else ROOT / args.out
    if out_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {out_dir}")
    if sha256(PREREG) != PREREG_SHA256:
        raise RuntimeError("Pre-registration file changed; refusing to run.")

    harness = pd.read_csv(HARNESS)
    main_pd = {f"{p}|{d}" for p, d in U.MAIN}
    h = harness[harness["fold_complete"] & (harness["program"] + "|" + harness["domain"]).isin(main_pd)].copy()
    h["abs_err"] = (h["y_pred"] - h["y_true"]).abs()

    # TimesFM rows on their subset (true scores already checked in the harness)
    tf = U.load_timesfm()
    base = h[h["model"] == "Persistence"][KEY + ["y_true"]]
    tfm = base.merge(tf, on=KEY)
    tfm = tfm[(tfm["true_score"] - tfm["y_true"]).abs() <= 0.01]
    tf_rows = tfm.assign(model="TimesFM2_5_zero_shot", y_pred=tfm["prediction"],
                         abs_err=(tfm["prediction"] - tfm["y_true"]).abs())[KEY + ["model", "y_true", "y_pred", "abs_err"]]

    rq2 = pd.read_csv(RQ2)
    rq2["abs_err"] = (rq2["y_pred"] - rq2["y_true"]).abs()

    results = []
    for label, excl in (("primary_excl2025", True), ("sensitivity_incl2025", False)):
        keep = (lambda df: df[~df["test_cycle"].isin(U.PRIMARY_EXCLUDED_TEST_CYCLES)]) if excl else (lambda df: df)
        results.append(contrasts(keep(rq2), [("Ridge_M0A", "Ridge_M1"), ("Ridge_M0", "Ridge_M1")], label, "RQ2"))
        models = sorted(m for m in h["model"].unique() if m != "Persistence")
        results.append(contrasts(keep(h), [("Persistence", m) for m in models], label, "RQ1"))
        tf_sub = pd.concat([keep(h).merge(tf_rows[KEY], on=KEY)[KEY + ["model", "abs_err"]],
                            keep(tf_rows)[KEY + ["model", "abs_err"]]])
        results.append(contrasts(tf_sub, [("Persistence", "TimesFM2_5_zero_shot")], label, "RQ1_timesfm_subset"))
    res = pd.concat(results, ignore_index=True)

    # Pre-registered RQ2 decision (primary, country scheme)
    c = pd.read_csv(RQ2_CONTRASTS).set_index(["program", "domain"])
    r = res[(res["analysis"] == "primary_excl2025") & (res["scheme"] == "country") & (res["family"] == "RQ2")]
    dec = []
    for (prog, dom), g in r.groupby(["program", "domain"]):
        a = g[g["reference"] == "Ridge_M0A"].iloc[0]
        b = g[g["reference"] == "Ridge_M0"].iloc[0]
        c1 = bool(a["delta"] >= MARGIN and a["ci95_lo"] > 0 and a["p_superiority_holm"] < 0.05)
        c2 = bool(c.loc[(prog, dom), "rule_c2_beats_perm_p05"])
        c3 = bool(c.loc[(prog, dom), "rule_c3_beats_M0"])
        dec.append({"program": prog, "domain": dom,
                    "delta_M0A_minus_M1": a["delta"], "ci95": f"[{a['ci95_lo']:.2f}, {a['ci95_hi']:.2f}]",
                    "p_sup_holm": a["p_superiority_holm"], "c1": c1, "c2": c2, "c3": c3,
                    "kb_improves": c1 and c2 and c3,
                    "M1_vs_M0A_equivalent_holm": bool(a["equivalent_90ci"] and a["p_tost_holm"] < 0.05),
                    "M1_vs_M0_equivalent_holm": bool(b["equivalent_90ci"] and b["p_tost_holm"] < 0.05)})
    dec = pd.DataFrame(dec)

    # Scale context
    est, _ = U.load_estimates()
    se = {(r.program, r.domain, r.country_iso3, int(r.cycle)): float(r.se) for r in est.itertuples()}
    p = h[h["model"] == "Persistence"].copy()
    prevc = {}
    for prog, dom in U.MAIN:
        cyc = sorted(est[(est["program"] == prog) & (est["domain"] == dom)]["cycle"].unique())
        prevc[(prog, dom)] = {c_: cyc[i - 1] for i, c_ in enumerate(cyc) if i > 0}
    floors = []
    for (prog, dom), g in p.groupby(["program", "domain"]):
        g = g[~g["test_cycle"].isin(U.PRIMARY_EXCLUDED_TEST_CYCLES)]
        s_t, s_p = [], []
        for r_ in g.itertuples():
            pc = prevc[(prog, dom)][r_.test_cycle]
            s_t.append(se.get((prog, dom, r_.country_iso3, r_.test_cycle), np.nan))
            s_p.append(se.get((prog, dom, r_.country_iso3, pc), np.nan))
        s_t, s_p = np.array(s_t), np.array(s_p)
        floor = np.sqrt(2 / np.pi) * np.sqrt(s_t ** 2 + s_p ** 2)
        floors.append({"program": prog, "domain": dom, "median_se_target": np.nanmedian(s_t),
                       "sampling_floor_MAE_persistence": np.nanmean(floor),
                       "MAE_persistence": g["abs_err"].mean(), "n_with_se": int(np.isfinite(floor).sum()),
                       "n": len(g)})
    floors = pd.DataFrame(floors)

    out_dir.mkdir(parents=True)
    fmt = "%.6f"
    res.to_csv(out_dir / "contrasts_all.csv", index=False, float_format=fmt)
    dec.to_csv(out_dir / "rq2_preregistered_decision.csv", index=False, float_format=fmt)
    floors.to_csv(out_dir / "scale_context.csv", index=False, float_format=fmt)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {str(x.relative_to(ROOT)): sha256(x) for x in (HARNESS, RQ2, RQ2_CONTRASTS, PREREG)},
        "B": B, "seed": SEED, "margin": MARGIN,
        "stream_seed": "SEED + first 8 hex of sha256('program|domain|scheme|analysis|family')",
        "schemes": ["country (primary)", "cycle (sensitivity)", "twoway pigeonhole (sensitivity)"],
        "n_kb_improves": int(dec["kb_improves"].sum()),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    pd.set_option("display.width", 220)
    print(dec.round(3).to_string(index=False))
    print(floors.round(2).to_string(index=False))
    print(f"Saved to {out_dir}/")


if __name__ == "__main__":
    main()
