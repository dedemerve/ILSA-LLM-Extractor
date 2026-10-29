#!/usr/bin/env python3
"""
RQ1 reporting tables (Phase 5): one table per program-domain, same n for every
model, MAE with country-cluster bootstrap CI, relative MAE vs persistence, and
bootstrap rank uncertainty (all models resampled jointly in each replicate).

Inputs: outputs/unified_backtest_v1/predictions.csv (complete folds, main
program-domains), saved TimesFM forecasts (subset table), and Phase 4 paired
contrasts (outputs/inference_v1/contrasts_all.csv) for Holm/TOST flags.
No cross-scale averaging of MAE: the aggregate uses the geometric mean of
relative MAE (model / persistence) across program-domains.

Outputs (refuses to overwrite): outputs/rq1_tables_v1/
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
CONTRASTS = ROOT / "outputs" / "inference_v1" / "contrasts_all.csv"
DEFAULT_OUT = ROOT / "outputs" / "rq1_tables_v1"
SEED = 20261027
B = 2000
KEY = ["program", "domain", "test_cycle", "country_iso3"]
LABEL = {"PISA": "PISA", "TIMSS_G8": "TIMSS Grade 8", "TIMSS_G4": "TIMSS Grade 4", "PIRLS": "PIRLS"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def table(err: pd.DataFrame, prog: str, dom: str, analysis: str, subset: str, b: int = B) -> pd.DataFrame:
    """err: wide frame, KEY columns + one abs-error column per model."""
    models = [c for c in err.columns if c not in KEY]
    E = err[models].to_numpy()
    countries = err["country_iso3"].to_numpy()
    uniq, inv = np.unique(countries, return_inverse=True)
    tag = hashlib.sha256(f"{prog}|{dom}|{analysis}|{subset}".encode()).hexdigest()
    rng = np.random.default_rng([SEED, int(tag[:8], 16)])
    counts = rng.multinomial(len(uniq), np.full(len(uniq), 1 / len(uniq)), size=b)
    W = counts[:, inv].astype(float)
    mae_star = (W @ E) / W.sum(axis=1, keepdims=True)          # (B, models)
    ranks = mae_star.argsort(axis=1).argsort(axis=1) + 1
    p = models.index("Persistence")
    rel_star = mae_star / mae_star[:, [p]]
    mae = E.mean(axis=0)
    point_rank = pd.Series(mae).rank(method="min").astype(int).to_numpy()
    rows = []
    for j, m in enumerate(models):
        rows.append({
            "analysis": analysis, "subset": subset, "program": prog, "domain": dom, "model": m,
            "n": len(err), "n_countries": len(uniq), "n_folds": int(err["test_cycle"].nunique()),
            "MAE": mae[j], "MAE_ci95_lo": np.percentile(mae_star[:, j], 2.5),
            "MAE_ci95_hi": np.percentile(mae_star[:, j], 97.5),
            "rel_MAE": mae[j] / mae[p], "rel_MAE_ci95_lo": np.percentile(rel_star[:, j], 2.5),
            "rel_MAE_ci95_hi": np.percentile(rel_star[:, j], 97.5),
            "rank": int(point_rank[j]), "rank_median": float(np.median(ranks[:, j])),
            "rank_ci95_lo": int(np.percentile(ranks[:, j], 2.5)), "rank_ci95_hi": int(np.percentile(ranks[:, j], 97.5)),
            "p_rank1": float(np.mean(ranks[:, j] == 1)),
        })
    return pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--B", type=int, default=B, help="bootstrap replicates (v1 used 2000)")
    args = ap.parse_args()
    out_dir = args.out if args.out.is_absolute() else ROOT / args.out
    if out_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {out_dir}")

    h = pd.read_csv(HARNESS)
    main_pd = {f"{p}|{d}" for p, d in U.MAIN}
    h = h[h["fold_complete"] & (h["program"] + "|" + h["domain"]).isin(main_pd)].copy()
    h["abs_err"] = (h["y_pred"] - h["y_true"]).abs()
    tf = U.load_timesfm()
    base = h[h["model"] == "Persistence"][KEY + ["y_true"]]
    tfm = base.merge(tf, on=KEY)
    tfm = tfm[(tfm["true_score"] - tfm["y_true"]).abs() <= 0.01]
    tf_rows = tfm.assign(model="TimesFM2_5_zero_shot", abs_err=(tfm["prediction"] - tfm["y_true"]).abs())

    con = pd.read_csv(CONTRASTS)
    con = con[(con["scheme"] == "country") & (con["family"].isin(["RQ1", "RQ1_timesfm_subset"]))]

    tables = []
    for label, excl in (("primary_excl2025", True), ("sensitivity_incl2025", False)):
        keep = (lambda df: df[~df["test_cycle"].isin(U.PRIMARY_EXCLUDED_TEST_CYCLES)]) if excl else (lambda df: df)
        hh, tt = keep(h), keep(tf_rows)
        for (prog, dom), g in hh.groupby(["program", "domain"], sort=False):
            wide = g.pivot_table(index=KEY, columns="model", values="abs_err").reset_index()
            tables.append(table(wide, prog, dom, label, "full_common_set", args.B))
            gt = pd.concat([g.merge(tt[KEY], on=KEY), tt[(tt["program"] == prog) & (tt["domain"] == dom)]])
            if len(gt):
                wide_t = gt.pivot_table(index=KEY, columns="model", values="abs_err").reset_index().dropna()
                tables.append(table(wide_t, prog, dom, label, "timesfm_subset", args.B))
    res = pd.concat(tables, ignore_index=True)
    c = con[["analysis", "program", "domain", "model", "family", "p_two_sided_holm", "p_tost_holm", "equivalent_90ci"]].copy()
    c["subset"] = np.where(c["family"] == "RQ1", "full_common_set", "timesfm_subset")
    res = res.merge(c.drop(columns="family"), on=["analysis", "subset", "program", "domain", "model"], how="left")
    res["sig_vs_persistence"] = np.where(res["p_two_sided_holm"] < 0.05,
                                         np.where(res["rel_MAE"] < 1, "better", "worse"), "")
    res["equivalent_to_persistence"] = (res["equivalent_90ci"].astype("boolean").fillna(False).astype(bool)
                                        & (res["p_tost_holm"] < 0.05))

    # Aggregate across program-domains: geometric mean of relative MAE (full common set)
    agg = (res[res["subset"] == "full_common_set"].groupby(["analysis", "model"])
              .agg(n_program_domains=("rel_MAE", "size"),
                   geo_mean_rel_MAE=("rel_MAE", lambda v: float(np.exp(np.log(v).mean()))),
                   n_point_rank1=("rank", lambda v: int((v == 1).sum())),
                   mean_p_rank1=("p_rank1", "mean"),
                   n_sig_better=("sig_vs_persistence", lambda v: int((v == "better").sum())),
                   n_sig_worse=("sig_vs_persistence", lambda v: int((v == "worse").sum())),
                   n_equivalent=("equivalent_to_persistence", "sum"))
              .reset_index().sort_values(["analysis", "geo_mean_rel_MAE"]))

    out_dir.mkdir(parents=True)
    fmt = "%.6f"
    res.to_csv(out_dir / "rq1_per_program_domain.csv", index=False, float_format=fmt)
    agg.to_csv(out_dir / "rq1_aggregate_relative_mae.csv", index=False, float_format=fmt)

    # Markdown tables for the thesis appendix (primary analysis)
    lines = ["# RQ1 tables (primary analysis: complete folds, 2025 excluded)", "",
             f"MAE in score points with 95% country-cluster bootstrap CI (B = {args.B}). Relative MAE = model / Persistence.",
             "Rank interval and P(rank 1) come from joint bootstrap replicates. Significance and equivalence vs",
             "Persistence: Holm-adjusted paired tests from outputs/inference_v1 (worse/better: two-sided p < .05;",
             "equivalent: TOST, margin 1 point). Every model in a table is scored on the same rows.", ""]
    for subset, title in (("full_common_set", "Common test set"), ("timesfm_subset", "TimesFM subset")):
        for prog, dom in U.MAIN:
            t = res[(res["analysis"] == "primary_excl2025") & (res["subset"] == subset)
                    & (res["program"] == prog) & (res["domain"] == dom)]
            if t.empty:
                continue
            r0 = t.iloc[0]
            lines += [f"## {LABEL[prog]} {dom} — {title} (n = {r0.n}, {r0.n_countries} countries, {r0.n_folds} folds)", "",
                      "| Rank | Model | MAE [95% CI] | Rel. MAE [95% CI] | Rank 95% interval | P(rank 1) | vs Persistence |",
                      "|---|---|---|---|---|---|---|"]
            for r in t.itertuples():
                if r.model == "Persistence":
                    flag = "reference"
                else:
                    sig = r.sig_vs_persistence if isinstance(r.sig_vs_persistence, str) and r.sig_vs_persistence else ""
                    flag = (f"{sig}, within ±1" if sig else "equivalent (±1)") if r.equivalent_to_persistence \
                        else (sig or "n.s.")
                lines.append(f"| {r.rank} | {r.model} | {r.MAE:.2f} [{r.MAE_ci95_lo:.2f}, {r.MAE_ci95_hi:.2f}] | "
                             f"{r.rel_MAE:.3f} [{r.rel_MAE_ci95_lo:.3f}, {r.rel_MAE_ci95_hi:.3f}] | "
                             f"{r.rank_ci95_lo}–{r.rank_ci95_hi} | {r.p_rank1:.2f} | {flag} |")
            lines.append("")
    a = agg[agg["analysis"] == "primary_excl2025"]
    lines += ["## Aggregate across the 8 program-domains (common test set)", "",
              "| Model | Geometric mean rel. MAE | Point rank 1 | Mean P(rank 1) | Sig. better | Sig. worse | Equivalent |",
              "|---|---|---|---|---|---|---|"]
    for r in a.itertuples():
        lines.append(f"| {r.model} | {r.geo_mean_rel_MAE:.3f} | {r.n_point_rank1} | {r.mean_p_rank1:.2f} | "
                     f"{r.n_sig_better} | {r.n_sig_worse} | {r.n_equivalent} |")
    (out_dir / "rq1_tables.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "B": args.B, "seed": SEED,
                "inputs": {str(x.relative_to(ROOT)): sha256(x) for x in (HARNESS, CONTRASTS)}}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    pd.set_option("display.width", 220)
    print(a.round(3).to_string(index=False))
    print(f"Saved to {out_dir}/")


if __name__ == "__main__":
    main()
