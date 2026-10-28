#!/usr/bin/env python3
"""
Thesis numbers (Phase 6): every number quoted in the revised Methods, Results
and Discussion, computed from the verified output files in one place.

Each row carries its source file (with sha256), the raw value, the value
formatted for the text (MAE 2 dp, relative MAE 3 dp, kappa 2 dp, p 3 dp), and,
for the claims in the Phase 6.0 claim map, the expected value. Any mismatch
aborts the run before anything is written, so the text cannot drift from the
outputs. Corrections K1-K5 of the claim map are encoded here:
  K1 classroom-management self-efficacy: 0 of 6 labels correct (6/6 mislabelled)
  K2 permutation null = share of 200 draws as good as or better than M1
  K3 KB top-k vs random testable in 4 of 8 program-domains only
  K4 RQ1 table CIs from rq1_tables_v2 (B = 5000, same as Phase 4 tests)
  K5 M1 worse than M0A in PISA only under the country-cluster scheme

Outputs (refuses to overwrite): outputs/thesis_numbers_v1/
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
DEFAULT_OUT = ROOT / "outputs" / "thesis_numbers_v1"
O = ROOT / "outputs"
SRC = {
    "rq1_agg": O / "rq1_tables_v2" / "rq1_aggregate_relative_mae.csv",
    "rq1_pd": O / "rq1_tables_v2" / "rq1_per_program_domain.csv",
    "rq1_manifest": O / "rq1_tables_v2" / "manifest.json",
    "contrasts": O / "inference_v1" / "contrasts_all.csv",
    "inference_manifest": O / "inference_v1" / "manifest.json",
    "rq2_decision": O / "inference_v1" / "rq2_preregistered_decision.csv",
    "scale": O / "inference_v1" / "scale_context.csv",
    "rq2_contrasts": O / "rq2_v3" / "contrasts_primary_excl2025.csv",
    "kappa": O / "kb_integrity_v1" / "kappa_summary.json",
    "validation": O / "kb_integrity_v1" / "evidence_validation_agreement_v1.json",
    "validation_sample": O / "kb_integrity_v1" / "evidence_matrix_validation_sample_v1.csv",
    "validation_blind": O / "kb_integrity_v1" / "evidence_matrix_validation_sample_v1_blind.csv",
    "per_construct": O / "kb_integrity_v1" / "evidence_validation_per_construct_v1.csv",
    "weights_corrected": O / "kb_integrity_v1" / "predictor_weights_v3_sample_corrected.csv",
    "evidence": O / "stage5" / "evidence_matrix.csv",
    "gold": O / "finetuning" / "data" / "gold_standard_full_corpus_v1.4.1.csv",
    "no_history": O / "unified_backtest_v1" / "no_history.csv",
}
PRIMARY = "primary_excl2025"
TREES = ["RF_score", "RF_cov", "XGB_score", "XGB_cov"]
FMT = {"mae": "{:.2f}", "rel": "{:.3f}", "kappa": "{:.2f}", "p": "{:.3f}", "share": "{:.2f}", "int": "{:d}"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class Numbers:
    def __init__(self) -> None:
        self.rows: list[dict] = []
        self.failures: list[str] = []

    def add(self, id_: str, section: str, description: str, value, kind: str, source: str,
            expected=None, tol: float = 0.0, label: str = "VERIFIED") -> None:
        value = int(value) if kind == "int" else float(value)
        text = FMT[kind].format(value)
        ok = None
        if expected is not None:
            ok = (value == expected) if kind == "int" else abs(value - expected) <= tol
            if not ok:
                self.failures.append(f"{id_}: got {text}, expected {expected} (tol {tol})")
        self.rows.append({"id": id_, "section": section, "description": description, "value": value,
                          "text": text, "kind": kind, "expected": expected, "check_ok": ok,
                          "label": label, "source": str(SRC[source].relative_to(ROOT))})

    def add_range(self, id_: str, section: str, description: str, values, kind: str, source: str,
                  expected: tuple | None = None, tol: float = 0.0) -> None:
        v = np.asarray(list(values), dtype=float)
        lo_exp, hi_exp = expected if expected else (None, None)
        self.add(f"{id_}.min", section, f"{description} (min)", v.min(), kind, source, lo_exp, tol)
        self.add(f"{id_}.max", section, f"{description} (max)", v.max(), kind, source, hi_exp, tol)


def rq1(n: Numbers) -> None:
    agg = pd.read_csv(SRC["rq1_agg"])
    agg = agg[agg["analysis"] == PRIMARY].set_index("model")
    expected_geo = {"Persistence": 1.000, "AR1": 1.027, "Drift": 1.030, "Ridge_M0A": 1.035, "Ridge_M1": 1.093,
                    "Ridge_M0": 1.095, "XGB_score": 1.182, "XGB_cov": 1.195, "RF_score": 1.218, "RF_cov": 1.220,
                    "LinearTrend": 1.299, "InternationalMean": 3.281}
    for m, e in expected_geo.items():
        n.add(f"rq1.geo_rel_mae.{m}", "RQ1", f"Geometric-mean relative MAE vs persistence, {m}",
              agg.loc[m, "geo_mean_rel_MAE"], "rel", "rq1_agg", e, 0.0005)
    point_rank1 = {"Persistence": 3, "AR1": 2, "Drift": 1}
    for m, e in point_rank1.items():
        n.add(f"rq1.n_point_rank1.{m}", "RQ1", f"Program-domains where {m} has the lowest point MAE",
              agg.loc[m, "n_point_rank1"], "int", "rq1_agg", e)
    n.add("rq1.n_point_rank1.Ridge", "RQ1", "Program-domains where a Ridge model has the lowest point MAE",
          agg.loc[["Ridge_M0", "Ridge_M1", "Ridge_M0A"], "n_point_rank1"].sum(), "int", "rq1_agg", 2)

    pd_ = pd.read_csv(SRC["rq1_pd"])
    full = pd_[(pd_["analysis"] == PRIMARY) & (pd_["subset"] == "full_common_set")]
    winners = full[full["rank"] == 1]
    n.add_range("rq1.point_winner_p_rank1", "RQ1", "Bootstrap P(rank 1) of the point-MAE winner",
                winners["p_rank1"], "share", "rq1_pd", (0.21, 0.79), 0.005)
    pers = full[full["model"] == "Persistence"]
    n.add("rq1.persistence_rank_interval_includes_1", "RQ1",
          "Program-domains whose persistence 95% rank interval includes rank 1",
          (pers["rank_ci95_lo"] == 1).sum(), "int", "rq1_pd", 8)
    n.add_range("rq1.persistence_mae", "RQ1", "Persistence MAE (score points), common test set",
                pers["MAE"], "mae", "rq1_pd", (9.02, 18.34), 0.005)
    n.add("rq1.bootstrap_B_tables", "Methods", "Bootstrap replicates for RQ1 table CIs and rank intervals",
          json.loads(SRC["rq1_manifest"].read_text())["B"], "int", "rq1_manifest", 5000)

    con = pd.read_csv(SRC["contrasts"])
    con = con[con["analysis"] == PRIMARY]
    vs_p = con[con["family"].isin(["RQ1", "RQ1_timesfm_subset"])]
    n.add("rq1.n_pairings_per_scheme", "RQ1", "Model-vs-persistence pairings per bootstrap scheme",
          (vs_p["scheme"] == "country").sum(), "int", "contrasts", 96)
    n.add("rq1.n_sig_better_any_scheme", "RQ1",
          "Pairings significantly better than persistence (Holm, any of three schemes)",
          ((vs_p["p_superiority_holm"] < 0.05)).sum(), "int", "contrasts", 0)
    c = con[(con["family"] == "RQ1") & (con["scheme"] == "country")]
    sig_worse = c[(c["p_two_sided_holm"] < 0.05) & (c["delta"] < 0)]
    for m in TREES:
        doms = sorted(sig_worse[(sig_worse["model"] == m) & (sig_worse["program"] == "PISA")]["domain"])
        exp = 3 if m.startswith("RF") else 2
        n.add(f"rq1.{m}.n_pisa_sig_worse", "RQ1", f"PISA domains where {m} is significantly worse ({', '.join(doms)})",
              len(doms), "int", "contrasts", exp)
    n.add("rq1.trees.n_point_better", "RQ1", "Program-domains where RF/XGBoost have lower point MAE than persistence",
          (c[c["model"].isin(TREES)]["delta"] > 0).sum(), "int", "contrasts", 0)
    tw = sig_worse[sig_worse["model"].isin(TREES) & (sig_worse["program"] == "PISA")]
    n.add_range("rq1.trees.pisa_sig_worse_gap", "RQ1", "RF/XGBoost excess MAE over persistence where significant (PISA)",
                -tw["delta"], "mae", "contrasts", (1.29, 1.92), 0.005)
    tn = c[c["model"].isin(TREES) & (c["program"] != "PISA")]
    n.add_range("rq1.trees.timss_pirls_gap", "RQ1", "RF/XGBoost excess MAE over persistence, TIMSS and PIRLS",
                -tn["delta"], "mae", "contrasts")
    n.add_range("rq1.trees.timss_pirls_ci_halfwidth", "RQ1", "Half-width of 95% CI for those contrasts",
                (tn["ci95_hi"] - tn["ci95_lo"]) / 2, "mae", "contrasts")
    n.add("rq1.trees.timss_pirls_n_sig", "RQ1", "Significant RF/XGBoost contrasts in TIMSS and PIRLS",
          ((tn["p_two_sided_holm"] < 0.05)).sum(), "int", "contrasts", 0)
    ridge = sig_worse[sig_worse["model"].isin(["Ridge_M0", "Ridge_M1"])]
    n.add_range("rq1.ridge_m0_m1.pisa_sig_worse_gap", "RQ1",
                "Ridge M0/M1 excess MAE over persistence where significant (all PISA)", -ridge["delta"], "mae",
                "contrasts", (2.35, 2.70), 0.005)

    t = con[(con["family"] == "RQ1_timesfm_subset") & (con["scheme"] == "country")].set_index(["program", "domain"])
    g4 = t.loc[("TIMSS_G4", "mathematics")]
    n.add("rq1.timesfm.timss_g4_math_delta", "RQ1", "TimesFM minus persistence MAE, TIMSS G4 mathematics",
          -g4["delta"], "mae", "contrasts", 2.28, 0.005)
    n.add("rq1.timesfm.timss_g4_math_p_holm", "RQ1", "Holm p, TimesFM vs persistence, TIMSS G4 mathematics",
          g4["p_two_sided_holm"], "p", "contrasts", 0.008, 0.0005)
    n.add("rq1.timesfm.n_equivalent", "RQ1", "Program-domains where TimesFM is equivalent to persistence (PISA reading)",
          (t["equivalent_90ci"] & (t["p_tost_holm"] < 0.05)).sum(), "int", "contrasts", 1)

    sc = pd.read_csv(SRC["scale"])
    n.add_range("scale.median_se", "Methods", "Median standard error of a country mean", sc["median_se_target"],
                "mae", "scale", (2.51, 3.29), 0.005)
    n.add_range("scale.sampling_floor_mae", "Results", "Sampling-error floor for persistence MAE",
                sc["sampling_floor_MAE_persistence"], "mae", "scale", (2.95, 3.91), 0.005)
    n.add_range("scale.persistence_to_floor_ratio", "Results", "Persistence MAE as a multiple of the sampling floor",
                sc["MAE_persistence"] / sc["sampling_floor_MAE_persistence"], "share", "scale", (3.06, 5.40), 0.005)
    n.add("methods.no_history_country_cycles", "Methods", "Country-cycles without prior same-program score",
          len(pd.read_csv(SRC["no_history"])), "int", "no_history", 580)
    n.add("methods.bootstrap_B_tests", "Methods", "Bootstrap replicates for paired tests (Holm, TOST)",
          json.loads(SRC["inference_manifest"].read_text())["B"], "int", "inference_manifest", 5000)


def rq2(n: Numbers) -> None:
    dec = pd.read_csv(SRC["rq2_decision"])
    n.add("rq2.n_kb_improves", "RQ2", "Program-domains meeting the pre-registered KB criterion",
          dec["kb_improves"].sum(), "int", "rq2_decision", 0)
    con = pd.read_csv(SRC["contrasts"])
    r2 = con[(con["analysis"] == PRIMARY) & (con["family"] == "RQ2")]
    for scheme in ("country", "cycle", "twoway"):
        m0 = r2[(r2["reference"] == "Ridge_M0") & (r2["scheme"] == scheme)]
        n.add(f"rq2.m1_equiv_m0.{scheme}", "RQ2", f"Program-domains with M1 equivalent to M0 (TOST ±1, Holm), {scheme}",
              (m0["equivalent_90ci"] & (m0["p_tost_holm"] < 0.05)).sum(), "int", "contrasts", 8)
    m0c = r2[(r2["reference"] == "Ridge_M0") & (r2["scheme"] == "country")]
    sig_gain = m0c[(m0c["ci95_lo"] > 0)]
    n.add_range("rq2.m1_minus_m0.sig_gain_share_of_margin", "RQ2",
                f"M1 gain over M0 as share of the 1-point margin where the CI excludes 0 ({', '.join(sig_gain['program'] + ' ' + sig_gain['domain'])})",
                sig_gain["delta"] / 1.0, "share", "contrasts", (0.05, 0.08), 0.005)
    a = r2[(r2["reference"] == "Ridge_M0A")]
    pisa = a[a["program"] == "PISA"]
    pc = pisa[pisa["scheme"] == "country"]
    n.add_range("rq2.m1_minus_m0a.pisa_gap", "RQ2", "M1 excess MAE over M0A, PISA", -pc["delta"], "mae",
                "contrasts", (1.59, 1.70), 0.005)
    n.add("rq2.m1_vs_m0a.pisa_country_p_holm_max", "RQ2", "Max Holm p, M1 vs M0A, PISA, country scheme",
          pc["p_two_sided_holm"].max(), "p", "contrasts", 0.002, 0.0005)
    n.add("rq2.m1_vs_m0a.pisa_cycle_twoway_p_holm_min", "RQ2",
          "Min Holm p, M1 vs M0A, PISA, cycle and two-way schemes (K5: not significant)",
          pisa[pisa["scheme"] != "country"]["p_two_sided_holm"].min(), "p", "contrasts", 0.499, 0.0005)
    pirls = a[(a["program"] == "PIRLS") & (a["scheme"] == "country")].iloc[0]
    n.add("rq2.m1_vs_m0a.pirls_ci90_hi", "RQ2", "Upper 90% CI bound, M0A minus M1, PIRLS (excludes 0)",
          pirls["ci90_hi"], "mae", "contrasts", -0.06, 0.005)
    n.add("rq2.m1_vs_m0a.pirls_p_holm", "RQ2", "Holm p, M1 vs M0A, PIRLS (K5: not significant)",
          pirls["p_two_sided_holm"], "p", "contrasts", 0.315, 0.0005)

    rc = pd.read_csv(SRC["rq2_contrasts"])
    rc = rc[rc["analysis"] == PRIMARY]
    share = rc["M1_share_of_perms_better_or_equal"]
    n.add_range("rq2.perm_share_all", "RQ2", "Share of 200 KB-weight permutations as good as or better than M1, all 8",
                share, "share", "rq2_contrasts", (0.23, 0.99), 0.005)
    n.add_range("rq2.perm_share_pisa", "RQ2", "Same share, PISA only", share[rc["program"] == "PISA"], "share",
                "rq2_contrasts", (0.42, 0.99), 0.005)
    n.add("rq2.n_beats_perm_p05", "RQ2", "Program-domains where M1 beats the permutation 5th percentile",
          rc["rule_c2_beats_perm_p05"].sum(), "int", "rq2_contrasts", 0)
    n.add("rq2.m0_minus_m1_max_gain", "RQ2", "Largest M1 gain over M0 (score points)", rc["dMAE_M0_minus_M1"].max(),
          "mae", "rq2_contrasts", 0.08, 0.005)
    n.add("rq2.n_placebo_ge_m1", "RQ2", "Program-domains where the best shrinkage placebo matches or beats M1",
          (rc["best_placebo_minus_M1"] <= 0).sum(), "int", "rq2_contrasts", 8)
    testable = rc["MAE_KB_top_1"].notna()
    n.add("rq2.topk.n_testable", "RQ2", "Program-domains where KB top-k vs random top-k is testable (K3)",
          testable.sum(), "int", "rq2_contrasts", 4)
    better = 0
    for k in (1, 2, 3):
        kb, p05 = rc[f"MAE_KB_top_{k}"], rc[f"KB_top_{k}_random_p05"]
        better += int(((kb < p05) & kb.notna()).sum())
    n.add("rq2.topk.n_better_than_random", "RQ2", "KB top-k configurations below the random top-k 5th percentile",
          better, "int", "rq2_contrasts", 0)


def kb(n: Numbers) -> None:
    k = json.loads(SRC["kappa"].read_text())
    n.add("kb.intra.n_records", "KB", "Gold records re-coded (intra-rater)", k["n_records"], "int", "kappa", 173)
    n.add("kb.intra.n_studies", "KB", "Studies in intra-rater re-coding", k["n_studies"], "int", "kappa", 30)
    n.add("kb.intra.kappa", "KB", "Intra-rater kappa vs pre-adjudication labels", k["kappa_pre_adjudication"],
          "kappa", "kappa", 0.65, 0.005)
    lo, hi = k["kappa_pre_adjudication_study_bootstrap_95ci"]
    n.add("kb.intra.kappa_ci_lo", "KB", "Intra-rater kappa, 95% study-bootstrap CI lower", lo, "kappa", "kappa", 0.45, 0.005)
    n.add("kb.intra.kappa_ci_hi", "KB", "Intra-rater kappa, 95% study-bootstrap CI upper", hi, "kappa", "kappa", 0.78, 0.005)

    v = json.loads(SRC["validation"].read_text())
    n.add("kb.validation.n_records", "KB", "Study-construct pairs in blind expert validation", v["n_records"], "int",
          "validation", 51)
    n.add("kb.validation.n_studies", "KB", "Studies in blind expert validation", v["n_studies"], "int", "validation", 35)
    expected_k = {"canonical_construct": 0.70, "effect_type": 0.21, "analysis_level": 0.38, "direction": 0.01}
    for field, e in expected_k.items():
        f = v["fields"][field]
        n.add(f"kb.validation.kappa.{field}", "KB", f"LLM-vs-expert kappa, {field}", f["kappa"], "kappa",
              "validation", e, 0.005)
    lo, hi = v["fields"]["canonical_construct"]["kappa_study_bootstrap_95ci"]
    n.add("kb.validation.kappa_construct_ci_lo", "KB", "Construct kappa, 95% CI lower", lo, "kappa", "validation", 0.57, 0.005)
    n.add("kb.validation.kappa_construct_ci_hi", "KB", "Construct kappa, 95% CI upper", hi, "kappa", "validation", 0.83, 0.005)

    s = pd.read_csv(SRC["validation_sample"])[["sample_id", "llm_direction"]].merge(
        pd.read_csv(SRC["validation_blind"], encoding="utf-8-sig")[["sample_id", "human_direction"]], on="sample_id")
    signed = s[s["llm_direction"].isin(["+", "-"])]
    n.add("kb.validation.n_llm_signed", "KB", "Pairs with an LLM-assigned sign", len(signed), "int",
          "validation_sample", 10)
    n.add("kb.validation.n_sign_confirmed", "KB", "LLM-assigned signs confirmed by the expert",
          (signed["llm_direction"] == signed["human_direction"]).sum(), "int", "validation_sample", 1)

    pc = pd.read_csv(SRC["per_construct"])
    cm = pc[pc.iloc[:, 0] == "TEACHER_SELF_EFFICACY_CLS_MGMT"].iloc[0]
    n.add("kb.cls_mgmt.n_sampled", "KB", "Sampled pairs labelled classroom-management self-efficacy", cm.iloc[1], "int",
          "per_construct", 6)
    n.add("kb.cls_mgmt.n_correct", "KB", "Of those, labels confirmed correct (K1: 6/6 mislabelled)", cm.iloc[2], "int",
          "per_construct", 0)
    w = pd.read_csv(SRC["weights_corrected"]).set_index("canonical_construct").loc["TEACHER_SELF_EFFICACY_CLS_MGMT"]
    n.add("kb.cls_mgmt.weight", "KB", "KB weight, classroom-management self-efficacy", w["w"], "rel",
          "weights_corrected", 0.854, 0.0005)
    n.add("kb.cls_mgmt.weight_corrected", "KB", "Sample-corrected weight", w["w_corrected"], "rel",
          "weights_corrected", 0.677, 0.0005)

    e = pd.read_csv(SRC["evidence"])
    n.add("kb.evidence.n_rows", "KB", "Evidence rows in the knowledge base", len(e), "int", "evidence", 530)
    n.add("kb.evidence.n_studies", "KB", "Studies in the knowledge base", e["study_id"].nunique(), "int", "evidence", 80)
    lvl = e["analysis_level"].value_counts()
    n.add("kb.evidence.n_student_level", "KB", "Student-level evidence rows", lvl.get("student", 0), "int", "evidence", 486)
    n.add("kb.evidence.n_country_level", "KB", "Country-level evidence rows", lvl.get("country", 0), "int", "evidence", 4)
    g = pd.read_csv(SRC["gold"])
    n.add("kb.gold.n_rows", "KB", "Gold corpus rows", len(g), "int", "gold", 246)
    n.add("kb.gold.n_studies", "KB", "Gold corpus studies", g["study_id"].nunique(), "int", "gold", 42)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    out_dir = args.out if args.out.is_absolute() else ROOT / args.out
    if out_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {out_dir}")

    n = Numbers()
    rq1(n)
    rq2(n)
    kb(n)
    if n.failures:
        sys.exit("Thesis numbers do not match the claim map; nothing written:\n  " + "\n  ".join(n.failures))

    res = pd.DataFrame(n.rows)
    out_dir.mkdir(parents=True)
    res.to_csv(out_dir / "thesis_numbers.csv", index=False)
    lines = ["# Thesis numbers (generated; do not edit by hand)", "",
             "Every number below is computed from the listed output file by `scripts/build_thesis_numbers_v1.py`.",
             "Rows with an expected value were checked against the Phase 6.0 claim map.", ""]
    for section, t in res.groupby("section", sort=False):
        lines += [f"## {section}", "", "| ID | Description | Value | Check | Source |", "|---|---|---|---|---|"]
        for r in t.itertuples():
            check = "" if r.check_ok is None else ("ok" if r.check_ok else "FAIL")
            lines.append(f"| `{r.id}` | {r.description} | {r.text} | {check} | `{r.source}` |")
        lines.append("")
    (out_dir / "thesis_numbers.md").write_text("\n".join(lines), encoding="utf-8")
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "n_numbers": len(res),
                "n_checked": int(res["check_ok"].notna().sum()),
                "inputs": {str(p.relative_to(ROOT)): sha256(p) for p in SRC.values()}}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"{len(res)} numbers, {manifest['n_checked']} checked, all match. Saved to {out_dir}/")


if __name__ == "__main__":
    main()
