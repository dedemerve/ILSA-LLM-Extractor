#!/usr/bin/env python3
"""
Thesis appendix tables (Phase 6.5) as LaTeX booktabs fragments, generated from
the verified output files. One .tex file per table plus appendix_tables.tex,
which \\input's them in order. Required packages: booktabs, longtable, graphicx,
pdflscape (for the landscape A2-A17 block), amsmath (for \\approx in text mode
via $...$).

Rounding: MAE 2 dp, relative MAE 3 dp, probabilities 2 dp, kappa 2 dp,
weights 3 dp, SD 3 dp.

Inputs: rq1_tables_v2, inference_v1, rq2_v3, unified_backtest_v1, stage5,
kb_integrity_v1, thesis_numbers_v2 (all under outputs/).
Outputs (refuses to overwrite): outputs/appendix_tables_v1/
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
O = ROOT / "outputs"
DEFAULT_OUT = O / "appendix_tables_v1"
SRC = {
    "rq1_pd": O / "rq1_tables_v2" / "rq1_per_program_domain.csv",
    "rq1_agg": O / "rq1_tables_v2" / "rq1_aggregate_relative_mae.csv",
    "contrasts": O / "inference_v1" / "contrasts_all.csv",
    "rq2_decision": O / "inference_v1" / "rq2_preregistered_decision.csv",
    "rq2_contrasts": O / "rq2_v3" / "contrasts_primary_excl2025.csv",
    "appendix_complete": O / "unified_backtest_v1" / "summary_appendix_complete_folds.csv",
    "appendix_incomplete": O / "unified_backtest_v1" / "summary_appendix_baselines_incomplete_folds.csv",
    "folds": O / "unified_backtest_v1" / "folds.csv",
    "weights": O / "stage5" / "predictor_weights_v3.csv",
    "feature_map": O / "stage5" / "kb_feature_map_v3.csv",
    "evidence": O / "stage5" / "evidence_matrix.csv",
    "talis": O / "stage5" / "talis_covariate_estimates.csv",
    "validation": O / "kb_integrity_v1" / "evidence_validation_agreement_v1.json",
    "per_construct": O / "kb_integrity_v1" / "evidence_validation_per_construct_v1.csv",
    "confusion": O / "kb_integrity_v1" / "evidence_validation_confusion_v1.csv",
    "weights_corrected": O / "kb_integrity_v1" / "predictor_weights_v3_sample_corrected.csv",
    "kappa": O / "kb_integrity_v1" / "kappa_summary.json",
    "thesis_numbers": O / "thesis_numbers_v2" / "thesis_numbers.csv",
}
PRIMARY, SENS = "primary_excl2025", "sensitivity_incl2025"
MAIN = [("PISA", "mathematics"), ("PISA", "reading"), ("PISA", "science"),
        ("TIMSS_G8", "mathematics"), ("TIMSS_G8", "science"),
        ("TIMSS_G4", "mathematics"), ("TIMSS_G4", "science"), ("PIRLS", "reading")]
PROGRAM = {"PISA": "PISA", "TIMSS_G8": "TIMSS Grade 8", "TIMSS_G4": "TIMSS Grade 4", "PIRLS": "PIRLS",
           "ICCS": "ICCS", "ICILS": "ICILS", "PIAAC": "PIAAC"}
MODEL = {
    "Persistence": "Persistence", "Drift": "Drift", "AR1": "AR(1)", "LinearTrend": "Linear trend",
    "InternationalMean": "International mean", "Ridge_M0A": "Ridge M0A (scores only)",
    "Ridge_M0": "Ridge M0 (unweighted covariates)", "Ridge_M1": "Ridge M1 (KB-weighted covariates)",
    "RF_score": "Random forest (scores)", "RF_cov": "Random forest (scores + covariates)",
    "XGB_score": "XGBoost (scores)", "XGB_cov": "XGBoost (scores + covariates)",
    "TimesFM2_5_zero_shot": "TimesFM 2.5 (zero-shot)",
}
SIG_NOTE = (r"$^{*}$ significantly worse than persistence (Holm-adjusted two-sided $p<.05$, country-cluster "
            r"bootstrap, $B=5{,}000$); $\approx$ practically equivalent to persistence (TOST, $\pm1$ point, Holm). "
            r"No model was significantly better than persistence.")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def tex(s) -> str:
    s = str(s)
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"),
                 ("_", r"\_"), ("{", r"\{"), ("}", r"\}"), ("~", r"\textasciitilde{}"), ("^", r"\^{}"),
                 ("±", r"$\pm$"), ("≈", r"$\approx$"), ("≥", r"$\geq$"), ("≤", r"$\leq$"), ("−", "$-$"),
                 ("–", "--"), ("—", "---")):
        s = s.replace(a, b)
    return s


def code(s) -> str:
    return r"\texttt{" + tex(s) + "}"


def pd_label(prog: str, dom: str) -> str:
    return f"{PROGRAM.get(prog, prog)} {dom.replace('_', ' ')}"


def f(x, dp: int) -> str:
    return "--" if pd.isna(x) else f"{x:.{dp}f}".replace("-", "$-$")


def ci(lo, hi, dp: int) -> str:
    return f"[{f(lo, dp)}, {f(hi, dp)}]"


def table(label: str, caption: str, colspec: str, header: list[str], rows: list[list[str]],
          note: str = "", size: str = r"\footnotesize", long: bool = False, resize: bool = False) -> str:
    head = " & ".join(header) + r" \\"
    body = "\n".join(" & ".join(r) + r" \\" for r in rows)
    note_tex = (r"\par\smallskip{\scriptsize " + note + "}") if note else ""
    if long:
        return "\n".join([
            "{" + size, r"\begin{longtable}{" + colspec + "}",
            r"\caption{" + caption + r"}\label{" + label + r"}\\", r"\toprule", head, r"\midrule", r"\endfirsthead",
            r"\toprule", head, r"\midrule", r"\endhead", r"\bottomrule", r"\endlastfoot",
            body, r"\end{longtable}", (r"{\scriptsize " + note + "}") if note else "", "}", ""])
    tab = "\n".join([r"\begin{tabular}{" + colspec + "}", r"\toprule", head, r"\midrule", body,
                     r"\bottomrule", r"\end{tabular}"])
    if resize:
        tab = r"\resizebox{\textwidth}{!}{%" + "\n" + tab + "}"
    return "\n".join([r"\begin{table}[htbp]", r"\centering", size, r"\caption{" + caption + "}",
                      r"\label{" + label + "}", tab, note_tex, r"\end{table}", ""])


def flag(r) -> str:
    if r.model == "Persistence":
        return "ref."
    worse = isinstance(r.sig_vs_persistence, str) and r.sig_vs_persistence == "worse"
    eq = bool(r.equivalent_to_persistence)
    return ", ".join([s for s, on in (("worse$^{*}$", worse), (r"$\approx$", eq)) if on]) or "n.s."


# ---------------------------------------------------------------------------
# Appendix A: RQ1
# ---------------------------------------------------------------------------

def a_aggregate(agg: pd.DataFrame, analysis: str, label: str, caption: str) -> str:
    a = agg[agg["analysis"] == analysis].sort_values("geo_mean_rel_MAE")
    rows = [[tex(MODEL[r.model]), f(r.geo_mean_rel_MAE, 3), str(r.n_point_rank1), f(r.mean_p_rank1, 2),
             str(r.n_sig_better), str(r.n_sig_worse), str(int(r.n_equivalent))] for r in a.itertuples()]
    return table(label, caption, "lrrrrrr",
                 ["Model", "Geo.\\ mean rel.\\ MAE", "Point rank 1", "Mean P(rank 1)", "Sig.\\ better",
                  "Sig.\\ worse", "Equivalent"], rows,
                 note="Relative MAE = model MAE / persistence MAE, geometric mean over the 8 program-domains. "
                      "Counts are numbers of program-domains (out of 8). " + SIG_NOTE)


def a_program_domain(res: pd.DataFrame, prog: str, dom: str, subset: str, label: str) -> str:
    t = res[(res["analysis"] == PRIMARY) & (res["subset"] == subset) & (res["program"] == prog)
            & (res["domain"] == dom)].sort_values(["MAE", "model"])
    r0 = t.iloc[0]
    rows = []
    for r in t.itertuples():
        rel_ci = "--" if r.model == "Persistence" else ci(r.rel_MAE_ci95_lo, r.rel_MAE_ci95_hi, 3)
        rows.append([str(r.rank), tex(MODEL[r.model]), f(r.MAE, 2), ci(r.MAE_ci95_lo, r.MAE_ci95_hi, 2),
                     f(r.rel_MAE, 3), rel_ci, f"{r.rank_ci95_lo}--{r.rank_ci95_hi}", f(r.p_rank1, 2), flag(r)])
    what = "common test set" if subset == "full_common_set" else "TimesFM subset"
    caption = (f"{pd_label(prog, dom)}, {what} (primary analysis, 2025 excluded; n = {r0.n}, "
               f"{r0.n_countries} countries, {r0.n_folds} forecast origins)")
    return table(label, tex(caption), "rlrlrlcrl",
                 ["Rank", "Model", "MAE", "95\\% CI", "Rel.\\ MAE", "95\\% CI", "Rank 95\\% int.", "P(rank 1)",
                  "vs.\\ persistence"], rows,
                 note="MAE in score points. CIs, rank intervals and P(rank 1) from a joint country-cluster "
                      "bootstrap of all models ($B=5{,}000$). " + SIG_NOTE)


def a_appendix_programs(complete: pd.DataFrame, incomplete: pd.DataFrame) -> str:
    rows = []
    for name, df in (("complete fold", complete), ("baselines only, incomplete fold", incomplete)):
        for (prog, dom), g in df.groupby(["program", "domain"], sort=False):
            for i, r in enumerate(g.sort_values(["MAE", "model"]).itertuples()):
                rows.append([tex(pd_label(prog, dom)) if i == 0 else "", tex(name) if i == 0 else "",
                             tex(MODEL.get(r.model, r.model)), str(r.n), str(r.n_folds), f(r.MAE, 2),
                             f(r.rel_MAE_vs_persistence, 3)])
    return table("tab:A19", "Additional programs (descriptive only; one forecast origin each)", "lllrrrr",
                 ["Program-domain", "Fold type", "Model", "n", "Origins", "MAE", "Rel.\\ MAE"], rows,
                 note="Single-origin results; not used for inference or ranking. PIAAC: no regression model "
                      "could be trained without same-wave information.", long=True)


def a_folds(folds: pd.DataFrame) -> str:
    rows = []
    for r in folds.itertuples():
        rows.append([tex(PROGRAM.get(r.program, r.program)), tex(r.domain.replace("_", " ")), str(r.test_cycle),
                     str(r.prev_cycle), tex(r.origin_year_month), str(r.n_train), str(r.n_test),
                     str(r.n_no_history), "yes" if r.complete else "no",
                     tex(str(r.covariates_used).replace("|", ", ")) if isinstance(r.covariates_used, str) else "--"])
    return table("tab:A20", "Fold structure of the rolling-origin backtest (Definition A origins)",
                 "llrrlrrrlp{4.2cm}",
                 ["Program", "Domain", "Test", "Prev.", "Origin", "n train", "n test", "No hist.", "Complete",
                  "Covariates used"], rows,
                 note="Origin = publication month of the previous cycle (release dates are an assumption; see "
                      "Appendix F). A fold is complete if it has at least 10 training rows and every model "
                      "forecast every test row. No hist. = countries without an earlier score (excluded).",
                 size=r"\scriptsize", long=True)


# ---------------------------------------------------------------------------
# Appendix B: pre-registered RQ2 decision
# ---------------------------------------------------------------------------

def b_decision(dec: pd.DataFrame, con: pd.DataFrame, rc: pd.DataFrame) -> str:
    m0 = con[(con["analysis"] == PRIMARY) & (con["family"] == "RQ2") & (con["scheme"] == "country")
             & (con["reference"] == "Ridge_M0")].set_index(["program", "domain"])
    rc = rc[rc["analysis"] == PRIMARY].set_index(["program", "domain"])
    d = dec.set_index(["program", "domain"])
    yn = lambda b: "yes" if bool(b) else "no"  # noqa: E731
    rows = []
    for key in MAIN:
        r, m, p = d.loc[key], m0.loc[key], rc.loc[key]
        lo, hi = [float(x) for x in r["ci95"].strip("[]").split(",")]
        rows.append([tex(pd_label(*key)), f(r["delta_M0A_minus_M1"], 2), ci(lo, hi, 2), f(m["delta"], 2),
                     ci(m["ci95_lo"], m["ci95_hi"], 2), f(p["M1_share_of_perms_better_or_equal"], 2),
                     yn(r["c1"]), yn(r["c2"]), yn(r["c3"]), r"\textbf{" + yn(r["kb_improves"]) + "}",
                     yn(r["M1_vs_M0_equivalent_holm"]), yn(r["M1_vs_M0A_equivalent_holm"])])
    return table("tab:B4", "Pre-registered RQ2 decision (primary analysis, country-cluster bootstrap)",
                 "lrlrlrcccccc",
                 ["Program-domain", r"$\Delta$(M0A$-$M1)", "95\\% CI", r"$\Delta$(M0$-$M1)", "95\\% CI",
                  "Perm.\\ share", "C1", "C2", "C3", "KB improves", "M1$\\approx$M0", "M1$\\approx$M0A"], rows,
                 note=r"$\Delta$ = MAE(reference) $-$ MAE(M1); positive values favour M1. Perm.\ share = share of "
                      r"200 random reassignments of KB weights that performed as well as or better than M1. "
                      r"C1: $\Delta$(M0A$-$M1) $\geq 1$ with Holm-adjusted CI lower bound $>0$; C2: M1 below the "
                      r"permutation 5th percentile; C3: $\Delta$(M0$-$M1) $>0$. Equivalence: TOST, $\pm1$ point, "
                      r"Holm. Small negative values round to $-$0.00.", resize=True)


def b_comparators(rc: pd.DataFrame) -> str:
    rc = rc[rc["analysis"] == PRIMARY].set_index(["program", "domain"])
    spec = [("MAE_Ridge_M0A", "Ridge M0A (scores only)"), ("MAE_Ridge_M0", "Ridge M0 (unweighted)"),
            ("MAE_Ridge_M1", "Ridge M1 (KB-weighted)"), ("MAE_Uniform_0.75", "Uniform $w=0.75$"),
            ("MAE_Uniform_0.5", "Uniform $w=0.5$"), ("MAE_Uniform_0.25", "Uniform $w=0.25$"),
            ("MAE_DataDriven_block", "Data-driven block shrinkage"),
            ("MAE_DataDriven_adaptive", "Data-driven adaptive ridge"), ("MAE_KB_top_1", "KB top-1"),
            ("MAE_KB_top_2", "KB top-2"), ("MAE_KB_top_3", "KB top-3"),
            ("perm_p05", "Permutation null, 5th pct."), ("perm_median", "Permutation null, median")]
    header = ["Comparator"] + [tex(pd_label(*k)).replace("TIMSS Grade ", "G").replace("mathematics", "math.")
                               .replace("science", "sci.").replace("reading", "read.") for k in MAIN]
    rows = [[lab] + [f(rc.loc[k, col], 2) for k in MAIN] for col, lab in spec]
    rows.append(["Covariates in program"] + [str(int(rc.loc[k, "n_covariates_program"])) for k in MAIN])
    return table("tab:B5", "RQ2 comparators: MAE by program-domain (primary analysis)", "l" + "r" * len(MAIN),
                 header, rows, note="MAE in score points. -- = not defined (fewer covariates than k).",
                 resize=True)


# ---------------------------------------------------------------------------
# Appendix C: knowledge base
# ---------------------------------------------------------------------------

def c_weights(w: pd.DataFrame) -> str:
    rows = [[code(r.canonical_construct), str(r.n_studies), str(r.n_rows), str(r.n_programs), str(r.n_cycles),
             str(r.n_countries), f(r.frac_predictive, 3), "yes" if r.forecastable else "no", f(r.w_F, 3),
             f(r.w_P, 3), f(r.w_C, 3), f(r.w_N, 3), f(r.w_E, 3), f(r.w, 3)]
            for r in w.sort_values(["w", "canonical_construct"], ascending=[False, True]).itertuples()]
    return table("tab:C1", "Knowledge-base weights (v3) and their components", "lrrrrrrcrrrrrr",
                 ["Construct", "Studies", "Rows", "Prog.", "Cycles", "Countries", "Pred.\\ share", "Forecastable",
                  "$w_F$", "$w_P$", "$w_C$", "$w_N$", "$w_E$", "$w$"], rows,
                 note="$w_F$, $w_P$, $w_C$, $w_N$: frequency, program, cycle and country diversity components; "
                      "$w_E$: predictive-evidence factor. $w$ = normalized composite (maximum 1). "
                      "Non-forecastable constructs receive $w=0$.", size=r"\scriptsize", resize=True)


def c_feature_map(m: pd.DataFrame) -> str:
    rows = [[code(r.ilsa_variable), tex(r.source_program.replace("|", ", ")), code(r.canonical_construct),
             tex(r.status), tex(r.excluded_source_cycles) if isinstance(r.excluded_source_cycles, str) else "--",
             tex(r.note) if isinstance(r.note, str) else ""] for r in m.itertuples()]
    return table("tab:C2", "Mapping of ILSA covariates to knowledge-base constructs", "lllllp{5cm}",
                 ["Variable", "Program", "Construct", "Status", "Excluded cycles", "Note"], rows,
                 size=r"\scriptsize", long=True)


def c_agreement(v: dict) -> str:
    names = {"canonical_construct": "Construct", "effect_type": "Effect type (predictive/associative)",
             "analysis_level": "Analysis level", "direction": "Direction"}
    rows = []
    for k, lab in names.items():
        x = v["fields"][k]
        lo, hi = x["kappa_study_bootstrap_95ci"]
        rows.append([tex(lab), f(x["kappa"], 2), ci(lo, hi, 2), f(100 * x["agreement"], 1) + r"\%"])
    return table("tab:C4", tex(f"Agreement between LLM labels and blind expert coding "
                               f"({v['n_records']} study-construct pairs, {v['n_studies']} studies)"),
                 "lrlr", ["Field", r"$\kappa$", "95\\% CI", "Agreement"], rows,
                 note="Single expert coder, blind to LLM labels, coding from the original PDFs. CIs from a "
                      "study-cluster bootstrap (5,000 replicates). This is LLM-versus-expert agreement, not "
                      "agreement between two human coders. "
                      f"Program or cycle did not match the extraction in {v['programs_cycles_ok']['no']} of "
                      f"{v['n_records']} pairs.")


def c_per_construct(pc: pd.DataFrame) -> str:
    rows = [[code(r.llm_canonical_construct), str(r.n), str(r.n_correct), f(r.precision, 2)]
            for r in pc.sort_values(["precision", "llm_canonical_construct"]).itertuples()]
    return table("tab:C5", "Precision of LLM construct labels in the validation sample", "lrrr",
                 ["LLM construct", "n", "Correct", "Precision"], rows)


def c_confusion(cm: pd.DataFrame) -> str:
    cols = [c for c in cm.columns if c != "llm"]
    abbr = {c: f"E{i + 1}" for i, c in enumerate(cols)}
    rows = [[code(r["llm"])] + [str(int(r[c])) if r[c] else "." for c in cols] for _, r in cm.iterrows()]
    legend = "; ".join(f"{abbr[c]} = {code(c)}" for c in cols)
    return table("tab:C6", "Construct confusion matrix (rows: LLM label; columns: expert label)",
                 "l" + "r" * len(cols), ["LLM label"] + [abbr[c] for c in cols], rows,
                 note="Expert labels: " + legend + ". A dot denotes zero.", size=r"\scriptsize")


def c_corrected(wc: pd.DataFrame) -> str:
    rows = [[code(r.canonical_construct), str(int(r.n_studies)), f(r.w, 3),
             "--" if pd.isna(r.n_studies_corrected) else str(int(r.n_studies_corrected)), f(r.w_corrected, 3)]
            for r in wc.itertuples()]
    return table("tab:C7", "Weight sensitivity: sampled pairs relabelled to the expert codes", "lrrrr",
                 ["Construct", "Studies", "$w$", "Studies (corrected)", "$w$ (corrected)"], rows,
                 note="Only the 51 sampled pairs were relabelled (about 10\\% of evidence rows). -- = no evidence "
                      "left for the construct after relabelling.")


def c_intra(k: dict) -> str:
    lo, hi = k["kappa_pre_adjudication_study_bootstrap_95ci"]
    rows = [["Records / studies", f"{k['n_records']} / {k['n_studies']}"],
            [r"$\kappa$, against pre-adjudication labels", f(k["kappa_pre_adjudication"], 2)],
            ["95\\% CI (study bootstrap)", ci(lo, hi, 2)],
            ["Agreement, pre-adjudication", f(100 * k["agreement_pre_adjudication"], 1) + r"\%"],
            [r"$\kappa$ after adjudication (not valid as reliability)",
             f(k["kappa_post_adjudication_NOT_VALID_AS_RELIABILITY"], 2)]]
    return table("tab:C8", "Intra-rater agreement on the gold corpus (canonical construct)", "lr",
                 ["Statistic", "Value"], rows,
                 note="The same researcher verified the gold labels and produced the re-codes. Reference labels "
                      "frozen at commit b39eb0f. The post-adjudication value was computed after five labels were "
                      "changed to the re-codes and is reported only for transparency.")


def c_levels(e: pd.DataFrame) -> str:
    vc = e["analysis_level"].value_counts()
    rows = [[tex(lvl), str(int(n)), f(100 * n / len(e), 1) + r"\%"] for lvl, n in vc.items()]
    rows.append([r"\textit{Total}", str(len(e)), r"100.0\%"])
    return table("tab:C9", tex(f"Analysis level of knowledge-base evidence rows ({e['study_id'].nunique()} studies)"),
                 "lrr", ["Analysis level", "Rows", "Share"], rows)


# ---------------------------------------------------------------------------
# Appendix D and E
# ---------------------------------------------------------------------------

def d_talis(t: pd.DataFrame) -> str:
    u = t.drop_duplicates(["cycle", "country_iso3", "variable"])
    g = u.groupby(["construct", "cycle", "variable"])["mean"].agg(["count", "std", "min", "max"]).reset_index()
    rows = []
    for c, gc in g.groupby("construct", sort=True):
        for i, r in enumerate(gc.itertuples()):
            bold = r.cycle == 2018 and r.std < 0.01
            sd = r"\textbf{" + f(r.std, 3) + "}" if bold else f(r.std, 3)
            rows.append([tex(c.replace("_", " ")) if i == 0 else "", str(r.cycle), code(r.variable),
                         str(r.count), sd, f(r.min, 3), f(r.max, 3)])
    return table("tab:D1", "Between-country dispersion of TALIS country means by cycle", "lrlrrrr",
                 ["Construct", "Cycle", "Variable", "Countries", "SD", "Min", "Max"], rows,
                 note="Bold: degenerate TALIS 2018 estimates (between-country SD $\\approx 0.001$). The TALIS 2018 "
                      "sources of SECLSS, TCDISCS and TJSPROS were masked in every fold; T3EFFPD (normal variance) "
                      "was retained.", size=r"\scriptsize")


def e_numbers(tn: pd.DataFrame) -> str:
    rows = [[code(r.id), tex(r.description), tex(r.text), code(r.source.replace("outputs/", ""))]
            for r in tn.itertuples()]
    return table("tab:E3", "Numbers quoted in the text and their source files", "p{4.6cm}p{5.6cm}rp{3.6cm}",
                 ["ID", "Description", "Value", "Source (outputs/)"], rows,
                 note="Generated by scripts/build\\_thesis\\_numbers\\_v1.py; rows with an expected value were "
                      "checked against the claim map.", size=r"\scriptsize", long=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    out_dir = args.out if args.out.is_absolute() else ROOT / args.out
    if out_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {out_dir}")

    res = pd.read_csv(SRC["rq1_pd"])
    agg = pd.read_csv(SRC["rq1_agg"])
    rc = pd.read_csv(SRC["rq2_contrasts"])
    files: list[tuple[str, str]] = []  # (filename, content)
    files.append(("A01_aggregate.tex", a_aggregate(agg, PRIMARY, "tab:A1",
                                                   "RQ1 aggregate ranking across the 8 program-domains (primary analysis)")))
    for i, (p, d) in enumerate(MAIN):
        files.append((f"A{i + 2:02d}_{p}_{d}.tex", a_program_domain(res, p, d, "full_common_set", f"tab:A{i + 2}")))
    for i, (p, d) in enumerate(MAIN):
        files.append((f"A{i + 10:02d}_{p}_{d}_timesfm.tex",
                      a_program_domain(res, p, d, "timesfm_subset", f"tab:A{i + 10}")))
    files.append(("A18_aggregate_incl2025.tex", a_aggregate(agg, SENS, "tab:A18",
                                                            "RQ1 aggregate ranking, sensitivity analysis including 2025")))
    files.append(("A19_appendix_programs.tex", a_appendix_programs(pd.read_csv(SRC["appendix_complete"]),
                                                                   pd.read_csv(SRC["appendix_incomplete"]))))
    files.append(("A20_folds.tex", a_folds(pd.read_csv(SRC["folds"]))))
    files.append(("B4_rq2_decision.tex", b_decision(pd.read_csv(SRC["rq2_decision"]), pd.read_csv(SRC["contrasts"]), rc)))
    files.append(("B5_rq2_comparators.tex", b_comparators(rc)))
    files.append(("C1_weights.tex", c_weights(pd.read_csv(SRC["weights"]))))
    files.append(("C2_feature_map.tex", c_feature_map(pd.read_csv(SRC["feature_map"]))))
    files.append(("C4_agreement.tex", c_agreement(json.loads(SRC["validation"].read_text()))))
    files.append(("C5_per_construct.tex", c_per_construct(pd.read_csv(SRC["per_construct"]))))
    files.append(("C6_confusion.tex", c_confusion(pd.read_csv(SRC["confusion"]))))
    files.append(("C7_weights_corrected.tex", c_corrected(pd.read_csv(SRC["weights_corrected"]))))
    files.append(("C8_intra_rater.tex", c_intra(json.loads(SRC["kappa"].read_text()))))
    files.append(("C9_analysis_levels.tex", c_levels(pd.read_csv(SRC["evidence"]))))
    files.append(("D1_talis_dispersion.tex", d_talis(pd.read_csv(SRC["talis"]))))
    files.append(("E3_thesis_numbers.tex", e_numbers(pd.read_csv(SRC["thesis_numbers"], dtype={"text": str}))))

    out_dir.mkdir(parents=True)
    for name, content in files:
        (out_dir / name).write_text(content, encoding="utf-8")
    master = ["% Generated by scripts/build_appendix_tables_v1.py; do not edit by hand.",
              "% Preamble needs: \\usepackage{booktabs,longtable,graphicx,pdflscape,amsmath}", ""]
    for name, _ in files:
        if name == "A02_PISA_mathematics.tex":
            master.append(r"\begin{landscape}")
        master.append(r"\input{" + name[:-4] + "}")
        if name.startswith("A17_"):
            master.append(r"\end{landscape}")
    (out_dir / "appendix_tables.tex").write_text("\n".join(master) + "\n", encoding="utf-8")
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "n_tables": len(files),
                "files": [n for n, _ in files],
                "inputs": {str(p.relative_to(ROOT)): sha256(p) for p in SRC.values()}}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"{len(files)} tables written to {out_dir}/")


if __name__ == "__main__":
    main()
